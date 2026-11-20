"""Training, evaluation, and checkpointing for PhiTapMicro."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from phigros_gen.dataset import (
    TapDataset,
    Vocab,
    load_jsonl,
    make_dataloader,
    select_rows,
    split_rows,
)
from phigros_gen.metrics import (
    evaluate_baselines,
    evaluate_chart_metrics,
    hungarian_loss,
    matched_pos_l1,
    template_aux_loss,
)
from phigros_gen.model import PhiTapMicro, build_model, default_config


DEFAULT_HYPERPARAMS = {
    "lr": 3e-4,
    "weight_decay": 0.01,
    "warmup_frac": 0.05,
    "grad_clip": 1.0,
    "label_smoothing": 0.02,
    "side_weight": 0.2,
    "aux_weight": 1.0,
    "pos_weight": 0.5,
    "batch_size": 4,
    "effective_batch": 8,
    "max_tokens": 8192,
    "epochs": 200,
    "patience": 30,
}


def get_device(prefer: Optional[str] = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        info = check_gpu()
        if info.get("status") == "ok":
            return torch.device("cuda")
        print(
            f"GPU capability check failed ({info.get('status')}); falling back to CPU",
            flush=True,
        )
    return torch.device("cpu")


def check_gpu() -> Dict[str, Any]:
    """Verify CUDA capability and run a real kernel before training."""
    info: Dict[str, Any] = {"available": torch.cuda.is_available()}
    if not torch.cuda.is_available():
        info["status"] = "no_cuda"
        return info
    try:
        capability = tuple(int(v) for v in torch.cuda.get_device_capability(0))
        info["capability"] = list(capability)
        info["device_name"] = torch.cuda.get_device_name(0)
        a = torch.randn(512, 512, device="cuda")
        b = a @ a.t()
        b.sum().item()
        torch.cuda.synchronize()
        info["matmul_ok"] = True
        info["status"] = "ok" if capability == (12, 0) else "capability_mismatch"
    except Exception as exc:  # pragma: no cover - hardware dependent
        info["status"] = "error"
        info["error"] = str(exc)
    return info


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _forward(model: PhiTapMicro, batch: Dict[str, Any], device: torch.device):
    return model.forward_full(
        batch["scalars"].to(device),
        batch["type_idx"].to(device),
        batch["diff_idx"].to(device),
        batch["src_idx"].to(device),
        batch["audio"].to(device),
        batch["times"].to(device),
        batch["dts"].to(device),
        u_slot=batch["u_slot"].to(device),
        group_ids=batch["group_ids"].to(device),
        src_key_padding_mask=(~batch["mask"]).to(device),
    )


def _batch_loss(
    batch: Dict[str, Any],
    out: Dict[str, torch.Tensor],
    device: torch.device,
    side_weight: float,
    label_smoothing: float,
    aux_weight: float,
    pos_weight: float,
    side_class_weights: Optional[torch.Tensor],
) -> torch.Tensor:
    losses: List[torch.Tensor] = []
    n_charts = batch["x_bin"].shape[0]
    for b in range(n_charts):
        n = int(batch["num_notes"][b])
        if n == 0:
            continue
        hung = hungarian_loss(
            out["x_bin_logits"][b, :n],
            out["side_logits"][b, :n],
            batch["x_bin"][b, :n],
            batch["side"][b, :n],
            batch["group_ranges"][b],
            side_weight=side_weight,
            label_smoothing=label_smoothing,
            side_class_weights=side_class_weights,
        )
        aux = template_aux_loss(
            out["center"][b, :n],
            out["log_spread"][b, :n],
            batch["x_abs"][b, :n],
            batch["group_ranges"][b],
        )
        x_abs_dev = batch["x_abs"][b, :n].to(out["pos"].device)
        pos_l1 = matched_pos_l1(
            out["pos"][b, :n],
            x_abs_dev,
            batch["group_ranges"][b],
            hung["assignments"],
        )
        loss = hung["total"] + aux_weight * aux + pos_weight * pos_l1
        losses.append(loss)
    if not losses:
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()


def run_epoch(
    model: PhiTapMicro,
    dataloader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR],
    side_weight: float,
    label_smoothing: float,
    aux_weight: float,
    pos_weight: float,
    side_class_weights: Optional[torch.Tensor],
    grad_accum: int,
    grad_clip: float,
    use_amp: bool,
) -> float:
    training = optimizer is not None
    model.train(training)
    running = 0.0
    seen_charts = 0
    accum_count = 0
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp)

    for batch in dataloader:
        with amp:
            out = _forward(model, batch, device)
            loss = _batch_loss(
                batch, out, device, side_weight, label_smoothing, aux_weight,
                pos_weight, side_class_weights,
            )
        if training:
            scaled = loss / max(1, grad_accum)
            with amp:
                scaled.backward()
            accum_count += 1
            if accum_count % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        running += float(loss.detach().item()) * len(batch["num_notes"])
        seen_charts += len(batch["num_notes"])

    if training and accum_count % grad_accum != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    return running / max(1, seen_charts)


def evaluate_model(
    model: PhiTapMicro,
    dataloader,
    device: torch.device,
    side_weight: float = 0.2,
    label_smoothing: float = 0.02,
    aux_weight: float = 0.1,
    pos_weight: float = 0.5,
    side_class_weights: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    model.eval()
    metrics_agg = {
        "loss": 0.0,
        "x_mae": 0.0,
        "x_rmse": 0.0,
        "top1": 0.0,
        "top5": 0.0,
        "side_macro_f1": 0.0,
        "side_acc": 0.0,
        "pos_mae": 0.0,
        "center_mae": 0.0,
        "spread_mae_k2": 0.0,
        "log_spread_mae_k2": 0.0,
    }
    total_notes = 0
    total_charts = 0
    per_chart: Dict[str, Dict[str, float]] = {}
    with torch.no_grad():
        for batch in dataloader:
            out = _forward(model, batch, device)
            loss = _batch_loss(
                batch, out, device, side_weight, label_smoothing, aux_weight,
                pos_weight, side_class_weights,
            )
            n_charts = batch["x_bin"].shape[0]
            metrics_agg["loss"] += float(loss.item()) * n_charts
            total_charts += n_charts
            for b in range(n_charts):
                n = int(batch["num_notes"][b])
                if n == 0:
                    continue
                chart_metrics = evaluate_chart_metrics(
                    out["x_bin_logits"][b, :n],
                    out["side_logits"][b, :n],
                    batch["x_abs"][b, :n],
                    batch["x_bin"][b, :n],
                    batch["side"][b, :n],
                    batch["group_ranges"][b],
                    pos=out["pos"][b, :n],
                    center=out["center"][b, :n],
                    log_spread=out["log_spread"][b, :n],
                )
                chart_id = batch["chart_id"][b]
                per_chart[chart_id] = chart_metrics
                w = chart_metrics["num_notes"]
                total_notes += w
                for key in (
                    "x_mae", "x_rmse", "top1", "top5", "side_macro_f1", "side_acc",
                    "pos_mae", "center_mae", "spread_mae_k2", "log_spread_mae_k2",
                ):
                    metrics_agg[key] += chart_metrics[key] * w

    if total_notes:
        for key in (
            "x_mae", "x_rmse", "top1", "top5", "side_macro_f1", "side_acc",
            "pos_mae", "center_mae", "spread_mae_k2", "log_spread_mae_k2",
        ):
            metrics_agg[key] /= total_notes
    if total_charts:
        metrics_agg["loss"] /= total_charts
    metrics_agg["num_notes"] = total_notes
    metrics_agg["num_charts"] = total_charts
    return {"overall": metrics_agg, "per_chart": per_chart}


def train_and_evaluate(
    rows: Sequence[Dict[str, Any]],
    vocab: Vocab,
    features_dir: str,
    output_dir: str,
    train_song_ids: Sequence[str],
    val_song_ids: Sequence[str],
    test_song_ids: Sequence[str],
    fold: Optional[int],
    seed: int,
    use_audio: bool,
    smoke: bool = False,
    hyperparams: Optional[Dict[str, Any]] = None,
    device: Optional[torch.device] = None,
    num_workers: int = 0,
) -> Dict[str, Any]:
    hp = dict(DEFAULT_HYPERPARAMS)
    if hyperparams:
        hp.update(hyperparams)
    model_kind = str(hp.get("model_kind", "transformer"))
    if model_kind == "group_mlp":
        hp["lr"] = float((hyperparams or {}).get("lr", 1e-3))
        hp["weight_decay"] = float((hyperparams or {}).get("weight_decay", 0.0))
        hp["patience"] = int((hyperparams or {}).get("patience", 60))
        hp["aux_weight"] = float((hyperparams or {}).get("aux_weight", 1.0))
        hp["pos_weight"] = float((hyperparams or {}).get("pos_weight", 1.0))
        hp["dropout"] = float((hyperparams or {}).get("dropout", 0.0))
        hp["group_hidden"] = int((hyperparams or {}).get("group_hidden", 256))
        hp["group_layers"] = int((hyperparams or {}).get("group_layers", 3))
    if smoke:
        hp["epochs"] = min(int(hp.get("epochs", 200)), 20)
        hp["patience"] = min(int(hp.get("patience", 30)), 8)

    device = device or get_device()
    train_rows, val_rows, test_rows = split_rows(
        rows, list(test_song_ids), list(val_song_ids), list(train_song_ids)
    )
    if smoke and len(train_rows) > 2:
        train_rows = train_rows[:2]
        val_rows = val_rows[:1]
        test_rows = test_rows[:1]

    side_counts = [0, 0]
    for row in train_rows:
        for group in row.get("groups", []):
            for note in group.get("notes", []):
                side_counts[int(note.get("side_id", 0))] += 1
    side_total = max(1, sum(side_counts))
    side_class_weights = torch.tensor(
        [side_total / (2 * max(1, c)) for c in side_counts],
        dtype=torch.float32,
        device=device,
    )

    config = default_config(
        use_audio=use_audio,
        n_difficulties=vocab.num_difficulties,
        n_sources=vocab.num_sources,
        model_kind=model_kind,
    )
    config.update(vocab.to_config())
    if model_kind == "group_mlp":
        config["dropout"] = float(hp.get("dropout", 0.0))
        config["group_hidden"] = int(hp.get("group_hidden", 256))
        config["group_layers"] = int(hp.get("group_layers", 3))
    config.update(
        {
            "seed": seed,
            "fold": fold,
            "lr": float(hp["lr"]),
            "weight_decay": float(hp["weight_decay"]),
            "label_smoothing": float(hp["label_smoothing"]),
            "side_weight": float(hp["side_weight"]),
            "aux_weight": float(hp["aux_weight"]),
            "pos_weight": float(hp["pos_weight"]),
            "batch_size": int(hp["batch_size"]),
            "effective_batch": int(hp["effective_batch"]),
            "max_tokens": int(hp["max_tokens"]),
            "epochs": int(hp["epochs"]),
            "patience": int(hp["patience"]),
        }
    )

    model = build_model(config).to(device)
    n_params = model.num_parameters()

    train_ds = TapDataset(train_rows, vocab, features_dir, use_audio=use_audio)
    val_ds = TapDataset(val_rows, vocab, features_dir, use_audio=use_audio)
    test_ds = TapDataset(test_rows, vocab, features_dir, use_audio=use_audio)

    batch_size = int(hp["batch_size"])
    effective_batch = int(hp["effective_batch"])
    grad_accum = max(1, math.ceil(effective_batch / batch_size))
    train_loader = make_dataloader(
        train_ds,
        batch_size=batch_size,
        max_tokens=int(hp["max_tokens"]),
        shuffle=True,
        seed=seed,
        num_workers=num_workers,
    )
    val_loader = make_dataloader(
        val_ds,
        batch_size=batch_size,
        max_tokens=int(hp["max_tokens"]),
        shuffle=False,
        seed=seed,
        num_workers=num_workers,
    )
    test_loader = make_dataloader(
        test_ds,
        batch_size=batch_size,
        max_tokens=int(hp["max_tokens"]),
        shuffle=False,
        seed=seed,
        num_workers=num_workers,
    )

    epochs = int(hp["epochs"])
    patience = int(hp["patience"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(hp["lr"]), weight_decay=float(hp["weight_decay"])
    )
    steps_per_epoch = max(1, math.ceil(len(train_loader) / grad_accum))
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(1, int(total_steps * float(hp["warmup_frac"])))
    scheduler = build_scheduler(optimizer, total_steps, warmup_steps)
    use_amp = device.type == "cuda"

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.time()

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            scheduler,
            float(hp["side_weight"]),
            float(hp["label_smoothing"]),
            float(hp["aux_weight"]),
            float(hp["pos_weight"]),
            side_class_weights,
            grad_accum,
            float(hp["grad_clip"]),
            use_amp,
        )
        record: Dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        if model_kind == "group_mlp":
            train_eval = evaluate_model(
                model,
                train_loader,
                device,
                float(hp["side_weight"]),
                float(hp["label_smoothing"]),
                float(hp["aux_weight"]),
                float(hp["pos_weight"]),
                side_class_weights,
            )
            record["train_center_mae"] = float(train_eval["overall"]["center_mae"])
            record["train_x_mae"] = float(train_eval["overall"]["x_mae"])
            record["train_pos_mae"] = float(train_eval["overall"]["pos_mae"])
        if len(val_rows) == 0:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            history.append(record)
            continue
        val_result = evaluate_model(
            model,
            val_loader,
            device,
            float(hp["side_weight"]),
            float(hp["label_smoothing"]),
            float(hp["aux_weight"]),
            float(hp["pos_weight"]),
            side_class_weights,
        )
        val_loss = float(val_result["overall"]["loss"])
        record["val_loss"] = val_loss
        record["val_x_mae"] = float(val_result["overall"]["x_mae"])
        record["val_top1"] = float(val_result["overall"]["top1"])
        record["val_side_macro_f1"] = float(val_result["overall"]["side_macro_f1"])
        history.append(record)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break

    elapsed = time.time() - start
    peak_mem = None
    if device.type == "cuda":
        peak_mem = int(torch.cuda.max_memory_allocated(device))

    if best_state is not None:
        model.load_state_dict(best_state)

    best_train_record = next(
        (r for r in history if r.get("epoch") == best_epoch), {}
    )
    checkpoint = {
        "model_state": best_state if best_state is not None else model.state_dict(),
        "config": config,
        "fold": fold,
        "seed": seed,
        "epoch": best_epoch,
        "uses_audio": bool(use_audio),
    }
    os.makedirs(output_dir, exist_ok=True)
    fold_tag = "final" if fold is None else f"fold{fold}"
    ckpt_path = os.path.join(output_dir, f"model_{fold_tag}_seed{seed}.pt")
    torch.save(checkpoint, ckpt_path)

    test_result: Dict[str, Any] = {}
    if len(test_rows):
        test_result = evaluate_model(
            model,
            test_loader,
            device,
            float(hp["side_weight"]),
            float(hp["label_smoothing"]),
            float(hp["aux_weight"]),
            float(hp["pos_weight"]),
            side_class_weights,
        )

    baselines = evaluate_baselines(train_rows, test_rows)

    report: Dict[str, Any] = {
        "fold": fold,
        "seed": seed,
        "uses_audio": use_audio,
        "device": str(device),
        "model_kind": model_kind,
        "num_parameters": n_params,
        "train_charts": len(train_rows),
        "val_charts": len(val_rows),
        "test_charts": len(test_rows),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss if len(val_rows) else None,
        "epochs_trained": len(history),
        "train_center_mae": best_train_record.get("train_center_mae"),
        "train_x_mae": best_train_record.get("train_x_mae"),
        "train_pos_mae": best_train_record.get("train_pos_mae"),
        "elapsed_seconds": elapsed,
        "peak_memory_bytes": peak_mem,
        "checkpoint": ckpt_path,
        "config": config,
        "history": history,
        "test": test_result,
        "baselines": baselines,
    }
    report_path = os.path.join(output_dir, f"fold_{fold_tag}_seed{seed}.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, default=_json_default)
    return report


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def gpu_forward_backward_smoke(
    device: torch.device,
    use_audio: bool = True,
    vocab: Optional[Vocab] = None,
    model_kind: str = "transformer",
) -> Dict[str, Any]:
    vocab = vocab or Vocab(["IN", "HD", "EZ", "AT", "SP"], ["official", "special", "community"])
    config = default_config(
        use_audio=use_audio,
        n_difficulties=vocab.num_difficulties,
        n_sources=vocab.num_sources,
        model_kind=model_kind,
    )
    if model_kind == "group_mlp":
        config["dropout"] = 0.0
        config["group_hidden"] = 256
        config["group_layers"] = 3
    model = build_model(config).to(device)
    b, n = 2, 64
    batch = {
        "scalars": torch.randn(b, n, 7, device=device),
        "type_idx": torch.randint(0, 4, (b, n), device=device),
        "diff_idx": torch.randint(0, vocab.num_difficulties, (b, n), device=device),
        "src_idx": torch.randint(0, vocab.num_sources, (b, n), device=device),
        "audio": torch.randn(b, n, 9, 64, device=device),
        "times": torch.rand(b, n, device=device) * 120.0,
        "dts": torch.rand(b, n, device=device) + 0.02,
    }
    mask = torch.ones(b, n, dtype=torch.bool, device=device)
    mask[1, n // 2 :] = False
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")
    with amp:
        out = model(
            batch["scalars"],
            batch["type_idx"],
            batch["diff_idx"],
            batch["src_idx"],
            batch["audio"],
            batch["times"],
            batch["dts"],
            src_key_padding_mask=(~mask),
        )
        loss = out["x_bin_logits"].float().mean() + out["side_logits"].float().mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    keys = sorted(out.keys())
    return {
        "status": "ok",
        "output_keys": keys,
        "x_bin_logits_shape": list(out["x_bin_logits"].shape),
        "side_logits_shape": list(out["side_logits"].shape),
        "num_parameters": model.num_parameters(),
        "grad_ok": bool(torch.isfinite(loss).item()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PhiTapMicro for one CV fold.")
    parser.add_argument("--dataset", default="data/processed/dataset.jsonl")
    parser.add_argument("--splits", default="data/processed/splits.json")
    parser.add_argument("--features-dir", default="data/processed/features")
    parser.add_argument("--output", default="artifacts")
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    rows = load_jsonl(args.dataset)
    vocab = Vocab.from_rows(rows)
    use_audio = not args.no_audio

    device = get_device(args.device)
    print(json.dumps(check_gpu(), ensure_ascii=False, indent=2))

    if args.smoke:
        smoke_result = gpu_forward_backward_smoke(device, use_audio, vocab)
        print(json.dumps({"gpu_forward_backward_smoke": smoke_result}, indent=2))

    hyperparams: Dict[str, Any] = {}
    if args.epochs is not None:
        hyperparams["epochs"] = args.epochs
    if args.batch_size is not None:
        hyperparams["batch_size"] = args.batch_size
    if args.lr is not None:
        hyperparams["lr"] = args.lr

    if args.fold is not None:
        with open(args.splits, "r", encoding="utf-8") as handle:
            splits = json.load(handle)
        fold_info = next(f for f in splits["folds"] if f["fold"] == args.fold)
        train_song_ids = fold_info["train_song_ids"]
        val_song_ids = fold_info["inner_val_song_ids"]
        test_song_ids = fold_info["test_song_ids"]
    else:
        song_ids = sorted({r["song_id"] for r in rows})
        if len(song_ids) <= 1:
            train_song_ids = song_ids
            val_song_ids: List[str] = []
            test_song_ids: List[str] = []
        else:
            val_song_ids = song_ids[:1]
            train_song_ids = song_ids[1:]
            test_song_ids: List[str] = []

    report = train_and_evaluate(
        rows,
        vocab,
        args.features_dir,
        os.path.join(args.output, "checkpoints"),
        train_song_ids,
        val_song_ids,
        test_song_ids,
        args.fold,
        args.seed,
        use_audio,
        smoke=args.smoke,
        hyperparams=hyperparams,
        device=device,
        num_workers=args.num_workers,
    )
    print(json.dumps(
        {k: v for k, v in report.items() if k not in ("history", "test", "config")},
        ensure_ascii=False,
        indent=2,
        default=_json_default,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
