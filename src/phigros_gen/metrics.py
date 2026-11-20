"""Losses, matched metrics, and non-learned baselines."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from phigros_gen.model import bin_centers


N_BINS = 128


def _assignment(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if cost.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    return linear_sum_assignment(cost)


def label_smoothed_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smoothing: float = 0.02,
) -> torch.Tensor:
    logits = logits.float()
    log_probs = F.log_softmax(logits, dim=-1)
    n_classes = logits.shape[-1]
    with torch.no_grad():
        one_hot = torch.zeros_like(log_probs)
        one_hot.scatter_(1, targets.unsqueeze(1), 1.0)
        smooth = (1.0 - smoothing) * one_hot + smoothing / n_classes
    return -(smooth * log_probs).sum(dim=-1).mean()


def hungarian_assignments(
    x_logits: torch.Tensor,
    x_bin_targets: torch.Tensor,
    group_ranges: Sequence[Tuple[int, int]],
) -> List[np.ndarray]:
    """Return, per group, the target index assigned to each slot."""
    log_probs = F.log_softmax(x_logits.float(), dim=-1).detach().cpu().numpy()
    targets = x_bin_targets.detach().cpu().numpy()
    assignments: List[np.ndarray] = []
    for start, end in group_ranges:
        if end <= start:
            assignments.append(np.array([], dtype=np.int64))
            continue
        slots = log_probs[start:end]
        tgt = targets[start:end]
        cost = -slots[:, tgt]
        row, col = _assignment(cost)
        assignments.append(col.astype(np.int64))
    return assignments


def hungarian_loss(
    x_logits: torch.Tensor,
    side_logits: torch.Tensor,
    x_bin: torch.Tensor,
    side: torch.Tensor,
    group_ranges: Sequence[Tuple[int, int]],
    side_weight: float = 0.2,
    label_smoothing: float = 0.02,
    side_class_weights: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Compute ``CE_x + side_weight * CE_side`` with a shared assignment."""
    x_bin = x_bin.to(x_logits.device)
    side = side.to(side_logits.device)
    assignments = hungarian_assignments(x_logits, x_bin, group_ranges)
    x_losses: List[torch.Tensor] = []
    side_losses: List[torch.Tensor] = []
    for gi, (start, end) in enumerate(group_ranges):
        col = assignments[gi]
        if len(col) == 0:
            continue
        slot_x = x_logits[start:end]
        slot_side = side_logits[start:end]
        target_x = x_bin[start:end][col]
        target_side = side[start:end][col]
        x_losses.append(label_smoothed_ce(slot_x, target_x, label_smoothing))
        weight = None if side_class_weights is None else side_class_weights.to(slot_side.device)
        side_losses.append(F.cross_entropy(slot_side, target_side, weight=weight))
    if not x_losses:
        return {
            "x_loss": torch.zeros((), device=x_logits.device),
            "side_loss": torch.zeros((), device=side_logits.device),
            "total": torch.zeros((), device=x_logits.device),
            "assignments": assignments,
        }
    x_loss = torch.stack(x_losses).mean()
    side_loss = torch.stack(side_losses).mean()
    return {
        "x_loss": x_loss,
        "side_loss": side_loss,
        "total": x_loss + side_weight * side_loss,
        "assignments": assignments,
    }


def template_aux_loss(
    center_pred: torch.Tensor,
    log_spread_pred: torch.Tensor,
    x_abs: torch.Tensor,
    group_ranges: Sequence[Tuple[int, int]],
) -> torch.Tensor:
    """Supervise group center/spread against a sorted-x template fit.

    Single-note groups only supervise center; their spread is undefined.
    """
    losses: List[torch.Tensor] = []
    for start, end in group_ranges:
        y = x_abs[start:end].float()
        k = y.numel()
        if k == 0:
            continue
        if k == 1:
            losses.append((center_pred[start] - y[0]).abs())
            continue
        u = torch.arange(k, device=y.device, dtype=torch.float32)
        u = (u + 1.0) / (k + 1.0) - 0.5
        y_sorted, _ = torch.sort(y)
        u_mean = u.mean()
        y_mean = y_sorted.mean()
        var_u = ((u - u_mean) ** 2).sum().clamp_min(1e-8)
        spread = ((u - u_mean) * (y_sorted - y_mean)).sum() / var_u
        spread_target = spread.clamp_min(1e-2)
        center_target = y_mean - spread_target * u_mean
        log_spread_target = torch.log(spread_target)
        losses.append(
            (center_pred[start] - center_target).abs()
            + (log_spread_pred[start] - log_spread_target).abs()
        )
    if not losses:
        return torch.zeros((), device=center_pred.device)
    return torch.stack(losses).mean()


def matched_pos_l1(
    pos: torch.Tensor,
    x_abs: torch.Tensor,
    group_ranges: Sequence[Tuple[int, int]],
    assignments: Sequence[np.ndarray],
) -> torch.Tensor:
    """Hungarian-matched L1 between predicted positions and target x."""
    total = torch.zeros((), device=pos.device)
    count = 0
    for gi, (start, end) in enumerate(group_ranges):
        col = assignments[gi]
        if len(col) == 0:
            continue
        pred = pos[start:end]
        target = x_abs[start:end][col]
        total = total + (pred - target).abs().sum()
        count += int(pred.numel())
    return total / max(1, count)


def macro_f1(y_true: torch.Tensor, y_pred: torch.Tensor, n_classes: int = 2) -> float:
    y_true = y_true.long()
    y_pred = y_pred.long()
    eps = 1e-8
    f1s = []
    for c in range(n_classes):
        tp = torch.sum((y_pred == c) & (y_true == c)).float()
        fp = torch.sum((y_pred == c) & (y_true != c)).float()
        fn = torch.sum((y_pred != c) & (y_true == c)).float()
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1s.append((2.0 * precision * recall / (precision + recall + eps)).item())
    return float(np.mean(f1s))


def evaluate_chart_metrics(
    x_logits: torch.Tensor,
    side_logits: torch.Tensor,
    x_abs: torch.Tensor,
    x_bin: torch.Tensor,
    side: torch.Tensor,
    group_ranges: Sequence[Tuple[int, int]],
    n_bins: int = N_BINS,
    pos: Optional[torch.Tensor] = None,
    center: Optional[torch.Tensor] = None,
    log_spread: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Evaluate one chart with Hungarian-aligned targets."""
    x_abs = x_abs.to(x_logits.device)
    x_bin = x_bin.to(x_logits.device)
    side = side.to(side_logits.device)
    assignments = hungarian_assignments(x_logits, x_bin, group_ranges)
    x_pred_bin = torch.argmax(x_logits, dim=-1)
    bin_c = bin_centers(n_bins).to(x_logits.device)
    x_pred_abs = bin_c[x_pred_bin]
    side_pred = torch.argmax(side_logits, dim=-1)

    abs_errors: List[float] = []
    sq_errors: List[float] = []
    pos_errors: List[float] = []
    center_errors: List[float] = []
    spread_errors_k2: List[float] = []
    log_spread_errors_k2: List[float] = []
    top1 = 0
    top5 = 0
    side_true_all: List[torch.Tensor] = []
    side_pred_all: List[torch.Tensor] = []
    total = 0
    topk = torch.topk(x_logits, k=min(5, n_bins), dim=-1).indices

    for gi, (start, end) in enumerate(group_ranges):
        col = assignments[gi]
        if len(col) == 0:
            continue
        slot_ids = torch.arange(start, end, device=x_logits.device)
        matched_target_x = x_abs[start:end][col]
        matched_target_bin = x_bin[start:end][col]
        matched_target_side = side[start:end][col]
        pred_abs = x_pred_abs[start:end]
        err = (pred_abs - matched_target_x).abs()
        abs_errors.extend(err.tolist())
        sq_errors.extend((err ** 2).tolist())
        if pos is not None:
            pos_errors.extend((pos[start:end] - matched_target_x).abs().tolist())
        top1 += int(torch.sum(x_pred_bin[start:end] == matched_target_bin).item())
        hit5 = topk[start:end] == matched_target_bin.unsqueeze(1)
        top5 += int(hit5.any(dim=1).sum().item())
        side_true_all.append(matched_target_side)
        side_pred_all.append(side_pred[start:end])
        total += len(col)

        if center is not None:
            y = x_abs[start:end]
            k = y.numel()
            if k == 1:
                center_target = y[0]
            else:
                u = torch.arange(k, device=y.device, dtype=torch.float32)
                u = (u + 1.0) / (k + 1.0) - 0.5
                y_sorted, _ = torch.sort(y)
                u_mean = u.mean()
                y_mean = y_sorted.mean()
                var_u = ((u - u_mean) ** 2).sum().clamp_min(1e-8)
                spread = ((u - u_mean) * (y_sorted - y_mean)).sum() / var_u
                spread_target = spread.clamp_min(1e-2)
                center_target = y_mean - spread_target * u_mean
                if log_spread is not None:
                    spread_errors_k2.append(
                        (torch.exp(log_spread[start]) - spread_target).abs().item()
                    )
                    log_spread_errors_k2.append(
                        (log_spread[start] - torch.log(spread_target)).abs().item()
                    )
            center_errors.append((center[start] - center_target).abs().item())

    if total == 0:
        return {
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
            "num_notes": 0,
            "num_groups": len(group_ranges),
        }

    side_true = torch.cat(side_true_all)
    side_pred_cat = torch.cat(side_pred_all)
    result: Dict[str, float] = {
        "x_mae": float(np.mean(abs_errors)),
        "x_rmse": float(np.sqrt(np.mean(sq_errors))),
        "top1": top1 / total,
        "top5": top5 / total,
        "side_macro_f1": macro_f1(side_true, side_pred_cat),
        "side_acc": float((side_true == side_pred_cat).float().mean().item()),
        "num_notes": total,
        "num_groups": len(group_ranges),
    }
    result["pos_mae"] = float(np.mean(pos_errors)) if pos_errors else 0.0
    result["center_mae"] = float(np.mean(center_errors)) if center_errors else 0.0
    result["spread_mae_k2"] = (
        float(np.mean(spread_errors_k2)) if spread_errors_k2 else 0.0
    )
    result["log_spread_mae_k2"] = (
        float(np.mean(log_spread_errors_k2)) if log_spread_errors_k2 else 0.0
    )
    return result


def matched_x_mae(
    pred_x: np.ndarray,
    true_x: np.ndarray,
    group_ranges: Sequence[Tuple[int, int]],
) -> float:
    """Hungarian-matched x MAE between slot predictions and targets."""
    total_err = 0.0
    total_n = 0
    for start, end in group_ranges:
        p = np.asarray(pred_x[start:end], dtype=np.float64)
        t = np.asarray(true_x[start:end], dtype=np.float64)
        if len(p) == 0:
            continue
        cost = np.abs(p[:, None] - t[None, :])
        row, col = _assignment(cost)
        total_err += float(cost[row, col].sum())
        total_n += len(p)
    return total_err / max(1, total_n)


def flatten_chart_notes(
    rows: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[List[Tuple[int, int]]], List[Dict[str, Any]]]:
    """Flatten rows into per-note records while retaining group boundaries."""
    note_infos: List[Dict[str, Any]] = []
    group_ranges_all: List[List[Tuple[int, int]]] = []
    chart_meta: List[Dict[str, Any]] = []
    cursor = 0
    for row in rows:
        ranges: List[Tuple[int, int]] = []
        for group in row.get("groups", []):
            notes = sorted(group.get("notes", []), key=lambda n: int(n.get("slot_id", 0)))
            start = cursor
            for note in notes:
                note_infos.append(
                    {
                        "chart_id": row["chart_id"],
                        "song_id": row.get("song_id", row["chart_id"]),
                        "difficulty": row.get("difficulty", "unknown"),
                        "type_id": int(note.get("type_id", 1)),
                        "x_abs": float(note.get("x_abs", 0.5)),
                        "side_id": int(note.get("side_id", 0)),
                        "time_sec": float(note.get("time_sec", group.get("time_sec", 0.0))),
                        "slot_id": int(note.get("slot_id", 0)),
                    }
                )
                cursor += 1
            if cursor > start:
                ranges.append((start, cursor))
        group_ranges_all.append(ranges)
        chart_meta.append(
            {
                "chart_id": row["chart_id"],
                "song_id": row.get("song_id", row["chart_id"]),
                "difficulty": row.get("difficulty", "unknown"),
            }
        )
    return note_infos, group_ranges_all, chart_meta


def fit_baseline_stats(train_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Fit non-learned baselines using only training charts."""
    note_infos, _, _ = flatten_chart_notes(train_rows)
    if not note_infos:
        return {
            "global_median": 0.5,
            "grouped_median": {},
            "difficulty_median": {},
            "type_median": {},
            "majority_side": 0,
        }
    xs = np.array([n["x_abs"] for n in note_infos], dtype=np.float64)
    grouped: Dict[Tuple[str, int], List[float]] = {}
    difficulty: Dict[str, List[float]] = {}
    type_med: Dict[int, List[float]] = {}
    sides: List[int] = []
    for n in note_infos:
        key = (str(n["difficulty"]), int(n["type_id"]))
        grouped.setdefault(key, []).append(n["x_abs"])
        difficulty.setdefault(str(n["difficulty"]), []).append(n["x_abs"])
        type_med.setdefault(int(n["type_id"]), []).append(n["x_abs"])
        sides.append(int(n["side_id"]))
    return {
        "global_median": float(np.median(xs)),
        "grouped_median": {f"{k[0]}|{k[1]}": float(np.median(v)) for k, v in grouped.items()},
        "difficulty_median": {k: float(np.median(v)) for k, v in difficulty.items()},
        "type_median": {k: float(np.median(v)) for k, v in type_med.items()},
        "majority_side": int(np.bincount(sides, minlength=2).argmax()),
    }


def evaluate_baselines(
    train_rows: Sequence[Dict[str, Any]],
    test_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute x MAE for the four required baselines on held-out charts."""
    stats = fit_baseline_stats(train_rows)
    test_notes, test_ranges, test_meta = flatten_chart_notes(test_rows)
    if not test_notes:
        return {
            "baseline_x_mae": {},
            "baseline_side_macro_f1": {},
            "majority_side": stats["majority_side"],
        }

    x_all = np.array([n["x_abs"] for n in test_notes], dtype=np.float64)
    side_all = np.array([n["side_id"] for n in test_notes], dtype=np.int64)

    preds: Dict[str, np.ndarray] = {}
    preds["global_median"] = np.full(len(test_notes), stats["global_median"])
    preds["grouped_median"] = np.array(
        [
            stats["grouped_median"].get(
                f"{n['difficulty']}|{n['type_id']}", stats["global_median"]
            )
            for n in test_notes
        ],
        dtype=np.float64,
    )
    preds["difficulty_median"] = np.array(
        [
            stats["difficulty_median"].get(n["difficulty"], stats["global_median"])
            for n in test_notes
        ],
        dtype=np.float64,
    )
    preds["type_median"] = np.array(
        [
            stats["type_median"].get(n["type_id"], stats["global_median"])
            for n in test_notes
        ],
        dtype=np.float64,
    )

    # Equidistant spread within each group.
    eq = np.zeros(len(test_notes), dtype=np.float64)
    for ranges in test_ranges:
        for start, end in ranges:
            k = end - start
            if k <= 0:
                continue
            eq[start:end] = (np.arange(k) + 1.0) / (k + 1.0)
    preds["equidistant"] = eq

    # Nearest-previous-note copy within each chart.
    near = np.zeros(len(test_notes), dtype=np.float64)
    for ranges in test_ranges:
        prev = 0.5
        flat = sorted(
            [(start, end) for start, end in ranges], key=lambda r: r[0]
        )
        for start, end in flat:
            for i in range(start, end):
                near[i] = prev
            if end > start:
                prev = float(x_all[end - 1])
    preds["nearest_copy"] = near

    baseline_mae = {
        name: matched_x_mae(pred, x_all, _ranges_for_notes(test_ranges))
        for name, pred in preds.items()
    }

    # Side majority baseline macro-F1.
    majority = stats["majority_side"]
    side_pred = np.full(len(side_all), majority, dtype=np.int64)
    baseline_side_f1 = {
        "majority": _numpy_macro_f1(side_all, side_pred),
    }

    return {
        "baseline_x_mae": baseline_mae,
        "baseline_side_macro_f1": baseline_side_f1,
        "majority_side": majority,
        "strongest_baseline": min(baseline_mae, key=baseline_mae.get),
        "strongest_baseline_mae": min(baseline_mae.values()),
    }


def _ranges_for_notes(
    group_ranges_all: Sequence[Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    return [r for ranges in group_ranges_all for r in ranges]


def _numpy_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 2) -> float:
    eps = 1e-8
    f1s = []
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1s.append(2.0 * precision * recall / (precision + recall + eps))
    return float(np.mean(f1s))
