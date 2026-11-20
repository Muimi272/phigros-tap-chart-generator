"""PhiTapMicro: a tiny transformer for chart x/side prediction."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


DEFAULT_MODEL_CONFIG = {
    "d_model": 128,
    "n_heads": 4,
    "ffn_dim": 512,
    "dropout": 0.1,
    "n_layers": 4,
    "n_bins": 128,
    "n_types": 4,
    "n_difficulties": 8,
    "n_sources": 4,
    "use_audio": True,
    "n_scalars": 7,
    "audio_window": 9,
    "audio_mels": 64,
    "residual_scale": 0.0,
    "x_sigma": 0.08,
}


class TimeIntervalPositionalEncoding(nn.Module):
    """Learnable positional encoding driven by absolute time and time gaps.

    Fixed log-spaced sinusoidal bands are computed from both ``times`` and
    ``dts`` and projected to ``d_model``.  The projection weights are learned,
    so the scheme adapts to chart rhythm while remaining length-agnostic.
    """

    def __init__(
        self,
        d_model: int,
        max_period: float = 4096.0,
        min_period: float = 1.0 / 64.0,
    ):
        super().__init__()
        self.d_model = d_model
        n_bands = max(1, d_model // 4)
        freqs = np.geomspace(1.0 / max_period, 1.0 / min_period, n_bands)
        self.register_buffer("freqs", torch.tensor(freqs, dtype=torch.float32))
        self.proj = nn.Linear(4 * n_bands, d_model)

    def forward(self, times: torch.Tensor, dts: torch.Tensor) -> torch.Tensor:
        t_phase = times.unsqueeze(-1) * self.freqs * (2.0 * math.pi)
        d_phase = dts.unsqueeze(-1) * self.freqs * (2.0 * math.pi)
        feats = torch.cat(
            [torch.sin(t_phase), torch.cos(t_phase), torch.sin(d_phase), torch.cos(d_phase)],
            dim=-1,
        )
        return self.proj(feats)


class AudioMLP(nn.Module):
    def __init__(self, in_features: int, hidden: int, out_features: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_features),
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        flat = audio.flatten(start_dim=-2)
        return self.net(flat)


class PhiTapMicro(nn.Module):
    """4-layer transformer encoder with x-bin and side heads only."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = dict(config)
        d_model = int(config.get("d_model", 128))
        dropout = float(config.get("dropout", 0.1))
        n_scalars = int(config.get("n_scalars", 7))
        self.n_bins = int(config.get("n_bins", 128))
        self.use_audio = bool(config.get("use_audio", True))

        self.scalar_proj = nn.Linear(n_scalars, d_model)
        self.type_emb = nn.Embedding(int(config.get("n_types", 4)), d_model)
        self.diff_emb = nn.Embedding(int(config.get("n_difficulties", 8)), d_model)
        self.src_emb = nn.Embedding(int(config.get("n_sources", 4)), d_model)

        audio_window = int(config.get("audio_window", 9))
        audio_mels = int(config.get("audio_mels", 64))
        self.audio_mlp = AudioMLP(
            audio_window * audio_mels,
            hidden=int(config.get("audio_hidden", 256)),
            out_features=d_model,
            dropout=dropout,
        )

        self.time_pe = TimeIntervalPositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=int(config.get("n_heads", 4)),
            dim_feedforward=int(config.get("ffn_dim", 512)),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=int(config.get("n_layers", 4))
        )
        self.side_head = nn.Linear(d_model, 2)
        self.group_center_head = nn.Linear(d_model, 1)
        self.group_spread_head = nn.Linear(d_model, 1)
        self.residual_head = nn.Linear(d_model, 1)
        self.register_buffer(
            "residual_scale",
            torch.tensor(float(config.get("residual_scale", 0.0)), dtype=torch.float32),
        )
        self.register_buffer(
            "x_sigma",
            torch.tensor(float(config.get("x_sigma", 0.08)), dtype=torch.float32),
        )

    def forward(
        self,
        scalars: torch.Tensor,
        type_idx: torch.Tensor,
        diff_idx: torch.Tensor,
        src_idx: torch.Tensor,
        audio: torch.Tensor,
        times: torch.Tensor,
        dts: torch.Tensor,
        u_slot: Optional[torch.Tensor] = None,
        group_ids: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        full = self.forward_full(
            scalars,
            type_idx,
            diff_idx,
            src_idx,
            audio,
            times,
            dts,
            u_slot=u_slot,
            group_ids=group_ids,
            src_key_padding_mask=src_key_padding_mask,
        )
        return {
            "x_bin_logits": full["x_bin_logits"],
            "side_logits": full["side_logits"],
        }

    def forward_full(
        self,
        scalars: torch.Tensor,
        type_idx: torch.Tensor,
        diff_idx: torch.Tensor,
        src_idx: torch.Tensor,
        audio: torch.Tensor,
        times: torch.Tensor,
        dts: torch.Tensor,
        u_slot: Optional[torch.Tensor] = None,
        group_ids: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        x = self.scalar_proj(scalars)
        x = x + self.type_emb(type_idx) + self.diff_emb(diff_idx) + self.src_emb(src_idx)
        if self.use_audio:
            x = x + self.audio_mlp(audio)
        x = x + self.time_pe(times, dts)
        x = self.dropout(x)
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        side_logits = self.side_head(x)
        residual = torch.tanh(self.residual_head(x)) * torch.abs(self.residual_scale)

        b, n, _ = x.shape
        device = x.device
        if group_ids is None:
            group_ids = torch.arange(n, device=device).unsqueeze(0).expand(b, n)
        if u_slot is None:
            u_slot = torch.zeros(b, n, 1, device=device, dtype=x.dtype)
        else:
            u_slot = u_slot.unsqueeze(-1)

        n_groups = int(group_ids.max().item()) + 1
        valid_mask = None if src_key_padding_mask is None else (~src_key_padding_mask)
        group_repr = self._group_mean(x, group_ids, n_groups, valid_mask)
        group_center = torch.sigmoid(self.group_center_head(group_repr))  # [B,G,1]
        group_log_spread = torch.clamp(
            self.group_spread_head(group_repr), -3.0, 2.0
        )  # [B,G,1]

        gather_idx = group_ids.unsqueeze(-1).expand(-1, -1, 1)
        center = torch.gather(group_center, 1, gather_idx)
        log_spread = torch.gather(group_log_spread, 1, gather_idx)
        pos = center + torch.exp(log_spread) * u_slot + residual

        bin_c = bin_centers(self.n_bins).to(device).view(1, 1, -1)
        diff = bin_c - pos
        x_bin_logits = -0.5 * (diff / torch.abs(self.x_sigma)) ** 2

        return {
            "x_bin_logits": x_bin_logits,
            "side_logits": side_logits,
            "pos": pos.squeeze(-1),
            "center": center.squeeze(-1),
            "log_spread": log_spread.squeeze(-1),
            "residual": residual.squeeze(-1),
        }

    @staticmethod
    def _group_mean(
        x: torch.Tensor,
        group_ids: torch.Tensor,
        n_groups: int,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, n, d = x.shape
        device = x.device
        if valid_mask is None:
            valid_mask = torch.ones(b, n, dtype=torch.bool, device=device)
        weight = valid_mask.unsqueeze(-1).to(x.dtype)
        masked_x = x * weight
        out = torch.zeros(b, n_groups, d, device=device, dtype=x.dtype)
        counts = torch.zeros(b, n_groups, 1, device=device, dtype=x.dtype)
        idx = group_ids.unsqueeze(-1).expand(-1, -1, d)
        out.scatter_add_(1, idx, masked_x)
        counts.scatter_add_(
            1, group_ids.unsqueeze(-1).expand(-1, -1, 1), weight
        )
        return out / counts.clamp_min(1.0)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class PhiTapGroupMLP(nn.Module):
    """Explicit group-aggregated MLP for center/spread + per-note side."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = dict(config)
        d_model = int(config.get("d_model", 128))
        dropout = float(config.get("dropout", 0.0))
        self.n_bins = int(config.get("n_bins", 128))
        self.use_audio = bool(config.get("use_audio", True))

        self.scalar_proj = nn.Linear(int(config.get("n_scalars", 7)), d_model)
        self.type_emb = nn.Embedding(int(config.get("n_types", 4)), d_model)
        self.diff_emb = nn.Embedding(int(config.get("n_difficulties", 8)), d_model)
        self.src_emb = nn.Embedding(int(config.get("n_sources", 4)), d_model)
        self.side_head = nn.Linear(d_model, 2)

        audio_mels = int(config.get("audio_mels", 64))
        audio_window = int(config.get("audio_window", 9))
        group_in = 6 + 4 + 3 * audio_window * audio_mels + 2 * d_model
        hidden = int(config.get("group_hidden", 256))
        layers = [nn.Linear(group_in, hidden), nn.ReLU()]
        for _ in range(max(1, int(config.get("group_layers", 3)) - 1)):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        if dropout > 0:
            layers = [layers[0]] + [nn.Dropout(dropout)] + layers[1:]
        self.group_mlp = nn.Sequential(*layers)
        self.center_head = nn.Linear(hidden, 1)
        self.spread_head = nn.Linear(hidden, 1)
        self.register_buffer(
            "x_sigma", torch.tensor(float(config.get("x_sigma", 0.08)), dtype=torch.float32)
        )

    def forward(
        self,
        scalars: torch.Tensor,
        type_idx: torch.Tensor,
        diff_idx: torch.Tensor,
        src_idx: torch.Tensor,
        audio: torch.Tensor,
        times: torch.Tensor,
        dts: torch.Tensor,
        u_slot: Optional[torch.Tensor] = None,
        group_ids: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        full = self.forward_full(
            scalars, type_idx, diff_idx, src_idx, audio, times, dts,
            u_slot=u_slot, group_ids=group_ids,
            src_key_padding_mask=src_key_padding_mask,
        )
        return {"x_bin_logits": full["x_bin_logits"], "side_logits": full["side_logits"]}

    def forward_full(
        self,
        scalars: torch.Tensor,
        type_idx: torch.Tensor,
        diff_idx: torch.Tensor,
        src_idx: torch.Tensor,
        audio: torch.Tensor,
        times: torch.Tensor,
        dts: torch.Tensor,
        u_slot: Optional[torch.Tensor] = None,
        group_ids: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        b, n, _ = scalars.shape
        device = scalars.device
        d_model = int(self.config.get("d_model", 128))

        side_feat = (
            self.scalar_proj(scalars)
            + self.type_emb(type_idx)
            + self.diff_emb(diff_idx)
            + self.src_emb(src_idx)
        )
        side_logits = self.side_head(side_feat)

        if group_ids is None:
            group_ids = torch.arange(n, device=device).unsqueeze(0).expand(b, n)
        if u_slot is None:
            u_slot = torch.zeros(b, n, 1, device=device, dtype=scalars.dtype)
        else:
            u_slot = u_slot.unsqueeze(-1)

        valid = torch.ones(b, n, dtype=torch.bool, device=device)
        if src_key_padding_mask is not None:
            valid = ~src_key_padding_mask

        n_groups = int(group_ids.max().item()) + 1
        counts = self._group_count(group_ids, valid, n_groups).squeeze(-1).long()
        group_start = torch.cumsum(counts, dim=1) - counts
        group_end = group_start + counts - 1

        group_time = self._group_mean(times.unsqueeze(-1), group_ids, valid, n_groups)
        bpm_scaled = self._group_mean(scalars[:, :, 6:7], group_ids, valid, n_groups)
        density_scaled = self._group_mean(scalars[:, :, 5:6], group_ids, valid, n_groups)
        log_group_size = torch.log(counts.clamp_min(1).float().unsqueeze(-1))

        type_onehot = torch.nn.functional.one_hot(
            type_idx, num_classes=int(self.config.get("n_types", 4))
        ).float()
        type_ratio = self._group_sum(type_onehot, group_ids, valid, n_groups) / counts.clamp_min(1).unsqueeze(-1)

        audio_flat = audio.flatten(start_dim=-2)
        audio_mean = self._group_mean(audio_flat, group_ids, valid, n_groups)
        audio_first = self._gather_idx(audio_flat, group_start.clamp(0, n - 1))
        audio_last = self._gather_idx(audio_flat, group_end.clamp(0, n - 1))

        dt_prev = torch.zeros(b, n_groups, 1, device=device, dtype=scalars.dtype)
        dt_next = torch.zeros(b, n_groups, 1, device=device, dtype=scalars.dtype)
        if n_groups > 1:
            dt_prev[:, 1:] = (group_time[:, 1:] - group_time[:, :-1]).clamp_min(0.0)
            dt_next[:, :-1] = (group_time[:, 1:] - group_time[:, :-1]).clamp_min(0.0)

        diff_emb_g = self._group_mean(self.diff_emb(diff_idx), group_ids, valid, n_groups)
        src_emb_g = self._group_mean(self.src_emb(src_idx), group_ids, valid, n_groups)

        group_feat = torch.cat(
            [
                group_time,
                bpm_scaled,
                log_group_size,
                density_scaled,
                torch.log1p(dt_prev),
                torch.log1p(dt_next),
                type_ratio,
                audio_mean,
                audio_first,
                audio_last,
                diff_emb_g,
                src_emb_g,
            ],
            dim=-1,
        )
        h = self.group_mlp(group_feat)
        group_center = torch.sigmoid(self.center_head(h))
        group_log_spread = torch.clamp(self.spread_head(h), -3.0, 2.0)

        gather_idx = group_ids.unsqueeze(-1).expand(-1, -1, 1)
        center = torch.gather(group_center, 1, gather_idx)
        log_spread = torch.gather(group_log_spread, 1, gather_idx)
        pos = center + torch.exp(log_spread) * u_slot

        bin_c = bin_centers(self.n_bins).to(device).view(1, 1, -1)
        x_bin_logits = -0.5 * ((bin_c - pos) / torch.abs(self.x_sigma)) ** 2

        return {
            "x_bin_logits": x_bin_logits,
            "side_logits": side_logits,
            "pos": pos.squeeze(-1),
            "center": center.squeeze(-1),
            "log_spread": log_spread.squeeze(-1),
        }

    @staticmethod
    def _group_count(
        group_ids: torch.Tensor, valid: torch.Tensor, n_groups: int
    ) -> torch.Tensor:
        b, n = group_ids.shape
        weight = valid.unsqueeze(-1).to(torch.float32)
        out = torch.zeros(b, n_groups, 1, device=group_ids.device)
        out.scatter_add_(1, group_ids.unsqueeze(-1).expand(-1, -1, 1), weight)
        return out

    @staticmethod
    def _group_sum(
        x: torch.Tensor, group_ids: torch.Tensor, valid: torch.Tensor, n_groups: int
    ) -> torch.Tensor:
        b, n, c = x.shape
        weight = valid.unsqueeze(-1).to(x.dtype)
        out = torch.zeros(b, n_groups, c, device=x.device, dtype=x.dtype)
        out.scatter_add_(1, group_ids.unsqueeze(-1).expand(-1, -1, c), x * weight)
        return out

    @staticmethod
    def _group_mean(
        x: torch.Tensor, group_ids: torch.Tensor, valid: torch.Tensor, n_groups: int
    ) -> torch.Tensor:
        total = PhiTapGroupMLP._group_sum(x, group_ids, valid, n_groups)
        counts = PhiTapGroupMLP._group_count(group_ids, valid, n_groups)
        return total / counts.clamp_min(1.0)

    @staticmethod
    def _gather_idx(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        b, _, c = x.shape
        return torch.gather(x, 1, idx.unsqueeze(-1).expand(b, -1, c))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(config: Dict[str, Any]) -> nn.Module:
    if config.get("model_kind") == "group_mlp":
        return PhiTapGroupMLP(config)
    return PhiTapMicro(config)


def default_config(
    use_audio: bool = True,
    n_difficulties: int = 8,
    n_sources: int = 4,
    **overrides: Any,
) -> Dict[str, Any]:
    config = dict(DEFAULT_MODEL_CONFIG)
    config.setdefault("model_kind", "transformer")
    config["use_audio"] = bool(use_audio)
    config["n_difficulties"] = int(n_difficulties)
    config["n_sources"] = int(n_sources)
    config.update(overrides)
    return config


def bin_centers(n_bins: int = 128) -> torch.Tensor:
    return torch.arange(n_bins, dtype=torch.float32).add_(0.5).div_(n_bins)


def predict_from_logits(
    x_logits: torch.Tensor,
    side_logits: torch.Tensor,
    group_ranges: Optional[Sequence[Tuple[int, int]]] = None,
) -> Dict[str, torch.Tensor]:
    """Convert logits to per-note predictions.

    Returns ``x_bin``, ``x_abs``, ``side_id``.  If ``group_ranges`` is given,
    it also returns ``sorted_idx``: within each group the notes are reordered by
    predicted x, which is the deterministic order used when writing charts.
    """
    x_bin = torch.argmax(x_logits, dim=-1)
    centers = bin_centers(x_logits.shape[-1]).to(x_logits.device)
    x_abs = centers[x_bin]
    side_id = torch.argmax(side_logits, dim=-1)
    out: Dict[str, torch.Tensor] = {
        "x_bin": x_bin,
        "x_abs": x_abs,
        "side_id": side_id,
    }
    if group_ranges is not None:
        sorted_idx = torch.empty_like(x_bin)
        for start, end in group_ranges:
            idx = torch.arange(start, end, device=x_bin.device)
            order = torch.argsort(x_abs[idx], stable=True)
            sorted_idx[start:end] = idx[order]
        out["sorted_idx"] = sorted_idx
    return out


def infer_chart(
    model: PhiTapMicro,
    batch: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    """Run inference on one padded batch and return numpy-friendly predictions."""
    model.eval()
    with torch.no_grad():
        out = model.forward_full(
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
    preds = predict_from_logits(
        out["x_bin_logits"], out["side_logits"], None
    )
    return {
        "x_bin_logits": out["x_bin_logits"].detach().cpu(),
        "side_logits": out["side_logits"].detach().cpu(),
        "x_bin": preds["x_bin"].cpu(),
        "x_abs": preds["x_abs"].cpu(),
        "side_id": preds["side_id"].cpu(),
    }
