import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from phigros_gen.model import (  # noqa: E402
    PhiTapMicro,
    build_model,
    default_config,
    predict_from_logits,
)


def make_batch(b, n, vocab, use_audio=True, device="cpu"):
    return {
        "scalars": torch.randn(b, n, 7, device=device),
        "type_idx": torch.randint(0, 4, (b, n), device=device),
        "diff_idx": torch.randint(0, vocab.num_difficulties, (b, n), device=device),
        "src_idx": torch.randint(0, vocab.num_sources, (b, n), device=device),
        "audio": torch.randn(b, n, 9, 64, device=device) if use_audio else torch.zeros(b, n, 9, 64, device=device),
        "times": torch.rand(b, n, device=device) * 100,
        "dts": torch.rand(b, n, device=device) + 0.01,
    }


class _Vocab:
    num_difficulties = 5
    num_sources = 3


def test_forward_shapes_and_no_speed_head():
    config = default_config(use_audio=True, n_difficulties=5, n_sources=3)
    model = build_model(config)
    batch = make_batch(2, 32, _Vocab(), use_audio=True)
    mask = torch.ones(2, 32, dtype=torch.bool)
    mask[1, 20:] = False
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
    assert set(out.keys()) == {"x_bin_logits", "side_logits"}
    assert out["x_bin_logits"].shape == (2, 32, 128)
    assert out["side_logits"].shape == (2, 32, 2)
    assert "speed" not in out


def test_variable_length_mask():
    config = default_config(use_audio=True, n_difficulties=5, n_sources=3)
    model = build_model(config)
    batch = make_batch(1, 10, _Vocab(), use_audio=True)
    mask = torch.ones(1, 10, dtype=torch.bool)
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
    assert out["x_bin_logits"].shape == (1, 10, 128)


def test_no_audio_config_forward():
    config = default_config(use_audio=False, n_difficulties=5, n_sources=3)
    model = build_model(config)
    batch = make_batch(2, 16, _Vocab(), use_audio=False)
    mask = torch.ones(2, 16, dtype=torch.bool)
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
    assert out["x_bin_logits"].shape == (2, 16, 128)


def test_predict_sorts_group_by_x():
    # Two groups of two slots; logits are chosen so predicted x decreases.
    x_logits = torch.zeros(4, 128)
    x_logits[0, 100] = 10.0  # slot 0 -> high x
    x_logits[1, 10] = 10.0   # slot 1 -> low x
    x_logits[2, 5] = 10.0
    x_logits[3, 120] = 10.0
    side_logits = torch.zeros(4, 2)
    pred = predict_from_logits(x_logits, side_logits, [(0, 2), (2, 4)])
    sorted_idx = pred["sorted_idx"]
    assert sorted_idx[:2].tolist() == [1, 0]
    assert sorted_idx[2:].tolist() == [2, 3]
    assert pred["x_bin"].shape == (4,)
    assert pred["side_id"].shape == (4,)


def test_num_parameters_is_reasonable():
    model = PhiTapMicro(default_config(use_audio=True, n_difficulties=8, n_sources=4))
    params = model.num_parameters()
    assert 500_000 <= params <= 5_000_000


def test_hungarian_uses_shared_assignment_for_x_and_side():
    from phigros_gen.metrics import hungarian_loss

    x_logits = torch.zeros(2, 128)
    x_logits[0, 1] = 10.0  # slot 0 prefers target bin 1
    x_logits[1, 0] = 10.0  # slot 1 prefers target bin 0
    side_logits = torch.zeros(2, 2)
    side_logits[0, 1] = 10.0  # aligned with slot 0 -> target side 1
    side_logits[1, 0] = 10.0  # aligned with slot 1 -> target side 0
    x_bin = torch.tensor([0, 1])
    side = torch.tensor([0, 1])
    loss = hungarian_loss(
        x_logits, side_logits, x_bin, side, [(0, 2)], side_weight=1.0
    )
    assert loss["assignments"][0].tolist() == [1, 0]
    assert loss["side_loss"].item() < 0.1


def test_forward_and_forward_full_agree_with_group_ids():
    config = default_config(use_audio=True, n_difficulties=5, n_sources=3)
    model = build_model(config)
    model.eval()
    batch = make_batch(2, 8, _Vocab(), use_audio=True)
    batch["u_slot"] = torch.randn(2, 8)
    batch["group_ids"] = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3], [0, 0, 1, 1, 2, 2, 3, 3]])
    mask = torch.ones(2, 8, dtype=torch.bool)
    mask[1, 6:] = False
    out_short = model(
        batch["scalars"], batch["type_idx"], batch["diff_idx"], batch["src_idx"],
        batch["audio"], batch["times"], batch["dts"],
        u_slot=batch["u_slot"], group_ids=batch["group_ids"],
        src_key_padding_mask=(~mask),
    )
    out_full = model.forward_full(
        batch["scalars"], batch["type_idx"], batch["diff_idx"], batch["src_idx"],
        batch["audio"], batch["times"], batch["dts"],
        u_slot=batch["u_slot"], group_ids=batch["group_ids"],
        src_key_padding_mask=(~mask),
    )
    assert torch.allclose(out_short["x_bin_logits"], out_full["x_bin_logits"])
    assert torch.allclose(out_short["side_logits"], out_full["side_logits"])


def test_group_mean_ignores_padding():
    config = default_config(use_audio=False, n_difficulties=2, n_sources=2)
    model = build_model(config)
    x = torch.randn(1, 4, 4)
    group_ids = torch.tensor([[0, 0, 0, 0]])  # all group 0, last one is padding
    valid = torch.tensor([[True, True, True, False]])
    pooled = model._group_mean(x, group_ids, 1, valid)
    expected = x[:, :3].mean(dim=1, keepdim=True)
    assert torch.allclose(pooled, expected, atol=1e-6)
