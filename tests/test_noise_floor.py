import pytest
import torch

from conform.noise_floor import (
    ABOVE, FLOAT32_RESOLUTION, INSIDE, UNCLEAR, classify, measure_floor, relative_gap,
)
from conform.probes.p1_grad_accum import build_rows
from conform.proxy_model import LLAMA_TYPE, build_proxy


def test_classify_follows_figure_7():
    assert classify(0.0, floor=1e-6, multiplier=10, pass_band=2) == INSIDE
    assert classify(2e-6, floor=1e-6, multiplier=10, pass_band=2) == INSIDE
    assert classify(5e-6, floor=1e-6, multiplier=10, pass_band=2) == UNCLEAR
    assert classify(1e-4, floor=1e-6, multiplier=10, pass_band=2) == ABOVE


def test_relative_gap():
    assert relative_gap(1.1, 1.0) == pytest.approx(0.1)
    assert relative_gap(torch.tensor([3.0, 4.0]), torch.tensor([3.0, 0.0])) == pytest.approx(4 / 3)


def test_floor_is_small_but_measured():
    try:
        model, tokenizer, _ = build_proxy(LLAMA_TYPE)
    except OSError:
        pytest.skip("tokenizer not downloaded; run scripts/download_targets.py")
    rows = build_rows(len(tokenizer), tokenizer.pad_token_id, seed=0)
    batches = [{k: v.unsqueeze(0) for k, v in row.items()} for row in rows]
    _, _, floor = measure_floor(model, batches)
    assert FLOAT32_RESOLUTION <= floor.loss < 1e-4
    assert 0 < floor.gradient < 1e-4
