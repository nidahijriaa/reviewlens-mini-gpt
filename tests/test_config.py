"""Unit tests for reviewlens.config (INTERFACES.md section 3)."""

from __future__ import annotations

import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

# Make `src/` importable without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reviewlens.config import (  # noqa: E402
    IGNORE_INDEX,
    ModelConfig,
    Paths,
    TrainConfig,
    get_device,
    set_seed,
)


# ---------------------------------------------------------------------------
# IGNORE_INDEX (INTERFACES.md section 2.3)
# ---------------------------------------------------------------------------
def test_ignore_index_value():
    assert IGNORE_INDEX == -100
    assert type(IGNORE_INDEX) is int


def test_ignore_index_never_collides_with_token_ids():
    # Special tokens use IDs 0-9 (PAD = 0) and every vocab ID is >= 0, so a
    # negative IGNORE_INDEX can never mask a real target token.
    assert IGNORE_INDEX < 0
    assert IGNORE_INDEX not in range(0, 10)


def test_ignore_index_matches_cross_entropy_default():
    import inspect

    import torch.nn.functional as F

    default = inspect.signature(F.cross_entropy).parameters["ignore_index"].default
    assert IGNORE_INDEX == default


def test_ignore_index_masks_targets_in_cross_entropy():
    logits = torch.randn(4, 7)
    targets = torch.tensor([1, 2, IGNORE_INDEX, IGNORE_INDEX])
    loss = torch.nn.functional.cross_entropy(logits, targets, ignore_index=IGNORE_INDEX)
    expected = torch.nn.functional.cross_entropy(logits[:2], targets[:2])
    torch.testing.assert_close(loss, expected)


def test_model_reexports_same_ignore_index():
    from reviewlens import model

    assert model.IGNORE_INDEX is IGNORE_INDEX
    assert "IGNORE_INDEX" in model.__all__


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------
def test_model_config_defaults():
    cfg = ModelConfig(vocab_size=6000)
    assert asdict(cfg) == {
        "vocab_size": 6000,
        "block_size": 128,
        "n_layer": 4,
        "n_head": 4,
        "d_model": 256,
        "d_ff": 1024,
        "dropout": 0.1,
    }


def test_model_config_requires_vocab_size():
    with pytest.raises(TypeError):
        ModelConfig()  # type: ignore[call-arg]


def test_default_d_model_divisible_by_n_head():
    cfg = ModelConfig(vocab_size=100)
    assert cfg.d_model % cfg.n_head == 0
    assert cfg.d_head == 64


def test_model_config_rejects_indivisible_heads():
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(vocab_size=100, d_model=250, n_head=4)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"vocab_size": 0},
        {"vocab_size": 100, "block_size": 0},
        {"vocab_size": 100, "n_head": -1},
        {"vocab_size": 100, "dropout": 1.0},
        {"vocab_size": 100, "dropout": -0.1},
    ],
)
def test_model_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        ModelConfig(**kwargs)


# ---------------------------------------------------------------------------
# TrainConfig
# ---------------------------------------------------------------------------
def test_train_config_defaults():
    cfg = TrainConfig()
    assert asdict(cfg) == {
        "batch_size": 64,
        "lr": 3e-4,
        "weight_decay": 0.1,
        "max_epochs": 6,
        "warmup_steps": 200,
        "grad_clip": 1.0,
        "log_interval": 50,
        "seed": 42,
        "device": None,
    }


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def test_paths_defaults_and_derived():
    p = Paths()
    assert p.raw_csv == Path("data/raw/Processed Dataset.csv")
    assert p.data_dir == Path("data/processed")
    assert p.checkpoint_dir == Path("checkpoints")

    assert p.train_jsonl == Path("data/processed/train.jsonl")
    assert p.val_jsonl == Path("data/processed/val.jsonl")
    assert p.test_jsonl == Path("data/processed/test.jsonl")
    assert p.vocab == Path("data/processed/vocab.json")

    assert p.best_ckpt == Path("checkpoints/best.pt")
    assert p.last_ckpt == Path("checkpoints/last.pt")
    assert p.history_json == Path("checkpoints/history.json")
    assert p.ckpt_vocab == Path("checkpoints/vocab.json")
    assert p.eval_json == Path("checkpoints/eval.json")


def test_paths_accept_strings_from_cli():
    p = Paths(
        raw_csv="/kaggle/input/x/Processed Dataset.csv",  # type: ignore[arg-type]
        data_dir="/kaggle/working/data/processed",  # type: ignore[arg-type]
        checkpoint_dir="/kaggle/working/checkpoints",  # type: ignore[arg-type]
    )
    assert isinstance(p.raw_csv, Path)
    assert p.train_jsonl == Path("/kaggle/working/data/processed/train.jsonl")
    assert p.last_ckpt == Path("/kaggle/working/checkpoints/last.pt")


# ---------------------------------------------------------------------------
# get_device
# ---------------------------------------------------------------------------
def test_get_device_auto_matches_cuda_availability():
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert get_device() == expected
    # The returned string must be usable by torch.
    torch.zeros(1, device=get_device())


def test_get_device_prefers_cuda_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert get_device() == "cuda"
    assert get_device("cuda:0") == "cuda:0"


def test_get_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert get_device() == "cpu"


def test_get_device_override_cpu():
    assert get_device("cpu") == "cpu"
    assert get_device(" CPU ") == "cpu"


def test_get_device_cuda_override_without_cuda_raises(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        get_device("cuda")


def test_get_device_rejects_second_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(ValueError, match="single GPU"):
        get_device("cuda:1")


def test_get_device_rejects_unknown():
    with pytest.raises(ValueError):
        get_device("tpu")


# ---------------------------------------------------------------------------
# set_seed
# ---------------------------------------------------------------------------
def _draw():
    return (
        random.random(),
        np.random.rand(3).tolist(),
        torch.rand(3).tolist(),
        torch.nn.Linear(4, 4).weight.detach().clone(),
    )


def test_set_seed_is_deterministic():
    set_seed(123)
    a = _draw()
    set_seed(123)
    b = _draw()
    assert a[0] == b[0]
    assert a[1] == b[1]
    assert a[2] == b[2]
    assert torch.equal(a[3], b[3])


def test_set_seed_different_seeds_differ():
    set_seed(1)
    a = _draw()
    set_seed(2)
    b = _draw()
    assert a[0] != b[0]
    assert a[2] != b[2]
