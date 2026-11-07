"""Configuration objects and runtime helpers for ReviewLens GPT.

Contract: INTERFACES.md section 3.

- ``ModelConfig``  : Mini-GPT architecture hyperparameters.
- ``TrainConfig``  : optimisation / training-loop hyperparameters.
- ``Paths``        : filesystem locations (overridable from CLI) and derived files.
- ``get_device``   : single-device selection (``cuda`` = ``cuda:0``, else ``cpu``).
- ``set_seed``     : seeds ``random``, ``numpy``, ``torch`` and ``torch.cuda``.

Device policy (PROJECT_CONTEXT.md section 9.1): exactly one device is used.
No ``DataParallel`` / ``DistributedDataParallel`` / multi-GPU.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

__all__ = ["ModelConfig", "TrainConfig", "Paths", "get_device", "set_seed"]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    """Hyperparameters of the decoder-only Mini-GPT."""

    vocab_size: int            # filled from the tokenizer (6.000-10.000)
    block_size: int = 128      # maximum input length (pos-emb size & causal-mask size)
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 256         # must be divisible by n_head
    d_ff: int = 1024
    dropout: float = 0.1

    def __post_init__(self) -> None:
        for name in ("vocab_size", "block_size", "n_layer", "n_head", "d_model", "d_ff"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"ModelConfig.{name} must be a positive int, got {value!r}")
        if self.d_model % self.n_head != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_head ({self.n_head})"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

    @property
    def d_head(self) -> int:
        """Dimension of a single attention head (d_model // n_head)."""
        return self.d_model // self.n_head


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    """Hyperparameters of the training loop."""

    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.1
    max_epochs: int = 6        # total target epochs (not "additional" epochs on resume)
    warmup_steps: int = 200
    grad_clip: float = 1.0
    log_interval: int = 50     # steps; train-loss logging only
    seed: int = 42
    device: str | None = None  # None -> get_device()


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
@dataclass
class Paths:
    """Filesystem layout. All three roots can be overridden from the CLI
    (``--csv``, ``--data-dir``, ``--checkpoint-dir``) so the same code runs
    locally, on ``/kaggle/working`` and on Google Drive (Colab)."""

    raw_csv: Path = Path("data/raw/Processed Dataset.csv")
    data_dir: Path = Path("data/processed")      # train/val/test.jsonl, vocab.json
    checkpoint_dir: Path = Path("checkpoints")

    def __post_init__(self) -> None:
        # Accept plain strings (e.g. straight from argparse).
        self.raw_csv = Path(self.raw_csv)
        self.data_dir = Path(self.data_dir)
        self.checkpoint_dir = Path(self.checkpoint_dir)

    # -- under data_dir ------------------------------------------------------
    @property
    def train_jsonl(self) -> Path:
        return self.data_dir / "train.jsonl"

    @property
    def val_jsonl(self) -> Path:
        return self.data_dir / "val.jsonl"

    @property
    def test_jsonl(self) -> Path:
        return self.data_dir / "test.jsonl"

    @property
    def vocab(self) -> Path:
        return self.data_dir / "vocab.json"

    # -- under checkpoint_dir ------------------------------------------------
    @property
    def best_ckpt(self) -> Path:
        return self.checkpoint_dir / "best.pt"

    @property
    def last_ckpt(self) -> Path:
        return self.checkpoint_dir / "last.pt"

    @property
    def history_json(self) -> Path:
        return self.checkpoint_dir / "history.json"

    @property
    def ckpt_vocab(self) -> Path:
        return self.checkpoint_dir / "vocab.json"

    @property
    def eval_json(self) -> Path:
        return self.checkpoint_dir / "eval.json"


# ---------------------------------------------------------------------------
# Runtime helpers
# ---------------------------------------------------------------------------
def get_device(override: str | None = None) -> str:
    """Return the single device to run on.

    - ``override`` given -> validated and returned (``"cpu"``, ``"cuda"`` or ``"cuda:0"``).
    - otherwise ``"cuda"`` if CUDA is available, else ``"cpu"``.

    ``"cuda"`` always means the first GPU (``cuda:0``); multi-GPU is out of
    scope, so ``cuda:1`` etc. are rejected even on a T4 x2 Kaggle machine.
    """
    if override is None:
        return "cuda" if torch.cuda.is_available() else "cpu"

    device = override.strip().lower()
    if device == "cpu":
        return "cpu"
    if device in ("cuda", "cuda:0"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"device {override!r} requested but CUDA is not available; use 'cpu' or omit --device"
            )
        return device
    if device.startswith("cuda:"):
        raise ValueError(
            f"device {override!r} not allowed: only a single GPU (cuda:0) is supported"
        )
    raise ValueError(f"unknown device {override!r}; expected 'cpu', 'cuda' or 'cuda:0'")


def set_seed(seed: int) -> None:
    """Seed every RNG used by the project: ``random``, ``numpy``, ``torch``, ``torch.cuda``."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
