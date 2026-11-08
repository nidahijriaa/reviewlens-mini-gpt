"""Sequence format, truncation and batching for Mini-GPT training.

Contract: INTERFACES.md sections 2 and 7.

Canonical sequence (string before tokenisation)::

    <bos> <review> {review_clean} <pos> {p1} <sep> {p2} <neg> {n1} <sug> {s1} <eos>
    \\_ prefix 2 _/ \\_ review r _/ \\__________ target t: <pos> ... <eos> ___________/

- ``encode_example`` keeps the whole sequence at ``<= block_size + 1`` tokens.
  The target is encoded first and capped at ``MAX_TARGET_TOKENS``; the review
  (never the target) is truncated to ``MAX_REVIEW_TOKENS`` words.
- ``collate_fn`` shifts every sequence (``x = ids[:-1]``, ``y = ids[1:]``) and
  right-pads ``x`` with ``PAD`` (0) and ``y`` with ``IGNORE_INDEX`` (-100), so
  padding never contributes to the loss while the real ``<eos>`` target does.

Train/val/test splitting is done by ``preprocess.split_data`` (80/10/10,
stratified, seed 42) inside ``scripts/prepare_data.py``; this module only reads
the resulting ``*.jsonl`` files.

The tokenizer is used only through the contract method
``tokenizer.encode(text: str) -> list[int]`` (INTERFACES.md section 6). This
module does not import ``tokenizer.py``.
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Protocol

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .config import IGNORE_INDEX, ModelConfig
from .labeling import InsightLabel

__all__ = [
    "PREFIX_TOKENS",
    "MAX_TARGET_TOKENS",
    "MAX_REVIEW_TOKENS",
    "MAX_SEQ_TOKENS",
    "MAX_NEW_TOKENS",
    "format_prompt",
    "format_target",
    "encode_example",
    "ReviewInsightDataset",
    "collate_fn",
]

# ---------------------------------------------------------------------------
# Special tokens (INTERFACES.md 2.1). The canonical owner of these values is
# tokenizer.py; they are mirrored here (private) only so this module can format
# strings and pad batches without importing the tokenizer. tests/test_dataset.py
# checks they agree with tokenizer.py as soon as that module exists.
# ---------------------------------------------------------------------------
_PAD_ID = 0          # PAD: padding of the INPUT x only
_BOS, _REVIEW = "<bos>", "<review>"
_POS, _NEG, _SUG = "<pos>", "<neg>", "<sug>"
_SEP, _NONE, _EOS = "<sep>", "<none>", "<eos>"
_SPECIAL_STRINGS = frozenset(
    {"<pad>", "<unk>", _BOS, _EOS, _REVIEW, _POS, _NEG, _SUG, _SEP, _NONE}
)

# ---------------------------------------------------------------------------
# Length limits (INTERFACES.md 2.2), derived from the default block_size (128).
# ---------------------------------------------------------------------------
_DEFAULT_BLOCK_SIZE: int = next(
    f.default for f in fields(ModelConfig) if f.name == "block_size"
)
PREFIX_TOKENS = 2                                   # <bos> <review>
MAX_TARGET_TOKENS = 48                              # <pos> ... <eos>
MAX_SEQ_TOKENS = _DEFAULT_BLOCK_SIZE + 1            # 129
MAX_REVIEW_TOKENS = MAX_SEQ_TOKENS - PREFIX_TOKENS - MAX_TARGET_TOKENS   # 79
MAX_NEW_TOKENS = MAX_TARGET_TOKENS - 1              # 47

_SECTIONS = ("positives", "pain_points", "suggestions")
# Truncation order for an over-long target: drop the last item of
# suggestions, then pain_points, then positives (one item per turn, round-robin)
# and repeat until it fits.
_DROP_ORDER = ("suggestions", "pain_points", "positives")


class _Encoder(Protocol):
    """The only tokenizer capability this module needs (INTERFACES.md 6)."""

    def encode(self, text: str) -> list[int]: ...


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _check_review(review_clean: str) -> list[str]:
    if not isinstance(review_clean, str):
        raise TypeError(f"review_clean must be str, got {type(review_clean).__name__}")
    words = review_clean.split()
    if not words:
        raise ValueError("review_clean is empty; empty reviews must be dropped before formatting")
    if any(w in _SPECIAL_STRINGS for w in words):
        raise ValueError(f"review_clean must not contain special tokens: {review_clean!r}")
    return words


def _check_phrase(phrase: str, section: str) -> str:
    if not isinstance(phrase, str) or not phrase.split():
        raise ValueError(f"{section} contains an empty or non-string phrase: {phrase!r}")
    if any(w in _SPECIAL_STRINGS for w in phrase.split()):
        raise ValueError(f"{section} phrase must not contain special tokens: {phrase!r}")
    return " ".join(phrase.split())


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def format_prompt(review_clean: str) -> str:
    """``"<bos> <review> {review <= MAX_REVIEW_TOKENS words} <pos>"`` (used at inference)."""
    words = _check_review(review_clean)[:MAX_REVIEW_TOKENS]
    return " ".join([_BOS, _REVIEW, *words, _POS])


def format_target(label: InsightLabel) -> str:
    """``"<pos> p1 <sep> p2 <neg> n1 <sug> s1 <eos>"``; an empty section becomes ``<none>``."""
    parts: list[str] = []
    for marker, section in zip((_POS, _NEG, _SUG), _SECTIONS):
        items = getattr(label, section)
        if not isinstance(items, (list, tuple)):
            raise TypeError(f"label.{section} must be a list of str, got {type(items).__name__}")
        phrases = [_check_phrase(p, section) for p in items]
        parts.append(marker)
        parts.append(f" {_SEP} ".join(phrases) if phrases else _NONE)
    parts.append(_EOS)
    return " ".join(parts)


def _encode_target(label: InsightLabel, tokenizer: _Encoder) -> list[int]:
    """Encode the target, dropping trailing items until it fits MAX_TARGET_TOKENS."""
    current = InsightLabel(*(list(getattr(label, s)) for s in _SECTIONS))
    turn = 0
    while True:
        ids = tokenizer.encode(format_target(current))
        if len(ids) <= MAX_TARGET_TOKENS:
            return ids
        # Round-robin: drop ONE trailing item, taking turns suggestions ->
        # pain_points -> positives (skipping empty sections), then re-check.
        # Taking turns keeps suggestions[i] aligned with pain_points[i].
        for _ in range(len(_DROP_ORDER)):
            items = getattr(current, _DROP_ORDER[turn % len(_DROP_ORDER)])
            turn += 1
            if items:
                items.pop()
                break
        else:  # nothing left to drop (cannot happen with the canonical format)
            raise ValueError(f"target does not fit in {MAX_TARGET_TOKENS} tokens")


def encode_example(
    review_clean: str, label: InsightLabel, tokenizer: _Encoder, block_size: int
) -> list[int]:
    """Token IDs of one full training sequence, ``len <= block_size + 1``.

    1. Target first, capped at ``MAX_TARGET_TOKENS`` (items dropped, markers kept).
    2. Review truncated to ``MAX_REVIEW_TOKENS`` words (and to whatever room is
       left for smaller test configs). The target is never truncated.
    """
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError(f"block_size must be a positive int, got {block_size!r}")
    _check_review(review_clean)

    target_ids = _encode_target(label, tokenizer)
    prefix_ids = tokenizer.encode(f"{_BOS} {_REVIEW}")
    room = block_size + 1 - len(prefix_ids) - len(target_ids)
    if room < 1:
        raise ValueError(
            f"block_size={block_size} too small: prefix ({len(prefix_ids)}) + target "
            f"({len(target_ids)}) leave no room for the review"
        )
    review_ids = tokenizer.encode(review_clean)[: min(MAX_REVIEW_TOKENS, room)]

    ids = prefix_ids + review_ids + target_ids
    assert len(ids) <= block_size + 1, (len(ids), block_size)
    return ids


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def _read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{lineno}: invalid JSON ({exc.msg})") from exc
            if not isinstance(rec, dict):
                raise ValueError(f"{path.name}:{lineno}: expected a JSON object")
            missing = [k for k in ("review", *_SECTIONS) if k not in rec]
            if missing:
                raise ValueError(f"{path.name}:{lineno}: missing keys {missing}")
            records.append(rec)
    return records


class ReviewInsightDataset(Dataset):
    """One item = the ``LongTensor`` returned by ``encode_example`` for one jsonl row.

    All rows are encoded once at construction, so malformed data fails early.
    An empty file (no records) is rejected.
    """

    def __init__(self, jsonl_path: Path, tokenizer: _Encoder, block_size: int) -> None:
        path = Path(jsonl_path)
        if not path.is_file():
            raise FileNotFoundError(f"jsonl not found: {path}")
        records = _read_jsonl(path)
        if not records:
            raise ValueError(f"{path} contains no examples")

        self.block_size = block_size
        self.examples: list[Tensor] = []
        for lineno, rec in enumerate(records, start=1):
            label = InsightLabel(
                positives=rec["positives"],
                pain_points=rec["pain_points"],
                suggestions=rec["suggestions"],
            )
            try:
                ids = encode_example(rec["review"], label, tokenizer, block_size)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path.name}: example {lineno}: {exc}") from exc
            self.examples.append(torch.tensor(ids, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> Tensor:
        return self.examples[i]


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------
def collate_fn(batch: list[Tensor]) -> tuple[Tensor, Tensor]:
    """Shift and right-pad a batch of full sequences.

    For each sequence ``x = ids[:-1]``, ``y = ids[1:]``. Then ``x`` is padded
    with ``PAD`` (0) and ``y`` with ``IGNORE_INDEX`` (-100) to the longest
    ``T`` in the batch. Returns ``(x, y)``, both ``(B, T)`` ``torch.long``.
    """
    if len(batch) == 0:
        raise ValueError("collate_fn received an empty batch")
    for i, seq in enumerate(batch):
        if not isinstance(seq, Tensor) or seq.dim() != 1:
            raise ValueError(f"batch[{i}] must be a 1-D tensor")
        if seq.dtype != torch.long:
            raise ValueError(f"batch[{i}] must be torch.long, got {seq.dtype}")
        if seq.numel() < 2:
            raise ValueError(f"batch[{i}] needs at least 2 tokens to form (x, y)")

    T = max(seq.numel() for seq in batch) - 1
    x = torch.full((len(batch), T), _PAD_ID, dtype=torch.long)
    y = torch.full((len(batch), T), IGNORE_INDEX, dtype=torch.long)
    for i, seq in enumerate(batch):
        n = seq.numel() - 1
        x[i, :n] = seq[:-1]
        y[i, :n] = seq[1:]
    return x, y
