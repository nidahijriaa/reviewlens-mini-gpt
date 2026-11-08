"""Word-level tokenizer built from scratch on the project's own corpus.

Contract: INTERFACES.md sections 2.1 and 6.

- Text is split on whitespace only: one word -> one token. Special tokens
  (``<bos>``, ``<review>``, ``<pos>``, ...) are whole words, so they are never
  split. A word glued to a special token (``"bagus<pos>"``) is just an unknown
  word -> ``<unk>``.
- Special token IDs are fixed (0-9). ``PAD`` ... ``NONE`` are **int IDs**;
  the strings are ``PAD_TOKEN`` ... ``NONE_TOKEN`` (and ``SPECIAL_TOKENS``,
  where the index equals the ID).
- ``PAD`` (0) pads the input ``x`` only; target padding uses
  ``config.IGNORE_INDEX`` (-100), which is never a valid token ID here.
- The vocabulary is deterministic: words are ordered by descending frequency,
  ties broken alphabetically. Only words with ``count >= min_freq`` are kept,
  and the total size (special tokens included) is at most ``max_size``.
- The vocabulary must be built from the TRAIN split only. ``build_vocab_from_jsonl``
  does this from ``train.jsonl``: review text + ``dataset.format_target`` of the
  silver label, counting each unique record once so oversampled copies do not
  inflate frequencies.

No pretrained tokenizer or external library is used (standard library only).
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Sequence

__all__ = [
    "PAD", "UNK", "BOS", "EOS", "REVIEW", "POS", "NEG", "SUG", "SEP", "NONE",
    "PAD_TOKEN", "UNK_TOKEN", "BOS_TOKEN", "EOS_TOKEN", "REVIEW_TOKEN",
    "POS_TOKEN", "NEG_TOKEN", "SUG_TOKEN", "SEP_TOKEN", "NONE_TOKEN",
    "SPECIAL_TOKENS", "DEFAULT_MIN_FREQ", "DEFAULT_MAX_SIZE",
    "WordTokenizer", "build_vocab_from_jsonl",
]

# ---------------------------------------------------------------------------
# Special tokens (INTERFACES.md 2.1): string form and fixed integer ID.
# ---------------------------------------------------------------------------
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
REVIEW_TOKEN = "<review>"
POS_TOKEN = "<pos>"
NEG_TOKEN = "<neg>"
SUG_TOKEN = "<sug>"
SEP_TOKEN = "<sep>"
NONE_TOKEN = "<none>"

#: Index in this tuple == token ID.
SPECIAL_TOKENS: tuple[str, ...] = (
    PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, REVIEW_TOKEN,
    POS_TOKEN, NEG_TOKEN, SUG_TOKEN, SEP_TOKEN, NONE_TOKEN,
)

PAD: int = 0      # input padding (x) only; target padding is config.IGNORE_INDEX
UNK: int = 1
BOS: int = 2
EOS: int = 3
REVIEW: int = 4
POS: int = 5
NEG: int = 6
SUG: int = 7
SEP: int = 8
NONE: int = 9

DEFAULT_MIN_FREQ = 3
DEFAULT_MAX_SIZE = 10_000   # total, special tokens included

_N_SPECIAL = len(SPECIAL_TOKENS)


class WordTokenizer:
    """Whitespace word-level tokenizer with fixed special-token IDs."""

    def __init__(self, stoi: dict[str, int]):
        self._stoi, self._itos = self._validate(stoi)

    # -- construction --------------------------------------------------------
    @staticmethod
    def _validate(stoi: Any) -> tuple[dict[str, int], list[str]]:
        if not isinstance(stoi, dict):
            raise ValueError(f"stoi must be a dict[str, int], got {type(stoi).__name__}")
        for tok, idx in stoi.items():
            if not isinstance(tok, str) or not tok or tok != tok.strip() or len(tok.split()) != 1:
                raise ValueError(f"invalid token {tok!r}: must be one non-empty word")
            if not isinstance(idx, int) or isinstance(idx, bool):
                raise ValueError(f"ID of {tok!r} must be an int, got {idx!r}")
        ids = sorted(stoi.values())
        if ids != list(range(len(stoi))):
            raise ValueError("token IDs must be unique and contiguous from 0")
        for idx, tok in enumerate(SPECIAL_TOKENS):
            if stoi.get(tok) != idx:
                raise ValueError(f"special token {tok!r} must have ID {idx}, got {stoi.get(tok)!r}")
        itos = [""] * len(stoi)
        for tok, idx in stoi.items():
            itos[idx] = tok
        return dict(sorted(stoi.items(), key=lambda kv: kv[1])), itos

    @classmethod
    def build(cls, texts: Iterable[str], min_freq: int = DEFAULT_MIN_FREQ,
              max_size: int | None = DEFAULT_MAX_SIZE) -> "WordTokenizer":
        """Build a vocabulary from ``texts`` (TRAIN split only).

        Words with ``count >= min_freq`` are added after the 10 special tokens,
        sorted by descending count then alphabetically, until the vocabulary
        holds ``max_size`` tokens in total (``None`` = no limit). Special-token
        strings found in ``texts`` are not counted (they already exist).
        """
        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be an iterable of strings, not a single string")
        if not isinstance(min_freq, int) or isinstance(min_freq, bool) or min_freq < 1:
            raise ValueError(f"min_freq must be an int >= 1, got {min_freq!r}")
        if max_size is not None and (
            not isinstance(max_size, int) or isinstance(max_size, bool) or max_size < _N_SPECIAL
        ):
            raise ValueError(f"max_size must be None or an int >= {_N_SPECIAL}, got {max_size!r}")

        counts: Counter[str] = Counter()
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(f"texts[{i}] must be str, got {type(text).__name__}")
            counts.update(text.split())
        for tok in SPECIAL_TOKENS:
            counts.pop(tok, None)

        words = sorted((w for w, c in counts.items() if c >= min_freq),
                       key=lambda w: (-counts[w], w))
        if max_size is not None:
            words = words[: max_size - _N_SPECIAL]

        stoi = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        for w in words:
            stoi[w] = len(stoi)
        return cls(stoi)

    # -- encode / decode -----------------------------------------------------
    def encode(self, text: str) -> list[int]:
        """Whitespace split -> IDs; unknown words -> ``UNK`` (1). ``""`` -> ``[]``."""
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        stoi = self._stoi
        return [stoi.get(w, UNK) for w in text.split()]

    def decode(self, ids: Sequence[int], skip_special: bool = False) -> str:
        """IDs -> words joined by single spaces.

        ``skip_special=True`` drops all special tokens (IDs 0-9). Accepts a list,
        tuple or 1-D tensor of ints. IDs outside the vocabulary (including
        ``IGNORE_INDEX`` = -100) raise ``ValueError``.
        """
        if isinstance(ids, (str, bytes)):
            raise TypeError("ids must be a sequence of ints, not a string")
        if hasattr(ids, "tolist"):          # torch.Tensor / numpy array
            ids = ids.tolist()
        words: list[str] = []
        n = len(self._itos)
        for pos, raw in enumerate(ids):
            if isinstance(raw, bool) or not isinstance(raw, int):
                raise TypeError(f"ids[{pos}] must be int, got {type(raw).__name__}")
            if not 0 <= raw < n:
                raise ValueError(f"ids[{pos}] = {raw} is outside the vocabulary [0, {n})")
            if skip_special and raw < _N_SPECIAL:
                continue
            words.append(self._itos[raw])
        return " ".join(words)

    # -- persistence ---------------------------------------------------------
    def save(self, path: Path) -> None:
        """Write ``{"special_tokens": [...], "vocab_size": N, "stoi": {...}}``.

        UTF-8, ``stoi`` in ID order, deterministic bytes; written to a temp
        file then atomically renamed. Parent folders are created.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "special_tokens": list(SPECIAL_TOKENS),
            "vocab_size": self.vocab_size,
            "stoi": self._stoi,
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8", newline="\n") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=1)
            fh.write("\n")
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> "WordTokenizer":
        """Read a vocabulary written by ``save`` (only ``"stoi"`` is required)."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"vocab file not found: {path}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: invalid JSON ({exc.msg})") from exc
        if not isinstance(data, dict) or "stoi" not in data:
            raise ValueError(f"{path}: expected a JSON object with a 'stoi' key")
        tok = cls(data["stoi"])
        if "vocab_size" in data and data["vocab_size"] != tok.vocab_size:
            raise ValueError(f"{path}: vocab_size {data['vocab_size']} != len(stoi) {tok.vocab_size}")
        if "special_tokens" in data and list(data["special_tokens"]) != list(SPECIAL_TOKENS):
            raise ValueError(f"{path}: special_tokens differ from {list(SPECIAL_TOKENS)}")
        return tok

    # -- info ----------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return len(self._itos)

    @property
    def stoi(self) -> dict[str, int]:
        """Copy of the token -> ID mapping (in ID order)."""
        return dict(self._stoi)

    @property
    def itos(self) -> list[str]:
        """Copy of the ID -> token list."""
        return list(self._itos)

    def __len__(self) -> int:
        return self.vocab_size

    def __eq__(self, other: object) -> bool:
        return isinstance(other, WordTokenizer) and self._stoi == other._stoi

    def __repr__(self) -> str:
        return f"WordTokenizer(vocab_size={self.vocab_size})"


# ---------------------------------------------------------------------------
# Vocabulary from train.jsonl (TRAIN split only)
# ---------------------------------------------------------------------------
_RECORD_KEYS = ("review", "positives", "pain_points", "suggestions")


def _train_corpus(train_jsonl: Path) -> tuple[list[str], int, int]:
    """Texts for ``build``: review + formatted target of each UNIQUE record.

    Returns ``(texts, n_records, n_unique)``. Oversampled copies written by
    prepare_data are identical records, so they are counted once.
    """
    from .dataset import format_target      # lazy: keeps tokenizer import light
    from .labeling import InsightLabel

    path = Path(train_jsonl)
    if not path.is_file():
        raise FileNotFoundError(f"train jsonl not found: {path}")
    seen: set[str] = set()
    texts: list[str] = []
    n_records = 0
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{lineno}: invalid JSON ({exc.msg})") from exc
            if not isinstance(rec, dict) or any(k not in rec for k in _RECORD_KEYS):
                raise ValueError(f"{path.name}:{lineno}: expected keys {list(_RECORD_KEYS)}")
            n_records += 1
            key = json.dumps(rec, sort_keys=True, ensure_ascii=False)
            if key in seen:
                continue
            seen.add(key)
            label = InsightLabel(rec["positives"], rec["pain_points"], rec["suggestions"])
            texts.append(rec["review"])
            texts.append(format_target(label))
    if n_records == 0:
        raise ValueError(f"{path} contains no records")
    return texts, n_records, len(seen)


def build_vocab_from_jsonl(train_jsonl: Path, min_freq: int = DEFAULT_MIN_FREQ,
                           max_size: int | None = DEFAULT_MAX_SIZE) -> WordTokenizer:
    """Build the vocabulary from ``train.jsonl`` only (never val/test).

    Corpus = ``review`` + ``dataset.format_target(label)`` of every unique
    record (oversampled duplicates counted once).
    """
    texts, _, _ = _train_corpus(train_jsonl)
    return WordTokenizer.build(texts, min_freq=min_freq, max_size=max_size)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: ``python -m reviewlens.tokenizer --data-dir data/processed``."""
    parser = argparse.ArgumentParser(description="Build vocab.json from train.jsonl (TRAIN only)")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"),
                        help="folder with train.jsonl; vocab.json is written here")
    parser.add_argument("--min-freq", type=int, default=DEFAULT_MIN_FREQ)
    parser.add_argument("--max-size", type=int, default=DEFAULT_MAX_SIZE)
    args = parser.parse_args(argv)

    train_jsonl = args.data_dir / "train.jsonl"
    texts, n_records, n_unique = _train_corpus(train_jsonl)
    tok = WordTokenizer.build(texts, min_freq=args.min_freq, max_size=args.max_size)
    out = args.data_dir / "vocab.json"
    tok.save(out)

    n_words = sum(len(t.split()) for t in texts[0::2])
    n_unk = sum(i == UNK for t in texts[0::2] for i in tok.encode(t))
    print(f"train records: {n_records} ({n_unique} unique used for counting)")
    print(f"vocab_size   : {tok.vocab_size} (min_freq={args.min_freq}, max_size={args.max_size})")
    print(f"<unk> rate on train reviews: {100 * n_unk / max(n_words, 1):.2f}%")
    print(f"saved        : {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
