"""Load, clean, normalise, and split the review dataset.

Contract: INTERFACES.md section 4.

- ``load_reviews`` reads the original CSV (read-only) and renames columns in
  memory only. The source file is never written.
- ``clean_text`` is the single cleaning function for training, labeling, and
  inference. Negation normalisation happens here and nowhere else.
- ``split_data`` performs a deterministic stratified split on ``sentiment``,
  grouped by ``clean_text(review)`` so identical reviews never cross splits;
  groups with conflicting sentiment labels are dropped
  (``split_data_with_stats`` also returns the statistics).
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Column mapping (original CSV name -> internal name)
# ---------------------------------------------------------------------------

COLUMN_MAP: dict[str, str] = {
    "reviewContent": "review",
    "sentiment_label": "sentiment",
    "rating": "rating",
    "itemId": "item_id",
}
OPTIONAL_COLUMNS: tuple[str, ...] = ("confidence",)  # read if present, unused in MVP

# ---------------------------------------------------------------------------
# Normalisation dictionaries
# ---------------------------------------------------------------------------

NEGATION_MAP: dict[str, str] = {  # informal variant -> canonical form
    "gak": "tidak", "ga": "tidak", "nggak": "tidak", "ngga": "tidak",
    "gk": "tidak", "tdk": "tidak", "enggak": "tidak", "kagak": "tidak",
    "blm": "belum", "krg": "kurang", "bkn": "bukan",
}
NEGATIONS: tuple[str, ...] = ("tidak", "kurang", "belum", "bukan")  # canonical forms

# Other slang / common typos. Must not share keys with NEGATION_MAP, and every
# value must be a fixed point (not itself a key) so clean_text is idempotent.
SLANG_MAP: dict[str, str] = {
    # intensifiers / adjectives
    "bgt": "banget", "bngt": "banget", "bget": "banget",
    "mantab": "mantap", "mantep": "mantap", "mantul": "mantap",
    "cpt": "cepat", "cepet": "cepat",
    "bener": "benar", "bnr": "benar",
    "ok": "oke", "okey": "oke", "okeh": "oke",
    # nouns frequently seen in reviews
    "brg": "barang", "barng": "barang", "brgnya": "barangnya",
    "pengirimn": "pengiriman", "pngiriman": "pengiriman",
    # typos
    "sesui": "sesuai", "sesuaii": "sesuai",
    # function words
    "yg": "yang", "dgn": "dengan", "dg": "dengan",
    "utk": "untuk", "untk": "untuk",
    "krn": "karena", "karna": "karena",
    "tp": "tapi", "tpi": "tapi",
    "sdh": "sudah", "udh": "sudah", "udah": "sudah", "dah": "sudah",
    "aja": "saja", "aj": "saja",
    "jg": "juga", "sm": "sama",
    "bs": "bisa", "bsa": "bisa",
    "lg": "lagi", "lgi": "lagi",
    "msh": "masih", "masi": "masih",
    "lbh": "lebih", "bnyk": "banyak", "byk": "banyak",
    "dr": "dari", "dri": "dari",
    "kalo": "kalau", "klo": "kalau",
    "emang": "memang", "trs": "terus", "trus": "terus",
    "sampe": "sampai", "smpe": "sampai", "nyampe": "sampai",
    "tlg": "tolong",
}

# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

_URL_RE = re.compile(r"(?:https?://|www\.)\S+")
# Anything that is not a word character or whitespace (punctuation, emoji,
# symbols). "_" is a word character in Python regex, so remove it explicitly.
_NON_WORD_RE = re.compile(r"[^\w\s]|_")
# A letter (not digit/underscore) repeated 3+ times -> keep 2.
# Digits are excluded on purpose so numbers like "1000" stay intact.
_REPEAT_RE = re.compile(r"([^\W\d_])\1{2,}")
_SPACE_RE = re.compile(r"\s+")


def normalize_token(token: str) -> str:
    """Normalise one lowercase word: NEGATION_MAP first, then SLANG_MAP."""
    if token in NEGATION_MAP:
        return NEGATION_MAP[token]
    return SLANG_MAP.get(token, token)


def clean_text(text: str) -> str:
    """Clean a raw review. Must be used identically for training and inference.

    Steps: lowercase -> remove URL -> remove emoji/punctuation -> collapse
    letters repeated 3+ times to 2 -> ``normalize_token`` per word -> tidy
    whitespace. ``None``, NaN, and non-string/empty input return ``""``.
    """
    if text is None:
        return ""
    if isinstance(text, float) and math.isnan(text):
        return ""
    if not isinstance(text, str):
        if pd.isna(text):  # pd.NA, np.nan of other dtypes
            return ""
        text = str(text)

    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _NON_WORD_RE.sub(" ", text)
    text = _REPEAT_RE.sub(r"\1\1", text)
    tokens = [normalize_token(tok) for tok in text.split()]
    return _SPACE_RE.sub(" ", " ".join(tokens)).strip()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_reviews(csv_path: Path) -> pd.DataFrame:
    """Read the review CSV (read-only) and map columns to internal names.

    Returns a DataFrame with columns ``review, sentiment, rating, item_id``
    (plus ``confidence`` when present). Rows with an empty ``review`` and
    duplicate ``(item_id, review)`` pairs are dropped. ``rating`` may be NaN.

    Raises:
        FileNotFoundError: if ``csv_path`` does not exist.
        ValueError: if a required column is missing.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV tidak ditemukan: {csv_path}")

    # utf-8-sig also handles files saved with a BOM.
    header = pd.read_csv(csv_path, nrows=0, encoding="utf-8-sig").columns
    header = [str(c).strip() for c in header]
    missing = [c for c in COLUMN_MAP if c not in header]
    if missing:
        raise ValueError(
            f"Kolom wajib tidak ditemukan di {csv_path.name}: {missing}. "
            f"Kolom yang tersedia: {list(header)}"
        )
    optional = [c for c in OPTIONAL_COLUMNS if c in header]
    usecols = list(COLUMN_MAP) + optional

    df = pd.read_csv(
        csv_path,
        usecols=lambda c: str(c).strip() in usecols,
        encoding="utf-8-sig",
    )
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns=COLUMN_MAP)

    # Empty review: NaN or whitespace only.
    df = df[df["review"].notna()].copy()
    df["review"] = df["review"].astype(str)
    df = df[df["review"].str.strip() != ""]

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")  # NaN allowed
    if "confidence" in df.columns:
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df["sentiment"] = df["sentiment"].where(
        df["sentiment"].isna(), df["sentiment"].astype(str).str.strip().str.lower()
    )

    df = df.drop_duplicates(subset=["item_id", "review"], keep="first")

    ordered = ["review", "sentiment", "rating", "item_id"] + [
        c for c in OPTIONAL_COLUMNS if c in df.columns
    ]
    return df[ordered].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


_SPLIT_NAMES = ("train", "val", "test")
_MISSING_SENTIMENT = "<NA>"
_MAX_CONFLICT_EXAMPLES = 10


def split_data(
    df: pd.DataFrame,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Deterministic, leakage-free stratified split into train/val/test.

    Same behaviour as ``split_data_with_stats`` without the statistics.
    """
    train, val, test, _ = split_data_with_stats(df, val_frac, test_frac, seed)
    return train, val, test


def split_data_with_stats(
    df: pd.DataFrame,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Group-aware stratified split that keeps identical reviews in one split.

    1. Rows are grouped by ``clean_text(review)``, so texts that differ only in
       case, punctuation, slang or stretched letters form one group.
    2. **Conflict groups** (more than one distinct ``sentiment``; a missing
       sentiment counts as its own value) are dropped entirely. Labels are
       never changed.
    3. Every remaining group (one sentiment, possibly many rows) goes to exactly
       one split. Stratification is per sentiment: inside each stratum, groups
       are placed largest first (equal sizes in seeded random order) into the
       split currently furthest below its row target (``round(n * frac)``).
       With only single-row groups the per-class counts are exactly the
       targets; large groups may make proportions approximate.
    4. Rows inside each split are shuffled with the same seed. Deterministic
       for a given input and ``seed``. Input columns and values are unchanged.

    Returns ``(train, val, test, stats)``; ``stats`` holds the number of
    groups, conflict groups, dropped/kept rows and per-split rows/groups.
    """
    for col in ("review", "sentiment"):
        if col not in df.columns:
            raise ValueError(f"split_data membutuhkan kolom '{col}'")
    if not (0 <= val_frac < 1 and 0 <= test_frac < 1 and val_frac + test_frac < 1):
        raise ValueError(
            f"Fraksi tidak valid: val_frac={val_frac}, test_frac={test_frac}"
        )

    rng = np.random.default_rng(seed)
    keys = df["review"].map(clean_text).to_numpy(dtype=object)
    labels = (
        df["sentiment"].astype(object)
        .where(df["sentiment"].notna(), _MISSING_SENTIMENT)
        .astype(str).to_numpy(dtype=object)
    )

    # key -> row positions, and key -> set of sentiments.
    group_pos: dict[str, list[int]] = {}
    group_labels: dict[str, set[str]] = {}
    for pos, (key, label) in enumerate(zip(keys, labels)):
        group_pos.setdefault(key, []).append(pos)
        group_labels.setdefault(key, set()).add(label)

    conflict_keys = sorted(k for k, s in group_labels.items() if len(s) > 1)
    conflict_set = set(conflict_keys)
    n_dropped = sum(len(group_pos[k]) for k in conflict_keys)

    # stratum (sentiment) -> list of (key, positions)
    strata: dict[str, list[tuple[str, list[int]]]] = {}
    for key in sorted(group_pos):          # sorted -> independent of row order
        if key in conflict_set:
            continue
        (label,) = group_labels[key]
        strata.setdefault(label, []).append((key, group_pos[key]))

    split_pos: dict[str, list[int]] = {name: [] for name in _SPLIT_NAMES}
    split_groups = dict.fromkeys(_SPLIT_NAMES, 0)
    for label in sorted(strata):
        groups = strata[label]
        n = sum(len(p) for _, p in groups)
        n_test = int(round(n * test_frac))
        n_val = min(int(round(n * val_frac)), n - n_test)
        target = {"train": n - n_val - n_test, "val": n_val, "test": n_test}
        filled = dict.fromkeys(_SPLIT_NAMES, 0)

        # Seeded shuffle, then stable sort by size (largest first).
        order = rng.permutation(len(groups))
        order = sorted(order, key=lambda g: -len(groups[g][1]))
        for g in order:
            _, positions = groups[g]
            # Split furthest below its target; ties -> train, val, test.
            name = max(_SPLIT_NAMES, key=lambda s: target[s] - filled[s])
            split_pos[name].extend(positions)
            filled[name] += len(positions)
            split_groups[name] += 1

    def _take(positions: list[int]) -> pd.DataFrame:
        idx = rng.permutation(np.asarray(positions, dtype=int))
        return df.iloc[idx].reset_index(drop=True)

    parts = {name: _take(split_pos[name]) for name in _SPLIT_NAMES}
    n_kept = len(df) - n_dropped
    stats = {
        "n_rows_input": len(df),
        "n_groups": len(group_pos),
        "n_conflict_groups": len(conflict_keys),
        "n_rows_dropped": n_dropped,
        "n_rows_kept": n_kept,
        "split_rows": {name: len(parts[name]) for name in _SPLIT_NAMES},
        "split_groups": split_groups,
        "split_frac": {
            name: (len(parts[name]) / n_kept if n_kept else 0.0) for name in _SPLIT_NAMES
        },
        "conflict_examples": conflict_keys[:_MAX_CONFLICT_EXAMPLES],
    }
    return parts["train"], parts["val"], parts["test"], stats
