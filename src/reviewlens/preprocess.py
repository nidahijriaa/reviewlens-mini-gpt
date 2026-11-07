"""Load, clean, normalise, and split the review dataset.

Contract: INTERFACES.md section 4.

- ``load_reviews`` reads the original CSV (read-only) and renames columns in
  memory only. The source file is never written.
- ``clean_text`` is the single cleaning function for training, labeling, and
  inference. Negation normalisation happens here and nowhere else.
- ``split_data`` performs a deterministic stratified split on ``sentiment``.
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


def split_data(
    df: pd.DataFrame,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Deterministic stratified split on ``sentiment`` into train/val/test.

    Each sentiment class is shuffled with a seeded RNG and split by the given
    fractions, so class proportions are preserved in every split. Missing
    sentiment values form their own stratum (no rows are lost). Splits are
    disjoint and together contain every input row.
    """
    if "sentiment" not in df.columns:
        raise ValueError("split_data membutuhkan kolom 'sentiment'")
    if not (0 <= val_frac < 1 and 0 <= test_frac < 1 and val_frac + test_frac < 1):
        raise ValueError(
            f"Fraksi tidak valid: val_frac={val_frac}, test_frac={test_frac}"
        )

    rng = np.random.default_rng(seed)
    strata = df["sentiment"].astype(object).where(df["sentiment"].notna(), "<NA>")
    strata = strata.astype(str)

    train_pos: list[np.ndarray] = []
    val_pos: list[np.ndarray] = []
    test_pos: list[np.ndarray] = []
    # Sorted class order + positional indices -> deterministic for a given seed.
    for label in sorted(strata.unique()):
        pos = np.flatnonzero((strata == label).to_numpy())
        pos = rng.permutation(pos)
        n = len(pos)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        n_val = min(n_val, n - n_test)
        test_pos.append(pos[:n_test])
        val_pos.append(pos[n_test:n_test + n_val])
        train_pos.append(pos[n_test + n_val:])

    def _take(parts: list[np.ndarray]) -> pd.DataFrame:
        idx = np.concatenate(parts) if parts else np.array([], dtype=int)
        idx = rng.permutation(idx)  # mix classes; still deterministic
        return df.iloc[idx].reset_index(drop=True)

    return _take(train_pos), _take(val_pos), _take(test_pos)
