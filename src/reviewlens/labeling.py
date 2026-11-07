"""Rule-based silver labels: aspect lexicon -> insight phrases.

Contract: INTERFACES.md section 5.

The input must already be the output of ``preprocess.clean_text``: negation
variants ("gak", "ga", "tdk", ...) are canonical, so this module only knows the
canonical ``NEGATIONS`` imported from ``preprocess``.

Algorithm (deterministic):
1. Split the review into tokens and into clauses at contrast words ("tapi",
   "namun", ...). Cues never cross a clause boundary.
2. Each lexicon lookup uses a matching key: a trailing "nya" is stripped
   ("pengirimannya" -> "pengiriman") and doubled letters are squashed
   ("mantaap" -> "mantap"), so stretched words still match.
3. Every cue token is owned by exactly one aspect: the aspect with a keyword
   nearest to it (at most ``_WINDOW`` words away, same clause). A token that
   is both keyword and cue of an aspect ("rusak", "murah") is a weaker match,
   so "kemasan rusak" belongs to ``kemasan``, not ``fungsi_kerusakan``.
4. A word from ``NEGATIONS`` within 2 words before a cue flips its polarity
   ("tidak cepat" -> neg, "tidak rusak" -> pos).
5. An aspect is detected when it has a keyword not owned by another aspect,
   or owns a cue. Polarity = majority vote of its cues; with no cues (or a
   tie) it falls back to ``sentiment``; neutral/None -> the aspect is dropped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .preprocess import NEGATIONS


@dataclass(frozen=True)
class Aspect:
    name: str                  # "pengiriman"
    keywords: tuple[str, ...]  # aspect triggers (normalised form)
    pos_cues: tuple[str, ...]  # "cepat", "kilat"
    neg_cues: tuple[str, ...]  # "lama", "lambat", "telat"
    pos_phrase: str            # "pengiriman cepat"
    neg_phrase: str            # "pengiriman lambat"
    suggestion: str            # "percepat proses pengiriman"


@dataclass
class InsightLabel:
    positives: list[str]
    pain_points: list[str]
    suggestions: list[str]


# ---------------------------------------------------------------------------
# Lexicon. Every token is a single word that is a fixed point of clean_text.
# Every phrase is unique across the whole lexicon (checked by tests).
# ---------------------------------------------------------------------------

ASPECTS: list[Aspect] = [
    Aspect(
        name="kualitas_produk",
        keywords=("kualitas", "barang", "produk", "bahan", "material",
                  "layar", "gambar", "suara"),
        pos_cues=("bagus", "baik", "mantap", "keren", "oke", "kokoh", "kuat",
                  "halus", "jernih", "tajam", "berkualitas", "premium", "jos",
                  "top"),
        neg_cues=("jelek", "buruk", "murahan", "rapuh", "tipis", "kasar",
                  "buram", "ringkih"),
        pos_phrase="kualitas produk bagus",
        neg_phrase="kualitas produk buruk",
        suggestion="tingkatkan kualitas bahan produk",
    ),
    Aspect(
        name="fungsi_kerusakan",
        keywords=("fungsi", "berfungsi", "nyala", "menyala", "mati", "rusak",
                  "kerusakan", "cacat", "error", "eror", "bekerja", "macet"),
        pos_cues=("berfungsi", "normal", "lancar", "nyala", "menyala", "awet",
                  "sempurna"),
        neg_cues=("rusak", "kerusakan", "mati", "cacat", "error", "eror",
                  "macet", "lemot", "hang", "bermasalah", "korslet"),
        pos_phrase="produk berfungsi dengan baik",
        neg_phrase="produk rusak atau cacat",
        suggestion="perketat quality control sebelum pengiriman",
    ),
    Aspect(
        name="pengiriman",
        keywords=("pengiriman", "kirim", "dikirim", "kiriman", "sampai",
                  "datang", "kurir", "ekspedisi", "paket",
                  # standalone speed words ("tidak cepat") also trigger it
                  "cepat", "kilat", "lambat", "telat", "terlambat"),
        pos_cues=("cepat", "kilat", "tepat", "ekspres", "lancar", "selamat"),
        neg_cues=("lama", "lambat", "telat", "terlambat", "lelet", "molor",
                  "nyasar"),
        pos_phrase="pengiriman cepat",
        neg_phrase="pengiriman lambat",
        suggestion="percepat proses pengiriman",
    ),
    Aspect(
        name="kemasan",
        keywords=("kemasan", "packing", "packaging", "bungkus", "bungkusan",
                  "pembungkus", "dikemas", "dus", "kardus", "box", "bubble"),
        pos_cues=("rapi", "rapih", "aman", "tebal", "rapat", "bagus"),
        neg_cues=("rusak", "penyok", "sobek", "robek", "hancur", "tipis",
                  "asal", "berantakan", "basah", "lecek"),
        pos_phrase="kemasan rapi dan aman",
        neg_phrase="kemasan rusak atau kurang aman",
        suggestion="perbaiki kemasan agar lebih aman",
    ),
    Aspect(
        name="kesesuaian_deskripsi",
        keywords=("deskripsi", "foto", "pesanan", "spesifikasi", "spek",
                  "sesuai", "iklan", "keterangan", "warna", "tipe", "varian"),
        pos_cues=("sesuai", "persis", "cocok"),
        neg_cues=("beda", "berbeda", "salah"),
        pos_phrase="barang sesuai deskripsi",
        neg_phrase="barang tidak sesuai deskripsi",
        suggestion="sesuaikan deskripsi dengan produk asli",
    ),
    Aspect(
        name="harga",
        keywords=("harga", "murah", "mahal", "diskon", "promo", "worth",
                  "budget", "uang"),
        pos_cues=("murah", "terjangkau", "worth", "worthit", "sepadan",
                  "sebanding", "hemat", "ekonomis", "bersahabat"),
        neg_cues=("mahal", "kemahalan", "overprice", "overpriced", "rugi"),
        pos_phrase="harga terjangkau",
        neg_phrase="harga terlalu mahal",
        suggestion="sesuaikan harga dengan kualitas",
    ),
    Aspect(
        name="pelayanan_penjual",
        keywords=("seller", "penjual", "toko", "pelayanan", "layanan",
                  "respon", "respons", "admin", "cs", "olshop", "lapak"),
        pos_cues=("ramah", "responsif", "cepat", "sigap", "baik", "amanah",
                  "jujur", "mantap", "recommended", "rekomended", "rekomen",
                  "terpercaya"),
        neg_cues=("lambat", "lama", "slow", "judes", "jutek", "cuek", "kasar",
                  "lelet", "buruk", "mengecewakan"),
        pos_phrase="pelayanan penjual ramah",
        neg_phrase="pelayanan penjual kurang responsif",
        suggestion="tingkatkan respons dan keramahan penjual",
    ),
    Aspect(
        name="keaslian",
        keywords=("ori", "original", "orisinal", "asli", "kw", "palsu",
                  "replika", "tiruan", "resmi"),
        pos_cues=("ori", "original", "orisinal", "asli", "resmi", "genuine"),
        neg_cues=("kw", "palsu", "replika", "tiruan", "abal", "refurbish",
                  "bekas"),
        pos_phrase="produk asli dan original",
        neg_phrase="keaslian produk diragukan",
        suggestion="pastikan keaslian produk yang dijual",
    ),
    Aspect(
        name="garansi",
        keywords=("garansi", "warranty", "klaim", "servis", "service",
                  "retur", "return"),
        pos_cues=("resmi", "panjang", "aman", "berlaku", "mudah", "jelas",
                  "lengkap"),
        neg_cues=("hangus", "ditolak", "susah", "ribet", "sulit", "palsu"),
        pos_phrase="garansi resmi dan jelas",
        neg_phrase="klaim garansi bermasalah",
        suggestion="permudah proses klaim garansi",
    ),
]

# Fallback phrases when no aspect is detected.
_GENERAL_POSITIVE = "produk memuaskan secara umum"
_GENERAL_PAIN_POINT = "pelanggan kurang puas secara umum"
_GENERAL_SUGGESTION = "tingkatkan kualitas layanan secara umum"

_MAX_ITEMS = 3        # per section
_NEG_WINDOW = 2       # negation must be within 2 words before a cue
_WINDOW = 3           # max distance between a cue and its aspect keyword
_SELF_DISTANCE = _WINDOW + 0.5  # keyword-and-cue in one token: weakest match

_CLAUSE_BREAKS = frozenset({
    "tapi", "tetapi", "namun", "cuma", "cuman", "sayang", "sayangnya",
    "walaupun", "meskipun", "padahal", "sedangkan",
})

_SENTIMENT_TO_POLARITY = {"positive": "pos", "negative": "neg"}

_DOUBLE_LETTER_RE = re.compile(r"([^\W\d_])\1")


def _key(token: str) -> str:
    """Matching key: strip a trailing "nya", squash doubled letters."""
    if len(token) > 5 and token.endswith("nya"):
        token = token[:-3]
    return _DOUBLE_LETTER_RE.sub(r"\1", token)


# Pre-computed lookup tables (index i follows ASPECTS order).
_KEYWORDS: list[frozenset[str]] = [
    frozenset(_key(k) for k in a.keywords) for a in ASPECTS
]
_CUES: list[dict[str, str]] = [
    {**{_key(c): "pos" for c in a.pos_cues}, **{_key(c): "neg" for c in a.neg_cues}}
    for a in ASPECTS
]


def _flip(polarity: str) -> str:
    return "neg" if polarity == "pos" else "pos"


def _detect(review_clean: str, sentiment: str | None) -> list[tuple[int, int, str, bool]]:
    """Return ``(first position, aspect idx, polarity, from_cue)`` sorted by position."""
    tokens = review_clean.split() if review_clean else []
    keys = [_key(t) for t in tokens]

    # Clause id per token; clause-break tokens get -1 and match nothing.
    clause: list[int] = []
    current = 0
    for tok in tokens:
        if tok in _CLAUSE_BREAKS:
            current += 1
            clause.append(-1)
        else:
            clause.append(current)

    def same_clause(p: int, q: int) -> bool:
        return clause[p] >= 0 and clause[p] == clause[q]

    keyword_pos = [
        [p for p, k in enumerate(keys) if clause[p] >= 0 and k in _KEYWORDS[i]]
        for i in range(len(ASPECTS))
    ]

    # Assign every cue token to exactly one aspect (nearest keyword wins;
    # ties go to the earlier aspect in ASPECTS).
    owner: dict[int, tuple[int, str]] = {}
    for p, k in enumerate(keys):
        if clause[p] < 0:
            continue
        best: tuple[tuple[float, int], str] | None = None
        for i in range(len(ASPECTS)):
            polarity = _CUES[i].get(k)
            if polarity is None:
                continue
            dists = [abs(p - q) for q in keyword_pos[i]
                     if q != p and same_clause(p, q) and abs(p - q) <= _WINDOW]
            if dists:
                dist: float = min(dists)
            elif p in keyword_pos[i]:
                dist = _SELF_DISTANCE
            else:
                continue
            if best is None or (dist, i) < best[0]:
                best = ((dist, i), polarity)
        if best is None:
            continue
        polarity = best[1]
        negated = any(
            p - j >= 0 and same_clause(p, p - j) and tokens[p - j] in NEGATIONS
            for j in range(1, _NEG_WINDOW + 1)
        )
        owner[p] = (best[0][1], _flip(polarity) if negated else polarity)

    fallback = _SENTIMENT_TO_POLARITY.get(
        sentiment.strip().lower() if isinstance(sentiment, str) else ""
    )

    found: list[tuple[int, int, str, bool]] = []
    for i in range(len(ASPECTS)):
        triggers = [q for q in keyword_pos[i] if q not in owner or owner[q][0] == i]
        owned = [p for p, (j, _) in owner.items() if j == i]
        if not triggers and not owned:
            continue
        n_pos = sum(1 for p in owned if owner[p][1] == "pos")
        n_neg = len(owned) - n_pos
        if n_pos > n_neg:
            polarity, from_cue = "pos", True
        elif n_neg > n_pos:
            polarity, from_cue = "neg", True
        elif fallback is not None:  # no cue, or a tie
            polarity, from_cue = fallback, False
        else:                       # neutral / None -> ignore
            continue
        found.append((min(triggers + owned), i, polarity, from_cue))

    found.sort()  # order of first mention in the review, then ASPECTS order
    return found


def _select(items: list[tuple[int, int, str, bool]]) -> list[int]:
    """Keep at most ``_MAX_ITEMS`` aspects, cue-based ones first, in mention order."""
    chosen = sorted(items, key=lambda t: (not t[3], t[0], t[1]))[:_MAX_ITEMS]
    return [i for _, i, _, _ in sorted(chosen)]


def detect_aspects(review_clean: str) -> dict[str, str]:
    """Return ``{aspect_name: "pos" | "neg"}`` decided by lexicon cues only.

    Aspects mentioned without any polarity cue are omitted here (there is no
    sentiment to fall back on); ``label_review`` adds those via ``sentiment``.
    Keys are ordered by first mention in the review.
    """
    return {ASPECTS[i].name: polarity for _, i, polarity, _ in _detect(review_clean, None)}


def label_review(review_clean: str, sentiment: str | None) -> InsightLabel:
    """Build the silver insight label for one cleaned review.

    ``review_clean`` must be the output of ``preprocess.clean_text``.
    ``sentiment`` is the weak label ("positive" / "negative" / "neutral") or
    ``None``. At most 3 items per section; ``suggestions[i]`` is the suggestion
    for ``pain_points[i]``. When more than 3 aspects share a polarity,
    cue-based aspects are kept before sentiment-fallback ones.
    """
    found = _detect(review_clean, sentiment)
    pos_idx = _select([t for t in found if t[2] == "pos"])
    neg_idx = _select([t for t in found if t[2] == "neg"])

    positives = [ASPECTS[i].pos_phrase for i in pos_idx]
    pain_points = [ASPECTS[i].neg_phrase for i in neg_idx]
    suggestions = [ASPECTS[i].suggestion for i in neg_idx]

    if not found:
        polarity = _SENTIMENT_TO_POLARITY.get(
            sentiment.strip().lower() if isinstance(sentiment, str) else ""
        )
        if polarity == "pos":
            positives = [_GENERAL_POSITIVE]
        elif polarity == "neg":
            pain_points = [_GENERAL_PAIN_POINT]
            suggestions = [_GENERAL_SUGGESTION]

    return InsightLabel(positives=positives, pain_points=pain_points,
                        suggestions=suggestions)
