"""Unit tests for reviewlens.tokenizer (INTERFACES.md sections 2.1, 6, 14).

All data is synthetic; the real dataset is never used.
"""

from __future__ import annotations

import ast
import inspect
import json
import random
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

# Make `src/` importable without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reviewlens import dataset as ds  # noqa: E402
from reviewlens import tokenizer as tk  # noqa: E402
from reviewlens.config import IGNORE_INDEX, ModelConfig  # noqa: E402
from reviewlens.dataset import (  # noqa: E402
    ReviewInsightDataset,
    collate_fn,
    encode_example,
    format_prompt,
    format_target,
)
from reviewlens.labeling import InsightLabel, label_review  # noqa: E402
from reviewlens.model import MiniGPT  # noqa: E402
from reviewlens.preprocess import clean_text  # noqa: E402
from reviewlens.tokenizer import (  # noqa: E402
    BOS, EOS, NEG, NONE, PAD, POS, REVIEW, SEP, SPECIAL_TOKENS, SUG, UNK,
    WordTokenizer,
    build_vocab_from_jsonl,
)

EXPECTED_SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>", "<review>",
                     "<pos>", "<neg>", "<sug>", "<sep>", "<none>"]


def tok_from(*texts: str, **kw) -> WordTokenizer:
    kw.setdefault("min_freq", 1)
    return WordTokenizer.build(list(texts), **kw)


def write_jsonl(path: Path, records) -> Path:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                    encoding="utf-8")
    return path


def record(review, pos=(), neg=(), sug=(), sentiment="positive", item_id=1):
    return {"review": review, "sentiment": sentiment, "rating": 5.0, "item_id": item_id,
            "positives": list(pos), "pain_points": list(neg), "suggestions": list(sug)}


# ---------------------------------------------------------------------------
# Special tokens
# ---------------------------------------------------------------------------


def test_special_token_ids_are_fixed_ints():
    assert list(SPECIAL_TOKENS) == EXPECTED_SPECIALS
    ids = (PAD, UNK, BOS, EOS, REVIEW, POS, NEG, SUG, SEP, NONE)
    assert ids == tuple(range(10))
    assert all(type(i) is int for i in ids)
    assert PAD == 0 and type(PAD) is int
    assert tk.PAD_TOKEN == "<pad>" and tk.NONE_TOKEN == "<none>"


def test_special_ids_in_built_vocab():
    for tok in (tok_from(), tok_from("barang bagus"), tok_from("zzz " * 5, min_freq=3)):
        assert tok.encode(" ".join(EXPECTED_SPECIALS)) == list(range(10))
        assert tok.itos[:10] == EXPECTED_SPECIALS


def test_ignore_index_is_not_a_token_id():
    assert IGNORE_INDEX == -100 and IGNORE_INDEX not in range(10)
    with pytest.raises(ValueError):
        tok_from("a").decode([IGNORE_INDEX])


def test_agrees_with_dataset_private_mirror():
    # dataset.py keeps private copies; they must match the canonical owner.
    assert ds._PAD_ID == PAD
    assert (ds._BOS, ds._REVIEW, ds._POS, ds._NEG, ds._SUG, ds._SEP, ds._NONE, ds._EOS) == (
        "<bos>", "<review>", "<pos>", "<neg>", "<sug>", "<sep>", "<none>", "<eos>")
    assert ds._SPECIAL_STRINGS == frozenset(SPECIAL_TOKENS)


# ---------------------------------------------------------------------------
# Encode / decode
# ---------------------------------------------------------------------------


def test_encode_decode_round_trip():
    tok = tok_from("barang bagus pengiriman cepat")
    s = "pengiriman cepat barang bagus"
    ids = tok.encode(s)
    assert len(ids) == 4 and all(i >= 10 for i in ids)
    assert tok.decode(ids) == s


def test_round_trip_with_special_tokens():
    tok = tok_from("barang bagus kualitas produk")
    s = "<bos> <review> barang bagus <pos> kualitas produk bagus <neg> <none> <sug> <none> <eos>"
    ids = tok.encode(s)
    assert ids[:2] == [BOS, REVIEW] and ids[-1] == EOS
    assert tok.decode(ids) == s
    assert tok.decode(ids, skip_special=True) == "barang bagus kualitas produk bagus"


def test_one_word_one_token_and_whitespace():
    tok = tok_from("a b c")
    assert tok.encode("a  b\tc\n") == tok.encode("a b c")
    assert len(tok.encode("a b c d e")) == 5
    assert tok.decode(tok.encode("  a   b ")) == "a b"


def test_decode_accepts_tensor_and_tuple():
    tok = tok_from("barang bagus")
    ids = tok.encode("<bos> barang bagus <eos>")
    assert tok.decode(torch.tensor(ids)) == tok.decode(tuple(ids)) == "<bos> barang bagus <eos>"


# ---------------------------------------------------------------------------
# <unk>
# ---------------------------------------------------------------------------


def test_unknown_words_become_unk():
    tok = tok_from("barang bagus")
    assert tok.encode("barang jelek sekali") == [tok.stoi["barang"], UNK, UNK]
    assert UNK == 1
    assert tok.decode(tok.encode("jelek")) == "<unk>"


def test_unknown_words_are_case_and_form_sensitive():
    tok = tok_from("bagus")
    assert tok.encode("Bagus") == [UNK]   # clean_text lowercases before encoding


# ---------------------------------------------------------------------------
# Special tokens are never split
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("special", EXPECTED_SPECIALS)
def test_special_token_is_one_token(special):
    tok = tok_from("barang")
    assert tok.encode(special) == [EXPECTED_SPECIALS.index(special)]
    assert tok.encode(f"barang {special} barang")[1] == EXPECTED_SPECIALS.index(special)


def test_specials_glued_to_words_are_not_split_into_pieces():
    tok = tok_from("bagus")
    # Whitespace-only rule: "bagus<pos>" is one (unknown) word, not "bagus" + "<pos>".
    assert tok.encode("bagus<pos>") == [UNK]
    assert tok.encode("<review>") != tok.encode("< review >")


def test_special_strings_in_texts_are_not_counted_twice():
    tok = tok_from("<pos> <pos> <pos> barang")
    assert tok.vocab_size == 11
    assert list(tok.stoi.values()) == list(range(11))


# ---------------------------------------------------------------------------
# Vocabulary size and min_freq
# ---------------------------------------------------------------------------


def test_default_parameters():
    sig = inspect.signature(WordTokenizer.build)
    assert sig.parameters["min_freq"].default == 3
    assert sig.parameters["max_size"].default == 10_000


def test_vocab_capped_at_10000_including_specials():
    texts = [" ".join(f"w{i:05d}" for i in range(20_000))] * 3   # 20k words, each freq 3
    tok = WordTokenizer.build(texts)
    assert tok.vocab_size == 10_000
    assert tok.itos[:10] == EXPECTED_SPECIALS
    assert tok.itos[10] == "w00000" and tok.itos[-1] == "w09989"


def test_small_max_size_keeps_most_frequent():
    tok = tok_from("a a a a b b b c c d", max_size=12)
    assert tok.itos[10:] == ["a", "b"]


def test_max_size_none_means_no_limit():
    tok = tok_from(" ".join(f"x{i}" for i in range(50)), max_size=None)
    assert tok.vocab_size == 60


def test_min_freq_three():
    texts = ["satu dua dua tiga tiga tiga empat empat empat empat"]
    tok = WordTokenizer.build(texts)            # default min_freq = 3
    assert "satu" not in tok.stoi and "dua" not in tok.stoi
    assert "tiga" in tok.stoi and "empat" in tok.stoi
    assert tok.encode("dua") == [UNK]
    assert tok.vocab_size == 12


def test_min_freq_counts_across_texts():
    tok = WordTokenizer.build(["bagus", "bagus", "bagus jelek", "jelek"], min_freq=3)
    assert "bagus" in tok.stoi and "jelek" not in tok.stoi


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_frequency_order_with_alphabetical_ties():
    tok = tok_from("zeta alpha alpha beta beta gamma gamma gamma delta")
    assert tok.itos[10:] == ["gamma", "alpha", "beta", "delta", "zeta"]


def test_vocab_independent_of_text_order():
    words = [f"kata{i}" for i in range(200)] * 4
    texts = [" ".join(words[i:i + 7]) for i in range(0, len(words), 7)]
    a = WordTokenizer.build(texts)
    shuffled = texts[:]
    random.Random(0).shuffle(shuffled)
    b = WordTokenizer.build(shuffled)
    assert a.stoi == b.stoi and a == b
    assert WordTokenizer.build(texts).stoi == a.stoi


# ---------------------------------------------------------------------------
# Training split only (build_vocab_from_jsonl)
# ---------------------------------------------------------------------------


def test_vocab_built_from_train_only(tmp_path):
    train = write_jsonl(tmp_path / "train.jsonl", [
        record("barang bagus", pos=["kualitas produk bagus"]),
        record("pengiriman lama", neg=["pengiriman lambat"], sug=["percepat proses pengiriman"],
               sentiment="negative", item_id=2),
    ])
    write_jsonl(tmp_path / "val.jsonl", [record("valkata khusus", pos=["harga terjangkau"])])
    tok = build_vocab_from_jsonl(train, min_freq=1)
    # review words and target phrase words (via format_target) are in the vocab
    for w in ("barang", "bagus", "kualitas", "produk", "lambat", "percepat", "proses"):
        assert w in tok.stoi, w
    # words that only appear in val are unknown
    assert tok.encode("valkata khusus terjangkau") == [UNK, UNK, UNK]


def test_oversampled_copies_do_not_inflate_frequency(tmp_path):
    neg = record("kardus penyok", neg=["kemasan rusak atau kurang aman"],
                 sug=["perbaiki kemasan agar lebih aman"], sentiment="negative")
    pos = record("barang bagus", pos=["kualitas produk bagus"], item_id=2)
    train = write_jsonl(tmp_path / "train.jsonl", [neg, pos, dict(neg), dict(neg)])
    tok = build_vocab_from_jsonl(train)              # min_freq = 3
    texts, n_records, n_unique = tk._train_corpus(train)
    assert (n_records, n_unique) == (4, 2)
    assert "penyok" not in tok.stoi                  # 1 unique occurrence, not 3
    # "bagus": review + target phrase = 2 occurrences -> below min_freq 3
    assert "bagus" not in tok.stoi


def test_same_text_different_item_counts_twice(tmp_path):
    rows = [record("barang mantap", item_id=i) for i in range(3)]
    tok = build_vocab_from_jsonl(write_jsonl(tmp_path / "train.jsonl", rows))
    assert "mantap" in tok.stoi                      # 3 genuinely different records


def test_build_vocab_from_jsonl_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_vocab_from_jsonl(tmp_path / "missing.jsonl")
    (tmp_path / "empty.jsonl").write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError):
        build_vocab_from_jsonl(tmp_path / "empty.jsonl")
    (tmp_path / "bad.jsonl").write_text('{"review": "a"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="expected keys"):
        build_vocab_from_jsonl(tmp_path / "bad.jsonl")


def test_cli_writes_vocab_json(tmp_path, capsys):
    rows = [record("barang bagus sekali", pos=["kualitas produk bagus"], item_id=i)
            for i in range(3)]
    write_jsonl(tmp_path / "train.jsonl", rows)
    assert tk.main(["--data-dir", str(tmp_path)]) == 0
    tok = WordTokenizer.load(tmp_path / "vocab.json")
    assert "sekali" in tok.stoi
    assert "vocab_size" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Save / load vocab.json
# ---------------------------------------------------------------------------


def test_save_load_round_trip(tmp_path):
    tok = tok_from("barang bagus pengiriman cepat kemasan rapi ñandú")
    path = tmp_path / "nested" / "dir" / "vocab.json"
    tok.save(path)
    loaded = WordTokenizer.load(path)
    assert loaded == tok
    assert loaded.stoi == tok.stoi and loaded.itos == tok.itos
    assert loaded.vocab_size == tok.vocab_size
    s = "<bos> <review> barang bagus <pos> ñandú <eos>"
    assert loaded.encode(s) == tok.encode(s)
    assert not list(path.parent.glob("*.tmp"))


def test_vocab_json_structure_is_stable(tmp_path):
    tok = tok_from("b a a")
    p1, p2 = tmp_path / "v1.json", tmp_path / "v2.json"
    tok.save(p1)
    WordTokenizer.load(p1).save(p2)
    assert p1.read_bytes() == p2.read_bytes()
    data = json.loads(p1.read_text(encoding="utf-8"))
    assert data == {
        "special_tokens": EXPECTED_SPECIALS,
        "vocab_size": 12,
        "stoi": {**{t: i for i, t in enumerate(EXPECTED_SPECIALS)}, "a": 10, "b": 11},
    }
    assert list(data["stoi"].values()) == list(range(12))   # written in ID order


def test_load_accepts_minimal_contract_format(tmp_path):
    path = tmp_path / "vocab.json"
    stoi = {t: i for i, t in enumerate(EXPECTED_SPECIALS)} | {"halo": 10}
    path.write_text(json.dumps({"stoi": stoi}), encoding="utf-8")
    assert WordTokenizer.load(path).encode("halo") == [10]


@pytest.mark.parametrize("payload", [
    [],                                                              # not an object
    {"vocab": {}},                                                   # missing stoi
    {"stoi": {"<pad>": 1, "<unk>": 0}},                              # wrong special IDs
    {"stoi": {t: i for i, t in enumerate(EXPECTED_SPECIALS)} | {"a": 11}},   # gap
    {"stoi": {t: i for i, t in enumerate(EXPECTED_SPECIALS)} | {"a": 9}},    # duplicate ID
    {"stoi": {t: i for i, t in enumerate(EXPECTED_SPECIALS)} | {"a": "10"}}, # non-int ID
    {"stoi": {t: i for i, t in enumerate(EXPECTED_SPECIALS)} | {"a b": 10}}, # 2 words
    {"stoi": {t: i for i, t in enumerate(EXPECTED_SPECIALS)}, "vocab_size": 99},
])
def test_load_rejects_invalid_vocab(tmp_path, payload):
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        WordTokenizer.load(path)


def test_load_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        WordTokenizer.load(tmp_path / "nope.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        WordTokenizer.load(bad)


# ---------------------------------------------------------------------------
# Empty and invalid input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_encode_empty(text):
    assert tok_from("a").encode(text) == []


def test_decode_empty():
    assert tok_from("a").decode([]) == ""


@pytest.mark.parametrize("bad", [None, 123, ["barang"], b"barang"])
def test_encode_rejects_non_string(bad):
    with pytest.raises(TypeError):
        tok_from("a").encode(bad)


def test_decode_rejects_invalid_ids():
    tok = tok_from("a")
    with pytest.raises(ValueError):
        tok.decode([tok.vocab_size])
    with pytest.raises(ValueError):
        tok.decode([-1])
    with pytest.raises(TypeError):
        tok.decode(["a"])
    with pytest.raises(TypeError):
        tok.decode([1.0])
    with pytest.raises(TypeError):
        tok.decode("abc")


def test_build_from_empty_corpus_has_only_specials():
    tok = WordTokenizer.build([])
    assert tok.vocab_size == 10
    assert tok.encode("apa saja") == [UNK, UNK]


@pytest.mark.parametrize("kwargs,exc", [
    (dict(texts="barang bagus"), TypeError),          # bare string, not a list of texts
    (dict(texts=["ok", None]), TypeError),
    (dict(texts=["ok", 5]), TypeError),
    (dict(texts=["ok"], min_freq=0), ValueError),
    (dict(texts=["ok"], min_freq=1.5), ValueError),
    (dict(texts=["ok"], max_size=5), ValueError),     # smaller than the 10 specials
])
def test_build_rejects_invalid_arguments(kwargs, exc):
    with pytest.raises(exc):
        WordTokenizer.build(**kwargs)


@pytest.mark.parametrize("stoi", [None, {}, {"<pad>": 0}])
def test_constructor_validates_stoi(stoi):
    with pytest.raises(ValueError):
        WordTokenizer(stoi)


def test_stoi_and_itos_are_copies():
    tok = tok_from("a")
    tok.stoi["zzz"] = 99
    tok.itos.append("zzz")
    assert "zzz" not in tok.stoi and tok.vocab_size == 11


# ---------------------------------------------------------------------------
# Compatibility with ReviewInsightDataset, collate_fn and MiniGPT
# ---------------------------------------------------------------------------

RAW = [
    ("Barang bagus bgt, pengiriman cepat!", "positive"),
    ("pengiriman lama, kardus penyok", "negative"),
    ("harga murah tp kualitas jelek", "negative"),
    ("seller ramah, packing rapi", "positive"),
    ("biasa aja", "neutral"),
    ("gak nyala, kecewa", "negative"),
]


def _records(n_copies=4):
    recs = []
    for k in range(n_copies):
        for i, (raw, sentiment) in enumerate(RAW):
            review = clean_text(raw)
            lab = label_review(review, sentiment)
            recs.append(record(review, lab.positives, lab.pain_points, lab.suggestions,
                               sentiment=sentiment, item_id=100 * k + i))
    return recs


def test_dataset_collate_and_minigpt(tmp_path):
    train = write_jsonl(tmp_path / "train.jsonl", _records())
    tok = build_vocab_from_jsonl(train)              # real defaults, min_freq = 3
    assert tok.vocab_size > 10

    data = ReviewInsightDataset(train, tok, block_size=128)
    assert len(data) == 24
    for item in data:
        assert item.dtype == torch.long
        assert int(item.min()) >= 0 and int(item.max()) < tok.vocab_size
        assert item[0] == BOS and item[1] == REVIEW and item[-1] == EOS
        assert UNK not in item.tolist()               # every train word is in vocab

    first = tok.decode(data[0])
    assert first.startswith("<bos> <review> ") and first.endswith(" <eos>")
    assert " <pos> " in first and " <neg> " in first and " <sug> " in first

    loader = DataLoader(data, batch_size=8, shuffle=False, collate_fn=collate_fn)
    x, y = next(iter(loader))
    assert x.shape == y.shape and x.shape[1] <= 128
    assert torch.all((y == IGNORE_INDEX) | ((y >= 0) & (y < tok.vocab_size)))

    torch.manual_seed(0)
    model = MiniGPT(ModelConfig(vocab_size=tok.vocab_size, block_size=128, n_layer=1,
                                n_head=2, d_model=32, d_ff=64, dropout=0.0))
    logits, loss = model(x, y)
    assert logits.shape == (*x.shape, tok.vocab_size)
    assert torch.isfinite(loss)


def test_prompt_encoding_matches_dataset_format():
    tok = tok_from("barang bagus")
    prompt_ids = tok.encode(format_prompt("barang bagus"))
    assert prompt_ids == [BOS, REVIEW, tok.stoi["barang"], tok.stoi["bagus"], POS]
    target = format_target(InsightLabel(["harga terjangkau"], [], []))
    tgt_ids = tok.encode(target)
    assert tgt_ids[0] == POS and tgt_ids[-1] == EOS
    assert tgt_ids.count(NONE) == 2 and NEG in tgt_ids and SUG in tgt_ids


def test_encode_example_with_real_tokenizer():
    tok = tok_from("barang bagus harga murah")
    ids = encode_example("barang bagus", InsightLabel(["harga murah"], [], []), tok, 128)
    assert ids[:2] == [BOS, REVIEW] and ids[-1] == EOS and len(ids) <= 129


# ---------------------------------------------------------------------------
# No pretrained tokenizer
# ---------------------------------------------------------------------------


def test_uses_only_standard_library_and_project_modules():
    source = inspect.getsource(tk)
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            imported.add(node.module.split(".")[0])
    stdlib = {"__future__", "argparse", "json", "os", "collections", "pathlib", "typing"}
    assert imported <= stdlib, imported - stdlib
    lowered = source.lower()
    for forbidden in ("transformers", "sentencepiece", "tiktoken", "from tokenizers",
                      "import tokenizers", "autotokenizer", "berttokenizer", "gpt2tokenizer"):
        assert forbidden not in lowered, forbidden
