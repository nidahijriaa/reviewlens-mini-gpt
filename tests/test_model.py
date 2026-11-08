"""Unit tests for reviewlens.model (INTERFACES.md sections 8 and 14)."""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

# Make `src/` importable without installing the package.
SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from reviewlens.attention import MultiHeadSelfAttention  # noqa: E402
from reviewlens.config import ModelConfig  # noqa: E402
from reviewlens.model import (  # noqa: E402
    IGNORE_INDEX,
    FeedForward,
    MiniGPT,
    TransformerBlock,
)


def small_cfg(**overrides) -> ModelConfig:
    kw = dict(vocab_size=50, block_size=16, n_layer=2, n_head=2, d_model=32, d_ff=64, dropout=0.1)
    kw.update(overrides)
    return ModelConfig(**kw)


def make_model(seed: int = 0, **overrides) -> MiniGPT:
    torch.manual_seed(seed)
    return MiniGPT(small_cfg(**overrides))


def expected_params(V: int, T: int, L: int, d: int, d_ff: int) -> int:
    """PROJECT_CONTEXT.md 8.2: tok_emb + pos_emb + L * block + ln_f (lm_head tied -> 0)."""
    attn = (d * 3 * d + 3 * d) + (d * d + d)
    ff = (d * d_ff + d_ff) + (d_ff * d + d)
    ln = 2 * d
    block = attn + ff + 2 * ln
    return V * d + T * d + L * block + ln


# ---------------------------------------------------------------------------
# 1-4. forward: shapes, loss
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("B, T", [(1, 1), (2, 7), (4, 16)])
def test_logits_shape(B, T):
    model = make_model()
    idx = torch.randint(0, 50, (B, T))
    logits, _ = model(idx)
    assert logits.shape == (B, T, 50)


def test_forward_without_targets_returns_none_loss():
    model = make_model()
    logits, loss = model(torch.randint(0, 50, (2, 8)))
    assert loss is None
    assert torch.isfinite(logits).all()


def test_forward_with_targets_returns_scalar_loss():
    model = make_model()
    idx = torch.randint(0, 50, (2, 8))
    tgt = torch.randint(0, 50, (2, 8))
    logits, loss = model(idx, tgt)
    assert loss is not None and loss.dim() == 0
    expected = F.cross_entropy(logits.view(-1, 50), tgt.view(-1))
    torch.testing.assert_close(loss, expected)


def test_loss_is_finite_and_backpropagates():
    model = make_model()
    idx = torch.randint(0, 50, (3, 16))
    _, loss = model(idx, torch.randint(0, 50, (3, 16)))
    assert torch.isfinite(loss)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


def test_loss_ignores_minus_100_targets():
    model = make_model().eval()
    idx = torch.randint(0, 50, (2, 10))
    tgt = torch.randint(0, 50, (2, 10))
    tgt[0, 6:] = IGNORE_INDEX  # right padding
    tgt[1, 9:] = IGNORE_INDEX
    logits, loss = model(idx, tgt)
    keep = tgt != IGNORE_INDEX
    manual = F.cross_entropy(logits[keep], tgt[keep])
    torch.testing.assert_close(loss, manual)
    assert IGNORE_INDEX == -100


def test_rejects_mismatched_targets_shape():
    model = make_model()
    with pytest.raises(ValueError, match="targets"):
        model(torch.randint(0, 50, (2, 8)), torch.randint(0, 50, (2, 7)))


# ---------------------------------------------------------------------------
# 5. random initialisation
# ---------------------------------------------------------------------------
def test_initial_loss_close_to_ln_vocab():
    """Fresh random init -> near-uniform predictions -> loss ~ ln(V)."""
    V = 1000
    model = make_model(vocab_size=V).eval()
    idx = torch.randint(0, V, (8, 16))
    _, loss = model(idx, torch.randint(0, V, (8, 16)))
    assert abs(loss.item() - math.log(V)) < 0.1


def test_init_weights_distribution():
    model = make_model(vocab_size=2000, d_model=64, d_ff=256)
    for name, m in model.named_modules():
        if isinstance(m, (torch.nn.Linear, torch.nn.Embedding)):
            assert abs(m.weight.std().item() - 0.02) < 0.004, name
            assert abs(m.weight.mean().item()) < 0.004, name
            if getattr(m, "bias", None) is not None:
                assert torch.count_nonzero(m.bias) == 0, name
        elif isinstance(m, torch.nn.LayerNorm):
            assert torch.all(m.weight == 1), name
            assert torch.all(m.bias == 0), name


def test_different_seeds_give_different_weights():
    a, b = make_model(seed=0), make_model(seed=1)
    assert not torch.equal(a.tok_emb.weight, b.tok_emb.weight)
    assert not torch.equal(a.blocks[0].attn.qkv.weight, b.blocks[0].attn.qkv.weight)


# ---------------------------------------------------------------------------
# 6. weight tying
# ---------------------------------------------------------------------------
def test_weight_tying():
    model = make_model()
    assert model.lm_head.bias is None
    assert model.lm_head.weight is model.tok_emb.weight
    assert model.lm_head.weight.data_ptr() == model.tok_emb.weight.data_ptr()
    # Still tied after a gradient step.
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    _, loss = model(torch.randint(0, 50, (2, 8)), torch.randint(0, 50, (2, 8)))
    loss.backward()
    opt.step()
    assert torch.equal(model.lm_head.weight, model.tok_emb.weight)


def test_tying_survives_state_dict_roundtrip():
    src, dst = make_model(seed=0), make_model(seed=1)
    dst.load_state_dict(src.state_dict())
    assert dst.lm_head.weight is dst.tok_emb.weight
    assert torch.equal(dst.tok_emb.weight, src.tok_emb.weight)


# ---------------------------------------------------------------------------
# 7. parameter count
# ---------------------------------------------------------------------------
def test_param_count_small_config():
    model = make_model()
    assert model.num_parameters() == expected_params(V=50, T=16, L=2, d=32, d_ff=64)
    # Tied weight counted once: naive count with duplicates would be larger by V*d.
    naive = sum(p.numel() for m in model.modules() for p in m.parameters(recurse=False))
    assert naive - model.num_parameters() == 50 * 32


@pytest.mark.parametrize("V, expected", [(6000, 4_728_320), (10_000, 5_752_320)])
def test_param_count_default_config(V, expected):
    """Matches the official estimate in PROJECT_CONTEXT.md 8.2."""
    model = MiniGPT(ModelConfig(vocab_size=V))
    assert model.num_parameters() == expected == expected_params(V, 128, 4, 256, 1024)


def test_block_param_count_default():
    # PROJECT_CONTEXT 8.2: 789,760 per block.
    block = TransformerBlock(ModelConfig(vocab_size=10))
    assert sum(p.numel() for p in block.parameters()) == 789_760


# ---------------------------------------------------------------------------
# 8. block_size limit
# ---------------------------------------------------------------------------
def test_rejects_sequence_longer_than_block_size():
    model = make_model()
    with pytest.raises(ValueError, match="block_size"):
        model(torch.randint(0, 50, (1, 17)))


def test_accepts_exactly_block_size():
    model = make_model()
    logits, _ = model(torch.randint(0, 50, (1, 16)))
    assert logits.shape == (1, 16, 50)


# ---------------------------------------------------------------------------
# Architecture checks
# ---------------------------------------------------------------------------
def test_architecture_uses_manual_attention_and_pre_ln():
    model = make_model()
    assert len(model.blocks) == 2
    for block in model.blocks:
        assert isinstance(block.attn, MultiHeadSelfAttention)
        assert isinstance(block.ff, FeedForward)
        assert isinstance(block.ln1, torch.nn.LayerNorm)
        assert isinstance(block.ln2, torch.nn.LayerNorm)
    assert model.pos_emb.weight.shape == (16, 32)
    assert isinstance(model.ln_f, torch.nn.LayerNorm)


def test_feedforward_structure():
    ff = FeedForward(32, 64, 0.1)
    assert isinstance(ff.act, torch.nn.GELU)
    assert ff.fc1.in_features == 32 and ff.fc1.out_features == 64
    assert ff.fc2.in_features == 64 and ff.fc2.out_features == 32


def test_model_is_causal():
    """Changing future tokens must not change logits at earlier positions."""
    model = make_model().eval()
    idx = torch.randint(0, 50, (2, 12))
    idx2 = idx.clone()
    idx2[:, 6:] = (idx2[:, 6:] + 1) % 50
    with torch.no_grad():
        a, _ = model(idx)
        b, _ = model(idx2)
    torch.testing.assert_close(a[:, :6], b[:, :6], rtol=0, atol=1e-5)
    assert not torch.allclose(a[:, 6:], b[:, 6:])


def test_no_forbidden_imports():
    src = (SRC / "reviewlens" / "model.py").read_text(encoding="utf-8")
    code = re.sub(r'^"""[\s\S]*?"""', "", src, count=1)
    for forbidden in (
        "transformers",
        "nn.Transformer",
        "TransformerEncoder",
        "TransformerDecoder",
        "MultiheadAttention",
        "scaled_dot_product_attention",
        "tokenizer",
        "preprocess",
    ):
        assert forbidden not in code, forbidden


# ---------------------------------------------------------------------------
# 9-11. generate
# ---------------------------------------------------------------------------
def test_generate_length():
    model = make_model()
    idx = torch.randint(0, 50, (3, 5))
    out = model.generate(idx, max_new_tokens=7, temperature=1.0)
    assert out.shape == (3, 12)
    assert out.dtype == torch.long
    assert torch.equal(out[:, :5], idx)
    assert int(out.min()) >= 0 and int(out.max()) < 50


def test_generate_zero_new_tokens():
    model = make_model()
    idx = torch.randint(0, 50, (2, 4))
    assert torch.equal(model.generate(idx, max_new_tokens=0), idx)


def test_generate_crops_context_beyond_block_size():
    model = make_model()
    idx = torch.randint(0, 50, (2, 16))  # already at block_size
    out = model.generate(idx, max_new_tokens=10, temperature=0.0)
    assert out.shape == (2, 26)


def test_greedy_is_deterministic_and_restores_mode():
    model = make_model(dropout=0.5)
    model.train()
    idx = torch.randint(0, 50, (2, 4))
    a = model.generate(idx, max_new_tokens=8, temperature=0.0)
    b = model.generate(idx, max_new_tokens=8, temperature=0.0)
    assert torch.equal(a, b)
    assert model.training  # mode restored

    # Greedy == argmax of the last-position logits, step by step.
    model.eval()
    seq = idx.clone()
    for _ in range(8):
        logits, _ = model(seq[:, -16:])
        seq = torch.cat([seq, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
    assert torch.equal(a, seq)


def test_top_k_one_equals_greedy():
    model = make_model()
    idx = torch.randint(0, 50, (2, 4))
    torch.manual_seed(123)
    sampled = model.generate(idx, max_new_tokens=6, temperature=1.0, top_k=1)
    greedy = model.generate(idx, max_new_tokens=6, temperature=0.0)
    assert torch.equal(sampled, greedy)


def test_top_k_restricts_candidates():
    model = make_model()
    idx = torch.randint(0, 50, (1, 4))
    model.eval()
    with torch.no_grad():
        logits, _ = model(idx)
    allowed = set(torch.topk(logits[0, -1], 3).indices.tolist())
    for seed in range(20):
        torch.manual_seed(seed)
        out = model.generate(idx, max_new_tokens=1, temperature=1.5, top_k=3)
        assert int(out[0, -1]) in allowed


def test_generate_rejects_invalid_args():
    model = make_model()
    idx = torch.randint(0, 50, (1, 4))
    with pytest.raises(ValueError):
        model.generate(idx, max_new_tokens=-1)
    with pytest.raises(ValueError):
        model.generate(idx, max_new_tokens=2, temperature=-1.0)
    with pytest.raises(ValueError):
        model.generate(idx, max_new_tokens=2, top_k=0)


class ScriptedGPT(MiniGPT):
    """MiniGPT whose next token follows a fixed per-row script (to test EOS logic)."""

    def __init__(self, cfg: ModelConfig, script: list[list[int]], prompt_len: int):
        super().__init__(cfg)
        self.script = script
        self.prompt_len = prompt_len
        self.calls = 0

    def forward(self, idx, targets=None):
        self.calls += 1
        B, T = idx.shape
        step = self.calls - 1
        logits = torch.zeros(B, T, self.cfg.vocab_size)
        for row in range(B):
            logits[row, -1, self.script[row][step]] = 10.0
        return logits, None


def test_generate_stops_when_all_rows_hit_eos():
    EOS = 3
    cfg = small_cfg(block_size=64)
    # Row 0 emits EOS at step 1 then would emit 7s; row 1 emits EOS at step 3.
    script = [[11, EOS, 7, 7, 7, 7], [12, 13, 14, EOS, 7, 7]]
    model = ScriptedGPT(cfg, script, prompt_len=2)
    idx = torch.tensor([[1, 2], [1, 2]])
    out = model.generate(idx, max_new_tokens=6, temperature=0.0, eos_id=EOS)

    assert model.calls == 4  # stopped early, not after 6 steps
    assert out.tolist() == [
        [1, 2, 11, EOS, EOS, EOS],  # finished row keeps getting EOS
        [1, 2, 12, 13, 14, EOS],
    ]


def test_generate_stops_on_eos_real_model():
    model = make_model()
    idx = torch.randint(0, 50, (1, 4))
    first = model.generate(idx, max_new_tokens=1, temperature=0.0)[0, -1].item()
    out = model.generate(idx, max_new_tokens=10, temperature=0.0, eos_id=first)
    assert out.shape == (1, 5)
    assert out[0, -1].item() == first


def test_generate_without_eos_runs_full_length():
    model = make_model()
    idx = torch.randint(0, 50, (2, 3))
    out = model.generate(idx, max_new_tokens=9, temperature=0.0, eos_id=None)
    assert out.shape == (2, 12)


# ---------------------------------------------------------------------------
# 12. overfit one small batch
# ---------------------------------------------------------------------------
def test_overfit_batch():
    torch.manual_seed(0)
    model = MiniGPT(small_cfg(vocab_size=40, dropout=0.0))
    data = torch.randint(0, 40, (4, 17))
    x, y = data[:, :-1].clone(), data[:, 1:].clone()
    y[3, 12:] = IGNORE_INDEX  # simulate right padding

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    model.train()
    for _ in range(200):
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.eval()
    with torch.no_grad():
        _, final = model(x, y)
    assert final.item() < 0.1, final.item()
