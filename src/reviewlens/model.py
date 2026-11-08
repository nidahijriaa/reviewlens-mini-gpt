"""Decoder-only Mini-GPT built from scratch.

Contract: INTERFACES.md section 8 / PROJECT_CONTEXT.md section 8.

    idx (B, T)
      -> tok_emb(idx) + pos_emb(0..T-1)          # learned positional embedding
      -> dropout
      -> n_layer x TransformerBlock (pre-LayerNorm)
           x = x + attn(ln1(x))                  # manual causal MHSA (attention.py)
           x = x + ff(ln2(x))                    # Linear -> GELU -> Linear -> Dropout
      -> ln_f
      -> lm_head (Linear(d_model, vocab_size, bias=False), weight tied to tok_emb)
      -> logits (B, T, vocab_size)

This module knows nothing about tokens/data: it only consumes integer IDs and
never imports tokenizer.py or preprocess.py. Loss ignores target positions
equal to ``config.IGNORE_INDEX`` (-100); dataset.collate_fn pads ``y`` with
-100 (and ``x`` with PAD = 0), so PAD is never used as a padding target.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .attention import MultiHeadSelfAttention
from .config import IGNORE_INDEX, ModelConfig

# IGNORE_INDEX is defined in config.py (single source of truth) and re-exported
# here for convenience/backward compatibility.
__all__ = ["IGNORE_INDEX", "FeedForward", "TransformerBlock", "MiniGPT"]


class FeedForward(nn.Module):
    """Position-wise MLP: Linear(d_model, d_ff) -> GELU -> Linear(d_ff, d_model) -> Dropout."""

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class TransformerBlock(nn.Module):
    """Pre-LayerNorm decoder block: x = x + attn(ln1(x)); x = x + ff(ln2(x))."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadSelfAttention(cfg.d_model, cfg.n_head, cfg.block_size, cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    """Decoder-only Transformer language model trained with next-token prediction."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg.block_size

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)   # learned positions
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying: the LM head reuses the token-embedding matrix (V, d_model).
        self.lm_head.weight = self.tok_emb.weight

        # Random initialisation (no external weights). Re-initialising the shared
        # tensor via either module keeps it shared, so tying survives this call.
        self.apply(self._init_weights)

    # ------------------------------------------------------------------
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    def forward(
        self, idx: torch.LongTensor, targets: torch.LongTensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        """idx (B, T) with T <= block_size -> (logits (B, T, V), loss or None)."""
        if idx.dim() != 2:
            raise ValueError(f"idx must have shape (B, T), got {tuple(idx.shape)}")
        if idx.dtype != torch.long:
            raise ValueError(f"idx must be a LongTensor, got dtype {idx.dtype}")
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.block_size}")

        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))     # (B, T, d) + (T, d)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))                      # (B, T, V)

        if targets is None:
            return logits, None
        if targets.shape != idx.shape:
            raise ValueError(
                f"targets shape {tuple(targets.shape)} must match idx shape {tuple(idx.shape)}"
            )
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )
        return logits, loss

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        idx: torch.LongTensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_id: int | None = None,
    ) -> torch.LongTensor:
        """Autoregressively append up to ``max_new_tokens`` tokens to ``idx`` (B, T0).

        - ``temperature == 0`` -> greedy (argmax); otherwise sample from
          ``softmax(logits / temperature)``, optionally restricted to the top-k.
        - The context fed to the model is cropped to the last ``block_size`` tokens.
        - With ``eos_id``: a row that has produced EOS keeps receiving ``eos_id``
          (so the batch stays rectangular) and generation stops as soon as every
          row has produced EOS. Output length is ``<= T0 + max_new_tokens``.
        - Runs in eval mode (dropout off) and restores the previous mode afterwards.
        """
        if idx.dim() != 2:
            raise ValueError(f"idx must have shape (B, T), got {tuple(idx.shape)}")
        if max_new_tokens < 0:
            raise ValueError(f"max_new_tokens must be >= 0, got {max_new_tokens}")
        if temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {temperature}")
        if top_k is not None and top_k <= 0:
            raise ValueError(f"top_k must be a positive int or None, got {top_k}")

        was_training = self.training
        self.eval()
        try:
            finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.block_size:]               # safety crop
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :]                          # last position only (B, V)

                if temperature == 0:
                    next_id = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    logits = logits / temperature
                    if top_k is not None:
                        k = min(top_k, logits.size(-1))
                        kth = torch.topk(logits, k, dim=-1).values[:, [-1]]
                        logits = logits.masked_fill(logits < kth, float("-inf"))
                    probs = F.softmax(logits, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1)

                if eos_id is not None:
                    next_id = next_id.masked_fill(finished.unsqueeze(1), eos_id)
                    finished |= next_id.squeeze(1) == eos_id

                idx = torch.cat([idx, next_id], dim=1)
                if eos_id is not None and bool(finished.all()):
                    break
        finally:
            self.train(was_training)
        return idx

    # ------------------------------------------------------------------
    def num_parameters(self) -> int:
        """Number of unique parameters (the tied tok_emb / lm_head counted once)."""
        seen: set[int] = set()
        total = 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
        return total
