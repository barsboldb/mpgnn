"""Decoder-only transformer for autoregressive chain-of-thought (cot_mode:
autoregressive).

The model is a plain causal LM over the discrete graph/trace vocabulary built by
cot_tokens.py: it reads the serialized graph prompt, is teacher-forced on the
BFS trace + answer during training, and generates greedily at eval. Reuses the
pre-norm _EncoderBlock from transformer.py — a causal attention mask is the only
difference from the encoder stack. Depth and heads come from config.layers, so
the depth-vs-trace-length grid uses the same config axis as every other model.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import GNNConfig
from .cot_tokens import CoTVocab
from .transformer import _EncoderBlock


class CoTTransformer(nn.Module):
    def __init__(self, config: GNNConfig):
        super().__init__()
        self.config = config
        self.vocab = CoTVocab(config.max_nodes)
        d = config.hidden_channels

        self.tok_emb = nn.Embedding(self.vocab.size, d)
        # 'none' (NoPE) leaves ordering to the causal mask alone — worse in
        # distribution, but doesn't hard-code trained sequence lengths (length OOD)
        self.pos_emb = (nn.Embedding(config.max_seq_len, d)
                        if config.cot_pos == "learned" else None)

        default_heads = config.layers[0].get("heads", 4) if config.layers else 4
        self.blocks = nn.ModuleList([
            _EncoderBlock(d, lc.get("heads", default_heads), config.dropout)
            for lc in config.layers
        ])
        self.norm = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, self.vocab.size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight   # weight tying
        # small-std init (matches the 0.02 used for the scratchpad/task tokens);
        # the default N(0,1) embedding init blows up the tied logits (~sqrt(d))
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        if not config.tie_embeddings:
            nn.init.normal_(self.lm_head.weight, std=0.02)
        if self.pos_emb is not None:
            nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens [B, L] -> next-token logits [B, L, vocab].

        Padding needs no key_padding_mask: sequences are right-padded, so under a
        causal mask every real query position only ever sees real (earlier) keys.
        Logits at padded positions are garbage and must be ignored by the caller
        (the loss does via ignore_index; generate() reads its own cursor).
        """
        B, L = tokens.shape
        h = self.tok_emb(tokens)
        if self.pos_emb is not None:
            assert L <= self.pos_emb.num_embeddings, \
                f"sequence length {L} exceeds max_seq_len={self.pos_emb.num_embeddings}"
            h = h + self.pos_emb(torch.arange(L, device=tokens.device))

        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=tokens.device),
                            diagonal=1).expand(B, L, L)
        no_pad = torch.zeros(B, L, dtype=torch.bool, device=tokens.device)
        for blk in self.blocks:
            h = blk(h, no_pad, causal)
        return self.lm_head(self.norm(h))

    @torch.no_grad()
    def generate(self, tokens: torch.Tensor, prompt_len: torch.Tensor,
                 max_new: int) -> torch.Tensor:
        """Greedy decode. tokens [B, L] right-padded; row i's prompt is
        tokens[i, :prompt_len[i]]. Returns [B, Lp_max + max_new] where each row is
        its prompt followed by generated tokens (PAD after EOS). No KV cache —
        sequences are short (<= max_seq_len); each step is one full forward.
        """
        self.eval()
        B = tokens.size(0)
        device = tokens.device
        pad, eos = self.vocab.PAD, self.vocab.EOS
        Lp = int(prompt_len.max().item())
        total = min(Lp + max_new, self.config.max_seq_len)

        buf = torch.full((B, total), pad, dtype=torch.long, device=device)
        for i in range(B):
            n = int(prompt_len[i].item())
            buf[i, :n] = tokens[i, :n]
        cursor = prompt_len.clone().to(device)          # next position to write
        done = torch.zeros(B, dtype=torch.bool, device=device)

        while not bool(done.all()) and int(cursor.max().item()) < total:
            logits = self(buf)                          # [B, L, V]
            step = logits[torch.arange(B, device=device), cursor - 1].argmax(dim=-1)
            active = ~done & (cursor < total)
            buf[active, cursor[active]] = step[active]
            done |= active & (step == eos)
            cursor[active] += 1
        return buf

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
