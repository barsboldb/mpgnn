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

    def _kv(self, blk, hn: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project normed hidden states to per-head K, V with the block's own
        MultiheadAttention weights (in_proj rows d:2d = K, 2d:3d = V)."""
        d = self.config.hidden_channels
        W, bias = blk.attn.in_proj_weight, blk.attn.in_proj_bias
        k = hn @ W[d:2 * d].T + bias[d:2 * d]
        v = hn @ W[2 * d:].T + bias[2 * d:]
        B, L, _ = hn.shape
        hd = d // blk.heads
        return (k.view(B, L, blk.heads, hd).transpose(1, 2),
                v.view(B, L, blk.heads, hd).transpose(1, 2))   # [B, H, L, hd]

    def _decode_step(self, buf: torch.Tensor, pos: torch.Tensor,
                     cacheK: list[torch.Tensor], cacheV: list[torch.Tensor]) -> torch.Tensor:
        """One cached decode step: run ONLY the token at buf[i, pos[i]] through
        the stack, attending against the K/V caches (its own k/v is written to
        slot pos[i] first, so it sees itself like the causal mask allows).
        Returns next-token logits [B, vocab]. Dropout is skipped — the module
        is in eval mode where it is the identity."""
        B, total = buf.shape
        d = self.config.hidden_channels
        device = buf.device
        rows = torch.arange(B, device=device)

        h = self.tok_emb(buf[rows, pos])
        if self.pos_emb is not None:
            h = h + self.pos_emb(pos)
        h = h.unsqueeze(1)                                       # [B, 1, d]
        # row i may attend to key slots 0..pos[i]; later slots are stale/unwritten
        key_ok = torch.arange(total, device=device)[None] <= pos[:, None]

        for li, blk in enumerate(self.blocks):
            hn = blk.norm1(h)
            hd = d // blk.heads
            W, bias = blk.attn.in_proj_weight, blk.attn.in_proj_bias
            q = (hn @ W[:d].T + bias[:d]).view(B, 1, blk.heads, hd).transpose(1, 2)
            k, v = self._kv(blk, hn)
            cacheK[li][rows, :, pos] = k[:, :, 0]
            cacheV[li][rows, :, pos] = v[:, :, 0]
            att = (q @ cacheK[li].transpose(-1, -2)) / (hd ** 0.5)   # [B, H, 1, total]
            att = att.masked_fill(~key_ok[:, None, None, :], float("-inf")).softmax(dim=-1)
            a = (att @ cacheV[li]).transpose(1, 2).reshape(B, 1, d)
            h = h + blk.attn.out_proj(a)
            h = h + blk.ffn(blk.norm2(h))
        return self.lm_head(self.norm(h))[:, 0]                  # [B, vocab]

    @torch.no_grad()
    def generate(self, tokens: torch.Tensor, prompt_len: torch.Tensor,
                 max_new: int) -> torch.Tensor:
        """Greedy decode with a per-layer KV cache. tokens [B, L] right-padded;
        row i's prompt is tokens[i, :prompt_len[i]]. Returns [B, Lp_max + max_new]
        where each row is its prompt followed by generated tokens (PAD after EOS).

        Inference-only optimization — identical greedy outputs to the full
        re-forward decode it replaced (weights are frozen at eval and attention
        is causal, so past tokens' K/V never change): one causal forward over
        the prompt region fills the caches, then each step embeds only the
        newest token per row and attends against the cache. Rows advance at
        their own cursor; cache slots past a row's cursor are masked until the
        row writes them."""
        self.eval()
        B = tokens.size(0)
        device = tokens.device
        pad, eos = self.vocab.PAD, self.vocab.EOS
        Lp = int(prompt_len.max().item())
        total = min(Lp + max_new, self.config.max_seq_len)
        d = self.config.hidden_channels

        buf = torch.full((B, total), pad, dtype=torch.long, device=device)
        for i in range(B):
            n = int(prompt_len[i].item())
            buf[i, :n] = tokens[i, :n]
        cursor = prompt_len.clone().to(device)          # next position to write
        done = torch.zeros(B, dtype=torch.bool, device=device)
        rows = torch.arange(B, device=device)

        cacheK = [torch.zeros(B, blk.heads, total, d // blk.heads, device=device)
                  for blk in self.blocks]
        cacheV = [torch.zeros_like(c) for c in cacheK]

        # prefill: one causal forward over the prompts; rows shorter than Lp get
        # stale PAD-derived k/v in slots >= their prompt_len — masked in decode
        # and overwritten as the row's cursor reaches each slot.
        h = self.tok_emb(buf[:, :Lp])
        if self.pos_emb is not None:
            h = h + self.pos_emb(torch.arange(Lp, device=device))
        causal = torch.triu(torch.ones(Lp, Lp, dtype=torch.bool, device=device),
                            diagonal=1).expand(B, Lp, Lp)
        no_pad = torch.zeros(B, Lp, dtype=torch.bool, device=device)
        for li, blk in enumerate(self.blocks):
            k, v = self._kv(blk, blk.norm1(h))
            cacheK[li][:, :, :Lp] = k
            cacheV[li][:, :, :Lp] = v
            h = blk(h, no_pad, causal)
        step_logits = self.lm_head(self.norm(h))[rows, cursor - 1]   # [B, vocab]

        while not bool(done.all()) and int(cursor.max().item()) < total:
            step = step_logits.argmax(dim=-1)
            active = ~done & (cursor < total)
            buf[active, cursor[active]] = step[active]
            done |= active & (step == eos)
            cursor[active] += 1
            if bool(done.all()) or int(cursor.max().item()) >= total:
                break
            step_logits = self._decode_step(buf, cursor - 1, cacheK, cacheV)
        return buf

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
