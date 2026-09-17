"""Simply - a small decoder-only transformer that improves itself.

Generation 4: a decoder-only transformer over byte-level BPE tokens
(see tokenizer.py). Defaults (MODEL_KWARGS in self_improve.py): 6
layers, 6 heads, 384 embedding dims, 512-token context, 2048-token
vocabulary, tied embeddings - about 11.6 million parameters, roughly
3.7x the gen-3 brain, with twice the context (one window now holds
roughly two pages of text). Same architecture family as the big LLMs
(tokenized input, causal self-attention, MLP blocks, tied output
head), scaled so it can still train on a CPU runner inside a
self-improvement loop.

Generation comes with a KV-cache for generation: prompt tokens are
prefetched once, and every sampled token afterwards only runs one
position through the model instead of the whole context. That makes
quizzes, self-writing, and the diary several times faster, which is
what pays for the bigger brain inside the same runner budget.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GPTConfig:
    def __init__(self, vocab_size, block_size=128, n_layer=4, n_head=4,
                 n_embd=128, tie_weights=True):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        # Tie the output head to the token embedding (GPT-2 style): fewer
        # parameters, and small models train better with it.
        self.tie_weights = tie_weights


class CausalSelfAttention(nn.Module):
    """Multi-head masked self-attention with an optional KV-cache."""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer(
            "mask", mask.view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x, past=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if past is not None:
            pk, pv = past
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        cached = (k, v)
        kt = k.size(2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        if past is None:
            # Full prefill: classic triangular mask.
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0,
                                  float("-inf"))
        elif T > 1:
            # Cached multi-token step: queries sit at positions kt-T..kt-1.
            offset = kt - T
            m = torch.tril(torch.ones(T, kt, device=x.device),
                           diagonal=offset)
            att = att.masked_fill(m.view(1, 1, T, kt) == 0,
                                  float("-inf"))
        # else: single cached token attends to everything so far - no mask.
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y), cached


class Block(nn.Module):
    """Transformer block: attention + MLP, both with residuals."""

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
        )

    def forward(self, x, past=None):
        a, cached = self.attn(self.ln1(x), past)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, cached


class Simply(nn.Module):
    """The Simply model: a minimal decoder-only transformer."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_weights:
            self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, past_kv=None):
        B, T = idx.shape
        past_len = past_kv[0][0].size(2) if past_kv else 0
        pos = torch.arange(past_len, past_len + T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        new_kv = []
        for i, block in enumerate(self.blocks):
            x, cached = block(x, past_kv[i] if past_kv else None)
            new_kv.append(cached)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss, new_kv

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        """Autoregressive sampling. idx: (B, T) tensor of token ids.

        Uses a KV-cache: the prompt runs through the model once, then
        each new token is a single-position forward. Falls back to the
        old recompute-everything path when the request would overflow
        the context window."""
        if idx.size(1) + max_new_tokens > self.config.block_size:
            return self._generate_windowed(
                idx, max_new_tokens, temperature, top_k)
        out = idx
        cur, past = idx, None
        for _ in range(max_new_tokens):
            logits, _, past = self(cur, past_kv=past)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            cur = torch.multinomial(probs, num_samples=1)
            out = torch.cat([out, cur], dim=1)
        return out

    @torch.no_grad()
    def _generate_windowed(self, idx, max_new_tokens, temperature=0.8,
                           top_k=40):
        """Pre-cache generation path: crops to the last block_size tokens
        every step, like gen 1-3 did."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
