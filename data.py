"""Dataset utilities for Simply.

Two dataset classes: CharDataset (generation 1, character-level) and
TokenDataset (generation 2, byte-level BPE tokens - see tokenizer.py).
Both expose the same interface so the self-improvement loop can use
either: encode/decode/batch/eval_loss, plus a frozen validation slice
(indices stored in metrics.json on first use) so validation loss stays
comparable across every iteration of a generation.
"""
import torch


class CharDataset:
    def __init__(self, text, block_size=128, val_frac=0.05,
                 val_start=None, val_end=None):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = chars
        self.block_size = block_size
        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        if val_start is None or val_end is None:
            val_end = len(data)
            val_start = int(len(data) * (1 - val_frac))
        self.val_start, self.val_end = val_start, val_end
        # Training data = everything except the frozen validation slice.
        self.train_data = torch.cat([data[:val_start], data[val_end:]])
        self.val_data = data[val_start:val_end]

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    def batch(self, bs, device, split="train"):
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - self.block_size - 1, (bs,))
        x = torch.stack([data[i:i + self.block_size] for i in ix]).to(device)
        y = torch.stack([data[i + 1:i + 1 + self.block_size] for i in ix]).to(device)
        return x, y

    @torch.no_grad()
    def eval_loss(self, model, device, n_batches=10, bs=8, seed=1234):
        """Deterministic validation loss (fixed seed -> same batches)."""
        g = torch.Generator().manual_seed(seed)
        model.eval()
        total = 0.0
        for _ in range(n_batches):
            ix = torch.randint(
                len(self.val_data) - self.block_size - 1, (bs,), generator=g
            )
            x = torch.stack([self.val_data[i:i + self.block_size] for i in ix]).to(device)
            y = torch.stack([self.val_data[i + 1:i + 1 + self.block_size] for i in ix]).to(device)
            _, loss, _ = model(x, y)
            total += loss.item()
        return total / n_batches


class TokenDataset:
    """Generation 2: same contract as CharDataset, but over BPE tokens.

    encode() never fails on any input - byte-level means zero unknown
    tokens - and decode() is exact, so answers the model gives can be
    sliced and verified character-for-character exactly like before.
    """

    def __init__(self, text, tok, block_size=192, val_frac=0.05,
                 val_start=None, val_end=None):
        self.tok = tok
        self.vocab_size = tok.vocab_size
        self.stoi = tok.stoi
        self.itos = tok.itos
        self.block_size = block_size
        data = torch.tensor(tok.encode(text), dtype=torch.long)
        if val_start is None or val_end is None:
            val_end = len(data)
            val_start = int(len(data) * (1 - val_frac))
        self.val_start, self.val_end = val_start, val_end
        self.train_data = torch.cat([data[:val_start], data[val_end:]])
        self.val_data = data[val_start:val_end]

    def encode(self, s):
        return self.tok.encode(s)

    def decode(self, ids):
        return self.tok.decode(ids)

    def batch(self, bs, device, split="train"):
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - self.block_size - 1, (bs,))
        x = torch.stack([data[i:i + self.block_size] for i in ix]).to(device)
        y = torch.stack([data[i + 1:i + 1 + self.block_size] for i in ix]).to(device)
        return x, y

    @torch.no_grad()
    def eval_loss(self, model, device, n_batches=10, bs=8, seed=1234):
        """Deterministic validation loss (fixed seed -> same batches).

        Sized for the gen-4 brain: fewer, wider batches keep the same
        wall-clock cost per iteration as gen 3 despite 3.7x parameters."""
        g = torch.Generator().manual_seed(seed)
        model.eval()
        total = 0.0
        for _ in range(n_batches):
            ix = torch.randint(
                len(self.val_data) - self.block_size - 1, (bs,), generator=g
            )
            x = torch.stack([self.val_data[i:i + self.block_size] for i in ix]).to(device)
            y = torch.stack([self.val_data[i + 1:i + 1 + self.block_size] for i in ix]).to(device)
            _, loss, _ = model(x, y)
            total += loss.item()
        return total / n_batches
