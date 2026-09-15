"""Character-level dataset utilities for Simply.

The validation slice is frozen on first use (indices stored in
metrics.json) so that validation loss stays comparable across every
self-improvement iteration, even as the corpus grows.
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
    def eval_loss(self, model, device, n_batches=20, bs=16, seed=1234):
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
            _, loss = model(x, y)
            total += loss.item()
        return total / n_batches
