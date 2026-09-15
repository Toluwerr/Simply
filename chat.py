#!/usr/bin/env python3
"""Chat with Simply - the current best version of the model.

Usage:
    python chat.py
    python chat.py "What is 7 plus 5?"
"""
import os
import sys

import torch

from model import Simply, GPTConfig

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu")
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)


def main():
    path = os.path.join("best", "model.pt")
    if not os.path.exists(path):
        print("No trained model found. Run: python self_improve.py --iterations 1")
        sys.exit(1)
    blob = load_checkpoint(path)
    cfg = GPTConfig(**blob["config"])
    model = Simply(cfg)
    model.load_state_dict(blob["model"])
    model.eval()
    stoi, itos = blob["stoi"], blob["itos"]
    print(f"Simply v{blob.get('version', '?')} is ready "
          f"(val_loss {blob.get('val_loss', '?'):.4f}). "
          "Type 'quit' to exit.")
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q or q.lower() in ("quit", "exit"):
            break
        prompt = f"Q: {q}\nA:"
        ids = torch.tensor([[stoi.get(c, 0) for c in prompt]])
        with torch.no_grad():
            out = model.generate(ids, 140, temperature=0.7, top_k=30)
        text = "".join(itos[i] for i in out[0].tolist())[len(prompt):]
        print(f"simply> {text.splitlines()[0].strip()}")


if __name__ == "__main__":
    main()
