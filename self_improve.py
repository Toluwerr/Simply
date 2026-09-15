#!/usr/bin/env python3
"""Simply - Recursive Self-Improvement loop.

Every iteration Simply:
  1. loads the newest dataset and its latest weights,
  2. trains for a number of gradient steps,
  3. grades itself on a frozen validation set,
  4. writes new training examples from its own imagination, filters them
     for quality, and absorbs the survivors (self-distillation),
  5. promotes itself to a new version if it actually improved,
  6. records everything and (optionally) commits + pushes to GitHub.

Designed to run forever: locally, or on GitHub Actions on a schedule.
"""
import argparse
import json
import os
import random
import re
import subprocess
import time

import torch

from model import Simply, GPTConfig
from data import CharDataset

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

CORPUS_PATH = "corpus.txt"
METRICS_PATH = "metrics.json"
HISTORY_PATH = "history.jsonl"
LOG_PATH = "IMPROVEMENT_LOG.md"
BEST_DIR = "best"
LAST_DIR = "last"
SYN_DIR = "synthetic"

MODEL_KWARGS = dict(block_size=128, n_layer=4, n_head=4, n_embd=128)

BOOTSTRAP_STEPS = 500   # used for the very first training run
DEFAULT_STEPS = 300     # used for every later iteration
VAL_FRAC = 0.05
IMPROVE_EPSILON = 0.002  # val loss must drop by at least this much
LR_START = 3e-3
LR_MIN = 1e-4
MAX_CORPUS_CHARS = 3_500_000
MAX_SYNTH_PER_ITER = 24


def log(msg):
    print(msg, flush=True)


def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            return json.load(f)
    return {
        "version": 0,
        "iteration": -1,
        "best_val_loss": None,
        "lr": LR_START,
        "plateaus": 0,
        "val_start": None,
        "val_end": None,
        "dataset_chars": 0,
        "synthetic_total": 0,
        "model_kwargs": MODEL_KWARGS,
    }


def save_metrics(m):
    with open(METRICS_PATH, "w") as f:
        json.dump(m, f, indent=2)


def load_corpus():
    if not os.path.exists(CORPUS_PATH):
        log("corpus.txt missing - generating seed corpus ...")
        import seed_data
        seed_data.build()
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        return f.read()


def load_model(ds, device):
    """Return (model, from_scratch). Prefers last/, falls back to best/."""
    cfg = GPTConfig(vocab_size=len(ds.itos), **MODEL_KWARGS)
    model = Simply(cfg).to(device)
    for path in (os.path.join(LAST_DIR, "model.pt"),
                 os.path.join(BEST_DIR, "model.pt")):
        if os.path.exists(path):
            blob = torch.load(path, map_location=device)
            model.load_state_dict(blob["model"])
            log(f"loaded weights from {path}")
            return model, False
    log("no checkpoint found - bootstrapping from scratch")
    return model, True


def save_checkpoint(model, ds, val_loss, version, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "config": {**MODEL_KWARGS, "vocab_size": len(ds.itos)},
        "stoi": ds.stoi,
        "itos": ds.itos,
        "val_loss": val_loss,
        "version": version,
    }, path)


def train_steps(model, ds, steps, batch_size, lr, device):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    t0 = time.time()
    running = []
    for i in range(1, steps + 1):
        x, y = ds.batch(batch_size, device, split="train")
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        running.append(loss.item())
        if i % 50 == 0 or i == steps:
            recent = sum(running[-50:]) / len(running[-50:])
            log(f"  step {i:>5}/{steps}  loss {recent:.4f}  "
                f"{(time.time() - t0) / i:.2f}s/step")
    return sum(running[-50:]) / len(running[-50:])


def extract_pairs(text, allowed_chars):
    """Pull well-formed Q/A pairs out of raw model samples."""
    pairs = []
    for q, a in re.findall(r"Q: ([^\n]+)\nA: ([^\n]+)", text):
        q, a = q.strip(), a.strip()
        if not (10 <= len(q) <= 90 and 5 <= len(a) <= 220):
            continue
        if "Q:" in a or "A:" in q:
            continue
        blob = f"Q: {q}\nA: {a}"
        if not all(c in allowed_chars for c in blob):
            continue
        pairs.append(blob)
    return pairs


def too_repetitive(blob):
    for line in blob.split("\n"):
        if len(line) > 120:
            return True
        for i in range(0, max(1, len(line) - 30), 10):
            window = line[i:i + 30]
            if len(window) == 30 and line.count(window) >= 3:
                return True
    return False


def generate_and_absorb(model, ds, device, iteration):
    """Sample from itself, filter, and append the best examples."""
    model.eval()
    allowed = set(ds.stoi.keys())
    batch = 6
    prompt_ids = [ds.stoi.get(c, 0) for c in "Q: "]
    prompt = torch.tensor([prompt_ids] * batch, device=device)
    samples = []
    with torch.no_grad():
        for _ in range(5):
            out = model.generate(prompt.clone(), 220,
                                 temperature=0.95, top_k=40)
            for row in out.tolist():
                samples.append(ds.decode(row))

    candidates = []
    for text in samples:
        candidates.extend(extract_pairs(text, allowed))

    with open(CORPUS_PATH) as f:
        existing = set(re.findall(r"Q: [^\n]+\nA: [^\n]+", f.read()))

    accepted = []
    for blob in candidates:
        if blob in existing or blob in accepted:
            continue
        if too_repetitive(blob):
            continue
        accepted.append(blob)
        if len(accepted) >= MAX_SYNTH_PER_ITER:
            break

    if not accepted:
        log("  self-written data: 0 examples survived the filters")
        return 0

    os.makedirs(SYN_DIR, exist_ok=True)
    with open(os.path.join(SYN_DIR, f"iter_{iteration:03d}.txt"), "w") as f:
        f.write("\n\n".join(accepted) + "\n")

    if os.path.getsize(CORPUS_PATH) < MAX_CORPUS_CHARS:
        with open(CORPUS_PATH, "a") as f:
            f.write("\n\n" + "\n\n".join(accepted) + "\n")
    log(f"  self-written data: accepted {len(accepted)} new examples")
    return len(accepted)


def sample_answer(model, ds, device, question="Who are you?"):
    prompt = f"Q: {question}\nA:"
    ids = torch.tensor([[ds.stoi.get(c, 0) for c in prompt]], device=device)
    with torch.no_grad():
        out = model.generate(ids, 120, temperature=0.7, top_k=30)
    return ds.decode(out[0].tolist())[len(prompt):].split("\n")[0].strip()


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True)


def commit_and_push(message, push):
    git("add", "-A")
    st = git("status", "--porcelain")
    if not st.stdout.strip():
        log("nothing to commit")
        return
    c = git("commit", "-m", message)
    if c.returncode != 0:
        log(f"commit failed: {c.stderr.strip()}")
        return
    log(f"committed: {message}")
    if not push:
        return
    if not git("remote").stdout.strip():
        return
    p = git("push", "origin", "HEAD")
    if p.returncode == 0:
        log("pushed to GitHub")
    else:
        log(f"push failed: {p.stderr.strip()}")


def append_log(entry):
    header = (
        "# Simply - Self-Improvement Log\n\n"
        "This file is written automatically by `self_improve.py`.\n"
        "Every entry is one loop of: train -> evaluate -> self-write data ->\n"
        "promote-if-better. Newest entries are at the bottom.\n"
    )
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w") as f:
            f.write(header + "\n")
    with open(LOG_PATH, "a") as f:
        f.write(entry + "\n")


def main():
    ap = argparse.ArgumentParser(description="Simply RSI loop")
    ap.add_argument("--iterations", type=int, default=1)
    ap.add_argument("--steps", type=int, default=None,
                    help=f"default: {BOOTSTRAP_STEPS} if fresh, else {DEFAULT_STEPS}")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--bench", type=int, default=0,
                    help="train N steps in memory, print speed, exit")
    args = ap.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    torch.manual_seed(random.randrange(1 << 30))

    metrics = load_metrics()
    corpus = load_corpus()
    ds = CharDataset(corpus, block_size=MODEL_KWARGS["block_size"],
                     val_frac=VAL_FRAC,
                     val_start=metrics["val_start"], val_end=metrics["val_end"])
    if metrics["val_start"] is None:
        metrics["val_start"], metrics["val_end"] = ds.val_start, ds.val_end
        save_metrics(metrics)

    if args.bench:
        model, fresh = load_model(ds, device)
        t0 = time.time()
        train_steps(model, ds, args.bench, args.batch_size, metrics["lr"], device)
        dt = time.time() - t0
        log(f"BENCH: {args.bench} steps in {dt:.1f}s ({dt / args.bench:.2f}s/step)")
        return

    for _ in range(args.iterations):
        metrics["iteration"] += 1
        it_num = metrics["iteration"]
        model, from_scratch = load_model(ds, device)
        n_params = model.num_params()
        steps = args.steps if args.steps is not None else (
            BOOTSTRAP_STEPS if from_scratch else DEFAULT_STEPS)
        log(f"=== iteration {it_num} === from_scratch={from_scratch} "
            f"params={n_params / 1e6:.2f}M steps={steps} "
            f"lr={metrics['lr']:.2e} device={device}")

        train_loss = train_steps(model, ds, steps, args.batch_size,
                                 metrics["lr"], device)
        val_loss = ds.eval_loss(model, device)
        improved = (metrics["best_val_loss"] is None
                    or val_loss < metrics["best_val_loss"] - IMPROVE_EPSILON)
        synth_n = generate_and_absorb(model, ds, device, it_num)

        if improved:
            metrics["version"] += 1
            metrics["plateaus"] = 0
            status = "BOOTSTRAP" if from_scratch else "IMPROVED"
            save_checkpoint(model, ds, val_loss, metrics["version"],
                            os.path.join(BEST_DIR, "model.pt"))
        else:
            metrics["plateaus"] += 1
            status = "NO_GAIN"
            if metrics["plateaus"] >= 2:
                metrics["lr"] = max(LR_MIN, metrics["lr"] / 2)
                metrics["plateaus"] = 0
                log(f"plateau - lowering lr to {metrics['lr']:.2e}")
        save_checkpoint(model, ds, val_loss, metrics["version"],
                        os.path.join(LAST_DIR, "model.pt"))

        metrics["best_val_loss"] = (val_loss if improved
                                    else metrics["best_val_loss"])
        metrics["dataset_chars"] = os.path.getsize(CORPUS_PATH)
        metrics["synthetic_total"] += synth_n
        save_metrics(metrics)

        rec = {
            "iteration": it_num, "version": metrics["version"],
            "status": status, "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "best_val_loss": round(metrics["best_val_loss"], 4),
            "synthetic": synth_n, "lr": metrics["lr"],
            "dataset_chars": metrics["dataset_chars"],
            "params_M": round(n_params / 1e6, 3),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(HISTORY_PATH, "a") as f:
            f.write(json.dumps(rec) + "\n")

        portrait = sample_answer(model, ds, device)
        append_log(
            f"## Iteration {it_num} - v{metrics['version']} - {status}\n"
            f"- **when**: {rec['ts']}\n"
            f"- **train loss**: {train_loss:.4f} | "
            f"**val loss**: {val_loss:.4f} "
            f"(best {metrics['best_val_loss']:.4f})\n"
            f"- **corpus**: {metrics['dataset_chars']:,} chars "
            f"(+{synth_n} self-written examples)\n"
            f"- **lr**: {metrics['lr']:.4f} | **steps**: {steps}\n"
            f"- self-portrait, asked \"Who are you?\": "
            f"*\"{portrait}\"\n"
        )

        commit_and_push(
            f"Simply v{metrics['version']}: iter {it_num} {status} - "
            f"val_loss {val_loss:.4f} (best {metrics['best_val_loss']:.4f}), "
            f"+{synth_n} self-written examples",
            push=not args.no_push,
        )
        log(f"=== iteration {it_num} done: {status} ===")

    log("all done")


if __name__ == "__main__":
    main()
