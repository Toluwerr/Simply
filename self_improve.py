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
import seed_data as SD  # ground-truth tables used by the verifier

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


ARITH_Q = re.compile(r"^(?:What is|How much is) (\d+) (plus|minus|times) (\d+)\?$")
LETTERS_Q = re.compile(r"^How many letters (?:are in the word|does the word) '([a-z]+)'(?: have)?\?$")
SPELL_Q = re.compile(r"^How do you spell '([a-z]+)'\?$")
TRIPLE = re.compile(r"([a-zA-Z])\1{2,}")


def verified(q, a):
    """Strict quality gate for self-written examples.

    Policy: Simply only absorbs examples it can PROVE correct. A sample
    must match a known canonical template (arithmetic, letters, spelling,
    capitals, animal sounds, opposites, meanings) and its answer must
    check out against ground truth. Anything unverifiable is rejected,
    so the model can never teach itself nonsense.
    """
    if TRIPLE.search(q) or TRIPLE.search(a):
        return False
    if "  " in q or "  " in a:
        return False
    if len(q.split()) < 3 or len(a.split()) < 2:
        return False
    if not q.endswith("?") or not re.match(r"^[A-Z]", q):
        return False
    if not a.rstrip().endswith((".", "!", "?")):
        return False
    if re.search(r"\d[a-zA-Z]|[a-zA-Z]\d", q):
        return False

    m = ARITH_Q.match(q)
    if m:
        x, op, y = int(m.group(1)), m.group(2), int(m.group(3))
        result = (x + y if op == "plus"
                  else x - y if op == "minus" else x * y)
        # Require the full canonical equation, otherwise "1 times 1 is 8"
        # would pass on a word-boundary match against the operand "1".
        return f"{x} {op} {y} is {result}" in a
    m = LETTERS_Q.match(q)
    if m:
        return re.search(rf"\b{len(m.group(1))}\b", a) is not None
    m = SPELL_Q.match(q)
    if m:
        return "-".join(m.group(1).upper()) in a
    m = re.match(r"^What is the capital of (.+)\?$", q)
    if m:
        cap = SD.CAPITALS.get(m.group(1))
        return cap is not None and cap in a
    m = re.match(r"^What sound does an? (.+) make\?$", q)
    if m:
        sound = SD.ANIMAL_SOUNDS.get(m.group(1))
        return sound is not None and sound in a
    m = re.match(r"^What is the opposite of '(.+)'\?$", q)
    if m:
        opp = dict(SD.OPPOSITES).get(m.group(1))
        return opp is not None and f"'{opp}'" in a
    m = re.match(r"^What does '(.+)' mean\?$", q)
    if m:
        meaning = SD.DEFINITIONS.get(m.group(1))
        return meaning is not None and meaning in a
    # No verifiable template matched -> do not absorb.
    return False


def extract_pairs(text, allowed_chars):
    """Pull well-formed, verified Q/A pairs out of raw model samples."""
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
        if not verified(q, a):
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


def generate_and_absorb(model, ds, device, iteration, batches=5):
    """Sample from itself, filter, and append the best examples."""
    model.eval()
    allowed = set(ds.stoi.keys())
    batch = 6
    prompt_ids = [ds.stoi.get(c, 0) for c in "Q: "]
    prompt = torch.tensor([prompt_ids] * batch, device=device)
    samples = []
    with torch.no_grad():
        for _ in range(batches):
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
        log("  self-written data: 0 verified examples this iteration")
        return 0

    os.makedirs(SYN_DIR, exist_ok=True)
    with open(os.path.join(SYN_DIR, f"iter_{iteration:03d}.txt"), "w") as f:
        f.write("\n\n".join(accepted) + "\n")

    if os.path.getsize(CORPUS_PATH) < MAX_CORPUS_CHARS:
        with open(CORPUS_PATH, "a") as f:
            f.write("\n\n" + "\n\n".join(accepted) + "\n")
    log(f"  self-written data: accepted {len(accepted)} verified new examples")
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
    # Runner-safe: ensure a git identity exists (GitHub Actions
    # checkouts have none during the script step).
    git("config", "user.name", "Simply RSI Bot")
    git("config", "user.email", "simply-bot@users.noreply.github.com")
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
    ap.add_argument("--synth-batches", type=int, default=5,
                    help="sampling batches per iteration for self-written data")
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

    model, from_scratch = load_model(ds, device)
    n_params = model.num_params()
    # baseline = val loss of the COMMITTED best weights; every iteration
    # is one commit (log/metrics/corpus), weights are promoted once per
    # run so the repo does not balloon with multi-MB checkpoints.
    baseline = metrics["best_val_loss"]
    run_best_loss = baseline
    run_best_state = None

    for _ in range(args.iterations):
        metrics["iteration"] += 1
        it_num = metrics["iteration"]
        steps = args.steps if args.steps is not None else (
            BOOTSTRAP_STEPS if from_scratch else DEFAULT_STEPS)
        log(f"=== iteration {it_num} === "
            f"params={n_params / 1e6:.2f}M steps={steps} "
            f"lr={metrics['lr']:.2e} device={device}")

        train_loss = train_steps(model, ds, steps, args.batch_size,
                                 metrics["lr"], device)
        val_loss = ds.eval_loss(model, device)
        improved = (run_best_loss is None
                    or val_loss < run_best_loss - IMPROVE_EPSILON)
        if run_best_loss is None or val_loss < run_best_loss:
            run_best_loss = val_loss
            run_best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
        status = "IMPROVED" if improved else "NO_GAIN"
        synth_n = generate_and_absorb(model, ds, device, it_num,
                                      batches=args.synth_batches)

        metrics["plateaus"] = 0 if improved else metrics["plateaus"] + 1
        if metrics["plateaus"] >= 2:
            metrics["lr"] = max(LR_MIN, metrics["lr"] / 2)
            metrics["plateaus"] = 0
            log(f"plateau - lowering lr to {metrics['lr']:.2e}")

        os.makedirs(LAST_DIR, exist_ok=True)
        save_checkpoint(model, ds, val_loss, metrics["version"],
                        os.path.join(LAST_DIR, "model.pt"))

        metrics["dataset_chars"] = os.path.getsize(CORPUS_PATH)
        metrics["synthetic_total"] += synth_n
        save_metrics(metrics)

        rec = {
            "iteration": it_num, "version": metrics["version"],
            "status": status, "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "run_best": round(run_best_loss, 4),
            "synthetic": synth_n, "lr": metrics["lr"],
            "dataset_chars": metrics["dataset_chars"],
            "params_M": round(n_params / 1e6, 3),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(HISTORY_PATH, "a") as f:
            f.write(json.dumps(rec) + "\n")

        portrait = sample_answer(model, ds, device)
        append_log(
            f"## Iteration {it_num} - {status}\n"
            f"- **when**: {rec['ts']}\n"
            f"- **train loss**: {train_loss:.4f} | "
            f"**val loss**: {val_loss:.4f} | "
            f"**run best**: {run_best_loss:.4f}\n"
            f"- **corpus**: {metrics['dataset_chars']:,} chars "
            f"(+{synth_n} self-written examples)\n"
            f"- **lr**: {metrics['lr']:.4f} | **steps**: {steps}\n"
            f"- self-portrait, asked \"Who are you?\": "
            f"*\"{portrait}\"\n"
        )

        commit_and_push(
            f"Simply iter {it_num} {status} - "
            f"val_loss {val_loss:.4f}, +{synth_n} self-written examples",
            push=not args.no_push,
        )
        log(f"=== iteration {it_num} done: {status} ===")

    # End-of-run promotion: commit weights once if this run actually
    # beat the committed best. Version bumps only here.
    if run_best_state is not None and (baseline is None or
                                       run_best_loss < baseline - IMPROVE_EPSILON):
        model.load_state_dict(run_best_state)
        metrics["version"] += 1
        metrics["best_val_loss"] = run_best_loss
        metrics["plateaus"] = 0
        save_checkpoint(model, ds, run_best_loss, metrics["version"],
                        os.path.join(BEST_DIR, "model.pt"))
        save_metrics(metrics)
        was = f"{baseline:.4f}" if baseline is not None else "n/a"
        portrait = sample_answer(model, ds, device)
        append_log(
            f"## Promotion - v{metrics['version']} - weights committed\n"
            f"- **best val loss**: {run_best_loss:.4f} (was {was})\n"
            f"- self-portrait, asked \"Who are you?\": *\"{portrait}\"\n"
        )
        commit_and_push(
            f"Simply v{metrics['version']}: weights promoted "
            f"(val_loss {run_best_loss:.4f}, was {was})",
            push=not args.no_push,
        )
        log(f"=== promoted weights: v{metrics['version']} "
            f"val {run_best_loss:.4f} ===")

    save_metrics(metrics)
    log("all done")


if __name__ == "__main__":
    main()
