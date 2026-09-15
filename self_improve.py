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
KNOW_EPSILON = 0.02      # knowledge-quiz accuracy must rise by this much
LR_START = 3e-3
LR_MIN = 1e-4
MAX_CORPUS_CHARS = 8_000_000
MAX_SYNTH_PER_ITER = 40


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
    """Return (model, from_scratch). Prefers best/, falls back to last/.

    Every run retrains the CHAMPION weights. Training from last/ let a
    degraded state (overfit, LR-floored) chain forward forever; starting
    from best/ is self-correcting - a bad run is simply not promoted and
    the next run starts clean from the champion again.
    """
    cfg = GPTConfig(vocab_size=len(ds.itos), **MODEL_KWARGS)
    model = Simply(cfg).to(device)
    for path in (os.path.join(BEST_DIR, "model.pt"),
                 os.path.join(LAST_DIR, "model.pt")):
        if os.path.exists(path):
            blob = torch.load(path, map_location=device)
            if blob.get("config", {}).get("vocab_size") not in (None, len(ds.itos)):
                log(f"vocab drift at {path} - cannot load safely, skipping")
                continue
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

# --- self-expanding knowledge: verifiable question templates -------------

IS_BIGGER_Q = re.compile(r"^Is (\d+) bigger than (\d+)\?$")
BIGGER_Q = re.compile(r"^Which is bigger, (\d+) or (\d+)\?$")
SORT_Q = re.compile(r"^How do you sort (\d+), (\d+), and (\d+)\?$")
DOUBLE_Q = re.compile(r"^What is double (\d+)\?$")
HALF_Q = re.compile(r"^What is half of (\d+)\?$")
COUNT_Q = re.compile(r"^Can you count by (ones|twos|fives|tens) from (\d+) to (\d+)\?$")
NUMWORDS_Q = re.compile(r"^How do you write (\d+) in words\?$")
FIRST_LETTER_Q = re.compile(r"^What is the first letter of '([A-Za-z]+)'\?$")
LAST_LETTER_Q = re.compile(r"^What is the last letter of '([A-Za-z]+)'\?$")

COUNT_WORDS = {"ones": 1, "twos": 2, "fives": 5, "tens": 10}

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven",
         "eight", "nine", "ten", "eleven", "twelve", "thirteen",
         "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
         "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty",
         "sixty", "seventy", "eighty", "ninety"]


def num2words(n):
    """Deterministic English spelling of 0-100 (the number GT table)."""
    if not 0 <= n <= 100:
        raise ValueError(n)
    if n < 20:
        return _ONES[n]
    if n == 100:
        return "one hundred"
    t, r = divmod(n, 10)
    return _TENS[t] + ("-" + _ONES[r] if r else "")


def _single_words():
    """Every provable single word in the ground-truth tables."""
    ws = set()
    for table in (SD.ANIMAL_SOUNDS, SD.DEFINITIONS):
        for w in table:
            if w.isalpha():
                ws.add(w.lower())
    for a, b in SD.OPPOSITES:
        for w in (a, b):
            if w.isalpha():
                ws.add(w.lower())
    for k, v in SD.CAPITALS.items():
        for w in (k, v):
            if w.isalpha():
                ws.add(w.lower())
    return ws


KNOWN_WORDS = _single_words()

# Curriculum: prompt styles used to prime self-question-generation per
# category. The loop feeds the model its own weakest categories first.
CATEGORY_PROMPTS = {
    "arithmetic": ["Q: What is ", "Q: How much is "],
    "compare": ["Q: Which is bigger, ", "Q: Is "],
    "sort": ["Q: How do you sort "],
    "double_half": ["Q: What is double ", "Q: What is half of "],
    "count": ["Q: Can you count by "],
    "numwords": ["Q: How do you write "],
    "letters": ["Q: How many letters does the word '",
                "Q: How do you spell '"],
    "first_last": ["Q: What is the first letter of '",
                   "Q: What is the last letter of '"],
    "capitals": ["Q: What is the capital of "],
    "animals": ["Q: What sound does a "],
    "opposites": ["Q: What is the opposite of '"],
    "definitions": ["Q: What does '"],
}

# Fixed knowledge probe: one question per category, graded with the same
# verifier used for self-written data. Accuracy per category is committed
# to metrics.json every iteration -> measurable knowledge growth.
QUIZ_PROBES = [
    ("arithmetic", "What is 13 plus 48?"),
    ("compare", "Which is bigger, 38 or 71?"),
    ("sort", "How do you sort 14, 3, and 27?"),
    ("double_half", "What is half of 34?"),
    ("count", "Can you count by fives from 20 to 45?"),
    ("numwords", "How do you write 47 in words?"),
    ("letters", "How many letters does the word 'elephant' have?"),
    ("first_last", "What is the first letter of 'Brasilia'?"),
    ("capitals", "What is the capital of Norway?"),
    ("animals", "What sound does an owl make?"),
    ("opposites", "What is the opposite of 'full'?"),
    ("definitions", "What does 'sturdy' mean?"),
]


def template_check(q, a):
    """Per-template ground-truth check. Returns True only when the
    question matches a known canonical template AND its answer is
    provably correct. Single source of truth for the verifier and the
    knowledge quiz."""
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
    m = IS_BIGGER_Q.match(q)
    if m:
        x, y = int(m.group(1)), int(m.group(2))
        if x == y:
            return False
        yes = re.search(r"\byes\b", a.lower()) is not None
        no = re.search(r"\bno\b", a.lower()) is not None
        if x > y:
            return yes and not no and f"{x} is bigger" in a
        return no and not yes and f"{y} is bigger" in a
    m = BIGGER_Q.match(q)
    if m:
        x, y = int(m.group(1)), int(m.group(2))
        return x != y and f"{max(x, y)} is bigger" in a
    m = SORT_Q.match(q)
    if m:
        xs = sorted(int(v) for v in m.groups())
        if len(set(xs)) != 3:
            return False
        return " ".join(str(v) for v in xs) in a
    m = DOUBLE_Q.match(q)
    if m:
        x = int(m.group(1))
        return f"double {x} is {2 * x}" in a.lower()
    m = HALF_Q.match(q)
    if m:
        x = int(m.group(1))
        return x % 2 == 0 and f"half of {x} is {x // 2}" in a.lower()
    m = COUNT_Q.match(q)
    if m:
        step = COUNT_WORDS[m.group(1)]
        lo, hi = int(m.group(2)), int(m.group(3))
        if lo < step or lo % step or hi < lo:
            return False
        terms = list(range(lo, hi + 1, step))
        if not 3 <= len(terms) <= 8:
            return False
        return " ".join(str(v) for v in terms) in a
    m = NUMWORDS_Q.match(q)
    if m:
        n = int(m.group(1))
        if not 1 <= n <= 100:
            return False
        w = num2words(n)
        return (f"{n} in words is {w}" in a.lower()
                or f"{n} in words is {w.replace('-', ' ')}" in a.lower())
    m = FIRST_LETTER_Q.match(q)
    if m:
        w = m.group(1)
        if w.lower() not in KNOWN_WORDS:
            return False
        return re.search(rf"is '?{w[0].upper()}'?[^a-zA-Z]", a) is not None
    m = LAST_LETTER_Q.match(q)
    if m:
        w = m.group(1)
        if w.lower() not in KNOWN_WORDS:
            return False
        return re.search(rf"is '?{w[-1].upper()}'?[^a-zA-Z]", a) is not None
    # No verifiable template matched -> do not absorb.
    return False


def verified(q, a):
    """Strict quality gate for self-written examples.

    Policy: Simply only absorbs examples it can PROVE correct. A sample
    must match a known canonical template (arithmetic, letters, spelling,
    capitals, animal sounds, opposites, meanings, comparisons, sorting,
    doubles/halves, counting, number words, first/last letters) and its
    answer must check out against ground truth. Anything unverifiable is
    rejected, so the model can never teach itself nonsense.
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
    return template_check(q, a.rstrip())


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


def weakest_categories(knowledge):
    """Rank knowledge categories weakest-first (the self-study plan)."""
    def acc(c):
        v = knowledge.get(c) if isinstance(knowledge, dict) else None
        return v if isinstance(v, (int, float)) else 0.0
    cats = list(CATEGORY_PROMPTS)
    cats.sort(key=lambda c: (acc(c), random.random()))
    return cats


def run_knowledge_quiz(model, ds, device):
    """Grade one probe per knowledge category with the real verifier.

    Returns (accuracies, summary_line). Accuracies go into
    metrics['knowledge'] every iteration, so knowledge growth is
    committed, versioned and auditable like everything else.
    """
    model.eval()
    scores, seen = {}, {}
    for cat, q in QUIZ_PROBES:
        # Greedy decoding: the quiz measures knowledge, not creativity.
        ans = sample_answer(model, ds, device, question=q,
                            temperature=0.1, max_new=48, top_k=1)
        ok = template_check(q, ans)
        scores[cat] = scores.get(cat, 0) + int(ok)
        seen[cat] = seen.get(cat, 0) + 1
    acc = {c: round(scores[c] / seen[c], 3) for c in scores}
    ranked = sorted(acc.items(), key=lambda kv: kv[1])
    avg = sum(acc.values()) / len(acc)
    worst = ", ".join(f"{c} {v:.2f}" for c, v in ranked[:3])
    line = f"avg {avg:.2f} | weakest: {worst}"
    return acc, line


def knowledge_seed_examples():
    """One-time curriculum seed for the NEW self-teachable categories.

    Gives Simply the answer patterns to imitate; from then on it invents
    its own questions in these categories and only absorbs the ones it
    can prove correct. Every example is asserted to pass the strict
    verifier, so the seed itself can never smuggle in nonsense.
    """
    ex = []

    def add(q, a):
        assert verified(q, a), f"seed failed verification: {q!r} -> {a!r}"
        ex.append(f"Q: {q}\nA: {a}")

    for x, y in [(71, 38), (52, 9), (104, 220), (17, 92), (63, 41),
                 (85, 130)]:
        add(f"Which is bigger, {x} or {y}?",
            f"{max(x, y)} is bigger than {min(x, y)}.")
    for x, y in [(26, 19), (44, 57), (8, 3), (90, 99)]:
        if x > y:
            add(f"Is {x} bigger than {y}?", f"Yes, {x} is bigger than {y}.")
        else:
            add(f"Is {x} bigger than {y}?", f"No, {y} is bigger than {x}.")
    for a, b, c in [(14, 3, 27), (42, 7, 19), (60, 8, 31), (5, 88, 23)]:
        s = " ".join(str(v) for v in sorted([a, b, c]))
        add(f"How do you sort {a}, {b}, and {c}?",
            f"In order, they are {s}.")
    for x in [12, 25, 31, 46]:
        add(f"What is double {x}?", f"Double {x} is {2 * x}.")
    for x in [18, 26, 40, 54]:
        add(f"What is half of {x}?", f"Half of {x} is {x // 2}.")
    for word, lo, hi in [("twos", 2, 12), ("fives", 15, 40),
                         ("tens", 20, 70), ("ones", 6, 10)]:
        step = COUNT_WORDS[word]
        seq = " ".join(str(v) for v in range(lo, hi + 1, step))
        add(f"Can you count by {word} from {lo} to {hi}?",
            f"Counting by {word} gives {seq}.")
    for n in [7, 15, 21, 38, 42, 64]:
        add(f"How do you write {n} in words?",
            f"{n} in words is {num2words(n)}.")
    for w in ["paris", "frog", "eager", "oslo"]:
        add(f"What is the first letter of '{w}'?",
            f"The first letter of '{w}' is {w[0].upper()}.")
    for w in ["frog", "calm", "tokyo", "tiny"]:
        add(f"What is the last letter of '{w}'?",
            f"The last letter of '{w}' is {w[-1].upper()}.")
    return ex


def append_knowledge_seed(metrics):
    """Append the new-category seed to the corpus tail once.

    Safe for loss comparability: the frozen validation slice is the tail
    of the OLD corpus, and appends land strictly after it, so val indices
    are unchanged. Examples with characters outside the current charset
    are dropped (the vocab must never drift under a live checkpoint).
    """
    allowed = set(load_corpus())
    blobs = knowledge_seed_examples()
    keep = [b for b in blobs if all(c in allowed for c in b)]
    dropped = len(blobs) - len(keep)
    os.makedirs(SYN_DIR, exist_ok=True)
    with open(os.path.join(SYN_DIR, "knowledge_seed.txt"), "w") as f:
        f.write("\n\n".join(keep) + "\n")
    with open(CORPUS_PATH, "a") as f:
        f.write("\n\n" + "\n\n".join(keep) + "\n")
    metrics["knowledge_seed_v2"] = True
    # Warm restart: the floor LR is too cold to digest brand-new material.
    # Promote-or-hold still guards quality, so a fresh LR is safe.
    if metrics["lr"] < 1e-3:
        metrics["lr"] = 1e-3
        log(f"knowledge seed: LR warm restart -> {metrics['lr']:.4f}")
    save_metrics(metrics)
    log(f"knowledge seed: appended {len(keep)} new-category examples "
        f"({dropped} skipped for charset safety)")


def generate_and_absorb(model, ds, device, iteration, batches=5,
                        cat_order=None):
    """Sample from itself, filter, and append the best examples.

    cat_order ranks knowledge categories weakest-first: each sampling
    batch is primed with a category-flavored prompt, so Simply practises
    inventing questions where it scores worst. The verifier still rejects
    everything it cannot prove, so expansion is always sound.
    """
    model.eval()
    allowed = set(ds.stoi.keys())
    batch = 6
    if not cat_order:
        cat_order = list(CATEGORY_PROMPTS)
    samples = []
    with torch.no_grad():
        for i in range(batches):
            cat = cat_order[i % len(cat_order)]
            prefix = random.choice(CATEGORY_PROMPTS[cat])
            prompt_ids = [ds.stoi.get(c, 0) for c in prefix]
            prompt = torch.tensor([prompt_ids] * batch, device=device)
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


def sample_answer(model, ds, device, question="Who are you?",
                  temperature=0.7, max_new=120, top_k=30):
    prompt = f"Q: {question}\nA:"
    ids = torch.tensor([[ds.stoi.get(c, 0) for c in prompt]], device=device)
    with torch.no_grad():
        out = model.generate(ids, max_new, temperature=temperature, top_k=top_k)
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

    # One-time: seed the NEW teachable categories so the model can start
    # imitating them, then expand them by itself (see knowledge_seed_*).
    if not metrics.get("knowledge_seed_v2"):
        append_knowledge_seed(metrics)
        corpus = load_corpus()
        ds = CharDataset(corpus, block_size=MODEL_KWARGS["block_size"],
                         val_frac=VAL_FRAC,
                         val_start=metrics["val_start"],
                         val_end=metrics["val_end"])

    model, from_scratch = load_model(ds, device)
    n_params = model.num_params()
    # baseline = the COMMITTED best weights' scores; every iteration is
    # one commit (log/metrics/corpus), weights are promoted once per run
    # so the repo does not balloon with multi-MB checkpoints.
    baseline = metrics["best_val_loss"]
    run_best_loss = baseline
    run_best_val_state = None
    metrics.setdefault("knowledge", {})
    start_know = (sum(metrics["knowledge"].values()) / len(metrics["knowledge"])
                  if metrics["knowledge"] else 0.0)
    run_best_know = start_know
    run_best_know_state = None
    run_best_know_val = None
    run_best_know_acc = None
    log(f"run baseline: val_loss {'n/a' if baseline is None else f'{baseline:.4f}'}"
        f", knowledge {start_know:.2f}")

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
        synth_n = generate_and_absorb(model, ds, device, it_num,
                                      batches=args.synth_batches,
                                      cat_order=weakest_categories(
                                          metrics["knowledge"]))
        quiz_acc, quiz_line = run_knowledge_quiz(model, ds, device)
        know_avg = sum(quiz_acc.values()) / len(quiz_acc)
        metrics["knowledge"] = quiz_acc
        known_all = sum(1 for v in quiz_acc.values() if v >= 0.999)
        log(f"  knowledge quiz: {quiz_line} "
            f"({known_all}/{len(quiz_acc)} categories perfect)")

        val_loss = ds.eval_loss(model, device)
        val_gain = (run_best_loss is None
                    or val_loss < run_best_loss - IMPROVE_EPSILON)
        know_gain = know_avg > run_best_know + KNOW_EPSILON
        improved = val_gain or know_gain
        if run_best_loss is None or val_loss < run_best_loss:
            run_best_loss = val_loss
            run_best_val_state = {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}
        if know_avg > run_best_know:
            run_best_know = know_avg
            run_best_know_val = val_loss
            run_best_know_acc = quiz_acc
            run_best_know_state = {k: v.detach().cpu().clone()
                                   for k, v in model.state_dict().items()}
        status = "IMPROVED" if improved else "NO_GAIN"

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
            "know_avg": round(know_avg, 3),
            "synthetic": synth_n, "lr": metrics["lr"],
            "dataset_chars": metrics["dataset_chars"],
            "params_M": round(n_params / 1e6, 3),
            "knowledge": quiz_acc,
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
            f"- **knowledge quiz**: {quiz_line}\n"
            f"- self-portrait, asked \"Who are you?\": "
            f"*\"{portrait}\"\n"
        )

        commit_and_push(
            f"Simply iter {it_num} {status} - val {val_loss:.4f}, "
            f"know {know_avg:.2f}, +{synth_n} self-written examples",
            push=not args.no_push,
        )
        log(f"=== iteration {it_num} done: {status} ===")

    # End-of-run promotion: commit weights once if this run actually
    # beat the committed best - on frozen val loss OR on the knowledge
    # quiz (knowledge gains may not wreck fluency: their val loss must
    # stay within +0.01 of the champion). Version bumps only here.
    cand, why, cand_val, cand_acc = None, None, None, None
    if (run_best_know_state is not None
            and run_best_know >= start_know + KNOW_EPSILON
            and (baseline is None
                 or run_best_know_val <= baseline + 0.01)):
        cand = run_best_know_state
        cand_val = run_best_know_val
        cand_acc = run_best_know_acc
        why = (f"knowledge {run_best_know:.2f}, was {start_know:.2f}; "
               f"val {run_best_know_val:.4f}")
    elif (run_best_val_state is not None
          and (baseline is None or run_best_loss < baseline - IMPROVE_EPSILON)):
        cand = run_best_val_state
        cand_val = run_best_loss
        why = f"val_loss {run_best_loss:.4f}, was {baseline:.4f}"

    if cand is not None:
        model.load_state_dict(cand)
        metrics["version"] += 1
        metrics["best_val_loss"] = cand_val
        if cand_acc is not None:
            metrics["best_knowledge"] = round(run_best_know, 3)
            metrics["knowledge"] = cand_acc
        metrics["plateaus"] = 0
        save_checkpoint(model, ds, cand_val, metrics["version"],
                        os.path.join(BEST_DIR, "model.pt"))
        save_metrics(metrics)
        portrait = sample_answer(model, ds, device)
        append_log(
            f"## Promotion - v{metrics['version']} - weights committed\n"
            f"- **reason**: {why}\n"
            f"- self-portrait, asked \"Who are you?\": *\"{portrait}\"\n"
        )
        commit_and_push(
            f"Simply v{metrics['version']}: weights promoted ({why})",
            push=not args.no_push,
        )
        log(f"=== promoted weights: v{metrics['version']} ({why}) ===")

    save_metrics(metrics)
    log("all done")


if __name__ == "__main__":
    main()
