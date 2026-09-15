# Simply

A tiny, real, self-improving AI model. Simply is a GPT-style transformer
(character-level, ~0.8M parameters) that trains itself, grades itself,
writes new training examples from its own imagination, expands the
knowledge categories it can learn from, and keeps whatever version
actually scores better - on loop, forever, right inside this public
repository. No human required.

## How the self-improvement (RSI) loop works

Every iteration of `self_improve.py` runs six steps:

1. **Train** - continues training on everything it knows so far.
2. **Self-write data** - samples new Q/A examples from its own imagination
   (primed toward its weakest knowledge categories) and keeps only the
   ones it can *prove* correct: the example must match a known canonical
   template and the answer must check out against ground truth.
   Unverifiable output is discarded, so Simply can never teach itself
   nonsense.
3. **Knowledge quiz** - grades itself on 12 fixed probe questions, one
   per knowledge category, decoded greedily and checked with the same
   verifier used for self-written data. Per-category accuracy is
   committed every iteration, so knowledge growth is measurable.
4. **Self-evaluate** - measures validation loss on a frozen held-out set,
   so scores stay honest and comparable across versions.
5. **Promote or hold** - if validation loss dropped *or* knowledge-quiz
   accuracy rose (without wrecking fluency), the new weights become the
   new `best/model.pt`, the version number goes up, and the learning
   rate re-arms. If not, the version is held and the loop tries again.
6. **Record + commit** - appends to `IMPROVEMENT_LOG.md`,
   `history.jsonl`, `metrics.json`, and pushes a commit to this repo.

The loop is fully automated via GitHub Actions (`.github/workflows/
self-improve.yml`): **every 30 minutes** the model wakes up on GitHub's
servers, improves itself 12 times (one commit per iteration), and pushes
its progress. You never need to be online.

## Self-expanding knowledge

Simply's learnable world is defined by *provable question templates* -
and it expands that world by itself:

- **12 knowledge categories**, each with deterministic ground truth:
  arithmetic, comparisons (yes/no + which-is-bigger), sorting, doubles &
  halves, count-by sequences, numbers-in-words, letter counting &
  spelling, first/last letters, capitals, animal sounds, opposites, and
  definitions.
- **Curriculum**: every iteration the quiz ranks categories weakest
  first, and the self-data sampler is primed with the weakest category's
  question style. Simply practises inventing questions where it scores
  worst - and the verifier only lets *provable* answers into the corpus.
- **Growth is auditable**: `metrics.json` carries per-category accuracy
  on every commit; `IMPROVEMENT_LOG.md` prints the weakest categories;
  `history.jsonl` stores the full quiz record per iteration.
- New self-written examples are appended to `corpus.txt` (up to 8M
  chars), so the training material grows with the model's abilities.

## Quickstart

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu

python seed_data.py                        # (re)build the seed corpus
python self_improve.py --iterations 3      # run 3 self-improvement loops
python chat.py                             # talk to the current best Simply
```

Chat example:

```
you> Who are you?
simply> I am Simply, a small language model that improves itself a little bit every day.
you> What is 7 plus 5?
simply> 7 plus 5 is 12.
```

## Repository layout

```
model.py                 the Simply transformer (decoder-only GPT)
data.py                  char-level dataset + frozen validation split
seed_data.py             builds the seed corpus (corpus.txt)
self_improve.py          the RSI loop (train/eval/self-write/promote/commit)
chat.py                  talk to the best version
corpus.txt               everything Simply knows (seed + self-written)
synthetic/               examples Simply wrote for itself, per iteration
best/model.pt            current best weights (the official Simply)
last/model.pt            latest weights (kept locally, not committed)
metrics.json             version, best val loss, lr, corpus size
history.jsonl            one metrics record per iteration
IMPROVEMENT_LOG.md       human-readable auto-generated progress log
.github/workflows/       the automation that keeps Simply improving
```

## Automation

- Runs on GitHub's `ubuntu-latest` runner, CPU-only. **48 scheduled
  runs/day** (every 30 minutes at :17 and :47 UTC), each packing **12
  self-improvement iterations** (200 training steps each). **Every
  iteration is one commit**, so the schedule plans **576 commits/day**
  - realistically ~500+ after GitHub's occasional scheduler skips.
- That is **~15,000-17,000 commits/month** and roughly
  **183,000-192,000 commits/year**.
- The repo is public, so Actions minutes are **free and unlimited** -
  the cadence is bounded only by run duration (~12 min) vs the 30-min
  slot, not by any quota.
- Weights (`best/model.pt`) are promoted and committed once per run,
  only when the run actually beat the committed best (on validation
  loss or on knowledge-quiz accuracy) - this keeps the repo from
  ballooning with checkpoint blobs while every iteration still commits
  its log, metrics, and corpus updates.
- Every run retrains from the **champion** weights (`best/`), so a bad
  run is simply not promoted and the next run starts clean.
- GitHub's scheduler occasionally skips/delays a slot (their cron is
  best-effort); instant runs: **Actions -> Simply RSI Loop ->
  Run workflow**.

## Architecture

| Config    | Value |
|-----------|-------|
| Type      | decoder-only transformer (GPT-style) |
| Tokens    | character-level (~60 vocab) |
| Layers    | 4 |
| Heads     | 4 |
| Embed dim | 128 |
| Context   | 128 characters |
| Params    | ~0.8M |
| Optimizer | AdamW, weight decay 0.01, grad clip 1.0 |

Simply is honest about its size: it is a real neural network that really
learns and really improves, but it is a toy-scale model - a working
demonstration of an automated, measurable self-improvement loop, not a
chatbot-scale LLM. Its knowledge spans arithmetic, comparisons, sorting,
doubles/halves, counting, number words, letters, capitals, animal
sounds, opposites, definitions, fun facts, jokes, and short stories
about woodland creatures.

## Philosophy

`best/model.pt` only ever changes when the model genuinely improved -
on the frozen validation set or on the provable knowledge quiz. Version
numbers in `metrics.json`, per-category quiz scores, and the commit
history are the full audit trail of Simply becoming itself.
