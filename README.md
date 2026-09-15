# Simply

A tiny, real, self-improving AI model. Simply is a GPT-style transformer
(character-level, ~0.8M parameters) that trains itself, grades itself,
writes new training examples from its own imagination, and keeps whatever
version actually scores better - on loop, forever, right inside this
repository. No human required.

## How the self-improvement (RSI) loop works

Every iteration of `self_improve.py` runs five steps:

1. **Train** - continues training on everything it knows so far.
2. **Self-evaluate** - measures validation loss on a frozen held-out set,
   so scores stay honest and comparable across versions.
3. **Self-write data** - samples new Q/A examples from its own imagination
   and keeps only the ones that pass quality filters (well-formed, novel,
   not repetitive). This is self-distillation: the model teaches itself.
4. **Promote or hold** - if validation loss dropped, the new weights
   become the new `best/model.pt` and the version number goes up.
   If not, the version is held and the loop tries again.
5. **Record + commit** - appends to `IMPROVEMENT_LOG.md`,
   `history.jsonl`, `metrics.json`, and pushes a commit to this repo.

The loop is fully automated via GitHub Actions (`.github/workflows/
self-improve.yml`): **every 6 hours** the model wakes up on GitHub's
servers, improves itself, and commits its progress. You never need to
be online.

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

- Runs on GitHub's `ubuntu-latest` runner, CPU-only, ~10-15 min per run.
- Schedule: every 6 hours (edit the `cron:` line in
  `.github/workflows/self-improve.yml` to change the cadence).
- Can also be triggered manually: **Actions -> Simply RSI Loop ->
  Run workflow**.
- Note: private repos get 2,000 free Actions minutes/month; the default
  schedule uses roughly 1,800. Lower the frequency if you need headroom.

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
learns and really improves its validation loss, but it is a toy-scale
model - a working demonstration of an automated self-improvement loop,
not a chatbot-scale LLM. Its favorite topics are the ones in its seed
corpus: arithmetic, spelling, capitals, animal sounds, fun facts, jokes,
and short stories about woodland creatures.

## Philosophy

`best/model.pt` only ever changes when the model genuinely improved on
the frozen validation set. Version numbers in `metrics.json` and the
commit history are the full audit trail of Simply becoming itself.
