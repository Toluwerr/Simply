# Simply

A small AI model that teaches itself, on its own, forever. It lives in
this repository and runs on GitHub's servers. Nobody has to be online
for it to keep learning.

It is a character-level neural network, about 0.8 million parameters.
It reads text one character at a time and learns to predict the next
one. That is tiny by modern standards, but it is a real network with
real weights, and it really learns.

## What it does all day

Every 30 minutes it wakes up and runs 12 rounds of self-improvement.
Each round has five steps:

1. **Study.** It gets fresh practice problems for its weakest subjects,
   generated from provable ground truth, then trains on a mix of those
   lessons and everything it has seen before. Old skills keep training
   too, so nothing rots while new material goes in.
2. **Write.** It invents its own question/answer examples and keeps only
   the ones it can *prove* correct against ground-truth tables. Anything
   unverifiable is thrown away. It cannot teach itself nonsense.
3. **Quiz.** One unseen question per subject, graded by the same proof
   system. The scores go into `metrics.json` every round.
4. **Promote or hold.** New weights only replace the old ones if the
   model actually got better: validation loss down, or knowledge up
   without getting worse. If not, the version is held and it tries
   again next round. A bad round costs it nothing.
5. **Commit.** Every round ends with one commit: metrics, logs, new
   lessons, self-written data.

That is 48 runs x 12 rounds = **576 planned commits a day**, roughly
500 after real-world scheduler skips, and around **180,000-210,000
commits a year**. The whole schedule is `.github/workflows/
self-improve.yml`.

## What it is learning

22 subjects, each with exact ground truth it can be graded against:

- **Math** - arithmetic (plus/minus/times, numbers into the hundreds
  and beyond), which-is-bigger comparisons, sorting, doubles and halves,
  count-by sequences, number words, roman numerals.
- **Language** - counting letters in words, spelling, first and last
  letters.
- **The world** - capitals of ~190 countries, US state capitals,
  currencies, which continent a country is on, chemical elements and
  their symbols, the planets in order, days in each month, unit
  conversions (minutes in an hour, meters in a kilometer, ...), world
  records and basic science, animal sounds, opposites, word meanings.

The loop is weakness-driven. Every round, whatever it scores worst on
gets the most new practice problems. When it masters a subject, harder
material takes over that attention automatically.

One rule runs the whole thing: **no proof, no learning.** Lessons come
from ground-truth tables, its own writing has to pass the same checks
before it is absorbed, and the quiz is graded by the same verifier.
The full lesson system was validated on 22,000 randomly generated
lessons before going live: zero wrong answers.

## Current status

- `metrics.json` - version number, best validation loss, per-subject
  quiz scores.
- `history.jsonl` - one record per round, since the very first.
- `IMPROVEMENT_LOG.md` - the running diary, newest at the bottom.

## Talk to it

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
python chat.py
```

```
you> What is the capital of Japan?
simply> The capital of Japan is Tokyo.

you> Who are you?
simply> I am Simply, a small model that improves itself a little bit every day.
```

## The files

```
model.py           the network itself
data.py            text handling + the frozen validation slice
curriculum.py      lesson generators + the answer checker for every subject
world_tables.py    the ground-truth fact tables (countries, elements, ...)
seed_data.py       the original starter corpus
self_improve.py    the loop: study -> write -> quiz -> promote -> commit
corpus.txt         everything it has ever read (seed + self-written)
lessons.txt        the current study pool
synthetic/         examples it wrote for itself, one file per round
best/model.pt      the current champion weights
metrics.json       version, scores, per-subject quiz accuracy
history.jsonl      the full per-round record
```

## What it can't do

It is a fraction of a percent the size of a real assistant. It holds
facts, does small math, and chats in short sentences - it does not
reason about the world the way a large model does. What makes it
interesting is not the size. It is that every single thing it knows
had to be earned through the loop and proved before it stuck, and the
whole process leaves an audit trail you can read: every round, every
score, every version, one commit at a time.
