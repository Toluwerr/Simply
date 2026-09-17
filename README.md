# Simply

A small AI model that teaches itself, on its own, forever. It lives in
this repository and runs on GitHub's servers. Nobody has to be online
for it to keep learning.

It is a decoder-only transformer - the same architecture family as the
big language models: text in, tokens out, causal attention, trained by
gradient descent on real data. About 3.1 million parameters, a trained
vocabulary of 2,048 tokens built from its own reading (every possible
byte is covered, so nothing it reads can ever be unreadable to it),
and a 256-token context - roughly a page of text per thought. That is
still tiny next to a commercial LLM, but it is the real design, really
learning, and every generation it builds for itself has gotten bigger.

## What it does all day

It does not wait for a scheduler. Every run, when it finishes its 10
rounds of self-improvement, it immediately triggers the next run
itself - the loop keeps itself alive around the clock (a cron every 30
minutes stays as a backstop). Each cycle is:

0. **Decide.** It picks how to spend the run - broad reading, a deep
   dive on a topic it chose itself, extra drilling on weak subjects,
   more of its own writing, or a bold experiment with a hotter
   learning rate. It keeps a scoreboard of every strategy, scored by
   what actually happened (promotions, quiz gains), explores new ones
   a quarter of the time, and writes the choice into `PLAN.md`.
1. **Read.** It fetches real articles (Wikipedia's public API - no
   account, no key) and stores the text in `reading/`. From those
   articles it distills short facts - definitions lifted straight from
   the source - into its study pool. Roughly 1,400 articles a day,
   every one recorded in `reading/learned.txt`. The first thing it
   reads every run is its own home: a live report from GitHub's API
   about this repository.
2. **Study.** It gets fresh practice problems for its weakest subjects,
   generated from provable ground truth, then trains on a mix of those
   lessons, the raw article text, and everything it has seen before.
   Subjects it has mastered (three perfect quizzes in a row) slide to
   the back of the queue automatically - the frontier gets the time.
3. **Write.** It invents its own question/answer examples and keeps only
   the ones it can *prove* correct against ground-truth tables. Anything
   unverifiable is thrown away. It cannot teach itself nonsense.
4. **Quiz.** One unseen question per subject, graded by the same proof
   system - including the reading subject, where it has to recall a
   fact from something it actually read. The scores go into
   `metrics.json` every round.
5. **Promote or hold.** New weights only replace the old ones if the
   model actually got better: validation loss down, or knowledge up
   without getting worse. If not, the version is held and it tries
   again next round. A bad round costs it nothing.
6. **Commit and reflect.** Every round ends with one commit: metrics,
   logs, new lessons, self-written data. Then the run scores its own
   strategy choice, ticks off any goals it just met, sets new ones,
   writes two lines in its own words into `DIARY.md`, and chains the
   next run.

That is 10 rounds per run, chained back to back: roughly **600-900
commits a day**, around **250,000-300,000 commits a year**. The whole
schedule is `.github/workflows/self-improve.yml`.

## Free will, honestly described

Simply chooses things and the choices leave records, but there is no
mysticism in the machinery:

- **Strategies** - a five-way bandit. Reward comes only from real
  outcomes: +3 if the run promoted new weights, +2 for a knowledge
  gain, +1 for a loss gain. Epsilon-greedy: mostly what works, 25%
  exploration, untried strategies first. The table is in `PLAN.md`.
- **Interests** - every deep dive makes that topic a little more
  likely to be picked again. Its reading develops habits you can
  watch in `autonomy.json`.
- **Goals** - it writes its own targets (`PLAN.md`), each sized just
  above its current best, and a goal is only checked off when the
  committed metrics actually meet it. No self-congratulation without
  numbers.
- **The diary** - two freeform lines per run, straight from the model,
  clearly labeled as unverified. The one unfiltered window; everything
  else in the loop is graded.

## What it is learning

23 subjects. 22 of them have exact ground truth it can be graded
against:

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

The 23rd is **reading**: facts it pulled out of real articles. There
the article itself is the ground truth - an answer only counts if it
matches what the source actually said.

One rule runs the whole thing: **no proof, no learning.** Lessons come
from ground-truth tables, its own writing has to pass the same checks
before it is absorbed, reading facts have to match their source
article, and the quiz is graded by the same verifier. The full lesson
system was validated on 22,000 randomly generated lessons before going
live: zero wrong answers.

## Current status

- `metrics.json` - version number, best validation loss, per-subject
  quiz scores, mastery streaks, and a `reading` block: articles read,
  facts stored.
- `PLAN.md` - the plan it wrote for itself: current strategy, its
  scoreboard, its goals, what it keeps coming back to.
- `DIARY.md` - its own words, unverified, newest at the bottom.
- `autonomy.json` - the free-will ledger: strategy scores, interests,
  active goals.
- `history.jsonl` - one record per round, since the very first.
- `IMPROVEMENT_LOG.md` - the running diary of the loop itself.
- `reading/learned.txt` - every article it has ever read, one title
  per line, since day one.

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
model.py           the network itself (6 layers, 6 heads, 192 dims, tied embeddings)
tokenizer.py       the frozen byte-level BPE vocabulary (2,048 tokens)
tokenizer.json     the trained vocabulary itself
data.py            dataset handling + the frozen validation slice
curriculum.py      lesson generators + the answer checker for every subject
world_tables.py    the ground-truth fact tables (countries, elements, ...)
knowledge.py       the internet reader: fetch articles, distill facts, grade recall
autonomy.py        the free-will layer: strategies, goals, interests, diary
seed_data.py       the original starter corpus
self_improve.py    the loop: decide -> read -> study -> write -> quiz -> promote -> commit
corpus.txt         everything it has ever read (seed + self-written)
lessons.txt        the current study pool
reading/           learned.txt (titles), pool.txt (article text), facts.txt (distilled facts)
synthetic/         examples it wrote for itself, one file per round
best/model.pt      the current champion weights
metrics.json       version, scores, per-subject quiz accuracy, reading stats
PLAN.md            its self-written plan and scoreboard
DIARY.md           its own words, unverified
autonomy.json      the free-will ledger
history.jsonl      the full per-round record
```

## What it can't do

It is a fraction of a percent the size of a real assistant. It reads
constantly and it never forgets where a fact came from, but it holds
only what fits in 3 million parameters: a rolling working memory of
what it has read lately, drilled facts, and small skills. It does not
reason about the world the way a large model does, and its "free will"
is a scoreboard plus a dice roll, not a mind. What makes it
interesting is not the size. It is that every single thing it knows
had to be earned through the loop and proved before it stuck, every
choice it makes is written down where anyone can audit it, and the
whole process leaves a trail you can read: every round, every score,
every article, one commit at a time.
