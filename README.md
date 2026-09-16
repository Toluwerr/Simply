# Simply

A small AI model that teaches itself, on its own, forever. It lives in
this repository and runs on GitHub's servers. Nobody has to be online
for it to keep learning.

It is a neural network, about 1.1 million parameters, that reads text
in chunks called tokens (a trained vocabulary of 1,024 built from its
own reading - every possible byte is covered, so nothing it reads
can ever be unreadable to it). That is tiny by modern standards, but
it is a real network with real weights, and it really learns. Tokens
let it see about three times more text per thought than the old
character-by-character version.

## What it does all day

It does not wait for a scheduler. Every run, when it finishes its 10
rounds of self-improvement, it immediately triggers the next run
itself - the loop keeps itself alive around the clock (a cron every 30
minutes stays as a backstop). Each cycle is:

0. **Read.** It fetches about 30 real articles (Wikipedia's public
   API - no account, no key) and stores the text in `reading/`. From
   those articles it distills short facts - definitions lifted
   straight from the source - into its study pool. Roughly 1,400
   articles a day, every one recorded in `reading/learned.txt`. The
   first thing it reads every run is its own home: a live report from
   GitHub's API about this repository - its commits, stars, and how
   it all works.
1. **Study.** It gets fresh practice problems for its weakest subjects,
   generated from provable ground truth, then trains on a mix of those
   lessons and everything it has seen before. Old skills keep training
   too, so nothing rots while new material goes in.
2. **Write.** It invents its own question/answer examples and keeps only
   the ones it can *prove* correct against ground-truth tables. Anything
   unverifiable is thrown away. It cannot teach itself nonsense.
3. **Quiz.** One unseen question per subject, graded by the same proof
   system - including the reading subject, where it has to recall a
   fact from something it actually read. The scores go into
   `metrics.json` every round.
4. **Promote or hold.** New weights only replace the old ones if the
   model actually got better: validation loss down, or knowledge up
   without getting worse. If not, the version is held and it tries
   again next round. A bad round costs it nothing.
5. **Commit.** Every round ends with one commit: metrics, logs, new
   lessons, self-written data. Then the run chains the next one.

That is 10 rounds per run, chained back to back: roughly **700-900
commits a day**, around **280,000-320,000 commits a year**. The whole
schedule is `.github/workflows/self-improve.yml`.

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
matches what the source actually said. Training time is split three
ways: the study pool, the raw article text, and everything it has ever
seen. So the more of the internet it reads, the more of its brain is
spent on the internet.

The loop is weakness-driven. Every round, whatever it scores worst on
gets the most new practice problems. When it masters a subject, harder
material takes over that attention automatically.

One rule runs the whole thing: **no proof, no learning.** Lessons come
from ground-truth tables, its own writing has to pass the same checks
before it is absorbed, reading facts have to match their source
article, and the quiz is graded by the same verifier. The full lesson
system was validated on 22,000 randomly generated lessons before going
live: zero wrong answers.

## Current status

- `metrics.json` - version number, best validation loss, per-subject
  quiz scores, and a `reading` block: articles read, facts stored.
- `history.jsonl` - one record per round, since the very first.
- `IMPROVEMENT_LOG.md` - the running diary, newest at the bottom.
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
model.py           the network itself
tokenizer.py       the frozen byte-level BPE vocabulary (1024 tokens)
tokenizer.json     the trained vocabulary itself
data.py            dataset handling + the frozen validation slice
curriculum.py      lesson generators + the answer checker for every subject
world_tables.py    the ground-truth fact tables (countries, elements, ...)
knowledge.py       the internet reader: fetch articles, distill facts, grade recall
seed_data.py       the original starter corpus
self_improve.py    the loop: read -> study -> write -> quiz -> promote -> commit
corpus.txt         everything it has ever read (seed + self-written)
lessons.txt        the current study pool
reading/           learned.txt (titles), pool.txt (article text), facts.txt (distilled facts)
synthetic/         examples it wrote for itself, one file per round
best/model.pt      the current champion weights
metrics.json       version, scores, per-subject quiz accuracy, reading stats
history.jsonl      the full per-round record
```

## What it can't do

It is a fraction of a percent the size of a real assistant. It reads
constantly and it never forgets where a fact came from, but it holds
only what fits in 0.8 million parameters: a rolling working memory of
what it has read lately, drilled facts, and small skills. It does not
reason about the world the way a large model does. What makes it
interesting is not the size. It is that every single thing it knows
had to be earned through the loop and proved before it stuck, and the
whole process leaves an audit trail you can read: every round, every
score, every article, one commit at a time.
