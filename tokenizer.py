"""Simply's tokenizer: a byte-level BPE trained on its own reading.

Why byte-level: 256 base tokens cover EVERY possible byte, so no piece
of internet text can ever contain an unknown token, and the vocabulary
never drifts under a live checkpoint. On top of the 256 bytes it
learns merges until the vocabulary reaches TOKEN_VOCAB (2048). Words
the model sees a lot become single tokens, so the same context window
holds several times more text than the old character-level model.

The tokenizer is trained ONCE (from the corpus + study pool), saved to
tokenizer.json, and frozen from then on - exactly like the model's
weights, so every later run and checkpoint stays compatible.

Pure Python, no dependencies. Encoding is memoized per chunk, so
re-encoding the whole archive every run costs milliseconds.
"""
import json
import os
import re
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_PATH = os.path.join(ROOT, "tokenizer.json")

TOKEN_VOCAB = 2048  # 256 base bytes + 1792 learned merges (gen 3)

_CHUNK_RE = re.compile(r"\s?\w+|[^\w\s]|\s", re.UNICODE)


def bytes_to_unicode():
    """Readable, JSON-safe printable mapping of every byte 0-255."""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("\xa1"), ord("\xac") + 1))
          + list(range(ord("\xae"), ord("\xff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


_BYTE2UNI = bytes_to_unicode()


def _to_token_string(byte_ids):
    return "".join(_BYTE2UNI[b] for b in byte_ids)


class BPETokenizer:
    def __init__(self, merges):
        """merges: list of (token_string, token_string) pairs."""
        self.merges = merges
        self.vocab_size = 256 + len(merges)
        # id -> token string (unicode-mapped), and id -> raw bytes
        self.itos = [_to_token_string((i,)) for i in range(256)]
        self.itos_bytes = [bytes([i]) for i in range(256)]
        rank_of = {}
        a_strs, b_strs = zip(*merges) if merges else ((), ())
        str_bytes = {}
        for i in range(256):
            str_bytes[self.itos[i]] = bytes([i])
        for i, (a, b) in enumerate(merges):
            new = a + b
            self.itos.append(new)
            str_bytes[new] = str_bytes[a] + str_bytes[b]
            self.itos_bytes.append(str_bytes[new])
            rank_of[(a, b)] = i
        self._rank = rank_of
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self._memo = {}

    # --- encoding ---------------------------------------------------------
    def _encode_chunk(self, chunk_bytes):
        ids = list(chunk_bytes)
        if len(ids) < 2:
            return ids
        while True:
            best, best_rank = None, None
            for i in range(len(ids) - 1):
                p = (self.itos[ids[i]], self.itos[ids[i + 1]])
                r = self._rank.get(p)
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = p, r
            if best is None:
                return ids
            a, b = best
            out, i = [], 0
            a_id = self.stoi[a]
            b_id = self.stoi[b]
            new_id = 256 + self._rank[(a, b)]
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == a_id and ids[i + 1] == b_id:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids = out

    def encode(self, text):
        ids = []
        memo = self._memo
        for chunk in _CHUNK_RE.findall(text):
            got = memo.get(chunk)
            if got is None:
                got = self._encode_chunk(chunk.encode("utf-8"))
                memo[chunk] = got
            ids.extend(got)
        return ids

    def decode(self, ids):
        raw = b"".join(self.itos_bytes[i] for i in ids)
        return raw.decode("utf-8", errors="ignore")

    # --- persistence --------------------------------------------------------
    def save(self, path=TOKENIZER_PATH):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "vocab_size": self.vocab_size,
                       "merges": [list(m) for m in self.merges]}, f)

    @classmethod
    def load(cls, path=TOKENIZER_PATH):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls([tuple(m) for m in d["merges"]])


def train(text, vocab_size=TOKEN_VOCAB):
    """Learn merges: repeatedly merge the most frequent adjacent token
    pair across the archive. Word-internal only (standard BPE), counted
    over UNIQUE chunks so this stays fast even on megabytes of text."""
    chunk_counts = Counter(_CHUNK_RE.findall(text))
    words = {}
    for chunk, _c in chunk_counts.items():
        words[chunk] = list(chunk.encode("utf-8"))
    pair_counts = defaultdict(int)
    where = defaultdict(set)  # pair -> set of chunk keys containing it
    for chunk, ids in words.items():
        for i in range(len(ids) - 1):
            p = (ids[i], ids[i + 1])
            pair_counts[p] += 1
            where[p].add(chunk)
    merges = []
    while 256 + len(merges) < vocab_size and pair_counts:
        best = max(pair_counts.items(), key=lambda kv: kv[1])[0]
        new_id = 256 + len(merges)
        merges.append(best)
        # apply the merge to every chunk that contains it, updating
        # counts incrementally
        touched = where.pop(best, set())
        for chunk in touched:
            ids = words[chunk]
            out, i = [], 0
            while i < len(ids):
                if (i < len(ids) - 1 and ids[i] == best[0]
                        and ids[i + 1] == best[1]):
                    out.append(new_id)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            words[chunk] = out
            # update pair stats for this chunk: subtract old, add new
            old_pairs = Counter(zip(ids, ids[1:]))
            new_pairs = Counter(zip(out, out[1:]))
            for p, c in old_pairs.items():
                pair_counts[p] -= c
                where[p].discard(chunk)
            for p, c in new_pairs.items():
                pair_counts[p] += c
                where[p].add(chunk)
        # drop non-positive counts
        for p in [p for p, c in pair_counts.items() if c <= 0]:
            del pair_counts[p]
    # convert byte-id merges to token-string pairs for persistence
    uni = {}
    for i in range(256):
        uni[i] = _to_token_string((i,))
    out = []
    for i, (a, b) in enumerate(merges):
        out.append((uni[a], uni.get(b) or ""))
        # rebuild progressively: b may itself be a merged token
        uni[256 + i] = uni[a] + uni[b]
    return BPETokenizer(out)


def get():
    """Load the frozen tokenizer (must already exist - see train())."""
    return BPETokenizer.load()


if __name__ == "__main__":
    corpus = open(os.path.join(ROOT, "corpus.txt"), encoding="utf-8").read()
    for extra in ("lessons.txt", os.path.join("reading", "facts.txt")):
        p = os.path.join(ROOT, extra)
        if os.path.exists(p):
            corpus += open(p, encoding="utf-8").read()
    print(f"training BPE on {len(corpus):,} chars ...")
    tok = train(corpus)
    tok.save()
    print(f"vocab {tok.vocab_size} saved to {TOKENIZER_PATH}")
    s = "What is the capital of France? Q: Paris is a city."
    ids = tok.encode(s)
    print("roundtrip ok:", tok.decode(ids) == s, "|", len(s), "chars ->",
          len(ids), "tokens")
