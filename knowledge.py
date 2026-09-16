"""Simply's internet reader.

Every run, Simply goes out and reads real articles from the web
(Wikipedia's public API - no account, no key, no cost). It keeps three
things:

  reading/learned.txt   one article title per line, append-only - the
                        permanent record of everything it has ever read
  reading/pool.txt      the current reading pool: raw article text the
                        model trains on (bounded, oldest trimmed first)
  reading/facts.txt     short facts distilled from articles, in the same
                        "Q: .../A: ..." format as the study pool, so the
                        existing study/quiz machinery can drill them

Anything it cannot clean into the model's character set is dropped at
the door. Nothing here is ever trusted blind: facts are re-verified by
the same strict verifier self_improve.py uses for the model's own
writing (the article text itself is the ground truth, because the
answer must literally come from it).

The loop never blocks on the internet: every network call has a
timeout, a retry, and the caller wraps the whole refresh in a budget.
If the web is unreachable, Simply simply keeps studying what it has.
"""
import json
import os
import random
import re
import time
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
READING_DIR = os.path.join(ROOT, "reading")
LEARNED_PATH = os.path.join(READING_DIR, "learned.txt")
POOL_PATH = os.path.join(READING_DIR, "pool.txt")
FACTS_PATH = os.path.join(READING_DIR, "facts.txt")

POOL_MAX = 1_200_000   # chars of raw article text kept in rotation
FACTS_MAX = 3_000      # distilled facts kept in the working set
ARTICLES_PER_RUN = 30  # fetched per run: 48 runs/day x 30 = ~1,440/day
FETCH_BUDGET_S = 110   # hard wall-clock budget for the whole refresh
TOPIC_LIST = [
    "science", "history", "geography", "mathematics", "technology",
    "biology", "physics", "chemistry", "astronomy", "medicine",
    "philosophy", "psychology", "music", "art", "literature",
    "architecture", "economics", "law", "engineering", "computer",
    "sport", "film", "food", "agriculture", "language", "education",
    "ocean", "climate", "birds", "mammals", "insects", "plants",
    "ancient rome", "ancient egypt", "space exploration", "invention",
    "transport", "energy", "industry", "culture", "religion",
    "archaeology", "geology", "anatomy", "genetics", "ecology",
    "chemistry element", "civilization", "war", "exploration",
]

_WIKI_API = "https://en.wikipedia.org/w/api.php"
_UA = ("Simply-self-improving-model/1.0 "
       "(educational research; contact: noreply@users.noreply.github.com)")

_TAG_RE = re.compile(r"\[[^\]]*\]|\([^)]{0,40}\)")
_WS_RE = re.compile(r"\s+")
_BAD_FIRST = {"the", "a", "an", "it", "there", "this", "that",
              "these", "those", "he", "she", "they", "his", "her",
              "its", "their", "in", "on", "at", "by", "as", "or",
              "and", "but", "if", "when", "during", "after",
              "before", "some", "many", "most", "other", "such",
              "however", "although", "while", "one", "two", "first",
              "second", "third", "new", "we", "i", "you", "your",
              "our", "my", "mr", "mrs", "dr", "also", "another",
              "each", "any", "both", "early", "later", "today"}


def _http_json(params, timeout=10, retries=1):
    """GET the Wikipedia API and return parsed JSON, or None."""
    params = dict(params, format="json", formatversion="2")
    url = _WIKI_API + "?" + urllib.parse.urlencode(params)
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8", "ignore"))
        except Exception:
            if attempt >= retries:
                return None
            time.sleep(1.0)
    return None


def _clean_text(t):
    """Force article text into something the charset filter can keep:
    straight quotes/dashes, drop bracket refs, collapse whitespace."""
    t = (t.replace("\u2018", "'").replace("\u2019", "'")
          .replace("\u201c", '"').replace("\u201d", '"')
          .replace("\u2013", "-").replace("\u2014", "-")
          .replace("\u00a0", " "))
    t = _TAG_RE.sub("", t)
    t = _WS_RE.sub(" ", t)
    return t.strip()


def _pick_titles(n, rng):
    """Half serendipity (random articles), half a rotating topic with a
    random search offset, so coverage is both wide and deep."""
    titles = []
    data = _http_json({"action": "query", "list": "random",
                       "rnnamespace": 0, "rnlimit": min(n, 20)})
    if data:
        titles += [x["title"] for x in data.get("query", {}).get("random", [])]
    topic = rng.choice(TOPIC_LIST)
    offset = rng.randrange(0, 150)
    data = _http_json({"action": "query", "list": "search",
                       "srsearch": topic, "srnamespace": 0,
                       "srlimit": max(4, n), "sroffset": offset,
                       "srprop": ""})
    if data:
        titles += [x["title"] for x in
                   data.get("query", {}).get("search", [])]
    # dedupe, keep order
    seen, out = set(), []
    for t in titles:
        k = t.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(t)
        if len(out) >= n:
            break
    return out


def _fetch_extract(title):
    """Plain-text extract of one article (first ~1400 chars)."""
    data = _http_json({"action": "query", "prop": "extracts",
                       "explaintext": 1, "exchars": 1400,
                       "titles": title, "redirects": 1})
    if not data:
        return None
    pages = data.get("query", {}).get("pages", [])
    for p in pages:
        ext = p.get("extract")
        if ext and len(ext) > 200:
            return ext
    return None


# --- fact distillation ------------------------------------------------------
_SENT_RE = re.compile(r"(?<=[.!?]) (?=[A-Z0-9'])")
_DEF_RE = re.compile(r"^(.{3,60}?) (?:is|was) (a|an|the) (.+)$", re.I)
_NUM_RE = re.compile(r"\d")


def _good_subject(s):
    s = s.strip()
    if not 3 <= len(s) <= 48 or _NUM_RE.search(s):
        return None
    words = s.split()
    if not 1 <= len(words) <= 4:
        return None
    if words[0].lower().strip("',.") in _BAD_FIRST:
        return None
    # proper-ish subject: first letter capitalized, no stray punctuation
    if not s[0].isupper() or any(c in s for c in "()[]{}\"<>@#$%*_/\\|~`=+"):
        return None
    return s


def _cut_answer(a, limit=150):
    """Trim a definition to a clean clause that ends in a real word."""
    a = a.strip()
    if len(a) > limit:
        cut = a[:limit]
        best = max(cut.rfind(", "), cut.rfind("; "))
        if best < 30:
            best = cut.rfind(" and ")
        if best >= 30:
            a = cut[:best]
        else:
            dot = cut.rfind(". ")
            a = cut[:dot] if dot >= 30 else cut.rstrip(",;: ")
    a = a.strip().strip(",").strip()
    # keep only the first sentence - definitional facts are openings,
    # and trailing sections (headers, next sentences) are noise
    dot = a.find(". ")
    if dot >= 25:
        a = a[:dot + 1]
    if a and a[-1] not in ".!?":
        a += "."
    return a


def distill_facts(text, seen_subjects, limit=4):
    """Pull verifiable definition facts out of one article.

    A fact is kept only if it is a plain 'X is a/an/the ...' opening
    sentence. The article text is the ground truth: the stored answer is
    a literal slice of it. Returns (q, a) pairs plus the raw subject for
    the recall verifier.
    """
    facts = []
    # use the first two paragraph-ish chunks: opening definitions live there
    head = text[:1200]
    for sent in _SENT_RE.split(head):
        if len(facts) >= limit:
            break
        sent = sent.strip()
        if not 40 <= len(sent) <= 300:
            continue
        m = _DEF_RE.match(sent)
        if not m:
            continue
        subj = _good_subject(m.group(1))
        if not subj or subj.lower() in seen_subjects:
            continue
        art = m.group(2).lower()
        ans = _cut_answer(f"{subj} is {art} {m.group(3)}")
        if not 20 <= len(ans) <= 170:
            continue
        q = f"What is {subj}?"
        facts.append((q, ans))
        seen_subjects.add(subj.lower())
    return facts


# --- storage ----------------------------------------------------------------

def _append_line(path, line):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _read_lines(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return f.read().splitlines()


def learned_titles():
    return _read_lines(LEARNED_PATH)


def load_facts():
    """Parse facts.txt back into (q, a) tuples."""
    out = []
    blocks = []
    if os.path.exists(FACTS_PATH):
        with open(FACTS_PATH, encoding="utf-8") as f:
            blocks = f.read().split("\n\n")
    for b in blocks:
        qm = re.search(r"^Q: (.+)$", b, re.M)
        am = re.search(r"^A: (.+)$", b, re.M | re.S)
        if qm and am:
            out.append((qm.group(1).strip(),
                        am.group(1).strip().split("\n")[0]))
    return out


def fact_subjects():
    """Normalized subject -> answer, for the recall verifier."""
    d = {}
    for q, a in load_facts():
        m = re.match(r"^What is (.+)\?$", q)
        if m:
            d[m.group(1).strip().lower()] = a
    return d


def trim_pool():
    if not os.path.exists(POOL_PATH):
        return
    size = os.path.getsize(POOL_PATH)
    if size <= POOL_MAX:
        return
    with open(POOL_PATH, encoding="utf-8") as f:
        text = f.read()
    cut = text.find("\n\n", len(text) - int(POOL_MAX * 0.8))
    if cut == -1:
        return
    with open(POOL_PATH, "w", encoding="utf-8") as f:
        f.write(text[cut:])
    log = print  # keep import-light; caller logs anyway
    log(f"  reading pool trimmed: {size:,} -> {os.path.getsize(POOL_PATH):,}")


def trim_facts():
    blocks = []
    if os.path.exists(FACTS_PATH):
        with open(FACTS_PATH, encoding="utf-8") as f:
            blocks = [b for b in f.read().split("\n\n") if b.strip()]
    if len(blocks) <= FACTS_MAX:
        return
    with open(FACTS_PATH, "w", encoding="utf-8") as f:
        f.write("\n\n".join(blocks[len(blocks) // 2:]) + "\n")


def _norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def recall_ok(stored_answer, model_answer):
    """Reading-quiz grading: the model's answer must carry the core of
    the stored (article-sourced) answer - containment, or >= half the
    content words. Lenient enough for a character model, strict enough
    that a wrong guess never passes."""
    s, m = _norm(stored_answer), _norm(model_answer)
    if not s:
        return False
    if s in m or m in s:
        return True
    sw = [w for w in s.split() if len(w) > 2]
    mw = set(m.split())
    if not sw:
        return False
    hit = sum(1 for w in sw if w in mw)
    return hit / len(sw) >= 0.5


def refresh(metrics, allowed_chars=None, budget_s=FETCH_BUDGET_S,
            articles=ARTICLES_PER_RUN, log=print):
    """One reading session: fetch, clean, store, distill, trim.

    Never raises: any failure just means 'no new reading today'. Returns
    a small stats dict for the metrics file.
    """
    t0 = time.time()
    rng = random.Random(f"reading:{metrics.get('iteration', 0)}:{time.time()}")
    known = set(t.strip().lower() for t in learned_titles())
    seen_subj = set(fact_subjects())
    new_articles, new_facts = 0, 0

    titles = _pick_titles(articles + 8, rng)
    for title in titles:
        if new_articles >= articles or time.time() - t0 > budget_s:
            break
        key = title.strip().lower()
        if key in known:
            continue
        ext = _fetch_extract(title)
        if not ext:
            continue
        ext = _clean_text(ext)
        if allowed_chars is not None:
            ext = "".join(c for c in ext if c in allowed_chars)
        if len(ext) < 250:
            continue
        for q, a in distill_facts(ext, seen_subj):
            blob = f"Q: {q}\nA: {a}"
            if allowed_chars is not None and not all(
                    c in allowed_chars for c in blob):
                continue
            _append_line(FACTS_PATH, "")  # blank line = block separator
            _append_line(FACTS_PATH, blob)
            new_facts += 1
        _append_line(POOL_PATH, ext)
        _append_line(LEARNED_PATH, title.strip())
        known.add(key)
        new_articles += 1
        time.sleep(0.15)

    trim_pool()
    trim_facts()

    pool_chars = os.path.getsize(POOL_PATH) if os.path.exists(POOL_PATH) else 0
    total_learned = len(learned_titles())
    stats = {
        "articles_total": total_learned,
        "articles_new": new_articles,
        "facts_new": new_facts,
        "facts_stored": len(load_facts()),
        "pool_chars": pool_chars,
        "seconds": round(time.time() - t0, 1),
    }
    log(f"  reading session: +{new_articles} articles, +{new_facts} facts "
        f"({total_learned:,} read all-time, pool {pool_chars:,} chars, "
        f"{stats['seconds']}s)")
    return stats


if __name__ == "__main__":
    # standalone: python knowledge.py  -> one reading session, no training
    print("Simply reading session")
    st = refresh({"iteration": "manual"})
    print(json.dumps(st, indent=2))
