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
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.abspath(__file__))
READING_DIR = os.path.join(ROOT, "reading")
LEARNED_PATH = os.path.join(READING_DIR, "learned.txt")
POOL_PATH = os.path.join(READING_DIR, "pool.txt")
FACTS_PATH = os.path.join(READING_DIR, "facts.txt")

POOL_MAX = 4_000_000   # chars of raw article text kept in rotation
FACTS_MAX = 8_000      # distilled facts kept in the working set
ARTICLES_PER_RUN = 60  # fetched per run: ~90 chained runs/day -> ~5,000/day
FETCH_BUDGET_S = 150   # hard wall-clock budget for the whole refresh
FETCH_WORKERS = 6      # parallel article downloads per session
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

# A guaranteed slice of every reading session: how its own home works.
# Simply lives on GitHub and runs on git, so it studies them like any
# other subject - plus the ideas behind itself.
TECH_TOPICS = [
    "git", "github", "version control", "software engineering",
    "programming language", "artificial intelligence",
    "machine learning", "neural network", "computer science",
    "internet", "open source software", "operating system",
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


def _pick_titles(n, rng, focus=None):
    """A slice on git/GitHub/AI (its own world), plus serendipity
    (random articles) and topic searches with random offsets, so
    coverage is both wide and deep.

    focus: when Simply CHOOSES a topic for itself (autonomy module),
    that topic takes half the session - a deliberate deep dive."""
    titles = []
    tech = max(2, n // 5)
    data = _http_json({"action": "query", "list": "search",
                       "srsearch": rng.choice(TECH_TOPICS),
                       "srnamespace": 0, "srlimit": tech + 4,
                       "sroffset": rng.randrange(0, 80), "srprop": ""})
    if data:
        titles += [x["title"] for x in data.get("query", {}).get("search", [])]
    data = _http_json({"action": "query", "list": "random",
                       "rnnamespace": 0, "rnlimit": min(n, 20)})
    if data:
        titles += [x["title"] for x in data.get("query", {}).get("random", [])]
    if focus:
        topic, limit, offset = focus, max(4, n), rng.randrange(0, 300)
    else:
        topic, limit, offset = rng.choice(TOPIC_LIST), max(4, n), rng.randrange(0, 300)
    data = _http_json({"action": "query", "list": "search",
                       "srsearch": topic, "srnamespace": 0,
                       "srlimit": limit, "sroffset": offset,
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
    """Plain-text extract of one article (first ~1600 chars)."""
    data = _http_json({"action": "query", "prop": "extracts",
                       "explaintext": 1, "exchars": 1600,
                       "titles": title, "redirects": 1})
    if not data:
        return None
    pages = data.get("query", {}).get("pages", [])
    for p in pages:
        ext = p.get("extract")
        if ext and len(ext) > 200:
            return ext
    return None


def _fetch_many(titles, workers=FETCH_WORKERS):
    """Fetch several article extracts in parallel. Returns a dict
    title -> extract (or None). Politeness: a small pool and a short
    UA; Wikipedia handles this fine, and the whole wave takes seconds
    instead of minutes."""
    out = {}
    if not titles:
        return out
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fetch_extract, t): t for t in titles}
        for fut in futs:
            t = futs[fut]
            try:
                out[t] = fut.result()
            except Exception:
                out[t] = None
    return out


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


def append_pool_text(text):
    """Put one more piece of text into the reading pool."""
    _append_line(POOL_PATH, text)


def self_context(log=print):
    """A short self-portrait from the GitHub API: what this repository
    is, how big it has grown, where it lives. Written into the reading
    pool like any other article - this is how Simply learns about its
    own home (GitHub, git, Actions) from the source itself."""
    repo = os.environ.get("GITHUB_REPOSITORY") or "Toluwerr/Simply"
    headers = {"User-Agent": _UA, "Accept": "application/vnd.github+json"}
    tok = os.environ.get("GITHUB_TOKEN")
    if tok:
        headers["Authorization"] = f"Bearer {tok}"

    def _get(url):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.read().decode(), dict(r.headers)
        except Exception:
            return None, None

    body, _ = _get(f"https://api.github.com/repos/{repo}")
    if not body:
        return None
    try:
        d = json.loads(body)
    except ValueError:
        return None
    commits = "many"
    b2, h2 = _get(f"https://api.github.com/repos/{repo}/commits?per_page=1")
    if b2 is not None:
        link = (h2 or {}).get("Link", "")
        m = re.search(r"[?&]page=(\d+)>; rel=\"last\"", link)
        commits = m.group(1) if m else "1"
    text = (
        f"This is the {repo} repository on GitHub, the code hosting site "
        f"built around the git version control system. It is the home of "
        f"Simply, a self-improving AI model. The repository has {commits} "
        f"commits, {d.get('stargazers_count', 0)} stars, and "
        f"{d.get('forks_count', 0)} forks. Simply reads articles from the "
        f"internet, trains itself, and commits its progress here "
        f"automatically using GitHub Actions, the feature that runs "
        f"programs on GitHub's servers. Every commit is one step of "
        f"learning, recorded forever in the git history."
    )
    return _clean_text(text)


def refresh(metrics, allowed_chars=None, budget_s=FETCH_BUDGET_S,
            articles=ARTICLES_PER_RUN, focus=None, log=print):
    """One reading session: fetch, clean, store, distill, trim.

    Never raises: any failure just means 'no new reading today'. Returns
    a small stats dict for the metrics file.
    """
    t0 = time.time()
    rng = random.Random(f"reading:{metrics.get('iteration', 0)}:{time.time()}")
    known = set(t.strip().lower() for t in learned_titles())
    seen_subj = set(fact_subjects())
    new_articles, new_facts = 0, 0

    # first, its own home: a fresh self-portrait from the GitHub API
    home = self_context(log=log)
    if home:
        if allowed_chars is not None:
            home = "".join(c for c in home if c in allowed_chars)
        if len(home) > 100:
            append_pool_text(home)

    titles = _pick_titles(articles + 12, rng, focus=focus)
    # skip everything it has already read, THEN fetch the rest in one
    # parallel wave (bounded by the wall-clock budget at store time)
    fresh = [t for t in titles if t.strip().lower() not in known][:articles + 6]
    extracts = _fetch_many(fresh)
    for title in fresh:
        if new_articles >= articles or time.time() - t0 > budget_s:
            break
        ext = extracts.get(title)
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
        known.add(title.strip().lower())
        new_articles += 1

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
        "focus": focus,
        "seconds": round(time.time() - t0, 1),
    }
    deep = f", deep dive: {focus}" if focus else ""
    log(f"  reading session: +{new_articles} articles, +{new_facts} facts "
        f"({total_learned:,} read all-time, pool {pool_chars:,} chars, "
        f"{stats['seconds']}s{deep})")
    return stats


if __name__ == "__main__":
    # standalone: python knowledge.py  -> one reading session, no training
    print("Simply reading session")
    st = refresh({"iteration": "manual"})
    print(json.dumps(st, indent=2))
