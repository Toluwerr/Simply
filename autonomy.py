"""Simply's free will.

This module is how Simply decides things for itself. It does not make
Simply conscious and it does not pretend to - it gives the loop a real,
auditable decision layer:

  - STRATEGIES: each run, Simply picks HOW to spend its time. Five ways,
    from broad reading to deep dives to bold experiments. It keeps a
    score for every strategy (measured in actual promotions and quiz
    gains) and picks with epsilon-greedy exploration - mostly what
    worked before, sometimes something new, untried ones first.
  - INTERESTS: every deep dive on a topic makes that topic a little
    more interesting to it. Interests bias future picks, so its reading
    develops tastes over time - all of them visible in autonomy.json.
  - GOALS: Simply writes its own measurable goals (PLAN.md), sizes them
    just above its current best, and ticks them off when the metrics
    actually meet them. No goal is declared "done" by vibes - only by
    numbers it committed.
  - DIARY: one freeform sample per run, its own words, clearly labeled
    as unverified. The rest of the loop stays strict; the diary is
    where it gets to just talk.

Every file here is committed to the repo, so anyone can audit what it
chose, why, and whether the choice paid off.
"""
import json
import os
import random
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
AUTONOMY_PATH = os.path.join(ROOT, "autonomy.json")
PLAN_PATH = os.path.join(ROOT, "PLAN.md")
DIARY_PATH = os.path.join(ROOT, "DIARY.md")

EPSILON = 0.25          # chance of exploring a random strategy
MAX_ACTIVE_GOALS = 3

# The action space: how a run can be spent. Descriptions are written
# into PLAN.md so the choice is always readable.
STRATEGIES = {
    "survey": "read broadly - more articles, no fixed direction",
    "deep_dive": "read hard on one topic it picked itself",
    "drill": "double the study lessons for its weakest subjects",
    "create": "spend more time writing and verifying its own examples",
    "experiment": "study with a bolder learning rate for this run",
}

_DIARY_PROMPTS = [
    "Today I learned that",
    "Something I read today:",
    "I want to understand",
    "The most interesting thing I know is",
    "I am getting better at",
]


def load():
    if os.path.exists(AUTONOMY_PATH):
        with open(AUTONOMY_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {
        "strategies": {name: {"tries": 0, "reward": 0.0}
                       for name in STRATEGIES},
        "interests": {},
        "last": None,
        "goals": [],
        "runs": 0,
    }


def save(a):
    with open(AUTONOMY_PATH, "w", encoding="utf-8") as f:
        json.dump(a, f, indent=2)


def _avg(st):
    return st["reward"] / st["tries"] if st["tries"] else 0.0


def choose(a, rng):
    """Epsilon-greedy over strategies; untried ones go first."""
    untried = [n for n in STRATEGIES if a["strategies"][n]["tries"] == 0]
    if untried:
        return rng.choice(untried)
    if rng.random() < EPSILON:
        return rng.choice(list(STRATEGIES))
    return max(STRATEGIES, key=lambda n: _avg(a["strategies"][n]))


def pick_focus(a, rng):
    """Pick a topic to go deep on, biased by its own interests.

    Interests grow every time a topic is studied, so the more it has
    read about something, the more likely it returns to it - with a
    steady chance of starting a brand-new interest instead."""
    import knowledge as KN  # lazily: topic tables live there
    pool = list(KN.TOPIC_LIST) + list(KN.TECH_TOPICS)
    weights = [1.0 + 3.0 * a["interests"].get(t, 0) for t in pool]
    return rng.choices(pool, weights=weights, k=1)[0]


def record_interest(a, topic):
    if topic:
        a["interests"][topic] = a["interests"].get(topic, 0) + 1


def begin_run(a, metrics, rng):
    """Pick a strategy and translate it into concrete settings.

    Returns a plan dict the loop applies: how much to read, what to
    focus on, how many extra study steps to take, whether to push the
    learning rate. The choice is also stored, so end_run() can score
    it against what actually happened."""
    name = choose(a, rng)
    a["last"] = name
    a["runs"] += 1
    plan = {"strategy": name, "desc": STRATEGIES[name],
            "articles": 60, "focus": None, "steps_mult": 1.0,
            "synth_extra": 0, "bold_lr": False}
    if name == "survey":
        plan["articles"] = 90
    elif name == "deep_dive":
        plan["articles"] = 80
        plan["focus"] = pick_focus(a, rng)
        record_interest(a, plan["focus"])
    elif name == "drill":
        plan["steps_mult"] = 1.25
    elif name == "create":
        plan["synth_extra"] = 3
        plan["steps_mult"] = 1.1
    elif name == "experiment":
        plan["bold_lr"] = True
        plan["steps_mult"] = 1.15
    save(a)
    return plan


def score_last(a, know_gain, val_gain, promoted):
    """Reward the last strategy in REAL outcomes: promotions weigh
    most, then quiz/knowledge gains, then loss gains."""
    if not a["last"]:
        return
    reward = (3.0 if promoted else 0.0) + 2.0 * know_gain + 1.0 * val_gain
    st = a["strategies"][a["last"]]
    st["tries"] += 1
    st["reward"] += reward
    save(a)
    return reward


# --- goals ---------------------------------------------------------------

def knowledge_avg(metrics):
    k = metrics.get("knowledge") or {}
    return sum(k.values()) / len(k) if k else 0.0


def _new_goal(metrics, rng):
    """Size one fresh goal just above the current state. Ambition is
    not a mood here - it is a target that the committed metrics must
    actually reach before it counts."""
    kind = rng.choice(["know", "master", "read", "write"])
    k = metrics.get("knowledge") or {}
    if kind == "know":
        target = round(min(0.98, knowledge_avg(metrics) + 0.12), 2)
        return {"kind": "know", "target": target,
                "text": f"lift the knowledge quiz average to {target:.2f}"}
    if kind == "master":
        unmastered = [c for c, v in k.items() if v < 0.999]
        if not unmastered:
            return None
        cat = rng.choice(unmastered)
        return {"kind": "master", "cat": cat,
                "text": f"master the '{cat}' subject (quiz 1.0)"}
    if kind == "read":
        cur = (metrics.get("reading") or {}).get("articles", 0)
        return {"kind": "read", "target": cur + 200,
                "text": f"read {cur + 200:,} articles all-time "
                        f"(now {cur:,})"}
    cur = metrics.get("synthetic_total", 0)
    return {"kind": "write", "target": cur + 60,
            "text": f"write {cur + 60} verified examples of its own "
                    f"(now {cur})"}


def _goal_met(g, metrics):
    k = metrics.get("knowledge") or {}
    if g["kind"] == "know":
        return knowledge_avg(metrics) >= g["target"]
    if g["kind"] == "master":
        return k.get(g["cat"], 0.0) >= 0.999
    if g["kind"] == "read":
        return (metrics.get("reading") or {}).get("articles", 0) >= g["target"]
    if g["kind"] == "write":
        return metrics.get("synthetic_total", 0) >= g["target"]
    return False


def update_goals(a, metrics):
    """Tick met goals, refill to 3 active. Returns list of just-met."""
    met = []
    for g in a["goals"]:
        if g.get("done") is None and _goal_met(g, metrics):
            g["done"] = metrics.get("iteration", 0)
            met.append(g)
    a["goals"] = [g for g in a["goals"] if g.get("done") is None]
    rng = random.Random(f"goals:{metrics.get('iteration', 0)}")
    # Some goal kinds can be unavailable (e.g. every subject mastered),
    # so retry a few times before giving up on the refill.
    tries = 0
    while len(a["goals"]) < MAX_ACTIVE_GOALS and tries < 12:
        tries += 1
        g = _new_goal(metrics, rng)
        if g is None or g in a["goals"]:
            continue
        a["goals"].append(g)
    save(a)
    return met


# --- artifacts ------------------------------------------------------------

def write_plan(a, metrics, plan, met=None):
    """Regenerate PLAN.md: its current drive, its scoreboard, its
    goals with honest status."""
    lines = [
        "# Simply's own plan",
        "",
        "_Written by Simply itself, every run. Goals are only checked",
        "off when the committed metrics actually meet them._",
        "",
        f"**This run's choice**: `{plan['strategy']}` - {plan['desc']}.",
    ]
    if plan.get("focus"):
        lines.append(f"**Topic it picked to go deep on**: "
                     f"{plan['focus']} (its own interest, "
                     f"studied {a['interests'][plan['focus']]}x).")
    lines += ["", "## Goals it set for itself", ""]
    if met:
        for g in met:
            lines.append(f"- [x] {g['text']} - DONE (iteration "
                         f"{g.get('done')})")
    active = [g for g in a["goals"] if g.get("done") is None]
    for g in active:
        lines.append(f"- [ ] {g['text']}")
    if not active:
        lines.append("- (all current goals met - new ones next run)")
    lines += ["", "## How it chooses", "",
              "Each run it picks a strategy; rewards come only from real",
              "outcomes (promotions, quiz gains, loss gains). Explore",
              f"{int(EPSILON * 100)}% of the time, exploit the rest.", ""]
    lines.append("| strategy | tries | avg reward |")
    lines.append("|---|---|---|")
    for name in STRATEGIES:
        st = a["strategies"][name]
        avg = f"{_avg(st):.2f}" if st["tries"] else "-"
        lines.append(f"| {name} | {st['tries']} | {avg} |")
    interests = sorted(a["interests"].items(), key=lambda kv: -kv[1])[:6]
    if interests:
        lines += ["", "## What it keeps coming back to", ""]
        lines += [f"- {t} ({n}x)" for t, n in interests]
    lines.append("")
    with open(PLAN_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_diary(model, ds, device, plan, temperature=0.9):
    """Two freeform lines in its own words. Explicitly NOT verified -
    this is the one unfiltered window into the model, and it is labeled
    as such right in the file. Kept to the newest 30 entries."""
    import torch
    lines = []
    rng = random.Random(f"diary:{plan['strategy']}:{time.time()}")
    model.eval()
    for prompt in rng.sample(_DIARY_PROMPTS, 2):
        ids = torch.tensor([ds.encode(prompt + " ")], device=device)
        with torch.no_grad():
            out = model.generate(ids, 60, temperature=temperature, top_k=30)
        text = ds.decode(out[0].tolist())[len(prompt) + 1:]
        text = text.split("\n")[0].strip()
        if text:
            lines.append(f"> {prompt} {text}")
    if not lines:
        return
    header = (f"### {time.strftime('%Y-%m-%d %H:%M')} - "
              f"after a '{plan['strategy']}' run\n\n"
              "_Freeform output from the model itself - not verified, "
              "not graded, just its own words._\n\n")
    existing = ""
    if os.path.exists(DIARY_PATH):
        with open(DIARY_PATH, encoding="utf-8") as f:
            existing = f.read()
    if existing:
        # keep the newest 30 entries (oldest released)
        parts = existing.split("\n### ")
        if len(parts) > 30:
            existing = "\n### " + "\n### ".join(
                p for p in parts[len(parts) - 30:] if p.strip())
    with open(DIARY_PATH, "w", encoding="utf-8") as f:
        f.write(existing + header + "\n".join(lines) + "\n\n")
