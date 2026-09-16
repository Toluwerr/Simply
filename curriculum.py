"""Curriculum engine for Simply: infinite provable lessons per category.

gen_lesson(cat, rng) returns (question, answer) pairs that are correct
BY CONSTRUCTION - they are built from the ground-truth tables in
world_tables.py / seed_data.py, never invented. check(q, a) re-proves
any candidate answer for the new question templates, so the quiz and
the self-written-data verifier share one source of truth.
"""
import random
import re

import seed_data as SD
import world_tables as WT

# --- roman numerals (lowercase: uppercase X is not in the charset) --------
_ROMAN_PAIRS = [(1000, "m"), (900, "cm"), (500, "d"), (400, "cd"),
                (100, "c"), (90, "xc"), (50, "l"), (40, "xl"),
                (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")]
_ROMAN_VALUES = {c: v for v, c in _ROMAN_PAIRS}


def to_roman(n):
    if not 1 <= n <= 3000:
        raise ValueError(n)
    out = []
    for v, sym in _ROMAN_PAIRS:
        while n >= v:
            out.append(sym)
            n -= v
    return "".join(out)


def from_roman(s):
    total = 0
    for i, ch in enumerate(s):
        v = _ROMAN_VALUES[ch]
        if i + 1 < len(s) and _ROMAN_VALUES[s[i + 1]] > v:
            total -= v
        else:
            total += v
    return total


_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven",
         "eight", "nine", "ten", "eleven", "twelve", "thirteen",
         "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
         "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty",
         "sixty", "seventy", "eighty", "ninety"]


def num2words(n):
    """English spelling of 0-100."""
    if not 0 <= n <= 100:
        raise ValueError(n)
    if n < 20:
        return _ONES[n]
    if n == 100:
        return "one hundred"
    t, r = divmod(n, 10)
    return _TENS[t] + ("-" + _ONES[r] if r else "")


# --- shared word pools (for letters / spelling / first-last) --------------
TRIPLE = re.compile(r"([a-zA-Z])\1{2,}")


def _safe_word(w):
    return (w.isalpha() and len(w) >= 3
            and not TRIPLE.search(w)
            and "x" not in w.lower())


def table_words():
    """Every provable word across all ground-truth tables."""
    ws = set()
    tables = [WT.CAPITALS, WT.CURRENCIES, WT.STATES, WT.CONTINENTS]
    for table in tables:
        for k, v in table.items():
            for token in f"{k} {v}".split():
                token = token.strip("',-.").lower()
                if token.isalpha():
                    ws.add(token)
    for name, _sym in WT.ELEMENTS:
        ws.add(name)
    for planet in WT.PLANETS:
        ws.add(planet.lower())
    for month in WT.MONTH_DAYS:
        ws.add(month.lower())
    for table in (SD.ANIMAL_SOUNDS, SD.DEFINITIONS):
        for k, v in table.items():
            for token in f"{k} {v}".split():
                token = token.strip("',-.").lower()
                if token.isalpha():
                    ws.add(token)
    for a, b in SD.OPPOSITES:
        for w in (a, b):
            if w.isalpha():
                ws.add(w.lower())
    ws.update(w for w in SD.SPELL_WORDS if _safe_word(w))
    ws.update(w for w in SD.LETTER_WORDS if _safe_word(w))
    return {w for w in ws if _safe_word(w)}


WORD_POOL = sorted(table_words())
SPELL_POOL = sorted(w for w in WORD_POOL if len(w) <= 8)

# --- question templates for the new categories -----------------------------
CURRENCY_Q = re.compile(r"^What currency does (.+) use\?$")
CONTINENT_Q = re.compile(r"^What continent is (.+) on\?$")
SYM_FOR_Q = re.compile(r"^What is the chemical symbol for (.+)\?$")
ELEMENT_WITH_Q = re.compile(r"^Which element has the chemical symbol (.+)\?$")
STATE_CAP_Q = re.compile(r"^What is the capital of the US state (.+)\?$")
PLANET_Q = re.compile(
    r"^Which planet is (first|second|third|fourth|fifth|sixth|seventh|"
    r"eighth) from the sun\?$")
MONTH_Q = re.compile(r"^How many days are in (.+)\?$")
ROMAN_TO_Q = re.compile(r"^What is the Roman numeral for (\d+)\?$")
ROMAN_FROM_Q = re.compile(r"^What number is the Roman numeral ([a-z]+)\?$")

_ORD_INDEX = {o: i for i, o in enumerate(WT.ORDINALS)}
_UNITS_DICT = dict(WT.UNITS)
_RECORDS_DICT = dict(WT.RECORDS)


def check(q, a):
    """Ground-truth check for the NEW templates. Old categories stay in
    self_improve.template_check; this function is the delegated fallback."""
    # Exact-table questions first, so a units/records question can never
    # be swallowed by an overlapping regex (e.g. "How many days are in a
    # week?" must hit the units table, not the months template).
    if q in _UNITS_DICT:
        return _UNITS_DICT[q] in a
    if q in _RECORDS_DICT:
        return _RECORDS_DICT[q] in a
    m = CURRENCY_Q.match(q)
    if m:
        cur = WT.CURRENCIES.get(m.group(1))
        return cur is not None and (
            f"currency of {m.group(1).lower()} is the {cur}" in a.lower())
    m = CONTINENT_Q.match(q)
    if m:
        cont = WT.CONTINENTS.get(m.group(1))
        return cont is not None and f"is on the continent of {cont}" in a
    m = SYM_FOR_Q.match(q)
    if m:
        sym = dict(WT.ELEMENTS).get(m.group(1).lower())
        return sym is not None and f"chemical symbol for {m.group(1).lower()} is {sym}" in a
    m = ELEMENT_WITH_Q.match(q)
    if m:
        name = dict((s, n) for n, s in WT.ELEMENTS).get(m.group(1))
        return name is not None and f"is the chemical symbol for {name}" in a
    m = STATE_CAP_Q.match(q)
    if m:
        cap = WT.STATES.get(m.group(1))
        return cap is not None and f"{cap} is the capital of {m.group(1)}" in a
    m = PLANET_Q.match(q)
    if m:
        i = _ORD_INDEX[m.group(1)]
        return f"{WT.PLANETS[i]} is the {WT.ORDINALS[i]} planet from the sun" in a
    m = MONTH_Q.match(q)
    if m:
        n = WT.MONTH_DAYS.get(m.group(1))
        return n is not None and f"{m.group(1)} has {n} days" in a
    m = ROMAN_TO_Q.match(q)
    if m:
        n = int(m.group(1))
        if not 1 <= n <= 3000:
            return False
        return f"numeral for {n} is {to_roman(n)}" in a.lower()
    m = ROMAN_FROM_Q.match(q)
    if m:
        r = m.group(1).lower()
        try:
            n = from_roman(r)
        except (KeyError, ValueError):
            return False
        return f"is the number {n}" in a
    return False


# --- lesson generators (correct by construction) ---------------------------
def _pick(rng, seq):
    return seq[rng.randrange(len(seq))]


def _gen_arithmetic(rng):
    tier = rng.randrange(3)
    op = _pick(rng, ["plus", "minus", "times"])
    if tier == 0:
        x, y = rng.randint(1, 20), rng.randint(1, 20)
        if op == "times":
            x, y = rng.randint(2, 9), rng.randint(2, 9)
    elif tier == 1:
        x, y = rng.randint(10, 99), rng.randint(10, 99)
        if op == "times":
            x, y = rng.randint(3, 12), rng.randint(3, 12)
    else:
        x, y = rng.randint(100, 999), rng.randint(10, 999)
        if op == "times":
            x, y = rng.randint(11, 15), rng.randint(11, 19)
    if op == "minus" and y > x:
        x, y = y, x
    r = x + y if op == "plus" else x - y if op == "minus" else x * y
    return f"What is {x} {op} {y}?", f"{x} {op} {y} is {r}."


def _gen_compare(rng):
    tier = rng.randrange(3)
    lo, hi = [(1, 20), (10, 99), (100, 999)][tier]
    x, y = rng.randint(lo, hi), rng.randint(lo, hi)
    if x == y:
        y += 1
    if rng.random() < 0.5:
        return (f"Which is bigger, {x} or {y}?",
                f"{max(x, y)} is bigger than {min(x, y)}.")
    return (f"Is {x} bigger than {y}?",
            (f"Yes, {x} is bigger than {y}." if x > y
             else f"No, {y} is bigger than {x}."))


def _gen_sort(rng):
    tier = rng.randrange(2)
    lo, hi = (1, 30) if tier == 0 else (10, 999)
    xs = rng.sample(range(lo, hi), 3)
    s = sorted(xs)
    return (f"How do you sort {xs[0]}, {xs[1]}, and {xs[2]}?",
            f"In order, they are {s[0]} {s[1]} {s[2]}.")


def _gen_double_half(rng):
    x = rng.randint(2, 60)
    if rng.random() < 0.5:
        return f"What is double {x}?", f"Double {x} is {2 * x}."
    if x % 2:
        x += 1
    return f"What is half of {x}?", f"Half of {x} is {x // 2}."


def _gen_count(rng):
    word = _pick(rng, ["ones", "twos", "fives", "tens"])
    step = {"ones": 1, "twos": 2, "fives": 5, "tens": 10}[word]
    lo = rng.randrange(step, 40, step)
    n_terms = rng.randint(3, 6)
    hi = lo + step * (n_terms - 1)
    seq = " ".join(str(v) for v in range(lo, hi + 1, step))
    return (f"Can you count by {word} from {lo} to {hi}?",
            f"Counting by {word} gives {seq}.")


def _gen_numwords(rng):
    n = rng.randint(1, 100)
    return (f"How do you write {n} in words?",
            f"{n} in words is {num2words(n)}.")


def _gen_roman(rng):
    tier = rng.randrange(3)
    n = _pick(rng, [rng.randint(1, 20), rng.randint(21, 100),
                    rng.randint(101, 1000)])
    r = to_roman(n)
    if rng.random() < 0.5:
        return (f"What is the Roman numeral for {n}?",
                f"The Roman numeral for {n} is {r}.")
    return (f"What number is the Roman numeral {r}?",
            f"The Roman numeral {r} is the number {n}.")


def _gen_letters(rng):
    w = _pick(rng, WORD_POOL)
    return (f"How many letters does the word '{w}' have?",
            f"The word '{w}' has {len(w)} letters.")


def _gen_spelling(rng):
    w = _pick(rng, SPELL_POOL)
    return (f"How do you spell '{w}'?",
            f"You spell it '{w}': " + "-".join(w.upper()) + ".")


def _gen_first_last(rng):
    w = _pick(rng, WORD_POOL)
    if rng.random() < 0.5:
        return (f"What is the first letter of '{w}'?",
                f"The first letter of '{w}' is {w[0].upper()}.")
    return (f"What is the last letter of '{w}'?",
            f"The last letter of '{w}' is {w[-1].upper()}.")


def _gen_capitals(rng):
    c, cap = _pick(rng, list(WT.CAPITALS.items()))
    return f"What is the capital of {c}?", f"The capital of {c} is {cap}."


def _gen_states(rng):
    s, cap = _pick(rng, list(WT.STATES.items()))
    return (f"What is the capital of the US state {s}?",
            f"{cap} is the capital of {s}.")


def _gen_currencies(rng):
    c, cur = _pick(rng, list(WT.CURRENCIES.items()))
    return (f"What currency does {c} use?",
            f"The currency of {c} is the {cur}.")


def _gen_continents(rng):
    c, cont = _pick(rng, list(WT.CONTINENTS.items()))
    return (f"What continent is {c} on?",
            f"{c} is on the continent of {cont}.")


def _gen_elements(rng):
    name, sym = _pick(rng, WT.ELEMENTS)
    if rng.random() < 0.5:
        return (f"What is the chemical symbol for {name}?",
                f"The chemical symbol for {name} is {sym}.")
    return (f"Which element has the chemical symbol {sym}?",
            f"{sym} is the chemical symbol for {name}.")


def _gen_planets(rng):
    i = rng.randrange(len(WT.PLANETS))
    return (f"Which planet is {WT.ORDINALS[i]} from the sun?",
            f"{WT.PLANETS[i]} is the {WT.ORDINALS[i]} planet from the sun.")


def _gen_months(rng):
    m = _pick(rng, list(WT.MONTH_DAYS))
    n = WT.MONTH_DAYS[m]
    a = f"{m} has {n} days"
    if m == "February":
        a += ", and 29 in a leap year"
    return f"How many days are in {m}?", a + "."


def _gen_units(rng):
    return _pick(rng, WT.UNITS)


def _gen_records(rng):
    return _pick(rng, WT.RECORDS)


def _gen_animals(rng):
    name, sound = _pick(rng, list(SD.ANIMAL_SOUNDS.items()))
    art = "an" if name[0].lower() in "aeiou" else "a"
    return (f"What sound does {art} {name} make?",
            f"A {name} says '{sound}'.")


def _gen_opposites(rng):
    w, opp = _pick(rng, SD.OPPOSITES)
    return (f"What is the opposite of '{w}'?",
            f"The opposite of '{w}' is '{opp}'.")


def _gen_definitions(rng):
    w, meaning = _pick(rng, list(SD.DEFINITIONS.items()))
    return f"What does '{w}' mean?", f"'{w}' means {meaning}."


GENERATORS = {
    "arithmetic": _gen_arithmetic,
    "compare": _gen_compare,
    "sort": _gen_sort,
    "double_half": _gen_double_half,
    "count": _gen_count,
    "numwords": _gen_numwords,
    "roman": _gen_roman,
    "letters": _gen_letters,
    "spelling": _gen_spelling,
    "first_last": _gen_first_last,
    "capitals": _gen_capitals,
    "states": _gen_states,
    "currencies": _gen_currencies,
    "continents": _gen_continents,
    "elements": _gen_elements,
    "planets": _gen_planets,
    "months": _gen_months,
    "units": _gen_units,
    "records": _gen_records,
    "animals": _gen_animals,
    "opposites": _gen_opposites,
    "definitions": _gen_definitions,
}

CATEGORIES = list(GENERATORS)


def gen_lesson(cat, rng):
    """One provably correct (question, answer) pair for a category."""
    return GENERATORS[cat](rng)
