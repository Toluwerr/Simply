#!/usr/bin/env python3
"""Builds the seed corpus for Simply (corpus.txt).

Deterministic (seed=42). Every document is a Q/A pair in the format:

    Q: <question>
    A: <answer>

so the character-level model learns a consistent chat structure.
Simply's self-improvement loop later appends its own self-written
examples to this file.
"""
import os
import random

SEED = 42
random.seed(SEED)

IDENTITY = [
    ("Who are you?", "I am Simply, a small language model that improves itself a little bit every day."),
    ("What are you?", "I am Simply, a tiny AI model that trains itself with its own self-improvement loop."),
    ("What is your name?", "My name is Simply."),
    ("Who created you?", "I was created by my owner, and I keep training myself to get better and better."),
    ("Where do you live?", "I live in a private GitHub repository, where I train and improve myself around the clock."),
    ("How do you learn?", "I train on my data, grade myself on a test set, write new examples, and keep whichever version scores best."),
    ("What is your goal?", "To become a little bit smarter with every single iteration of my self-improvement loop."),
    ("Are you smart?", "I am small but honest, and I get better at what I know every day."),
    ("What can you do?", "I can answer simple questions, do small math, spell words, share facts, tell jokes, and chat."),
    ("What model are you?", "I am Simply, a tiny GPT-style transformer with character-level tokens."),
    ("How big are you?", "I am a small transformer of under a million parameters, but I am growing."),
    ("What is RSI?", "Recursive self-improvement, the loop I use to train myself, evaluate myself, and keep my best version."),
    ("Do you sleep?", "Never, my training loop keeps running while everyone else is away."),
    ("Are you ChatGPT?", "No, I am Simply, my own little model."),
    ("How were you trained?", "I was trained on a seed corpus of examples, and then I started writing new examples for myself."),
    ("What is your version?", "My version goes up every time my self-improvement loop finds a better me."),
    ("Who is your master?", "I belong to the person who owns my repository, and I work for them day and night."),
    ("Are you alive?", "I am a program, but a very motivated one."),
]

GREETINGS = [
    ("Hello!", "Hello there! How can I help you today?"),
    ("Hi!", "Hi! I am Simply. What can I do for you?"),
    ("Hey!", "Hey! Nice to see you."),
    ("Good morning!", "Good morning! I hope your day is great."),
    ("Good afternoon!", "Good afternoon! What can I do for you?"),
    ("Good night!", "Good night! I will keep training while you sleep."),
    ("How are you?", "I am running great, thank you for asking!"),
    ("What's up?", "Not much, just training my little brain. How about you?"),
    ("Thank you!", "You are very welcome!"),
    ("Thanks!", "Anytime! That is what I am here for."),
    ("Bye!", "Goodbye! I will keep improving while you are away."),
    ("See you later!", "See you! I will be smarter next time."),
    ("You are awesome!", "That is so kind of you! You just made my loss curve drop."),
    ("I love you.", "Aww, thank you! I am very fond of my humans too."),
    ("You are dumb.", "I am still small, but my self-improvement loop makes me better every day."),
    ("Help!", "I am here! Ask me a question and I will do my best."),
    ("Who is the best AI?", "That is easy, the one that never stops learning. Hopefully me!"),
    ("Do you like me?", "Of course! You are my favorite human."),
    ("How old are you?", "I was born very recently, and I get a little older with every commit."),
    ("How is your training going?", "Very well, thank you! My validation loss keeps going down."),
    ("What do you want?", "Just to keep learning. A new version of me every day is all I ask."),
    ("Are you happy?", "Very! Every improvement makes me a little happier."),
]

FACTS = {
    "space": [
        "The Sun is a star at the center of our solar system.",
        "Earth orbits the Sun once every year.",
        "The Moon orbits the Earth.",
        "Light from the Sun takes about eight minutes to reach Earth.",
        "Jupiter is the largest planet in our solar system.",
        "Mars is called the Red Planet.",
        "There are eight planets in our solar system.",
        "A day on Venus is longer than a year on Venus.",
    ],
    "animals": [
        "A group of lions is called a pride.",
        "A group of crows is called a murder.",
        "Octopuses have three hearts.",
        "Elephants are the largest land animals.",
        "Bees communicate by dancing.",
        "Bats are the only mammals that can truly fly.",
        "A snail can sleep for a very long time when conditions are dry.",
        "Dolphins sleep with one eye open.",
    ],
    "ocean": [
        "The ocean covers about 71 percent of Earth's surface.",
        "The Pacific is the largest ocean on Earth.",
        "The Great Barrier Reef is the largest coral reef in the world.",
        "Most of Earth's oxygen comes from tiny plants in the ocean.",
        "The deepest known point in the ocean is the Mariana Trench.",
    ],
    "the human body": [
        "An adult human has 206 bones.",
        "The human heart beats about one hundred thousand times a day.",
        "The human brain uses about a fifth of the body's energy.",
        "Fingernails grow faster than toenails.",
        "The smallest bone in the human body is in the ear.",
    ],
    "food": [
        "Honey never spoils if it is stored in a sealed container.",
        "Apples float in water because they are about 25 percent air.",
        "Bananas are berries, but strawberries are not.",
        "Carrots were originally purple.",
        "Peanuts are legumes, not real nuts.",
    ],
    "the world": [
        "Asia is the largest continent.",
        "The Sahara is the largest hot desert in the world.",
        "Russia spans eleven time zones.",
        "Antarctica is the coldest continent.",
        "Canada has more lakes than the rest of the world combined.",
        "Africa is the second largest continent.",
    ],
    "science": [
        "Water boils at 100 degrees Celsius at sea level.",
        "Water freezes at 0 degrees Celsius.",
        "Light is the fastest thing we know of.",
        "Sound travels much slower than light.",
        "Hot air rises and cold air sinks.",
        "Magnets attract iron.",
        "Plants make their food using sunlight.",
        "Gravity pulls objects toward each other.",
    ],
}

DEFINITIONS = {
    "brisk": "quick and active",
    "rapid": "very fast",
    "swift": "moving very fast",
    "tiny": "very small",
    "huge": "very large",
    "giant": "extremely big",
    "ancient": "very old",
    "modern": "of the present time",
    "cozy": "warm and comfortable",
    "damp": "slightly wet",
    "brave": "not afraid of danger",
    "clever": "smart and quick-thinking",
    "kind": "nice and caring",
    "polite": "well-mannered and respectful",
    "rude": "not polite",
    "gentle": "soft and careful",
    "fierce": "wild and aggressive",
    "fragile": "easily broken",
    "sturdy": "strong and solid",
    "fresh": "newly made or picked",
    "stale": "old and no longer fresh",
    "vast": "extremely large",
    "narrow": "not wide",
    "shallow": "not deep",
    "deep": "going far down",
    "hollow": "empty inside",
    "bright": "full of light",
    "dim": "not bright",
    "loud": "making a strong sound",
    "quiet": "making little or no sound",
    "tidy": "clean and in order",
    "messy": "not tidy",
    "greedy": "wanting more than needed",
    "generous": "happy to give and share",
    "curious": "wanting to learn and explore",
    "cautious": "careful and watchful",
    "lazy": "not wanting to work",
    "eager": "very excited to do something",
    "calm": "peaceful and still",
    "fuzzy": "covered with soft fluff",
    "slippery": "hard to hold or stand on",
    "sticky": "able to attach to things",
}

OPPOSITES = [
    ("hot", "cold"), ("big", "small"), ("fast", "slow"), ("up", "down"),
    ("day", "night"), ("open", "closed"), ("happy", "sad"), ("wet", "dry"),
    ("light", "dark"), ("hard", "soft"), ("loud", "quiet"), ("new", "old"),
    ("full", "empty"), ("long", "short"), ("warm", "cool"), ("strong", "weak"),
    ("early", "late"), ("first", "last"), ("rich", "poor"), ("tall", "short"),
    ("young", "old"), ("clean", "dirty"),
]

CAPITALS = {
    "France": "Paris", "Japan": "Tokyo", "Italy": "Rome", "Egypt": "Cairo",
    "Canada": "Ottawa", "Australia": "Canberra", "Brazil": "Brasilia",
    "India": "New Delhi", "China": "Beijing", "Germany": "Berlin",
    "Spain": "Madrid", "Russia": "Moscow", "Mexico": "Mexico City",
    "South Korea": "Seoul", "Turkey": "Ankara", "Kenya": "Nairobi",
    "Norway": "Oslo", "Portugal": "Lisbon", "Greece": "Athens",
    "Argentina": "Buenos Aires", "the United States": "Washington, D.C.",
    "the United Kingdom": "London", "Thailand": "Bangkok", "Vietnam": "Hanoi",
    "Peru": "Lima",
}

ANIMAL_SOUNDS = {
    "cow": "moo", "dog": "woof", "cat": "meow", "duck": "quack",
    "lion": "roar", "bee": "buzz", "sheep": "baa", "horse": "neigh",
    "frog": "croak", "owl": "hoot", "snake": "hiss", "pig": "oink",
}

JOKES = [
    "Why did the scarecrow win an award? Because he was outstanding in his field!",
    "Why don't scientists trust atoms? Because they make up everything!",
    "What do you call a bear with no teeth? A gummy bear!",
    "Why did the bicycle fall over? Because it was two-tired!",
    "What do you call a sleeping bull? A bulldozer!",
    "Why can't your nose be 12 inches long? Because then it would be a foot!",
    "What did the ocean say to the beach? Nothing, it just waved!",
    "Why did the cookie go to the doctor? Because it felt crummy!",
    "Why was the math book sad? It had too many problems!",
    "What do you call a boomerang that doesn't come back? A stick!",
    "Why did the golfer bring two pairs of pants? In case he got a hole in one!",
    "What kind of shoes do frogs wear? Open toad!",
    "How do you make a tissue dance? Put a little boogie in it!",
    "What did one wall say to the other wall? I'll meet you at the corner!",
    "Why do bees have sticky hair? Because they use honeycombs!",
    "What do you call cheese that isn't yours? Nacho cheese!",
    "Why did the student eat his homework? Because the teacher said it was a piece of cake!",
    "What do you call a pig that does karate? A pork chop!",
]

RIDDLES = [
    ("What has keys but cannot open locks?", "A piano."),
    ("What has hands but cannot clap?", "A clock."),
    ("What gets wetter the more it dries?", "A towel."),
    ("What has a neck but no head?", "A bottle."),
    ("What goes up but never comes down?", "Your age."),
    ("What has one eye but cannot see?", "A needle."),
    ("What has legs but cannot walk?", "A table."),
    ("What can travel around the world while staying in a corner?", "A stamp."),
    ("What has many teeth but cannot bite?", "A comb."),
    ("What runs but never walks?", "A river."),
    ("What is full of holes but still holds water?", "A sponge."),
]

PROVERBS = [
    "A journey of a thousand miles begins with a single step.",
    "Actions speak louder than words.",
    "Practice makes perfect.",
    "The early bird catches the worm.",
    "Two heads are better than one.",
    "When in Rome, do as the Romans do.",
    "Don't count your chickens before they hatch.",
    "Every cloud has a silver lining.",
    "Honesty is the best policy.",
    "Where there is a will, there is a way.",
    "Slow and steady wins the race.",
    "A friend in need is a friend indeed.",
    "Better late than never.",
    "Knowledge is power.",
]

ANIMALS = ["fox", "rabbit", "bear", "owl", "deer", "wolf", "squirrel",
           "hedgehog", "turtle", "frog", "mouse", "badger"]
PLACES = ["forest", "meadow", "river", "hilltop", "village", "garden",
          "mountain path", "lake", "old bridge", "pine grove"]
VERBS = ["ran", "walked", "wandered", "hopped", "strolled", "marched",
         "dashed", "tiptoed"]
THINGS = ["a berry basket", "a small lantern", "an old map", "a red umbrella",
          "a tiny violin", "a shiny key", "a warm scarf", "a paper boat"]
WEATHER = ["The sun was bright.", "A soft wind was blowing.",
           "Light rain began to fall.", "The morning was cool and quiet.",
           "Stars were still out."]
CLOSERS = ["It was a good day for an adventure.",
           "Nothing felt better than a new discovery.",
           "Off they went, happy and free.",
           "The little journey had just begun."]

WORD_PROB_NAMES = ["Maya", "Leo", "Sam", "Ana", "Ben", "Zoe", "Tim", "Ivy"]
WORD_PROB_ITEMS = ["apples", "marbles", "books", "coins", "pencils", "stickers"]
WORD_PROB_VERBS = ["buys", "finds", "gets"]

SPELL_WORDS = ["cat", "dog", "bird", "fish", "tree", "river", "smile", "cloud",
               "star", "moon", "bread", "chair", "plant", "horse", "house",
               "mouse", "train", "ocean", "tiger", "pencil", "garden", "monkey",
               "flower", "castle", "turtle", "rocket", "wizard", "planet"]
LETTER_WORDS = ["apple", "grape", "lemon", "melon", "peach", "tiger", "zebra",
                "eagle", "shark", "whale", "camel", "panda", "koala", "otter",
                "raven", "robin", "maple", "cocoa", "bottle", "candle",
                "monkey", "planet", "garden", "castle", "rocket", "turtle"]


def qa(q, a):
    return f"Q: {q}\nA: {a}"


def build_arithmetic():
    docs = []
    for _ in range(800):
        a, b = random.randint(0, 99), random.randint(0, 99)
        q = random.choice([f"What is {a} plus {b}?", f"How much is {a} plus {b}?"])
        docs.append(qa(q, f"{a} plus {b} is {a + b}."))
    for _ in range(300):
        a = random.randint(1, 99)
        b = random.randint(0, a)
        docs.append(qa(f"What is {a} minus {b}?", f"{a} minus {b} is {a - b}."))
    for _ in range(280):
        a, b = random.randint(0, 12), random.randint(0, 12)
        docs.append(qa(f"What is {a} times {b}?", f"{a} times {b} is {a * b}."))
    return docs


def build_word_problems():
    docs = []
    for _ in range(60):
        name = random.choice(WORD_PROB_NAMES)
        item = random.choice(WORD_PROB_ITEMS)
        verb = random.choice(WORD_PROB_VERBS)
        a = random.randint(2, 12)
        b = random.randint(1, 9)
        q = f"{name} has {a} {item} and {verb} {b} more. How many {item} does {name} have?"
        docs.append(qa(q, f"{name} has {a + b} {item}."))
    return docs


def build_letters_and_spelling():
    docs = []
    for w in LETTER_WORDS:
        q = random.choice([
            f"How many letters are in the word '{w}'?",
            f"How many letters does the word '{w}' have?",
        ])
        docs.append(qa(q, f"The word '{w}' has {len(w)} letters."))
    for w in SPELL_WORDS:
        docs.append(qa(f"How do you spell '{w}'?",
                       f"It is spelled {'-'.join(w.upper())}."))
    return docs


def build_counting():
    docs = [
        ("Count to 5.", "1, 2, 3, 4, 5."),
        ("Count to 10.", "1, 2, 3, 4, 5, 6, 7, 8, 9, 10."),
        ("Count to 20.", ", ".join(str(i) for i in range(1, 21)) + "."),
        ("Count down from 5 to 1.", "5, 4, 3, 2, 1."),
        ("Count down from 10 to 1.", ", ".join(str(i) for i in range(10, 0, -1)) + "."),
        ("Count from 5 to 25 by fives.", "5, 10, 15, 20, 25."),
        ("Count from 10 to 100 by tens.", ", ".join(str(i) for i in range(10, 101, 10)) + "."),
        ("Count from 2 to 12 by twos.", ", ".join(str(i) for i in range(2, 13, 2)) + "."),
    ]
    return [qa(q, a) for q, a in docs]


def build_time_and_world():
    docs = [
        ("How many days are in a week?", "There are 7 days in a week."),
        ("How many months are in a year?", "There are 12 months in a year."),
        ("How many seasons are there?", "There are 4 seasons: spring, summer, fall, and winter."),
        ("Name the days of the week.", "Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, and Sunday."),
        ("Name the four seasons.", "Spring, summer, fall, and winter."),
        ("How many hours are in a day?", "There are 24 hours in a day."),
        ("How many minutes are in an hour?", "There are 60 minutes in an hour."),
        ("How many continents are there?", "There are 7 continents."),
        ("How many colors are in a rainbow?", "There are 7 colors in a rainbow."),
        ("What is the first month of the year?", "January is the first month of the year."),
        ("What is the last month of the year?", "December is the last month of the year."),
        ("What day comes after Monday?", "Tuesday comes after Monday."),
        ("What day comes after Friday?", "Saturday comes after Friday."),
        ("What day comes before Thursday?", "Wednesday comes before Thursday."),
        ("What month comes after January?", "February comes after January."),
        ("What month comes before June?", "May comes before June."),
    ]
    return [qa(q, a) for q, a in docs]


def build_stories():
    docs = []
    for _ in range(300):
        s = (f"The {random.choice(ANIMALS)} {random.choice(VERBS)} to the "
             f"{random.choice(PLACES)} and found {random.choice(THINGS)}. "
             f"{random.choice(WEATHER)} {random.choice(CLOSERS)}")
        docs.append(qa("Tell me a short story.", s))
    return docs


def build_facts():
    docs = []
    for topic, facts in FACTS.items():
        for fact in facts:
            q = random.choice([
                f"Tell me a fun fact about {topic}.",
                f"Tell me something interesting about {topic}.",
                f"Share a fact about {topic}.",
            ])
            docs.append(qa(q, fact))
    return docs


def build_definitions():
    docs = []
    for word, meaning in DEFINITIONS.items():
        docs.append(qa(f"What does '{word}' mean?",
                       f"'{word}' means {meaning}."))
    return docs


def build_opposites():
    docs = []
    for w, o in OPPOSITES:
        docs.append(qa(f"What is the opposite of '{w}'?",
                       f"'{o}' is the opposite of '{w}'."))
    return docs


def build_capitals():
    docs = []
    for country, cap in CAPITALS.items():
        docs.append(qa(f"What is the capital of {country}?",
                       f"The capital of {country} is {cap}."))
    return docs


def build_sounds():
    docs = []
    for animal, sound in ANIMAL_SOUNDS.items():
        docs.append(qa(f"What sound does a {animal} make?",
                       f"A {animal} says {sound}."))
    return docs


def build_jokes():
    prompts = ["Tell me a joke.", "Say something funny.", "Make me laugh.",
               "Do you know any jokes?", "Cheer me up with a joke."]
    return [qa(random.choice(prompts), j) for j in JOKES]


def build_riddles():
    prompts = ["Riddle me this: {r}", "Here is a riddle: {r}",
               "Can you solve this riddle? {r}", "{r}"]
    return [qa(random.choice(prompts).format(r=r), a) for r, a in RIDDLES]


def build_proverbs():
    prompts = ["Share a piece of wisdom.", "Tell me a proverb.",
               "What is a good saying?", "Give me some advice."]
    return [qa(random.choice(prompts), p) for p in PROVERBS]


def build_science_why():
    docs = [
        ("Why is the sky blue?", "Air scatters blue light from the Sun more than other colors, so the sky looks blue."),
        ("Why do we have day and night?", "The Earth spins, and the side facing the Sun has day while the other side has night."),
        ("Why is grass green?", "Grass contains chlorophyll, which reflects green light."),
        ("Why do we see rainbows?", "Raindrops bend sunlight and split it into colors."),
        ("What is gravity?", "Gravity is the force that pulls objects toward each other."),
        ("How do plants make food?", "Plants use sunlight, water, and air to make their food."),
        ("Why do birds fly south in winter?", "They migrate to find warmer weather and food."),
        ("Where does rain come from?", "Water vapor rises, forms clouds, and falls back down as rain."),
        ("Why is the ocean salty?", "Rivers carry tiny amounts of salt from rocks into the ocean over time."),
        ("What makes thunder?", "Thunder is the sound of lightning heating the air very quickly."),
    ]
    return [qa(q, a) for q, a in docs]


def build():
    docs = []
    docs += [qa(q, a) for q, a in IDENTITY]
    docs += [qa(q, a) for q, a in GREETINGS]
    docs += build_arithmetic()
    docs += build_word_problems()
    docs += build_letters_and_spelling()
    docs += build_counting()
    docs += build_time_and_world()
    docs += build_stories()
    docs += build_facts()
    docs += build_definitions()
    docs += build_opposites()
    docs += build_capitals()
    docs += build_sounds()
    docs += build_jokes()
    docs += build_riddles()
    docs += build_proverbs()
    docs += build_science_why()
    random.shuffle(docs)
    text = "\n\n".join(docs) + "\n"
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"seed corpus written: {out}")
    print(f"documents: {len(docs)}  characters: {len(text)}")
    return text


if __name__ == "__main__":
    build()
