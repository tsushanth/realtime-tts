#!/usr/bin/env python3
"""Expand a small set of authored template sentences into a much larger,
still-natural training corpus by substituting varied values into the
numbers/names/dates/times/amounts each template already contains.

Why this approach: hand-authoring ~1.2M characters (the standard baseline
for training a single-speaker TTS voice from scratch — see ../README.md)
isn't practical. But most of these sentences already contain a slot that's
naturally variable (a time, a dollar amount, a name) without changing the
sentence's structure or making it sound unnatural — so filling those slots
with many different real-sounding values is a legitimate, standard way to
multiply a small authored set into a much larger one without just repeating
the same handful of sentences over and over (which would teach the model
almost nothing new per repetition).

Usage: python3 expand_corpus.py corpus_batch_001.txt > corpus_expanded.txt
"""
import random
import re
import sys

random.seed(42)  # reproducible output

FIRST_NAMES = [
    "John", "Sarah", "Michael", "Emily", "David", "Jessica", "Daniel", "Ashley",
    "Robert", "Amanda", "James", "Melissa", "William", "Nicole", "Christopher",
    "Elizabeth", "Matthew", "Megan", "Anthony", "Rachel", "Mark", "Laura",
    "Steven", "Stephanie", "Andrew", "Jennifer", "Joshua", "Kimberly", "Kevin",
    "Lisa", "Brian", "Angela", "Jason", "Heather", "Ryan", "Michelle",
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
]
DAYS = [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "next Monday", "next Tuesday", "next Wednesday", "next Thursday", "next Friday",
]
MONTHS = [
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
]
TIMES = [
    "nine in the morning", "nine thirty", "ten o'clock", "ten fifteen",
    "eleven in the morning", "eleven forty-five", "noon", "twelve thirty",
    "one o'clock", "one fifteen in the afternoon", "two thirty", "two forty-five",
    "three o'clock", "three fifteen in the afternoon", "four o'clock",
    "four thirty in the afternoon", "five o'clock", "five forty-five in the evening",
    "six in the evening", "seven thirty in the evening",
]
DOLLAR_AMOUNTS = [
    "twelve dollars and fifty cents", "seventeen ninety-nine", "twenty-three dollars",
    "twenty-nine ninety-five", "thirty-four dollars and twenty cents", "forty-two fifty",
    "forty-eight dollars", "fifty-six ninety-nine", "sixty-three ninety-nine",
    "seventy-one dollars and ten cents", "eighty-nine dollars", "ninety-four fifty",
    "one hundred and eighteen dollars", "one hundred and fifty dollars and thirty cents",
    "two hundred and four dollars", "two hundred and eighty-nine ninety-nine",
]
ORDER_NUMBERS = [f"{random.randint(100000, 999999)}" for _ in range(40)]
CONFIRMATION_CODES = [
    f"{a} {b} {random.randint(1,9)} {random.randint(1,9)} {random.randint(1,9)}"
    for a, b in zip(
        random.choices(["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf", "Hotel"], k=40),
        random.choices(["Tango", "Whiskey", "Victor", "Zulu", "Sierra", "Romeo", "Oscar"], k=40),
    )
]
DAY_ORDINALS = [
    "first", "second", "third", "fifth", "eighth", "ninth", "tenth", "twelfth",
    "fifteenth", "eighteenth", "twentieth", "twenty-second", "twenty-fifth", "thirtieth",
]

# Map each recognizable literal phrase in the authored batch to the pool that
# should replace it. Longest/most-specific patterns first so a substring
# match doesn't clobber a more specific one.
SUBSTITUTIONS = [
    (r"\bTuesday, September the ninth, at two thirty in the afternoon\b",
     lambda: f"{random.choice(DAYS)}, {random.choice(MONTHS)} the {random.choice(DAY_ORDINALS)}, at {random.choice(TIMES)}"),
    (r"\bthe fifteenth of next month\b",
     lambda: f"the {random.choice(DAY_ORDINALS)} of next month"),
    (r"\bTuesday at three o'clock\b", lambda: f"{random.choice(DAYS)} at {random.choice(TIMES)}"),
    (r"\bWednesday at ten in the morning\b", lambda: f"{random.choice(DAYS)} at {random.choice(TIMES)}"),
    (r"\bthis Friday at one o'clock\b", lambda: f"{random.choice(DAYS)} at {random.choice(TIMES)}"),
    (r"\bfour o'clock\b", lambda: random.choice(TIMES)),
    (r"\bThursday to next Monday\b", lambda: f"{random.choice(DAYS)} to {random.choice(DAYS)}"),
    (r"\bMonday or Wednesday next week\b", lambda: f"{random.choice(DAYS)} or {random.choice(DAYS)}"),
    (r"\bjohn dot smith at example dot com\b",
     lambda: f"{random.choice(FIRST_NAMES).lower()} dot {random.choice(LAST_NAMES).lower()} at example dot com"),
    (r"\bone two three Maple Street, apartment four B\b",
     lambda: f"{random.randint(100,999)} {random.choice(['Maple','Oak','Cedar','Pine','Elm','Main'])} Street, apartment {random.randint(1,9)}{random.choice('ABCD')}"),
    (r"\bforty-two dollars and fifty cents\b", lambda: random.choice(DOLLAR_AMOUNTS)),
    (r"\bone hundred and eighteen dollars\b", lambda: random.choice(DOLLAR_AMOUNTS)),
    (r"\bsixty-three ninety-nine\b", lambda: random.choice(DOLLAR_AMOUNTS)),
    (r"\beighty-nine dollars\b", lambda: random.choice(DOLLAR_AMOUNTS)),
    (r"\bthirty dollars a month\b", lambda: f"{random.choice(DOLLAR_AMOUNTS)} a month"),
    (r"\bfour four seven one two three\b", lambda: " ".join(random.choice(ORDER_NUMBERS))),
    (r"\bone two three four five six\b", lambda: " ".join(random.choice(ORDER_NUMBERS))),
    (r"\bBravo Tango seven three nine\b", lambda: random.choice(CONFIRMATION_CODES)),
    (r"\bfour two one two\b", lambda: str(random.randint(1000, 9999))),
    (r"\bsix forty-five in the morning\b", lambda: random.choice(TIMES)),
    (r"\bthree to five business days\b",
     lambda: f"{random.randint(1,3)} to {random.randint(4,7)} business days"),
    (r"\bfive to seven business days\b",
     lambda: f"{random.randint(3,5)} to {random.randint(6,9)} business days"),
    (r"\bfifteen minutes\b", lambda: f"{random.choice([5,10,15,20,25,30,45])} minutes"),
    (r"\bJohn\b", lambda: random.choice(FIRST_NAMES)),
]

COMPILED = [(re.compile(pat), fn) for pat, fn in SUBSTITUTIONS]


def has_slot(line: str) -> bool:
    return any(pat.search(line) for pat, _ in COMPILED)


def expand_line(line: str) -> str:
    out = line
    for pat, fn in COMPILED:
        out = pat.sub(lambda m: fn(), out)
    return out


def main():
    if len(sys.argv) != 2:
        print("usage: expand_corpus.py <input.txt>", file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1], encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    seen = set()
    for line in lines:
        if has_slot(line):
            # Templated sentence: generate many varied, de-duplicated instances.
            variants = 0
            attempts = 0
            while variants < 60 and attempts < 300:
                attempts += 1
                candidate = expand_line(line)
                if candidate not in seen:
                    seen.add(candidate)
                    print(candidate)
                    variants += 1
        else:
            # No variable slot — one authored sentence is one authored sentence.
            if line not in seen:
                seen.add(line)
                print(line)


if __name__ == "__main__":
    main()
