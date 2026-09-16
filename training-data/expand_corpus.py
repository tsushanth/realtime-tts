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
    # batch_002 additions
    (r"\bAlex\b", lambda: random.choice(FIRST_NAMES)),
    (r"\btwenty dollar\b", lambda: random.choice(DOLLAR_AMOUNTS).split(" and")[0].replace(" dollars", " dollar")),
    (r"\bnineteen dollars and ninety-five cents\b", lambda: random.choice(DOLLAR_AMOUNTS)),
    (r"\bthirty-five dollars\b", lambda: random.choice(DOLLAR_AMOUNTS)),
    (r"\btwenty-four hours\b", lambda: f"{random.choice([12, 24, 36, 48, 72])} hours"),
    (r"\bninety days\b", lambda: f"{random.choice([30, 60, 90, 120])} days"),
    (r"\bthirty days\b", lambda: f"{random.choice([14, 21, 30, 45, 60])} days"),
    (r"\bone year\b", lambda: f"{random.choice(['six months', 'one year', 'two years', 'eighteen months'])}"),
    (r"\bthirty minutes\b", lambda: f"{random.choice([10, 15, 20, 30, 45])} minutes"),
    (r"\bthe next two hours\b", lambda: f"the next {random.choice([1,2,3,4])} hours"),
    (r"\beight and eleven in the morning\b",
     lambda: f"{random.choice(['seven','eight','nine'])} and {random.choice(['ten','eleven','noon'])} in the morning"),
    (r"\bten minutes\b", lambda: f"{random.choice([5,10,15,20])} minutes"),
    (r"\boriginally for Tuesday\b", lambda: f"originally for {random.choice(DAYS)}"),
    (r"\bWednesday or skip ahead\b", lambda: f"{random.choice(DAYS)} or skip ahead"),
    # batch_003 additions
    (r"\bten seconds\b", lambda: f"{random.choice([5,10,15,20])} seconds"),
    (r"\bthirty seconds\b", lambda: f"{random.choice([15,20,30,45,60])} seconds"),
    (r"\btwo gigabytes\b", lambda: f"{random.choice([1,2,4,8,16])} gigabytes"),
    (r"\bpast month\b", lambda: f"past {random.choice(['week','month','few months'])}"),
    (r"\ban hour usually\b", lambda: f"{random.choice(['thirty minutes','an hour','two hours'])} usually"),
    (r"\bten minutes early\b", lambda: f"{random.choice([5,10,15,20])} minutes early"),
    (r"\bthree business days\b", lambda: f"{random.choice([1,2,3,4,5])} business days"),
    (r"\btwenty-four hours on weekdays\b", lambda: f"{random.choice([12,24,48])} hours on weekdays"),
    (r"\ba couple of weeks\b", lambda: random.choice(["a few days", "a couple of weeks", "about a month"])),
    # batch_004 additions
    (r"\bfour, seven, one, nine\b",
     lambda: ", ".join(random.choice(["zero","one","two","three","four","five","six","seven","eight","nine"]) for _ in range(4))),
    (r"\beight, two, five, six\b",
     lambda: ", ".join(random.choice(["zero","one","two","three","four","five","six","seven","eight","nine"]) for _ in range(4))),
    (r"\bfour one seven, dash, two two three\b",
     lambda: f"{random.randint(100,999)}, dash, {random.randint(100,999)}"),
    (r"\bnine four one zero three\b", lambda: str(random.randint(10000, 99999))),
    (r"\bone one four two\b", lambda: str(random.randint(1000, 9999))),
    (r"\bC S dash four four one two\b", lambda: f"C S dash {random.randint(1000,9999)}"),
    (r"\bcaller number three\b", lambda: f"caller number {random.randint(1,9)}"),
    (r"\bseven minutes\b", lambda: f"{random.choice([2,3,5,7,10,15])} minutes"),
    (r"\btwo hundred and thirteen dollars\b", lambda: random.choice(DOLLAR_AMOUNTS)),
    (r"\bone hundred and sixty-seven dollars\b", lambda: random.choice(DOLLAR_AMOUNTS)),
    (r"\bninety-four dollars and eighteen cents\b", lambda: random.choice(DOLLAR_AMOUNTS)),
    (r"\beleven dollars and fifty cents\b", lambda: random.choice(DOLLAR_AMOUNTS)),
    (r"\bnine dollars and ninety-nine cents\b", lambda: random.choice(DOLLAR_AMOUNTS)),
    (r"\bseventy-eight dollars\b", lambda: random.choice(DOLLAR_AMOUNTS)),
    (r"\btwenty-two dollars\b", lambda: random.choice(DOLLAR_AMOUNTS)),
    (r"\balmost forty dollars\b", lambda: f"almost {random.choice(DOLLAR_AMOUNTS)}"),
    (r"\bthirty-one dollars and forty cents\b", lambda: random.choice(DOLLAR_AMOUNTS)),
    (r"\beight dollars a month\b", lambda: f"{random.choice(DOLLAR_AMOUNTS)} a month"),
    (r"\bsixty days\b", lambda: f"{random.choice([30,45,60,90])} days"),
    (r"\bFifth Avenue, about two miles away\b",
     lambda: f"{random.choice(['Fifth Avenue','Main Street','Oak Boulevard','Second Street'])}, about {random.choice([1,2,3,5])} miles away"),
    (r"\beight tonight\b", lambda: random.choice(TIMES)),
    # batch_005 additions
    (r"\bThursday at noon\b", lambda: f"{random.choice(DAYS)} at {random.choice(TIMES)}"),
    # batch_006 additions
    (r"\bfifteen percent\b", lambda: f"{random.choice([5,10,15,20,25,30,35,40])} percent"),
    (r"\bnearly twenty percent\b", lambda: f"nearly {random.choice([10,15,20,25,30])} percent"),
    (r"\broughly eighty percent\b", lambda: f"roughly {random.choice([60,70,75,80,85,90])} percent"),
    (r"\babout five percent\b", lambda: f"about {random.choice([2,3,5,8,10])} percent"),
    (r"\balmost ninety percent\b", lambda: f"almost {random.choice([80,85,90,95])} percent"),
    (r"\bjust under ten percent\b", lambda: f"just under {random.choice([5,8,10,12,15])} percent"),
    (r"\bclose to seventy percent\b", lambda: f"close to {random.choice([50,60,65,70,75])} percent"),
    (r"\bless than one percent\b", lambda: f"less than {random.choice(['half a percent','one percent','two percent'])}"),
    (r"\bmore than sixty percent\b", lambda: f"more than {random.choice([40,50,60,70])} percent"),
    (r"\bsix inches by three inches\b", lambda: f"{random.choice([4,5,6,7,8])} inches by {random.choice([2,3,4,5])} inches"),
    (r"\bten hours per charge\b", lambda: f"{random.choice([6,8,10,12,15,20])} hours per charge"),
    (r"\bsix point one inches\b", lambda: f"{random.choice(['five point five','six point one','six point seven'])} inches"),
    (r"\bsixty-four gigabytes up to five hundred twelve gigabytes\b",
     lambda: f"{random.choice([32,64,128])} gigabytes up to {random.choice(['five hundred twelve','one terabyte','two terabytes'])} gigabytes"),
    (r"\btwo hours to charge\b", lambda: f"{random.choice([1,2,3])} hours to charge"),
    (r"\bfour times optical zoom\b", lambda: f"{random.choice([2,3,4,5,10])} times optical zoom"),
    (r"\bover thirty countries\b", lambda: f"over {random.choice([15,20,30,50])} countries"),
    (r"\bfifteen years\b", lambda: f"{random.choice([5,8,10,15,20,25])} years"),
    (r"\babout five years ago\b", lambda: f"about {random.choice([2,3,5,7,10])} years ago"),
    (r"\bthree different time zones\b", lambda: f"{random.choice([2,3,4,5])} different time zones"),
    # batch_007 additions
    (r"\bseventy-five degrees\b", lambda: f"{random.choice([45,55,65,72,75,82,90])} degrees"),
    (r"\btwenty miles per hour\b", lambda: f"{random.choice([10,15,20,25,30,40])} miles per hour"),
    (r"\bgate twenty-two\b", lambda: f"gate {random.choice(['fourteen','twenty-two','thirty-one','C nine'])}"),
    (r"\bforty-five minutes before\b", lambda: f"{random.choice([30,40,45,60])} minutes before"),
    (r"\ban hour between\b", lambda: f"{random.choice(['forty minutes','an hour','ninety minutes'])} between"),
    (r"\btwo hours before an international\b", lambda: f"{random.choice([2,3])} hours before an international"),
    (r"\beighteen C\b", lambda: f"{random.choice([12,14,18,22,27])}{random.choice('ABCDEF')}"),
    (r"\bunder three hours\b", lambda: f"under {random.choice([1,2,3,4,5])} hours"),
    (r"\bcarousel five\b", lambda: f"carousel {random.choice([2,3,5,7,9])}"),
    (r"\bthrough Denver\b", lambda: f"through {random.choice(['Denver','Chicago','Atlanta','Dallas'])}"),
    (r"\bfifteen minutes\b", lambda: f"{random.choice([10,15,20,30])} minutes"),
    (r"\btwo miles ahead\b", lambda: f"{random.choice([1,2,3,5])} miles ahead"),
    (r"\bfour in the afternoon\b", lambda: random.choice(TIMES)),
    (r"\btwenty minutes to get\b", lambda: f"{random.choice([10,15,20,30,45])} minutes to get"),
    # batch_008 additions
    (r"\bfour five six Oak Lane\b",
     lambda: f"{random.randint(100,999)} {random.choice(['Oak Lane','Birch Court','Cedar Way','Willow Drive'])}"),
    (r"\bfour seven two, one one nine\b", lambda: f"{random.randint(100,999)}, {random.randint(100,999)}"),
    (r"\bM, A, R, T, I, N, E, Z\b", lambda: ", ".join(random.choice(LAST_NAMES).upper())),
    (r"\bten thirty, not ten fifteen\b", lambda: f"{random.choice(TIMES)}, not {random.choice(TIMES)}"),
    (r"\bseven, seven, zero, three\b",
     lambda: ", ".join(random.choice(["zero","one","two","three","four","five","six","seven","eight","nine"]) for _ in range(4))),
    (r"\bthirty minutes or so\b", lambda: f"{random.choice([10,15,30,45,60])} minutes or so"),
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
            while variants < 3000 and attempts < 15000:
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
