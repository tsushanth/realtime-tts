import re

_D = "zero one two three four five six seven eight nine".split()


def norm(t):
    t = re.sub(r"\d", lambda m: " " + _D[int(m.group())] + " ", t)   # digits -> spoken (refs are spelled out)
    t = t.lower().replace("-", " ")
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()
