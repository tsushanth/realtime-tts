from textnorm import norm


def keyterm_recall(terms, hyp):
    h = " " + norm(hyp) + " "
    usable = [norm(t) for t in terms if norm(t)]
    return sum(1 for t in usable if " " + t + " " in h), len(usable)
