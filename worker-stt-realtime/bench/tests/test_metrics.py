from metrics import keyterm_recall
from textnorm import norm


def test_norm_spells_digits_and_strips_punctuation():
    assert norm("Call 555-1234!") == "call five five five one two three four"


def test_keyterm_recall_counts_names_and_numbers():
    hyp = "this is sushant call five five five one two three four"
    assert keyterm_recall(["Sushant", "555 1234"], hyp) == (2, 2)


def test_keyterm_recall_partial_and_word_boundaries():
    assert keyterm_recall(["Sushant", "Ashant"], "hi ashant") == (1, 2)
    assert keyterm_recall(["ash"], "ashant") == (0, 1)
    assert keyterm_recall([], "anything") == (0, 0)
