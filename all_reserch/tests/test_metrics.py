from research_suite.common.metrics import cer, hit_at_k, levenshtein_distance, mean, reciprocal_rank, wer


def test_levenshtein_distance_basic() -> None:
    assert levenshtein_distance(list("kitten"), list("sitting")) == 3


def test_levenshtein_distance_empty() -> None:
    assert levenshtein_distance([], list("abc")) == 3
    assert levenshtein_distance(list("abc"), []) == 3
    assert levenshtein_distance([], []) == 0


def test_cer_and_wer_zero_on_match() -> None:
    assert cer("HELLO WORLD", "HELLO WORLD") == 0.0
    assert wer("HELLO WORLD", "HELLO WORLD") == 0.0


def test_cer_single_char_error() -> None:
    assert cer("HELLO", "HXLLO") == 1.0 / 5.0


def test_wer_substitution() -> None:
    assert wer("the cat sat", "the dog sat") == 1.0 / 3.0


def test_cer_empty_reference() -> None:
    assert cer("", "") == 0.0
    assert cer("", "X") == 1.0


def test_hit_at_k() -> None:
    assert hit_at_k({"a", "b"}, ["c", "a", "d"], k=1) == 0.0
    assert hit_at_k({"a", "b"}, ["c", "a", "d"], k=2) == 1.0
    assert hit_at_k({"a"}, ["a", "b", "c"], k=1) == 1.0


def test_reciprocal_rank() -> None:
    assert reciprocal_rank({"b"}, ["a", "b", "c"]) == 0.5
    assert reciprocal_rank({"a"}, ["a", "b", "c"]) == 1.0
    assert reciprocal_rank({"z"}, ["a", "b", "c"]) == 0.0


def test_mean_empty() -> None:
    assert mean([]) == 0.0
    assert mean([1.0, 2.0, 3.0]) == 2.0
