"""Unit tests for PER/WER and the Pareto frontier. Run: python -m pytest tests/
or simply `python tests/test_metrics.py`."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from g2p.metrics import levenshtein, per, wer, to_chars
from analysis.pareto import pareto_frontier


def test_to_chars_delimiter_invariant():
    # Un-delimited hypothesis (zero-shot CharsiuG2P) vs space-separated gold must
    # score as identical once normalized to characters.
    gold_tokens = ["a", "ŋ", "ɟ", "e", "l", "o", "s"]
    hyp_string = "aŋɟelos"
    g, h = to_chars(gold_tokens), to_chars(hyp_string)
    assert g == h == list("aŋɟelos")
    assert per([(g, h)]) == 0.0
    assert wer([(g, h)]) == 0.0
    # one wrong character -> 1/7 chars
    g2, h2 = to_chars(gold_tokens), to_chars("aŋɟelot")
    assert abs(per([(g2, h2)]) - 100.0 / 7) < 1e-9
    assert wer([(g2, h2)]) == 100.0


def test_levenshtein_basic():
    assert levenshtein(["a", "b", "c"], ["a", "b", "c"]) == 0
    assert levenshtein(["a", "b", "c"], ["a", "x", "c"]) == 1  # one substitution
    assert levenshtein(["a", "b"], ["a", "b", "c"]) == 1       # one insertion
    assert levenshtein(["a", "b", "c"], ["a", "c"]) == 1       # one deletion
    assert levenshtein([], ["a", "b"]) == 2
    # token-level, not char-level: multi-char tokens count as one unit
    assert levenshtein(["aa", "bb"], ["aa", "cc"]) == 1


def test_wer():
    pairs = [
        (["p", "o"], ["p", "o"]),       # correct
        (["a", "b"], ["a", "x"]),       # wrong
        (["k"], ["k"]),                 # correct
        (["t", "s"], ["s", "t"]),       # wrong (order matters)
    ]
    assert abs(wer(pairs) - 50.0) < 1e-9


def test_per():
    # gold tokens total = 2 + 2 = 4; edits = 0 + 1 = 1 -> 25%
    pairs = [
        (["p", "o"], ["p", "o"]),
        (["a", "b"], ["a", "x"]),
    ]
    assert abs(per(pairs) - 25.0) < 1e-9


def test_per_empty():
    assert per([]) == 0.0
    assert wer([]) == 0.0


def test_pareto_frontier():
    # (params, per): cheap+bad, mid+good, expensive+best, dominated
    pts = [(0, 40.0), (1000, 12.0), (500000, 8.0), (2000, 30.0)]
    fr = set(pareto_frontier(pts))
    # (0,40) cheapest; (1000,12) better; (500000,8) best. (2000,30) dominated by (1000,12).
    assert fr == {0, 1, 2}


def test_pareto_ties():
    pts = [(100, 10.0), (100, 9.0), (200, 9.0)]
    fr = pareto_frontier(pts)
    # among equal params keep the lower PER; (200,9) is dominated (same PER, more params)
    assert 1 in fr and 0 not in fr and 2 not in fr


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"All {len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
