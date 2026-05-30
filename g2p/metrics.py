"""Phoneme Error Rate (PER) and Word Error Rate (WER).

We score on CHARACTER sequences with delimiters removed:
  * PER  = 100 * (sum of char-level Levenshtein distances) / (sum of gold
           char counts) -- a micro-averaged character error rate.
  * WER  = 100 * (# words whose space-normalized phoneme string is not exactly
           correct) / (# words).

Why character-level / delimiter-normalized rather than the literal SIGMORPHON
phoneme-token convention: the pretrained CharsiuG2P checkpoint emits IPA WITHOUT
delimiters (e.g. `aŋɟelos`), whereas SIGMORPHON gold and our fine-tuned models
use space-separated phonemes (`a ŋ ɟ e l o s`). Splitting the zero-shot output on
whitespace would score a phonetically-correct prediction as 100% wrong. Removing
delimiters and comparing characters makes PER/WER consistent and fair across
zero-shot and adapted models alike (WER is then a delimiter-invariant exact
match, equivalent to the official WER for space-delimited outputs). For these
languages phonemes are predominantly single characters, so character-level PER
closely tracks phoneme-level PER.
"""
from __future__ import annotations

from typing import Sequence


def to_chars(seq) -> list[str]:
    """Normalize a phoneme sequence to a list of characters with delimiters
    removed. Accepts a token list or a raw decoded string."""
    s = seq if isinstance(seq, str) else " ".join(seq)
    return list(s.replace(" ", ""))


def levenshtein(a: Sequence, b: Sequence) -> int:
    """Token-level edit distance with unit insertion/deletion/substitution cost."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def per(pairs: list[tuple[Sequence, Sequence]]) -> float:
    """pairs: list of (gold_tokens, hyp_tokens). Returns PER as a percentage."""
    edits = 0
    length = 0
    for gold, hyp in pairs:
        edits += levenshtein(gold, hyp)
        length += len(gold)
    if length == 0:
        return 0.0
    return 100.0 * edits / length


def wer(pairs: list[tuple[Sequence, Sequence]]) -> float:
    """pairs: list of (gold_tokens, hyp_tokens). Returns WER as a percentage."""
    if not pairs:
        return 0.0
    incorrect = sum(1 for gold, hyp in pairs if list(gold) != list(hyp))
    return 100.0 * incorrect / len(pairs)


def make_compute_metrics(tokenizer):
    """Build a `compute_metrics` callable for a HuggingFace Seq2SeqTrainer.

    Decodes generated predictions and labels (replacing -100 with pad), splits
    on whitespace into phoneme token lists, and reports PER/WER.
    """
    import numpy as np

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        dec_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        dec_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        pairs = [(to_chars(g), to_chars(h)) for g, h in zip(dec_labels, dec_preds)]
        return {"per": round(per(pairs), 4), "wer": round(wer(pairs), 4)}

    return compute_metrics
