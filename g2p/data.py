"""Data loading and preprocessing.

A SIGMORPHON 2021 TSV line is `grapheme<TAB>space-separated phonemes`.
We NFC-normalize the grapheme, optionally filter overly long pairs, build the
CharsiuG2P-prefixed model input, and (when a tokenizer is supplied) produce a
HuggingFace dataset for training.

The pure-Python helpers (`read_tsv`, `filter_long`, `build_input`) import nothing
heavy, so they run locally without torch/datasets installed.
"""
from __future__ import annotations

import random
import unicodedata
from pathlib import Path

from .config import (
    LANG_TAG,
    PREFIX_TEMPLATE,
    MAX_CHARS_FILTER,
    MAX_SOURCE_LEN,
    MAX_TARGET_LEN,
)


def read_tsv(path) -> list[tuple[str, list[str]]]:
    """Read a SIGMORPHON TSV -> list of (grapheme, phoneme_tokens).

    Graphemes are NFC-normalized. Read as UTF-8 explicitly (Windows defaults to
    cp1252, which would corrupt IPA / Greek).
    """
    pairs: list[tuple[str, list[str]]] = []
    with open(path, encoding="utf-8", newline="") as f:
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            cols = line.split("\t")
            if len(cols) < 2:
                continue
            grapheme = unicodedata.normalize("NFC", cols[0])
            phonemes = cols[1].split()
            pairs.append((grapheme, phonemes))
    return pairs


def filter_long(pairs, max_chars: int = MAX_CHARS_FILTER):
    """Drop pairs whose grapheme or joined phoneme string exceeds `max_chars`."""
    out = []
    for g, ph in pairs:
        if len(g) > max_chars:
            continue
        if len(" ".join(ph)) > max_chars:
            continue
        out.append((g, ph))
    return out


def build_input(grapheme: str, file_code: str) -> str:
    """`"<ron>: cuvant"` — map the SIGMORPHON file code to the CharsiuG2P tag."""
    return PREFIX_TEMPLATE.format(tag=LANG_TAG[file_code], word=grapheme)


def to_hf_dataset(pairs, file_code: str, tokenizer):
    """Tokenize (grapheme, phonemes) pairs into a `datasets.Dataset`.

    Inputs use `add_special_tokens=False` to match CharsiuG2P's inference
    convention (no EOS on the encoder side). Targets keep their EOS so the model
    learns to stop. The collator later pads labels with -100.
    """
    from datasets import Dataset

    inputs = [build_input(g, file_code) for g, _ in pairs]
    targets = [" ".join(ph) for _, ph in pairs]
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_SOURCE_LEN,
        truncation=True,
        add_special_tokens=False,
    )
    labels = tokenizer(text_target=targets, max_length=MAX_TARGET_LEN, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return Dataset.from_dict(model_inputs)


def load_english_forgetting(eng_test_path, n: int = 1000, seed: int = 42):
    """Seeded random sample of the English IPA test set for the forgetting eval."""
    pairs = filter_long(read_tsv(eng_test_path))
    rng = random.Random(seed)
    if len(pairs) > n:
        pairs = rng.sample(pairs, n)
    return pairs
