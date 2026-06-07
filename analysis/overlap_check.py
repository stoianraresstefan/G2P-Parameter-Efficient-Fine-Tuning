"""Train/test overlap validity check (addresses the outline's threat-to-validity).

For each language, compares train+dev against test on five axes:
  1. exact grapheme overlap        - same orthographic word in both
  2. exact (grapheme, phonemes)    - full leakage of a labelled pair
  3. substring/lemma overlap       - test word is a substring of (or contains) a
                                     train word of length >= min_substr (a proxy
                                     for shared inflectional stems)
  4. prefix-stem overlap           - shares a >= min_prefix char prefix with a
                                     train word
  5. phoneme-sequence overlap      - identical gold phoneme string in both

Writes a JSON report and prints a human-readable summary.
"""
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

# Allow running as a script (python analysis/overlap_check.py) or as a module.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g2p.data import read_tsv  # noqa: E402


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def _stat(count: int, total: int, examples: list[str]) -> dict:
    return {
        "count": count,
        "pct_of_test": round(100.0 * count / total, 2) if total else 0.0,
        "examples": examples[:20],
    }


def compute_overlap(train, dev, test, min_substr: int = 4, min_prefix: int = 5) -> dict:
    train_all = list(train) + list(dev)
    train_graph = {_nfc(g).lower() for g, _ in train_all}
    train_pairs = {(_nfc(g).lower(), " ".join(ph)) for g, ph in train_all}
    train_phon = {" ".join(ph) for _, ph in train_all}
    train_long = [g for g in train_graph if len(g) >= min_substr]

    n_test = len(test)
    exact_g, exact_p, substr, prefix, phon = [], [], [], [], []
    for g, ph in test:
        gl = _nfc(g).lower()
        pstr = " ".join(ph)
        if gl in train_graph:
            exact_g.append(g)
        if (gl, pstr) in train_pairs:
            exact_p.append(g)
        if pstr in train_phon:
            phon.append(g)
        if len(gl) >= min_substr and any(
            (gl in t or t in gl) and gl != t for t in train_long
        ):
            substr.append(g)
        if len(gl) >= min_prefix and any(
            t[:min_prefix] == gl[:min_prefix] for t in train_graph if len(t) >= min_prefix
        ):
            prefix.append(g)

    return {
        "n_train": len(train),
        "n_dev": len(dev),
        "n_test": n_test,
        "exact_grapheme": _stat(len(exact_g), n_test, exact_g),
        "exact_pair": _stat(len(exact_p), n_test, exact_p),
        "substring_lemma": _stat(len(substr), n_test, substr),
        "prefix_stem": _stat(len(prefix), n_test, prefix),
        "phoneme_seq": _stat(len(phon), n_test, phon),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train/test overlap validity check.")
    ap.add_argument("--splits-dir", default="results/data_splits")
    ap.add_argument("--out", default="results/overlap_report.json")
    ap.add_argument("--langs", nargs="+", default=["rum", "gre"])
    a = ap.parse_args(argv)

    splits = Path(a.splits_dir)
    report = {}
    for lang in a.langs:
        try:
            train = read_tsv(splits / f"{lang}_train.tsv")
            dev = read_tsv(splits / f"{lang}_dev.tsv")
            test = read_tsv(splits / f"{lang}_test.tsv")
        except FileNotFoundError as e:
            print(f"[overlap] skipping {lang}: {e}")
            continue
        r = compute_overlap(train, dev, test)
        report[lang] = r
        print(
            f"[{lang}] test={r['n_test']}  exact-word={r['exact_grapheme']['pct_of_test']}%  "
            f"exact-pair={r['exact_pair']['pct_of_test']}%  "
            f"substr={r['substring_lemma']['pct_of_test']}%  "
            f"prefix={r['prefix_stem']['pct_of_test']}%  "
            f"phon-seq={r['phoneme_seq']['pct_of_test']}%"
        )

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[overlap] wrote {out}")
    return report


if __name__ == "__main__":
    main()
