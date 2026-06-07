"""Download SIGMORPHON 2021 Task 1 data for Romanian + Greek and build the
English (eng-us) IPA forgetting sample.

Writes UTF-8 TSVs into BOTH results/data_splits/ (for the local overlap check)
and kaggle/bundle/staging/data/ (for the Kaggle dataset bundle).

Data source: github.com/sigmorphon/2021-task1 (data/low for rum+gre, an English
IPA test set for forgetting). Filenames are discovered via the GitHub contents
API so the script is robust to exact-name drift.

    python scripts/01_prepare_data.py
"""
from __future__ import annotations

import argparse
import io
import random
import sys
import unicodedata
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from _util import REPO_ROOT, DATA_SPLITS  # noqa: E402

API = "https://api.github.com/repos/sigmorphon/2021-task1/contents/{path}"
RAW = "https://raw.githubusercontent.com/sigmorphon/2021-task1/master/{path}"

DEST_DIRS = [DATA_SPLITS, REPO_ROOT / "kaggle" / "_staging" / "data"]
ENG_N = 1000
ENG_SEED = 42


def list_dir(path: str) -> list[str]:
    r = requests.get(API.format(path=path), timeout=30)
    r.raise_for_status()
    return [item["name"] for item in r.json() if item["type"] == "file"]


def fetch(path: str) -> str:
    r = requests.get(RAW.format(path=path), timeout=60)
    r.raise_for_status()
    return r.text


def write_all(name: str, text: str):
    # NFC normalize graphemes on the way in; keep exact phoneme strings.
    out_lines = []
    for line in text.splitlines():
        if not line.strip():
            continue
        cols = line.split("\t")
        cols[0] = unicodedata.normalize("NFC", cols[0])
        out_lines.append("\t".join(cols))
    blob = "\n".join(out_lines) + "\n"
    for d in DEST_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        (d / name).write_text(blob, encoding="utf-8", newline="\n")
    print(f"  wrote {name} ({len(out_lines)} lines)")


def find_english_test() -> str:
    """Locate an English (US) IPA test file across the resource tiers."""
    for tier in ("high", "medium", "low"):
        try:
            files = list_dir(f"data/{tier}")
        except requests.HTTPError:
            continue
        # prefer eng_us, then eng_uk, then any eng* test file
        for pref in ("eng_us", "eng_uk", "eng"):
            for f in files:
                if f.startswith(pref) and f.endswith("_test.tsv"):
                    print(f"  English forgetting source: data/{tier}/{f}")
                    return f"data/{tier}/{f}"
    raise SystemExit("Could not locate an English IPA test set in the SIGMORPHON repo.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=["rum", "gre"])
    a = ap.parse_args()

    print("Downloading SIGMORPHON 2021 low-resource splits...")
    low_files = set(list_dir("data/low"))
    for lang in a.langs:
        for split in ("train", "dev", "test"):
            name = f"{lang}_{split}.tsv"
            if name not in low_files:
                raise SystemExit(f"{name} not found in data/low (have: {sorted(low_files)[:8]}...)")
            write_all(name, fetch(f"data/low/{name}"))

    print("Building English (eng-us) IPA forgetting sample...")
    eng_path = find_english_test()
    eng_text = fetch(eng_path)
    rows = [ln for ln in eng_text.splitlines() if ln.strip() and "\t" in ln]
    rng = random.Random(ENG_SEED)
    if len(rows) > ENG_N:
        rows = rng.sample(rows, ENG_N)
    write_all("eng_us_forgetting.tsv", "\n".join(rows))

    print(f"\nDone. Data in: {DATA_SPLITS}")
    print(f"            and: {REPO_ROOT / 'kaggle' / '_staging' / 'data'}")


if __name__ == "__main__":
    main()
