"""Run the full experiment matrix for one language and write per-run JSON.

Usage (typically invoked inside the Kaggle kernel):
    python -m g2p.run_matrix --language rum \
        --data-dir /kaggle/input/g2p-peft-bundle/data \
        --eng-test /kaggle/input/g2p-peft-bundle/data/eng_us_forgetting.tsv \
        --out-dir /kaggle/working
"""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from . import config as C
from .train import run_training, tag_smoke_test


def run_language(language: str, data_dir, eng_test_path, out_dir, smoke: bool = True):
    out = Path(out_dir)
    (out / "runs").mkdir(parents=True, exist_ok=True)

    if smoke:
        try:
            res = tag_smoke_test(language, data_dir)
            print(f"[smoke] zero-shot dev PER by tag for {language}: {res}", flush=True)
            with open(out / f"smoke_{language}.json", "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=2)
        except Exception:
            print("[smoke] failed (non-fatal):", flush=True)
            traceback.print_exc()

    records = []
    for spec in C.build_matrix(language):
        print(f"=== RUN {spec.name} ===", flush=True)
        rec = run_training(spec, data_dir, eng_test_path, out_dir)
        records.append(rec)
        with open(out / "runs" / f"{spec.name}.json", "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
        print(
            f"    PER={rec['per']} WER={rec['wer']} "
            f"trainable={rec['trainable_params']} ({rec['trainable_pct']}%) "
            f"forgetting_PER={rec['forgetting_per']} time={rec['train_seconds']}s",
            flush=True,
        )

    with open(out / f"results_{language}.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    return records


def main():
    ap = argparse.ArgumentParser(description="Run the G2P PEFT experiment matrix for one language.")
    ap.add_argument("--language", required=True, choices=list(C.LANGUAGES))
    ap.add_argument("--data-dir", required=True, help="dir containing <lang>_{train,dev,test}.tsv")
    ap.add_argument("--eng-test", required=True, help="English IPA TSV for the forgetting eval")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--no-smoke", action="store_true", help="skip the tag smoke test")
    a = ap.parse_args()
    run_language(a.language, a.data_dir, a.eng_test, a.out_dir, smoke=not a.no_smoke)


if __name__ == "__main__":
    main()
