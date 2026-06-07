"""Write plausible synthetic run records into results/runs/ so the analysis
pipeline (tables + figures) can be exercised end-to-end without spending Kaggle
GPU time. These numbers are FAKE placeholders, not experimental results.

    python scripts/make_demo_results.py            # writes to results/runs
    python scripts/make_demo_results.py --out tmp  # custom dir
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# (config, trainable_params, trainable_pct, per_rum, wer_rum, per_gre, wer_gre,
#  time_rum, time_gre, forget_rum, forget_gre)
DEMO = [
    ("zeroshot", 0, 0.0, 11.2, 28.0, 19.5, 41.0, 0, 0, 1.8, 1.8),
    ("frozen_head", 565248, 0.19, 9.8, 24.0, 16.1, 33.0, 95, 110, 1.9, 1.9),
    ("lora_r8", 294912, 0.10, 6.9, 18.0, 11.2, 24.0, 150, 165, 2.1, 2.4),
    ("lora_r16", 589824, 0.20, 6.7, 17.5, 11.0, 23.5, 158, 172, 2.2, 2.5),
    ("full_ft", 299900000, 100.0, 6.5, 17.0, 10.8, 23.0, 420, 440, 9.7, 12.3),
]
TOTAL = 299900000


def _record(lang, cfg, params, pct, per, wer, t, forget, seed):
    method = "lora" if cfg.startswith("lora_") else cfg
    rank = int(cfg.split("_r")[1]) if method == "lora" else None
    return {
        "language": lang,
        "lang_tag": {"rum": "ron", "gre": "gre"}[lang],
        "method": method,
        "lora_rank": rank,
        "lora_alpha": (2 * rank if rank else None),
        "seed": seed,
        "per": per,
        "wer": wer,
        "trainable_params": params,
        "total_params": TOTAL,
        "trainable_pct": pct,
        "train_seconds": float(t),
        "forgetting_per": forget,
        "base_model": "charsiu/g2p_multilingual_byT5_small",
        "epochs": (0 if method == "zeroshot" else 30),
        "lr": (None if method == "zeroshot" else 1e-3),
        "batch_size": (None if method == "zeroshot" else 16),
        "test_n": 100,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/runs")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    n = 0
    for cfg, params, pct, pr, wr, pg, wg, tr, tg, fr, fg in DEMO:
        for lang, per, wer, t, forget in (("rum", pr, wr, tr, fr), ("gre", pg, wg, tg, fg)):
            rec = _record(lang, cfg, params, pct, per, wer, t, forget, seed=42)
            name = f"{lang}_{rec['method']}" + (f"_r{rec['lora_rank']}" if rec['lora_rank'] else "") + "_s42"
            (out / f"{name}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
            n += 1
    # extra Romanian LoRA r=8 seeds for variance
    for seed, per, wer in ((1, 7.1, 18.5), (2, 6.6, 17.2)):
        rec = _record("rum", "lora_r8", 294912, 0.10, per, wer, 152, 2.1, seed=seed)
        (out / f"rum_lora_r8_s{seed}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        n += 1
    print(f"Wrote {n} synthetic run records to {out}")


if __name__ == "__main__":
    main()
