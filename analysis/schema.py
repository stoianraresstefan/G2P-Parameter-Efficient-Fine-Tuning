"""Run-record schema: load and validate the per-run JSON emitted by the Kaggle
training matrix. This is the contract between `g2p.train` (producer) and the
analysis layer (consumer)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

VALID_METHODS = {"zeroshot", "frozen_head", "lora", "full_ft"}
VALID_LANGS = {"rum", "gre"}


@dataclass
class RunRecord:
    language: str
    method: str
    lora_rank: int | None
    seed: int
    per: float
    wer: float
    trainable_params: int
    total_params: int
    trainable_pct: float
    train_seconds: float
    forgetting_per: float | None = None
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def config(self) -> str:
        """Config code stable across seeds: lora_r8 / lora_r16 / zeroshot / ..."""
        if self.method == "lora":
            return f"lora_r{self.lora_rank}"
        return self.method

    @property
    def config_key(self) -> str:
        return f"{self.language}/{self.config}"

    @property
    def seed_key(self) -> tuple:
        return (self.language, self.config, self.seed)


def _from_dict(d: dict) -> RunRecord:
    return RunRecord(
        language=d["language"],
        method=d["method"],
        lora_rank=d.get("lora_rank"),
        seed=int(d.get("seed", 0)),
        per=float(d["per"]),
        wer=float(d["wer"]),
        trainable_params=int(d["trainable_params"]),
        total_params=int(d["total_params"]),
        trainable_pct=float(d["trainable_pct"]),
        train_seconds=float(d.get("train_seconds", 0.0)),
        forgetting_per=(None if d.get("forgetting_per") is None else float(d["forgetting_per"])),
        raw=d,
    )


def load_results(runs_dir) -> list[RunRecord]:
    """Load all run records from a directory of JSON files.

    Handles both per-run dict files and combined `results_<lang>.json` lists,
    and de-duplicates on (language, config, seed) keeping the first seen.
    """
    runs_dir = Path(runs_dir)
    seen: dict[tuple, RunRecord] = {}
    for fp in sorted(runs_dir.glob("*.json")):
        try:
            with open(fp, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        items = data if isinstance(data, list) else [data]
        for d in items:
            if not isinstance(d, dict) or "method" not in d:
                continue
            rec = _from_dict(d)
            seen.setdefault(rec.seed_key, rec)
    return list(seen.values())


def validate_records(records: list[RunRecord]) -> list[str]:
    """Return a list of human-readable problems (empty == all good)."""
    problems: list[str] = []
    for r in records:
        tag = f"{r.language}/{r.method}(seed {r.seed})"
        if r.method not in VALID_METHODS:
            problems.append(f"{tag}: invalid method '{r.method}'")
        if r.language not in VALID_LANGS:
            problems.append(f"{tag}: invalid language '{r.language}'")
        if not (0.0 <= r.per <= 100.0) or not (0.0 <= r.wer <= 100.0):
            problems.append(f"{tag}: PER/WER out of [0,100] (per={r.per}, wer={r.wer})")
        if (r.method == "lora") != (r.lora_rank is not None):
            problems.append(f"{tag}: lora_rank must be set iff method=='lora'")
        if r.method == "zeroshot" and (r.trainable_params != 0 or r.train_seconds != 0):
            problems.append(f"{tag}: zero-shot must have 0 trainable params and 0 train time")
        if r.total_params:
            expected = 100.0 * r.trainable_params / r.total_params
            if abs(expected - r.trainable_pct) > 0.05:
                problems.append(
                    f"{tag}: trainable_pct {r.trainable_pct} != {expected:.4f} (stale?)"
                )

    # Coverage warnings for the expected matrix.
    have = {(r.language, r.config) for r in records}
    expected_cells = {
        (lang, cfg)
        for lang in VALID_LANGS
        for cfg in ("zeroshot", "frozen_head", "lora_r8", "lora_r16", "full_ft")
    }
    for cell in sorted(expected_cells - have):
        problems.append(f"missing run for {cell[0]}/{cell[1]}")

    rum_lora8_seeds = [r for r in records if r.language == "rum" and r.config == "lora_r8"]
    if 0 < len(rum_lora8_seeds) < 3:
        problems.append(
            f"Romanian LoRA r=8 has only {len(rum_lora8_seeds)} seed(s); 3 expected for variance"
        )
    return problems
