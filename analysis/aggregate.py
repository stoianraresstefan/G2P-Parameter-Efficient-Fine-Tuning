"""Aggregate run records across seeds into one cell per (language, config)."""
from __future__ import annotations

import statistics
from dataclasses import dataclass

from .constants import DISPLAY_METHOD, METHOD_ORDER
from .schema import RunRecord


@dataclass
class AggCell:
    language: str
    config: str  # zeroshot | frozen_head | lora_r8 | lora_r16 | full_ft
    per_mean: float
    per_std: float
    wer_mean: float
    wer_std: float
    n_seeds: int
    trainable_params: int
    total_params: int
    trainable_pct: float
    train_seconds: float
    forgetting_per: float | None

    @property
    def display_method(self) -> str:
        return DISPLAY_METHOD[self.config]

    @property
    def sort_order(self) -> int:
        return METHOD_ORDER.index(self.config)


def aggregate(records: list[RunRecord]) -> list[AggCell]:
    groups: dict[tuple, list[RunRecord]] = {}
    for r in records:
        groups.setdefault((r.language, r.config), []).append(r)

    cells: list[AggCell] = []
    for (lang, cfg), rs in groups.items():
        pers = [r.per for r in rs]
        wers = [r.wer for r in rs]
        forgets = [r.forgetting_per for r in rs if r.forgetting_per is not None]
        cells.append(
            AggCell(
                language=lang,
                config=cfg,
                per_mean=statistics.mean(pers),
                per_std=(statistics.pstdev(pers) if len(pers) > 1 else 0.0),
                wer_mean=statistics.mean(wers),
                wer_std=(statistics.pstdev(wers) if len(wers) > 1 else 0.0),
                n_seeds=len(rs),
                trainable_params=rs[0].trainable_params,
                total_params=rs[0].total_params,
                trainable_pct=rs[0].trainable_pct,
                train_seconds=statistics.mean([r.train_seconds for r in rs]),
                forgetting_per=(statistics.mean(forgets) if forgets else None),
            )
        )
    cells.sort(key=lambda c: (c.language, c.sort_order))
    return cells


def cells_by_key(cells: list[AggCell]) -> dict[tuple, AggCell]:
    return {(c.language, c.config): c for c in cells}


def variance_summary(records: list[RunRecord], language: str = "rum", config: str = "lora_r8") -> dict:
    """mean / std / per-seed values for the headline variance config."""
    rs = [r for r in records if r.language == language and r.config == config]
    rs.sort(key=lambda r: r.seed)
    if not rs:
        return {}
    pers = [r.per for r in rs]
    wers = [r.wer for r in rs]
    return {
        "language": language,
        "config": config,
        "n": len(rs),
        "seeds": [r.seed for r in rs],
        "per_mean": statistics.mean(pers),
        "per_std": (statistics.pstdev(pers) if len(pers) > 1 else 0.0),
        "per_values": pers,
        "wer_mean": statistics.mean(wers),
        "wer_std": (statistics.pstdev(wers) if len(wers) > 1 else 0.0),
        "wer_values": wers,
    }


def best_lora(cells: list[AggCell], language: str) -> AggCell | None:
    """The LoRA cell (r8 or r16) with the lowest mean PER for a language."""
    loras = [c for c in cells if c.language == language and c.config.startswith("lora_")]
    return min(loras, key=lambda c: c.per_mean) if loras else None
