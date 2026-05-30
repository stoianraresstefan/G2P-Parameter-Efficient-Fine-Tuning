"""Pareto-frontier computation for (trainable_params, PER) — both minimized."""
from __future__ import annotations

from .aggregate import AggCell


def pareto_frontier(points: list[tuple[float, float]]) -> list[int]:
    """Indices of non-dominated points for two-objective minimization.

    `points = [(params, per), ...]`. A point is non-dominated if no other point
    has both fewer-or-equal params and lower-or-equal PER (strictly better on at
    least one). Returns indices sorted by ascending params.
    """
    if not points:
        return []
    order = sorted(range(len(points)), key=lambda i: (points[i][0], points[i][1]))
    frontier: list[int] = []
    best_per = float("inf")
    for i in order:
        _, per = points[i]
        if per < best_per - 1e-9:
            frontier.append(i)
            best_per = per
    return frontier


def pareto_by_language(cells: list[AggCell]) -> dict[str, set[str]]:
    """For each language, the set of config codes on the PER/params frontier.

    Dominance is computed within a language only (Romanian and Greek PERs are
    not comparable). Zero-shot (0 params) is frontier-eligible as the cheapest
    extreme.
    """
    out: dict[str, set[str]] = {}
    langs = sorted({c.language for c in cells})
    for lang in langs:
        lc = [c for c in cells if c.language == lang]
        pts = [(c.trainable_params, c.per_mean) for c in lc]
        idx = pareto_frontier(pts)
        out[lang] = {lc[i].config for i in idx}
    return out
