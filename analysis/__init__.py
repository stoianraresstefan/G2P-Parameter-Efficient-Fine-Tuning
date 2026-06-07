"""Local, CPU-only results analysis: load per-run JSON, aggregate across seeds,
and emit the paper's LaTeX tables and vector PDF figures.

Only `figures` requires matplotlib/numpy; `schema`, `aggregate`, `pareto`,
`tables`, and `overlap_check` are pure stdlib.
"""

__all__ = [
    "constants",
    "schema",
    "aggregate",
    "pareto",
    "plotstyle",
    "tables",
    "figures",
    "overlap_check",
]
