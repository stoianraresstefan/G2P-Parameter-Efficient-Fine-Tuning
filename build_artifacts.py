"""Regenerate all paper artifacts (tables + figures) from results/runs/*.json.

    python build_artifacts.py              # tables + figures
    python build_artifacts.py --tables     # only tables
    python build_artifacts.py --figures    # only figures
    python build_artifacts.py --overlap    # run the train/test overlap check
    python build_artifacts.py --check      # validate run JSON only, no artifacts
    python build_artifacts.py --strict     # turn validation warnings into a nonzero exit

Determinism: stable orderings and fixed number formatting, so regenerated
artifacts diff cleanly in git.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from analysis.schema import load_results, validate_records
from analysis.aggregate import aggregate, variance_summary, best_lora
from analysis.tables import build_table1, build_table2, write_table


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results/runs")
    ap.add_argument("--out-tables", default="tables")
    ap.add_argument("--out-figures", default="figures")
    ap.add_argument("--tables", action="store_true", help="only build tables")
    ap.add_argument("--figures", action="store_true", help="only build figures")
    ap.add_argument("--overlap", action="store_true", help="also run the overlap validity check")
    ap.add_argument("--check", action="store_true", help="validate only; build no artifacts")
    ap.add_argument("--strict", action="store_true", help="nonzero exit if validation finds problems")
    ap.add_argument("--standalone", action="store_true", help="wrap tables in a full table float")
    a = ap.parse_args(argv)

    records = load_results(a.results_dir)
    if not records:
        print(f"No run records found in {a.results_dir}. Run the Kaggle matrix first "
              f"(or `python scripts/make_demo_results.py` for a dry-run fixture).")
        return 1

    problems = validate_records(records)
    if problems:
        print(f"Validation warnings ({len(problems)}):")
        for p in problems:
            print("  -", p)
    else:
        print(f"Validation OK: {len(records)} run records.")
    if a.strict and problems:
        return 2
    if a.check:
        return 0

    cells = aggregate(records)
    written = []

    do_tables = a.tables or not a.figures
    do_figures = a.figures or not a.tables

    if do_tables:
        t1 = Path(a.out_tables) / "table1_main.tex"
        t2 = Path(a.out_tables) / "table2_efficiency.tex"
        write_table(build_table1(cells, standalone=a.standalone), t1)
        write_table(build_table2(cells, standalone=a.standalone), t2)
        written += [t1, t2]

    if do_figures:
        from analysis.figures import make_pareto_figure, make_forgetting_figure

        f1 = Path(a.out_figures) / "fig1_pareto.pdf"
        f2 = Path(a.out_figures) / "fig2_forgetting.pdf"
        pareto = make_pareto_figure(cells, f1)
        forget = make_forgetting_figure(cells, f2)
        written += [f1, f2]
        print("Pareto frontier (config codes on frontier):", pareto)
        print("Forgetting summary (English PER):", forget)

    if a.overlap:
        from analysis.overlap_check import main as overlap_main

        overlap_main([])

    # Headline variance sentence for the paper text.
    vs = variance_summary(records, "rum", "lora_r8")
    if vs and vs.get("n", 0) > 1:
        print(
            f"Headline variance (Romanian LoRA r=8, {vs['n']} seeds): "
            f"PER {vs['per_mean']:.2f} ± {vs['per_std']:.2f}, "
            f"WER {vs['wer_mean']:.2f} ± {vs['wer_std']:.2f} "
            f"(PER per seed: {vs['per_values']})"
        )

    print("\nWrote:")
    for w in written:
        print("  ", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
