r"""Generate the paper's LaTeX tables (booktabs) from aggregated cells.

Table I  - main PER/WER per language x method (best per column in bold).
Table II - efficiency: trainable params (absolute + % of base) and train time.

`booktabs` (\toprule/\midrule/\bottomrule) is assumed in the paper preamble.
Best-per-column is used (lowest PER / WER / params / time among the trained
methods); the column header is the meaningful comparison axis, not the row.
"""
from __future__ import annotations

from .aggregate import AggCell, cells_by_key
from .constants import METHOD_ORDER, LANG_ORDER, DISPLAY_LANG, SIGMORPHON_BASELINE_WER


def _fmt_int(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def _fmt_num(x: float, std: float = 0.0, bold: bool = False) -> str:
    s = f"{x:.2f}"
    if bold:
        s = r"\textbf{" + s + "}"
    if std and std > 0:
        s += rf" $\pm$ {std:.2f}"
    return s


def _wrap(body: str, caption: str, label: str, colspec: str, standalone: bool) -> str:
    if not standalone:
        return body
    return (
        "\\begin{table}[t]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        f"{body}\n\\end{{table}}\n"
    )


def build_table1(cells: list[AggCell], standalone: bool = False) -> str:
    by = cells_by_key(cells)
    # best (min) per (language, metric) among method rows present
    best = {}
    for lang in LANG_ORDER:
        for metric in ("per", "wer"):
            vals = [
                (cfg, getattr(by[(lang, cfg)], f"{metric}_mean"))
                for cfg in METHOD_ORDER
                if (lang, cfg) in by
            ]
            if vals:
                best[(lang, metric)] = min(vals, key=lambda t: t[1])[0]

    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r" & \multicolumn{2}{c}{Romanian} & \multicolumn{2}{c}{Greek} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Method & PER & WER & PER & WER \\",
        r"\midrule",
    ]
    for cfg in METHOD_ORDER:
        row = [_disp(cfg)]
        for lang in LANG_ORDER:
            c = by.get((lang, cfg))
            if c is None:
                row += ["--", "--"]
                continue
            row.append(_fmt_num(c.per_mean, c.per_std, best.get((lang, "per")) == cfg))
            row.append(_fmt_num(c.wer_mean, c.wer_std, best.get((lang, "wer")) == cfg))
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\midrule")
    base = [r"SIGMORPHON'21 baseline"]
    for lang in LANG_ORDER:
        base += ["--", f"{SIGMORPHON_BASELINE_WER[lang]:.2f}"]
    lines.append(" & ".join(base) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    body = "\n".join(lines)
    return _wrap(
        body,
        "Phoneme Error Rate (PER) and Word Error Rate (WER), in percent, on the "
        "Romanian and Greek test sets. Best per column in bold; lower is better. "
        "The Romanian LoRA (r=8) cell reports mean $\\pm$ std over 3 seeds. The "
        "final row is the reported SIGMORPHON 2021 low-resource baseline WER.",
        "tab:main",
        "lcccc",
        standalone,
    )


def build_table2(cells: list[AggCell], standalone: bool = False) -> str:
    by = cells_by_key(cells)

    def get_any(cfg):
        for lang in LANG_ORDER:
            if (lang, cfg) in by:
                return by[(lang, cfg)]
        return None

    trained = [c for c in METHOD_ORDER if c != "zeroshot"]
    min_params = min(
        (get_any(c).trainable_params for c in trained if get_any(c)), default=None
    )
    min_time = {}
    for lang in LANG_ORDER:
        times = [by[(lang, c)].train_seconds for c in trained if (lang, c) in by]
        if times:
            min_time[lang] = min(times)

    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & Trainable & \% of & \multicolumn{2}{c}{Train time (s)} \\",
        r" & params & base & Ro & El \\",
        r"\midrule",
    ]
    for cfg in METHOD_ORDER:
        c0 = get_any(cfg)
        if c0 is None:
            continue
        if cfg == "zeroshot":
            lines.append(_disp(cfg) + r" & 0 & 0.00 & -- & -- \\")
            continue
        params = c0.trainable_params
        params_s = _fmt_int(params)
        if min_params is not None and params == min_params:
            params_s = r"\textbf{" + params_s + "}"
        pct_s = f"{c0.trainable_pct:.2f}"
        time_cells = []
        for lang in LANG_ORDER:
            c = by.get((lang, cfg))
            if c is None:
                time_cells.append("--")
                continue
            t = f"{c.train_seconds:.0f}"
            if min_time.get(lang) is not None and abs(c.train_seconds - min_time[lang]) < 1e-6:
                t = r"\textbf{" + t + "}"
            time_cells.append(t)
        lines.append(
            " & ".join([_disp(cfg), params_s, pct_s, *time_cells]) + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    body = "\n".join(lines)
    return _wrap(
        body,
        "Parameter efficiency and training cost per adaptation method. Trainable "
        "parameter counts are identical across languages for a given method; "
        "training time is reported separately for Romanian (Ro) and Greek (El). "
        "Fewest trainable parameters and shortest time per column in bold.",
        "tab:efficiency",
        "lrrrr",
        standalone,
    )


def _disp(cfg: str) -> str:
    from .constants import DISPLAY_METHOD

    return DISPLAY_METHOD[cfg]


def write_table(tex: str, path) -> None:
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(tex + "\n", encoding="utf-8")
