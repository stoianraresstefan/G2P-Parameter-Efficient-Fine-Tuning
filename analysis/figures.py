"""Generate the paper's vector PDF figures from aggregated cells.

Figure 1 - Pareto frontier: PER vs. trainable parameters (log x), one panel per
           language, non-dominated points highlighted and connected.
Figure 2 - Catastrophic forgetting: English IPA PER for zero-shot / best-LoRA /
           full-FT, grouped by adaptation language.
"""
from __future__ import annotations

from pathlib import Path

from .aggregate import AggCell, cells_by_key, best_lora
from .constants import METHOD_ORDER, LANG_ORDER, DISPLAY_LANG, DISPLAY_METHOD, MARKERS
from .pareto import pareto_by_language
from .plotstyle import apply_ieee_style, CONFIG_COLOR, FULL_WIDTH

_PARAM_FLOOR = 1  # zero-shot has 0 trainable params; floor for the log axis


def make_pareto_figure(cells: list[AggCell], out_path) -> dict:
    plt = apply_ieee_style()
    frontier = pareto_by_language(cells)
    by = cells_by_key(cells)
    langs = [l for l in LANG_ORDER if any(c.language == l for c in cells)]

    fig, axes = plt.subplots(1, len(langs), figsize=(FULL_WIDTH, 2.7), squeeze=False)
    summary = {}
    for ax, lang in zip(axes[0], langs):
        lc = [by[(lang, cfg)] for cfg in METHOD_ORDER if (lang, cfg) in by]
        fr = frontier.get(lang, set())
        # frontier line (sorted by params)
        fpts = sorted(
            [(max(c.trainable_params, _PARAM_FLOOR), c.per_mean) for c in lc if c.config in fr]
        )
        if len(fpts) >= 2:
            ax.plot([p[0] for p in fpts], [p[1] for p in fpts],
                    color="#444444", lw=0.8, ls="--", zorder=1, label="Pareto frontier")
        for c in lc:
            x = max(c.trainable_params, _PARAM_FLOOR)
            on = c.config in fr
            yerr = c.per_std if c.config == "lora_r8" and c.per_std > 0 else None
            ax.errorbar(
                x, c.per_mean, yerr=yerr, fmt=MARKERS.get(c.config, "o"),
                color=CONFIG_COLOR.get(c.config, "#0072B2"),
                markerfacecolor=(CONFIG_COLOR.get(c.config, "#0072B2") if on else "white"),
                markeredgecolor=CONFIG_COLOR.get(c.config, "#0072B2"),
                markeredgewidth=1.0, capsize=2, zorder=3,
            )
            ax.annotate(
                DISPLAY_METHOD[c.config], (x, c.per_mean),
                textcoords="offset points", xytext=(4, 4), fontsize=6,
            )
        ax.set_xscale("log")
        ax.set_xlabel("Trainable parameters (log scale)")
        ax.set_ylabel("Test PER (%)")
        ax.set_title(DISPLAY_LANG.get(lang, lang))
        summary[lang] = sorted(fr)
    # de-duplicate legend handles
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _save(fig, out_path)
    return summary


def make_forgetting_figure(cells: list[AggCell], out_path) -> dict:
    import numpy as np

    plt = apply_ieee_style()
    by = cells_by_key(cells)
    langs = [l for l in LANG_ORDER if (l, "zeroshot") in by]

    series = ["zeroshot", "best_lora", "full_ft"]
    series_label = {"zeroshot": "Zero-shot (base)", "best_lora": "After LoRA", "full_ft": "After full FT"}
    series_color = {"zeroshot": "#999999", "best_lora": "#0072B2", "full_ft": "#D55E00"}

    data = {s: [] for s in series}
    summary = {}
    for lang in langs:
        bl = best_lora(cells, lang)
        vals = {
            "zeroshot": _f(by.get((lang, "zeroshot"))),
            "best_lora": _f(bl),
            "full_ft": _f(by.get((lang, "full_ft"))),
        }
        for s in series:
            if vals[s] is None:
                print(f"[fig2] WARNING: missing forgetting PER for {lang}/{s} — bar omitted")
            # NaN renders as a gap (not a misleading 0.0 'no forgetting' bar).
            data[s].append(vals[s] if vals[s] is not None else float("nan"))
        summary[lang] = {
            "zeroshot": vals["zeroshot"],
            "best_lora": (bl.config if bl else None, vals["best_lora"]),
            "full_ft": vals["full_ft"],
        }

    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    x = np.arange(len(langs))
    w = 0.26
    for i, s in enumerate(series):
        offs = (i - 1) * w
        bars = ax.bar(x + offs, data[s], w, label=series_label[s], color=series_color[s])
        for b in bars:
            h = b.get_height()
            if np.isnan(h):
                continue
            ax.annotate(f"{h:.1f}", (b.get_x() + b.get_width() / 2, h),
                        textcoords="offset points", xytext=(0, 2), ha="center", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_LANG.get(l, l) + "-adapted" for l in langs])
    ax.set_ylabel("English PER (%)")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    _save(fig, out_path)
    return summary


def _f(cell):
    return None if cell is None else cell.forgetting_per


def _save(fig, out_path):
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(p))
    import matplotlib.pyplot as plt

    plt.close(fig)
