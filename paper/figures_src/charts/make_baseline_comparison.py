"""Baseline-comparison bar chart for the Results section (Table 1 visualized).

Horizontal bar chart of each metric's overall AUC (n=4,024) with 95% bootstrap CI whiskers, sorted
ascending, MorphEvalHeb highlighted. Real data from paper/results/eval/meta_eval.json -- no
placeholder/synthetic path, since the full evaluation is complete.

    uv run --no-project --with matplotlib --with numpy python make_baseline_comparison.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.abspath(os.path.join(HERE, "..", "..", "results", "eval"))
OUT = os.path.join(HERE, "out")

COL_W = 7.7 / 2.54
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8984"
GRID = "#e6e5e1"
OURS = "#2a78d6"
BASE = "#8a8984"


def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.edgecolor": INK_3,
        "axes.labelcolor": INK_2,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "text.color": INK,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.grid": True,
        "axes.grid.axis": "x",
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


PRETTY = {"BERTScore (AlephBERT)": "BERTScore (AlephBERT)", "BERTScore (heBERT)": "BERTScore (heBERT)"}


def main():
    setup_style()
    d = json.load(open(os.path.join(EVAL, "meta_eval.json")))["all"]
    rows = sorted(((k, v["auc"], v["auc_ci"]) for k, v in d.items()), key=lambda x: x[1])
    names = [PRETTY.get(k, k) for k, _, _ in rows]
    aucs = np.array([a for _, a, _ in rows])
    los = np.array([a - ci[0] for _, a, ci in rows])
    his = np.array([ci[1] - a for _, a, ci in rows])
    colors = [OURS if n == "MorphEvalHeb" else BASE for n in names]

    fig, ax = plt.subplots(figsize=(COL_W, 3.1))
    y = np.arange(len(names))
    ax.barh(y, aucs, color=colors, height=0.62, zorder=3)
    ax.errorbar(aucs, y, xerr=[los, his], fmt="none", ecolor=INK_2, elinewidth=0.7, capsize=1.5, zorder=4)
    ax.axvline(0.5, color=INK_3, linewidth=0.7, linestyle=(0, (2, 2)), zorder=2)
    ax.text(0.5, len(names) - 0.3, "chance", fontsize=6.5, color=INK_3, ha="center", va="bottom")

    for yi, a in zip(y, aucs):
        ax.text(a + 0.015, yi, f"{a:.3f}", va="center", fontsize=6.8,
                color=OURS if names[yi] == "MorphEvalHeb" else INK_2,
                fontweight="bold" if names[yi] == "MorphEvalHeb" else "normal")

    ax.set_yticks(y, names)
    for lbl, n in zip(ax.get_yticklabels(), names):
        if n == "MorphEvalHeb":
            lbl.set_fontweight("bold")
            lbl.set_color(OURS)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("ROC AUC (full benchmark, $n{=}4{,}024$)")
    ax.set_title("All metrics vs. MorphEvalHeb", loc="left", fontsize=8.5, color=INK)
    fig.tight_layout()

    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, "fig_baseline_comparison.pdf"))
    fig.savefig(os.path.join(OUT, "fig_baseline_comparison_preview.png"), dpi=220)
    plt.close(fig)
    print("wrote fig_baseline_comparison")


if __name__ == "__main__":
    main()
