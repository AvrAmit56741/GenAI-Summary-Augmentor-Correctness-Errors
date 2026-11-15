"""Translate-then-check vs. MorphEvalHeb (docs/experiments.md #17).

Grouped bar chart: AUC of FactCG run natively on Hebrew vs. after translating to English with two
commercial LLM translators (Claude Haiku 4.5, Amazon Nova Micro), compared with MorphEvalHeb, on all
three benchmark groupings. Computed directly from the verified raw data
(paper/results/eval/translate_test_api_results.csv, 250 pairs) -- bootstrap CIs recomputed here, not
copied from the peer-session summary file.

    uv run --no-project --with matplotlib --with numpy --with pandas --with scikit-learn python make_translate_test.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.abspath(os.path.join(HERE, "..", "..", "results", "eval"))
OUT = os.path.join(HERE, "out")

COL_W = 7.7 / 2.54
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8984"
GRID = "#e6e5e1"
OURS = "#2a78d6"
BASE = "#8a8984"
HAIKU_C = "#eb6834"
NOVA_C = "#1baf7a"

COLS = ["FactCG-HE", "FactCG-EN-haiku45", "FactCG-EN-novamicro", "MorphEvalHeb"]
LABELS = {"FactCG-HE": "FactCG (Hebrew)", "FactCG-EN-haiku45": "FactCG (after Haiku 4.5 MT)",
          "FactCG-EN-novamicro": "FactCG (after Nova Micro MT)", "MorphEvalHeb": "MorphEvalHeb"}
COLORS = {"FactCG-HE": BASE, "FactCG-EN-haiku45": HAIKU_C, "FactCG-EN-novamicro": NOVA_C, "MorphEvalHeb": OURS}
GROUPS = [("All", None), ("Injected", "MAFAT-Aug"), ("Natural", "mFACT-He")]


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
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def boot_ci(y, s, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    vals = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(set(y[b])) < 2:
            continue
        vals.append(roc_auc_score(y[b], s[b]))
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def main():
    setup_style()
    a = pd.read_csv(os.path.join(EVAL, "translate_test_api_results.csv"))

    fig, ax = plt.subplots(figsize=(COL_W, 3.3))
    n_metrics = len(COLS)
    width = 0.19
    x0 = np.arange(len(GROUPS))

    for i, col in enumerate(COLS):
        aucs, los, his = [], [], []
        for _, part in GROUPS:
            x = a if part is None else a[a.part == part]
            y, s = x.y.to_numpy(), x[col].to_numpy()
            auc = roc_auc_score(y, s)
            lo, hi = boot_ci(y, s)
            aucs.append(auc); los.append(auc - lo); his.append(hi - auc)
        xpos = x0 + (i - (n_metrics - 1) / 2) * width
        ax.bar(xpos, aucs, width=width * 0.92, color=COLORS[col], zorder=3,
               label=LABELS[col], edgecolor="white", linewidth=0.4)
        ax.errorbar(xpos, aucs, yerr=[los, his], fmt="none", ecolor=INK_2, elinewidth=0.6, capsize=1.3, zorder=4)

    ax.axhline(0.5, color=INK_3, linewidth=0.7, linestyle=(0, (2, 2)), zorder=2)
    ax.set_xticks(x0, [g for g, _ in GROUPS])
    ax.set_ylim(0.3, 1.05)
    ax.set_ylabel("ROC AUC")
    ax.set_title("Translate-then-check vs. MorphEvalHeb ($n{=}250$)", loc="left", fontsize=8.3, color=INK)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.42), ncols=2, fontsize=6.6,
              frameon=False, handlelength=1.1, columnspacing=0.9)
    fig.tight_layout()

    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, "fig_translate_test.pdf"))
    fig.savefig(os.path.join(OUT, "fig_translate_test_preview.png"), dpi=220)
    plt.close(fig)
    print("wrote fig_translate_test")


if __name__ == "__main__":
    main()
