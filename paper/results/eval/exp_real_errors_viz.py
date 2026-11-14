"""Thesis-only figure: does ANY metric's score track real-error severity? (lead #3/#6 completion)

Combines severity.csv (MorphEvalHeb, from exp_severity.py) and real_errors_ngram.csv (n-gram
baselines, from exp_real_errors_ngram.py) into one panel figure, severity bucket (1-5) on the
x-axis, normalized mean score on the y-axis, one line per metric. All lines are visually flat --
this is the point: nothing here tracks real, Gemini-confirmed error severity.

    python exp_real_errors_viz.py   -> thesis_figures/real_error_severity_tracking.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "thesis_figures")
os.makedirs(OUT_DIR, exist_ok=True)

NGRAM_METRICS = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU", "chrF", "Token F1"]


def main():
    sev = pd.read_csv(os.path.join(HERE, "severity.csv"))
    ngram = pd.read_csv(os.path.join(HERE, "real_errors_ngram.csv"))

    morph_by_sev = sev.groupby("max_severity").unfaithful_prob.mean()
    ngram_by_sev = ngram.groupby("max_severity")[NGRAM_METRICS].mean()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    morph_norm = morph_by_sev / morph_by_sev.max()
    ax.plot(morph_norm.index, morph_norm.values, marker="o", color="#2a78d6", linewidth=2, label="MorphEvalHeb")
    ax.set_xlabel("Severity (1=mild, 5=severe)")
    ax.set_ylabel("Unfaithfulness probability (normalized)")
    ax.set_title("MorphEvalHeb: score vs. real-error severity\n(n=265, Spearman $\\rho$=-0.04, n.s.)")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1]
    for mt in NGRAM_METRICS:
        y = ngram_by_sev[mt]
        y_norm = (y - y.min()) / (y.max() - y.min() + 1e-9)
        ax.plot(y_norm.index, y_norm.values, marker="o", label=mt, alpha=0.8)
    ax.set_xlabel("Severity (1=mild, 5=severe)")
    ax.set_ylabel("Score (normalized per metric)")
    ax.set_title("N-gram baselines: score vs. real-error severity\n(n=265, all $|\\rho|$<0.05, n.s.)")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)

    fig.suptitle("No metric tracks real LLM-error severity (synthetic-trained metrics vs. real errors)")
    fig.tight_layout()
    png = os.path.join(OUT_DIR, "real_error_severity_tracking.png")
    fig.savefig(png, dpi=200)
    print("saved", png)


if __name__ == "__main__":
    main()
