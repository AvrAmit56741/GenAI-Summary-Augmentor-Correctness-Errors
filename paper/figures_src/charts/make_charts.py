"""Build the MorphEvalHeb paper charts (PDF for the paper + PNG preview).

Run from anywhere:
    uv run --no-project --with pandas --with matplotlib python make_charts.py
    uv run --no-project --with pandas --with matplotlib python make_charts.py --scores scores.csv

Charts built from REAL data in the repo:
    fig_error_family_distribution   benchmark combined.csv, error families in negative summaries
    fig_errors_per_summary          benchmark combined.csv, number of distinct families per negative
    fig_llm_detected_errors         thesisdatasets/error_finder, errors that Gemini found in real LLM summaries

Charts that need per-summary metric scores. Without --scores they are drawn from
synthetic data and carry a "PLACEHOLDER DATA" watermark:
    fig_per_family_detection        detection rate per error family, ours vs. two baselines
    fig_score_distributions         faithful vs. unfaithful score distributions per metric
    fig_robustness                  score shift under meaning-preserving rewrites (placeholder only)
    fig_multilabel_confusion        multi-label detector confusions (placeholder only)

--scores CSV schema (one row per test summary):
    split            "dev" or "test" (thresholds are tuned on dev, plotted on test)
    label            "entailment" or "contradiction"
    error_types      comma-separated families (negatives only)
    score_<metric>   one column per metric, higher = more faithful
                     (e.g. score_MorphEvalHeb, score_GPT-4o-mini judge, score_mFACT)
"""

import argparse
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
BENCHMARK = os.path.join(REPO, "thesisdatasets/train_datasets/Correctness/data/benchmark/combined.csv")
FINDER_GLOB = os.path.join(REPO, "thesisdatasets/error_finder/*_gemini_finder.csv")
OUT = os.path.join(HERE, "out")

# ACL geometry: single column 7.7 cm, double column 16 cm.
COL_W = 7.7 / 2.54
PAGE_W = 16.0 / 2.54

# Palette: reference categorical slots 1-3 (validated all-pairs on white), text inks, recessive structure.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # ours, baseline A, baseline B
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8984"
GRID = "#e6e5e1"
FAITHFUL, UNFAITHFUL = SERIES[0], SERIES[1]

LOGICAL = {"sentence_negation", "number_swap", "measure_unit_swap"}
GROUP_ORDER = ["Morpho-syntactic", "Logical", "Entity"]


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
        "legend.fontsize": 7.5,
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
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def pretty(family: str) -> str:
    name = family.replace("_errors", "").replace("entity_", "").replace("_", " ")
    fixes = {"hebrew root pattern confusion": "root-pattern confusion",
             "morphological connective confusion": "connective confusion",
             "idiom collocation corruption": "idiom/collocation",
             "directional preposition swap": "directional preposition",
             "homographic gender": "homographic gender"}
    name = fixes.get(name, name)
    if family.startswith("entity_"):
        name = f"entity: {name.replace(' swap', '')}"
    return name


def group_of(family: str) -> str:
    if family.startswith("entity_"):
        return "Entity"
    if family in LOGICAL:
        return "Logical"
    return "Morpho-syntactic"


def watermark(fig):
    fig.text(0.5, 0.5, "PLACEHOLDER DATA", rotation=28, ha="center", va="center",
             fontsize=26, color="#d4483c", alpha=0.16, fontweight="bold", zorder=100)


def save(fig, name, placeholder=False):
    if placeholder:
        watermark(fig)
        name = f"{name}_PLACEHOLDER"
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, f"{name}.pdf"))
    fig.savefig(os.path.join(OUT, f"{name}_preview.png"), dpi=220)
    plt.close(fig)
    print("wrote", name)


def load_benchmark():
    d = pd.read_csv(BENCHMARK)
    neg = d[(d.label == "contradiction") & d.error_types.notna()].copy()
    neg["families"] = neg.error_types.str.split(",").apply(lambda xs: sorted({x.strip() for x in xs if x.strip()}))
    return d, neg


def families_by_group(neg):
    counts = neg.families.explode().value_counts()
    ordered = []
    for g in GROUP_ORDER:
        fams = [f for f in counts.index if group_of(f) == g]
        ordered += sorted(fams, key=lambda f: -counts[f])
    return ordered, counts


def group_bands(ax, fams, x_text=None):
    """Dashed separators between taxonomy groups and vertical group names outside the right edge."""
    start = 0
    for g in GROUP_ORDER:
        n = sum(group_of(f) == g for f in fams)
        if n == 0:
            continue
        if start > 0:
            ax.axhline(start - 0.5, color=INK_3, lw=0.5, ls=(0, (2, 2)))
        ax.text(1.015, start + (n - 1) / 2, g, transform=ax.get_yaxis_transform(), rotation=270,
                ha="left", va="center", fontsize=7, color=INK_2, style="italic")
        start += n


# ---------------------------------------------------------------- real-data charts
def fig_error_family_distribution(neg):
    fams, counts = families_by_group(neg)
    n_neg = len(neg)
    fig, ax = plt.subplots(figsize=(COL_W, 3.5))
    y = np.arange(len(fams))
    share = [counts[f] / n_neg * 100 for f in fams]
    ax.barh(y, share, height=0.62, color=SERIES[0], edgecolor="white", linewidth=0.6)
    ax.set_yticks(y, [pretty(f) for f in fams])
    ax.invert_yaxis()
    ax.set_xlabel(f"% of negative summaries (n = {n_neg:,})")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(share) * 1.15)
    for yi, s in zip(y, share):
        if s < 3:  # call out the near-empty families
            ax.text(s + 0.6, yi, f"{s:.1f}%", va="center", fontsize=6.5, color=INK_2)
    group_bands(ax, fams, x_text=0.99)
    save(fig, "fig_error_family_distribution")


def fig_errors_per_summary(neg):
    k = neg.families.apply(len)
    vc = k.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(COL_W, 1.7))
    ax.bar(vc.index, vc.values, width=0.72, color=SERIES[0], edgecolor="white", linewidth=0.6)
    ax.set_xlabel("Error families per negative summary")
    ax.set_ylabel("Negative summaries")
    ax.set_xticks(vc.index)
    ax.grid(axis="x", visible=False)
    med = float(k.median())
    ax.axvline(med, color=INK_2, lw=0.8, ls=(0, (3, 2)))
    ax.set_ylim(0, vc.max() * 1.18)
    ax.text(med, vc.max() * 1.17, f"median {med:g}", fontsize=7, color=INK_2, va="top", ha="center",
            bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
    save(fig, "fig_errors_per_summary")


def load_finder():
    rows = []
    for f in sorted(glob.glob(FINDER_GLOB)):
        m = re.search(r"finder_\d{8}_\d{6}_(\d+)_(\d+)_(.+?)_summaries_gemini_finder\.csv$", f)
        start, end, model = int(m.group(1)), int(m.group(2)), m.group(3)
        e = pd.read_csv(f)
        e["model"], e["start"], e["end"], e["file"] = model, start, end, os.path.basename(f)
        rows.append(e)
    e = pd.concat(rows, ignore_index=True)
    names = {"gpt3.5": "GPT-3.5-turbo", "dicta2.0": "DictaLM 2.0", "ollama_gemma3-12b": "Gemma 3 12B",
             "ollama_gpt-oss-20b": "gpt-oss-20b", "ollama_qwen3_14b": "Qwen3 14B"}
    e["model"] = e.model.map(names).fillna(e.model)
    return e


def fig_llm_detected_errors():
    e = load_finder()
    stats = []
    for model, g in e.groupby("model"):
        # Row ranges in run filenames do not identify the same documents across runs, so pool per run:
        # each run examined (end - start) summaries; documents listed in a run had >=1 detected error.
        runs = g.groupby("file").agg(start=("start", "first"), end=("end", "first"), docs=("doc_id", "nunique"))
        examined = int((runs.end - runs.start).sum())
        stats.append(dict(model=model, n=examined, with_err=int(runs.docs.sum()),
                          errors=len(g), sev=g.severity.value_counts()))
    st = pd.DataFrame(stats)
    st["share"] = st.with_err / st.n * 100
    st = st.sort_values("share")

    # Single ACL column: the two measures get their own stacked panels (never one shared axis).
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(COL_W, 2.75), gridspec_kw=dict(hspace=0.75))
    y = np.arange(len(st))
    labels = [f"{m} (n={n})" for m, n in zip(st.model, st.n)]
    a1.barh(y, st.share, height=0.62, color=SERIES[1], edgecolor="white", linewidth=0.6)
    a1.set_yticks(y, labels)
    a1.set_xlim(0, 100)
    a1.set_xticks([0, 25, 50, 75, 100])
    a1.set_title("(a) Summaries with \u22651 detected error (%)", loc="left", fontsize=7.5, color=INK)
    a1.grid(axis="y", visible=False)
    for yi, sh in zip(y, st.share):
        a1.text(sh - 2, yi, f"{sh:.0f}%", va="center", ha="right", fontsize=6.8, color="white", fontweight="bold")

    # Severity: sequential single-hue ramp (light = minor, dark = severe), 100%-stacked.
    levels = [1, 2, 3, 4, 5]
    ramp = ["#fde3d3", "#f9b896", "#f08a57", "#d95926", "#9a3510"]
    left = np.zeros(len(st))
    for lv, col in zip(levels, ramp):
        vals = np.array([r.sev.get(lv, 0) / r.errors * 100 for r in st.itertuples()])
        a2.barh(y, vals, left=left, height=0.62, color=col, edgecolor="white", linewidth=0.6, label=str(lv))
        left += vals
    a2.set_yticks(y, labels)
    a2.set_xlim(0, 100)
    a2.set_xticks([0, 25, 50, 75, 100])
    a2.set_title("(b) Severity of detected errors (% of errors)", loc="left", fontsize=7.5, color=INK)
    a2.grid(axis="y", visible=False)
    a2.legend(title="severity: 1 = minor, 5 = severe", ncols=5, loc="upper center", bbox_to_anchor=(0.42, -0.16),
              handlelength=1.0, columnspacing=0.9, title_fontsize=6.8, fontsize=6.8)
    save(fig, "fig_llm_detected_errors")
    tot_err = sum(r.errors for r in st.itertuples())
    sev45 = sum(r.sev.get(4, 0) + r.sev.get(5, 0) for r in st.itertuples())
    print("detected-errors stats:", st[["model", "n", "with_err", "errors"]].to_dict("records"),
          f"total n={st.n.sum()} with_err={st.with_err.sum()} share={st.with_err.sum()/st.n.sum():.3f} sev4-5={sev45/tot_err:.3f} errors={tot_err}")
    return st


# ---------------------------------------------------------------- score-dependent charts
def synthetic_scores(d, neg, seed=7):
    """Synthetic per-summary scores, ONLY for drafting chart layouts."""
    rng = np.random.default_rng(seed)
    t = d.copy()
    t["split"] = np.where(rng.random(len(t)) < 0.3, "dev", "test")
    fam_map = neg.set_index(neg.index).families
    t["families"] = [fam_map.get(i, []) for i in t.index]
    y = (t.label == "entailment").astype(float).values
    for name, sep in [("MorphEvalHeb", 2.2), ("GPT-4o-mini judge", 1.2), ("mFACT", 0.8)]:
        z = rng.normal(0, 1, len(t)) + sep * (y - 0.5) * 2
        t[f"score_{name}"] = 1 / (1 + np.exp(-z))
    return t


def load_scores(path, neg):
    t = pd.read_csv(path)
    t["families"] = t.error_types.fillna("").str.split(",").apply(lambda xs: sorted({x.strip() for x in xs if x.strip()}))
    return t


def best_threshold(scores, labels):
    best, best_t = -1, 0.5
    for th in np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 99))):
        pred = scores >= th
        tpr = (pred & labels).sum() / max(labels.sum(), 1)
        tnr = (~pred & ~labels).sum() / max((~labels).sum(), 1)
        if (tpr + tnr) / 2 > best:
            best, best_t = (tpr + tnr) / 2, th
    return best_t


def fig_per_family_detection(t, neg, placeholder):
    metrics = [c[len("score_"):] for c in t.columns if c.startswith("score_")][:3]
    fams, _ = families_by_group(neg)
    dev, test = t[t.split == "dev"], t[t.split == "test"]
    fig, ax = plt.subplots(figsize=(COL_W, 3.6))
    y = np.arange(len(fams))
    markers = ["o", "s", "D"]
    for i, m in enumerate(metrics):
        th = best_threshold(dev[f"score_{m}"].values, (dev.label == "entailment").values)
        rates = []
        for f in fams:
            sub = test[test.families.apply(lambda fs: f in fs)]
            rates.append((sub[f"score_{m}"] < th).mean() * 100 if len(sub) else np.nan)
        ax.plot(rates, y, ls="none", marker=markers[i], ms=4.2 if i == 0 else 3.6,
                mfc=SERIES[i] if i == 0 else "white", mec=SERIES[i], mew=1.1, label=m, zorder=3 + (i == 0))
    ax.set_yticks(y, [pretty(f) for f in fams])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Detection rate on negatives containing the family (%)")
    ax.grid(axis="y", color=GRID, lw=0.4)
    ax.legend(loc="lower center", bbox_to_anchor=(0.4, 1.0), ncols=3, handletextpad=0.3, columnspacing=1.0)
    group_bands(ax, fams, x_text=0.99)
    save(fig, "fig_per_family_detection", placeholder)


def fig_score_distributions(t, placeholder):
    metrics = [c[len("score_"):] for c in t.columns if c.startswith("score_")][:3]
    test = t[t.split == "test"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(PAGE_W * 0.62, 1.65), sharey=True,
                             gridspec_kw=dict(wspace=0.08))
    for ax, m in zip(axes, metrics):
        for pos, (lab, col) in enumerate([("entailment", FAITHFUL), ("contradiction", UNFAITHFUL)]):
            v = test.loc[test.label == lab, f"score_{m}"].dropna().values
            parts = ax.violinplot([v], positions=[pos], widths=0.8, showextrema=False, showmedians=False)
            for b in parts["bodies"]:
                b.set_facecolor(col)
                b.set_edgecolor(col)
                b.set_alpha(0.55)
                b.set_linewidth(0.6)
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            ax.vlines(pos, q1, q3, color=INK, lw=2.2)
            ax.scatter([pos], [med], s=10, color="white", zorder=3, edgecolor=INK, linewidth=0.6)
        ax.set_xticks([0, 1], ["faithful", "unfaithful"])
        ax.set_title(m, color=INK)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Metric score")
    save(fig, "fig_score_distributions", placeholder)


def fig_robustness():
    rng = np.random.default_rng(11)
    metrics = ["MorphEvalHeb", "GPT-4o-mini judge", "mFACT", "SummaC-ZS", "BERTScore", "ROUGE-1"]
    para = np.sort(rng.uniform(0.02, 0.3, len(metrics)))
    bt = para + rng.uniform(0.0, 0.08, len(metrics))
    fig, ax = plt.subplots(figsize=(COL_W, 1.9))
    y = np.arange(len(metrics))
    ax.hlines(y, para, bt, color=GRID, lw=2.2, zorder=1)
    ax.plot(para, y, "o", ms=4.2, color=SERIES[0], label="paraphrase", zorder=3)
    ax.plot(bt, y, "s", ms=3.8, mfc="white", mec=SERIES[1], mew=1.1, label="back-translation", zorder=3)
    ax.set_yticks(y, metrics)
    ax.invert_yaxis()
    ax.set_xlim(0, max(bt) * 1.15)
    ax.set_xlabel("Mean |score change| on faithful summaries (lower = more robust)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncols=2)
    save(fig, "fig_robustness", placeholder=True)


def fig_multilabel_confusion(neg):
    fams, _ = families_by_group(neg)
    k = len(fams)
    rng = np.random.default_rng(5)
    m = rng.gamma(0.6, 0.03, (k, k))
    np.fill_diagonal(m, 0)
    m = np.clip(m, 0, 0.35)
    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.98))
    im = ax.imshow(m * 100, cmap="Blues", vmin=0, vmax=35)
    labels = [pretty(f) for f in fams]
    ax.set_xticks(range(k), labels, rotation=90, fontsize=6)
    ax.set_yticks(range(k), labels, fontsize=6)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("Wrongly predicted family", fontsize=7)
    ax.set_ylabel("True family", fontsize=7)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("% of true-family instances", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_visible(False)
    save(fig, "fig_multilabel_confusion", placeholder=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", help="per-summary metric scores CSV (see module docstring)")
    args = ap.parse_args()
    setup_style()
    d, neg = load_benchmark()
    fig_error_family_distribution(neg)
    fig_errors_per_summary(neg)
    fig_llm_detected_errors()
    if args.scores:
        t, placeholder = load_scores(args.scores, neg), False
    else:
        t, placeholder = synthetic_scores(d, neg), True
    fig_per_family_detection(t, neg, placeholder)
    fig_score_distributions(t, placeholder)
    fig_robustness()
    fig_multilabel_confusion(neg)


if __name__ == "__main__":
    main()
