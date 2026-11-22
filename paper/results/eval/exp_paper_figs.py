"""Paper figures for the single-error and inside-the-model findings (reads saved results only, no models).

    python exp_paper_figs.py   -> ../../overleaf/figures/fig_single_error.{pdf,png},
                                  fig_probe_patching.{pdf,png}, fig_erasure.{pdf,png}
"""
import json
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "overleaf", "figures")
BLUE, ORANGE, AQUA, YELLOW, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#a3a29c"
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": .6, "axes.axisbelow": True,
                     "legend.frameon": False, "pdf.fonttype": 42})
NAME = {"MorphEvalHeb": "MorphEvalHeb (ours)", "FactCG": "FactCG", "SummaC-ZS": "SummaC-ZS"}


def save(fig, name):
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def fig_single_error():
    t = pd.read_csv(os.path.join(HERE, "single_edit_baselines_summary.csv"))
    s = pd.DataFrame([json.loads(x) for x in open(os.path.join(HERE, "cache", "single_edit_baselines.jsonl"))])
    p = s.drop_duplicates(["unit_id", "version", "metric"], keep="last").pivot_table(
        index=["unit_id", "metric"], columns="version", values="v").reset_index()
    mets = ["MorphEvalHeb", "FactCG", "SummaC-ZS"]
    n = int(t.n.iloc[0])
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw={"width_ratios": [1.1, 1]})
    a = ax[0]
    x = np.arange(len(mets))
    for j, (ver, lab, c) in enumerate([("multi", "Injected summary (many errors)", ORANGE),
                                       ("single", "Same summary, one wrong word", BLUE)]):
        v = t[t.version == ver].set_index("metric").loc[mets]
        a.bar(x + (j - .5) * .36, v.flagged, .34, color=c, label=lab,
              yerr=[v.flagged - v.ci_lo, v.ci_hi - v.flagged], ecolor=INK2, capsize=2, error_kw={"lw": .8})
        for xi, f in zip(x + (j - .5) * .36, v.flagged):
            a.text(xi, f + .04 if f < .9 else f - .1, f"{f:.0%}", ha="center", fontsize=7,
                   color="white" if f >= .9 else INK)
    a.set_xticks(x, [NAME[m] for m in mets])
    a.set(ylim=(0, 1.08), ylabel="share flagged as unfaithful", yticks=[0, .25, .5, .75, 1])
    a.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    a.set_title(f"(a) Detection at 5% false alarms (n={n} articles)", fontsize=8, loc="left")
    a.legend(loc="upper right", fontsize=7)
    a = ax[1]
    rows = []
    for m in mets:
        g = p[p.metric == m]
        rows.append([(g.single < g.gold).mean(), (g.single == g.gold).mean(), (g.single > g.gold).mean()])
    R = np.array(rows)
    left = np.zeros(len(mets))
    for k, (lab, c) in enumerate([("score drops", BLUE), ("identical score", GRAY), ("score rises", ORANGE)]):
        a.barh(np.arange(len(mets)), R[:, k], left=left, color=c, label=lab, height=.55, edgecolor="#fcfcfb", lw=1)
        for i in range(len(mets)):
            if R[i, k] >= .12:
                a.text(left[i] + R[i, k] / 2, i, f"{R[i, k]:.0%}", ha="center", va="center", fontsize=7,
                       color="white" if c != GRAY else INK)
        left += R[:, k]
    a.set_yticks(range(len(mets)), [NAME[m] for m in mets])
    a.invert_yaxis()
    a.set(xlim=(0, 1), xlabel="share of one-wrong-word summaries")
    a.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    a.grid(axis="y", visible=False)
    a.set_title("(b) Score vs. the faithful summary", fontsize=8, loc="left")
    a.legend(loc="upper center", bbox_to_anchor=(.5, -.28), ncol=3, fontsize=7)
    plt.tight_layout()
    save(fig, "fig_single_error")


def fig_probe_patching():
    t = pd.read_csv(os.path.join(HERE, "layers_probe.csv"))
    pv = t.pivot_table(index="layer", columns=["model", "pool"], values="paired")
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.5))
    a = ax[0]
    a.plot(pv.index, pv[("ft", "cls")], color=BLUE, lw=2, label="MorphEvalHeb, [CLS]")
    a.plot(pv.index, pv[("base", "cls")], color=ORANGE, lw=2, label="NeoDictaBERT base, [CLS]")
    a.plot(pv.index, pv[("ft", "hyp")], color=GRAY, lw=1.2, ls="--", label="either model, mean of summary tokens")
    a.axhline(.5, color=INK2, lw=.8, ls=":")
    a.text(1.5, .51, "chance", ha="left", fontsize=7, color=INK2)
    a.set(xlabel="layer", ylabel="probe paired accuracy", ylim=(.3, 1.0), xlim=(1, 28), xticks=[1, 4, 8, 12, 16, 20, 24, 28])
    a.set_title(f"(a) Linear probe: faithful vs. corrupted (n={int(t.n_units.iloc[0])})", fontsize=8, loc="left")
    a.legend(loc="lower right", fontsize=6.5)
    rows = [json.loads(x) for x in open(os.path.join(HERE, "cache", "patching.jsonl"))]
    ok = [r for r in rows if r["status"] == "ok"]
    d = pd.DataFrame([{"u": r["unit_id"], "pos": k.split("@")[0], "layer": int(k.split("@")[1]), "v": v}
                      for r in ok for k, v in r["restore"].items()])
    a = ax[1]
    rng = np.random.default_rng(0)
    for pos, lab, c in [("edit", "edited word", BLUE), ("cls", "[CLS]", ORANGE),
                        ("hyp_rest", "rest of summary", AQUA), ("article", "article", GRAY)]:
        g = d[d.pos == pos].pivot(index="u", columns="layer", values="v")
        m = g.mean()
        bs = np.array([g.iloc[rng.integers(0, len(g), len(g))].mean().values for _ in range(1000)])
        lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
        a.plot(m.index, m.values, color=c, lw=2, marker="o", ms=3, label=lab)
        a.fill_between(m.index, lo, hi, color=c, alpha=.15, lw=0)
    a.axhline(0, color=INK2, lw=.8, ls=":")
    a.set(xlabel="layer", ylabel="share of faithful score restored", xlim=(1, 28), xticks=[1, 4, 7, 10, 13, 16, 19, 22, 25, 28])
    a.set_title(f"(b) Activation patching, one-word minimal pairs (n={len(ok)})", fontsize=8, loc="left")
    a.legend(loc="center left", fontsize=6.5)
    plt.tight_layout()
    save(fig, "fig_probe_patching")


def fig_erasure():
    full = pd.read_csv(os.path.join(HERE, "layers_erase_summary.csv"))
    orth = pd.read_csv(os.path.join(HERE, "layers_erase_summary_orth.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw={"width_ratios": [1, 1.15]})
    a = ax[0]
    dirs = [("none", "none"), ("random", "random"), ("negation", "negation"), ("number", "number"),
            ("agreement_suffix", "agreement"), ("all_types", "shared")]
    acc = [(g.paired * g.n).sum() / g.n.sum() for g in (full[full.dir == k] for k, _ in dirs)]
    cols = [GRAY, GRAY, BLUE, BLUE, BLUE, ORANGE]
    a.bar(range(len(dirs)), acc, .6, color=cols)
    for i, v in enumerate(acc):
        a.text(i, v + .02, f"{v:.2f}", ha="center", fontsize=7)
    a.axhline(.5, color=INK2, lw=.8, ls=":")
    a.set_xticks(range(len(dirs)), [l for _, l in dirs], rotation=30, ha="right")
    a.set(ylim=(0, 1.08), ylabel="paired accuracy\n(dotted = chance)", xlabel="direction erased at all 28 layers")
    a.grid(axis="x", visible=False)
    a.set_title(f"(a) Erasing one direction (n={int(full[full.dir == 'none'].n.sum())} pairs)", fontsize=8, loc="left")
    a = ax[1]
    tests = ["negation", "number", "agreement_suffix", "other"]
    rws = ["negation", "number", "agreement_suffix", "random"]
    lab = {"agreement_suffix": "agreement"}
    pv = orth.pivot(index="dir", columns="rule", values="flagged")
    D = (pv.loc[rws, tests] - pv.loc["none", tests]).values
    im = a.imshow(D, cmap="RdBu", vmin=-.6, vmax=.6, aspect="auto")
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            a.text(j, i, f"{D[i, j]:+.2f}", ha="center", va="center", fontsize=7,
                   color="white" if abs(D[i, j]) > .35 else INK, fontweight="bold" if i == j else "normal")
    n = orth.groupby("rule").n.first()
    a.set_xticks(range(len(tests)), [f"{lab.get(c, c)}\n(n={n[c]})" for c in tests])
    a.set_yticks(range(len(rws)), [lab.get(r, r) for r in rws])
    a.grid(False)
    a.set(xlabel="error type of test sentences", ylabel="type-specific part erased")
    a.set_title("(b) Type-specific part only: change in share flagged", fontsize=8, loc="left")
    cb = plt.colorbar(im, ax=a, shrink=.85)
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_visible(False)
    plt.tight_layout()
    save(fig, "fig_erasure")


if __name__ == "__main__":
    fig_single_error()
    fig_probe_patching()
    fig_erasure()
