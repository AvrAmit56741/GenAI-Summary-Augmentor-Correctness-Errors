"""Meta-evaluation statistics for every metric column in scores.csv.

    python analyze.py            -> results/eval/meta_eval.json, meta_eval_table.tex, per_family.csv

Protocol (paper Section 6.3):
  * Kendall's tau_b and ROC AUC: threshold-free, on all benchmark pairs.
  * Balanced accuracy: the threshold is tuned to maximise balanced accuracy on the other folds
    (5-fold, grouped by article) and applied to the held-out fold -- no tuning on test data.
  * 95% CIs: percentile bootstrap over article-grouped resamples (B = 1000).
  * Significance vs MorphEvalHeb: paired bootstrap on AUC (same resamples), one-sided, Holm-corrected.
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
B = 1000
rng = np.random.default_rng(13)
META = {"uid", "part", "source", "label", "y", "error_types", "method", "doc_id"}
OURS = "MorphEvalHeb"


def bacc(y, s, t):
    p = s >= t
    tpr = (p & (y == 1)).sum() / max((y == 1).sum(), 1)
    tnr = (~p & (y == 0)).sum() / max((y == 0).sum(), 1)
    return (tpr + tnr) / 2


def best_t(y, s):
    cands = np.unique(np.quantile(s, np.linspace(0, 1, 201)))
    return max(cands, key=lambda t: bacc(y, s, t))


def cv_bacc(y, s, groups, k=5):
    ug = np.unique(groups)
    rng_local = np.random.default_rng(0)
    fold_of = dict(zip(ug, rng_local.integers(0, k, len(ug))))
    f = np.array([fold_of[g] for g in groups])
    pred = np.zeros(len(y), dtype=bool)
    for i in range(k):
        tr, te = f != i, f == i
        if te.sum() == 0:
            continue
        pred[te] = s[te] >= best_t(y[tr], s[tr])
    tpr = (pred & (y == 1)).sum() / (y == 1).sum()
    tnr = (~pred & (y == 0)).sum() / (y == 0).sum()
    return (tpr + tnr) / 2


def grouped_boot_idx(groups):
    ug, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == g)[0] for g in range(len(ug))]
    for _ in range(B):
        pick = rng.integers(0, len(ug), len(ug))
        yield np.concatenate([members[g] for g in pick])


def evaluate(df, metrics):
    y = df.y.values
    groups = df.group.values
    idxs = list(grouped_boot_idx(groups))
    out = {}
    aucs = {}
    for m in metrics:
        s = df[m].values.astype(float)
        ok = ~np.isnan(s)
        if ok.sum() < len(s):
            print(f"  {m}: {(~ok).sum()} missing scores -> dropped")
        yy, ss, gg = y[ok], s[ok], groups[ok]
        boot_auc = np.array([roc_auc_score(y[i], s[i]) if np.isfinite(s[i]).all() and 0 < y[i].sum() < len(i) else np.nan for i in idxs])
        boot_tau = np.array([kendalltau(s[i], y[i]).statistic if np.isfinite(s[i]).all() else np.nan for i in idxs])
        aucs[m] = boot_auc
        out[m] = {"n": int(ok.sum()),
                  "tau_b": float(kendalltau(ss, yy).statistic), "tau_b_ci": np.nanpercentile(boot_tau, [2.5, 97.5]).tolist(),
                  "auc": float(roc_auc_score(yy, ss)), "auc_ci": np.nanpercentile(boot_auc, [2.5, 97.5]).tolist(),
                  "bacc_cv": float(cv_bacc(yy, ss, gg))}
    if OURS in aucs:
        others = [m for m in metrics if m != OURS]
        raw = {m: float(np.nanmean(aucs[OURS] - aucs[m] <= 0)) for m in others}
        order = sorted(others, key=lambda m: raw[m])
        adj, running = {}, 0.0
        for r, m in enumerate(order):                       # Holm step-down
            running = max(running, min(1.0, (len(order) - r) * raw[m]))
            adj[m] = running
        for m in others:
            out[m]["p_vs_ours_holm"] = adj[m]
    return out


def main():
    df = pd.read_csv(os.path.join(HERE, "scores.csv"))
    df["group"] = np.where(df.doc_id.notna(), df.doc_id.astype(str), "u" + df.uid.astype(str))
    metrics = [c for c in df.columns if c not in META | {"group"} and not c.startswith(("setup2:", "setup3:"))]
    res = {"all": evaluate(df, metrics)}
    for part, g in df.groupby("part"):
        res[part] = evaluate(g.reset_index(drop=True), metrics)
    with open(os.path.join(HERE, "meta_eval.json"), "w") as fh:
        json.dump(res, fh, indent=1)

    # Per-family detection on MAFAT-Aug negatives, threshold tuned out-of-fold as above.
    mafat = df[df.part == "MAFAT-Aug"].reset_index(drop=True)
    neg = mafat[(mafat.y == 0) & mafat.error_types.notna()].copy()
    neg["fams"] = neg.error_types.str.split(",").apply(lambda xs: {x.strip() for x in xs})
    rows = []
    for m in metrics:
        t = best_t(mafat.y.values, mafat[m].values)
        flagged = neg[m].values < t
        for f in sorted(set().union(*neg.fams)):
            sel = neg.fams.apply(lambda s: f in s).values
            rows.append({"metric": m, "family": f, "n": int(sel.sum()), "detection_rate": float(flagged[sel].mean())})
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "per_family.csv"), index=False)

    # Console + LaTeX
    def fmt(r, k):
        return f"{r[k]:.3f}"
    lines = []
    order = sorted(metrics, key=lambda m: -res["all"][m]["auc"])
    for m in order:
        a = res["all"][m]
        star = "" if m == OURS or a.get("p_vs_ours_holm", 1) < 0.05 else "$^{\\dagger}$"
        cells = [m.replace("_", "\\_") + star]
        for part in ("all", "MAFAT-Aug", "mFACT-He"):
            r = res[part][m]
            cells += [fmt(r, "auc"), fmt(r, "bacc_cv"), fmt(r, "tau_b")]
        lines.append(" & ".join(cells) + " \\\\")
        print(f"{m:24s} AUC {a['auc']:.3f} [{a['auc_ci'][0]:.3f},{a['auc_ci'][1]:.3f}]  BAcc {a['bacc_cv']:.3f}  "
              f"tau_b {a['tau_b']:.3f}  | MAFAT AUC {res['MAFAT-Aug'][m]['auc']:.3f}  mFACT AUC {res['mFACT-He'][m]['auc']:.3f}"
              + (f"  p={a['p_vs_ours_holm']:.3g}" if "p_vs_ours_holm" in a else ""))
    with open(os.path.join(HERE, "meta_eval_table.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
