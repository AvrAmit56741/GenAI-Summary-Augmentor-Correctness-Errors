"""Compute paper-ready results from the prediction files published with the MorphEvalHeb models on Hugging Face.

    uv run --no-project --with pandas --with scikit-learn --with scipy python analyze_hf_results.py

Inputs (downloaded to results/hf/ from huggingface.co/Amit5674/...):
    NLI-hebrew-binary-correctness-metric/predictions.csv                        binary verifier, test split
    hebrew-multilabel-correctness-error-detection-evidence-to-claim/predictions.csv   multi-label Setup 3, test split
Outputs: results/binary_results.json, results/setup3_per_family.csv, printed summary.
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import (balanced_accuracy_score, f1_score, precision_recall_fscore_support,
                             roc_auc_score, accuracy_score)

HERE = os.path.dirname(os.path.abspath(__file__))
HF = os.path.join(HERE, "hf")
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
BENCH = os.path.join(REPO, "thesisdatasets/train_datasets/Correctness/data/benchmark/combined.csv")
B = 1000
rng = np.random.default_rng(42)


def boot_ci(fn, *arrays):
    n = len(arrays[0])
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        try:
            vals.append(fn(*[a[idx] for a in arrays]))
        except ValueError:
            continue
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def binary():
    p = pd.read_csv(os.path.join(HF, "NLI-hebrew-binary-correctness-metric", "predictions.csv"))
    y = (p.true_label == "entailment").astype(int).values          # 1 = faithful
    s = p.prob_entailment.values                                    # MorphEvalHeb score
    yhat = (p.predicted_label == "entailment").astype(int).values
    res = {
        "n": int(len(p)), "n_faithful": int(y.sum()), "n_unfaithful": int((1 - y).sum()),
        "accuracy": accuracy_score(y, yhat),
        "f1_unfaithful_as_positive": f1_score(1 - y, 1 - yhat),
        "f1_faithful_as_positive": f1_score(y, yhat),
        "balanced_accuracy_argmax": balanced_accuracy_score(y, yhat),
        "roc_auc": roc_auc_score(y, s),
        "kendall_tau_b": kendalltau(s, y).statistic,
    }
    res["roc_auc_ci95"] = boot_ci(roc_auc_score, y, s)
    res["balanced_accuracy_ci95"] = boot_ci(balanced_accuracy_score, y, yhat)
    res["kendall_tau_b_ci95"] = boot_ci(lambda a, b: kendalltau(b, a).statistic, y, s)

    # Match test pairs against the benchmark file to recover source / error families.
    bench = pd.read_csv(BENCH)
    key = lambda df: df.premise.astype(str).str.strip() + "\x1f" + df.hypothesis.astype(str).str.strip()
    bench = bench.assign(_k=key(bench)).drop_duplicates("_k")
    m = p.assign(_k=key(p)).merge(bench[["_k", "source", "error_types", "method", "doc_id"]], on="_k", how="left")
    m["part"] = np.where(m.source.notna(), "MAFAT-Aug", np.where(m._k.isin(bench._k), "benchmark-no-source", "not-in-combined.csv"))
    res["test_rows_matched_in_combined_csv"] = int(m._k.isin(bench._k).sum())
    by_part = {}
    for part, g in m.groupby("part"):
        yy = (g.true_label == "entailment").astype(int).values
        ss = g.prob_entailment.values
        d = {"n": int(len(g)), "faithful": int(yy.sum()),
             "accuracy": accuracy_score(yy, (g.predicted_label == "entailment").astype(int))}
        if 0 < yy.sum() < len(yy):
            d["roc_auc"] = roc_auc_score(yy, ss)
            d["kendall_tau_b"] = kendalltau(ss, yy).statistic
        by_part[part] = d
    res["by_part"] = by_part

    # Per-family detection rate on MAFAT-Aug negatives (argmax decision).
    neg = m[(m.true_label == "contradiction") & m.error_types.notna()].copy()
    neg["fams"] = neg.error_types.str.split(",").apply(lambda xs: {x.strip() for x in xs if x.strip()})
    fam_rows = []
    for f in sorted(set().union(*neg.fams)) if len(neg) else []:
        g = neg[neg.fams.apply(lambda s_: f in s_)]
        fam_rows.append({"family": f, "n_negatives": len(g),
                         "detection_rate": float((g.predicted_label == "contradiction").mean())})
    res["per_family_detection_mafat_aug"] = fam_rows
    m.drop(columns="_k").to_csv(os.path.join(HERE, "binary_predictions_with_metadata.csv"), index=False)
    return res


def setup3():
    p = pd.read_csv(os.path.join(HF, "hebrew-multilabel-correctness-error-detection-evidence-to-claim", "predictions.csv"))
    fams = [c for c in p.columns if c not in ("premise", "hypothesis", "ground_truth_error_types")]
    gt = p.ground_truth_error_types.fillna("").astype(str).str.split(",").apply(lambda xs: {x.strip() for x in xs if x.strip()})
    Y = np.array([[f in s for f in fams] for s in gt], dtype=int)
    P = (p[fams].values >= 0.5).astype(int)
    pr, rc, f1, sup = precision_recall_fscore_support(Y, P, average=None, zero_division=0)
    per = pd.DataFrame({"family": fams, "support": sup, "precision": pr, "recall": rc, "f1": f1}).sort_values("f1", ascending=False)
    per.to_csv(os.path.join(HERE, "setup3_per_family.csv"), index=False)
    micro = precision_recall_fscore_support(Y, P, average="micro", zero_division=0)[:3]
    macro = precision_recall_fscore_support(Y, P, average="macro", zero_division=0)[:3]
    clean = Y.sum(1) == 0
    return {"n_claims": int(len(p)), "n_clean_claims": int(clean.sum()), "n_families": len(fams),
            "micro_p_r_f1": [float(x) for x in micro], "macro_p_r_f1": [float(x) for x in macro],
            "false_alarm_rate_on_clean_claims": float((P[clean].sum(1) > 0).mean()),
            "per_family": per.round(3).to_dict("records")}


if __name__ == "__main__":
    out = {"binary": binary(), "setup3": setup3()}
    with open(os.path.join(HERE, "hf_results.json"), "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False, default=float)
    b, s3 = out["binary"], out["setup3"]
    print(json.dumps({k: v for k, v in b.items() if k != "per_family_detection_mafat_aug"}, indent=1, default=lambda x: round(float(x), 4)))
    print("per-family detection (MAFAT-Aug test negatives):")
    for r in b["per_family_detection_mafat_aug"]:
        print(f"  {r['family']:36s} n={r['n_negatives']:4d}  {r['detection_rate']:.3f}")
    print(json.dumps({k: v for k, v in s3.items() if k != "per_family"}, indent=1))
    print(pd.DataFrame(s3["per_family"]).to_string(index=False))
