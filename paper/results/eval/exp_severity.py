"""Severity calibration on real LLM output (research lead #6, docs/tasks.md).

Does MorphEvalHeb's unfaithfulness probability track the *severity* of *naturally occurring*
LLM errors (thesisdatasets/error_finder/*.csv, Gemini-identified errors, severity 1-5), not just
the binary label of *injected* errors it was trained on?

Uses `generative_summary` (the real LLM output), NOT `ai_summary` (misaligned/unrelated text in
these files) -- see docs/tasks.md lead #6 note.

    python exp_severity.py   -> results/eval/severity.csv, printed summary
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import pointbiserialr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eval as R  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "severity.csv")
ERROR_FINDER = os.path.join(R.REPO, "thesisdatasets/error_finder")


def load_error_finder():
    files = glob.glob(os.path.join(ERROR_FINDER, "*.csv"))
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d = d.dropna(subset=["generative_summary", "premise", "severity"])
    # one row per real error; take the worst (max) severity per unique summary for the pair-level score
    agg = (d.groupby(["doc_id", "source", "premise", "generative_summary"])
             .agg(max_severity=("severity", "max"), n_errors=("severity", "size"),
                  error_names=("correctness_error_name", lambda x: "; ".join(x.astype(str))))
             .reset_index())
    return agg


def main():
    pairs = load_error_finder()
    print(f"unique (premise, generative_summary) pairs: {len(pairs)}")
    tok, model = R.load_cls("NLI-hebrew-binary-correctness-metric")
    id2label = model.config.id2label
    ent = [i for i, l in id2label.items() if str(l).lower().startswith("entail")]
    ent = int(ent[0]) if ent else 1
    dev = R.device()

    scores = []
    for r in pairs.itertuples():
        enc = tok(r.premise, r.generative_summary, return_tensors="pt", truncation="only_first",
                  max_length=3072).to(dev)
        with torch.no_grad():
            p = torch.softmax(model(**enc).logits.float(), -1)[0, ent].item()
        scores.append(1 - p)  # unfaithfulness probability
    pairs["unfaithful_prob"] = scores
    pairs.to_csv(OUT, index=False)

    rho, p_rho = spearmanr(pairs.max_severity, pairs.unfaithful_prob)
    print(f"\nn={len(pairs)}  mean unfaithful_prob={pairs.unfaithful_prob.mean():.3f}")
    print(f"Spearman rho(severity, unfaithful_prob) = {rho:.3f}  (p={p_rho:.4f})")

    lo = pairs[pairs.max_severity <= 2].unfaithful_prob
    hi = pairs[pairs.max_severity >= 4].unfaithful_prob
    print(f"severity<=2 (n={len(lo)}): mean unfaithful_prob={lo.mean():.3f}")
    print(f"severity>=4 (n={len(hi)}): mean unfaithful_prob={hi.mean():.3f}")

    flagged = (pairs.unfaithful_prob > 0.5)
    print(f"\nflagged-as-unfaithful rate by severity:")
    print(pairs.assign(flagged=flagged).groupby("max_severity").flagged.mean().to_string())


if __name__ == "__main__":
    main()
