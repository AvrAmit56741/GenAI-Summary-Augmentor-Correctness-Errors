"""N-gram baselines (ROUGE/BLEU/chrF/Token F1) on the real LLM-error dataset (lead #3/#6, docs/tasks.md).

Completes severity.csv (MorphEvalHeb only) with the zero-model-load baselines, on the same 265
unique (premise, generative_summary) pairs from thesisdatasets/error_finder/*.csv. Deliberately run
WITHOUT loading any model, since a 12B translation model is running concurrently in another session
and memory is tight (checked: ~180MB free system-wide before this ran).

Question: do these source-overlap metrics' scores track real-error severity any better (or worse)
than MorphEvalHeb did (severity.csv: rho~0, 9.1% flagged)? N-gram metrics here measure premise-vs-
summary lexical overlap (as in Table 1), not a calibrated faithfulness probability, so "flagged" has
no natural threshold -- report correlation and score-by-severity-bucket instead.

    python exp_real_errors_ngram.py   -> real_errors_ngram.csv, printed summary
"""
import os
import sys

import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eval as R  # noqa: E402
from exp_severity import load_error_finder  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "real_errors_ngram.csv")
METRICS = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU", "chrF", "Token F1"]


def main():
    from summary_augmentation.pages.metrics.ngram_metrics import (RougeCalculator, BleuCalculator,
                                                                   ChrfCalculator, TokenF1Calculator)
    pairs = load_error_finder()
    print(f"unique (premise, generative_summary) pairs: {len(pairs)}")

    rows = []
    for r in pairs.itertuples():
        rg = RougeCalculator(r.premise, r.generative_summary)
        rows.append({
            "doc_id": r.doc_id, "source": r.source, "max_severity": r.max_severity, "n_errors": r.n_errors,
            "ROUGE-1": rg.compute_rouge_n(1)["f1"], "ROUGE-2": rg.compute_rouge_n(2)["f1"],
            "ROUGE-L": rg.compute_rouge_l()["f1"],
            "BLEU": BleuCalculator(r.premise, r.generative_summary).compute_bleu()["bleu"],
            "chrF": ChrfCalculator(r.premise, r.generative_summary).compute_chrf()["chrf"],
            "Token F1": TokenF1Calculator(r.premise, r.generative_summary).compute()["f1"],
        })
    t = pd.DataFrame(rows)
    t.to_csv(OUT, index=False)

    print("\nSpearman rho(severity, score) per metric:")
    for mt in METRICS:
        rho, p = spearmanr(t.max_severity, t[mt])
        print(f"  {mt:10s} rho={rho:+.3f}  p={p:.3f}")

    print("\nMean score by severity bucket:")
    print(t.groupby("max_severity")[METRICS].mean().round(3).to_string())


if __name__ == "__main__":
    main()
