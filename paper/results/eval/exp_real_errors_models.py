"""Model-dependent baselines on the real LLM-error dataset (completes lead #6, docs/tasks.md).

Severity calibration (exp_severity.py: MorphEvalHeb, exp_real_errors_ngram.py: n-gram baselines) was
split in two because the model-dependent baselines (ENFS, SummaC-ZS, BERTScore, FactCG) needed a model
load that was deferred while a 12B translation model was running concurrently in another session.
Memory is free now (13G+ unused) -- this script scores the same 265 unique (premise,
generative_summary) pairs with the remaining baselines, one model at a time, so no two large models
are resident together.

Question: same as exp_real_errors_ngram.py -- do these baselines' scores track real-error severity
any better than MorphEvalHeb did (ρ≈0, 9.1% flagged)?

    python exp_real_errors_models.py   -> real_errors_models.csv, printed summary
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eval as R  # noqa: E402
from exp_severity import load_error_finder  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "real_errors_models.csv")


def _progress(name, i, n, t0):
    if i % 50 == 0 or i == n - 1:
        print(f"  {name} {i + 1}/{n}  ({(time.time() - t0) / (i + 1):.2f}s/pair)", flush=True)


def run_bertscore(pairs):
    from bert_score import score
    from transformers import AutoTokenizer
    print(f"[BERTScore] device={R.device()}", flush=True)
    tk = AutoTokenizer.from_pretrained("avichr/heBERT")
    trunc = lambda t: tk.convert_tokens_to_string(tk.tokenize(t)[:500])
    out = np.zeros(len(pairs))
    t0 = time.time()
    for i in range(0, len(pairs), 32):
        b = pairs.iloc[i:i + 32]
        _, _, f1 = score([trunc(h) for h in b.generative_summary], [trunc(p) for p in b.premise],
                          model_type="avichr/heBERT", num_layers=12, lang="he", device=R.device(),
                          batch_size=16, verbose=False)
        out[i:i + 32] = f1.tolist()
        print(f"  BERTScore {min(i + 32, len(pairs))}/{len(pairs)}  ({(time.time() - t0):.0f}s elapsed)", flush=True)
    return out


def run_summac(pairs):
    from summary_augmentation.pages.metrics.summac import SummacFactory, _auto_device
    print(f"[SummaC-ZS] device={_auto_device()}", flush=True)
    pipe = SummacFactory.zero_shot("MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
    out = []
    t0 = time.time()
    for i, r in enumerate(pairs.itertuples()):
        out.append(pipe.score(r.premise, r.generative_summary).summary_score)
        _progress("SummaC-ZS", i, len(pairs), t0)
    return np.array(out)


def run_factcg(pairs):
    from summary_augmentation.pages.metrics.factcg import FactCGScorer
    print(f"[FactCG] device={R.device()}", flush=True)
    scorer = FactCGScorer(device=R.device())
    out = []
    t0 = time.time()
    for i, r in enumerate(pairs.itertuples()):
        out.append(scorer.score(r.premise, r.generative_summary).score)
        _progress("FactCG", i, len(pairs), t0)
    return np.array(out)


def run_enfs(pairs):
    from summary_augmentation.pages.metrics.misc.hallucinations_metrics import (
        load_ner_pipeline, extract_entities, calculate_enfs, _auto_device)
    print(f"[ENFS/NER] device={_auto_device()}", flush=True)
    ner = load_ner_pipeline("dicta-il/dictabert-ner")
    out = []
    t0 = time.time()
    for i, r in enumerate(pairs.itertuples()):
        pe = extract_entities(r.premise, ner, 0.5)
        he = extract_entities(r.generative_summary, ner, 0.5)
        enfs, n_ent, _, _ = calculate_enfs(pe, he)
        out.append(0.0 if enfs is None else enfs)
        _progress("ENFS", i, len(pairs), t0)
    return np.array(out)


def main():
    pairs = load_error_finder()
    print(f"unique (premise, generative_summary) pairs: {len(pairs)}")

    results = {}
    for name, fn in [("BERTScore (heBERT)", run_bertscore), ("SummaC-ZS", run_summac),
                      ("FactCG", run_factcg), ("ENFS", run_enfs)]:
        print(f"running {name}...", flush=True)
        results[name] = fn(pairs)

    t = pairs[["doc_id", "source", "max_severity", "n_errors"]].copy()
    for name, vals in results.items():
        t[name] = vals
    t.to_csv(OUT, index=False)

    print("\nSpearman rho(severity, score) per metric:")
    for name in results:
        rho, p = spearmanr(t.max_severity, t[name])
        print(f"  {name:20s} rho={rho:+.3f}  p={p:.3f}")

    print("\nMean score by severity bucket:")
    print(t.groupby("max_severity")[list(results.keys())].mean().round(3).to_string())


if __name__ == "__main__":
    main()
