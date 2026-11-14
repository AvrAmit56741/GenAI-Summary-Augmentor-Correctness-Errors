"""MorphEvalHeb local evaluation on the benchmark (combined.csv).

Steps (each caches to results/eval/cache/*.csv and can be re-run / resumed):
    python run_eval.py binary        # MorphEvalHeb binary verifier, server settings (pair input, 3,072 tokens)
    python run_eval.py ngram         # ROUGE-1/2/L, BLEU, chrF, Token F1 (repo implementations, source vs summary)
    python run_eval.py bertscore     # BERTScore F1 with heBERT and AlephBERT (source vs summary, 350-word truncation as on the server)
    python run_eval.py summac        # SummaC-ZS with mDeBERTa-v3-base-mnli-xnli (repo implementation)
    python run_eval.py multilabel    # Setups 2 and 3 on MAFAT-Aug, as served (article/evidence + one claim)
    python run_eval.py merge         # -> results/eval/scores.csv (one row per benchmark pair, one column per metric)

Environment: ~/.cache/morphevalheb_env (torch + transformers 4.53.3). Models: ~/.cache/morphevalheb_models.
"""
import os
import re
import sys
import time

import numpy as np
try:
    import truststore; truststore.inject_into_ssl()   # corporate TLS
except Exception:
    pass
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
BENCH = os.path.join(REPO, "thesisdatasets/train_datasets/Correctness/data/benchmark/combined.csv")
MODELS = os.path.expanduser("~/.cache/morphevalheb_models")
CACHE = os.path.join(HERE, "cache")
os.makedirs(CACHE, exist_ok=True)
sys.path.insert(0, REPO)


def load_bench():
    d = pd.read_csv(BENCH)
    d["uid"] = np.arange(len(d))
    d["part"] = np.where(d.source.notna(), "MAFAT-Aug", "mFACT-He")
    d["y"] = (d.label == "entailment").astype(int)          # 1 = faithful
    d["premise"] = d.premise.astype(str)
    d["hypothesis"] = d.hypothesis.astype(str)
    return d


def device():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def cached(name):
    p = os.path.join(CACHE, f"{name}.csv")
    return (pd.read_csv(p) if os.path.exists(p) else pd.DataFrame(columns=["uid"])), p


def append(path, rows):
    pd.DataFrame(rows).to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def load_cls(name):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    path = os.path.join(MODELS, name)
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.float32)
    model.to(device()).eval()
    return tok, model


# ------------------------------------------------------------------ MorphEvalHeb binary
def run_binary(max_len=3072, bs=4):
    import torch
    d = load_bench()
    done, path = cached("binary")
    todo = d[~d.uid.isin(done.uid)]
    if todo.empty:
        print("binary: cached"); return
    tok, model = load_cls("NLI-hebrew-binary-correctness-metric")
    id2label = model.config.id2label
    ent = [i for i, l in id2label.items() if str(l).lower().startswith("entail")]
    ent = int(ent[0]) if ent else 1
    print("binary id2label", id2label, "-> faithful index", ent, "| todo", len(todo))
    # length-sorted batches: identical predictions to the server's max_length padding, much faster
    lens = todo.premise.str.len() + todo.hypothesis.str.len()
    todo = todo.assign(_l=lens).sort_values("_l")
    t0 = time.time()
    for i in range(0, len(todo), bs):
        b = todo.iloc[i:i + bs]
        enc = tok(list(b.premise), list(b.hypothesis), return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len).to(device())
        with torch.no_grad():
            p = torch.softmax(model(**enc).logits.float(), -1)[:, ent].cpu().numpy()
        append(path, [{"uid": u, "MorphEvalHeb": float(s)} for u, s in zip(b.uid, p)])
        if (i // bs) % 25 == 0:
            n = i + len(b)
            print(f"  binary {n}/{len(todo)}  {(time.time()-t0)/n:.2f}s/pair", flush=True)


# ------------------------------------------------------------------ n-gram baselines (repo code)
def run_ngram():
    from summary_augmentation.pages.metrics.ngram_metrics import (RougeCalculator, BleuCalculator,
                                                                   ChrfCalculator, TokenF1Calculator)
    d = load_bench()
    done, path = cached("ngram")
    rows = []
    for r in d[~d.uid.isin(done.uid)].itertuples():
        rg = RougeCalculator(r.premise, r.hypothesis)
        rows.append({"uid": r.uid,
                     "ROUGE-1": rg.compute_rouge_n(1)["f1"], "ROUGE-2": rg.compute_rouge_n(2)["f1"],
                     "ROUGE-L": rg.compute_rouge_l()["f1"],
                     "BLEU": BleuCalculator(r.premise, r.hypothesis).compute_bleu()["bleu"],
                     "chrF": ChrfCalculator(r.premise, r.hypothesis).compute_chrf()["chrf"],
                     "Token F1": TokenF1Calculator(r.premise, r.hypothesis).compute()["f1"]})
    if rows:
        append(path, rows)
    print("ngram: done", len(rows))


# ------------------------------------------------------------------ BERTScore (server settings)
def run_bertscore():
    from bert_score import score
    from transformers import AutoTokenizer
    d = load_bench()
    for name, col in [("avichr/heBERT", "BERTScore (heBERT)"), ("onlplab/alephbert-base", "BERTScore (AlephBERT)")]:
        done, path = cached("bertscore_" + col.split("(")[1][:-1])
        todo = d[~d.uid.isin(done.uid)]
        if todo.empty:
            continue
        tk = AutoTokenizer.from_pretrained(name)
        # BERT limit is 512 positions incl. [CLS]/[SEP]: cut by tokens, not words (Hebrew words split into many pieces)
        trunc = lambda t: tk.convert_tokens_to_string(tk.tokenize(t)[:500])
        for i in range(0, len(todo), 64):
            b = todo.iloc[i:i + 64]
            _, _, f1 = score([trunc(h) for h in b.hypothesis], [trunc(p) for p in b.premise], model_type=name,
                             num_layers=12, lang="he", device=device(), batch_size=16, verbose=False)
            append(path, [{"uid": u, col: float(v)} for u, v in zip(b.uid, f1.tolist())])
        print("bertscore done", col)


# ------------------------------------------------------------------ SummaC-ZS (repo pipeline)
def run_summac():
    from summary_augmentation.pages.metrics.summac import SummacFactory
    d = load_bench()
    done, path = cached("summac")
    todo = d[~d.uid.isin(done.uid)]
    pipe = SummacFactory.zero_shot("MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
    t0 = time.time()
    for k, r in enumerate(todo.itertuples()):
        res = pipe.score(r.premise, r.hypothesis)
        append(path, [{"uid": r.uid, "SummaC-ZS": float(res.summary_score)}])
        if k % 100 == 0:
            print(f"  summac {k}/{len(todo)} {(time.time()-t0)/(k+1):.2f}s/pair", flush=True)


# ------------------------------------------------------------------ multi-label Setups 2/3, as served
FAMS = None


def split_claims(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def run_multilabel(bs=8):
    import torch
    from summary_augmentation.evidence_engine import rank_evidence
    d = load_bench()
    d = d[d.part == "MAFAT-Aug"]
    for name, col, max_len, use_evidence in [
        ("hebrew-multilabel-correctness-error-detection", "setup2", 3072, False),
        ("hebrew-multilabel-correctness-error-detection-evidence-to-claim", "setup3", 512, True),
    ]:
        done, path = cached("ml_" + col)
        todo = d[~d.uid.isin(done.uid)]
        if todo.empty:
            continue
        tok, model = load_cls(name)
        labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
        print(col, "labels", len(labels), "| todo", len(todo))
        t0 = time.time()
        for k, r in enumerate(todo.itertuples()):
            claims = split_claims(r.hypothesis)
            if use_evidence:
                art_sents = split_claims(r.premise)
                prem = [" ".join(s for s, _ in rank_evidence(c, art_sents, top_k=3, metric="sequence_matcher")) for c in claims]
            else:
                prem = [r.premise] * len(claims)
            probs = []
            for i in range(0, len(claims), bs):
                enc = tok(prem[i:i + bs], claims[i:i + bs], return_tensors="pt", padding=True,
                          truncation=True, max_length=max_len).to(device())
                with torch.no_grad():
                    probs.append(torch.sigmoid(model(**enc).logits.float()).cpu().numpy())
            pmax = np.concatenate(probs).max(0) if probs else np.zeros(len(labels))
            append(path, [{"uid": r.uid, **{f"{col}:{l}": float(v) for l, v in zip(labels, pmax)}}])
            if k % 50 == 0:
                print(f"  {col} {k}/{len(todo)} {(time.time()-t0)/(k+1):.2f}s/summary", flush=True)


def run_factcg():
    """FactCG (Lei et al., 2025) via the shared module used by the server and GUI."""
    from summary_augmentation.pages.metrics.factcg import FactCGScorer
    d = load_bench()
    done, path = cached("factcg")
    todo = d[~d.uid.isin(done.uid)]
    if todo.empty:
        print("factcg: cached"); return
    scorer = FactCGScorer(device=device())
    t0 = time.time()
    for k, r in enumerate(todo.itertuples()):
        res = scorer.score(r.premise, r.hypothesis)
        append(path, [{"uid": r.uid, "FactCG": res.score}])
        if k % 200 == 0:
            print(f"  factcg {k}/{len(todo)} {(time.time()-t0)/(k+1):.2f}s/pair", flush=True)


def merge():
    d = load_bench()[["uid", "part", "source", "label", "y", "error_types", "method", "doc_id"]]
    for f in sorted(x for x in os.listdir(CACHE) if x.endswith(".csv")):
        c = pd.read_csv(os.path.join(CACHE, f)).drop_duplicates("uid", keep="last")
        d = d.merge(c, on="uid", how="left")
    d.to_csv(os.path.join(HERE, "scores.csv"), index=False)
    print("scores.csv", d.shape, "\n", d.notna().sum().to_string())


if __name__ == "__main__":
    {"binary": run_binary, "ngram": run_ngram, "bertscore": run_bertscore, "summac": run_summac,
     "multilabel": run_multilabel, "factcg": run_factcg, "merge": merge}[sys.argv[1]]()
