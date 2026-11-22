"""Read the error before it is diluted: per-token mid-layer probe, max over the summary.

Probe: logistic regression on word vectors from layers_words_{ft}.npz (faithful word = 0, corrupted word = 1),
trained only on articles NOT in the test set. Detector score for a summary = max over summary tokens of the probe
logit at layer L. Compared with MorphEvalHeb's own score on the same faithful vs. one-wrong-word summaries.

    python exp_token_max.py run N     -> cache/token_max.jsonl (first N articles of the single-error set; chunks of 20)
    python exp_token_max.py analyze   -> token_max_summary.csv
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cache", "token_max.jsonl")
LAYERS = [8, 12, 16, 20, 24]


def probes(test_docs):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    w = pd.read_csv(os.path.join(HERE, "layers_words_sample.csv"))
    z = np.load(os.path.join(HERE, "cache", "layers_words_ft.npz"))
    doc = w.doc_id.values[z["row"]]
    keep = ~np.isin(doc, list(test_docs)) & ~np.isnan(z["g"][:, 0, 0]) & ~np.isnan(z["c"][:, 0, 0])
    P = {}
    for L in LAYERS:
        X = np.vstack([z["g"][keep, L - 1], z["c"][keep, L - 1]]).astype(np.float32)
        y = np.r_[np.zeros(keep.sum()), np.ones(keep.sum())]
        sc = StandardScaler().fit(X)
        clf = LogisticRegression(C=0.05, max_iter=2000).fit(sc.transform(X), y)
        P[L] = (sc.mean_, sc.scale_, clf.coef_[0], clf.intercept_[0])
    print(f"probes trained on {keep.sum()} word pairs (test articles excluded)", flush=True)
    return P


def run(n=20, chunk=20):
    import torch
    import run_eval as R
    from exp_layers import MAX_LEN
    from exp_single_edit_baselines import items
    it = items().head(n)                      # same order as exp_single_edit_baselines (first n units)
    done = {(j["unit_id"], j["version"]) for j in map(json.loads, open(OUT))} if os.path.exists(OUT) else set()
    P = probes(set(items().doc_id))            # exclude ALL single-error test articles, not only these n
    tok, model = R.load_cls("NLI-hebrew-binary-correctness-metric")
    dev = R.device()
    buf = []
    for k, r in enumerate(it.itertuples()):
        for ver in ["gold", "single"]:
            if (r.unit_id, ver) in done:
                continue
            e = tok(r.premise, getattr(r, ver), return_tensors="pt", truncation="only_first", max_length=MAX_LEN)
            seq = np.array([s if s is not None else -1 for s in e.sequence_ids(0)])
            x = {a: b.to(dev) for a, b in e.items() if a != "token_type_ids"}
            with torch.no_grad():
                o = model(**x, output_hidden_states=True)
            p = torch.softmax(o.logits, -1)[0, 1].item()
            rec = {"unit_id": int(r.unit_id), "kind": r.kind, "version": ver, "p": p, "n_tok": int((seq == 1).sum())}
            for L in LAYERS:
                H = o.hidden_states[L - 1][0][seq == 1].float().cpu().numpy()
                mu, sd, wv, b = P[L]
                s = ((H - mu) / sd) @ wv + b
                rec[f"max@{L}"] = float(s.max())
                rec[f"mean@{L}"] = float(s.mean())
            buf.append(rec)
        if (k + 1) % chunk == 0 or k + 1 == len(it):
            with open(OUT, "a") as f:
                for b in buf:
                    f.write(json.dumps(b) + "\n")
            buf = []
            print(f"  {k + 1}/{len(it)} articles", flush=True)
            analyze()


def analyze():
    from sklearn.metrics import roc_auc_score
    d = pd.DataFrame([json.loads(x) for x in open(OUT)]).drop_duplicates(["unit_id", "version"], keep="last")
    g = d[d.version == "gold"].set_index("unit_id")
    s = d[d.version == "single"].set_index("unit_id")
    ids = g.index.intersection(s.index)
    g, s = g.loc[ids], s.loc[ids]
    from exp_single_edit_baselines import items
    doc = items().set_index("unit_id").doc_id.loc[ids].values
    y = np.r_[np.zeros(len(ids)), np.ones(len(ids))]
    rng = np.random.default_rng(0)
    udocs = np.unique(doc)
    boots = [np.concatenate([np.where(doc == u)[0] for u in rng.choice(udocs, len(udocs))]) for _ in range(1000)]

    def row(name, a, b):                      # a = gold scores, b = single scores; higher = more suspicious
        auc = roc_auc_score(y, np.r_[a, b])
        bs = [roc_auc_score(np.r_[np.zeros(len(i)), np.ones(len(i))], np.r_[a[i], b[i]]) for i in boots]
        thr = np.quantile(a, .95)
        return {"score": name, "auc": auc, "auc_lo": np.percentile(bs, 2.5), "auc_hi": np.percentile(bs, 97.5),
                "flag_at_5fa": (b > thr).mean(), "single_above_gold": (b > a).mean()}
    rows = [row("MorphEvalHeb (1 - p)", -g.p.values, -s.p.values)]
    for L in LAYERS:
        for agg in ["max", "mean"]:
            c = f"{agg}@{L}"
            rows.append(row(f"probe {agg} over tokens, L{L}", g[c].values, s[c].values))
    t = pd.DataFrame(rows).assign(n=len(ids), n_articles=len(udocs))
    t.to_csv(os.path.join(HERE, "token_max_summary.csv"), index=False)
    print(t.round(3).to_string(index=False), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(int(sys.argv[2]) if len(sys.argv) > 2 else 20)
    else:
        analyze()
