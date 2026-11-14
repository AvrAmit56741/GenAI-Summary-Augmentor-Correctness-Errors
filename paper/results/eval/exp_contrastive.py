"""Contrastive-embedding feasibility probe (research lead #7, docs/tasks.md).

Before considering a contrastive-loss retrain (expensive, needs the GPU server), a cheap check:
do faithful/unfaithful pairs already separate in the *existing* cross-entropy-trained binary
verifier's own embedding space (pooled final-layer hidden state)? If yes, a contrastive objective
is unlikely to add much; if not, that's a concrete argument *for* trying it.

    python exp_contrastive.py [N]   -> printed summary, results/eval/contrastive_probe.json
"""
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eval as R  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "contrastive_probe.json")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1200


def main():
    d = R.load_bench()
    per_group = max(1, N // 4)
    n_per_group = min(int(d.groupby(["part", "y"]).size().min()), per_group)
    sample = d.groupby(["part", "y"]).sample(n=n_per_group, random_state=0).reset_index(drop=True)
    print(f"sample: {len(sample)} pairs, part x y = {sample.groupby(['part','y']).size().to_dict()}")

    tok, model = R.load_cls("NLI-hebrew-binary-correctness-metric")
    dev = R.device()
    embs = []
    with torch.no_grad():
        for r in sample.itertuples():
            enc = tok(r.premise, r.hypothesis, return_tensors="pt", truncation="only_first",
                      max_length=3072).to(dev)
            out = model(**enc, output_hidden_states=True)
            last = out.hidden_states[-1][0]        # (seq, dim)
            pooled = last.mean(0).float().cpu().numpy()  # mean-pool, no training-time CLS-head dependence
            embs.append(pooled)
    X = np.stack(embs)
    y = sample.y.values

    # 1) does a *linear* probe on the frozen embedding separate faithful/unfaithful? (5-fold CV AUC)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    aucs = []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    cv_auc = float(np.mean(aucs))

    # 2) unsupervised separation: silhouette score of the two label clusters in embedding space
    sil = float(silhouette_score(X, y))

    # 3) mean cosine distance: within-class vs between-class
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    sim = Xn @ Xn.T
    same = y[:, None] == y[None, :]
    np.fill_diagonal(same, False)
    within = sim[same].mean()
    between = sim[~same & ~np.eye(len(y), dtype=bool)].mean()

    result = dict(n=len(sample), cv_logreg_auc=cv_auc, silhouette=sil,
                  within_class_cos_sim=float(within), between_class_cos_sim=float(between),
                  gap=float(within - between))
    json.dump(result, open(OUT, "w"), indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
