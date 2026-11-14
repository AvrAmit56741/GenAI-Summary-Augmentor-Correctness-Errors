"""2D visualization of the contrastive-embedding probe (experiment 11, docs/experiments.md).

Thesis-only visual companion to exp_contrastive.py's quantitative result (0.983 frozen-probe AUC,
0.87/0.72 within/between cosine similarity). Not intended for the ACL paper (see docs/experiments.md
and the author's decision not to add a figure there) -- outputs go to thesis_figures/, separate from
paper/figures/ (which the ACL LaTeX source references).

Reuses the identical sample (same groupby/sample call, random_state=0) as exp_contrastive.py so the
plotted points match the reported AUC/cosine-similarity numbers exactly.

    python exp_contrastive_viz.py   -> thesis_figures/contrastive_embedding_tsne.png
                                       thesis_figures/contrastive_embedding_tsne.npz (raw embeddings + labels)
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eval as R  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "thesis_figures")
os.makedirs(OUT_DIR, exist_ok=True)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1200


def main():
    d = R.load_bench()
    per_group = max(1, N // 4)
    n_per_group = min(int(d.groupby(["part", "y"]).size().min()), per_group)
    sample = d.groupby(["part", "y"]).sample(n=n_per_group, random_state=0).reset_index(drop=True)
    print(f"sample: {len(sample)} pairs (same as exp_contrastive.py)")

    tok, model = R.load_cls("NLI-hebrew-binary-correctness-metric")
    dev = R.device()
    embs = []
    with torch.no_grad():
        for i, r in enumerate(sample.itertuples()):
            enc = tok(r.premise, r.hypothesis, return_tensors="pt", truncation="only_first",
                      max_length=3072).to(dev)
            out = model(**enc, output_hidden_states=True)
            pooled = out.hidden_states[-1][0].mean(0).float().cpu().numpy()
            embs.append(pooled)
            if (i + 1) % 200 == 0:
                print(f"  embedded {i+1}/{len(sample)}", flush=True)
    X = np.stack(embs)
    y = sample.y.values
    part = sample.part.values

    np.savez(os.path.join(OUT_DIR, "contrastive_embedding_tsne.npz"), X=X, y=y, part=part)

    tsne = TSNE(n_components=2, random_state=0, perplexity=30, init="pca")
    X2 = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(7, 6))
    markers = {"MAFAT-Aug": "o", "mFACT-He": "^"}
    colors = {1: "#2a7a3b", 0: "#c0392b"}
    labels = {1: "faithful", 0: "unfaithful"}
    for p in ["MAFAT-Aug", "mFACT-He"]:
        for yy in [1, 0]:
            m = (part == p) & (y == yy)
            ax.scatter(X2[m, 0], X2[m, 1], c=colors[yy], marker=markers[p], s=18, alpha=0.6,
                      edgecolors="none", label=f"{labels[yy]} ({p})")
    ax.set_title("MorphEvalHeb verifier embeddings (t-SNE, n=%d)\nfrozen final-layer mean pool, cross-entropy-trained" % len(sample))
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="best", fontsize=8, markerscale=1.5)
    fig.tight_layout()
    png = os.path.join(OUT_DIR, "contrastive_embedding_tsne.png")
    fig.savefig(png, dpi=200)
    print("saved", png)


if __name__ == "__main__":
    main()
