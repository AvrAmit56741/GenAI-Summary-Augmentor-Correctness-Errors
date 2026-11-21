import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

data = np.load("hidden_states.npz")
res = np.load("analysis_results.npz")
Xf, Xe = data["X_faithful"], data["X_erroneous"]
records = json.load(open("records.json", encoding="utf-8"))
n, L, H = Xf.shape
error_types = [r["error_type"] for r in records]

BEST_LAYER = 24  # from analyze.py: peak within/between family-geometry gap

diff = Xe[:, BEST_LAYER, :] - Xf[:, BEST_LAYER, :]
pca = PCA(n_components=2)
coords = pca.fit_transform(diff)
explained = pca.explained_variance_ratio_

families = sorted(set(error_types), key=lambda t: -error_types.count(t))
cmap = plt.get_cmap("tab10")
color_of = {fam: cmap(i) for i, fam in enumerate(families)}

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# Panel 1: binary probe accuracy vs layer
ax = axes[0]
ax.plot(range(L), res["binary_acc"] * 100, marker="o", color="#444", lw=1.5, ms=3)
ax.axhline(50, color="gray", ls="--", lw=1, label="chance (50%)")
ax.set_xlabel("layer")
ax.set_ylabel("leave-one-pair-out accuracy (%)")
ax.set_title(f"(a) faithful vs. erroneous\nbinary probe, per layer (n={n} pairs)")
ax.legend(fontsize=8)
ax.set_ylim(30, 80)

# Panel 2: within- vs between-family cosine similarity gap vs layer
ax = axes[1]
ax.plot(range(L), res["within_by_layer"], marker="o", ms=3, lw=1.5, label="within-family (all 7 types)", color="#0072B2")
ax.plot(range(L), res["between_by_layer"], marker="o", ms=3, lw=1.5, label="between-family (all 7 types)", color="#D55E00")
ax.plot(range(L), res["within2_by_layer"], marker="s", ms=3, lw=1.2, ls="--", label="within (2 big types only)", color="#0072B2", alpha=0.55)
ax.plot(range(L), res["between2_by_layer"], marker="s", ms=3, lw=1.2, ls="--", label="between (2 big types only)", color="#D55E00", alpha=0.55)
ax.axhline(0, color="gray", lw=0.8)
ax.axvline(BEST_LAYER, color="black", lw=0.8, ls=":")
ax.set_xlabel("layer")
ax.set_ylabel("mean cosine similarity of\n(erroneous - faithful) direction vectors")
p_all = float(res["perm_pvalue_all"]) if "perm_pvalue_all" in res else None
p_big = float(res["perm_pvalue_big2"]) if "perm_pvalue_big2" in res else None
p_note = f"\npermutation p={p_all:.4f} (all 7), p={p_big:.4f} (2 big)" if p_all is not None else ""
ax.set_title(f"(b) error-family geometry, per layer\n(dotted line = layer used in panel c){p_note}")
ax.legend(fontsize=7, loc="upper left")

# Panel 3: PCA of difference vectors at the best layer
ax = axes[2]
for fam in families:
    idx = [i for i, t in enumerate(error_types) if t == fam]
    ax.scatter(coords[idx, 0], coords[idx, 1], label=f"{fam} (n={len(idx)})",
               color=color_of[fam], s=70, edgecolor="black", linewidth=0.4)
ax.set_xlabel(f"PC1 ({explained[0]*100:.0f}% var)")
ax.set_ylabel(f"PC2 ({explained[1]*100:.0f}% var)")
ax.set_title(f"(c) PCA of (erroneous-faithful) direction\nvectors at layer {BEST_LAYER} of 28, colored by taxonomy family")
ax.legend(fontsize=6.5, loc="best")

fig.suptitle(f"Pilot: Hebrew morphological error-type geometry in DictaLM-3.0-1.7B-Instruct hidden states (N={n} confirmed project findings)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("repr_pilot_figure.png", dpi=200)
print("saved repr_pilot_figure.png")
