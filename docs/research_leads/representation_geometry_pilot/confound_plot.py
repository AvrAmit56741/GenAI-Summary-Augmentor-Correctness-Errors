import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

# Companion figure to confound_check.py: explains WHY the family-geometry result isn't the
# trivial "different sentences look different" claim, and shows the actual numbers that rule
# out a topic/lexical confound.

data = np.load("hidden_states.npz")
Xf, Xe = data["X_faithful"], data["X_erroneous"]
records = json.load(open("records.json", encoding="utf-8"))
ids = [r["id"] for r in records]
types = [r["error_type"] for r in records]
n, L, H = Xf.shape
BEST_LAYER = 24

DOG_IDS = {"dicta2.0_uzi", "qwen14b_juan_finder", "qwen14b_chupi_finder", "qwen14b_hason_finder"}
is_dog = np.array([i in DOG_IDS for i in ids])
type_arr = np.array(types)
diff = Xe[:, BEST_LAYER, :] - Xf[:, BEST_LAYER, :]
v = diff / (np.linalg.norm(diff, axis=1, keepdims=True) + 1e-8)
sim = v @ v.T

vgs_idx = np.where(type_arr == "verb_gender_swap")[0]
dd, dn, nn = [], [], []
for a in range(len(vgs_idx)):
    for b in range(a + 1, len(vgs_idx)):
        i, j = vgs_idx[a], vgs_idx[b]
        s = sim[i, j]
        if is_dog[i] and is_dog[j]:
            dd.append(s)
        elif is_dog[i] != is_dog[j]:
            dn.append(s)
        else:
            nn.append(s)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.15, 1]})

# ---------------- Panel (a): conceptual schematic ----------------
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")
ax.set_title("(a) What is actually being compared", fontsize=12, pad=14)

box_style = dict(boxstyle="round,pad=0.35", linewidth=1.2)


def add_box(ax, x, y, w, h, text, fc, ec, fontsize=9.5):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12", linewidth=1.2,
                        facecolor=fc, edgecolor=ec)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True)


# --- NOT the test ---
ax.text(5, 9.6, "NOT the test (would be trivial)", ha="center", fontsize=10.5, color="#B00020", fontweight="bold")
add_box(ax, 0.3, 8.35, 3.8, 1.0, "sentence about\ndogs / shelters", "#FCE8E6", "#B00020")
add_box(ax, 5.9, 8.35, 3.8, 1.0, "sentence about\na Knesset debate", "#FCE8E6", "#B00020")
ax.annotate("", xy=(5.7, 8.85), xytext=(4.2, 8.85),
            arrowprops=dict(arrowstyle="<->", color="#B00020", lw=1.6))
ax.text(5, 7.85, "raw text / topic similarity -- always low, uninformative",
        ha="center", fontsize=8.6, color="#B00020", style="italic")

ax.plot([0.5, 9.5], [7.35, 7.35], color="#cccccc", lw=1, ls="--")

# --- The actual test ---
ax.text(5, 6.9, "The actual test (this pilot)", ha="center", fontsize=10.5, color="#0B6E4F", fontweight="bold")
add_box(ax, 0.3, 4.55, 3.8, 1.75,
        "Topic A, verb_gender_swap:\nfaithful $\\rightarrow$ erroneous\n(dog-shelter listing)",
        "#E6F4EA", "#0B6E4F", fontsize=9)
add_box(ax, 5.9, 4.55, 3.8, 1.75,
        "Topic B, verb_gender_swap:\nfaithful $\\rightarrow$ erroneous\n(general's wartime account)",
        "#E6F4EA", "#0B6E4F", fontsize=9)
ax.annotate("", xy=(2.2, 4.45), xytext=(2.2, 3.55), arrowprops=dict(arrowstyle="-|>", color="#0B6E4F", lw=1.6))
ax.annotate("", xy=(7.8, 4.45), xytext=(7.8, 3.55), arrowprops=dict(arrowstyle="-|>", color="#0B6E4F", lw=1.6))
ax.text(2.2, 3.25, "$d_A$ = erroneous $-$ faithful", ha="center", fontsize=8.6)
ax.text(7.8, 3.25, "$d_B$ = erroneous $-$ faithful", ha="center", fontsize=8.6)
ax.text(5, 2.15, "compare $\\cos(d_A, d_B)$ -- the shift direction, not the topic",
        ha="center", fontsize=9.3, color="#0B6E4F", fontweight="bold")
ax.text(5, 1.5, "(measured: $+0.212$ for unrelated topics sharing the same error family)",
        ha="center", fontsize=8.8, color="#0B6E4F")

# ---------------- Panel (b): empirical confound check ----------------
ax = axes[1]
groups = [dd, dn, nn]
labels = ["dog-dog pairs\n(same topic)\nn={}".format(len(dd)),
          "dog-nondog pairs\n(mixed topic)\nn={}".format(len(dn)),
          "nondog-nondog pairs\n(unrelated topics)\nn={}".format(len(nn))]
means = [np.mean(g) for g in groups]
sems = [np.std(g, ddof=1) / np.sqrt(len(g)) if len(g) > 1 else 0 for g in groups]
colors = ["#D55E00", "#E69F00", "#0072B2"]

x = np.arange(3)
bars = ax.bar(x, means, yerr=sems, capsize=5, color=colors, edgecolor="black", linewidth=0.8, width=0.6)
for xi, g in zip(x, groups):
    jitter = (np.random.default_rng(0).random(len(g)) - 0.5) * 0.18
    ax.scatter(np.full(len(g), xi) + jitter, g, color="black", s=18, alpha=0.6, zorder=3)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8.8)
ax.axhline(0, color="gray", lw=0.8)
ax.set_ylabel("cosine similarity of (erroneous $-$ faithful)\ndirection vectors, layer 24")
ax.set_title("(b) Within verb_gender_swap: topic-matched\nvs. topic-unrelated pairs", fontsize=12)
ax.text(0.5, 0.97, "If topic drove the effect, the left (same-topic)\nbar would be tallest. It is not.",
        transform=ax.transAxes, ha="center", va="top", fontsize=8.8, style="italic",
        bbox=dict(boxstyle="round", fc="white", ec="#999999", alpha=0.9))

fig.suptitle("Robustness check: the family-geometry gap is not a topic/lexical artifact", fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("fig_confound_robustness.png", dpi=200)
print("saved fig_confound_robustness.png")
print(f"dog-dog mean={np.mean(dd):+.3f}  dog-nondog mean={np.mean(dn):+.3f}  nondog-nondog mean={np.mean(nn):+.3f}")
