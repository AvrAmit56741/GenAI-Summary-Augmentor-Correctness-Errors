import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = np.load("hidden_states.npz")
Xf = data["X_faithful"]  # (n, L, H)
Xe = data["X_erroneous"]  # (n, L, H)
records = json.load(open("records.json", encoding="utf-8"))
n, L, H = Xf.shape
error_types = [r["error_type"] for r in records]
ids = [r["id"] for r in records]

print(f"n_examples={n}  n_layers(incl. embed)={L}  hidden={H}")

# ---------- 1. Binary probe: faithful (0) vs erroneous (1), leave-one-pair-out CV ----------
print("\n=== Binary probe: faithful vs erroneous hidden state, per layer ===")
print("(leave-one-PAIR-out CV: both members of a pair always held out together)")
binary_acc = []
for layer in range(L):
    X = np.concatenate([Xf[:, layer, :], Xe[:, layer, :]], axis=0)
    y = np.concatenate([np.zeros(n), np.ones(n)])
    groups = np.concatenate([np.arange(n), np.arange(n)])
    correct = 0
    total = 0
    for held_out in range(n):
        train_mask = groups != held_out
        test_mask = groups == held_out
        Xtr, ytr = X[train_mask], y[train_mask]
        Xte, yte = X[test_mask], y[test_mask]
        scaler = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(scaler.transform(Xtr), ytr)
        pred = clf.predict(scaler.transform(Xte))
        correct += (pred == yte).sum()
        total += len(yte)
    acc = correct / total
    binary_acc.append(acc)

for layer in range(L):
    bar = "#" * int(binary_acc[layer] * 40)
    print(f"  layer {layer:2d}: {binary_acc[layer]*100:5.1f}%  {bar}")
best_binary_layer = int(np.argmax(binary_acc))
print(f"Best layer: {best_binary_layer}  acc={binary_acc[best_binary_layer]*100:.1f}%  (chance=50%, n={2*n} points, {n} leave-one-pair-out folds)")

# ---------- 2. Family geometry: within-family vs between-family cosine similarity of diff vectors ----------
print("\n=== Difference-vector geometry: within-family vs between-family cosine similarity ===")
diff = Xe - Xf  # (n, L, H)  erroneous-minus-faithful direction per example, per layer

type_arr = np.array(error_types)
unique_types, counts = np.unique(type_arr, return_counts=True)
print("Family sizes:", dict(zip(unique_types.tolist(), counts.tolist())))

within_by_layer = []
between_by_layer = []
for layer in range(L):
    v = diff[:, layer, :]
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
    sim = v @ v.T  # (n, n) cosine similarity matrix
    within_vals = []
    between_vals = []
    for i in range(n):
        for j in range(i + 1, n):
            s = sim[i, j]
            if type_arr[i] == type_arr[j]:
                within_vals.append(s)
            else:
                between_vals.append(s)
    within_by_layer.append(np.mean(within_vals))
    between_by_layer.append(np.mean(between_vals))

within_by_layer = np.array(within_by_layer)
between_by_layer = np.array(between_by_layer)
gap = within_by_layer - between_by_layer

for layer in range(L):
    print(f"  layer {layer:2d}: within={within_by_layer[layer]:+.3f}  between={between_by_layer[layer]:+.3f}  gap={gap[layer]:+.3f}")
best_gap_layer = int(np.argmax(gap))
print(f"Best (largest within-between gap) layer: {best_gap_layer}  gap={gap[best_gap_layer]:+.3f}")

# Also restrict to the two larger families only (verb_gender_swap, noun_gender_swap) --
# these are the two families with the most members, for a within-family mean less sensitive to
# small-n noise than the thinner families.
mask_big = np.isin(type_arr, ["verb_gender_swap", "noun_gender_swap"])
print("\n--- restricted to the 2 biggest families (verb_gender_swap, noun_gender_swap) ---")
within2_by_layer = []
between2_by_layer = []
for layer in range(L):
    v = diff[mask_big][:, layer, :]
    t2 = type_arr[mask_big]
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
    sim = v @ v.T
    m = mask_big.sum()
    wv, bv = [], []
    for i in range(m):
        for j in range(i + 1, m):
            s = sim[i, j]
            (wv if t2[i] == t2[j] else bv).append(s)
    within2_by_layer.append(np.mean(wv))
    between2_by_layer.append(np.mean(bv))
within2_by_layer = np.array(within2_by_layer)
between2_by_layer = np.array(between2_by_layer)
gap2 = within2_by_layer - between2_by_layer
for layer in range(L):
    print(f"  layer {layer:2d}: within={within2_by_layer[layer]:+.3f}  between={between2_by_layer[layer]:+.3f}  gap={gap2[layer]:+.3f}")
best_gap2_layer = int(np.argmax(gap2))
print(f"Best layer (2-family subset): {best_gap2_layer}  gap={gap2[best_gap2_layer]:+.3f}")

# ---------- 3. Permutation significance test on the within/between gap at the best layer ----------
print("\n=== Permutation test: is the within/between gap at the best layer bigger than chance? ===")
rng = np.random.default_rng(0)
N_PERM = 10000


def gap_for_labels(layer, labels, pool_mask=None):
    v = diff[pool_mask][:, layer, :] if pool_mask is not None else diff[:, layer, :]
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
    sim = v @ v.T
    m = len(labels)
    wv, bv = [], []
    for i in range(m):
        for j in range(i + 1, m):
            (wv if labels[i] == labels[j] else bv).append(sim[i, j])
    return np.mean(wv) - np.mean(bv)


def permutation_pvalue(layer, pool_mask=None):
    labels = type_arr if pool_mask is None else type_arr[pool_mask]
    observed = gap_for_labels(layer, labels, pool_mask)
    null_gaps = np.empty(N_PERM)
    for p in range(N_PERM):
        shuffled = rng.permutation(labels)
        null_gaps[p] = gap_for_labels(layer, shuffled, pool_mask)
    pval = (np.sum(null_gaps >= observed) + 1) / (N_PERM + 1)
    return observed, pval


obs_all, p_all = permutation_pvalue(best_gap_layer)
print(f"All {len(unique_types)} families, layer {best_gap_layer}: observed gap={obs_all:+.3f}, "
      f"permutation p-value={p_all:.4f} (N={N_PERM} shuffles, label-shuffle null)")

obs_big, p_big = permutation_pvalue(best_gap2_layer, pool_mask=mask_big)
print(f"2 biggest families only, layer {best_gap2_layer}: observed gap={obs_big:+.3f}, "
      f"permutation p-value={p_big:.4f} (N={N_PERM} shuffles, label-shuffle null)")

np.savez(
    "analysis_results.npz",
    binary_acc=np.array(binary_acc),
    within_by_layer=within_by_layer,
    between_by_layer=between_by_layer,
    within2_by_layer=within2_by_layer,
    between2_by_layer=between2_by_layer,
    perm_pvalue_all=p_all,
    perm_pvalue_big2=p_big,
)
print("\nSaved analysis_results.npz")
print(f"\nSUMMARY: best_binary_layer={best_binary_layer} ({binary_acc[best_binary_layer]*100:.1f}%), "
      f"best_family_gap_layer(all {len(unique_types)} fam)={best_gap_layer} (gap={gap[best_gap_layer]:+.3f}, p={p_all:.4f}), "
      f"best_family_gap_layer(2 big fam)={best_gap2_layer} (gap={gap2[best_gap2_layer]:+.3f}, p={p_big:.4f})")
