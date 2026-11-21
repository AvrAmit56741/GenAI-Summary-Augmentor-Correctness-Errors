"""Test 1 (word level): where does MorphEvalHeb encode each MORPHOLOGICAL edit type, and does it use it?

Unit = one changed word pair (faithful word -> corrupted word) inside a sentence pair from multisource_units.csv,
typed by Dicta's analyzer (exp_morph_labels.word_kinds): gender, tense, prefix, definiteness, construct, root_pattern,
letter_swap, entity, number_digit, lexical; plus negation (word right after an inserted/removed negation).
Each sentence is encoded with its full article (as MorphEvalHeb is used). Vector of a word = mean of its sub-tokens.

    python exp_layers_words.py sample        -> layers_words_sample.csv  (<=120 words per kind, <=3 per sentence)
    python exp_layers_words.py embed ft|base -> cache/layers_words_{ft,base}.npz   (chunks of 20 sentences)
    python exp_layers_words.py probe         -> layers_words_probe.csv, layers_words_recall.csv, figures/layers_words_*.png
"""
import difflib
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_morph_labels as M  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
UNITS = os.path.join(HERE, "multisource_units.csv")
SAMPLE = os.path.join(HERE, "layers_words_sample.csv")
EMB = os.path.join(HERE, "cache", "layers_words_{}.npz")
FIG = os.path.join(HERE, "figures")
KINDS = ["negation", "gender", "tense", "prefix", "definiteness", "construct", "root_pattern", "letter_swap",
         "entity", "number_digit", "lexical"]
MORPH = ["gender", "tense", "prefix", "definiteness", "construct", "root_pattern"]


def word_pairs(tg, tc):
    """(gold word idx, corr word idx, kind) for single-kind changed words; negation -> next aligned word."""
    wg, wc = [t["w"] for t in tg], [t["w"] for t in tc]
    out = []
    ops = difflib.SequenceMatcher(None, wg, wc, autojunk=False).get_opcodes()
    for n, (op, i1, i2, j1, j2) in enumerate(ops):
        if op == "equal":
            continue
        A, B = tg[i1:i2], tc[j1:j2]
        neg_a = [t for t in A if t["lex"] in M.NEG or t["w"] in M.NEG]
        neg_b = [t for t in B if t["lex"] in M.NEG or t["w"] in M.NEG]
        if bool(neg_a) != bool(neg_b) and n + 1 < len(ops) and ops[n + 1][0] == "equal":
            out.append((ops[n + 1][1], ops[n + 1][3], "negation"))      # first unchanged word after the negation edit
        if op == "replace" and len(A) == len(B):
            for k, (a, b) in enumerate(zip(A, B)):
                kd = M.word_kinds(a, b)
                if len(kd) == 1:
                    out.append((i1 + k, j1 + k, kd.pop()))
    return out


def sample(per_kind=120, per_sent=3, seed=7):
    u = pd.read_csv(UNITS)
    u["unit_id"] = np.arange(len(u))
    c = M.parse_all(list(u.gold_sent) + list(u.corr_sent))
    rows = []
    for r in u.itertuples():
        for gi, ci, k in word_pairs(c[r.gold_sent], c[r.corr_sent]):
            rows.append({"unit_id": r.unit_id, "doc_id": r.doc_id, "gi": gi, "ci": ci, "kind": k,
                         "gold_word": c[r.gold_sent][gi]["w"], "corr_word": c[r.corr_sent][ci]["w"]})
    w = pd.DataFrame(rows)
    print("all words:", w.kind.value_counts().to_dict())
    w = w.sample(frac=1, random_state=seed)
    w = w.groupby(["unit_id", "kind"]).head(1).groupby("unit_id").head(per_sent)
    # rarest kinds first, cap per kind
    w = w.groupby("kind").head(per_kind)
    w = w.merge(u[["unit_id", "gold_sent", "corr_sent"]], on="unit_id")
    w.to_csv(SAMPLE, index=False)
    print(len(w), "words in", w.unit_id.nunique(), "sentences:", w.kind.value_counts().to_dict())


def char_spans(sent, words):
    spans, cur = [], 0
    for wd in words:
        i = sent.find(wd, cur)
        if i < 0:
            spans.append(None)
            continue
        spans.append((i, i + len(wd)))
        cur = i + len(wd)
    return spans


def embed(which="ft", chunk=20):
    import torch
    import run_eval as R
    from exp_layers import load_model, premise_of, MAX_LEN
    w = pd.read_csv(SAMPLE)
    c = M.parse_all([])
    out = EMB.format(which)
    acc = {k: list(v) for k, v in np.load(out, allow_pickle=True).items()} if os.path.exists(out) else {}
    for k in ["row", "g", "cg", "c", "cc"]:
        acc.setdefault(k, [])
    done = set(acc["row"])
    w["row"] = np.arange(len(w))
    todo = w[~w.row.isin(done)]
    tok, model, enc = load_model(which)
    dev = R.device()
    units = todo.drop_duplicates("unit_id")
    arts = dict(zip(units.doc_id, premise_of(units.doc_id.values)))
    print(f"{which}: {len(todo)} words in {len(units)} sentences to embed")
    for n, (uid, grp) in enumerate(todo.groupby("unit_id", sort=False)):
        r0 = grp.iloc[0]
        vec = {}
        for tag, sent in [("g", r0.gold_sent), ("c", r0.corr_sent)]:
            e = tok(arts[r0.doc_id], sent, return_tensors="pt", truncation="only_first", max_length=MAX_LEN,
                    return_offsets_mapping=True)
            offs = e.pop("offset_mapping")[0].numpy()
            seq = np.array([s if s is not None else -1 for s in e.sequence_ids(0)])
            x = {k: v.to(dev) for k, v in e.items() if k != "token_type_ids"}
            with torch.no_grad():
                hs = (model.model if model is not None else enc)(**x, output_hidden_states=True).hidden_states
            H = torch.stack([h[0] for h in hs]).float().cpu().numpy()      # L x T x D
            spans = char_spans(sent, [t["w"] for t in c[sent]])
            vec[tag] = (H, offs, seq, spans, H[:, 0])
        for r in grp.itertuples():
            vs = []
            for tag, idx in [("g", r.gi), ("c", r.ci)]:
                H, offs, seq, spans, cls = vec[tag]
                sp = spans[idx]
                m = (seq == 1) & (offs[:, 0] < sp[1]) & (offs[:, 1] > sp[0]) if sp else np.zeros(len(seq), bool)
                vs.append((H[:, m].mean(1) if m.any() else np.full(H.shape[::2], np.nan)).astype(np.float16))
                vs.append(cls.astype(np.float16))
            acc["row"].append(r.row)
            acc["g"].append(vs[0]); acc["cg"].append(vs[1]); acc["c"].append(vs[2]); acc["cc"].append(vs[3])
        if (n + 1) % chunk == 0 or n + 1 == len(units):
            np.savez(out, **{k: np.array(v) for k, v in acc.items()})
            bad = np.isnan(np.array(acc["g"])[:, 0, 0]).sum() + np.isnan(np.array(acc["c"])[:, 0, 0]).sum()
            print(f"  {n + 1}/{len(units)} sentences, {len(acc['row'])} words, unmapped vectors {bad}", flush=True)


def data(which):
    z = dict(np.load(EMB.format(which), allow_pickle=True))
    w = pd.read_csv(SAMPLE)
    w["row"] = np.arange(len(w))
    w = w.set_index("row").loc[z["row"]].reset_index()
    ok = ~(np.isnan(z["g"][:, 0, 0].astype(float)) | np.isnan(z["c"][:, 0, 0].astype(float)))
    return w[ok].reset_index(drop=True), {k: v[ok] for k, v in z.items()}


def probe():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, recall_score, silhouette_score
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rows, rec = [], []
    for which in ["ft", "base"]:
        if not os.path.exists(EMB.format(which)):
            continue
        w, z = data(which)
        keep = w.kind.map(w.kind.value_counts()) >= 20
        w, z = w[keep].reset_index(drop=True), {k: v[keep.values] for k, v in z.items()}
        kinds = sorted(w.kind.unique())
        for L in range(z["g"].shape[1]):
            D = (z["c"][:, L] - z["g"][:, L]).astype(np.float32)
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced"))
            pred = cross_val_predict(clf, D, w.kind.values, groups=w.doc_id.values, cv=GroupKFold(5))
            Dn = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-6)
            rows.append({"model": which, "layer": L + 1, "bal_acc": balanced_accuracy_score(w.kind, pred),
                         "chance": 1 / len(kinds), "silhouette_cos": silhouette_score(Dn, w.kind, metric="cosine"),
                         "n": len(w)})
            r = recall_score(w.kind, pred, labels=kinds, average=None)
            for k, v in zip(kinds, r):
                rec.append({"model": which, "layer": L + 1, "kind": k, "recall": v, "n": int((w.kind == k).sum())})
        print(which, "done", flush=True)
    P, Rc = pd.DataFrame(rows), pd.DataFrame(rec)
    P.to_csv(os.path.join(HERE, "layers_words_probe.csv"), index=False)
    Rc.to_csv(os.path.join(HERE, "layers_words_recall.csv"), index=False)
    print(P.pivot(index="layer", columns="model", values=["bal_acc", "silhouette_cos"]).iloc[[0, 3, 7, 11, 15, 19, 23, 27]].round(2))
    print(Rc.pivot_table(index="kind", columns=["model", "layer"], values="recall")[
        [(m, l) for m in ["base", "ft"] for l in (1, 8, 16, 28)]].round(2))
    figures()


def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    P = pd.read_csv(os.path.join(HERE, "layers_words_probe.csv"))
    Rc = pd.read_csv(os.path.join(HERE, "layers_words_recall.csv"))
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.3), gridspec_kw={"width_ratios": [1, 1, 1.4]})
    for m, g in P.groupby("model"):
        lab = "fine-tuned MorphEvalHeb" if m == "ft" else "base NeoDictaBERT"
        ax[0].plot(g.layer, g.bal_acc, label=lab, ls="-" if m == "ft" else "--")
        ax[1].plot(g.layer, g.silhouette_cos, label=lab, ls="-" if m == "ft" else "--")
    ax[0].axhline(P.chance.iloc[0], color="gray", ls=":", lw=0.8, label="chance")
    ax[0].set(title="Which edit type? (probe on corrupted − faithful word)", xlabel="layer", ylabel="balanced accuracy")
    ax[1].set(title="Cluster separation of edit types", xlabel="layer", ylabel="silhouette (cosine)")
    ax[0].legend(fontsize=8); ax[1].legend(fontsize=8)
    h = Rc[Rc.model == "ft"].pivot(index="kind", columns="layer", values="recall")
    order = [k for k in KINDS if k in h.index]
    h = h.loc[order]
    im = ax[2].imshow(h.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    nk = Rc[Rc.model == "ft"].groupby("kind").n.first()
    ax[2].set_yticks(range(len(h)), [f"{k} (n={nk[k]})" for k in h.index], fontsize=8)
    ax[2].set_xticks(range(0, h.shape[1], 3), h.columns[::3])
    ax[2].set(title="Fine-tuned: per-type recall of the type probe", xlabel="layer")
    plt.colorbar(im, ax=ax[2])
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "layers_words_probe.png"), dpi=160); plt.close()
    for which in ["ft", "base"]:
        if not os.path.exists(EMB.format(which)):
            continue
        w, z = data(which)
        m = w.kind.isin(["negation", "gender", "tense", "prefix", "definiteness", "root_pattern", "letter_swap",
                         "entity", "number_digit"]).values
        fig, ax = plt.subplots(1, 4, figsize=(18, 4.4))
        for a, L in zip(ax, [1, 8, 16, 28]):
            D = (z["c"][m, L - 1] - z["g"][m, L - 1]).astype(np.float32)
            E2 = TSNE(2, random_state=0, perplexity=30, metric="cosine").fit_transform(D)
            for k in ["negation", "gender", "tense", "prefix", "definiteness", "root_pattern", "letter_swap",
                      "entity", "number_digit"]:
                s = w.kind.values[m] == k
                a.scatter(E2[s, 0], E2[s, 1], s=7, label=k)
            a.set(title=f"layer {L}", xticks=[], yticks=[])
        ax[0].legend(fontsize=7, markerscale=2)
        fig.suptitle(f"{'Fine-tuned MorphEvalHeb' if which == 'ft' else 'Base NeoDictaBERT'}: corrupted − faithful WORD vector, t-SNE")
        plt.tight_layout(); plt.savefig(os.path.join(FIG, f"layers_words_tsne_{which}.png"), dpi=150); plt.close()
    print("figures saved")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "sample":
        sample()
    elif cmd == "embed":
        embed(sys.argv[2] if len(sys.argv) > 2 else "ft")
    elif cmd == "probe":
        probe()
    elif cmd == "fig":
        figures()
