"""Where are Hebrew error types represented inside MorphEvalHeb? (representation-learning / activation-engineering lead)

Sentence-level (gold, corrupted) pairs from multisource_units.csv (Injected part). Each corrupted sentence gets ONE
error type: an LLM (Claude Haiku 4.5, Bedrock) picks it from the summary's own listed types; the rule-based
edit_kind() from exp_translation.py is an independent check for negation / number edits.

Stages (all cached, resumable, chunks of 20):
    python exp_layers.py sample            -> layers_sample.csv            (stratified by type, <=40 per type)
    python exp_layers.py label [N]         -> cache/layers_labels.jsonl    (N units, chunks of 20, prints checks)
    python exp_layers.py embed [ft|base]   -> cache/layers_emb_{ft,base}.npz (per-layer CLS + hypothesis-mean)
    python exp_layers.py probe             -> layers_probe*.csv, figures/layers_*.png
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
UNITS = os.path.join(HERE, "multisource_units.csv")
SAMPLE = os.path.join(HERE, "layers_sample.csv")
LABELS = os.path.join(HERE, "cache", "layers_labels.jsonl")
EMB = os.path.join(HERE, "cache", "layers_emb_{}.npz")
FIG = os.path.join(HERE, "figures")
LLM = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
MAX_LEN = 3072

GLOSS = {
    "construct_state_confusion": "smichut / construct-state structure changed (e.g. 'בית ספר' vs 'בית הספר', which noun is head)",
    "hebrew_root_pattern_confusion": "word with the same 3-letter root but different pattern/binyan, changing meaning",
    "pronoun_reference_errors": "pronoun or pronominal suffix now refers to a different entity",
    "number_swap": "a quantity/number changed",
    "sentence_negation": "negation added or removed",
    "entity_person_swap": "a person name replaced",
    "specificity_shift_errors": "claim made more/less specific or general than the article",
    "verb_tense_swap": "verb tense changed (past/present/future)",
    "entity_organization_swap": "an organization name replaced",
    "entity_location_swap": "a location replaced",
    "idiom_collocation_corruption": "an idiom or fixed collocation altered",
    "measure_unit_swap": "unit of measure changed (e.g. km->m, %->points)",
    "entity_date_swap": "a date/year/time changed",
    "verb_gender_swap": "verb gender agreement changed",
    "entity_event_swap": "an event replaced",
    "morphological_connective_confusion": "prefix connective (ו/ש/כש/מ/ל/ב) changed, altering logical relation",
    "noun_gender_swap": "noun/adjective gender changed",
    "entity_title_swap": "a title/role (e.g. minister, CEO) replaced",
    "directional_preposition_swap": "directional preposition changed (from/to, into/out of)",
    "homographic_gender_errors": "an unvocalized homograph read with the wrong gender",
}


def types_of(s):
    return [t.strip() for t in str(s).split(",") if t.strip()]


# ------------------------------------------------------------------ sample
def sample(per_type=40, seed=7):
    u = pd.read_csv(UNITS)
    u["unit_id"] = np.arange(len(u))
    rng = np.random.default_rng(seed)
    picked, out = set(), []
    # rarest types first so they get first pick of shared units
    counts = pd.Series([t for s in u.error_types for t in types_of(s)]).value_counts()
    for t in counts.index[::-1]:
        cand = u[u.error_types.apply(lambda s: t in types_of(s)) & ~u.unit_id.isin(picked)]
        cand = cand.sample(frac=1, random_state=int(rng.integers(1e9)))
        cand = cand.groupby("doc_id").head(2).head(per_type)
        picked |= set(cand.unit_id)
        out.append(cand.assign(sampled_for=t))
    s = pd.concat(out).sample(frac=1, random_state=seed).reset_index(drop=True)
    s.to_csv(SAMPLE, index=False)
    print(len(s), "units;", s.sampled_for.value_counts().to_dict())


# ------------------------------------------------------------------ label (LLM)
PROMPT = """You compare two Hebrew sentences. The CORRUPTED sentence was made from the ORIGINAL by an automatic
error-injection system that was asked to inject errors of these types (one summary, several types; this sentence
usually shows only one of them):
{types}

ORIGINAL: {g}
CORRUPTED: {c}

Which ONE of the listed types best describes the change between the two sentences? If the change matches none of
them, answer "none". Reply with JSON only: {{"type": "<type name or none>", "changed": "<the changed words, original -> corrupted>"}}"""


def label(n=None, chunk=20):
    from exp_translation import edit_kind
    from exp_length_corrected_val20_bedrock import load_creds
    load_creds()
    try:
        import truststore
        truststore.inject_into_ssl()
    except Exception:
        pass
    import boto3
    brt = boto3.Session(region_name="us-east-1").client("bedrock-runtime")
    s = pd.read_csv(SAMPLE)
    done = set()
    if os.path.exists(LABELS):
        done = {json.loads(line)["unit_id"] for line in open(LABELS)}
    todo = s[~s.unit_id.isin(done)]
    if n:
        todo = todo.head(n)
    print(f"done {len(done)}, labeling {len(todo)}")
    for i in range(0, len(todo), chunk):
        rows, bad = [], 0
        for r in todo.iloc[i:i + chunk].itertuples():
            ts = types_of(r.error_types)
            p = PROMPT.format(types="\n".join(f"- {t}: {GLOSS.get(t, '')}" for t in ts), g=r.gold_sent, c=r.corr_sent)
            resp = brt.converse(modelId=LLM, messages=[{"role": "user", "content": [{"text": p}]}],
                                inferenceConfig={"maxTokens": 200, "temperature": 0})
            txt = resp["output"]["message"]["content"][0]["text"]
            m = re.search(r"\{.*\}", txt, re.S)
            try:
                j = json.loads(m.group(0))
                t = j.get("type", "none")
            except Exception:
                j, t = {}, "parse_error"
            if t not in ts and t != "none":
                bad += 1
                t = "invalid:" + str(t)
            rows.append({"unit_id": int(r.unit_id), "type": t, "changed": j.get("changed", ""),
                         "rule": edit_kind(r.gold_sent, r.corr_sent), "raw": txt})
        with open(LABELS, "a") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        df = pd.DataFrame(rows)
        neg = df[df.type == "sentence_negation"]
        num = df[df.type.isin(["number_swap", "entity_date_swap", "measure_unit_swap"])]
        print(f"chunk {i // chunk + 1}: n={len(df)} none={int((df.type == 'none').sum())} invalid={bad} "
              f"| negation->rule negation {neg.rule.str.contains('negation').mean() if len(neg) else float('nan'):.2f} (n={len(neg)}) "
              f"| number-like->rule number {num.rule.str.contains('number').mean() if len(num) else float('nan'):.2f} (n={len(num)})",
              flush=True)


def labeled():
    """Sample with rule label (always) and LLM type (if labeled; NaN otherwise)."""
    from exp_translation import edit_kind
    s = pd.read_csv(SAMPLE)
    s["rule"] = [edit_kind(g, c) for g, c in zip(s.gold_sent, s.corr_sent)]
    s["type"] = np.nan
    if os.path.exists(LABELS):
        lab = pd.DataFrame([json.loads(x) for x in open(LABELS)]).drop_duplicates("unit_id", keep="last")
        lab = lab[~lab.type.str.startswith("invalid") & (lab.type != "parse_error")]
        s = s.drop(columns="type").merge(lab[["unit_id", "type"]], on="unit_id", how="left")
    return s


# ------------------------------------------------------------------ embed
def premise_of(doc_ids):
    import run_eval as R
    b = R.load_bench()
    m = b[b.part == "MAFAT-Aug"].drop_duplicates("doc_id").set_index("doc_id").premise
    return m.reindex(doc_ids).values


def load_model(which):
    import torch
    import run_eval as R
    if which == "ft":
        tok, model = R.load_cls("NLI-hebrew-binary-correctness-metric")
        enc = model.model
    else:
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained("dicta-il/neodictabert", trust_remote_code=True)
        enc = AutoModel.from_pretrained("dicta-il/neodictabert", trust_remote_code=True,
                                        torch_dtype=torch.float32).to(R.device()).eval()
        model = None
    return tok, model, enc


def encode(tok, art, hyp, dev):
    enc = tok(art, hyp, return_tensors="pt", truncation="only_first", max_length=MAX_LEN)
    hyp_ids = tok(hyp, add_special_tokens=False)["input_ids"]
    n = enc["input_ids"].shape[1]
    hmask = np.zeros(n, bool)
    hmask[n - 1 - len(hyp_ids):n - 1] = True          # [CLS] art [SEP] hyp [SEP]
    return {k: v.to(dev) for k, v in enc.items() if k != "token_type_ids"}, hmask


def embed(which="ft", chunk=20):
    import torch
    import run_eval as R
    d = labeled()
    out = EMB.format(which)
    cache = dict(np.load(out, allow_pickle=True)) if os.path.exists(out) else {}
    done = set(cache.get("unit_id", []))
    todo = d[~d.unit_id.isin(done)]
    print(f"{which}: done {len(done)}, todo {len(todo)}")
    tok, model, enc = load_model(which)
    dev = R.device()
    arts = premise_of(todo.doc_id.values)
    acc = {k: list(v) for k, v in cache.items()}
    for k in ["unit_id", "cls_g", "cls_c", "hyp_g", "hyp_c", "p_g", "p_c"]:
        acc.setdefault(k, [])
    for i, (r, art) in enumerate(zip(todo.itertuples(), arts)):
        res = {}
        for tag, hyp in [("g", r.gold_sent), ("c", r.corr_sent)]:
            x, hm = encode(tok, art, hyp, dev)
            with torch.no_grad():
                if model is not None:
                    o = model(**x, output_hidden_states=True)
                    res["p_" + tag] = torch.softmax(o.logits, -1)[0, 1].item()
                    hs = o.hidden_states
                else:
                    hs = enc(**x, output_hidden_states=True).hidden_states
                    res["p_" + tag] = np.nan
            H = torch.stack([h[0] for h in hs]).float().cpu().numpy()      # L x T x D
            res["cls_" + tag] = H[:, 0].astype(np.float16)
            res["hyp_" + tag] = H[:, hm].mean(1).astype(np.float16)
        acc["unit_id"].append(r.unit_id)
        for k, v in res.items():
            acc[k].append(v)
        if (i + 1) % chunk == 0 or i + 1 == len(todo):
            np.savez(out, **{k: np.array(v) for k, v in acc.items()})
            pc = np.array(acc["p_c"][-chunk:], float)
            print(f"  {len(acc['unit_id'])} embedded; last chunk mean p(faithful) corrupted={np.nanmean(pc):.2f}",
                  flush=True)


# ------------------------------------------------------------------ probe / clustering
RULE_TYPES = ["negation", "number", "definiteness", "agreement_suffix"]


def data(which):
    z = dict(np.load(EMB.format(which), allow_pickle=True))
    d = labeled().set_index("unit_id").loc[z["unit_id"]].reset_index()
    return d, z


def oof_scores(X, y, groups, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    s = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, groups):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))
        clf.fit(X[tr], y[tr])
        s[te] = clf.decision_function(X[te])
    return s


def probe():
    from sklearn.metrics import roc_auc_score, silhouette_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rows, trows, crows = [], [], []
    for which in ["ft", "base"]:
        if not os.path.exists(EMB.format(which)):
            continue
        d, z = data(which)
        n = len(d)
        g2 = np.r_[d.doc_id.values, d.doc_id.values]
        y2 = np.r_[np.zeros(n), np.ones(n)]                       # 1 = corrupted
        kind = d.rule.values
        for pool in ["cls", "hyp"]:
            for L in range(z[pool + "_g"].shape[1]):
                X = np.r_[z[pool + "_g"][:, L], z[pool + "_c"][:, L]].astype(np.float32)
                s = oof_scores(X, y2, g2)
                paired = (s[n:] > s[:n])
                rows.append({"model": which, "pool": pool, "layer": L + 1, "auc": roc_auc_score(y2, s),
                             "paired": paired.mean(), "n_units": n})
                for t in RULE_TYPES + ["other"]:
                    m = kind == t
                    if m.sum() >= 10:
                        trows.append({"model": which, "pool": pool, "layer": L + 1, "type": t, "n": int(m.sum()),
                                      "paired": paired[m].mean()})
                # type encoding: diff vectors, single-kind rule types only
                m = np.isin(kind, RULE_TYPES)
                D = (z[pool + "_c"][m, L] - z[pool + "_g"][m, L]).astype(np.float32)
                yt = kind[m]
                clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000))
                pred = cross_val_predict(clf, D, yt, groups=d.doc_id.values[m], cv=GroupKFold(5))
                Dn = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-6)
                crows.append({"model": which, "pool": pool, "layer": L + 1, "type_acc": (pred == yt).mean(),
                              "majority": pd.Series(yt).value_counts(normalize=True).iloc[0],
                              "silhouette_cos": silhouette_score(Dn, yt, metric="cosine"), "n": int(m.sum())})
            print(which, pool, "done", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "layers_probe.csv"), index=False)
    pd.DataFrame(trows).to_csv(os.path.join(HERE, "layers_probe_by_type.csv"), index=False)
    pd.DataFrame(crows).to_csv(os.path.join(HERE, "layers_type_encoding.csv"), index=False)
    # model's own output on the same units (reference line)
    d, z = data("ft")
    print("MorphEvalHeb output: paired detection", (z["p_c"] < z["p_g"]).mean().round(3),
          "| flagged corrupted", (z["p_c"] < 0.5).mean().round(3), "| false alarms gold", (z["p_g"] < 0.5).mean().round(3))


def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    os.makedirs(FIG, exist_ok=True)
    P = pd.read_csv(os.path.join(HERE, "layers_probe.csv"))
    T = pd.read_csv(os.path.join(HERE, "layers_probe_by_type.csv"))
    C = pd.read_csv(os.path.join(HERE, "layers_type_encoding.csv"))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for (m, pool), g in P.groupby(["model", "pool"]):
        ax[0].plot(g.layer, g.paired, label=f"{'fine-tuned' if m == 'ft' else 'base'} / {pool}",
                   ls="-" if m == "ft" else "--")
    ax[0].set(title="Linear probe: corrupted vs. faithful sentence\n(paired accuracy, 5-fold by article)",
              xlabel="layer", ylabel="paired accuracy", ylim=(0.4, 1.01)); ax[0].legend(fontsize=8)
    for (m, pool), g in C.groupby(["model", "pool"]):
        ax[1].plot(g.layer, g.type_acc, label=f"{'fine-tuned' if m == 'ft' else 'base'} / {pool}",
                   ls="-" if m == "ft" else "--")
    ax[1].axhline(C.majority.iloc[0], color="gray", lw=0.8, ls=":", label="majority")
    ax[1].set(title="Which error type? (probe on corrupted − faithful)", xlabel="layer", ylabel="accuracy")
    ax[1].legend(fontsize=8)
    h = T[(T.model == "ft") & (T.pool == "cls")].pivot(index="type", columns="layer", values="paired")
    im = ax[2].imshow(h.values, aspect="auto", cmap="viridis", vmin=0.5, vmax=1)
    ax[2].set_yticks(range(len(h)), [f"{t} (n={T[T.type == t].n.iloc[0]})" for t in h.index], fontsize=8)
    ax[2].set_xticks(range(0, h.shape[1], 3), h.columns[::3]); ax[2].set(title="Fine-tuned CLS: detection by error type", xlabel="layer")
    plt.colorbar(im, ax=ax[2])
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "layers_probe.png"), dpi=160); plt.close()
    # t-SNE of diff vectors at 4 layers
    for which in ["ft", "base"]:
        if not os.path.exists(EMB.format(which)):
            continue
        d, z = data(which)
        m = np.isin(d.rule.values, RULE_TYPES)
        layers = [1, 8, 16, 28]
        fig, ax = plt.subplots(1, 4, figsize=(16, 4))
        for a, L in zip(ax, layers):
            D = (z["hyp_c"][m, L - 1] - z["hyp_g"][m, L - 1]).astype(np.float32)
            E2 = TSNE(2, random_state=0, perplexity=30, metric="cosine").fit_transform(D)
            for t in RULE_TYPES:
                k = d.rule.values[m] == t
                a.scatter(E2[k, 0], E2[k, 1], s=8, label=t)
            a.set(title=f"layer {L}", xticks=[], yticks=[])
        ax[0].legend(fontsize=8, markerscale=2)
        fig.suptitle(f"{'Fine-tuned MorphEvalHeb' if which == 'ft' else 'Base NeoDictaBERT'}: corrupted − faithful sentence (hypothesis mean), t-SNE")
        plt.tight_layout(); plt.savefig(os.path.join(FIG, f"layers_tsne_{which}.png"), dpi=150); plt.close()
    print("figures saved")

# ------------------------------------------------------------------ causal erasure (activation engineering)
ERASE = os.path.join(HERE, "cache", "layers_erase.jsonl")


def erase(per_type=40, chunk=20, seed=0, orth=False):
    """Project out a per-type 'error direction' (mean corrupted−faithful, hypothesis tokens) at EVERY layer, then
    re-score held-out units. Directions come from one half of the articles, tests on the other half (no leakage).
    Specificity test: does erasing the negation direction hurt negation detection only?"""
    import hashlib
    import torch
    import run_eval as R
    d, z = data("ft")
    half = np.array([int(hashlib.md5(str(x).encode()).hexdigest(), 16) % 2 for x in d.doc_id])
    tr, te = half == 0, half == 1
    D = (z["hyp_c"] - z["hyp_g"]).astype(np.float32)                 # n x L x H
    dirs = {}
    for t in RULE_TYPES:
        m = tr & (d.rule.values == t)
        dirs[t] = D[m].mean(0)
    dirs["all_types"] = D[tr].mean(0)
    rng = np.random.default_rng(seed)
    dirs["random"] = rng.standard_normal(D.shape[1:]).astype(np.float32)
    dirs = {k: v / np.linalg.norm(v, axis=1, keepdims=True) for k, v in dirs.items()}
    if orth:   # keep only the type-specific part: remove the shared all-types component per layer
        a = dirs["all_types"]
        dirs = {k: v - (v * a).sum(1, keepdims=True) * a for k, v in dirs.items() if k in RULE_TYPES + ["random"]}
        dirs = {k: v / np.linalg.norm(v, axis=1, keepdims=True) for k, v in dirs.items()}
        print("cos(type, all_types) before orth:", {t: round(float(np.mean((D[tr & (d.rule.values == t)].mean(0) /
              np.linalg.norm(D[tr & (d.rule.values == t)].mean(0), axis=1, keepdims=True) * a).sum(1))), 2) for t in RULE_TYPES})
    print("train units per type:", {t: int((tr & (d.rule.values == t)).sum()) for t in RULE_TYPES})
    test = []
    for t in RULE_TYPES + ["other"]:
        idx = np.where(te & (d.rule.values == t))[0]
        test += list(rng.permutation(idx)[:per_type])
    test = d.iloc[test]
    out = ERASE.replace(".jsonl", "_orth.jsonl") if orth else ERASE
    done = set()
    if os.path.exists(out):
        done = {(j["dir"], j["unit_id"]) for j in map(json.loads, open(out))}
    tok, model = R.load_cls("NLI-hebrew-binary-correctness-metric")
    dev = R.device()
    blocks = model.model.transformer_encoder
    cur = {"U": None}

    def hook(L):
        def f(mod, inp, out):
            if cur["U"] is None:
                return out
            x = out[0]
            u = cur["U"][L]
            x = x - (x @ u)[..., None] * u
            return (x,) + tuple(out[1:])
        return f
    hs = [b.register_forward_hook(hook(L)) for L, b in enumerate(blocks)]
    arts = dict(zip(test.doc_id, premise_of(test.doc_id.values)))
    jobs = [(k, r) for k in ["none"] + list(dirs) for r in test.itertuples() if (k, r.unit_id) not in done]
    print(f"test units {len(test)}, jobs {len(jobs)} (done {len(done)})")
    buf = []
    for i, (k, r) in enumerate(jobs):
        cur["U"] = None if k == "none" else torch.tensor(dirs[k], device=dev)
        p = {}
        for tag, hyp in [("g", r.gold_sent), ("c", r.corr_sent)]:
            x, _ = encode(tok, arts[r.doc_id], hyp, dev)
            with torch.no_grad():
                p[tag] = torch.softmax(model(**x).logits, -1)[0, 1].item()
        buf.append({"dir": k, "unit_id": int(r.unit_id), "rule": r.rule, "p_g": p["g"], "p_c": p["c"]})
        if len(buf) == chunk or i + 1 == len(jobs):
            with open(out, "a") as f:
                for b in buf:
                    f.write(json.dumps(b) + "\n")
            b = pd.DataFrame(buf)
            print(f"  {i + 1}/{len(jobs)} dir={k}: paired det {(b.p_c < b.p_g).mean():.2f}, flagged {(b.p_c < .5).mean():.2f}, "
                  f"false alarm {(b.p_g < .5).mean():.2f}", flush=True)
            buf = []
    for h in hs:
        h.remove()
    erase_summary(orth)


def erase_summary(orth=False):
    src = ERASE.replace(".jsonl", "_orth.jsonl") if orth else ERASE
    e = pd.DataFrame([json.loads(x) for x in open(src)]).drop_duplicates(["dir", "unit_id"], keep="last")
    e["det"] = e.p_c < e.p_g
    e["flag"] = e.p_c < 0.5
    e["fa"] = e.p_g < 0.5
    t = e.groupby(["dir", "rule"]).agg(n=("det", "size"), paired=("det", "mean"), flagged=("flag", "mean"),
                                       false_alarm=("fa", "mean")).reset_index()
    t.to_csv(os.path.join(HERE, "layers_erase_summary_orth.csv" if orth else "layers_erase_summary.csv"), index=False)
    print(t.pivot(index="dir", columns="rule", values="flagged").round(2))
    return t

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "sample":
        sample()
    elif cmd == "label":
        label(int(sys.argv[2]) if len(sys.argv) > 2 else None)
    elif cmd == "embed":
        embed(sys.argv[2] if len(sys.argv) > 2 else "ft")
    elif cmd == "probe":
        probe()
    elif cmd == "fig":
        figures()
    elif cmd == "erase":
        erase(orth=len(sys.argv) > 2 and sys.argv[2] == "orth")
    elif cmd == "erase_summary":
        erase_summary()
