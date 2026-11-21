"""Test 3: activation patching. Where (layer x position) does MorphEvalHeb register each edit type?

Pairs: MINIMAL pairs built from layers_words_sample.csv: the faithful sentence with only ONE typed word changed to its
corrupted form (type from exp_morph_labels); kept if both tokenize to the same length.
Only pairs the model gets right (p_gold >= .5 > p_corr). For layer L and position set S, the faithful run's hidden
state (output of block L) at S is copied into the corrupted run; restoration = (p_patched - p_corr) / (p_gold - p_corr).
Positions: edit (the changed word's tokens), cls, hyp_rest (other summary tokens), article (all article tokens).

    python exp_patching.py run [per_kind]  -> cache/patching.jsonl (chunks of 20 pairs)
    python exp_patching.py analyze         -> patching_summary.csv, figures/patching.png
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_morph_labels as M  # noqa: E402
from exp_layers_words import SAMPLE, char_spans  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cache", "patching.jsonl")
LAYERS = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28]
POS = ["edit", "cls", "hyp_rest", "article"]


def run(per_kind=25, chunk=20, seed=7):
    import torch
    import run_eval as R
    from exp_layers import premise_of, MAX_LEN
    w = pd.read_csv(SAMPLE).sample(frac=1, random_state=seed)
    w = w[~w.kind.isin(["number_infl", "person", "lexical"])]
    c = M.parse_all([])
    done = set()
    if os.path.exists(OUT):
        done = {json.loads(x)["unit_id"] for x in open(OUT)}
    tok, model = R.load_cls("NLI-hebrew-binary-correctness-metric")
    dev = R.device()
    blocks = model.model.transformer_encoder
    state = {"L": None, "pos": None, "src": None}

    def hook(L):
        def f(mod, inp, out):
            if state["L"] != L:
                return out
            x = out[0].clone()
            x[0, state["pos"]] = state["src"][state["pos"]]
            return (x,) + tuple(out[1:])
        return f
    hs = [b.register_forward_hook(hook(L + 1)) for L, b in enumerate(blocks)]
    arts = dict(zip(w.doc_id.unique(), premise_of(w.doc_id.unique())))
    used = {}
    for x in (json.loads(l) for l in open(OUT)) if os.path.exists(OUT) else []:
        if x.get("status") == "ok":
            used[x["kind"]] = used.get(x["kind"], 0) + 1
    buf, n_try = [], 0
    for r in w.drop_duplicates("unit_id").itertuples():
        if r.unit_id in done or used.get(r.kind, 0) >= per_kind:
            continue
        art = arts[r.doc_id]
        gsp = char_spans(r.gold_sent, [t["w"] for t in c[r.gold_sent]])[r.gi]
        if gsp is None:
            continue
        # minimal pair: the faithful sentence with ONLY this one typed word changed
        csent = r.gold_sent[:gsp[0]] + r.corr_word + r.gold_sent[gsp[1]:]
        sp = (gsp[0], gsp[0] + len(r.corr_word))
        enc = {}
        for tag, sent in [("g", r.gold_sent), ("c", csent)]:
            e = tok(art, sent, return_tensors="pt", truncation="only_first", max_length=MAX_LEN,
                    return_offsets_mapping=True)
            offs = e.pop("offset_mapping")[0].numpy()
            seq = np.array([s if s is not None else -1 for s in e.sequence_ids(0)])
            enc[tag] = ({k: v.to(dev) for k, v in e.items() if k != "token_type_ids"}, offs, seq)
        xg, og, sg = enc["g"]
        xc, oc, sc = enc["c"]
        rec = {"unit_id": int(r.unit_id), "kind": r.kind, "gold_word": r.gold_word, "corr_word": r.corr_word,
               "minimal_sent": csent}
        if xg["input_ids"].shape != xc["input_ids"].shape or not (sg == sc).all():
            rec["status"] = "length_mismatch"
        else:
            diff = np.where((xg["input_ids"] != xc["input_ids"])[0].cpu().numpy())[0]
            word = np.where((sc == 1) & (oc[:, 0] < sp[1]) & (oc[:, 1] > sp[0]))[0]
            if r.kind == "negation" or len(diff) == 0 or not set(diff) <= set(word):
                rec["status"] = "not_single_edit"
            else:
                with torch.no_grad():
                    state["L"] = None
                    og_ = model(**xg, output_hidden_states=True)
                    pg = torch.softmax(og_.logits, -1)[0, 1].item()
                    H = [h[0] for h in og_.hidden_states]
                    pc = torch.softmax(model(**xc).logits, -1)[0, 1].item()
                rec.update({"p_g": pg, "p_c": pc})
                if not (pg >= .5 > pc):
                    rec["status"] = "not_detected"
                else:
                    pos = {"edit": diff, "cls": np.array([0]),
                           "hyp_rest": np.setdiff1d(np.where(sc == 1)[0], diff),
                           "article": np.where(sc == 0)[0]}
                    res = {}
                    for L in LAYERS:
                        for pn in POS:
                            state.update(L=L, pos=torch.tensor(pos[pn], device=dev), src=H[L - 1])
                            with torch.no_grad():
                                p = torch.softmax(model(**xc).logits, -1)[0, 1].item()
                            res[f"{pn}@{L}"] = (p - pc) / (pg - pc)
                    state["L"] = None
                    rec.update({"status": "ok", "restore": res})
                    used[r.kind] = used.get(r.kind, 0) + 1
        buf.append(rec)
        n_try += 1
        if sum(1 for b in buf if b["status"] == "ok") >= chunk or n_try % 100 == 0:
            flush(buf)
            buf = []
        if all(used.get(k, 0) >= per_kind for k in set(w.kind) - {"negation"}):
            break
    flush(buf)
    for h in hs:
        h.remove()


def flush(buf):
    if not buf:
        return
    with open(OUT, "a") as f:
        for b in buf:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")
    ok = [b for b in buf if b["status"] == "ok"]
    st = pd.Series([b["status"] for b in buf]).value_counts().to_dict()
    if ok:
        R = pd.DataFrame([b["restore"] for b in ok])
        best = R.mean().sort_values(ascending=False).head(3).round(2).to_dict()
        print(f"  chunk: {st} | kinds {pd.Series([b['kind'] for b in ok]).value_counts().to_dict()} | top mean restore {best}",
              flush=True)
    else:
        print(f"  chunk: {st}", flush=True)


def analyze():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [json.loads(x) for x in open(OUT)]
    print("status:", pd.Series([r["status"] for r in rows]).value_counts().to_dict())
    ok = [r for r in rows if r["status"] == "ok"]
    recs = []
    for r in ok:
        for k, v in r["restore"].items():
            pn, L = k.split("@")
            recs.append({"unit_id": r["unit_id"], "kind": r["kind"], "pos": pn, "layer": int(L), "restore": v})
    d = pd.DataFrame(recs)
    t = d.groupby(["kind", "pos", "layer"]).restore.agg(["mean", "median", "size"]).reset_index()
    t.to_csv(os.path.join(HERE, "patching_summary.csv"), index=False)
    allk = d.groupby(["pos", "layer"]).restore.mean().unstack(0)[POS]
    print("all kinds, mean restoration:\n", allk.round(2))
    # layer where the edit-token patch stops working and the CLS patch starts working, per kind
    piv = d.groupby(["kind", "pos", "layer"]).restore.mean().unstack("layer")
    print(piv.round(2).to_string())
    kinds = sorted(d.kind.unique())
    fig, ax = plt.subplots(1, len(kinds) + 1, figsize=(3.1 * (len(kinds) + 1), 3.2), sharey=True)
    for a, k in zip(ax, ["all"] + kinds):
        g = d if k == "all" else d[d.kind == k]
        m = g.groupby(["pos", "layer"]).restore.mean().unstack("layer").loc[POS]
        a.imshow(m.values, aspect="auto", cmap="magma", vmin=0, vmax=1)
        a.set_xticks(range(len(LAYERS)), LAYERS, fontsize=7)
        a.set_title(f"{k} (n={g.unit_id.nunique()})", fontsize=9)
        a.set_xlabel("layer", fontsize=8)
    ax[0].set_yticks(range(len(POS)), POS)
    fig.suptitle("Activation patching: share of the faithful score restored by copying the faithful run's state")
    plt.tight_layout(); plt.savefig(os.path.join(HERE, "figures", "patching.png"), dpi=160); plt.close()


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        run(int(sys.argv[2]) if len(sys.argv) > 2 else 25)
    elif cmd == "analyze":
        analyze()
