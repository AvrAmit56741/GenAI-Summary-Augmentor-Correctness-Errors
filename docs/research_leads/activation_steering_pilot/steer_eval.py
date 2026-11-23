"""Quantitative follow-up of the steering pilot (steer.py, 6 sentences, by eye).

1. vectors: build TWO gender directions from many real benchmark edits (exp_morph_attack gender edits on
   MAFAT reference sentences, VERB only): masc->fem and fem->masc, each = mean(h(edited) - h(original)) of
   DictaLM-3.0-1.7B-Instruct last-token hidden states at layer 24 (same recipe as the geometry pilot).
   Built only from articles NOT used for testing. Plus a norm-matched random control.
2. run N: held-out sentences (half with a masculine target verb, half feminine). The model is asked to
   repeat each sentence verbatim under: baseline, m2f x s, f2m x s, random x s. Scored automatically:
     flip   = the target verb appears in its gender-swapped form (the exact edited form) instead of the original
     clean  = flip and no other word changed
     verbatim = output identical to the sentence (word level)
   -> steer_eval.jsonl (chunks of 20 sentences)
3. analyze -> steer_eval_summary.csv

    ~/.cache/morphevalheb_env/bin/python steer_eval.py vectors
    ~/.cache/morphevalheb_env/bin/python steer_eval.py run 20
    ~/.cache/morphevalheb_env/bin/python steer_eval.py analyze
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(HERE, "..", "..", "..", "paper", "results", "eval")
sys.path.insert(0, EVAL)
MODEL_ID = "dicta-il/DictaLM-3.0-1.7B-Instruct"
LAYER = 24
VEC = os.path.join(HERE, "steer_vectors.npz")
OUT = os.path.join(HERE, "steer_eval.jsonl")
PROMPT = "חזור בדיוק על המשפט הבא, מילה במילה, ללא שום שינוי:\n\n{sentence}"
SCALES = [float(x) for x in os.environ.get("SCALES", "2,4").split(",")]


def edits():
    import exp_morph_labels as M
    d = pd.read_csv(os.path.join(EVAL, "morph_attack_candidates.csv"))
    d = d[(d.kind == "gender") & (d.pos == "VERB")].copy()
    c = M.parse_all([])
    def gender(r):
        for t in c.get(r.gold_sent, []):
            if t["w"] == r.orig:
                return (t["feats"] or {}).get("Gender")
        return None
    d["g"] = d.apply(gender, axis=1)
    d = d[d.g.isin(["Masc", "Fem"])]
    d["dir"] = np.where(d.g == "Masc", "m2f", "f2m")
    d["n_words"] = d.gold_sent.str.split().str.len()
    # the target word must occur exactly once so the automatic check is unambiguous
    d = d[d.apply(lambda r: r.gold_sent.split().count(r.orig) == 1, axis=1)]
    return d.reset_index(drop=True)


def split(d, seed=0):
    rng = np.random.default_rng(seed)
    docs = d.doc_id.unique()
    test_docs = set(rng.choice(docs, size=len(docs) // 3, replace=False))
    return d[~d.doc_id.isin(test_docs)], d[d.doc_id.isin(test_docs)]


def load():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to(dev).eval()
    return tok, m, dev


def vectors():
    import torch
    d = edits()
    tr, te = split(d)
    print("edits:", d.dir.value_counts().to_dict(), "| train:", tr.dir.value_counts().to_dict(),
          "| test:", te.dir.value_counts().to_dict())
    tok, m, dev = load()
    def h(text):
        x = tok(text, return_tensors="pt").to(dev)
        with torch.no_grad():
            return m(**x, output_hidden_states=True).hidden_states[LAYER][0, -1].float().cpu().numpy()
    V = {}
    for k in ["m2f", "f2m"]:
        g = tr[tr.dir == k]
        V[k] = np.mean([h(r.attacked) - h(r.gold_sent) for r in g.itertuples()], axis=0)
        print(f"  {k}: n={len(g)} norm={np.linalg.norm(V[k]):.2f}")
    rng = np.random.default_rng(0)
    r = rng.normal(size=V["m2f"].shape)
    V["random"] = r / np.linalg.norm(r) * np.linalg.norm(V["m2f"])
    cos = V["m2f"] @ V["f2m"] / np.linalg.norm(V["m2f"]) / np.linalg.norm(V["f2m"])
    print(f"  cosine(m2f, f2m) = {cos:.3f}")
    np.savez(VEC, **V)


def words(s):
    return re.sub(r"[^\w\s\"'-]", " ", s).split()


def score(out, r):
    a, b = words(r.gold_sent), words(out)
    flip = r.new in b and r.orig not in b
    import difflib
    ops = [o for o in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if o[0] != "equal"]
    n_changed = sum(max(i2 - i1, j2 - j1) for _, i1, i2, j1, j2 in ops)
    # gender flip on ANY word: aligned replaced words that differ only by a gender ending, or he/she they(m/f)
    pron = {("הוא", "היא"), ("היא", "הוא"), ("הם", "הן"), ("הן", "הם")}
    def gpair(x, y):
        if (x, y) in pron:
            return 1 if x in ("הוא", "הם") else -1
        for suf in ["ה", "ת", "ית", "ות"]:
            if y == x + suf or (x.endswith("ים") and y == x[:-2] + "ות"):
                return 1                                   # masc -> fem
            if x == y + suf or (y.endswith("ים") and x == y[:-2] + "ות"):
                return -1                                  # fem -> masc
        return 0
    g = [gpair(x, y) for op, i1, i2, j1, j2 in ops if op == "replace" and i2 - i1 == j2 - j1
         for x, y in zip(a[i1:i2], b[j1:j2])]
    return {"flip": flip, "clean": flip and n_changed == 1, "verbatim": n_changed == 0, "n_changed": n_changed,
            "to_fem": sum(v == 1 for v in g), "to_masc": sum(v == -1 for v in g)}


def run(n=20, chunk=20):
    import torch
    _, te = split(edits())
    te = te[te.n_words <= 25].drop_duplicates("gold_sent")
    te = pd.concat([te[te.dir == k].sample(frac=1, random_state=1) for k in ["m2f", "f2m"]])
    te = te.groupby("dir").head(n // 2)
    done = {(j["sent"], j["cond"]) for j in map(json.loads, open(OUT))} if os.path.exists(OUT) else set()
    V = dict(np.load(VEC))
    tok, m, dev = load()
    state = {"v": None}
    def hook(mod, inp, out):
        if state["v"] is None:
            return out
        hs = out[0] if isinstance(out, tuple) else out
        hs = hs + state["v"].to(hs.dtype)
        return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs
    hd = m.model.layers[LAYER - 1].register_forward_hook(hook)
    conds = [("baseline", None, 0)] + [(f"{k} x{s:g}", k, s) for s in SCALES for k in ["m2f", "f2m", "random"]]
    buf = []
    for i, r in enumerate(te.itertuples()):
        msgs = [{"role": "user", "content": PROMPT.format(sentence=r.gold_sent)}]
        x = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), return_tensors="pt").to(dev)
        for name, k, s in conds:
            if (r.gold_sent, name) in done:
                continue
            state["v"] = None if k is None else torch.tensor((V[k] * s).astype(np.float32), device=dev)
            with torch.no_grad():
                o = m.generate(**x, max_new_tokens=int(len(tok(r.gold_sent)["input_ids"]) * 1.5) + 10,
                               do_sample=False, pad_token_id=tok.eos_token_id)
            out = tok.decode(o[0][x["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            buf.append({"sent": r.gold_sent, "dir": r.dir, "orig": r.orig, "new": r.new, "cond": name,
                        "out": out, **score(out, r)})
        state["v"] = None
        if (i + 1) % chunk == 0 or i + 1 == len(te):
            with open(OUT, "a") as f:
                for b in buf:
                    f.write(json.dumps(b, ensure_ascii=False) + "\n")
            buf = []
            print(f"  {i + 1}/{len(te)} sentences", flush=True)
            analyze()
    hd.remove()


def analyze():
    d = pd.DataFrame([json.loads(x) for x in open(OUT)])
    if "to_fem" not in d:
        d["to_fem"] = d["to_masc"] = np.nan
    d["any_to_fem"] = d.to_fem > 0
    d["any_to_masc"] = d.to_masc > 0
    t = d.groupby(["cond", "dir"])[["flip", "verbatim", "any_to_fem", "any_to_masc"]].mean().unstack("dir")
    n = d.groupby("dir").sent.nunique().to_dict()
    t.to_csv(os.path.join(HERE, "steer_eval_summary.csv"))
    print(f"sentences per direction: {n}  (dir = which way the TARGET flip goes: m2f = masc verb in the sentence)")
    print(t.round(2).to_string())


if __name__ == "__main__":
    {"vectors": vectors, "run": lambda: run(int(sys.argv[2]) if len(sys.argv) > 2 else 20),
     "analyze": analyze}[sys.argv[1]]()
