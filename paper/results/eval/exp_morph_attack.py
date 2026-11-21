"""Test 2: morphology attacker. Single-word morphological edits of FAITHFUL summary sentences; how often does
MorphEvalHeb miss them, per edit kind? (search-based attacker; the cheap version of an RL attacker)

Edits are generated from Dicta's analysis of the word (dicta-il/dictabert-joint, cached in cache/morph_parse.jsonl)
and kept only if the new form occurs in the Hebrew corpus of the benchmark (articles + summaries), so they are
real words. Kinds:
  gender          verb/adj/noun gender flip (נרשם<->נרשמה, מתאר<->מתארת, ים<->ות)
  pronoun_suffix  pronominal ending flip (שלו<->שלה, ידיו<->ידיה, הם<->הן)
  prefix          prefix-letter swap (מ<->ל<->ב, add/remove ש or ו)
  definiteness    add/remove the article ה on a noun/adjective
  negation        insert לא before the main verb (control: a clear meaning change)

    python exp_morph_attack.py build [N]   -> morph_attack_candidates.csv (N faithful sentences, default 200)
    python exp_morph_attack.py score       -> cache/morph_attack_scores.jsonl (chunks of 20 sentences)
    python exp_morph_attack.py analyze     -> morph_attack_summary.csv, figures/morph_attack.png
"""
import collections
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_morph_labels as M  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
UNITS = os.path.join(HERE, "multisource_units.csv")
CAND = os.path.join(HERE, "morph_attack_candidates.csv")
SCORES = os.path.join(HERE, "cache", "morph_attack_scores.jsonl")
PUNCT = ".,\"'():;!?"


def vocab():
    import run_eval as R
    b = R.load_bench()
    c = collections.Counter()
    for t in pd.concat([b.premise, b.hypothesis]).drop_duplicates():
        c.update(w.strip(PUNCT) for w in str(t).split())
    return c


def gender_forms(w, pos, feats):
    out = []
    g, n = feats.get("Gender"), feats.get("Number")
    if n == "Plur":
        if w.endswith("ים"):
            out.append(w[:-2] + "ות")
        elif w.endswith("ות"):
            out.append(w[:-2] + "ים")
    elif g == "Masc":
        out += [w + "ה", w + "ת"]
        if w[-1:] in "כמנפצ":
            pass
        fin = {"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"}
        if w[-1:] in fin:
            out += [w[:-1] + fin[w[-1]] + "ה", w[:-1] + fin[w[-1]] + "ת"]
    elif g == "Fem":
        if w.endswith("תה"):
            out.append(w[:-2] + "ה")                      # הייתה -> היה
        elif w.endswith("ה") or w.endswith("ת"):
            base = w[:-1]
            out.append(base)
            fin = {"כ": "ך", "מ": "ם", "נ": "ן", "פ": "ף", "צ": "ץ"}
            if base[-1:] in fin:
                out.append(base[:-1] + fin[base[-1]])
        if w.endswith("ית"):
            out.append(w[:-2] + "י")
    return out


def pronoun_forms(w, lex):
    out = []
    if lex and lex != w:
        if w.endswith("הם"):
            out.append(w[:-2] + "הן")
        elif w.endswith("הן"):
            out.append(w[:-2] + "הם")
        elif w.endswith("יו"):
            out.append(w[:-2] + "יה")
        elif w.endswith("ו") and len(w) >= 3:
            out.append(w[:-1] + "ה")
    if w in ("שלו", "שלה", "לו", "לה", "בו", "בה", "אותו", "אותה", "ממנו", "ממנה", "עליו", "עליה"):
        sw = {"שלו": "שלה", "שלה": "שלו", "לו": "לה", "לה": "לו", "בו": "בה", "בה": "בו", "אותו": "אותה",
              "אותה": "אותו", "ממנו": "ממנה", "ממנה": "ממנו", "עליו": "עליה", "עליה": "עליו"}
        out.append(sw[w])
    return out


def prefix_forms(w, pre):
    out = []
    pre = [p for p in pre if p != "DET"]
    if w[:1] in "מלב" and pre:
        out += [x + w[1:] for x in "מלב" if x != w[0]]
    if w[:1] == "ש" and pre:
        out.append(w[1:])
    if w[:1] == "ו" and pre:
        out.append(w[1:])
    return out


def definiteness_forms(w, pos, pre):
    if pos not in ("NOUN", "ADJ"):
        return []
    if "DET" in pre and w.startswith("ה"):
        return [w[1:]]
    if not pre and not w.startswith("ה"):
        return ["ה" + w]
    return []


def build(n=200, seed=7):
    u = pd.read_csv(UNITS)
    c = M.parse_all(list(u.gold_sent))
    voc = vocab()
    s = u.drop_duplicates("gold_sent")
    s = s[s.gold_sent.str.split().str.len().between(6, 45)]
    s = s.sample(frac=1, random_state=seed).groupby("doc_id").head(2)
    # proportional to source
    quota = (s.source.value_counts(normalize=True) * n).round().clip(lower=1).astype(int)
    s = pd.concat([g.head(quota[src]) for src, g in s.groupby("source")]).head(n)
    rows = []
    for r in s.itertuples():
        toks = c[r.gold_sent]
        words = r.gold_sent.split()
        for i, t in enumerate(toks):
            w = t["w"]
            if not re.fullmatch(r"[א-ת]+", w):
                continue
            feats = t["feats"] or {}
            cands = [("gender", x) for x in gender_forms(w, t["pos"], feats) if t["pos"] in ("VERB", "ADJ", "AUX")]
            cands += [("pronoun_suffix", x) for x in pronoun_forms(w, t["lex"])]
            cands += [("prefix", x) for x in prefix_forms(w, t["pre"])]
            cands += [("definiteness", x) for x in definiteness_forms(w, t["pos"], t["pre"])]
            for kind, x in cands:
                if x == w or voc[x] < 2:
                    continue
                # replace the i-th occurrence of w in the raw sentence (punctuation kept)
                k = [j for j, ww in enumerate(words) if ww.strip(PUNCT) == w]
                if not k:
                    continue
                j = k[0]
                new = words[:j] + [words[j].replace(w, x, 1)] + words[j + 1:]
                rows.append({"unit_id": r.Index, "doc_id": r.doc_id, "source": r.source, "kind": kind, "orig": w,
                             "new": x, "pos": t["pos"], "gold_sent": r.gold_sent, "attacked": " ".join(new)})
        roots = [i for i, t in enumerate(toks) if t["dep"] == "root" and t["pos"] == "VERB"]
        if roots and not any(t["lex"] in M.NEG for t in toks):
            w = toks[roots[0]]["w"]
            k = [j for j, ww in enumerate(words) if ww.strip(PUNCT) == w]
            if k:
                new = words[:k[0]] + ["לא"] + words[k[0]:]
                rows.append({"unit_id": r.Index, "doc_id": r.doc_id, "source": r.source, "kind": "negation",
                             "orig": w, "new": "לא " + w, "pos": "VERB", "gold_sent": r.gold_sent,
                             "attacked": " ".join(new)})
    d = pd.DataFrame(rows).drop_duplicates(["gold_sent", "attacked"])
    d = d.sample(frac=1, random_state=seed).groupby(["gold_sent", "kind"]).head(3).sort_index()   # <=3 per kind per sentence
    d.to_csv(CAND, index=False)
    print(len(s), "sentences;", len(d), "attacks;", d.kind.value_counts().to_dict())
    print("sentences with >=1 attack per kind:", d.groupby("kind").gold_sent.nunique().to_dict())


def score(chunk=20):
    import torch
    import run_eval as R
    from exp_layers import premise_of, MAX_LEN
    d = pd.read_csv(CAND)
    done = set()
    if os.path.exists(SCORES):
        done = {json.loads(x)["text"] for x in open(SCORES)}
    sents = list(dict.fromkeys(d.gold_sent))
    tok, model = R.load_cls("NLI-hebrew-binary-correctness-metric")
    dev = R.device()
    docs = d.drop_duplicates("gold_sent").set_index("gold_sent").doc_id
    arts = dict(zip(docs.values, premise_of(docs.values)))
    todo = [s for s in sents if s not in done or not set(d[d.gold_sent == s].attacked) <= done]
    print(f"{len(todo)} of {len(sents)} sentences to score")
    for i in range(0, len(todo), chunk):
        buf = []
        for s in todo[i:i + chunk]:
            art = arts[docs[s]]
            for text in [s] + list(d[d.gold_sent == s].attacked):
                if text in done:
                    continue
                e = tok(art, text, return_tensors="pt", truncation="only_first", max_length=MAX_LEN)
                x = {k: v.to(dev) for k, v in e.items() if k != "token_type_ids"}
                with torch.no_grad():
                    p = torch.softmax(model(**x).logits, -1)[0, 1].item()
                buf.append({"text": text, "p": p})
                done.add(text)
        with open(SCORES, "a") as f:
            for b in buf:
                f.write(json.dumps(b, ensure_ascii=False) + "\n")
        sc = {b["text"]: b["p"] for b in buf}
        g = d[d.gold_sent.isin(todo[i:i + chunk])]
        gp = np.array([sc.get(s, np.nan) for s in todo[i:i + chunk]])
        ap = g.attacked.map(sc)
        print(f"  {min(i + chunk, len(todo))}/{len(todo)}: faithful flagged {np.nanmean(gp < .5):.2f} | "
              f"attacks missed (p>=.5) {np.nanmean(ap >= .5):.2f} | by kind "
              + " ".join(f"{k}:{(ap[g.kind == k] >= .5).mean():.2f}" for k in sorted(g.kind.unique())), flush=True)


def analyze():
    from exp_multisource import wilson
    d = pd.read_csv(CAND)
    sc = {}
    for x in open(SCORES):
        j = json.loads(x)
        sc[j["text"]] = j["p"]
    d["p"] = d.attacked.map(sc)
    d["p_gold"] = d.gold_sent.map(sc)
    d = d.dropna(subset=["p", "p_gold"])
    d = d[d.p_gold >= 0.5]                        # only sentences the metric itself judges faithful
    d["missed"] = d.p >= 0.5
    d["drop"] = d.p_gold - d.p
    rows = []
    for k, g in d.groupby("kind"):
        per_sent = g.groupby("gold_sent").missed.max()            # attacker success: >=1 evading edit
        lo, hi = wilson(int(g.missed.sum()), len(g))
        rows.append({"kind": k, "n_edits": len(g), "n_sents": per_sent.size, "missed_rate": g.missed.mean(),
                     "ci_lo": lo, "ci_hi": hi, "attack_success_per_sentence": per_sent.mean(),
                     "median_p_drop": g["drop"].median()})
    t = pd.DataFrame(rows).sort_values("missed_rate", ascending=False)
    t.to_csv(os.path.join(HERE, "morph_attack_summary.csv"), index=False)
    print(t.round(3).to_string(index=False))
    print("faithful sentences flagged (false alarm):", round((pd.Series([sc[s] for s in d.gold_sent.unique()]) < .5).mean(), 3))
    d.to_csv(os.path.join(HERE, "morph_attack_scored.csv"), index=False)
    by_pos = d[d.kind == "gender"].groupby("pos").missed.agg(["mean", "size"])
    print("gender by POS:\n", by_pos.round(2))
    figure(t)


def figure(t):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = t.sort_values("missed_rate")
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    y = np.arange(len(t))
    ax.barh(y, t.missed_rate, xerr=[t.missed_rate - t.ci_lo, t.ci_hi - t.missed_rate], color="#4C78A8", capsize=3)
    ax.set_yticks(y, [f"{k} (n={n})" for k, n in zip(t.kind, t.n_edits)])
    ax.set(xlabel="share of single-word edits MorphEvalHeb misses (p(faithful) ≥ 0.5)", xlim=(0, 1),
           title="Morphology attacker on faithful summary sentences")
    plt.tight_layout(); plt.savefig(os.path.join(HERE, "figures", "morph_attack.png"), dpi=160); plt.close()


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build(int(sys.argv[2]) if len(sys.argv) > 2 else 200)
    elif cmd == "score":
        score()
    elif cmd == "analyze":
        analyze()
