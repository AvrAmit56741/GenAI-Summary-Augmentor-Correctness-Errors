"""Lead 17 (4): finding the article source of a FALSE summary claim, and checking the claim against it.

The author's original question: for a summary claim -- especially a false one -- can we find the article sentences
it came from, and verify the claim against them?  A corrupted claim is about the same article sentences as its
correct version, so the hand-labeled sources of the 39 gold claims (evidence_gold_labels.csv) are also the sources
of their corrupted versions (multisource_units.csv: 52 corrupted versions of 35 labeled claims).

  A  retrieval robustness   do the finders still find the source when the claim is false?
                            same finders, correct vs corrupted claim (hit@1, recall@3, all@3, MRR)
  B  verify on evidence     MorphEvalHeb on (top-k retrieved sentences, claim) vs on (whole article, claim);
                            on all scored units of exp_multisource (single + multi), paired detection + false alarms
  C  error localization     which claim WORD is wrong? MorphEvalHeb |grad x input| over claim words (premise = evidence)
                            vs a deterministic baseline (claim word absent from the evidence, prefix-stripped)

    python exp_false_claims.py A | B | C        (cached in cache/false_claims_*.jsonl, logs every 20)
"""
import difflib
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eval as R  # noqa: E402
import exp_translation as T  # noqa: E402
import exp_evidence as E  # noqa: E402
import exp_multisource as MS  # noqa: E402

OUT = R.HERE
FINDERS = ["lex", "bm25", "e5", "bge", "rerank", "nli"]   # attr dropped: near chance on correct claims (hit@1 0.05)
CA = os.path.join(R.CACHE, "false_claims_A.jsonl")
CB = os.path.join(R.CACHE, "false_claims_B.jsonl")
CC = os.path.join(R.CACHE, "false_claims_C.jsonl")


def jl(path):
    out = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line); out[r["k"]] = r
    return out


def h(*xs):
    return hashlib.sha1("\x00".join(xs).encode()).hexdigest()


def premises():
    return R.load_bench().drop_duplicates("doc_id").set_index("doc_id").premise


def labeled_pairs():
    lab = pd.read_csv(E.GOLD_LABELS); lab = lab[lab.multi.notna()]
    U = pd.read_csv(MS.UNITS)
    m = U.merge(lab[["claim_id", "doc_id", "claim", "gold", "multi"]],
                left_on=["doc_id", "gold_sent"], right_on=["doc_id", "claim"])
    return m.reset_index(drop=True)


def variants(sc):
    v = {f: np.asarray(sc[f]) for f in FINDERS}
    v["rerank+lex"] = E.rrf([v["rerank"], v["lex"]])
    v["fuse_all"] = E.rrf([v[f] for f in FINDERS])
    return v


# ---------------------------------------------------------------- A
def run_A():
    P = labeled_pairs(); prem = premises(); cache = jl(CA)
    t0 = time.time()
    with open(CA, "a", encoding="utf-8") as fh:
        for i, r in enumerate(P.itertuples()):
            for which, claim in (("correct", r.gold_sent), ("false", r.corr_sent)):
                k = h(which, r.doc_id, claim)
                if k in cache:
                    continue
                _, sc = E.all_scores(claim, prem[r.doc_id], finders=FINDERS)
                rec = {"k": k, **{f: [float(x) for x in sc[f]] for f in FINDERS}}
                fh.write(json.dumps(rec) + "\n"); fh.flush(); cache[k] = rec
            if (i + 1) % 20 == 0 or i + 1 == len(P):
                print(f"  A {i + 1}/{len(P)} pairs  {time.time() - t0:.0f}s", flush=True)
    rows = []
    for r in P.itertuples():
        gold = {int(x) for x in str(r.gold).split()}
        for which, claim in (("correct", r.gold_sent), ("false", r.corr_sent)):
            for f, v in E.eval_finders(variants(cache[h(which, r.doc_id, claim)]), gold, r.multi).items():
                rows.append({"unit": r.Index, "claim_id": r.claim_id, "multi": int(r.multi), "error_types": r.error_types,
                             "claim": which, "finder": f, **v})
    e = pd.DataFrame(rows)
    e.to_csv(os.path.join(OUT, "false_claims_retrieval.csv"), index=False)
    agg = e.groupby(["finder", "claim"])[["hit1", "rec3", "all3", "mrr"]].mean().unstack("claim").round(2)
    print(f"A: {len(P)} false claims of {P.claim_id.nunique()} labeled claims "
          f"({P.drop_duplicates('claim_id').multi.sum():.0f} multi-source)")
    print(agg.sort_values(("mrr", "false"), ascending=False).to_string())
    # same top-1 sentence for correct and false version?
    same = []
    for r in P.itertuples():
        a = variants(cache[h("correct", r.doc_id, r.gold_sent)]); b = variants(cache[h("false", r.doc_id, r.corr_sent)])
        same.append({f: int(np.argmax(a[f]) == np.argmax(b[f])) for f in a})
    print("top-1 unchanged by the corruption:", pd.DataFrame(same).mean().round(2).to_dict())


# ---------------------------------------------------------------- B
_MOD = {}


def morph():
    if "m" not in _MOD:
        _MOD["m"] = R.load_cls("NLI-hebrew-binary-correctness-metric")
    return _MOD["m"]


def morph_score(prem, hyp):
    import torch
    tok, mod = morph()
    enc = tok(prem, hyp, return_tensors="pt", truncation=True, max_length=3072).to(R.device())
    with torch.no_grad():
        return float(torch.softmax(mod(**enc).logits.float(), -1)[0, 1])


def evidence_order(claim, sents):
    return np.argsort(-E.rrf([E.rerank_scores(claim, sents), E.lex_scores(claim, sents)]))


def evidence_text(claim, sents, k, order=None):
    order = evidence_order(claim, sents) if order is None else order
    top = sorted(order[:k])                                              # keep article order
    return " ".join(sents[i] for i in top), [int(i) for i in top]


def run_B(ks=(1, 3, 5)):
    S = pd.read_csv(os.path.join(OUT, "multisource_subset_scored.csv"))
    prem = premises(); cache = jl(CB); t0 = time.time()
    with open(CB, "a", encoding="utf-8") as fh:
        for i, r in enumerate(S.itertuples()):
            sents = T.split_sents(prem[r.doc_id])
            for which, claim in (("gold", r.gold_sent), ("corr", r.corr_sent)):
                k = h(which, r.doc_id, claim)
                if k in cache:
                    continue
                rec = {"k": k}
                order = evidence_order(claim, sents)
                for kk in ks:
                    ev, idx = evidence_text(claim, sents, kk, order)
                    rec[f"top{kk}"] = morph_score(ev, claim); rec[f"idx{kk}"] = idx
                fh.write(json.dumps(rec) + "\n"); fh.flush(); cache[k] = rec
            if (i + 1) % 20 == 0 or i + 1 == len(S):
                print(f"  B {i + 1}/{len(S)} units  {time.time() - t0:.0f}s", flush=True)
    rows = []
    for kk in ks:
        S[f"gold:top{kk}"] = [cache[h("gold", r.doc_id, r.gold_sent)][f"top{kk}"] for r in S.itertuples()]
        S[f"corr:top{kk}"] = [cache[h("corr", r.doc_id, r.corr_sent)][f"top{kk}"] for r in S.itertuples()]
    S["gold:article"] = S["gold:MorphEvalHeb"]; S["corr:article"] = S["corr:MorphEvalHeb"]
    for prem_kind in [f"top{k}" for k in ks] + ["article"]:
        for g in ("single", "multi", "all"):
            q = S if g == "all" else S[S.group == g]
            det = (q[f"gold:{prem_kind}"] > q[f"corr:{prem_kind}"])
            fa = q[f"gold:{prem_kind}"] < .5
            caught = q[f"corr:{prem_kind}"] < .5
            rows.append({"premise": prem_kind, "group": g, "n": len(q), "paired_detection": det.mean(),
                         "det_ci": "[%.2f,%.2f]" % MS.wilson(det.sum(), len(q)),
                         "false_alarm": fa.mean(), "fa_ci": "[%.2f,%.2f]" % MS.wilson(fa.sum(), len(q)),
                         "corrupted_flagged": caught.mean(), "auc": MS.auc(q[f"gold:{prem_kind}"], q[f"corr:{prem_kind}"])})
    o = pd.DataFrame(rows)
    o.to_csv(os.path.join(OUT, "false_claims_verify_on_evidence.csv"), index=False)
    S.to_csv(os.path.join(OUT, "false_claims_verify_on_evidence_units.csv"), index=False)
    print(o.round(3).to_string(index=False))


# ---------------------------------------------------------------- C
def word_spans(text):
    import re
    return [(m.start(), m.end(), m.group()) for m in re.finditer(r"\S+", text)]


def changed_word_idx(gold, corr):
    """indices of words in the corrupted claim that are not in the gold claim (insertions/substitutions)."""
    gw, cw = gold.split(), corr.split()
    out = set()
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, gw, cw).get_opcodes():
        if tag in ("replace", "insert"):
            out |= set(range(j1, j2))
        elif tag == "delete" and j1 < len(cw):        # deletion: blame the neighboring word
            out.add(j1)
    return out


def claim_word_attr(prem, claim):
    """|grad x input| of (contradiction - entailment) logit, summed per claim word."""
    import torch
    tok, mod = morph()
    if "cap" not in _MOD:
        cap = {}

        def hook(_, __, out):
            out.retain_grad(); cap["emb"] = out
        mod.model.encoder.register_forward_hook(hook)
        _MOD["cap"] = cap
    cap = _MOD["cap"]
    enc = tok(prem, claim, return_tensors="pt", truncation="only_first", max_length=3072, return_offsets_mapping=True)
    seq = enc.sequence_ids(0); offs = enc.pop("offset_mapping")[0].tolist()
    enc = enc.to(R.device())
    mod.zero_grad()
    lg = mod(**enc).logits[0]
    (lg[0] - lg[1]).backward()
    e = cap["emb"]
    g = (e.grad[0] * e[0]).sum(-1).abs().detach().float().cpu().numpy()
    ws = word_spans(claim); out = np.zeros(len(ws))
    for t, (sid, (a, b)) in enumerate(zip(seq, offs)):
        if sid != 1 or b <= a:
            continue
        for j, (s0, s1, _) in enumerate(ws):
            if s0 <= a < s1:
                out[j] += g[t]; break
    return out


def unsupported_words(evidence, claim):
    """deterministic: 1 for claim words whose normalized form is absent from the evidence (stopwords -> 0)."""
    ev = set(E.toks(evidence))
    return np.array([float(bool(E.toks(w)) and E.hnorm(w) not in ev and not (set(E.toks(w)) <= ev))
                     for _, _, w in word_spans(claim)])


def run_C():
    P = labeled_pairs(); prem = premises(); cache = jl(CC); t0 = time.time()
    with open(CC, "a", encoding="utf-8") as fh:
        for i, r in enumerate(P.itertuples()):
            sents = T.split_sents(prem[r.doc_id])
            gold_ev = " ".join(sents[int(x)] for x in sorted(int(y) for y in str(r.gold).split()))
            ret_ev, _ = evidence_text(r.corr_sent, sents, 3)
            for kind, ev in (("oracle", gold_ev), ("retrieved", ret_ev), ("article", prem[r.doc_id])):
                k = h(kind, r.doc_id, r.corr_sent)
                if k in cache:
                    continue
                rec = {"k": k, "attr": [float(x) for x in claim_word_attr(ev, r.corr_sent)],
                       "unsup": [float(x) for x in unsupported_words(ev, r.corr_sent)]}
                fh.write(json.dumps(rec) + "\n"); fh.flush(); cache[k] = rec
            if (i + 1) % 20 == 0 or i + 1 == len(P):
                print(f"  C {i + 1}/{len(P)}  {time.time() - t0:.0f}s", flush=True)
    rows = []
    for r in P.itertuples():
        ch = changed_word_idx(r.gold_sent, r.corr_sent)
        n = len(r.corr_sent.split())
        if not ch:
            continue
        for kind in ("oracle", "retrieved", "article"):
            c = cache[h(kind, r.doc_id, r.corr_sent)]
            a = np.asarray(c["attr"]); u = np.asarray(c["unsup"])
            rank_a = np.argsort(-a)
            # unsupported-word baseline: among flagged words pick the rarest-looking (longest) one; none flagged -> miss
            flagged = [j for j in range(n) if u[j] > 0]
            top_u = max(flagged, key=lambda j: len(r.corr_sent.split()[j])) if flagged else -1
            rows.append({"unit": r.Index, "evidence": kind, "n_words": n, "n_changed": len(ch),
                         "error_types": r.error_types,
                         "attr_hit1": int(rank_a[0] in ch), "attr_hit3": int(bool(set(rank_a[:3]) & ch)),
                         "unsup_hit1": int(top_u in ch), "unsup_any": int(bool(set(flagged) & ch)),
                         "unsup_n_flagged": len(flagged), "random_hit1": len(ch) / n})
    o = pd.DataFrame(rows)
    o.to_csv(os.path.join(OUT, "false_claims_localization.csv"), index=False)
    print(o.groupby("evidence")[["attr_hit1", "attr_hit3", "unsup_hit1", "unsup_any", "unsup_n_flagged", "random_hit1"]]
          .mean().round(2).to_string())


if __name__ == "__main__":
    {"A": run_A, "B": run_B, "C": run_C, "fig": lambda: None}[sys.argv[1]]()


def paper_figure():
    """fig_multisource.pdf (paper, one column): (a) false alarms on faithful sentences, single vs multi-source;
    (b) hit@1 of evidence finders for faithful vs corrupted claims. Wilson 95% CIs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
    ms = pd.read_csv(os.path.join(OUT, "multisource_summary.csv"))
    vb = pd.read_csv(os.path.join(OUT, "false_claims_verify_on_evidence.csv"))
    ret = pd.read_csv(os.path.join(OUT, "false_claims_retrieval.csv"))
    ci = lambda s: tuple(map(float, s.strip("[]").split(",")))
    # (a)
    bars = []
    for lab, m in (("SummaC\nfull art.", "SummaC-ZS"), ("Ours\nfull art.", "MorphEvalHeb")):
        r = ms[(ms.metric == m) & (ms.measure == "false_alarm_on_gold")].iloc[0]
        bars.append((lab, [(r[f"{g}_rate"], ci(r[f"{g}_ci"]), int(r[f"{g}_n"])) for g in ("single", "multi")]))
    q = vb[vb.premise == "top3"].set_index("group")
    bars.append(("Ours\ntop-3 sent.", [(q.at[g, "false_alarm"], ci(q.at[g, "fa_ci"]), int(q.at[g, "n"]))
                                                   for g in ("single", "multi")]))
    fig, ax = plt.subplots(1, 2, figsize=(3.3, 1.9), gridspec_kw={"width_ratios": [3, 2]})
    cols = ["#5b8cc8", "#e08a2e"]; w = .38; x = np.arange(len(bars))
    for j, g in enumerate(("single-source", "multi-source")):
        v = np.array([b[1][j][0] for b in bars]); lo = np.array([b[1][j][1][0] for b in bars])
        hi = np.array([b[1][j][1][1] for b in bars])
        ax[0].bar(x + (j - .5) * w, v, w, yerr=[v - lo, hi - v], capsize=1.5, color=cols[j], label=g,
                  error_kw={"lw": .6})
        for xx, vv, hh in zip(x + (j - .5) * w, v, hi):
            ax[0].text(xx, hh + .02, f"{vv * 100:.0f}", ha="center", fontsize=5.5)
    ax[0].set_xticks(x, [b[0] for b in bars], fontsize=5.5); ax[0].set_ylim(0, 1.25)
    ax[0].set_ylabel("false-alarm rate"); ax[0].set_yticks([0, .25, .5, .75, 1]); ax[0].legend(fontsize=5.5, frameon=False, loc="upper center", ncol=2, handlelength=1, columnspacing=.8)
    ax[0].set_title("(a) verification", fontsize=7)
    # (b)
    fins = [("NLI", "nli"), ("rerank\n+lexical", "rerank+lex")]
    cols2 = ["#9a9a9a", "#c0392b"]; x = np.arange(len(fins))
    for j, c in enumerate(("correct", "false")):
        vals = [ret[(ret.finder == f) & (ret.claim == c)].hit1 for _, f in fins]
        v = np.array([s.mean() for s in vals])
        lo = np.array([MS.wilson(s.sum(), len(s))[0] for s in vals]); hi = np.array([MS.wilson(s.sum(), len(s))[1] for s in vals])
        ax[1].bar(x + (j - .5) * w, v, w, yerr=[v - lo, hi - v], capsize=1.5, color=cols2[j],
                  label={"correct": "faithful", "false": "corrupted"}[c], error_kw={"lw": .6})
        for xx, vv, hh in zip(x + (j - .5) * w, v, hi):
            ax[1].text(xx, hh + .02, f"{vv * 100:.0f}", ha="center", fontsize=5.5)
    ax[1].set_xticks(x, [f[0] for f in fins], fontsize=5.5); ax[1].set_ylim(0, 1.25)
    ax[1].set_ylabel("source found at rank 1"); ax[1].set_yticks([0, .25, .5, .75, 1]); ax[1].legend(fontsize=5.5, frameon=False, loc="upper center", ncol=2, handlelength=1, columnspacing=.6)
    ax[1].set_title("(b) evidence finding", fontsize=7)
    plt.tight_layout(pad=.3, w_pad=.6)
    tex = os.path.join(R.REPO, "paper", "overleaf", "latex", "fig_multisource.pdf")
    plt.savefig(tex); plt.savefig(os.path.join(OUT, "figures", "fig_multisource.png"), dpi=200)
    print("saved", tex)


if __name__ == "__main__" and sys.argv[1] == "fig":
    paper_figure()
