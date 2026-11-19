"""Lead 17: evidence finding for summary claims (single- vs multi-sentence) and claim-level verification.

For each summary sentence ("claim") we score every article sentence with several evidence finders:
  lex      deterministic: morphology-normalized (prefix/final-letter) content-word recall, IDF-weighted
  bm25     deterministic: BM25 over the same normalized tokens
  e5       dense: intfloat/multilingual-e5-small (local cache)
  bge      dense: BAAI/bge-m3 (if downloaded)
  rerank   cross-encoder: BAAI/bge-reranker-v2-m3 (if downloaded)
  nli      model: mDeBERTa-v3-base-mnli-xnli P(entailment | article sentence -> claim)   (SummaC-style)
  attr     ours: MorphEvalHeb verifier read (whole article, claim); |gradient x input| mass per article sentence
  fuse     reciprocal-rank fusion of the above
Multi-sentence evidence: greedy coverage of the claim's normalized content words over the ranked sentences.

    python exp_evidence.py gold_candidates N   # build a labeling sheet: N claims x union of top candidates
    python exp_evidence.py score_finders       # recall@k / MRR of each finder vs the hand-labeled gold
Everything is cached in cache/evidence_*.jsonl / .csv
"""
import json
import math
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eval as R  # noqa: E402
import exp_translation as T  # noqa: E402

PREFIXES = "ובהכלמש"
STOP = set("של את על עם כי גם או אם לא אך כל זה זו זאת הוא היא הם הן אשר בין לפי עד יותר כמו אל מן".split())
GOLD_SHEET = os.path.join(R.HERE, "evidence_gold_sheet.csv")
GOLD_LABELS = os.path.join(R.HERE, "evidence_gold_labels.csv")


# ---------------------------------------------------------------- deterministic, morphology-aware

def hnorm(w):
    w = re.sub(r"[֑-ׇ\"'׳״]", "", w)
    w = T.unfinal(w)
    k = 0
    while k < 2 and len(w) > 3 and w[0] in PREFIXES:   # strip up to two proclitics: ו, ה, ב, כ, ל, מ, ש
        w, k = w[1:], k + 1
    return w


def toks(s):
    return [hnorm(w) for w in re.findall(r"[֐-׿A-Za-z0-9]+", s) if w not in STOP and len(w) > 1]


def lex_scores(claim, sents):
    df = Counter(w for s in sents for w in set(toks(s)))
    idf = {w: math.log((len(sents) + 1) / (c + 0.5)) for w, c in df.items()}
    cw = set(toks(claim))
    tot = sum(idf.get(w, math.log(len(sents) + 1)) for w in cw) or 1
    return np.array([sum(idf.get(w, 0) for w in cw & set(toks(s))) / tot for s in sents])


def bm25_scores(claim, sents, k1=1.5, b=0.75):
    docs = [toks(s) for s in sents]
    avg = np.mean([len(d) for d in docs]) or 1
    df = Counter(w for d in docs for w in set(d))
    N = len(docs)
    out = []
    for d in docs:
        tf = Counter(d)
        sc = 0.0
        for w in set(toks(claim)):
            if w in tf:
                idf = math.log(1 + (N - df[w] + 0.5) / (df[w] + 0.5))
                sc += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(d) / avg))
        out.append(sc)
    return np.array(out)


# ---------------------------------------------------------------- models (lazy)

_M = {}


def _dev():
    return R.device()


def dense_scores(claim, sents, name="e5"):
    import torch
    from transformers import AutoModel, AutoTokenizer
    mid = {"e5": "intfloat/multilingual-e5-small", "bge": "BAAI/bge-m3"}[name]
    if name not in _M:
        _M[name] = (AutoTokenizer.from_pretrained(mid), AutoModel.from_pretrained(mid).to(_dev()).eval())
    tok, mod = _M[name]
    q = ("query: " + claim) if name == "e5" else claim
    p = [("passage: " + s) if name == "e5" else s for s in sents]
    with torch.no_grad():
        enc = tok([q] + p, padding=True, truncation=True, max_length=256, return_tensors="pt").to(_dev())
        h = mod(**enc).last_hidden_state
        if name == "e5":
            m = enc["attention_mask"].unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1)
        else:
            v = h[:, 0]
        v = torch.nn.functional.normalize(v, dim=-1)
    return (v[1:] @ v[0]).float().cpu().numpy()


def rerank_scores(claim, sents):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    if "rerank" not in _M:
        mid = "BAAI/bge-reranker-v2-m3"
        _M["rerank"] = (AutoTokenizer.from_pretrained(mid),
                        AutoModelForSequenceClassification.from_pretrained(mid).to(_dev()).eval())
    tok, mod = _M["rerank"]
    with torch.no_grad():
        enc = tok([claim] * len(sents), sents, padding=True, truncation=True, max_length=384, return_tensors="pt").to(_dev())
        return mod(**enc).logits.view(-1).float().cpu().numpy()


def nli_scores(claim, sents):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    if "nli" not in _M:
        mid = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
        _M["nli"] = (AutoTokenizer.from_pretrained(mid),
                     AutoModelForSequenceClassification.from_pretrained(mid).to(_dev()).eval())
    tok, mod = _M["nli"]
    ent = mod.config.label2id.get("entailment", 0)
    with torch.no_grad():
        enc = tok(sents, [claim] * len(sents), padding=True, truncation=True, max_length=384, return_tensors="pt").to(_dev())
        return torch.softmax(mod(**enc).logits.float(), -1)[:, ent].cpu().numpy()


def attr_scores(claim, article, sents):
    """MorphEvalHeb reads (whole article, claim); |grad x input| of the entailment logit, summed per article sentence
    and divided by sqrt(length) so long sentences are not favored just for having more tokens."""
    import torch
    if "attr" not in _M:
        tok, model = R.load_cls("NLI-hebrew-binary-correctness-metric")
        cap = {}

        def hook(_, __, out):
            out.retain_grad(); cap["emb"] = out
        model.model.encoder.register_forward_hook(hook)
        _M["attr"] = (tok, model, cap)
    tok, model, cap = _M["attr"]
    enc = tok(article, claim, return_tensors="pt", truncation="only_first", max_length=3072, return_offsets_mapping=True)
    seq = enc.sequence_ids(0)
    offs = enc.pop("offset_mapping")[0].tolist()
    enc = enc.to(_dev())
    model.zero_grad()
    logits = model(**enc).logits[0]
    logits[1].backward()                                   # entailment logit
    e = cap["emb"]
    g = (e.grad[0] * e[0]).sum(-1).abs().detach().float().cpu().numpy()
    # map article sentences to char spans
    spans, pos = [], 0
    for s in sents:
        i = article.find(s[:30], pos)
        i = pos if i < 0 else i
        spans.append((i, i + len(s))); pos = i + len(s)
    out = np.zeros(len(sents))
    cnt = np.zeros(len(sents))
    for t, (sid, (a, b)) in enumerate(zip(seq, offs)):
        if sid != 0 or b <= a:
            continue
        for j, (s0, s1) in enumerate(spans):
            if s0 <= a < s1:
                out[j] += g[t]; cnt[j] += 1; break
    return out / np.sqrt(np.maximum(cnt, 1)), int(cnt.sum() < len(T.split_sents(article)) * 3)


def rrf(rank_lists, k=60):
    tot = np.zeros(len(rank_lists[0]))
    for sc in rank_lists:
        order = np.argsort(-sc)
        rank = np.empty_like(order); rank[order] = np.arange(len(sc))
        tot += 1 / (k + rank + 1)
    return tot


def greedy_cover(claim, sents, order, max_k=3, min_gain=0.1):
    cw = set(toks(claim)); cov, pick = set(), []
    for i in order[:10]:
        gain = len((cw & set(toks(sents[i]))) - cov) / max(1, len(cw))
        if gain >= min_gain or not pick:
            pick.append(int(i)); cov |= cw & set(toks(sents[i]))
        if len(pick) >= max_k:
            break
    return pick, len(cov) / max(1, len(cw))


FINDERS = ["lex", "bm25", "e5", "bge", "rerank", "nli", "attr"]


def all_scores(claim, article, finders=FINDERS):
    sents = T.split_sents(article)
    out = {}
    for f in finders:
        if f == "lex": out[f] = lex_scores(claim, sents)
        elif f == "bm25": out[f] = bm25_scores(claim, sents)
        elif f in ("e5", "bge"): out[f] = dense_scores(claim, sents, f)
        elif f == "rerank": out[f] = rerank_scores(claim, sents)
        elif f == "nli": out[f] = nli_scores(claim, sents)
        elif f == "attr": out[f] = attr_scores(claim, article, sents)[0]
    out["fuse"] = rrf([out[f] for f in finders])
    return sents, out


# ---------------------------------------------------------------- gold set

def gold_candidates(n=40, seed=11):
    """Stratified sample of reference-summary sentences; candidate pool = union of each finder's top 3."""
    m, gold = T.mafat()
    rng = np.random.default_rng(seed)
    src = m.drop_duplicates("doc_id").set_index("doc_id").source
    rows = []
    docs = list(gold.index)
    rng.shuffle(docs)
    per_src = Counter()
    for doc in docs:
        if len(rows) >= n:
            break
        if per_src[src[doc]] >= max(4, n // 5 + 2):
            continue
        claims = [c for c in T.split_sents(gold[doc]) if len(toks(c)) >= 5]
        if not claims:
            continue
        claim = claims[rng.integers(len(claims))]
        art = m[m.doc_id == doc].premise.iloc[0]
        sents, sc = all_scores(claim, art)
        pool = sorted({int(i) for f in FINDERS for i in np.argsort(-sc[f])[:3]})
        rows.append({"claim_id": len(rows), "doc_id": doc, "source": src[doc], "claim": claim,
                     "candidates": json.dumps({i: sents[i] for i in pool}, ensure_ascii=False)})
        per_src[src[doc]] += 1
        print(f"  claim {len(rows)}/{n} ({src[doc]}): {len(pool)} candidates", flush=True)
    pd.DataFrame(rows).to_csv(GOLD_SHEET, index=False)
    print("saved", GOLD_SHEET)


SCORES = os.path.join(R.CACHE, "evidence_scores.jsonl")


def cached_scores():
    out = {}
    if os.path.exists(SCORES):
        for line in open(SCORES, encoding="utf-8"):
            r = json.loads(line); out[r["claim_id"]] = r
    return out


def eval_finders(score_dict, gold, multi):
    """hit@1, recall@3 (share of gold sentences in top 3), all@3 (every gold sentence in top 3), MRR."""
    res = {}
    for f, sc in score_dict.items():
        order = list(np.argsort(-np.asarray(sc)))
        top3 = set(order[:3])
        first = next((r for r, i in enumerate(order) if i in gold), len(order))
        res[f] = {"hit1": order[0] in gold, "rec3": len(top3 & gold) / len(gold),
                  "all3": gold <= top3, "mrr": 1 / (first + 1)}
    return res


def score_finders():
    """Score every finder on the hand-labeled gold set (scores cached), then report overall / single / multi."""
    m, _ = T.mafat()
    lab = pd.read_csv(GOLD_LABELS)
    lab = lab[lab.multi.notna()]
    cache = cached_scores()
    with open(SCORES, "a", encoding="utf-8") as fh:
        for r in lab.itertuples():
            if r.claim_id in cache:
                continue
            art = m[m.doc_id == r.doc_id].premise.iloc[0]
            sents, sc = all_scores(r.claim, art)
            rec = {"claim_id": int(r.claim_id), "n_sents": len(sents), **{f: [float(x) for x in v] for f, v in sc.items()}}
            fh.write(json.dumps(rec) + "\n"); fh.flush(); cache[r.claim_id] = rec
            print(f"  scored claim {r.claim_id}", flush=True)
    rows = []
    for r in lab.itertuples():
        c = cache[r.claim_id]
        sc = {f: np.array(c[f]) for f in FINDERS}
        z = {f: (v - v.mean()) / (v.std() + 1e-9) for f, v in sc.items()}
        variants = dict(sc)
        variants["fuse_all"] = rrf([sc[f] for f in FINDERS])
        variants["fuse_det"] = rrf([sc["lex"], sc["bm25"]])
        variants["fuse_models"] = rrf([sc[f] for f in ("e5", "bge", "rerank", "nli")])
        variants["fuse_no_attr"] = rrf([sc[f] for f in FINDERS if f != "attr"])
        variants["zsum_rerank_lex_attr"] = z["rerank"] + z["lex"] + z["attr"]
        gold = {int(x) for x in str(r.gold).split()}
        for f, v in eval_finders(variants, gold, r.multi).items():
            rows.append({"claim_id": r.claim_id, "multi": int(r.multi), "finder": f, **v})
    e = pd.DataFrame(rows)
    e.to_csv(os.path.join(R.HERE, "evidence_finder_eval.csv"), index=False)
    agg = lambda d: d.groupby("finder")[["hit1", "rec3", "all3", "mrr"]].mean().round(2)
    out = pd.concat({"ALL": agg(e), "single": agg(e[e.multi == 0]), "multi": agg(e[e.multi == 1])}, axis=1)
    print(out.sort_values(("ALL", "mrr"), ascending=False).to_string())


NEUTRAL = ["זהו סיכום של הכתבה.", "הכתבה עוסקת בנושא כללי."]
NEUTRAL_CACHE = os.path.join(R.CACHE, "evidence_attr_neutral.jsonl")


def contrastive_attribution():
    """Raw verifier attribution is dominated by position (first/last sentence). Remove that per-article profile by
    subtracting the attribution the verifier gives the same article for neutral claims."""
    m, _ = T.mafat()
    lab = pd.read_csv(GOLD_LABELS)
    lab = lab[lab.multi.notna()]
    sc = cached_scores()
    neu = {}
    if os.path.exists(NEUTRAL_CACHE):
        for line in open(NEUTRAL_CACHE, encoding="utf-8"):
            r = json.loads(line); neu[r["doc_id"]] = r["attr_neutral"]
    with open(NEUTRAL_CACHE, "a", encoding="utf-8") as fh:
        for r in lab.itertuples():
            if r.doc_id in neu:
                continue
            art = m[m.doc_id == r.doc_id].premise.iloc[0]
            sents = T.split_sents(art)
            prof = np.mean([attr_scores(n, art, sents)[0] for n in NEUTRAL], axis=0)
            neu[r.doc_id] = [float(x) for x in prof]
            fh.write(json.dumps({"doc_id": r.doc_id, "attr_neutral": neu[r.doc_id]}) + "\n"); fh.flush()
            print(f"  neutral profile for claim {r.claim_id}", flush=True)
    z = lambda v: (v - v.mean()) / (v.std() + 1e-9)
    rows = []
    for r in lab.itertuples():
        c = sc[r.claim_id]
        a, n = np.array(c["attr"]), np.array(neu[r.doc_id])
        variants = {
            "attr_raw": a,
            "attr_ratio": np.log(a + 1e-6) - np.log(n + 1e-6),
            "attr_zdiff": z(a) - z(n),
            "lex": np.array(c["lex"]),
            "rerank": np.array(c["rerank"]),
            "lex+attr_ratio": z(np.array(c["lex"])) + z(np.log(a + 1e-6) - np.log(n + 1e-6)),
            "rerank+lex+attr_ratio": z(np.array(c["rerank"])) + z(np.array(c["lex"])) + z(np.log(a + 1e-6) - np.log(n + 1e-6)),
            "rerank+lex": z(np.array(c["rerank"])) + z(np.array(c["lex"])),
        }
        gold = {int(x) for x in str(r.gold).split()}
        for f, v in eval_finders(variants, gold, r.multi).items():
            rows.append({"claim_id": r.claim_id, "multi": int(r.multi), "finder": f, **v})
    e = pd.DataFrame(rows)
    e.to_csv(os.path.join(R.HERE, "evidence_contrastive_eval.csv"), index=False)
    agg = lambda d: d.groupby("finder")[["hit1", "rec3", "all3", "mrr"]].mean().round(2)
    print(pd.concat({"ALL": agg(e), "single": agg(e[e.multi == 0]), "multi": agg(e[e.multi == 1])}, axis=1)
          .sort_values(("ALL", "mrr"), ascending=False).to_string())


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "contrastive":
        contrastive_attribution()
    if cmd == "gold_candidates":
        gold_candidates(int(sys.argv[2]) if len(sys.argv) > 2 else 40)
    elif cmd == "score_finders":
        score_finders()
