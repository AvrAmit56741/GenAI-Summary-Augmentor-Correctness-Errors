"""Translation erasure (lead 2) and FactCG translate-test (task 1.11) on MAFAT-Aug.

Question: if a Hebrew summary is machine-translated to English, do the injected (often morphological)
errors survive? And does translating Hebrew -> English let an English checker (FactCG) catch them?

Unit of analysis for erasure = edited sentence. For each MAFAT-Aug negative, the gold summary of the
same article is in the benchmark. We align gold and corrupted summaries sentence by sentence; every
sentence pair that differs in Hebrew is an "edited sentence". It is *erased* if the two English
translations are identical after normalization, and *near-erased* if their chrF >= 95.

    python exp_translation.py smoke        # load the translator, translate 16 sentences, print timing
    python exp_translation.py summaries    # translate every summary sentence in MAFAT-Aug (cached, resumable)
    python exp_translation.py articles     # translate the 303 MAFAT-Aug articles (cached, resumable)
    python exp_translation.py erasure      # -> translation_erasure.csv (per edited sentence) + printed summary
    python exp_translation.py translate_test  # FactCG on English article/summary -> cache/factcg_en_<model>.csv

Translator: google/translategemma-12b-it (HF transformers, bf16, MPS), greedy decoding (deterministic).
Set MT_MODEL to swap the translator; each model gets its own cache file.
Environment: ~/.cache/morphevalheb_env.
"""
import difflib
import hashlib
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eval as R  # noqa: E402

MT_MODEL = os.environ.get("MT_MODEL", "google/translategemma-12b-it")
TAG = MT_MODEL.split("/")[-1]
TCACHE = os.path.join(R.CACHE, f"translations_{TAG}.jsonl")
BS = int(os.environ.get("MT_BS", "8"))
MAX_NEW = 256
MAX_CHARS = 400   # longer "sentences" are split at a comma/space so the translation fits in MAX_NEW tokens


# ---------------------------------------------------------------- data

def split_sents(text):
    """Hebrew sentence split on ., !, ? or newline followed by whitespace. Keeps punctuation."""
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", str(text).strip()) if p.strip()]
    out = []
    for p in parts:                       # unpunctuated runs (up to 6k chars) would exceed MAX_NEW
        while len(p) > MAX_CHARS:
            cut = p.rfind(",", 0, MAX_CHARS)
            cut = cut + 1 if cut > MAX_CHARS // 2 else (p.rfind(" ", 0, MAX_CHARS) if p.rfind(" ", 0, MAX_CHARS) > 0 else MAX_CHARS)
            out.append(p[:cut].strip()); p = p[cut:].strip()
        if p:
            out.append(p)
    return out


def mafat():
    d = R.load_bench()
    m = d[d.part == "MAFAT-Aug"].copy()
    gold = m[(m.y == 1) & m.method.isna()].drop_duplicates("doc_id").set_index("doc_id").hypothesis
    return m, gold


def key(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def load_tcache():
    out = {}
    if os.path.exists(TCACHE):
        with open(TCACHE, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                out[r["k"]] = r["en"]
    return out


# ---------------------------------------------------------------- translator

class Translator:
    def __init__(self):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        self.dev = R.device()
        self.proc = AutoProcessor.from_pretrained(MT_MODEL)
        self.tok = self.proc.tokenizer
        self.tok.padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(MT_MODEL, torch_dtype=torch.bfloat16).to(self.dev).eval()

    def prompt(self, he):
        msg = [{"role": "user", "content": [{"type": "text", "source_lang_code": "he",
                                              "target_lang_code": "en", "text": he}]}]
        return self.proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)

    def n_tokens(self, s):
        return len(self.tok(self.prompt(s), add_special_tokens=False)["input_ids"])

    def __call__(self, sents):
        enc = self.tok([self.prompt(s) for s in sents], return_tensors="pt", padding=True,
                       add_special_tokens=False).to(self.dev)
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False)
        gen = out[:, enc["input_ids"].shape[1]:]
        return [t.strip() for t in self.tok.batch_decode(gen, skip_special_tokens=True)]


def translate_all(sents, label):
    cache = load_tcache()
    todo = sorted({s for s in sents if key(s) not in cache}, key=len)   # length-sorted batches pad less
    print(f"{label}: {len(set(sents))} unique sentences, {len(todo)} to translate with {TAG}", flush=True)
    if not todo:
        return
    tr = Translator()
    t0, done = time.time(), 0
    with open(TCACHE, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), BS):
            batch = todo[i:i + BS]
            for he, en in zip(batch, tr(batch)):
                f.write(json.dumps({"k": key(he), "he": he, "en": en}, ensure_ascii=False) + "\n")
            f.flush()
            done += len(batch)
            if (i // BS) % 10 == 0:
                el = time.time() - t0
                print(f"  {done}/{len(todo)}  {el/done:.2f}s/sent  eta {el/done*(len(todo)-done)/60:.0f} min", flush=True)


# ---------------------------------------------------------------- steps

def smoke(n=5):
    """Translate n sentences, first one at a time, then as one batch; report speed and GPU memory."""
    import torch
    m, gold = mafat()
    sents = (split_sents(gold.iloc[0]) + split_sents(m[m.y == 0].hypothesis.iloc[0]))[:n]
    gb = lambda b: f"{b/1e9:.1f} GB"
    t0 = time.time()
    tr = Translator()
    print(f"load {time.time()-t0:.0f}s on {tr.dev}; MPS allocated {gb(torch.mps.current_allocated_memory())}, "
          f"driver {gb(torch.mps.driver_allocated_memory())}", flush=True)
    t0, en1 = time.time(), []
    for s in sents:
        en1 += tr([s])
    t_seq = time.time() - t0
    t0 = time.time()
    enb = tr(sents)
    t_bat = time.time() - t0
    n_tok = sum(len(tr.tok(e)["input_ids"]) for e in enb)
    print(f"sequential: {t_seq/n:.2f}s/sent | batch of {n}: {t_bat/n:.2f}s/sent ({n_tok/t_bat:.1f} output tok/s)")
    print(f"peak MPS driver memory {gb(torch.mps.driver_allocated_memory())}")
    print(f"batch output identical to sequential: {en1 == enb}")
    S, A = 8598, 19110   # unique summary / article sentences in MAFAT-Aug
    per = t_bat / n
    print(f"estimate at {per:.2f}s/sent: summaries {S*per/3600:.1f} h, articles {A*per/3600:.1f} h (batching more will be faster)")
    for he, e in zip(sents, enb):
        print("HE:", he[:150]); print("EN:", e[:200]); print()


def summaries():
    m, _ = mafat()
    translate_all([s for h in m.hypothesis for s in split_sents(h)], "summaries")


def articles():
    m, _ = mafat()
    translate_all([s for p in m.premise.unique() for s in split_sents(p)], "articles")


def norm(s):
    return re.sub(r"[^\w\s]", "", s.lower()).split()


def erasure():
    import sacrebleu
    m, gold = mafat()
    tc = load_tcache()
    rows = []
    for r in m[m.y == 0].itertuples():
        gs, cs = split_sents(gold[r.doc_id]), split_sents(r.hypothesis)
        sm = difflib.SequenceMatcher(None, gs, cs, autojunk=False)
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op != "replace" or (i2 - i1) != (j2 - j1):
                continue                      # only 1:1 sentence substitutions are comparable
            for g, c in zip(gs[i1:i2], cs[j1:j2]):
                if key(g) not in tc or key(c) not in tc:
                    continue
                ge, ce = tc[key(g)], tc[key(c)]
                rows.append({"uid": r.uid, "doc_id": r.doc_id, "error_types": r.error_types,
                             "n_families": str(r.error_types).count(",") + 1,
                             "he_char_sim": difflib.SequenceMatcher(None, g, c).ratio(),
                             "en_chrf": sacrebleu.sentence_chrf(ce, [ge]).score,
                             "erased": norm(ge) == norm(ce), "gold_he": g, "corr_he": c,
                             "gold_en": ge, "corr_en": ce})
    e = pd.DataFrame(rows)
    e["near_erased"] = e.erased | (e.en_chrf >= 95)
    out = os.path.join(R.HERE, f"translation_erasure_{TAG}.csv")
    e.to_csv(out, index=False)
    print(f"{len(e)} edited sentences from {e.uid.nunique()} negatives -> {out}")
    print(f"erased (identical EN): {e.erased.mean():.1%}   near-erased (chrF>=95): {e.near_erased.mean():.1%}")
    small = e[e.he_char_sim >= 0.9]
    print(f"minimal Hebrew edits (char-sim >= 0.9, n={len(small)}): erased {small.erased.mean():.1%}, "
          f"near {small.near_erased.mean():.1%}")
    fam = e.assign(f=e.error_types.astype(str).str.split(",")).explode("f")
    fam["f"] = fam.f.str.strip()
    t = fam.groupby("f").agg(n=("erased", "size"), erased=("erased", "mean"), near=("near_erased", "mean"))
    print("\nper family (a sentence counts toward every family injected in its summary):")
    print(t.sort_values("near", ascending=False).round(3).to_string())


def translate_test():
    from summary_augmentation.pages.metrics.factcg import FactCGScorer
    m, _ = mafat()
    tc = load_tcache()

    def en(text):
        ss = split_sents(text)
        miss = [s for s in ss if key(s) not in tc]
        return None if miss else " ".join(tc[key(s)] for s in ss)

    done, path = R.cached(f"factcg_en_{TAG}")
    todo = m[~m.uid.isin(done.uid)]
    scorer = FactCGScorer(device=R.device())
    skipped = 0
    for k, r in enumerate(todo.itertuples()):
        a, h = en(r.premise), en(r.hypothesis)
        if a is None or h is None:
            skipped += 1
            continue
        R.append(path, [{"uid": r.uid, f"FactCG-EN ({TAG})": scorer.score(a, h).score}])
        if k % 200 == 0:
            print(f"  factcg-en {k}/{len(todo)}", flush=True)
    print(f"done; skipped {skipped} pairs with untranslated sentences")


def edited_pairs(gold_text, corr_text):
    """1:1 sentence substitutions between gold and corrupted summary (the only sentences we translate)."""
    gs, cs = split_sents(gold_text), split_sents(corr_text)
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, gs, cs, autojunk=False).get_opcodes():
        if op == "replace" and (i2 - i1) == (j2 - j1):
            out += list(zip(gs[i1:i2], cs[j1:j2]))
    return out


def translate_bucketed(tr, sents, bs=16):
    """Translate with batches of equal prompt-token length only: no padding, so output == one at a time."""
    import collections
    by_len = collections.defaultdict(list)
    for s in sents:
        by_len[tr.n_tokens(s)].append(s)
    res = {}
    for L in sorted(by_len):
        group = by_len[L]
        for i in range(0, len(group), bs):
            b = group[i:i + bs]
            res.update(zip(b, tr(b)))
    return res, len(by_len)


HEB = re.compile(r"[֐-׿]")
NEG = {"לא", "אין", "אינו", "אינה", "אינם", "אינן", "בלי", "ללא"}   # not "אף" (also "even")
PREFIX = "ובלכמשה"


def unfinal(w):
    """Map Hebrew final letters to their base form (טוען vs טוענת share the stem טוענ)."""
    return w.translate(str.maketrans("ךםןףץ", "כמנפצ"))


def is_definite_pair(x, y):
    """y = x with the definite article: 'ה' + x, or one prefix (ו/ב/ל/כ/מ/ש) kept and 'ה' after it.
    Requires a stem of >= 2 letters, so short function words (שלו/שלה) never match."""
    x, y = re.sub(r"[֑-ׇ]", "", x), re.sub(r"[֑-ׇ]", "", y)   # drop niqqud
    for p in ["", *PREFIX[:-1]]:
        if x.startswith(p) and len(x) - len(p) >= 2:
            stem = x[len(p):]
            if y == p + "ה" + stem:
                return True
    return False


def edit_kind(g, c):
    """Rule-based label of what changed between gold and corrupted Hebrew sentence (a pre-label for manual check)."""
    gw, cw = g.split(), c.split()
    kinds = set()
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, gw, cw, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        a, b = gw[i1:i2], cw[j1:j2]
        if (NEG & set(a)) != (NEG & set(b)):
            kinds.add("negation")
        if re.findall(r"\d+", " ".join(a)) != re.findall(r"\d+", " ".join(b)):
            kinds.add("number")
        if op == "replace" and len(a) == len(b):
            for x, y in zip(a, b):
                x, y = x.strip(".,\"'"), y.strip(".,\"'")
                if is_definite_pair(x, y) or is_definite_pair(y, x):
                    kinds.add("definiteness")
                elif x != y and unfinal(re.sub(r"(ים|ות|ה|ת)$", "", x)) == unfinal(re.sub(r"(ים|ות|ה|ת)$", "", y)):
                    kinds.add("agreement_suffix")
    return "+".join(sorted(kinds)) or "other"


def qc(he, en, tok):
    """Problems with one translation, or '' if it looks fine."""
    if not en.strip():
        return "empty"
    if len(tok(en)["input_ids"]) >= MAX_NEW - 2:
        return "truncated"
    if len(HEB.findall(en)) > 0.2 * max(1, len(re.sub(r"\s", "", en))):
        return "untranslated_hebrew"
    r = len(en) / max(1, len(he))
    if r < 0.4 or r > 4:
        return f"length_ratio_{r:.1f}"
    return ""


def erasure_stream(chunk=20, max_bad=0.05):
    """Chunks of `chunk` negatives: translate only their edited sentences (cache-first), analyze, log, continue."""
    import sacrebleu
    m, gold = mafat()
    neg = m[m.y == 0].sample(frac=1, random_state=7)          # random order: running stats are representative
    out = os.path.join(R.HERE, f"translation_erasure_{TAG}.csv")
    done = set(pd.read_csv(out).uid) if os.path.exists(out) else set()
    neg = neg[~neg.uid.isin(done)]
    log = open(os.path.join(R.HERE, f"translation_erasure_{TAG}.log"), "a", encoding="utf-8")

    def say(msg):
        print(msg, flush=True); log.write(msg + "\n"); log.flush()

    say(f"=== start {time.strftime('%F %T')}  model={TAG}  remaining negatives={len(neg)}  already done={len(done)}")
    tr, tc = Translator(), load_tcache()
    tot = {"needed": 0, "cached": 0, "new": 0, "skipped_unchanged": 0}
    for c0 in range(0, len(neg), chunk):
        part = neg.iloc[c0:c0 + chunk]
        pairs = {r.uid: edited_pairs(gold[r.doc_id], r.hypothesis) for r in part.itertuples()}
        n_all = sum(len(split_sents(r.hypothesis)) for r in part.itertuples())
        need = {s for ps in pairs.values() for p in ps for s in p}
        new = sorted(s for s in need if key(s) not in tc)
        t0 = time.time()
        res, n_buckets = translate_bucketed(tr, new) if new else ({}, 0)
        dt = time.time() - t0
        bad = {}
        with open(TCACHE, "a", encoding="utf-8") as f:
            for he, en in res.items():
                tc[key(he)] = en
                f.write(json.dumps({"k": key(he), "he": he, "en": en}, ensure_ascii=False) + "\n")
                p = qc(he, en, tr.tok)
                if p:
                    bad[he] = p
        rows = []
        for r in part.itertuples():
            for g, c in pairs[r.uid]:
                ge, ce = tc[key(g)], tc[key(c)]
                rows.append({"uid": r.uid, "doc_id": r.doc_id, "error_types": r.error_types,
                             "he_char_sim": difflib.SequenceMatcher(None, g, c).ratio(),
                             "en_chrf": sacrebleu.sentence_chrf(ce, [ge]).score,
                             "erased": norm(ge) == norm(ce), "edit_kind": edit_kind(g, c),
                             "qc": bad.get(g, "") or bad.get(c, ""),
                             "gold_he": g, "corr_he": c, "gold_en": ge, "corr_en": ce})
        e = pd.DataFrame(rows)
        if not e.empty:
            e.to_csv(out, mode="a", header=not os.path.exists(out), index=False)
        tot["needed"] += len(need); tot["cached"] += len(need) - len(new); tot["new"] += len(new)
        tot["skipped_unchanged"] += n_all - sum(len(ps) for ps in pairs.values())
        allr = pd.read_csv(out)
        k = c0 // chunk + 1
        say(f"[chunk {k}] negatives {c0+1}-{c0+len(part)} | edited sentence pairs {len(e)} | sentences needed "
            f"{len(need)}: cached {len(need)-len(new)}, translated {len(new)} in {dt:.0f}s "
            f"({dt/max(1,len(new)):.2f}s/sent, {n_buckets} length buckets) | QC problems {len(bad)}")
        if not e.empty:
            p = allr.erased.mean()
            say(f"   this chunk: erased {e.erased.mean():.1%} ({e.erased.sum()}/{len(e)}) | cumulative "
                f"({allr.uid.nunique()} negatives, {len(allr)} edited sentences): erased {p:.1%} "
                f"(95% CI ±{1.96*np.sqrt(p*(1-p)/len(allr)):.1%})")
            ek = allr.edit_kind.str.split("+").explode()
            er = allr[allr.erased].edit_kind.str.split("+").explode()
            say("   cumulative erasure by edit kind (erased/total): " + ", ".join(
                f"{k} {int((er == k).sum())}/{int((ek == k).sum())}" for k in
                ["negation", "number", "definiteness", "agreement_suffix", "other"]))
            for _, x in e[e.erased].head(2).iterrows():
                say(f"   example erased [{x.edit_kind}]: HE gold «{x.gold_he[:90]}» / corr «{x.corr_he[:90]}» -> EN «{x.corr_en[:110]}»")
        for he, p in list(bad.items())[:3]:
            say(f"   QC {p}: «{he[:80]}» -> «{tc[key(he)][:100]}»")
        if new and len(bad) / len(new) > max_bad:
            say(f"STOP: {len(bad)}/{len(new)} translations failed QC (> {max_bad:.0%}). Inspect before continuing.")
            break
        if os.environ.get("MAX_CHUNKS") and k >= int(os.environ["MAX_CHUNKS"]):
            say(f"stopping after {k} chunks (MAX_CHUNKS)")
            break
    say(f"=== totals: sentences needed {tot['needed']}, from cache {tot['cached']}, newly translated {tot['new']}, "
        f"unchanged sentences never translated {tot['skipped_unchanged']}")


def translate_test_stream(chunk=20, n_docs=100, max_bad=0.05):
    """Task 1.11, in chunks of `chunk` articles: translate article + all its summaries (cache-first),
    score FactCG on the English pair, log running AUC vs Hebrew FactCG and MorphEvalHeb on the same pairs."""
    from sklearn.metrics import roc_auc_score
    from summary_augmentation.pages.metrics.factcg import FactCGScorer
    m, _ = mafat()
    docs = m.groupby("doc_id").y.agg(["min", "max"])
    docs = np.random.default_rng(7).permutation(docs[(docs["min"] == 0) & (docs["max"] == 1)].index)[:n_docs]
    col = f"FactCG-EN ({TAG})"
    done, path = R.cached(f"factcg_en_{TAG}")
    sc = pd.read_csv(os.path.join(R.HERE, "scores.csv")).set_index("uid")
    log = open(os.path.join(R.HERE, f"translate_test_{TAG}.log"), "a", encoding="utf-8")

    def say(msg):
        print(msg, flush=True); log.write(msg + "\n"); log.flush()

    say(f"=== start {time.strftime('%F %T')}  model={TAG}  articles={len(docs)}  pairs already scored={len(done)}")
    tr, tc, fc = Translator(), load_tcache(), FactCGScorer(device=R.device())
    rng = np.random.default_rng(0)
    for c0 in range(0, len(docs), chunk):
        part = m[m.doc_id.isin(docs[c0:c0 + chunk]) & ~m.uid.isin(done.uid)]
        if part.empty:
            continue
        need = {s for t in list(part.premise.unique()) + list(part.hypothesis) for s in split_sents(t)}
        new = sorted(s for s in need if key(s) not in tc)
        t0 = time.time()
        res, nb = translate_bucketed(tr, new) if new else ({}, 0)
        dt = time.time() - t0
        bad = {}
        with open(TCACHE, "a", encoding="utf-8") as f:
            for he, en in res.items():
                tc[key(he)] = en
                f.write(json.dumps({"k": key(he), "he": he, "en": en}, ensure_ascii=False) + "\n")
                p = qc(he, en, tr.tok)
                if p:
                    bad[he] = p
        en = lambda t: " ".join(tc[key(s)] for s in split_sents(t))
        t1 = time.time()
        rows = [{"uid": r.uid, col: fc.score(en(r.premise), en(r.hypothesis)).score} for r in part.itertuples()]
        R.append(path, rows)
        done = pd.read_csv(path)
        k = c0 // chunk + 1
        say(f"[chunk {k}] articles {c0+1}-{c0+chunk} | pairs {len(part)} | sentences needed {len(need)}: cached "
            f"{len(need)-len(new)}, translated {len(new)} in {dt:.0f}s ({dt/max(1,len(new)):.2f}s/sent, {nb} buckets) | "
            f"FactCG-EN {time.time()-t1:.0f}s | QC problems {len(bad)}")
        a = done.set_index("uid").join(sc[["y", "FactCG", "MorphEvalHeb"]])
        out = []
        for name in [col, "FactCG", "MorphEvalHeb"]:
            auc = roc_auc_score(a.y, a[name])
            bs = []
            for _ in range(500):
                i = rng.integers(0, len(a), len(a))
                if a.y.iloc[i].nunique() == 2:
                    bs.append(roc_auc_score(a.y.iloc[i], a[name].iloc[i]))
            out.append(f"{name} {auc:.3f} [{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}]")
        say(f"   cumulative AUC on {len(a)} pairs ({int(a.y.sum())} faithful): " + " | ".join(out))
        for he, p in list(bad.items())[:3]:
            say(f"   QC {p}: «{he[:80]}» -> «{tc[key(he)][:100]}»")
        if new and len(bad) / len(new) > max_bad:
            say(f"STOP: {len(bad)}/{len(new)} translations failed QC (> {max_bad:.0%}). Inspect before continuing.")
            break
    say(f"=== done {time.strftime('%F %T')}")


def check_batching(n=16):
    """Does batching equal-length (unpadded) sentences give the same output as one at a time?"""
    import collections
    m, _ = mafat()
    tr = Translator()
    pool = sorted({s for h in m.hypothesis for s in split_sents(h)})[:3000]
    by_len = collections.defaultdict(list)
    for s in pool:
        by_len[tr.n_tokens(s)].append(s)
    L = min((k for k, v in by_len.items() if len(v) >= n), key=lambda k: abs(k - 60))   # typical length
    sents = by_len[L][:n]
    print(f"{n} sentences, all {L} prompt tokens (no padding needed)", flush=True)
    t0 = time.time(); seq = [tr([s])[0] for s in sents]; t_seq = time.time() - t0
    t0 = time.time(); bat = tr(sents); t_bat = time.time() - t0
    same = sum(a == b for a, b in zip(seq, bat))
    print(f"one at a time: {t_seq/n:.2f}s/sent | batch of {n}: {t_bat/n:.2f}s/sent  (speed-up x{t_seq/t_bat:.1f})")
    print(f"identical outputs: {same}/{n}")
    for a, b in zip(seq, bat):
        if a != b:
            print("  SEQ:", a[:160]); print("  BAT:", b[:160])


if __name__ == "__main__":
    {"smoke": smoke, "check_batching": check_batching, "erasure_stream": erasure_stream,
     "translate_test_stream": translate_test_stream, "summaries": summaries, "articles": articles,
     "erasure": erasure, "translate_test": translate_test}[sys.argv[1]]()
