"""Document-level Hebrew -> English translation through Amazon Bedrock (replaces sentence-level local MT).

Whole summaries are translated in one call; articles in paragraph blocks (consecutive sentences up to
BLOCK_CHARS), so the translator keeps context. Prompt is deliberately neutral (a realistic
translate-then-check pipeline would not tell the translator to preserve errors).

    python exp_translation_api.py smoke      # 5 whole summaries + 2 article blocks, Claude Haiku 5.5 vs Nova Micro

Credentials: AWS env vars loaded from BEDROCK_CREDS_FILE (default: the urlf-classification experiments file).
"""
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eval as R  # noqa: E402
import exp_translation as T  # noqa: E402

CREDS = os.environ.get("BEDROCK_CREDS_FILE", os.path.expanduser(
    "~/PycharmProjects/urlf-classification/experiments/bedrock_classifier/bedrock_aws_creds.local.env"))
REGION = "us-east-1"
HAIKU = "us.anthropic.claude-haiku-4-5-20251001-v1:0"   # Haiku 5.5 is not enabled on this account (Marketplace 403)
NOVA = "us.amazon.nova-micro-v1:0"            # Bedrock Converse inference profile
SYSTEM = "You are a translator. Reply with the English translation only: no preamble, notes or quotation marks."
PROMPT = "Translate the following Hebrew text into English. Output only the English translation.\n\n{text}"
BLOCK_CHARS = 1000
# approximate list prices, $ per 1M tokens (input, output); Bedrock prices may differ slightly
PRICE = {HAIKU: (1.00, 5.00), NOVA: (0.035, 0.14)}


def load_creds():
    for line in open(CREDS):
        m = re.match(r"\s*(?:export\s+)?([A-Z_]+)=(.*)", line)
        if m:
            os.environ[m.group(1)] = m.group(2).strip().strip("\"'")


def haiku_client():
    from anthropic import AnthropicBedrock   # Mantle endpoint does not serve Haiku on this account
    return AnthropicBedrock(aws_region=REGION)


def translate_haiku(client, he):
    r = client.messages.create(model=HAIKU, max_tokens=8000, system=SYSTEM,
                               messages=[{"role": "user", "content": PROMPT.format(text=he)}])
    if r.stop_reason not in ("end_turn", "stop_sequence"):
        raise RuntimeError(f"stop_reason={r.stop_reason}")
    text = "".join(b.text for b in r.content if b.type == "text").strip()
    return text, r.usage.input_tokens, r.usage.output_tokens


def translate_nova(brt, he):
    r = brt.converse(modelId=NOVA, system=[{"text": SYSTEM}], messages=[{"role": "user", "content": [{"text": PROMPT.format(text=he)}]}],
                     inferenceConfig={"maxTokens": 5000, "temperature": 0})
    if r.get("stopReason") not in ("end_turn", "stop_sequence"):      # full articles: never accept a cut-off output
        raise RuntimeError(f"stopReason={r.get('stopReason')}")
    text = "".join(c.get("text", "") for c in r["output"]["message"]["content"]).strip()
    return text, r["usage"]["inputTokens"], r["usage"]["outputTokens"]


def article_blocks(text, limit=BLOCK_CHARS):
    blocks, cur = [], ""
    for s in T.split_sents(text):
        if cur and len(cur) + 1 + len(s) > limit:
            blocks.append(cur); cur = s
        else:
            cur = f"{cur} {s}".strip()
    return blocks + ([cur] if cur else [])


def smoke():
    load_creds()
    try:
        import truststore; truststore.inject_into_ssl()
    except Exception:
        pass
    import boto3
    m, gold = T.mafat()
    neg = m[m.y == 0]
    items = []
    for pat in ["היקב לא מייצר", "3918"]:
        c = neg[neg.hypothesis.str.contains(pat)].iloc[0]
        items += [("gold summary", gold[c.doc_id]), (f"corrupted summary («{pat}»)", c.hypothesis)]
    items.append(("corrupted summary («לא קיים פוזיטרון»)", neg[neg.hypothesis.str.contains("לא קיים פוזיטרון")].iloc[0].hypothesis))
    art = m[m.hypothesis.str.contains("היקב")].iloc[0].premise
    items += [(f"article block {i+1}", b) for i, b in enumerate(article_blocks(art)[:2])]

    hc, brt = haiku_client(), boto3.Session(region_name=REGION).client("bedrock-runtime")
    out, tot = [], {HAIKU: [0, 0, 0.0], NOVA: [0, 0, 0.0]}
    for label, he in items:
        row = {"label": label, "he": he}
        for name, fn, cl in [(HAIKU, translate_haiku, hc), (NOVA, translate_nova, brt)]:
            t0 = time.time()
            en, ti, to = fn(cl, he)
            row[name] = {"en": en, "in": ti, "out": to, "sec": round(time.time() - t0, 1),
                         "qc": T.qc(he, en, lambda s: {"input_ids": s.split()}) or "ok"}
            tot[name][0] += ti; tot[name][1] += to; tot[name][2] += time.time() - t0
        out.append(row)
        print(f"\n=== {label} ({len(he)} chars)\nHE: {he[:300]}")
        for name in (HAIKU, NOVA):
            r = row[name]
            print(f"--- {name}  [{r['in']} in / {r['out']} out tok, {r['sec']}s, qc={r['qc']}]\n{r['en'][:500]}")
    json.dump(out, open(os.path.join(R.CACHE, "api_translation_smoke.json"), "w"), ensure_ascii=False, indent=1)
    print("\n=== totals")
    for name, (ti, to, sec) in tot.items():
        pi, po = PRICE[name]
        print(f"{name}: {ti} in / {to} out tokens, {sec:.0f}s, ~${(ti*pi + to*po)/1e6:.4f}")


# ---------------------------------------------------------------- representative subsets

SUB1 = os.path.join(R.HERE, "subset_erasure.csv")         # test 1: 250 MAFAT corrupted summaries (+ their gold)
SUB2 = os.path.join(R.HERE, "subset_translate_test.csv")  # test 2: 250 article-summary pairs (MAFAT + mFACT)
KINDS = ["negation", "number", "definiteness", "agreement_suffix", "other"]


def build_subsets(n1=250, n2_each=125, seed=7):
    """Stratified, documented selection. Test 1 strata: source x summary-length group, then top-up so every
    error family has >= 10 summaries; reuses the sentence-level run's pairs for a paired comparison."""
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(seed)
    m, gold = T.mafat()
    sl = pd.read_csv(os.path.join(R.HERE, "translation_erasure_translategemma-12b-it.csv"))
    neg = m[(m.y == 0) & m.uid.isin(sl.uid)].copy()
    neg["n_fam"] = neg.error_types.str.count(",") + 1
    neg["char_sim"] = [__import__("difflib").SequenceMatcher(None, gold[r.doc_id], r.hypothesis, autojunk=False).ratio()
                       for r in neg.itertuples()]
    neg["edit_bin"] = pd.qcut(neg.char_sim, 3, labels=["large", "medium", "small"])
    neg["nfam_bin"] = pd.cut(neg.n_fam, [0, 3, 7, 99], labels=["1-3", "4-7", "8+"])
    # 1) proportional quota per source, with a floor so small sources (Knesset, Bagatz) are represented
    share = neg.source.value_counts(normalize=True)
    quota = (share * n1).round().astype(int).clip(lower=min(25, neg.source.value_counts().min()))
    quota = (quota * n1 / quota.sum()).round().astype(int)
    pick = []
    for src, q in quota.items():
        g = neg[neg.source == src]
        # stratified within source by length group x edit size x number of families
        strata = g.groupby(["len_group", "edit_bin", "nfam_bin"], observed=True)
        per = max(1, q // max(1, strata.ngroups))
        for _, s in strata:
            pick += list(rng.permutation(s.uid.to_numpy())[:per])
        rest = g[~g.uid.isin(pick)]
        need = q - int(g.uid.isin(pick).sum())
        if need > 0:
            pick += list(rng.permutation(rest.uid.to_numpy())[:need])
    sel = neg[neg.uid.isin(pick)]
    # 2) family coverage: top up any family below 10
    fam = lambda df: df.error_types.str.split(",").explode().str.strip().value_counts()
    for f in fam(neg).index:
        have = int(fam(sel).get(f, 0))
        if have < 10:
            cand = neg[~neg.uid.isin(sel.uid) & neg.error_types.str.contains(f)]
            sel = pd.concat([sel, cand.sample(min(10 - have, len(cand)), random_state=seed)])
    sel = sel.sample(frac=1, random_state=seed)                       # random order: each chunk representative
    sel[["uid", "doc_id", "source", "len_group", "edit_bin", "nfam_bin", "error_types", "char_sim"]].to_csv(SUB1, index=False)

    # Test 2: MAFAT half from test-1 articles (reuses summary translations), balanced faithful/corrupted,
    # faithful rewrites included; mFACT half stratified by article- and summary-length bins.
    d = R.load_bench()
    docs = sel.doc_id.unique()
    mf = d[(d.part == "MAFAT-Aug") & d.doc_id.isin(rng.permutation(docs)[:45])]
    pos = pd.concat([mf[(mf.y == 1) & mf.method.isna()], mf[(mf.y == 1) & mf.method.notna()]]).drop_duplicates("uid")
    mafat2 = pd.concat([pos.sample(min(len(pos), n2_each // 2), random_state=seed),
                        mf[mf.y == 0].sample(n2_each - min(len(pos), n2_each // 2), random_state=seed)])
    fa = d[d.part == "mFACT-He"]
    mfact2 = fa.sample(frac=1, random_state=seed).groupby(["y", "premise_bin", "hypothesis_bin"]).head(8)   # every length cell, then sampled down
    mfact2 = pd.concat([mfact2[mfact2.y == 1].sample(n2_each // 2 + 1, random_state=seed),
                        mfact2[mfact2.y == 0].sample(n2_each - n2_each // 2 - 1, random_state=seed)])
    t2 = pd.concat([mafat2, mfact2]).sample(frac=1, random_state=seed)
    t2["kind"] = np.where(t2.y == 0, "corrupted", np.where(t2.method.notna(), "rewrite:" + t2.method.astype(str), "faithful"))
    t2[["uid", "part", "doc_id", "source", "y", "kind", "error_types", "premise_bin", "hypothesis_bin"]].to_csv(SUB2, index=False)

    print(f"TEST 1 (erasure): {len(sel)} corrupted MAFAT summaries from {sel.doc_id.nunique()} articles")
    for c in ["source", "len_group", "edit_bin", "nfam_bin"]:
        print(f"  {c}: {sel[c].value_counts().to_dict()}")
    fs = fam(sel)
    print(f"  error families covered: {len(fs)}/{len(fam(neg))}, min per family {fs.min()}, median {int(fs.median())}")
    print(f"TEST 2 (translate-test): {len(t2)} article-summary pairs from {t2.premise.nunique()} articles")
    print("  " + str(t2.groupby(["part", "kind"]).size().to_dict()))
    print(f"  MAFAT sources: {t2[t2.part == 'MAFAT-Aug'].source.value_counts().to_dict()}")
    print(f"  mFACT article-length bins: {t2[t2.part == 'mFACT-He'].premise_bin.value_counts().sort_index().to_dict()}")


# ---------------------------------------------------------------- API translation with cache, in parallel

ACACHE = os.path.join(R.CACHE, "api_translations.jsonl")
PREAMBLE = re.compile(r"^\s*(here is|here's|the translation|translation:|sure[,!])", re.I)


def load_acache():
    out = {}
    if os.path.exists(ACACHE):
        for line in open(ACACHE, encoding="utf-8"):
            r = json.loads(line)
            out[(r["model"], r["k"])] = r
    return out


def api_translate_many(texts, models, workers=8):
    """Translate each unique text with each model (cache-first, parallel). Returns {(model, key): record}."""
    from concurrent.futures import ThreadPoolExecutor
    import boto3
    cache = load_acache()
    hc, brt = haiku_client(), boto3.Session(region_name=REGION).client("bedrock-runtime")
    jobs = [(mdl, he) for mdl in models for he in sorted(set(texts)) if (mdl, T.key(he)) not in cache]

    def run(job):
        mdl, he = job
        for attempt in range(4):
            try:
                t0 = time.time()
                en, ti, to = translate_haiku(hc, he) if mdl == HAIKU else translate_nova(brt, he)
                return {"model": mdl, "k": T.key(he), "he": he, "en": en, "in": ti, "out": to,
                        "sec": round(time.time() - t0, 2)}
            except Exception as e:  # throttling / transient
                err = f"{type(e).__name__}: {str(e)[:120]}"
                time.sleep(2 ** attempt)
        return {"model": mdl, "k": T.key(he), "he": he, "en": None, "error": err}

    with ThreadPoolExecutor(workers) as ex, open(ACACHE, "a", encoding="utf-8") as f:
        for r in ex.map(run, jobs):
            if r.get("en") is not None:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                cache[(r["model"], r["k"])] = r
            else:
                print("  FAILED:", r["model"], r.get("error"))
    return cache, len(jobs)


def en_sents(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])|\n+", text.strip()) if s.strip()]


# explicit negators plus implicit ones ("fails to emphasize", "lacks", "lack of") that keep a Hebrew לא
NEG_EN = re.compile(r"\b(not|cannot|no|never|none|nothing|without|neither|nor|fail(s|ed)? to|lack(s|ed|ing)?|lack of|absence of)\b|n't", re.I)
_TG = None


def _anchor(he_sent, en_sents_doc, tg):
    """Index of the document-level English sentence that best matches the sentence-level translation of
    he_sent (handles the translator splitting/merging sentences). Returns (index, similarity)."""
    import difflib
    ref = tg.get(T.key(he_sent))
    if ref is None or not en_sents_doc:
        return None, 0.0
    sims = [difflib.SequenceMatcher(None, T.norm(ref), T.norm(e)).ratio() for e in en_sents_doc]
    i = max(range(len(sims)), key=sims.__getitem__)
    return i, sims[i]


def doc_erasure_rows(c_uid, gold_he, corr_he, cache, mdl, min_sim=0.35):
    """Per edited Hebrew sentence: erased if its English counterparts in the two document-level translations
    are identical. Counterparts are anchored through the sentence-level translation of each Hebrew sentence."""
    import difflib
    global _TG
    if _TG is None:
        _TG = T.load_tcache()
    gs, cs = T.split_sents(gold_he), T.split_sents(corr_he)
    ges = en_sents(cache[(mdl, T.key(gold_he))]["en"])
    ces = en_sents(cache[(mdl, T.key(corr_he))]["en"])
    rows = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, gs, cs, autojunk=False).get_opcodes():
        if op != "replace" or (i2 - i1) != (j2 - j1):
            continue
        for g, c in zip(gs[i1:i2], cs[j1:j2]):
            ig, sg = _anchor(g, ges, _TG)
            ic, sc = _anchor(c, ces, _TG)
            ok = ig is not None and ic is not None and min(sg, sc) >= min_sim
            row = {"uid": c_uid, "model": mdl, "aligned": ok, "anchor_sim": round(min(sg, sc), 2),
                   "edit_kind": T.edit_kind(g, c), "gold_he": g, "corr_he": c}
            if ok:
                ce_ = ces[ic]
                new_digits = set(DIG.findall(c)) - set(DIG.findall(g))
                row.update(gold_en=ges[ig], corr_en=ce_, erased=T.norm(ges[ig]) == T.norm(ce_),
                           neg_lost=len(NEG_EN.findall(ges[ig])) == len(NEG_EN.findall(ce_)),
                           num_erased=(not new_digits <= set(DIG.findall(ce_))) if new_digits else None)
            rows.append(row)
    return rows


# ---------------------------------------------------------------- meaning judge (document-level wording varies)

JCACHE = os.path.join(R.CACHE, "api_meaning_judge.jsonl")
JUDGE_SYSTEM = "You compare two English sentences. Reply with exactly one word: same or different."
JUDGE_PROMPT = ("Sentence A: {a}\nSentence B: {b}\n\n"
                "Do A and B state different facts? Count as DIFFERENT: a negation or its removal, a different "
                "number or date, a different person, place, organization or object, a different actor or "
                "direction (who did what to whom), past vs present vs future, singular vs plural of an entity, "
                "a person's gender, or a different relation between facts.\n"
                "Count as SAME: synonyms, paraphrase, different word order or sentence structure, different "
                "spellings or transliterations of the same name (e.g. 'al-Dubaa' vs 'al-Daba'a'), and small "
                "wording shifts such as 'will address' vs 'would be required to address' that keep the fact.\n"
                "Reply with one word: different or same.")


def judge_many(pairs, workers=8):
    """pairs: list of (a, b) English sentences -> {key: 'same'|'different'} using Claude Haiku (cached)."""
    from concurrent.futures import ThreadPoolExecutor
    cache = {}
    if os.path.exists(JCACHE):
        for line in open(JCACHE, encoding="utf-8"):
            r = json.loads(line); cache[r["k"]] = r["label"]
    hc = haiku_client()
    k = lambda a, b: T.key(a + "␞" + b)
    todo = sorted({(a, b) for a, b in pairs if k(a, b) not in cache})

    def run(ab):
        a, b = ab
        for attempt in range(4):
            try:
                r = hc.messages.create(model=HAIKU, max_tokens=5, system=JUDGE_SYSTEM,
                                       messages=[{"role": "user", "content": JUDGE_PROMPT.format(a=a, b=b)}])
                t = "".join(x.text for x in r.content if x.type == "text").strip().lower()
                return ab, ("different" if t.startswith("diff") else "same" if t.startswith("same") else "unparsed:" + t)
            except Exception:
                time.sleep(2 ** attempt)
        return ab, "error"

    with ThreadPoolExecutor(workers) as ex, open(JCACHE, "a", encoding="utf-8") as f:
        for (a, b), lab in ex.map(run, todo):
            cache[k(a, b)] = lab
            f.write(json.dumps({"k": k(a, b), "a": a, "b": b, "label": lab}, ensure_ascii=False) + "\n")
    return {k(a, b): cache[k(a, b)] for a, b in pairs}, len(todo)


DIG = re.compile(r"\d+")


def add_meaning(df):
    """Adds sem_erased (judge says 'same' for an edited pair) and num_erased (changed Hebrew digits absent in EN)."""
    x = df[df.aligned].copy()
    labels, n_new = judge_many(list(zip(x.gold_en, x.corr_en)))
    x["judge"] = [labels[T.key(a + "␞" + b)] for a, b in zip(x.gold_en, x.corr_en)]
    x["sem_erased"] = x.judge == "same"
    x["num_erased"] = [(DIG.findall(g) != DIG.findall(c)) and (DIG.findall(ge) == DIG.findall(ce))
                       if "number" in k else None
                       for g, c, ge, ce, k in zip(x.gold_he, x.corr_he, x.gold_en, x.corr_en, x.edit_kind)]
    return x, n_new


EDIT_SYSTEM = "You check translations. Reply with exactly one word: yes or no."
EDIT_PROMPT = ("A Hebrew sentence was edited.\nOriginal Hebrew: {g}\nEdited Hebrew: {c}\n"
               "English translation of the EDITED sentence: {e}\n\n"
               "Does the English translation convey what the edit changed (so a reader of the English would see "
               "the edited meaning, not the original one)? Ignore wording, spelling of names and style. "
               "Reply 'yes' if the change is visible in the English, 'no' if the English reads like the original.")


def edit_visible_many(rows, workers=8):
    """rows: (gold_he, corr_he, corr_en) -> {key: 'yes'|'no'} ; 'no' = the edit was erased in translation."""
    from concurrent.futures import ThreadPoolExecutor
    path = os.path.join(R.CACHE, "api_edit_visible.jsonl")
    cache = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line); cache[r["k"]] = r["label"]
    hc = haiku_client()
    k = lambda g, c, e: T.key("␞".join([g, c, e]))
    todo = sorted({r for r in rows if k(*r) not in cache})

    def run(r):
        for attempt in range(4):
            try:
                out = hc.messages.create(model=HAIKU, max_tokens=5, system=EDIT_SYSTEM, messages=[
                    {"role": "user", "content": EDIT_PROMPT.format(g=r[0], c=r[1], e=r[2])}])
                t = "".join(x.text for x in out.content if x.type == "text").strip().lower()
                return r, "yes" if t.startswith("yes") else "no" if t.startswith("no") else "unparsed:" + t
            except Exception:
                time.sleep(2 ** attempt)
        return r, "error"

    with ThreadPoolExecutor(workers) as ex, open(path, "a", encoding="utf-8") as f:
        for r, lab in ex.map(run, todo):
            cache[k(*r)] = lab
            f.write(json.dumps({"k": k(*r), "gold_he": r[0], "corr_he": r[1], "corr_en": r[2], "label": lab},
                               ensure_ascii=False) + "\n")
    return {k(*r): cache[k(*r)] for r in rows}, len(todo)


def corr_en_for(rows_df, cache, mdl):
    """English counterpart of each corrupted Hebrew sentence inside the document-level translation."""
    tg = T.load_tcache()
    m, _ = T.mafat()
    hyp = m.set_index("uid").hypothesis
    out = []
    for r in rows_df.itertuples():
        ces = en_sents(cache[(mdl, T.key(hyp[r.uid]))]["en"])
        i, s = _anchor(r.corr_he, ces, tg)
        out.append(ces[i] if i is not None and s >= 0.35 else None)
    return out


def calibrate_edit_check(n_pairs=20):
    """Ground truth where it exists without a judge: negations (negator present?) and numbers (digits present?)."""
    import pandas as pd
    load_creds()
    try:
        import truststore; truststore.inject_into_ssl()
    except Exception:
        pass
    c = load_acache()
    a = pd.read_csv(os.path.join(R.HERE, "translation_erasure_doclevel.csv"))
    a = a[a.uid.isin(pd.read_csv(SUB1).uid.head(n_pairs))].drop_duplicates(["model", "gold_he", "corr_he"])
    res = []
    for mdl, x in a.groupby("model"):
        x = x.copy(); x["corr_en"] = corr_en_for(x, c, mdl); x = x[x.corr_en.notna()]
        lab, n_new = edit_visible_many(list(zip(x.gold_he, x.corr_he, x.corr_en)))
        x["visible"] = [lab[T.key("␞".join(r))] for r in zip(x.gold_he, x.corr_he, x.corr_en)]
        res.append(x); print(f"{mdl.split('.')[1][:9]}: {n_new} judge calls, {len(x)} edited sentences")
    x = pd.concat(res)
    # ground truth 1: pure negation insertions/removals -> visible iff English negator count differs from Hebrew-original translation
    ng = x[x.edit_kind == "negation"].copy()
    hneg = lambda s: len(re.findall(r"(^|\s)(לא|אין|אינו|אינה|אינם|אינן|ללא|בלי)(\s|$)", s))
    ng["truth"] = [(len(NEG_EN.findall(e)) > 0) == (hneg(cc) > hneg(g)) for g, cc, e in zip(ng.gold_he, ng.corr_he, ng.corr_en)]
    agree = (ng.truth == (ng.visible == "yes")).mean()
    print(f"negation-only edits: judge agrees with negator test on {agree:.0%} of {len(ng)}")
    # ground truth 2: pure number edits -> visible iff the edited digits appear in the English
    nb = x[x.edit_kind == "number"].copy()
    nb["truth"] = [set(DIG.findall(cc)) - set(DIG.findall(g)) <= set(DIG.findall(e)) for g, cc, e in zip(nb.gold_he, nb.corr_he, nb.corr_en)]
    print(f"number-only edits: judge agrees with digit test on {(nb.truth == (nb.visible == 'yes')).mean():.0%} of {len(nb)}")
    print("erased (judge says not visible) by model:", x.groupby("model").visible.apply(lambda v: f"{(v == 'no').mean():.1%} of {len(v)}").to_dict())
    for _, r in pd.concat([ng[ng.truth != (ng.visible == "yes")], nb[nb.truth != (nb.visible == "yes")]]).head(4).iterrows():
        print(f"  DISAGREE [{r.edit_kind}, judge={r.visible}]\n    G: {r.gold_he[:100]}\n    C: {r.corr_he[:100]}\n    E: {r.corr_en[:120]}")


def calibrate_judge(n_pairs=20):
    """Judge must say 'same' on unchanged sentences and 'different' where a negation demonstrably survived."""
    import difflib
    import pandas as pd
    load_creds()
    try:
        import truststore; truststore.inject_into_ssl()
    except Exception:
        pass
    c, tg = load_acache(), T.load_tcache()
    m, gold = T.mafat()
    hyp, doc = m.set_index("uid").hypothesis, m.set_index("uid").doc_id
    same_pairs = []
    for u in pd.read_csv(SUB1).uid.head(n_pairs):
        gs, cs = T.split_sents(gold[doc[u]]), T.split_sents(hyp[u])
        for mdl in (HAIKU, NOVA):
            ges, ces = en_sents(c[(mdl, T.key(gold[doc[u]]))]["en"]), en_sents(c[(mdl, T.key(hyp[u]))]["en"])
            for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, gs, cs, autojunk=False).get_opcodes():
                if op == "equal":
                    for g in gs[i1:i2]:
                        ig, s1 = _anchor(g, ges, tg); ic, s2 = _anchor(g, ces, tg)
                        if ig is not None and min(s1, s2) >= 0.35:
                            same_pairs.append((ges[ig], ces[ic]))
    a = pd.read_csv(os.path.join(R.HERE, "translation_erasure_doclevel.csv"))
    a = a[a.aligned & a.uid.isin(pd.read_csv(SUB1).uid.head(n_pairs))]
    neg_kept = a[a.edit_kind.str.contains("negation") & (a.neg_lost == False)]
    neg_lost = a[a.edit_kind.str.contains("negation") & (a.neg_lost == True)]
    lab, n_new = judge_many(same_pairs + list(zip(neg_kept.gold_en, neg_kept.corr_en)) + list(zip(neg_lost.gold_en, neg_lost.corr_en)))
    g = lambda ps: [lab[T.key(x + "␞" + y)] for x, y in ps]
    s1 = g(same_pairs); s2 = g(zip(neg_kept.gold_en, neg_kept.corr_en)); s3 = g(zip(neg_lost.gold_en, neg_lost.corr_en))
    print(f"judge calls {n_new}")
    print(f"  unchanged sentences (expect same):           same {s1.count('same')}/{len(s1)}")
    print(f"  negation survived in EN (expect different):  different {s2.count('different')}/{len(s2)}")
    print(f"  negation lost in EN (expect same):           same {s3.count('same')}/{len(s3)}")
    bad = [(x, y, l) for (x, y), l in zip(same_pairs, s1) if l != "same"][:3] + \
          [(x, y, l) for (x, y), l in zip(zip(neg_kept.gold_en, neg_kept.corr_en), s2) if l != "different"][:3]
    for x, y, l in bad:
        print(f"  MISS [{l}]\n    A: {x[:140]}\n    B: {y[:140]}")


TT_OUT = os.path.join(R.HERE, "translate_test_api_results.csv")   # one row per pair: all texts + all scores


def translate_test_api(chunk=20, max_pairs=None):
    """Test 2: full article + summary translated whole (one request each) by both models; FactCG on the English;
    compared with Hebrew FactCG and MorphEvalHeb on the same pairs. Everything is saved per row."""
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    from summary_augmentation.pages.metrics.factcg import FactCGScorer
    load_creds()
    try:
        import truststore; truststore.inject_into_ssl()
    except Exception:
        pass
    d = R.load_bench().set_index("uid")
    sc = pd.read_csv(os.path.join(R.HERE, "scores.csv")).set_index("uid")
    sub = pd.read_csv(SUB2)
    if max_pairs:
        sub = sub.head(max_pairs)
    done = set(pd.read_csv(TT_OUT).uid) if os.path.exists(TT_OUT) else set()
    todo = sub[~sub.uid.isin(done)]
    log = open(os.path.join(R.HERE, "translate_test_api.log"), "a", encoding="utf-8")

    def say(msg):
        print(msg, flush=True); log.write(msg + "\n"); log.flush()

    say(f"=== start {time.strftime('%F %T')}  pairs={len(sub)} done={len(done)} models={HAIKU}, {NOVA}")
    fc = FactCGScorer(device=R.device())
    rng = np.random.default_rng(0)
    for k, c0 in enumerate(range(0, len(todo), chunk), 1):
        part = todo.iloc[c0:c0 + chunk]
        texts = list(d.loc[part.uid, "premise"]) + list(d.loc[part.uid, "hypothesis"])
        t0 = time.time()
        cache, n_new = api_translate_many(texts, [HAIKU, NOVA])
        dt = time.time() - t0
        rows, issues = [], []
        for r in part.itertuples():
            prem, hyp = d.at[r.uid, "premise"], d.at[r.uid, "hypothesis"]
            row = {"uid": r.uid, "part": r.part, "doc_id": r.doc_id, "source": r.source, "y": r.y, "kind": r.kind,
                   "error_types": r.error_types, "premise_he": prem, "hypothesis_he": hyp,
                   "FactCG-HE": sc.at[r.uid, "FactCG"], "MorphEvalHeb": sc.at[r.uid, "MorphEvalHeb"]}
            for mdl, tag in [(HAIKU, "haiku45"), (NOVA, "novamicro")]:
                pe, he_ = cache.get((mdl, T.key(prem))), cache.get((mdl, T.key(hyp)))
                if pe is None or he_ is None:
                    issues.append(f"{tag} missing translation uid={r.uid}"); continue
                ratio = len(pe["en"]) / len(prem)
                if not 0.8 < ratio < 2.5 or PREAMBLE.search(pe["en"]) or len(T.HEB.findall(pe["en"])) > 5:
                    issues.append(f"{tag} article uid={r.uid} len_ratio={ratio:.2f}")
                row.update({f"premise_en_{tag}": pe["en"], f"hypothesis_en_{tag}": he_["en"],
                            f"premise_len_ratio_{tag}": round(ratio, 3),
                            f"FactCG-EN-{tag}": fc.score(pe["en"], he_["en"]).score})
            rows.append(row)
        pd.DataFrame(rows).to_csv(TT_OUT, mode="a", header=not os.path.exists(TT_OUT), index=False)
        a = pd.read_csv(TT_OUT)
        say(f"[chunk {k}] pairs {len(part)} | new API calls {n_new} in {dt:.0f}s | issues {len(issues)}")
        for i in issues[:5]:
            say(f"   ISSUE {i}")
        for part_name, x in [("ALL", a)] + list(a.groupby("part")):
            if x.y.nunique() < 2:
                continue
            res = []
            for col in ["FactCG-EN-haiku45", "FactCG-EN-novamicro", "FactCG-HE", "MorphEvalHeb"]:
                v = x[[col, "y"]].dropna()
                bs = [roc_auc_score(v.y.iloc[i], v[col].iloc[i]) for i in
                      (rng.integers(0, len(v), len(v)) for _ in range(300)) if v.y.iloc[i].nunique() == 2]
                res.append(f"{col} {roc_auc_score(v.y, v[col]):.3f} [{np.percentile(bs, 2.5):.2f},{np.percentile(bs, 97.5):.2f}]")
            say(f"   AUC {part_name} (n={len(x)}, faithful {int(x.y.sum())}): " + " | ".join(res))
    say(f"=== end {time.strftime('%F %T')}")


def erasure_doc_stream(chunk=20, max_chunks=None):
    """Test 1, document-level: chunks of 20 subset pairs; translate whole gold + corrupted summaries with both
    models, check format, compute erasure per edited sentence, compare with the sentence-level run."""
    import pandas as pd
    load_creds()
    try:
        import truststore; truststore.inject_into_ssl()
    except Exception:
        pass
    m, gold = T.mafat()
    sub = pd.read_csv(SUB1)
    hyp = m.set_index("uid").hypothesis
    out = os.path.join(R.HERE, "translation_erasure_doclevel.csv")
    done = set(pd.read_csv(out).uid) if os.path.exists(out) else set()
    sl = pd.read_csv(os.path.join(R.HERE, "translation_erasure_translategemma-12b-it.csv"))
    log = open(os.path.join(R.HERE, "translation_erasure_doclevel.log"), "a", encoding="utf-8")

    def say(msg):
        print(msg, flush=True); log.write(msg + "\n"); log.flush()

    todo = sub[~sub.uid.isin(done)]
    say(f"=== start {time.strftime('%F %T')}  subset={len(sub)} done={len(done)} models={HAIKU}, {NOVA}")
    for k, c0 in enumerate(range(0, len(todo), chunk), 1):
        part = todo.iloc[c0:c0 + chunk]
        texts = list(part.uid.map(hyp)) + list(part.doc_id.map(gold))
        t0 = time.time()
        cache, n_new = api_translate_many(texts, [HAIKU, NOVA])
        dt = time.time() - t0
        rows, fmt = [], {mdl: {"preamble": 0, "hebrew_left": 0, "len_ratio": 0, "n": 0} for mdl in (HAIKU, NOVA)}
        for t in set(texts):
            for mdl in (HAIKU, NOVA):
                en = cache[(mdl, T.key(t))]["en"]
                fmt[mdl]["n"] += 1
                fmt[mdl]["preamble"] += bool(PREAMBLE.search(en))
                fmt[mdl]["hebrew_left"] += len(T.HEB.findall(en)) > 5
                fmt[mdl]["len_ratio"] += not (0.6 < len(en) / len(t) < 2.5)
        for r in part.itertuples():
            for mdl in (HAIKU, NOVA):
                rows += doc_erasure_rows(r.uid, gold[r.doc_id], hyp[r.uid], cache, mdl)
        e = pd.DataFrame(rows)
        e.to_csv(out, mode="a", header=not os.path.exists(out), index=False)
        a = pd.read_csv(out)
        say(f"[chunk {k}] pairs {len(part)} | texts {len(set(texts))} | new API calls {n_new} in {dt:.0f}s")
        for mdl in (HAIKU, NOVA):
            f = fmt[mdl]; x = a[(a.model == mdl)]; xa = x[x.aligned]
            ng = xa[xa.edit_kind.str.contains("negation")]
            nb = xa[xa.num_erased.notna()]
            say(f"   {mdl.split('.')[1][:14]}: format preamble {f['preamble']}/{f['n']}, Hebrew left {f['hebrew_left']}, "
                f"odd length {f['len_ratio']} | alignment {x.aligned.mean():.0%} | cumulative ({x.uid.nunique()} pairs): "
                f"NEGATION LOST {ng.neg_lost.mean():.0%} of {len(ng)} | NEW NUMBER MISSING {nb.num_erased.astype(bool).mean():.0%} of {len(nb)}")
        # paired comparison with the sentence-level run on the same edited sentences
        slp = sl[sl.uid.isin(a.uid)].drop_duplicates(["gold_he", "corr_he"]).copy()
        slp["k"] = [T.edit_kind(g, c) for g, c in zip(slp.gold_he, slp.corr_he)]
        sn = slp[slp.k.str.contains("negation")]
        sn_lost = [len(NEG_EN.findall(g)) == len(NEG_EN.findall(c)) for g, c in zip(sn.gold_en, sn.corr_en)]
        say(f"   sentence-level TranslateGemma, same pairs: NEGATION LOST {sum(sn_lost)/max(1,len(sn_lost)):.0%} of {len(sn_lost)}")
        if max_chunks and k >= max_chunks:
            say(f"stopping after {k} chunk(s) (max_chunks)")
            break
    say(f"=== end {time.strftime('%F %T')}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "translate_test":
        translate_test_api(max_pairs=int(sys.argv[2]) if len(sys.argv) > 2 else None)
    elif cmd == "erasure_doc":
        erasure_doc_stream(max_chunks=int(sys.argv[2]) if len(sys.argv) > 2 else None)
    else:
        {"smoke": smoke, "build_subsets": build_subsets, "calibrate": calibrate_judge, "calibrate_edit": calibrate_edit_check}[cmd]()
