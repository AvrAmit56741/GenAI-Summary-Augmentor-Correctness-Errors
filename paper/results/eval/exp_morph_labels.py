"""Test 1: label each (faithful, corrupted) sentence pair of layers_sample.csv with the MORPHOLOGICAL kind of its edit,
using Dicta's local analyzer dicta-il/dictabert-joint (lemma, POS, gender/number/tense/person, prefixes, smixut).
No API. Validated against the summary-level injected error_types (independent signal).

    python exp_morph_labels.py        -> layers_morph_labels.csv, layers_morph_validation.csv

Word-level classes (changed word pairs aligned with difflib):
  negation        a negation word (לא/אין/אינו/...) added or removed
  number_digit    digits changed
  gender          same lemma, Gender changed (verb or noun/adj)
  number_infl     same lemma, grammatical Number changed
  tense           same lemma, Tense changed
  person          same lemma, Person changed
  prefix          same lemma, non-article prefix letters changed (ו/ש/כש/מ/ל/ב...)
  construct       same lemma, smixut (construct-state) role or definite article changed inside a smixut phrase
  definiteness    same lemma, only the definite article changed (outside smixut)
  root_pattern    different lemma sharing >=3 root-like consonants in order (approximate)
  entity          different lemma, a proper noun involved
  pronoun_suffix  same lemma, only the pronominal ending changed (שלו/שלה)
  letter_swap     different lemma, surface forms differ in exactly one letter (near-homograph, e.g. לשבור/לשכור)
  lexical         any other word substitution / insertion / deletion
"""
import difflib
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.path.join(HERE, "layers_sample.csv")
OUT = os.path.join(HERE, "layers_morph_labels.csv")
CACHE = os.path.join(HERE, "cache", "morph_parse.jsonl")
NEG = {"לא", "אין", "אינו", "אינה", "אינם", "אינן", "בלי", "ללא", "אל"}


def parse_all(sents, bs=16):
    import torch
    from transformers import AutoModel, AutoTokenizer
    cache = {}
    if os.path.exists(CACHE):
        for line in open(CACHE):
            j = json.loads(line)
            cache[j["s"]] = j["t"]
    todo = [s for s in dict.fromkeys(sents) if s not in cache]
    if todo:
        tok = AutoTokenizer.from_pretrained("dicta-il/dictabert-joint")
        m = AutoModel.from_pretrained("dicta-il/dictabert-joint", trust_remote_code=True).eval()
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        m.to(dev)
        with open(CACHE, "a") as f:
            for i in range(0, len(todo), bs):
                b = todo[i:i + bs]
                with torch.no_grad():
                    r = m.predict(b, tok, output_style="json")
                for s, x in zip(b, r):
                    t = [{"w": w["token"], "lex": w.get("lex"), "pos": w["morph"]["pos"], "feats": w["morph"]["feats"],
                          "pre": w["morph"]["prefixes"], "dep": (w.get("syntax") or {}).get("dep_func")}
                         for w in x["tokens"]]
                    cache[s] = t
                    f.write(json.dumps({"s": s, "t": t}, ensure_ascii=False) + "\n")
                if (i // bs) % 10 == 0:
                    print(f"  parsed {i + len(b)}/{len(todo)}", flush=True)
    return cache


def cons(lex):
    """Root-like consonant skeleton: drop matres lectionis ו/י and final-form differences."""
    lex = (lex or "").translate(str.maketrans("ךםןףץ", "כמנפצ"))
    return re.sub(r"[^א-ת]|[וי]", "", lex)


def lcs(a, b):
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return sum(blk.size for blk in sm.get_matching_blocks())


def word_kinds(a, b):
    """Kinds for one aligned changed word pair (a, b are parsed tokens)."""
    k = set()
    if re.findall(r"\d+", a["w"]) != re.findall(r"\d+", b["w"]):
        k.add("number_digit")
    if a["lex"] == b["lex"]:
        fa, fb = a["feats"] or {}, b["feats"] or {}
        for key, name in [("Gender", "gender"), ("Number", "number_infl"), ("Tense", "tense"), ("Person", "person")]:
            if fa.get(key) and fb.get(key) and fa[key] != fb[key]:
                k.add(name)
        pa = [p for p in a["pre"] if p != "DET"]
        pb = [p for p in b["pre"] if p != "DET"]
        if pa != pb:
            k.add("prefix")
        det_ch = ("DET" in a["pre"]) != ("DET" in b["pre"])
        smx_a, smx_b = a["dep"] == "compound:smixut", b["dep"] == "compound:smixut"
        if smx_a != smx_b or (det_ch and (smx_a or smx_b)):
            k.add("construct")
        elif det_ch:
            k.add("definiteness")
        if not k and a["w"] != b["w"]:
            wa, wb = a["w"].strip(".,\"'()"), b["w"].strip(".,\"'()")
            if len(wa) >= 2 and len(wb) >= 2 and wa[:-1] == wb[:-1]:
                k.add("pronoun_suffix")                       # שלו/שלה, בו/בה: pronominal ending changed
            elif re.sub(r"^[ושכמלבה]{1,3}", "", wa) == re.sub(r"^[ושכמלבה]{1,3}", "", wb) and len(re.sub(r"^[ושכמלבה]{1,3}", "", wa)) >= 2:
                k.add("prefix")
            else:
                k.add("lexical")
    else:
        wa, wb = re.sub(r"[^א-ת]", "", a["w"]), re.sub(r"[^א-ת]", "", b["w"])
        stem = lambda x: re.sub(r"(ית|ה|ת)$", "", x)
        if "PROPN" in (a["pos"], b["pos"]):
            k.add("entity")
        elif wa != wb and len(stem(wa)) >= 2 and stem(wa) == stem(wb) and a["pos"] == b["pos"]:
            k.add("gender")                                   # lemma split by the parser, but only a fem. ending differs
        else:
            ca, cb = cons(a["lex"]), cons(b["lex"])
            wa, wb = re.sub(r"[^א-ת]", "", a["w"]), re.sub(r"[^א-ת]", "", b["w"])
            if len(wa) == len(wb) >= 3 and sum(x != y for x, y in zip(wa, wb)) == 1:
                k.add("letter_swap")
            elif min(len(ca), len(cb)) >= 3 and lcs(ca, cb) >= 3 and lcs(ca, cb) >= 0.75 * min(len(ca), len(cb)):
                k.add("root_pattern")
            elif not k:
                k.add("lexical")
    return k


def pair_kinds(tg, tc):
    wg, wc = [t["w"] for t in tg], [t["w"] for t in tc]
    kinds, n_changed = set(), 0
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, wg, wc, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        n_changed += max(i2 - i1, j2 - j1)
        A, B = tg[i1:i2], tc[j1:j2]
        la = {t["lex"] for t in A} | {t["w"] for t in A}
        lb = {t["lex"] for t in B} | {t["w"] for t in B}
        if (NEG & la) != (NEG & lb):
            kinds.add("negation")
            A = [t for t in A if t["lex"] not in NEG and t["w"] not in NEG]
            B = [t for t in B if t["lex"] not in NEG and t["w"] not in NEG]
        if op == "replace" and len(A) == len(B):
            for a, b in zip(A, B):
                kinds |= word_kinds(a, b)
        elif A or B:
            # unequal spans: pair by best lemma match, rest is lexical insertion/deletion
            used = set()
            for a in A:
                j = next((j for j, b in enumerate(B) if j not in used and b["lex"] == a["lex"]), None)
                if j is None:
                    kinds.add("lexical")
                else:
                    used.add(j)
                    kinds |= word_kinds(a, B[j])
            if len(B) > len(used) and len(B) - len(used) > 0 and not A:
                kinds.add("lexical")
            elif len(B) - len(used) > len(A) - len(used):
                kinds.add("lexical")
    kinds.discard("")
    return kinds, n_changed


# which injected (summary-level) types each morphological kind should come from, for validation
EXPECT = {
    "negation": ["sentence_negation"],
    "number_digit": ["number_swap", "entity_date_swap", "measure_unit_swap"],
    "gender": ["verb_gender_swap", "noun_gender_swap", "homographic_gender_errors"],
    "tense": ["verb_tense_swap"],
    "construct": ["construct_state_confusion"],
    "prefix": ["morphological_connective_confusion"],
    "root_pattern": ["hebrew_root_pattern_confusion"],
    "entity": ["entity_person_swap", "entity_organization_swap", "entity_location_swap", "entity_event_swap",
               "entity_title_swap"],
    "pronoun_person": ["pronoun_reference_errors"],
}


def main():
    s = pd.read_csv(SAMPLE)
    cache = parse_all(list(s.gold_sent) + list(s.corr_sent))
    rows = []
    for r in s.itertuples():
        k, n = pair_kinds(cache[r.gold_sent], cache[r.corr_sent])
        rows.append({"unit_id": r.unit_id, "morph_kinds": "+".join(sorted(k)) or "none", "n_changed_words": n,
                     "morph": "+".join(sorted(k)) if len(k) == 1 else ("none" if not k else "mixed")})
    lab = pd.DataFrame(rows)
    lab.to_csv(OUT, index=False)
    print(lab.morph.value_counts().to_string())
    # validation: P(listed injected type | morph kind) vs base rate in the sample
    d = s.merge(lab, on="unit_id")
    listed = d.error_types.str.split(",")
    vrows = []
    for kind, exp in EXPECT.items():
        m = d.morph == kind
        if kind == "pronoun_person":
            m = d.morph == "person"
        if m.sum() == 0:
            continue
        hit = listed[m].apply(lambda L: any(e in L for e in exp)).mean()
        base = listed.apply(lambda L: any(e in L for e in exp)).mean()
        vrows.append({"morph": kind, "n": int(m.sum()), "expected_types": ",".join(exp),
                      "share_listing_expected": round(hit, 3), "base_rate": round(base, 3), "lift": round(hit / base, 2)})
    v = pd.DataFrame(vrows)
    v.to_csv(os.path.join(HERE, "layers_morph_validation.csv"), index=False)
    print(v.to_string(index=False))


if __name__ == "__main__":
    main()
