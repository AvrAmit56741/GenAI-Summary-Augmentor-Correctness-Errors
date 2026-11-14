"""ENFS% vs a morphology-aware Hebrew variant (ENFS-He).

ENFS (repo, hallucinations_metrics.calculate_enfs): a summary entity is hallucinated unless the exact
entity string is among the source's NER entities.
ENFS-He: strip Hebrew proclitics (ו ה ב כ ל מ ש, up to 3 chars, keeping >= 2 letters) from every token
of the entity and of the source, and count the entity as supported if its normalized token sequence
appears anywhere in the normalized source text.

    python exp_enfs.py   -> cache/ner_entities.jsonl (resumable), results printed + enfs_results.csv
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eval as R  # noqa: E402
from summary_augmentation.pages.metrics.misc.hallucinations_metrics import (  # noqa: E402
    load_ner_pipeline, extract_entities, calculate_enfs)

NER_CACHE = os.path.join(R.CACHE, "ner_entities.jsonl")
PREFIX = re.compile(r"^[והבכלמש]{1,3}(?=[א-ת]{2,})")
PUNCT = re.compile(r"[^\wא-ת]+")


def norm_tokens(text):
    toks = [PUNCT.sub("", t) for t in text.split()]
    return [PREFIX.sub("", t) for t in toks if t]


def enfs_he(src_text, hyp_ents):
    if not hyp_ents:
        return 0.0
    src = " " + " ".join(norm_tokens(src_text)) + " "
    src_raw = " " + " ".join(PUNCT.sub("", t) for t in src_text.split()) + " "
    halluc = 0
    for e in hyp_ents:
        nt = " ".join(norm_tokens(e["text"]))
        raw = " ".join(PUNCT.sub("", t) for t in e["text"].split())
        if not ((nt and f" {nt} " in src) or (raw and f" {raw} " in src_raw) or (raw and raw in src_raw)):
            halluc += 1
    return 100.0 * halluc / len(hyp_ents)


def main():
    d = R.load_bench()
    texts = pd.unique(pd.concat([d.premise, d.hypothesis]))
    done = {}
    if os.path.exists(NER_CACHE):
        for line in open(NER_CACHE, encoding="utf-8"):
            j = json.loads(line)
            done[j["t"]] = j["e"]
    todo = [t for t in texts if t not in done]
    if todo:
        ner = load_ner_pipeline("dicta-il/dictabert-ner")
        with open(NER_CACHE, "a", encoding="utf-8") as fh:
            for k, t in enumerate(todo):
                ents = extract_entities(t, ner, 0.5)
                done[t] = ents
                fh.write(json.dumps({"t": t, "e": ents}, ensure_ascii=False, default=float) + "\n")
                if k % 500 == 0:
                    print(f"  NER {k}/{len(todo)}", flush=True)
    rows = []
    for r in d.itertuples():
        pe, he = done[r.premise], done[r.hypothesis]
        enfs, n_ent, _, _ = calculate_enfs(pe, he)
        rows.append({"uid": r.uid, "part": r.part, "y": r.y, "n_ent": n_ent,
                     "ENFS": 0.0 if enfs is None else enfs, "ENFS-He": enfs_he(r.premise, he)})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(R.HERE, "enfs_results.csv"), index=False)
    for name, g in [("all", res)] + list(res.groupby("part")):
        out = [f"{name:10s} n={len(g)}"]
        for m in ["ENFS", "ENFS-He"]:
            faithful = g[g.y == 1]
            out.append(f"{m}: AUC {roc_auc_score(g.y, -g[m]):.3f}, false-alarm on faithful {(faithful[m] > 0).mean():.3f}")
        print(" | ".join(out))
    withent = res[res.n_ent > 0]
    print(f"summaries with >=1 entity: {len(withent)}/{len(res)}")
    for name, g in [("all", withent)] + list(withent.groupby("part")):
        print(f"  [entity subset] {name:10s} n={len(g)}  ENFS AUC {roc_auc_score(g.y, -g.ENFS):.3f}  "
              f"ENFS-He AUC {roc_auc_score(g.y, -g['ENFS-He']):.3f}")


if __name__ == "__main__":
    main()
