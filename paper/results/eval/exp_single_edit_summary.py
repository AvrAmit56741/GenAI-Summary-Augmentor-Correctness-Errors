"""Check for test 3: are SINGLE-word errors also missed when placed in the FULL summary (how MorphEvalHeb is used)?
Takes minimal pairs from cache/patching.jsonl (one typed word changed in one faithful sentence), puts the edit into
the full faithful reference summary, and scores both with the full article.
    python exp_single_edit_summary.py [per_kind]  -> single_edit_summary.csv (+ printed detection per kind)
"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eval as R
from exp_layers import MAX_LEN
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "single_edit_summary.csv")


def main(per_kind=25, seed=0):
    import torch
    r = pd.DataFrame([json.loads(x) for x in open(os.path.join(HERE, "cache", "patching.jsonl"))])
    r = r[r.status.isin(["ok", "not_detected"]) & (r.p_g >= .5)]
    w = pd.read_csv(os.path.join(HERE, "layers_words_sample.csv")).drop_duplicates("unit_id").set_index("unit_id")
    r["gold_sent"] = r.unit_id.map(w.gold_sent)
    r["doc_id"] = r.unit_id.map(w.doc_id)
    b = R.load_bench()
    g = b[(b.part == "MAFAT-Aug") & (b.y == 1)].drop_duplicates("doc_id").set_index("doc_id")
    r = r[r.doc_id.isin(g.index)]
    r = r[[gs in g.hypothesis[d] for gs, d in zip(r.gold_sent, r.doc_id)]]
    r = r.sample(frac=1, random_state=seed).groupby("kind").head(per_kind)
    tok, model = R.load_cls("NLI-hebrew-binary-correctness-metric")
    dev = R.device()
    cache = {}

    def p(art, hyp):
        if (art, hyp) not in cache:
            e = tok(art, hyp, return_tensors="pt", truncation="only_first", max_length=MAX_LEN)
            with torch.no_grad():
                cache[(art, hyp)] = torch.softmax(model(**{k: v.to(dev) for k, v in e.items() if k != "token_type_ids"}).logits, -1)[0, 1].item()
        return cache[(art, hyp)]
    rows = []
    for i, x in enumerate(r.itertuples()):
        art, summ = g.premise[x.doc_id], g.hypothesis[x.doc_id]
        edited = summ.replace(x.gold_sent, x.minimal_sent, 1)
        rows.append({"unit_id": x.unit_id, "kind": x.kind, "gold_word": x.gold_word, "corr_word": x.corr_word,
                     "p_sent_gold": x.p_g, "p_sent_edit": x.p_c, "p_summ_gold": p(art, summ), "p_summ_edit": p(art, edited),
                     "n_sents_summary": summ.count(".")})
        if (i + 1) % 20 == 0:
            d = pd.DataFrame(rows); print(f"  {i+1}/{len(r)}: summary-level flagged {(d.p_summ_edit < .5).mean():.2f}", flush=True)
    d = pd.DataFrame(rows); d.to_csv(OUT, index=False)
    d = d[d.p_summ_gold >= .5]
    print("faithful summaries kept:", len(d))
    print(d.groupby("kind").agg(n=("kind", "size"), sent_flagged=("p_sent_edit", lambda s: (s < .5).mean()),
                               summ_flagged=("p_summ_edit", lambda s: (s < .5).mean()),
                               summ_score_drop=("p_summ_edit", lambda s: 0)).round(2).drop(columns="summ_score_drop"))
    print("overall: sentence flagged", round((d.p_sent_edit < .5).mean(), 2), "| summary flagged", round((d.p_summ_edit < .5).mean(), 2),
          "| summary p lower than gold", round((d.p_summ_edit < d.p_summ_gold - 0.01).mean(), 2))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 25)
