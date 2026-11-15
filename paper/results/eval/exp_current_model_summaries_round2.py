"""Round 2: expand exp_current_model_summaries.py's DictaLM 3.0 sample with more articles.

Same model, same verbatim prompt, same generation config as round 1 -- just a bigger, non-overlapping
sample (round 1's 26 articles excluded), so the two rounds can be concatenated into one dataset for a
more statistically useful error-rate estimate. Bagatz has only 4 MAFAT-Aug articles total and all 4
were used in round 1, so it's skipped here (0 more available).

    python exp_current_model_summaries_round2.py   -> current_model_summaries_dictalm3_round2.csv
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eval as R  # noqa: E402
from exp_current_model_summaries import MODEL, PROMPT_TMPL  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROUND1 = os.path.join(HERE, "current_model_summaries_dictalm3.csv")
OUT = os.path.join(HERE, "current_model_summaries_dictalm3_round2.csv")
N_PER_SOURCE = {"Israel_Hayom": 15, "Wikipedia": 12, "Weizmann": 8, "Knesset": 8, "Bagatz": 0}


def sample_articles(seed=23):
    import pandas as pd
    d = R.load_bench()
    m = d[d.part == "MAFAT-Aug"][["doc_id", "source", "premise"]].drop_duplicates("doc_id")
    done = set(pd.read_csv(ROUND1).doc_id)
    m = m[~m.doc_id.isin(done)]
    rows = []
    for src, n in N_PER_SOURCE.items():
        g = m[m.source == src]
        rows.append(g.sample(min(n, len(g)), random_state=seed))
    return pd.concat(rows).reset_index(drop=True)


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    arts = sample_articles()
    print(f"sampled {len(arts)} NEW articles (round 1 excluded): {arts.source.value_counts().to_dict()}", flush=True)

    print(f"loading {MODEL} on {R.device()} ...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(R.device())
    print(f"loaded in {time.time() - t0:.1f}s", flush=True)

    rows = []
    for i, r in enumerate(arts.itertuples()):
        prompt = PROMPT_TMPL.format(article=r.premise)
        msgs = [{"role": "user", "content": prompt}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(R.device())
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=512, do_sample=True, temperature=0.2, top_p=0.9,
                                  pad_token_id=tok.eos_token_id)
        dt = time.time() - t0
        summary = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
        rows.append({"doc_id": r.doc_id, "source": r.source, "premise": r.premise,
                     "generative_summary": summary, "model": MODEL, "gen_seconds": round(dt, 1)})
        print(f"  [{i + 1}/{len(arts)}] {r.source} {r.doc_id} ({dt:.1f}s, {len(summary)} chars)", flush=True)

    import pandas as pd
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nsaved {len(rows)} summaries -> {OUT}")


if __name__ == "__main__":
    main()
