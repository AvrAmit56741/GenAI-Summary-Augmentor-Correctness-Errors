"""Do current models still produce Hebrew-morphology factual errors? Step 1: generate summaries.

Context: a co-author raised a real objection -- the original taxonomy/error-finder data was generated
with GPT-3.5, Gemini 1.5/2.0, DictaLM 2.0 (2025-era models), so maybe current (Nov 2026) frontier
models no longer make these Hebrew morphological errors at all. This script generates fresh Hebrew
summaries with a CURRENT model, using the EXACT prompt the original DictaLM 2.0 pipeline used (for a
fair, apples-to-apples comparison), on a fresh sample of MAFAT articles. Step 2 (a separate script)
will run the exact error-finder prompt on these summaries via Claude (Bedrock), matching the original
error-finder methodology (verified from summary_augmentation/correctness_errors_prompts.py and
correctness_errors_llm_helper_functions.py).

Model: dicta-il/DictaLM-3.0-1.7B-Instruct (Apache-2.0, Feb 2026) -- the current, Hebrew-native
generation of the same DictaLM family already represented (as v2.0) in the original error-finder data.

Prompt (verbatim, from summary_augmentation/summary_generator_open_llm.py, the active generator for
the `summaries_dicta_*.jsonl` files -- including its exact spacing/typo, preserved for fidelity):
    "סכם את המאמר הבא בעברית באורך של150  עד 200 מילים, תוך התמקדות ברעיונות המרכזיים בלבד.\n\n"
    "<תחילת הטקסט לסיכום>\n{article}\n<סוף הטקסט לסיכום>"
Generation config matches the original: do_sample=True, temperature=0.2, top_p=0.9, max_new_tokens=512.

    python exp_current_model_summaries.py   -> current_model_summaries_dictalm3.csv
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eval as R  # noqa: E402

MODEL = "dicta-il/DictaLM-3.0-1.7B-Instruct"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "current_model_summaries_dictalm3.csv")
N_PER_SOURCE = {"Israel_Hayom": 8, "Wikipedia": 6, "Weizmann": 4, "Knesset": 4, "Bagatz": 4}

PROMPT_TMPL = ("סכם את המאמר הבא בעברית באורך של150  עד 200 מילים, תוך התמקדות ברעיונות המרכזיים בלבד.\n\n"
               "<תחילת הטקסט לסיכום>\n{article}\n<סוף הטקסט לסיכום>")


def sample_articles(seed=11):
    import numpy as np
    import pandas as pd
    d = R.load_bench()
    m = d[d.part == "MAFAT-Aug"][["doc_id", "source", "premise"]].drop_duplicates("doc_id")
    rng = np.random.default_rng(seed)
    rows = []
    for src, n in N_PER_SOURCE.items():
        g = m[m.source == src]
        rows.append(g.sample(min(n, len(g)), random_state=seed))
    return pd.concat(rows).reset_index(drop=True)


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    arts = sample_articles()
    print(f"sampled {len(arts)} articles: {arts.source.value_counts().to_dict()}")

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
