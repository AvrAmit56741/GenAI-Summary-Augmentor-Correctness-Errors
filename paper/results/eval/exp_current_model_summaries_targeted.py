"""Does DictaLM 3.0 still err on articles KNOWN to trigger Hebrew-morphology errors? (power-up of #18)

#18's random 69-article sample found only 1/69 qualifying errors for DictaLM 3.0 -- too few to say
anything with confidence. This script targets the most error-prone articles instead of a random
sample: the source articles behind the highest-severity errors the production pipeline already found
in OLD models (thesisdatasets/error_finder/*.csv), ranked by max recorded severity then error count.
These are articles whose Hebrew structure (ambiguous pronouns, gendered role nouns, passive/causative
verbs, smixut chains) is already *confirmed* to be hard, by the project's own data -- a much higher-
power test of "does the current model still stumble here" than a random sample of mostly-easy articles.

Also switches to a deliberately MINIMAL ("thin") prompt per the author's request, to test natural
model behavior without an elaborate instruction that might coach the model into extra care:
    "סכם את הטקסט הבא:\n\n{article}"
(contrast with exp_current_model_summaries.py's detailed length/style instructions, or the English
two-step prompt used for the API-model entries).

    python exp_current_model_summaries_targeted.py [N]   -> current_model_summaries_dictalm3_targeted.csv
    (N = how many top-severity articles to use, default 50)
"""
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eval as R  # noqa: E402

MODEL = "dicta-il/DictaLM-3.0-1.7B-Instruct"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = R.REPO
OUT = os.path.join(HERE, "current_model_summaries_dictalm3_targeted.csv")

THIN_PROMPT = ("סכם את הטקסט הבא בעברית, סיכום מפורט ועשיר בפרטים ובעובדות, לא סיכום תמציתי או מופשט:"
               "\n\n{article}")


def pick_articles(n=50, offset=0):
    """Top-n most error-prone articles by (max_sev, n_err), skipping the first `offset` (so a second
    batch can pick up where the first left off, with zero overlap)."""
    import pandas as pd
    files = sorted(glob.glob(os.path.join(REPO, "thesisdatasets/error_finder/*.csv")))
    e = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    e = e.dropna(subset=["premise", "doc_id", "severity"])
    by_doc = (e.groupby("doc_id")
                .agg(max_sev=("severity", "max"), n_err=("severity", "size"),
                     premise=("premise", "first"), source=("source", "first"))
                .reset_index()
                .sort_values(["max_sev", "n_err"], ascending=False))
    return by_doc.iloc[offset:offset + n].reset_index(drop=True)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    arts = pick_articles(n)
    print(f"selected {len(arts)} known error-prone articles (max_sev distribution: "
          f"{arts.max_sev.value_counts().to_dict()})", flush=True)

    print(f"loading {MODEL} on {R.device()} ...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(R.device())
    print(f"loaded in {time.time() - t0:.1f}s", flush=True)

    rows = []
    for i, r in enumerate(arts.itertuples()):
        prompt = THIN_PROMPT.format(article=r.premise)
        msgs = [{"role": "user", "content": prompt}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(R.device())
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=2200, do_sample=True, temperature=0.2, top_p=0.9,
                                  pad_token_id=tok.eos_token_id)
        dt = time.time() - t0
        summary = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
        rows.append({"doc_id": r.doc_id, "source": r.source, "orig_max_severity": r.max_sev,
                     "orig_n_errors": r.n_err, "premise": r.premise, "generative_summary": summary,
                     "model": MODEL, "prompt_style": "thin", "gen_seconds": round(dt, 1)})
        print(f"  [{i + 1}/{len(arts)}] {r.doc_id} (orig_sev={r.max_sev}) ({dt:.1f}s, {len(summary)} chars)",
              flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    trunc = (~df.generative_summary.str.strip().str[-1].isin([".", "!", "?", ":", ")", '"'])).sum()
    print(f"\nsaved {len(rows)} summaries -> {OUT}")
    print(f"QC: {trunc}/{len(rows)} summaries do not end in sentence-final punctuation "
          f"(possible truncation -- check max_new_tokens if this is high)")


if __name__ == "__main__":
    main()
