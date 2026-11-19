"""Targeted test for GPT-5.6-Luna (OpenAI, via AWS Bedrock's "mantle" Responses API endpoint --
a different auth/endpoint than the standard bedrock-runtime Converse API used for Claude/Nova/GLM/
Ministral/Gemma). Same 50 known error-prone articles and same minimal ("thin") Hebrew prompt as
exp_current_model_summaries_targeted.py, for a direct comparison.

Luna is the lightweight/cheapest tier of GPT-5.6 on Bedrock (Sol = flagship reasoning, Terra =
balanced, Luna = fast/cheap) -- the closest current analogue to GPT-4o-mini's role in the original
baseline comparisons.

Must run under a Python env with `openai` and `aws_bedrock_token_generator` installed (NOT
morphevalheb_env, which lacks both) -- use the urlf-classification project's venv:
    /Users/amit.avraham/PycharmProjects/urlf-classification/url-classification/bin/python3

    python exp_current_model_summaries_mantle_targeted.py openai.gpt-5.6-luna 50
"""
import glob
import os
import re
import sys
import time

MODEL_DEFAULT = "openai.gpt-5.6-luna"
REGION = "us-east-1"
CREDS = os.environ.get("BEDROCK_CREDS_FILE", os.path.expanduser(
    "~/PycharmProjects/urlf-classification/experiments/bedrock_classifier/bedrock_aws_creds.local.env"))
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
BENCH = os.path.join(REPO, "thesisdatasets/train_datasets/Correctness/data/benchmark/combined.csv")

THIN_PROMPT = ("סכם את הטקסט הבא בעברית, סיכום מפורט ועשיר בפרטים ובעובדות, לא סיכום תמציתי או מופשט:"
               "\n\n{article}")


def load_creds():
    for line in open(CREDS):
        m = re.match(r"\s*(?:export\s+)?([A-Za-z_]+)=(.*)", line)
        if m:
            os.environ[m.group(1)] = m.group(2).strip().strip("\"'")


def pick_articles(n=50, offset=0):
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
    model = sys.argv[1] if len(sys.argv) > 1 else MODEL_DEFAULT
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    offset = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    tag = re.sub(r"[^A-Za-z0-9_-]", "_", model)
    suffix = f"_offset{offset}" if offset > 0 else ""
    out = os.path.join(HERE, f"current_model_summaries_{tag}_targeted{suffix}.csv")

    load_creds()
    try:
        import truststore; truststore.inject_into_ssl()
    except Exception:
        pass
    from openai import OpenAI
    from aws_bedrock_token_generator import provide_token

    base_url = f"https://bedrock-mantle.{REGION}.api.aws/openai/v1"
    token = provide_token(region=REGION)
    client = OpenAI(base_url=base_url, api_key=token)

    arts = pick_articles(n, offset=offset)
    print(f"model={model}  selected {len(arts)} known error-prone articles "
          f"(max_sev: {arts.max_sev.value_counts().to_dict()})", flush=True)

    rows = []
    for i, r in enumerate(arts.itertuples()):
        prompt = THIN_PROMPT.format(article=r.premise)
        t0 = time.time()
        for attempt in range(4):
            try:
                resp = client.responses.create(model=model, input=prompt, max_output_tokens=4000)
                break
            except Exception as e:
                if attempt == 3:
                    raise
                print(f"  retry {attempt} after error: {e}", flush=True)
                time.sleep(2 ** attempt)
        dt = time.time() - t0
        summary = (resp.output_text or "").strip()
        rows.append({"doc_id": r.doc_id, "source": r.source, "orig_max_severity": r.max_sev,
                     "orig_n_errors": r.n_err, "premise": r.premise, "generative_summary": summary,
                     "model": model, "prompt_style": "thin", "gen_seconds": round(dt, 1)})
        print(f"  [{i + 1}/{len(arts)}] {r.doc_id} (orig_sev={r.max_sev}) ({dt:.1f}s, {len(summary)} chars)",
              flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    trunc = (~df.generative_summary.str.strip().str[-1].isin([".", "!", "?", ":", ")", '"'])).sum()
    print(f"\nsaved {len(rows)} summaries -> {out}")
    print(f"QC: {trunc}/{len(rows)} summaries do not end in sentence-final punctuation "
          f"(possible truncation -- check max_output_tokens if this is high)")


if __name__ == "__main__":
    main()
