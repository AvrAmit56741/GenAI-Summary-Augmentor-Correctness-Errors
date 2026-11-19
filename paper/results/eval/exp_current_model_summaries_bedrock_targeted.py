"""Bedrock counterpart of exp_current_model_summaries_targeted.py -- same 50 known error-prone articles,
same minimal ("thin") Hebrew prompt, different current open-weight model, run via AWS Bedrock Converse.

    python exp_current_model_summaries_bedrock_targeted.py zai.glm-4.7-flash
    python exp_current_model_summaries_bedrock_targeted.py mistral.ministral-3-14b-instruct [n] [offset]

Output: current_model_summaries_<tag>_targeted.csv, where <tag> is the model id with dots/colons
stripped for a safe filename. Pass offset (3rd arg) to pick a non-overlapping next batch, e.g.
`... mistral.ministral-3-14b-instruct 50 50` for articles 51-100 by error-proneness rank; output
filename gets a _round2 suffix automatically when offset > 0.
"""
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eval as R  # noqa: E402
from exp_current_model_summaries_targeted import pick_articles, THIN_PROMPT  # noqa: E402

CREDS = os.environ.get("BEDROCK_CREDS_FILE", os.path.expanduser(
    "~/PycharmProjects/urlf-classification/experiments/bedrock_classifier/bedrock_aws_creds.local.env"))
REGION = "us-east-1"
HERE = os.path.dirname(os.path.abspath(__file__))


def load_creds():
    for line in open(CREDS):
        m = re.match(r"\s*(?:export\s+)?([A-Za-z_]+)=(.*)", line)
        if m:
            os.environ[m.group(1)] = m.group(2).strip().strip("\"'")


def main():
    model = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    offset = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    tag = re.sub(r"[^A-Za-z0-9_-]", "_", model)
    suffix = f"_offset{offset}" if offset > 0 else ""
    out = os.path.join(HERE, f"current_model_summaries_{tag}_targeted{suffix}.csv")
    log_path = os.path.join(HERE, f"current_model_summaries_{tag}_targeted{suffix}.log")

    load_creds()
    try:
        import truststore; truststore.inject_into_ssl()
    except Exception:
        pass
    import boto3
    brt = boto3.Session(region_name=REGION).client("bedrock-runtime")

    arts = pick_articles(n, offset=offset)
    print(f"model={model}  selected {len(arts)} known error-prone articles "
          f"(max_sev: {arts.max_sev.value_counts().to_dict()})", flush=True)

    import csv
    fieldnames = ["doc_id", "source", "orig_max_severity", "orig_n_errors", "premise",
                  "generative_summary", "model", "prompt_style", "gen_seconds"]
    done_ids = set()
    if os.path.exists(out):
        with open(out, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done_ids.add(row["doc_id"])
        print(f"resuming: {len(done_ids)} articles already saved in {out}, skipping those", flush=True)
    csv_file = open(out, "a" if done_ids else "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if not done_ids:
        writer.writeheader()

    rows = []
    for i, r in enumerate(arts.itertuples()):
        if r.doc_id in done_ids:
            continue
        prompt = THIN_PROMPT.format(article=r.premise)
        t0 = time.time()
        for attempt in range(4):
            try:
                resp = brt.converse(modelId=model, messages=[{"role": "user", "content": [{"text": prompt}]}],
                                     inferenceConfig={"maxTokens": 4000, "temperature": 0.2})
                break
            except Exception as e:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        dt = time.time() - t0
        summary = "".join(c.get("text", "") for c in resp["output"]["message"]["content"]).strip()
        row = {"doc_id": r.doc_id, "source": r.source, "orig_max_severity": r.max_sev,
               "orig_n_errors": r.n_err, "premise": r.premise, "generative_summary": summary,
               "model": model, "prompt_style": "thin", "gen_seconds": round(dt, 1)}
        rows.append(row)
        writer.writerow(row)
        csv_file.flush()
        print(f"  [{i + 1}/{len(arts)}] {r.doc_id} (orig_sev={r.max_sev}) ({dt:.1f}s, {len(summary)} chars)", flush=True)
    csv_file.close()

    import pandas as pd
    df = pd.read_csv(out)
    trunc = (~df.generative_summary.str.strip().str[-1].isin([".", "!", "?", ":", ")", '"'])).sum()
    print(f"\nsaved {len(rows)} summaries -> {out}")
    print(f"QC: {trunc}/{len(rows)} summaries do not end in sentence-final punctuation "
          f"(possible truncation -- check maxTokens if this is high)")


if __name__ == "__main__":
    main()
