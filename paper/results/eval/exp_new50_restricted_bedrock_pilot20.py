"""Pilot run of a candidate 6th model against the first 20 (of 50) never-before-used articles,
using the same validated "restricted" prompt (8-12 sentences, ~150-200 words) as
exp_new50_restricted_bedrock.py. Used to sanity-check a new model (structure/length compliance +
quick Hebrew-morphology error scan) before committing to a full 50-article run.

    python exp_new50_restricted_bedrock_pilot20.py amazon.nova-lite-v1:0

Output: current_model_summaries_<tag>_new50_restricted_pilot20.csv
"""
import csv
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_length_corrected_val20_bedrock import CORRECTED_PROMPT, load_creds  # noqa: E402
from exp_new50_restricted_bedrock import pick_articles  # noqa: E402

REGION = "us-east-1"
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    model = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    tag = re.sub(r"[^A-Za-z0-9_-]", "_", model)
    out = os.path.join(HERE, f"current_model_summaries_{tag}_new50_restricted_pilot{n}.csv")

    load_creds()
    try:
        import truststore; truststore.inject_into_ssl()
    except Exception:
        pass
    import boto3
    brt = boto3.Session(region_name=REGION).client("bedrock-runtime")

    arts = pick_articles().head(n)
    print(f"model={model}  pilot run on {len(arts)} never-used articles "
          f"(source breakdown: {arts.source.value_counts().to_dict()})", flush=True)

    fieldnames = ["doc_id", "source", "hardness_score", "premise",
                  "generative_summary", "model", "prompt_style", "gen_seconds"]
    csv_file = open(out, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    for i, r in enumerate(arts.itertuples()):
        prompt = CORRECTED_PROMPT.format(article=r.premise)
        t0 = time.time()
        for attempt in range(4):
            try:
                resp = brt.converse(modelId=model, messages=[{"role": "user", "content": [{"text": prompt}]}],
                                     inferenceConfig={"maxTokens": 1000, "temperature": 0.2})
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        dt = time.time() - t0
        summary = "".join(c.get("text", "") for c in resp["output"]["message"]["content"]).strip()
        row = {"doc_id": r.doc_id, "source": r.source, "hardness_score": r.score, "premise": r.premise,
               "generative_summary": summary, "model": model,
               "prompt_style": "restricted_8to12sentences", "gen_seconds": round(dt, 1)}
        writer.writerow(row)
        csv_file.flush()
        print(f"  [{i + 1}/{len(arts)}] {r.doc_id} ({r.source}) ({dt:.1f}s, {len(summary)} chars, "
              f"{len(summary.split())} words)", flush=True)
    csv_file.close()

    import pandas as pd
    df = pd.read_csv(out)
    comp = (df.generative_summary.str.len() / df.premise.str.len()).mean()
    print(f"\nsaved {len(df)} summaries -> {out}")
    print(f"mean compression ratio: {comp*100:.1f}% (target: ~16.7%, matching the real MAFAT baseline)")


if __name__ == "__main__":
    main()
