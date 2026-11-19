"""Second held-out pool of 50 never-before-used articles (see exp_build_next50_candidates.py),
run with the same validated "restricted" prompt (8-12 sentences, ~150-200 words) as
exp_new50_restricted_bedrock.py. Used to get a larger sample for models that showed a low error
rate (or none) on the first 50-article pool.

    python exp_next50_restricted_bedrock.py zai.glm-4.7-flash
    python exp_next50_restricted_bedrock.py mistral.ministral-3-14b-instruct
    python exp_next50_restricted_bedrock.py qwen.qwen3-32b-v1:0
    python exp_next50_restricted_bedrock.py amazon.nova-lite-v1:0

Output: current_model_summaries_<tag>_next50_restricted.csv
"""
import csv
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_length_corrected_val20_bedrock import CORRECTED_PROMPT, load_creds  # noqa: E402

REGION = "us-east-1"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
BENCH = os.path.join(REPO, "thesisdatasets/train_datasets/Correctness/data/benchmark/combined.csv")
CANDIDATES = os.path.join(HERE, "next50_candidates.csv")


def pick_articles():
    import pandas as pd
    cand = pd.read_csv(CANDIDATES)
    bench = pd.read_csv(BENCH)
    bench["premise"] = bench["premise"].astype(str)
    bench_uniq = bench.drop_duplicates(subset=["doc_id"])[["doc_id", "premise", "source"]]
    merged = cand.merge(bench_uniq, on="doc_id", how="left", suffixes=("_cand", ""))
    missing = merged[merged["premise"].isna()]
    if len(missing):
        raise RuntimeError(f"{len(missing)} candidate doc_ids not found in combined.csv: {missing.doc_id.tolist()}")
    return merged.reset_index(drop=True)


def main():
    model = sys.argv[1]
    tag = re.sub(r"[^A-Za-z0-9_-]", "_", model)
    out = os.path.join(HERE, f"current_model_summaries_{tag}_next50_restricted.csv")

    load_creds()
    try:
        import truststore; truststore.inject_into_ssl()
    except Exception:
        pass
    import boto3
    brt = boto3.Session(region_name=REGION).client("bedrock-runtime")

    arts = pick_articles()
    print(f"model={model}  selected {len(arts)} NEW never-used articles "
          f"(source breakdown: {arts.source.value_counts().to_dict()})", flush=True)

    fieldnames = ["doc_id", "source", "hardness_score", "premise",
                  "generative_summary", "model", "prompt_style", "gen_seconds"]
    done_ids = set()
    if os.path.exists(out):
        with open(out, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done_ids.add(row["doc_id"])
        print(f"resuming: {len(done_ids)} articles already saved, skipping those", flush=True)
    csv_file = open(out, "a" if done_ids else "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if not done_ids:
        writer.writeheader()

    for i, r in enumerate(arts.itertuples()):
        if r.doc_id in done_ids:
            continue
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
