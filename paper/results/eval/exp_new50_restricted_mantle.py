"""New, never-before-used 50-article pool (Luna / bedrock-mantle path). See
exp_new50_restricted_bedrock.py for the full rationale and new50_candidates.csv for the article list.

    python exp_new50_restricted_mantle.py openai.gpt-5.6-luna

Output: current_model_summaries_<tag>_new50_restricted.csv
"""
import csv
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_length_corrected_val20_mantle import CORRECTED_PROMPT, load_creds  # noqa: E402

MODEL_DEFAULT = "openai.gpt-5.6-luna"
REGION = "us-east-1"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
BENCH = os.path.join(REPO, "thesisdatasets/train_datasets/Correctness/data/benchmark/combined.csv")
CANDIDATES = os.path.join(HERE, "new50_candidates.csv")


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
    model = sys.argv[1] if len(sys.argv) > 1 else MODEL_DEFAULT
    tag = re.sub(r"[^A-Za-z0-9_-]", "_", model)
    out = os.path.join(HERE, f"current_model_summaries_{tag}_new50_restricted.csv")

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
                resp = client.responses.create(model=model, input=prompt, max_output_tokens=1000)
                break
            except Exception as e:
                if attempt == 3:
                    raise
                print(f"  retry {attempt} after error: {e}", flush=True)
                time.sleep(2 ** attempt)
        dt = time.time() - t0
        summary = (resp.output_text or "").strip()
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
