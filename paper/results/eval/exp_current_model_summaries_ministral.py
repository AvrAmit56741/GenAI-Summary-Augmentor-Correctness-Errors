"""Current-model check, model 3: Ministral 3 14B via AWS Bedrock (co-author objection follow-up).

Same research question as exp_current_model_summaries.py (DictaLM 3.0 1.7B) -- do current models still
produce Hebrew-morphology-caused factual errors -- but with a different, newer (~Nov/Dec 2025),
smaller (<=20B, per author's size preference), non-Hebrew-specialized open-weight model, to broaden the
evidence beyond one Hebrew-native model. Model: mistral.ministral-3-14b-instruct (Mistral AI), accessed
via AWS Bedrock Converse API (confirmed invokable on this account).

Uses the EXACT English two-step summary-generation prompt (verbatim from
summary_augmentation/summary_generator_ollama.py / summary_generator_close_llm.py, the same prompt
used for the GPT-3.5 and Ollama-model entries in the original error-finder data), so this is directly
comparable to those entries, not just to the DictaLM-prompt-style batch.

    python exp_current_model_summaries_ministral.py   -> current_model_summaries_ministral3.csv

Credentials: AWS env vars loaded from BEDROCK_CREDS_FILE (default: the urlf-classification experiments file).
"""
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eval as R  # noqa: E402

CREDS = os.environ.get("BEDROCK_CREDS_FILE", os.path.expanduser(
    "~/PycharmProjects/urlf-classification/experiments/bedrock_classifier/bedrock_aws_creds.local.env"))
REGION = "us-east-1"
MODEL = "mistral.ministral-3-14b-instruct"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "current_model_summaries_ministral3.csv")
N_PER_SOURCE = {"Israel_Hayom": 8, "Wikipedia": 6, "Weizmann": 4, "Knesset": 4, "Bagatz": 0}

SYSTEM = """You are an expert at creating detailed, fact-rich summaries in Hebrew.
Your task is to transform the source article into a single, comprehensive narrative summary that incorporates all key information.

### Your Two-Step Process to Follow ###

**Step 1: Internal Fact Collection (Analysis Only)**
First, silently read the entire article and identify ALL the key facts. This includes names, dates, statistics, locations, key events, and main conclusions.
Create a complete mental checklist of these facts.
**Do not write this list in the output.** This step is for your internal analysis only.

**Step 2: Narrative Summary Writing (The Final Output)**
Now, using the complete checklist of facts you just created, write a single, flowing, and detailed summary.
Your mission is to skillfully weave ALL the facts from your list into a coherent Hebrew narrative.
 The final text should be informative, easy to read, and feel like a complete, self-contained article.

### Mandatory Rules for the Final Output (Step 2) ###

1.  **Length:** The final summary MUST be **at least 15 sentences long**. This is a strict minimum.
2.  **Factual Completeness:** The summary MUST incorporate all of the important facts you identified in Step 1. Do not omit key details.
3.  **Style:** The output MUST be a flowing narrative, NOT a list of disconnected points.
4.  **Format:** The entire output must be a single, valid JSON object with the key "article_summary"."""

USER_TMPL = """Source article for processing:
<article>
{source_article}
</article>

---
Now, execute the two-step process and provide the final narrative summary in the required JSON format."""


def load_creds():
    for line in open(CREDS):
        m = re.match(r"\s*(?:export\s+)?([A-Za-z_]+)=(.*)", line)
        if m:
            os.environ[m.group(1)] = m.group(2).strip().strip("\"'")


def sample_articles(seed=31):
    import pandas as pd
    d = R.load_bench()
    m = d[d.part == "MAFAT-Aug"][["doc_id", "source", "premise"]].drop_duplicates("doc_id")
    rows = []
    for src, n in N_PER_SOURCE.items():
        if n == 0:
            continue
        g = m[m.source == src]
        rows.append(g.sample(min(n, len(g)), random_state=seed))
    return pd.concat(rows).reset_index(drop=True)


def extract_summary(text):
    """Best-effort JSON extraction; falls back to raw text if the model didn't wrap it in JSON."""
    m = re.search(r'\{[^{}]*"article_summary"\s*:\s*"((?:[^"\\]|\\.)*)"[^{}]*\}', text, re.S)
    if m:
        try:
            return json.loads(m.group(0))["article_summary"]
        except Exception:
            return m.group(1).encode().decode("unicode_escape")
    try:
        return json.loads(text)["article_summary"]
    except Exception:
        return text.strip()


def main():
    load_creds()
    try:
        import truststore; truststore.inject_into_ssl()
    except Exception:
        pass
    import boto3
    brt = boto3.Session(region_name=REGION).client("bedrock-runtime")

    arts = sample_articles()
    print(f"sampled {len(arts)} articles: {arts.source.value_counts().to_dict()}", flush=True)

    rows = []
    for i, r in enumerate(arts.itertuples()):
        user_msg = USER_TMPL.format(source_article=r.premise)
        t0 = time.time()
        resp = brt.converse(
            modelId=MODEL,
            system=[{"text": SYSTEM}],
            messages=[{"role": "user", "content": [{"text": user_msg}]}],
            inferenceConfig={"maxTokens": 2000, "temperature": 0.2},
        )
        dt = time.time() - t0
        raw = "".join(c.get("text", "") for c in resp["output"]["message"]["content"])
        summary = extract_summary(raw)
        rows.append({"doc_id": r.doc_id, "source": r.source, "premise": r.premise,
                     "generative_summary": summary, "raw_response": raw, "model": MODEL,
                     "gen_seconds": round(dt, 1)})
        print(f"  [{i + 1}/{len(arts)}] {r.source} {r.doc_id} ({dt:.1f}s, {len(summary)} chars)", flush=True)

    import pandas as pd
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nsaved {len(rows)} summaries -> {OUT}")


if __name__ == "__main__":
    main()
