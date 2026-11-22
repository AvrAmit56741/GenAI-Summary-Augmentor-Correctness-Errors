"""Targeted current-model check via OpenRouter (candidates: Qwen3.8 Flash, DeepSeek V4 Pro 0813,
etc. -- newer/more-popular models than the existing GLM-4.7-Flash/Ministral-3-14B/DictaLM3 rounds).

Same methodology as exp_current_model_summaries_targeted.py / exp_current_model_summaries_mantle_targeted.py:
the 50 (or N) most error-prone known articles (by max historical severity, then error count, from
thesisdatasets/error_finder/*.csv) and the same minimal ("thin") Hebrew prompt, so results are directly
comparable to those earlier rounds.

One deliberate addition over the historical THIN_PROMPT: a one-line instruction forbidding markdown
formatting (headers/bullets) and asking for flowing narrative prose. This was NOT needed for the
earlier DictaLM3/GLM-4.7-Flash/Ministral-3-14B rounds, which all produced narrative prose by default
-- but a 10-article qwen/qwen3.8-flash smoke test (2026-11-22) showed it defaults to a structured
markdown "report" (bold headers, bullet points) instead, sometimes expanding a 762-char source into a
2,181-char report (2.86x) and getting truncated mid-bullet at max_tokens=3000 (2/10 cases) -- a style
mismatch with the flowing-narrative-text the error-finder taxonomy prompt and its confirmed-error
examples are built around. The instruction is added for every model called through this script, not
just Qwen, so results stay comparable to each other even though the literal prompt text now diverges
slightly from the earlier local/Bedrock rounds.

    python exp_current_model_summaries_openrouter_targeted.py <openrouter-model-id> [N] [offset]
    python exp_current_model_summaries_openrouter_targeted.py qwen/qwen3.8-flash 10
    python exp_current_model_summaries_openrouter_targeted.py deepseek/deepseek-v4-pro-0813 10

Requires OPENROUTER_API_KEY in the repo-root .env (same convention as everywhere else this session).
"""
import glob
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

THIN_PROMPT = ("סכם את הטקסט הבא בעברית, סיכום מפורט ועשיר בפרטים ובעובדות, לא סיכום תמציתי או מופשט. "
               "כתוב טקסט רציף וזורם של פסקאות, ללא כותרות, ללא רשימות, וללא עיצוב Markdown "
               "(כגון **, #, או תבליטים), וללא משפט פתיחה המקדים את הסיכום -- התחל ישר בתוכן הסיכום:"
               "\n\n{article}")

HEBREW_RE = re.compile(r"[֐-׿]")
LATIN_RE = re.compile(r"[A-Za-z]")


def pick_articles(n=50, offset=0):
    """Identical selection logic to exp_current_model_summaries_targeted.py / _mantle_targeted.py,
    so this round is directly comparable to the GLM-4.7-Flash / Ministral-3-14B / DictaLM3 rounds."""
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


def qc_checks(df):
    """Smoke-test QC: structure, length ratio, Hebrew-only, truncation -- everything that should be
    verified before trusting a new model's summaries enough to run the full error-finder pass."""
    print("\n=== QC checks ===")
    empties = (df.generative_summary.str.strip() == "").sum()
    print(f"empty outputs: {empties}/{len(df)}")

    trunc = (~df.generative_summary.str.strip().str[-1].isin([".", "!", "?", ":", ")", '"', "”"])).sum()
    print(f"possible truncation (doesn't end in sentence-final punctuation): {trunc}/{len(df)}")

    ratios = df.generative_summary.str.len() / df.premise.str.len()
    print(f"length ratio (summary_chars/article_chars): min={ratios.min():.2f} "
          f"median={ratios.median():.2f} max={ratios.max():.2f}")
    degenerate = ((ratios < 0.03) | (ratios > 0.9)).sum()
    print(f"degenerate length ratio (<3% or >90% of article length): {degenerate}/{len(df)}")

    def hebrew_fraction(text):
        heb = len(HEBREW_RE.findall(text))
        lat = len(LATIN_RE.findall(text))
        total = heb + lat
        return heb / total if total > 0 else 1.0

    heb_frac = df.generative_summary.apply(hebrew_fraction)
    print(f"Hebrew-character fraction (of Hebrew+Latin chars): min={heb_frac.min():.2f} "
          f"median={heb_frac.median():.2f}")
    non_hebrew = (heb_frac < 0.8).sum()
    print(f"outputs with <80% Hebrew-vs-Latin characters (possible English drift): {non_hebrew}/{len(df)}")

    preamble = df.generative_summary.str.strip().str.match(r"^(הנה|סיכום|להלן|Here|Sure)").sum()
    print(f"outputs starting with a likely meta-preamble (\"הנה/סיכום/להלן/...\"): {preamble}/{len(df)}")


def _call_one(key, model, prompt):
    import requests
    last_err = None
    for attempt in range(6):
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                # reasoning disabled: on a reasoning model (confirmed with qwen/qwen3.8-flash),
                # hidden "thinking" tokens are drawn from the same max_tokens budget as the
                # visible answer and can silently consume the ENTIRE budget, leaving empty
                # content even at finish_reason="length" -- disabling it is the only reliable
                # fix (increasing max_tokens alone does NOT help, confirmed empirically: at
                # max_tokens=6000 with reasoning enabled, all 6000 tokens went to reasoning and
                # content was 0 chars). Models that aren't reasoning models ignore this flag.
                json={"model": model, "messages": [{"role": "user", "content": prompt}],
                      "temperature": 0.2, "max_tokens": 3000, "reasoning": {"enabled": False}},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            return (data["choices"][0]["message"]["content"] or "").strip()
        except Exception as e:  # noqa: BLE001 -- a transient network blip must not kill the whole batch
            last_err = e
            print(f"  retry {attempt} after error: {e}", flush=True)
            time.sleep(min(2 ** attempt, 30))
    raise last_err


def main():
    model = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    offset = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    tag = re.sub(r"[^A-Za-z0-9_-]", "_", model)
    suffix = f"_offset{offset}" if offset > 0 else ""
    out = os.path.join(HERE, f"current_model_summaries_{tag}_targeted{suffix}.csv")

    try:
        import truststore; truststore.inject_into_ssl()
    except Exception:
        pass
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(REPO, ".env"), override=True)

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY not found in environment / .env")

    arts = pick_articles(n, offset=offset)
    print(f"model={model}  selected {len(arts)} known error-prone articles "
          f"(max_sev: {arts.max_sev.value_counts().to_dict()})", flush=True)

    # Resumable: write each row as soon as it completes (one JSON object per line) rather than one
    # big CSV at the end -- a crash (e.g. a transient connection reset, which has happened in
    # practice) must not lose already-completed work. A rerun with the same model/n/offset skips
    # rows already present here.
    import json
    jsonl_path = out.replace(".csv", "_raw.jsonl")
    done_doc_ids = set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    done_doc_ids.add(json.loads(line)["doc_id"])
        print(f"Resuming: {len(done_doc_ids)} rows already done in {jsonl_path}", flush=True)

    with open(jsonl_path, "a", encoding="utf-8") as jf:
        for i, r in enumerate(arts.itertuples()):
            if r.doc_id in done_doc_ids:
                print(f"  [{i + 1}/{len(arts)}] {r.doc_id} (already done, skipping)", flush=True)
                continue
            prompt = THIN_PROMPT.format(article=r.premise)
            t0 = time.time()
            summary = _call_one(key, model, prompt)
            dt = time.time() - t0
            row = {"doc_id": r.doc_id, "source": r.source, "orig_max_severity": r.max_sev,
                   "orig_n_errors": r.n_err, "premise": r.premise, "generative_summary": summary,
                   "model": model, "prompt_style": "thin", "gen_seconds": round(dt, 1)}
            jf.write(json.dumps(row, ensure_ascii=False) + "\n")
            jf.flush()
            print(f"  [{i + 1}/{len(arts)}] {r.doc_id} (orig_sev={r.max_sev}) ({dt:.1f}s, {len(summary)} chars)",
                  flush=True)

    import pandas as pd
    rows = [json.loads(l) for l in open(jsonl_path, encoding="utf-8") if l.strip()]
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"\nsaved {len(rows)} summaries -> {out}")
    qc_checks(df)


if __name__ == "__main__":
    main()
