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


def pick_new_articles(n=50, seed=42, also_exclude_csvs=()):
    """A disjoint pool: articles that NO model (old local/Bedrock rounds OR the new OpenRouter
    Qwen/DeepSeek rounds) has ever had summarized, since pick_articles()'s 198-article
    thesisdatasets/error_finder/ pool is now fully exhausted (every doc_id in it has a generated
    summary from at least one model). Source: the mFACT-He portion of the project's own NLI
    benchmark (thesisdatasets/train_datasets/Correctness/data/benchmark/combined.csv), which has
    doc_id/source columns identical in spirit to the error_finder pool but was built for a
    different purpose (entailment pairs) and has only ever been used for that -- never fed through
    any summarization model. 303 unique doc_ids there, only 10 overlap with the exhausted pool,
    leaving 293 genuinely untouched candidates. A fixed seed makes the 50-article sample
    reproducible; there's no prior severity signal for these (never judged), so selection is a
    random sample rather than the historical "hardest known" ranking.

    `also_exclude_csvs`: paths to prior pick_new_articles() output CSVs (e.g. the Qwen/DeepSeek
    newpool rounds) whose doc_ids should also be excluded, so a later model's "new pool" round
    draws from a genuinely fresh subset rather than re-sampling the same 50 articles."""
    import pandas as pd
    files = sorted(glob.glob(os.path.join(REPO, "thesisdatasets/error_finder/*.csv")))
    e = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    e = e.dropna(subset=["premise", "doc_id", "severity"])
    exhausted_doc_ids = set(e.doc_id.unique())
    for p in also_exclude_csvs:
        exhausted_doc_ids |= set(pd.read_csv(p).doc_id.astype(str).unique())

    bench = pd.read_csv(os.path.join(
        REPO, "thesisdatasets/train_datasets/Correctness/data/benchmark/combined.csv"))
    bench = bench.dropna(subset=["doc_id", "premise", "source"])
    by_doc = bench.groupby("doc_id").agg(premise=("premise", "first"),
                                          source=("source", "first")).reset_index()
    by_doc = by_doc[~by_doc.doc_id.astype(str).isin(exhausted_doc_ids)]
    return by_doc.sample(n=n, random_state=seed).reset_index(drop=True)


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
                # reasoning disabled: on a reasoning model (confirmed with qwen/qwen3.8-flash and
                # moonshotai/kimi-k2.6), hidden "thinking" tokens are drawn from the same
                # max_tokens budget as the visible answer and can silently consume the ENTIRE
                # budget, leaving empty content even at finish_reason="length" -- disabling it is
                # the only reliable fix (increasing max_tokens alone does NOT help, confirmed
                # empirically: at max_tokens=6000 with reasoning enabled, all 6000 tokens went to
                # reasoning and content was 0 chars). Models that aren't reasoning models ignore
                # this flag. max_tokens=6000 (raised from 3000 after Kimi K2.6's smoke test showed
                # some article/summary pairs need up to ~3,700 completion tokens even with
                # reasoning off); timeout=180 (raised from 120 after Kimi took up to 127s on a
                # single call).
                json={"model": model, "messages": [{"role": "user", "content": prompt}],
                      "temperature": 0.2, "max_tokens": 6000, "reasoning": {"enabled": False}},
                timeout=180,
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
    offset_arg = sys.argv[3] if len(sys.argv) > 3 else "0"
    use_new_pool = offset_arg.startswith("new")
    # "new" = round 1 of the disjoint pool (suffix "_newpool"); "new2", "new3", ... = later rounds
    # once round 1 is exhausted for this or other models (suffix "_newpool_round{k}"), each
    # excluding every doc_id any model's any round of this disjoint pool has already used.
    new_round = int(offset_arg[3:]) if use_new_pool and len(offset_arg) > 3 else 1
    offset = 0 if use_new_pool else int(offset_arg)
    tag = re.sub(r"[^A-Za-z0-9_-]", "_", model)
    if use_new_pool:
        suffix = "_newpool" if new_round == 1 else f"_newpool_round{new_round}"
    else:
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

    if use_new_pool:
        # Exclude doc_ids any model's any round of the disjoint pool already used (this model's
        # own earlier rounds included), so each new round draws genuinely fresh articles rather
        # than re-sampling ones already tested. `out` itself is excluded from this list so a
        # crashed/resumed run for the same model+round still lines up with its own JSONL.
        prior_newpool_csvs = [f for f in glob.glob(os.path.join(HERE, "current_model_summaries_*_targeted_newpool*.csv"))
                              if f != out]
        arts = pick_new_articles(n, seed=42 + new_round, also_exclude_csvs=prior_newpool_csvs)
        print(f"model={model}  selected {len(arts)} never-summarized articles from the new pool "
              f"round {new_round} (excluding {len(prior_newpool_csvs)} prior new-pool round files; "
              f"sources: {arts.source.value_counts().to_dict()})", flush=True)
    else:
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
            row = {"doc_id": r.doc_id, "source": r.source,
                   "orig_max_severity": (None if use_new_pool else r.max_sev),
                   "orig_n_errors": (None if use_new_pool else r.n_err),
                   "premise": r.premise, "generative_summary": summary,
                   "model": model, "prompt_style": "thin", "gen_seconds": round(dt, 1)}
            jf.write(json.dumps(row, ensure_ascii=False) + "\n")
            jf.flush()
            sev_note = "n/a (new pool)" if use_new_pool else r.max_sev
            print(f"  [{i + 1}/{len(arts)}] {r.doc_id} (orig_sev={sev_note}) ({dt:.1f}s, {len(summary)} chars)",
                  flush=True)

    import pandas as pd
    rows = [json.loads(l) for l in open(jsonl_path, encoding="utf-8") if l.strip()]
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"\nsaved {len(rows)} summaries -> {out}")
    qc_checks(df)


if __name__ == "__main__":
    main()
