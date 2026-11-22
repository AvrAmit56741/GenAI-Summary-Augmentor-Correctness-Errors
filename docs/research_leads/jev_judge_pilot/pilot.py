"""
Pilot: does "Jev" (TypeSafe's hosted calibrated-decision-model judge, typesafe/jev-1.13,
reached here via OpenRouter) work at all on long Hebrew factuality-judging inputs?

We could not independently verify Jev's context length (sources conflict: 32K vs ~262K) or
any Hebrew-language support claim (see docs/research_leads/jev_judge_pilot/README.md once
written) -- so this script does NOT trust any vendor claim. It empirically probes the model
on our own benchmark, the same way the project already treats every other LLM-judge baseline
(see summary_augmentation/pages/metrics/llm_as_a_judge/vanila_zero_shot.py).

Two modes:
    python pilot.py smoke   # 3 hand-picked calls (1 short, 1 long premise, 1 contradiction) --
                             # prints the FULL raw response so we can see what format Jev
                             # actually returns before writing any parsing logic.
    python pilot.py batch   # stratified sample of N pairs from combined.csv, scored and saved
                             # to jev_pilot_results.json

Requires OPENROUTER_API_KEY in the repo-root .env (same convention as OPENAI_API_KEY elsewhere
in this project).
"""
from __future__ import annotations

import csv
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import truststore
from dotenv import load_dotenv

# This network sits behind corporate TLS interception (Cato Networks root CA, not in the
# certifi bundle `requests` ships with) -- use the OS trust store instead of certifi's, same
# fix already used elsewhere in this environment for the identical SSLCertVerificationError.
truststore.inject_into_ssl()

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
load_dotenv(dotenv_path=os.path.join(REPO_ROOT, ".env"), override=True)

BENCHMARK_CSV = os.path.join(
    REPO_ROOT, "thesisdatasets", "train_datasets", "Correctness", "data", "benchmark", "combined.csv"
)
# Jev is a "decisions model", not a chat model -- OpenRouter's normal /chat/completions endpoint
# rejects it (400: "...cannot be used with the chat/completions endpoint. Use the
# /api/alpha/decisions endpoint instead."). This is the real, typed-decision request shape
# documented at https://openrouter.ai/docs/guides/features/decisions -- a "noul" question is a
# binary (true/false) typed answer with a calibrated probability, not a free-text reply.
DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
MODEL = "typesafe/jev-1.14"
BATCH_N = 50
SEED = 0

QUESTION_INSTRUCTIONS = (
    "Is every claim in the generated summary supported by the source document -- i.e., is the "
    "summary factually consistent / faithful? The text may be in Hebrew or English."
)
QUESTION_CRITERIA = {
    "true": "Every claim in the summary is grounded in the source document; no contradictions "
            "or unsupported claims.",
    "false": "At least one claim in the summary contradicts or is not supported by the source "
             "document.",
}


def _headers():
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY not found in environment / .env -- see docs/research_leads/"
                 "jev_judge_pilot/README.md for setup.")
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/MorphEvalHeb",
        "X-Title": "MorphEvalHeb jev-judge pilot",
    }


def _state(premise: str, hypothesis: str) -> str:
    return f"Source document:\n{premise.strip()}\n\nGenerated summary:\n{hypothesis.strip()}"


def _call(premise: str, hypothesis: str, timeout: int = 60) -> requests.Response:
    """One raw Decisions API call. Returns the raw requests.Response -- caller decides what to do with it."""
    body = {
        "model": MODEL,
        "state": _state(premise, hypothesis),
        "questions": {
            "faithful": {
                "type": "noul",
                "instructions": QUESTION_INSTRUCTIONS,
                "criteria": QUESTION_CRITERIA,
            }
        },
    }
    return requests.post(DECISIONS_URL, headers=_headers(), json=body, timeout=timeout)


def _dump_raw(label: str, resp: requests.Response) -> None:
    print(f"\n=== {label} ===")
    print("status_code:", resp.status_code)
    try:
        print("json:")
        print(json.dumps(resp.json(), ensure_ascii=False, indent=2)[:4000])
    except Exception:
        print("raw text:", resp.text[:2000])


def load_rows():
    with open(BENCHMARK_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def smoke():
    rows = [r for r in load_rows() if r["premise_len"]]
    rows_by_len = sorted(rows, key=lambda r: float(r["premise_len"]))

    short_entail = min((r for r in rows if r["label"] == "entailment"), key=lambda r: float(r["premise_len"]))
    long_contra = rows_by_len[-1]  # longest premise in the whole benchmark -- the real context-length probe
    mid_contra = next(r for r in rows if r["label"] == "contradiction" and 800 < float(r["premise_len"]) < 1200)

    for label, row in [("SHORT / entailment", short_entail),
                        ("LONGEST PREMISE IN BENCHMARK / " + row_label(long_contra), long_contra),
                        ("MID-LENGTH / contradiction", mid_contra)]:
        print(f"\npremise_len={row['premise_len']} chars, label={row['label']}, source={row['source']!r}")
        t0 = time.time()
        try:
            resp = _call(row["premise"], row["hypothesis"])
            _dump_raw(label, resp)
        except requests.RequestException as e:
            print(f"=== {label} === REQUEST ERROR: {e}")
        print(f"(call took {time.time() - t0:.1f}s)")


def row_label(row):
    return row["label"]


def batch():
    rows = load_rows()
    rng = random.Random(SEED)

    by_key = {}
    for r in rows:
        by_key.setdefault(r["stratify_key"], []).append(r)
    keys = sorted(by_key)
    sample = []
    i = 0
    while len(sample) < BATCH_N and keys:
        k = keys[i % len(keys)]
        bucket = by_key[k]
        if bucket:
            sample.append(bucket.pop(rng.randrange(len(bucket))))
        else:
            keys.remove(k)
            continue
        i += 1

    print(f"Sampled {len(sample)} pairs (stratified by stratify_key) for Jev pilot scoring.")
    results = []
    for idx, row in enumerate(sample):
        try:
            resp = _call(row["premise"], row["hypothesis"])
            resp.raise_for_status()
            data = resp.json()
            answer = (data.get("questions") or data.get("answers") or {}).get("faithful", data)
            results.append({
                "index": row["index"], "label": row["label"], "source": row["source"],
                "premise_len": row["premise_len"], "raw_response": data,
            })
            print(f"[{idx+1}/{len(sample)}] label={row['label']} len={row['premise_len']} "
                  f"status={resp.status_code} answer={json.dumps(answer, ensure_ascii=False)[:120]}")
        except requests.RequestException as e:
            body = getattr(e.response, "text", "")[:300] if getattr(e, "response", None) is not None else ""
            print(f"[{idx+1}/{len(sample)}] ERROR: {e} {body}")
            results.append({"index": row["index"], "label": row["label"], "error": str(e), "body": body})
        time.sleep(0.2)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jev_pilot_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(results)} results to {out_path}")


FULL_RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jev_full_results.jsonl")
FULL_MAX_WORKERS = 6  # lowered from 10 -- in case the 402 bursts were a concurrency/velocity cap
FULL_MAX_RETRIES = 4


def _score_one_row(row_id, row, attempt_retries=FULL_MAX_RETRIES):
    last_err = None
    for attempt in range(attempt_retries):
        try:
            resp = _call(row["premise"], row["hypothesis"], timeout=90)
            resp.raise_for_status()
            data = resp.json()
            return {
                "row_id": row_id,
                "label": row["label"], "source": row["source"], "method": row["method"],
                "premise_len": len(row["premise"]), "noul": data["answers"]["faithful"]["noul"],
                "usage": data.get("usage", {}),
            }
        except requests.HTTPError as e:
            last_err = e
            # 402s showed up in bursts even with a healthy, well-funded account (verified by hand:
            # single calls and even a 10-call concurrent burst succeeded moments later) -- treat it
            # as a transient spend-rate/velocity limit on this alpha endpoint, not real depletion,
            # and back off much longer than a normal retry before trying again.
            if e.response is not None and e.response.status_code == 402:
                time.sleep(20 * (attempt + 1))
            else:
                time.sleep(1.5 * (attempt + 1))
        except Exception as e:  # noqa: BLE001 -- full run must not die on one bad row
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    return {"row_id": row_id, "label": row["label"], "source": row["source"], "error": str(last_err)}


def full():
    """Score the ENTIRE benchmark (combined.csv, 4,024 pairs) through Jev.

    Threaded (FULL_MAX_WORKERS workers) since sequential would take ~45+ minutes; writes each
    result to jev_full_results.jsonl as soon as it completes (not one big json.dump at the end)
    so progress survives an interruption. Resumable: row_id is just the row's position in
    combined.csv (stable as long as the file itself doesn't change). Only rows that already
    SUCCEEDED (have a "noul" field) are treated as done -- a failed row (error -- e.g. the 402s
    seen earlier) is deliberately NOT counted as done, so a re-run retries it instead of silently
    giving up on it forever. This means the output file can contain duplicate row_ids (an old
    failure line plus a newer success line) -- analysis should prefer the successful entry.
    """
    rows = load_rows()
    done_ids = set()
    if os.path.exists(FULL_RESULTS_PATH):
        with open(FULL_RESULTS_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    if "noul" in rec:  # only successes count as done -- failures get retried
                        done_ids.add(rec["row_id"])
        print(f"Resuming: {len(done_ids)} rows already successfully scored in {FULL_RESULTS_PATH}")

    todo = [(i, r) for i, r in enumerate(rows) if i not in done_ids]
    print(f"Scoring {len(todo)} of {len(rows)} total pairs with {FULL_MAX_WORKERS} workers...")

    lock = threading.Lock()
    completed = [0]
    t0 = time.time()
    with open(FULL_RESULTS_PATH, "a", encoding="utf-8") as out_f, \
            ThreadPoolExecutor(max_workers=FULL_MAX_WORKERS) as pool:
        futures = {pool.submit(_score_one_row, row_id, row): row_id for row_id, row in todo}
        for fut in as_completed(futures):
            result = fut.result()
            with lock:
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()
                completed[0] += 1
                if completed[0] % 50 == 0 or completed[0] == len(todo):
                    elapsed = time.time() - t0
                    rate = completed[0] / elapsed
                    eta = (len(todo) - completed[0]) / rate if rate > 0 else float("nan")
                    print(f"[{completed[0]}/{len(todo)}] elapsed={elapsed:.0f}s "
                          f"rate={rate:.2f}/s eta={eta:.0f}s")

    print(f"\nDone. Results in {FULL_RESULTS_PATH}")


SOURCES = ["Israel_Hayom", "Wikipedia", "Weizmann", "Knesset", "Bagatz", ""]  # "" = Natural (mFACT-He)


def diverse():
    """The first batch (BATCH_N=50 stratified by stratify_key) turned out 48/50 Natural ("")
    and only 1 Israel_Hayom + 1 Wikipedia -- it barely touched the five native-Hebrew Injected
    genres (news, encyclopedic, scientific, parliamentary, judicial), which are also the longer,
    harder documents (median ~1,343 chars vs. Natural's much shorter translated summaries).
    This samples explicitly per source x label, biased toward the longer end within each source,
    so every genre and the harder/longer end of the length distribution actually gets tested."""
    # premise_len is only populated for Natural rows ("") -- Injected rows (the five native-Hebrew
    # MAFAT sources) leave it blank and use a separate len_group bucket instead, so every Injected
    # source looked empty when filtering on that column. Compute length from len(premise) instead.
    rows = [r for r in load_rows() if r["premise"]]
    rng = random.Random(SEED + 1)

    sample = []
    for source in SOURCES:
        for label in ("entailment", "contradiction"):
            bucket = [r for r in rows if r["source"] == source and r["label"] == label]
            if not bucket:
                print(f"(no rows for source={source!r} label={label!r})")
                continue
            bucket.sort(key=lambda r: len(r["premise"]), reverse=True)
            top_quartile = bucket[: max(1, len(bucket) // 4)]
            sample.extend(rng.sample(top_quartile, k=min(3, len(top_quartile))))

    print(f"Sampled {len(sample)} pairs, explicitly covering sources {SOURCES}, "
          f"biased to the longer quartile within each source x label bucket.")
    results = []
    for idx, row in enumerate(sample):
        try:
            resp = _call(row["premise"], row["hypothesis"])
            resp.raise_for_status()
            data = resp.json()
            answer = data.get("answers", {}).get("faithful", data)
            results.append({
                "index": row["index"], "label": row["label"], "source": row["source"],
                "premise_len": len(row["premise"]), "raw_response": data,
            })
            print(f"[{idx+1}/{len(sample)}] source={row['source']!r:14} label={row['label']:12} "
                  f"len={len(row['premise']):>7} status={resp.status_code} "
                  f"answer={json.dumps(answer, ensure_ascii=False)}")
        except requests.RequestException as e:
            body = getattr(e.response, "text", "")[:300] if getattr(e, "response", None) is not None else ""
            print(f"[{idx+1}/{len(sample)}] source={row['source']!r} ERROR: {e} {body}")
            results.append({"index": row["index"], "label": row["label"], "source": row["source"],
                             "error": str(e), "body": body})
        time.sleep(0.2)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jev_diverse_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(results)} results to {out_path}")


def canary():
    """Truncation probe: build a synthetic document far longer than anything in the real
    benchmark (concatenate several real long articles), then append ONE final sentence with a
    unique, never-mentioned-elsewhere fact (an invented number). Two hypotheses: one that states
    that exact fact (should score HIGH faithful) and one that states a different number for the
    same fact (should score LOW). If Jev is silently truncating before the true end, it has no
    basis to tell these two apart and should score them similarly (near 0.5, or randomly) --
    if it reliably separates them, that is real evidence it processed all the way to the end."""
    rows = [r for r in load_rows() if r["premise_len"]]
    rows.sort(key=lambda r: float(r["premise_len"]), reverse=True)
    long_articles = [r["premise"].strip() for r in rows[:6]]  # 6 longest real articles, concatenated
    padded = "\n\n".join(long_articles)
    canary_fact = (
        "לבסוף, דובר החברה מסר כי התקציב הסופי שאושר לפרויקט המיוחד, שלא פורט בשום מקום אחר "
        "בכתבה, הסתכם ב-837,214 שקלים חדשים בדיוק."
    )
    full_doc = padded + "\n\n" + canary_fact
    print(f"Synthetic canary document: {len(full_doc)} chars "
          f"(~{len(full_doc) / max(1, len(long_articles[0])):.1f}x the single longest real article; "
          f"real benchmark max is 2,935 chars).")

    hyp_true = "התקציב הסופי שאושר לפרויקט המיוחד היה 837,214 שקלים חדשים."
    hyp_false = "התקציב הסופי שאושר לפרויקט המיוחד היה 412,900 שקלים חדשים."

    for label, hyp in [("TRUE fact (matches canary sentence)", hyp_true),
                        ("FALSE fact (contradicts canary sentence)", hyp_false)]:
        t0 = time.time()
        try:
            resp = _call(full_doc, hyp)
            resp.raise_for_status()
            data = resp.json()
            print(f"\n=== {label} ===")
            print("input_tokens:", data.get("usage", {}).get("input_tokens"))
            print("answer:", json.dumps(data.get("answers", {}).get("faithful", data), ensure_ascii=False))
        except requests.RequestException as e:
            body = getattr(e.response, "text", "")[:300] if getattr(e, "response", None) is not None else ""
            print(f"=== {label} === ERROR: {e} {body}")
        print(f"(call took {time.time() - t0:.1f}s)")


def diag():
    """Masked sanity check that OPENROUTER_API_KEY actually loaded from .env -- never prints the
    real value, only its length and first/last couple of characters, to debug 401s."""
    v = os.environ.get("OPENROUTER_API_KEY")
    if v is None:
        print("OPENROUTER_API_KEY: NOT SET in process environment after load_dotenv")
        return
    print(f"OPENROUTER_API_KEY: len={len(v)} repr_head={v[:6]!r} repr_tail={v[-4:]!r} "
          f"has_leading/trailing_whitespace={v != v.strip()!r} has_quotes={('\"' in v or chr(39) in v)!r}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if mode == "smoke":
        smoke()
    elif mode == "batch":
        batch()
    elif mode == "diag":
        diag()
    elif mode == "diverse":
        diverse()
    elif mode == "canary":
        canary()
    elif mode == "full":
        full()
    else:
        sys.exit(f"unknown mode {mode!r} -- use 'smoke', 'batch', 'diverse', 'canary', 'full', or 'diag'")
