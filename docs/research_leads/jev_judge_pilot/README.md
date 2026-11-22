# Pilot: Jev (TypeSafe decisions API) as a Hebrew factuality judge

Status: complete, full benchmark scored (2026-11-22), N=4,024. Added to the paper as a new
baseline (`paper/overleaf/latex/acl_latex.tex`, Table 1 `tab:main_results`, Table 2
`tab:full_results`, Appendix E `app:jev`) and to the thesis
(`../../../thesis/research/jev_judge_pilot.md`).

## What this tests

Research question: can a *decision-model* architecture -- a calibrated classifier over a frozen
reader, returning a typed probability in one forward pass, with no free text to generate or parse
-- judge Hebrew summary factuality, and how does it compare to the generative LLM judges
(G-EVAL, vanilla GPT-4o-mini) and trained checkers already in the paper? This came out of a user
question about "decision models like Jev" (the TypeSafe product referenced, under the name "Jev",
in our own earlier literature-review notes on the unattributed LLM-AggreFact leaderboard
entries -- see `paper/research/notes/MorphEvalHeb literature review update/
factuality_metrics_general.md`). A separate product, "Laya" (`convaiinnovations/laya`), was
investigated first and ruled out: its real released checkpoints are a 421M-parameter
ModernBERT-large encoder with a 512-token context window, far too short to read a full article,
with no confirmed Hebrew support. Jev, reached via OpenRouter, turned out to have a real,
documented, usable API and a context window large enough to matter -- this pilot is the result of
actually testing it, not trusting either product's marketing.

## Method (`pilot.py`)

Model: `typesafe/jev-1.14` (resolves at call time to snapshot `typesafe/jev-1.14-20261014`),
provider TypeSafe, reached through OpenRouter's **alpha Decisions API**
(`POST https://openrouter.ai/api/alpha/decisions`) -- not the standard chat-completions endpoint,
which rejects this model outright (`400: "... is a decisions model and cannot be used with the
chat/completions endpoint"`). Every call sends one binary (`"noul"`) typed question over a
`state` consisting of the **full, untruncated** source article and summary (no chunking):

```json
{
  "model": "typesafe/jev-1.14",
  "state": "Source document:\n<premise>\n\nGenerated summary:\n<hypothesis>",
  "questions": {
    "faithful": {
      "type": "noul",
      "instructions": "Is every claim in the generated summary supported by the source document -- i.e., is the summary factually consistent / faithful? The text may be in Hebrew or English.",
      "criteria": {
        "true": "Every claim in the summary is grounded in the source document; no contradictions or unsupported claims.",
        "false": "At least one claim in the summary contradicts or is not supported by the source document."
      }
    }
  }
}
```

The response is a single calibrated probability, e.g. `{"answers": {"faithful": {"noul": 0.11}}}`.

Scored all 4,024 pairs of `thesisdatasets/train_datasets/Correctness/data/benchmark/combined.csv`
(the exact file backing the paper's benchmark) with `python pilot.py full` -- threaded (6 workers),
results written incrementally to `jev_full_results.jsonl` so progress survives interruption, and
resumable (a rerun only retries rows that previously failed, never rows that already succeeded).
Total cost for all 4,024 calls: **$0.62**.

The resulting `noul` probabilities were merged into `paper/results/eval/scores.csv` (the project's
canonical per-metric score table) as a new `Jev` column, so the headline numbers below come from
the exact same evaluation protocol (`paper/results/eval/analyze.py`) as every other baseline: 5-fold
article-grouped cross-validated balanced accuracy, article-grouped bootstrap 95% CIs, and
Holm-corrected paired-bootstrap significance against MorphEvalHeb -- not an ad hoc analysis.

## Robustness checks (before trusting the full run)

Two things needed verifying empirically rather than trusting vendor claims, since Jev's documented
context length conflicts across sources (32K to ~262K tokens -- OpenRouter's own model page states
262,144) and no source anywhere confirms or denies Hebrew-language support:

- **Truncation canary** (`pilot.py canary`): built a synthetic 17,327-character (13,629-token)
  document by concatenating six real long articles, with a unique, never-mentioned-elsewhere fact
  (an invented budget figure) in its final sentence. Two test summaries -- one stating that exact
  fact, one contradicting it -- scored **0.92** and **0.02** respectively. It could only separate
  these if it read all the way to the end of a document ~6x longer than anything in the real
  benchmark (max 2,935 characters for Natural, up to 13,240 for the longest real Injected
  documents). No truncation detected.
- **Diverse-genre batch** (`pilot.py diverse`, `jev_diverse_results.json`): the first stratified
  batch accidentally sampled 48/50 from the short "Natural" part. A follow-up batch explicitly
  covered all five native-Hebrew Injected genres (Israel_Hayom news, Wikipedia, Weizmann
  scientific, Knesset parliamentary, Bagatz judicial) plus Natural, biased toward each bucket's
  longest quartile (documents up to 13,240 characters). AUC on that 35-pair batch: 0.964, strong
  on every genre.

## Results (full benchmark, N=4,024, exact paper protocol)

| | AUC [95% CI] | BAcc (5-fold CV) | $\tau_b$ |
|---|---|---|---|
| **All** | 0.963 [0.958, 0.969] | 0.893 | 0.660 |
| Injected (n=1,889) | 0.988 | 0.946 | 0.688 |
| Natural (n=2,135) | 0.942 | 0.868 | 0.630 |

Per Injected source (AUC): Bagatz 1.000 (n=30), Knesset 1.000 (n=141), Wikipedia 0.998 (n=655),
Israel_Hayom 0.990 (n=864), Weizmann 0.952 (n=199).

This is the strongest baseline in the paper by a wide margin -- above ENFS (0.677, the strongest
previously-recomputed baseline), FactCG (0.562), and the GPT-4o-mini LLM judges ($\tau$=0.599 and
0.242) -- though MorphEvalHeb still beats it significantly (AUC 0.992, $p<0.001$ overall,
Holm-corrected; $p=0.017$ specifically on the Injected part).

## Raw data (everything needed to inspect or re-analyze this)

- **`jev_full_raw_data.json` / `jev_full_raw_data.csv`** (4,024 rows each): the complete, joined
  dataset -- `row_id`, `source`, `part` (Injected/Natural), `label`, `method`, `doc_id`, the full
  **`premise`** (article) and **`hypothesis`** (summary) text, lengths, the **`jev_noul_score`**,
  and per-call `jev_input_tokens`/`jev_output_tokens`/`jev_cost_usd`. This is the file to open if
  you want to read an actual article + summary + Jev's score side by side.
- **`jev_full_results.jsonl`**: the raw API responses as scored (one JSON object per line,
  includes duplicate lines for rows that failed and were later retried -- the raw data files above
  already dedupe to the successful score per row).
- **`jev_diverse_results.json`**: the 35-pair diverse-genre robustness batch.
- **`jev_pilot_results.json`**: the original 50-pair stratified pilot batch (run before the full
  benchmark; superseded by the full run but kept for reference).
- **`pilot.py`**: all code -- `smoke` (3 hand-picked sanity calls), `batch` (stratified pilot
  sample), `diverse` (genre-coverage batch), `canary` (truncation probe), `full` (the complete
  4,024-pair run), `diag` (masked API-key sanity check).
- `paper/results/eval/scores.csv`: canonical project score table, now with a `Jev` column (backup
  of the pre-Jev version at `scores.csv.bak_before_jev`).

## Honest caveats

- **Not a published, peer-reviewed metric.** Jev is an unpublished commercial model reached
  through an endpoint explicitly marked "alpha," with no disclosed architecture, parameter count,
  or training data. A provider could change or withdraw it at any time -- the exact snapshot used
  here was `typesafe/jev-1.14-20261014`.
  - When we searched for "Jev" / this category of "decision model" more broadly (looking for a
    more established alternative), the surrounding ecosystem (TypeSafe's own marketing cluster,
    the unrelated "Laya" product, several mutually-citing 2026 arXiv preprints, and claimed
    "LLM-AggreFact leaderboard" entries that do not actually appear on that leaderboard) read as
    largely unverifiable self-promotion. Jev's OpenRouter API access and context-length figures
    were the only claims in that whole cluster we could independently confirm by testing directly.
- **Possible data contamination.** We cannot rule out that Jev's (undisclosed) training data
  overlaps with our benchmark's public source corpora (MAFAT, mFACT), since both are public
  datasets.
- **Threshold tuning matches protocol, not necessarily "fair" to a closed model.** The balanced
  accuracy figure uses the same held-out-fold tuning as every other baseline, which is correct for
  comparability, but a closed API's internal behavior (e.g., any server-side calibration) is
  opaque to us.
- Cost was negligible ($0.62 total), so this is easy to re-run or extend if the model changes.

## If this becomes a real chapter

Possible next steps, not yet done: (1) test Jev's `choice`/`score` typed-question types (not just
binary `noul`) to see if finer-grained outputs correlate with MorphEvalHeb's multi-label error
families; (2) re-run after some time has passed to check whether the model snapshot changed and
whether results are stable; (3) a lightweight contamination check (e.g., canary strings or
membership-inference-style probing) given the data-overlap caveat above.
