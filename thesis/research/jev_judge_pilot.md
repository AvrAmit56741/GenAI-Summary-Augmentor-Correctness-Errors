# Jev (TypeSafe decisions API): a decision-model judge baseline

Status: complete, full benchmark scored (2026-11-22), N=4,024. This is a thesis-facing summary;
the full writeup with method detail and raw-data pointers lives in
`../../docs/research_leads/jev_judge_pilot/README.md`, and the code lives in
`../../docs/research_leads/jev_judge_pilot/pilot.py`.

## Research question

Every LLM-judge baseline already in the paper (G-EVAL, the vanilla zero-shot judge) is
*generative*: a chat model writes free text ("Reasoning: ... Score: 4") that we then parse a
number out of. A different architecture exists -- a *decision model* -- that takes a fixed typed
question with explicit answer options and returns a calibrated probability directly in one forward
pass, with nothing generated and nothing to parse. Does this architecture work for Hebrew
factuality judging, and how does it compare to everything else in the paper?

## Why this came up

This started from a question about "decision models like Jev" after a vendor model card for a
product called "Laya" (`convaiinnovations/laya`) compared itself against something called "Jev."
Investigating led through several dead ends worth recording, since they're the kind of thing a
thesis committee might ask about:

- **Laya itself doesn't hold up**: its real released checkpoints are a small (421M-parameter)
  ModernBERT-large encoder with only a **512-token context window** -- far too short to read a
  full article -- and no confirmed Hebrew support. Not usable for this task.
- **"Jev" as a broader category** (searching for named "LLM-AggreFact leaderboard" entries like
  Clef, Decider, d1-ultra, Drex) turned out to be mostly unverifiable: those names do not actually
  appear on the real LLM-AggreFact leaderboard, and the surrounding ecosystem (several
  mutually-citing 2026 arXiv preprints, inconsistent claims across them) reads as largely
  self-promotional marketing rather than independently verifiable research.
- **Jev specifically, though, turned out to have a real, working, documented API** -- reachable
  through OpenRouter -- that we could test directly ourselves rather than trust any claim about it.
  That's what this pilot does: not cite Jev's marketing, but measure it, exactly as every other
  baseline in the paper is measured.

## Method

Model: `typesafe/jev-1.14` (provider TypeSafe), reached via OpenRouter's **alpha Decisions API**
(a different endpoint from normal chat -- the standard chat-completions endpoint explicitly
rejects this model). Each call sends the **full, untruncated** source article and summary as a
`state`, plus one binary typed question ("is every claim in the summary supported by the source
document?"), and gets back a single calibrated probability (`noul`, 0--1) -- no chunking, no
free-text parsing.

We scored all 4,024 pairs of the project's benchmark and merged the results into the exact same
`scores.csv` used by every other baseline, so Jev is evaluated with the *identical* protocol as
MorphEvalHeb and the rest of Table 2 in the paper: 5-fold article-grouped cross-validated balanced
accuracy, article-grouped bootstrap 95% confidence intervals, Holm-corrected significance testing.
Total cost for all 4,024 calls: **$0.62**.

## Robustness checks, done before trusting the result

Two things specifically needed verification rather than trust, since Jev's documented context
length conflicts across sources (32K vs. ~262K tokens) and no source confirms or denies Hebrew
support:

- **Does it actually read the whole document, or silently truncate?** We built a synthetic
  17,327-character document (six real articles concatenated) with a unique invented fact placed
  only in the final sentence, then asked it to judge two summaries -- one matching that fact, one
  contradicting it. It scored them 0.92 and 0.02 respectively: it could only tell these apart by
  reading to the true end of a document ~6x longer than anything in the real benchmark.
- **Does it actually work across every native-Hebrew genre, or only the easy/short ones?** An
  initial batch accidentally sampled mostly short, English-translated text. A follow-up batch
  deliberately covered all five native-Hebrew genres (news, encyclopedic, scientific,
  parliamentary, judicial) biased toward each genre's longest documents (up to 13,240 characters),
  and it held up strongly on every one.

## Results

Full benchmark (N=4,024), same protocol as every other baseline in the paper:

| | AUC [95% CI] | Balanced accuracy | $\tau_b$ |
|---|---|---|---|
| **All** | 0.963 [0.958, 0.969] | 0.893 | 0.660 |
| Injected (native Hebrew, n=1,889) | 0.988 | 0.946 | 0.688 |
| Natural (translated, n=2,135) | 0.942 | 0.868 | 0.630 |

Per-genre AUC within Injected: Bagatz (judicial) and Knesset (parliamentary) both 1.000, Wikipedia
0.998, Israel_Hayom (news) 0.990, Weizmann (scientific) 0.952 -- consistently strong, not just on
one easy genre.

**This is the strongest baseline in the entire paper, by a wide margin** -- clearly above ENFS
(0.677, previously the strongest recomputed baseline), FactCG (0.562), and the GPT-4o-mini LLM
judges ($\tau$=0.599 and 0.242). MorphEvalHeb still wins significantly (AUC 0.992, $p<0.001$
overall, Holm-corrected), but the gap to this baseline is far smaller than to anything else.

## Honest caveats

- **Not a peer-reviewed metric.** Jev is an unpublished commercial model behind an "alpha" API
  endpoint, with no disclosed architecture, parameter count, or training data, and a provider could
  change or withdraw it at any time. We report the exact resolved snapshot
  (`typesafe/jev-1.14-20261014`) for reproducibility, but "reproducible" here means "we can call
  the same named endpoint again," not "the underlying model is guaranteed unchanged."
- **Possible contamination.** We cannot rule out that its (undisclosed) training data overlaps
  with our benchmark's public source corpora (MAFAT, mFACT) -- both are public datasets a
  commercial model could plausibly have seen.
- Threshold tuning for balanced accuracy uses the same held-out-fold protocol as every other
  baseline, so the comparison is fair in that specific sense, even though the model itself is a
  black box.

## Raw data

Full article + summary + Jev score for all 4,024 pairs, plus per-call token counts and cost, is
saved in `../../docs/research_leads/jev_judge_pilot/jev_full_raw_data.{json,csv}`. Raw API
responses, the robustness-check scripts, and the pilot code are in that same folder.

## If this becomes a real chapter

Not yet done: testing Jev's other typed-question types (`choice`/`score`, not just binary `noul`)
against MorphEvalHeb's multi-label error families; re-running after time has passed to check
snapshot stability; a lightweight contamination check given the data-overlap caveat above.
