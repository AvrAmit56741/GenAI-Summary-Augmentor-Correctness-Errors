# Current-model check: do newly-released, popular LLMs still make Hebrew-morphology errors?

Status: first round complete (2026-11-22), N=50 articles x 2 models. Continues the project's existing
"does the current model still err here" line (DictaLM3/GLM-4.7-Flash/Ministral-3-14B rounds, not yet
written up elsewhere — see `paper/results/eval/exp_current_model_summaries_targeted.py` and siblings),
extended to two newer, more popular models not in the original evaluated roster: Qwen3.8 Flash and
DeepSeek V4 Pro 0813, both released mid-to-late 2026, picked from an OpenRouter "Most Popular" model
list per the user's request.

## Method

**Generation** (`paper/results/eval/exp_current_model_summaries_openrouter_targeted.py`): same targeted
article-selection methodology as the earlier rounds — the 50 historically most error-prone articles
(by max confirmed severity, then error count, from `thesisdatasets/error_finder/*.csv`'s 198-article
pool), same minimal ("thin") Hebrew summarization prompt. Generated via OpenRouter.

**Two pipeline bugs found and fixed during setup** (both models are "reasoning" models by default on
OpenRouter, a failure mode not seen in the project's earlier, non-reasoning-era model rounds):
- Qwen3.8 Flash was silently spending its entire completion-token budget on invisible "thinking"
  tokens, leaving 6/10 outputs empty on an initial smoke test. Fixed with `reasoning: {"enabled": false}`.
- Qwen3.8 Flash also defaulted to a markdown-structured "report" style (bold headers, bullet lists),
  sometimes *expanding* a 762-character source into a 2,181-character report (2.86x) and truncating
  mid-bullet. Fixed with an explicit prompt instruction forbidding markdown and asking for flowing
  narrative prose, matching the style the taxonomy prompt and its confirmed-error examples are built
  around. Both fixes verified via QC checks (empty/truncation/length-ratio/Hebrew-fraction) on a 10-article
  smoke test before committing to the full 50-article run; see `exp_current_model_summaries_openrouter_targeted.py`'s
  QC pass and inline comments for the full diagnosis.

**Judging**: manual, by Claude directly (not a delegated API judge) — the user explicitly required this
("you should be the judge! openrouter use only for the summaries generation"). All 100 summary judgments
(50 articles x 2 models) were read and judged against the real production taxonomy prompt
(`summary_augmentation/correctness_errors_prompts.py`'s `correctness_errors_finder_prompt()`), its
mandatory gate (must name one of: בניין / גוף-מגדר-מספר / סמיכות / ה"א הידיעה / כינוי קניין מוצמד /
קליטיק), and its English-contrast test (reject anything an English summarizer could have gotten wrong
the same way), per `docs/error_finder_instructions_for_hebrew_llm.md`. Split across 5 parallel forked
Claude instances (10 articles/20 judgments each) plus 2 small follow-up forks to cover a context-budget
gap in one batch.

**A discarded first attempt**: before the user's instruction to judge manually, a first pass used an
automated judge (Claude Sonnet 5.5 via OpenRouter's API) and hit the identical reasoning-token bug
(2,257 of 2,278 completion tokens spent on invisible thinking, leaving ~21 for the actual JSON) plus a
JSON-formatting bug (the judge sometimes emitted literal unescaped quote characters inside string
values). Both were fixed (`reasoning: {"max_tokens": 4000}` + `max_tokens: 16000`, plus an explicit
no-raw-quotes prompt instruction and a field-based JSON repair fallback — see
`exp_error_finder_openrouter.py`), but the job was stopped mid-run per the user's instruction before
producing a complete, saved result. Its partial live-log counts (Qwen: 6 findings/12 articles judged;
DeepSeek: 2 findings/16 articles judged) were never persisted to a file. The two DeepSeek articles it
had flagged were independently re-verified by hand against the final manual result and the finding was
**not reproduced** on careful re-reading (see `manual_judge_batches/rejudge_deepseek_flagged_articles.csv`)
— the faster, single-shot API judge appears to have had a real false-positive problem relative to the
more careful manual, multi-step review (which explicitly caught and rejected two separate near-misses
with documented reasoning in batch 2 — see that batch's notes).

## Results

**Qwen3.8 Flash: 3 confirmed Hebrew-morphology errors out of 50 summaries (6%).** All severity 2:

1. `magazine-7774873`: directional clitic flip מ/ל — "removed **from** the city" became "removed **to**
   the city," reversing the territorial direction of a described Jerusalem-boundary change.
2. `celebrities-8932596`: a named female friend (Liron Eini) misgendered via possessive clitic + verb
   agreement — חברו/זיהה (his [male] friend / he identified) instead of חברתו/זיהתה (her feminine forms).
3. `general-14061589`: שלושה (3) vs. שלושים (30) — same consonantal root ש-ל-ש, differing only in the
   number-suffix (ה"/-ים), confused within the same paragraph of a veterans-reunion piece.

**DeepSeek V4 Pro 0813: 0 confirmed errors out of 50 summaries.** Notably, on the *same* article where
Qwen misgendered Liron Eini, DeepSeek's independent summary kept her feminine correctly throughout
(חברה/אמרה/העלתה) — a direct same-source, different-model contrast.

Full raw data: `paper/results/eval/current_model_summaries_qwen_qwen3_8-flash_targeted.csv` and
`..._deepseek_deepseek-v4-pro-0813_targeted.csv` (source article + generated summary for all 100
pairs), confirmed findings in `manual_judge_all_findings.csv`, per-batch judging detail in
`manual_judge_batches/`.

## Honest caveats

- N=50 per model, targeted at the *historically hardest* known articles (not a random sample), so the
  6%/0% hit rates are not directly comparable to a base rate on typical text, and are not large enough
  to support a confident "DeepSeek is immune" claim — a null result on 50 examples is consistent with a
  real but low (e.g. 1-2%) error rate that a larger sample would catch.
- Single judge (Claude, manual), no second annotator cross-check on this round specifically, though the
  methodology mirrors the project's established taxonomy/gate/English-contrast protocol exactly.
- A second batch of 50 DeepSeek summaries (articles ranked 51-100 in the same pool, offset=50) has been
  generated but not yet judged as of this writing.

## If this becomes a real chapter

Natural next steps: judge the second DeepSeek batch (50 more articles already generated); generate and
judge a second Qwen3.8 Flash batch the same way for a larger N; consider whether the same two pipeline
bugs (reasoning-token starvation, markdown-report drift) generalize to other newly-released reasoning
models being considered for this project's evaluated roster.
