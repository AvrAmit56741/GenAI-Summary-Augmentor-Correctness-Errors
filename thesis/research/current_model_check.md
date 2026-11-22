# Current-model check: Qwen3.8 Flash and DeepSeek V4 Pro 0813

Status: first round complete (2026-11-22), N=50 articles x 2 models. Thesis-facing summary; full
method detail, pipeline-bug diagnosis, and raw-data pointers live in
`../../docs/research_leads/current_model_check/README.md`.

## Research question

The taxonomy was built from errors found in models evaluated up to mid-2026. Two newer, more popular
models (released later in 2026, picked from a live "Most Popular" API-aggregator list) were tested:
do current-generation LLMs still make Hebrew-morphology-specific factual errors, or has the problem
been addressed by newer training?

## Method, in brief

Same targeted-article methodology as the project's earlier DictaLM3/GLM-4.7-Flash/Ministral-3-14B
rounds: the 50 historically hardest known articles, a minimal Hebrew summarization prompt, via
OpenRouter. Both candidate models turned out to be "reasoning" models with a failure mode the earlier,
non-reasoning-era rounds never hit: Qwen3.8 Flash was silently spending its whole token budget on
invisible reasoning (6/10 outputs empty on an initial smoke test) and separately drifting into
markdown-report formatting instead of flowing narrative prose. Both were diagnosed and fixed before
the real run (disabling reasoning; an explicit no-markdown prompt instruction) -- documented in full in
the docs/ writeup, since this is a useful practical lesson for evaluating any newly-released reasoning
model, not just these two.

Judging was done manually, by Claude directly reading all 100 (article, summary) pairs against the
project's real production taxonomy, its mandatory gate, and its English-contrast test -- not delegated
to an automated judge API call. An initial automated-judge attempt was tried first, hit the identical
reasoning-token bug, and was abandoned before completion at the user's explicit direction that judging
should be done directly rather than via another model call; its partial, unsaved results were
independently spot-checked afterward and not reproduced on careful manual re-reading.

## Results

**Qwen3.8 Flash: 3 of 50 summaries (6%) had a confirmed Hebrew-morphology error** (all severity 2):
a directional-clitic (מ/ל) flip reversing which direction a territorial change went; a named person
(a female friend) misgendered through a possessive clitic and verb agreement; and a root-sharing
number confusion (שלושה/3 vs. שלושים/30) within the same paragraph.

**DeepSeek V4 Pro 0813: 0 of 50 confirmed errors**, including on the exact same article where Qwen
misgendered a named person -- DeepSeek's independent summary of that article kept her gender correct
throughout.

## What this adds

A direct, current (Nov 2026) confirmation that the taxonomy's target error class is not solved by
newer, more popular general-purpose models -- Qwen3.8 Flash, a flagship-tier, widely-used model,
still produces exactly the kind of error the taxonomy was built to catch. The DeepSeek null result is
suggestive of a genuine difference between models, but caveated below.

## Honest caveats

N=50 per model on the *historically hardest* known articles (not a random sample), so these rates
aren't a general base rate and the DeepSeek null result isn't strong evidence of "fixed" -- 50
examples can easily miss a real 1-2% error rate. Single manual judge, no cross-annotator check on this
specific round. A second 50-article DeepSeek batch has already been generated but not yet judged.

## Raw data

All 100 (article, summary) pairs with full text, confirmed findings with full reasoning/evidence, and
the per-batch manual judging notes are in `../../docs/research_leads/current_model_check/README.md`'s
file list (all under `paper/results/eval/`).
