# Current-model check: Qwen3.8 Flash and DeepSeek V4 Pro 0813

Status: complete through 3 DeepSeek rounds + 1 Qwen round (2026-11-22), N=50 Qwen summaries + N=150
DeepSeek summaries, all manually judged including a full adversarial re-check. Thesis-facing summary;
full method detail, pipeline-bug diagnosis, and raw-data pointers live in
`../../docs/research_leads/current_model_check/README.md`.

## Research question

The taxonomy was built from errors found in models evaluated up to mid-2026. Two newer, more popular
models (released later in 2026, picked from a live "Most Popular" API-aggregator list) were tested:
do current-generation LLMs still make Hebrew-morphology-specific factual errors, or has the problem
been addressed by newer training?

## Method, in brief

Same targeted-article methodology as the project's earlier DictaLM3/GLM-4.7-Flash/Ministral-3-14B
rounds: the historically hardest known articles, a minimal Hebrew summarization prompt, via OpenRouter
-- three DeepSeek rounds (150 articles total) and one Qwen round (50 articles) so far. Both candidate
models turned out to be "reasoning" models with failure modes the earlier, non-reasoning-era rounds
never hit: Qwen3.8 Flash was silently spending its whole token budget on invisible reasoning (6/10
outputs empty on an initial smoke test) and separately drifting into markdown-report formatting instead
of flowing narrative prose; a transient network error later crashed a DeepSeek generation run partway
through, which led to rewriting the generation script to save incrementally and resume cleanly rather
than losing completed work on a crash. Full diagnosis in the docs/ writeup.

Judging was done manually, by Claude directly reading every (article, summary) pair against the
project's real production taxonomy, its mandatory gate, and its English-contrast test -- not delegated
to an automated judge API call (an initial automated-judge attempt hit the identical reasoning-token
bug and was abandoned before completion at the user's explicit direction).

**A second, adversarial pass mattered.** After the first judging round reported zero confirmed DeepSeek
errors on two of its three rounds, the user explicitly pushed back, having independently found real
errors the first pass had missed. Rather than defending the original "0 findings" result, all of those
summaries were re-judged completely fresh by new reviewers instructed to look harder and treat
borderline cases as worth reporting rather than dismissing -- including specifically revisiting every
near-miss the first pass had already found and rejected. This reversed one prior rejection into a
confirmed finding and is the reason the final counts below differ from what was first reported.

## Results

**Qwen3.8 Flash: 3 of 50 summaries (6%) had a confirmed Hebrew-morphology error** (all severity 2):
a directional-clitic (מ/ל) flip reversing which direction a territorial change went; a named person
(a female friend) misgendered through a possessive clitic and verb agreement; and a root-sharing
number confusion (שלושה/3 vs. שלושים/30) within the same paragraph.

**DeepSeek V4 Pro 0813: 3 of 150 summaries (2%) had a confirmed error**, across three 50-article rounds:
- Round 1: 0/50 (held up on adversarial re-check).
- Round 2: 2/50, both in one dense poetry-anthology article -- a poem's grammatically-female-marked
  speaking voice described as male, and a root-pattern confusion turning "my beloved" (לְדוֹדִי,
  masculine possessive) into "an aunt" (לדודה, an unrelated feminine noun sharing the same consonants).
- Round 3: 1/50 (reversed from an initial 0 on adversarial re-check) -- a linguistics column built
  entirely around niqqud-distinguished minimal word pairs, where DeepSeek's summary stripped the niqqud
  from all four quoted pairs, making "the wrong form" and "the correct form" read as identical text and
  erasing the article's entire point.

Notably, DeepSeek's two confirmed error types both come from unusually literary/linguistic source
material (poetry, a column about Hebrew grammar itself), not ordinary news prose -- its clean round-1
result was entirely news-style content.

## What this adds

A direct, current (Nov 2026) confirmation that the taxonomy's target error class is not solved by
newer, more popular general-purpose models. It also demonstrates something methodologically useful for
the thesis more broadly: an adversarial second pass, specifically instructed to distrust the first
pass's "clean" conclusion and re-examine every near-miss, measurably changed the result -- a concrete
illustration of why a single judging pass (even a careful one) isn't necessarily the final word, and
why the project's existing calibration-anchor practice (re-checking against known hard cases) matters.

## Honest caveats

N per round is 50, targeted at the historically hardest known articles rather than a random sample, so
these rates aren't a general base rate. Single manual judge per round (though two of the four rounds got
a genuine second independent pass via the adversarial re-check). DeepSeek's error types being tied to
literary/linguistic content rather than news prose means its overall 2% rate may not transfer to
different article mixes. A fourth DeepSeek round and a second Qwen round (both offset to new,
not-yet-seen articles) are in progress.

## Raw data

All (article, summary) pairs with full text, confirmed findings with full reasoning/evidence, and the
per-batch manual judging and adversarial-re-check notes are in
`../../docs/research_leads/current_model_check/README.md`'s file list (all under `paper/results/eval/`).
