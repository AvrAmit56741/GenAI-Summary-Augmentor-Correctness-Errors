# Current-model check: Qwen3.8 Flash and DeepSeek V4 Pro 0813

Status: complete (2026-11-23). N=150 Qwen3.8 Flash summaries (149 judged, 1 broken row excluded) +
N=248 DeepSeek V4 Pro 0813 summaries (all judged), all manually judged including a full adversarial
re-check and a final round on a completely fresh, never-before-touched article pool. Thesis-facing
summary; full method detail, pipeline-bug diagnosis, and raw-data pointers live in
`../../docs/research_leads/current_model_check/README.md`.

## Research question

The taxonomy was built from errors found in models evaluated up to mid-2026. Two newer, more popular
models (released later in 2026, picked from a live "Most Popular" API-aggregator list) were tested:
do current-generation LLMs still make Hebrew-morphology-specific factual errors, or has the problem
been addressed by newer training?

## Method, in brief

Same targeted-article methodology as the project's earlier DictaLM3/GLM-4.7-Flash/Ministral-3-14B
rounds for the first several rounds: the historically hardest known articles (198-article pool), a
minimal Hebrew summarization prompt, via OpenRouter -- four DeepSeek rounds and two Qwen rounds
exhausted that entire pool (every doc_id in it now has a generated summary from at least one model).
A final 50-article round for each model then drew from a disjoint, genuinely untouched pool: the
mFACT-He portion of the project's own NLI benchmark (`combined.csv`, 303 unique articles spanning
Israel_Hayom, Wikipedia, Weizmann Institute science explainers, and Knesset transcripts), filtered to
the 293 articles with zero overlap with the exhausted pool, sampled with a fixed seed so both models
saw the identical 50 new articles.

Both candidate models turned out to be "reasoning" models with failure modes the earlier,
non-reasoning-era rounds never hit: Qwen3.8 Flash was silently spending its whole token budget on
invisible reasoning (6/10 outputs empty on an initial smoke test) and separately drifting into
markdown-report formatting instead of flowing narrative prose; a transient network error later
crashed a DeepSeek generation run partway through, which led to rewriting the generation script to
save incrementally and resume cleanly rather than losing completed work on a crash. Full diagnosis in
the docs/ writeup.

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

**Qwen3.8 Flash: 6 of 149 judged summaries (4.0%) had a confirmed Hebrew-morphology error:**
a directional-clitic (מ/ל) flip reversing which direction a territorial change went; a named person
(a female friend) misgendered through a possessive clitic and verb agreement; a root-sharing number
confusion (שלושה/3 vs. שלושים/30) within the same paragraph; a reality-show contestant's gender
flipped mid-sentence via verb agreement; a root-pattern confusion turning "my beloved" into "an aunt"
(see DeepSeek's identical error below -- a convergent finding); and, on the fresh never-before-used
article pool, a root confusion turning "she was ordered/commanded" (נצטוותה) into "she was flooded"
(הוצפה) inside an identical passive-binyan template.

**DeepSeek V4 Pro 0813: 4 of 248 judged summaries (1.6%) had a confirmed error**, across five rounds:
- Round 1 (50 articles): 0/50 (held up on adversarial re-check).
- Round 2 (50 articles): 2/50, both in one dense poetry-anthology article -- a poem's
  grammatically-female-marked speaking voice described as male, and a root-pattern confusion turning
  "my beloved" (לְדוֹדִי, masculine possessive) into "an aunt" (לדודה, an unrelated feminine noun
  sharing the same consonants).
- Round 3 (50 articles): 1/50 (reversed from an initial 0 on adversarial re-check) -- a linguistics
  column built entirely around niqqud-distinguished minimal word pairs, where DeepSeek's summary
  stripped the niqqud from all four quoted pairs, making "the wrong form" and "the correct form" read
  as identical text and erasing the article's entire point.
- Round 4 (48 articles, pool exhausted): 0/48.
- New-pool round (50 never-before-used articles): 1/50 -- a gender-suffix role confusion where "he
  serves as president" (הוא משמש כנשיא) became "he serves as [a female] president" (כנשיאה), directly
  contradicting "he" in the same clause.

**A striking convergent finding**: on the exact same article (`general-13160920`, a poetry anthology),
Qwen3.8 Flash and DeepSeek V4 Pro 0813 -- two architecturally unrelated models, queried completely
independently -- both made the identical root-pattern confusion on the identical line of the identical
poem, turning "my beloved" into "an aunt." Strong evidence this specific ambiguity is a structurally
hard case for current LLMs generally, not a random single-model slip.

Both models held their established error rates on the fresh, more varied article pool (adding
Wikipedia/Weizmann-science/Knesset content, not just Israel_Hayom news) -- no sign the earlier rounds
were an artifact of one narrow source type.

## What this adds

A direct, current (Nov 2026) confirmation that the taxonomy's target error class is not solved by
newer, more popular general-purpose models, and that it generalizes beyond the original targeted
article pool to a fresh, more varied one. It also demonstrates something methodologically useful for
the thesis more broadly: an adversarial second pass, specifically instructed to distrust the first
pass's "clean" conclusion and re-examine every near-miss, measurably changed the result -- a concrete
illustration of why a single judging pass (even a careful one) isn't necessarily the final word, and
why the project's existing calibration-anchor practice (re-checking against known hard cases) matters.
The cross-model convergent error on the same poem is additional evidence that at least some of the
taxonomy's error categories reflect genuine structural ambiguity in Hebrew morphology, not an
idiosyncrasy of any one model.

## Honest caveats

The first four rounds per model targeted the *historically hardest* known articles (not a random
sample); the final 50-article round per model used a random sample from a disjoint, never-before-tested
pool, so it is a cleaner (if still Israel_Hayom/Wikipedia/Weizmann/Knesset-skewed) complement to the
targeted rounds rather than a fully general base rate. Single manual judge per round, though the
adversarial re-check functions as a second, independent pass on two of the DeepSeek rounds (and
reversed one call), which is some evidence the process is self-correcting rather than just confirming
its own priors. A few genuine-but-non-qualifying issues were found and explicitly excluded throughout
(digit-transcription slips, a factual mislabeling, grammar-only gender slips on inanimate nouns that
don't change any identifiable fact) -- these are real reliability problems but outside this taxonomy's
Hebrew-morphology scope by design.

## Raw data

All (article, summary) pairs with full text, confirmed findings with full reasoning/evidence, and the
per-batch manual judging and adversarial-re-check notes are in
`../../docs/research_leads/current_model_check/README.md`'s file list (all under `paper/results/eval/`).
