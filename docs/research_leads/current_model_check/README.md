# Current-model check: do newly-released, popular LLMs still make Hebrew-morphology errors?

Status: complete through 3 DeepSeek rounds + 1 Qwen round (2026-11-22), N=50 Qwen3.8 Flash summaries +
N=150 DeepSeek V4 Pro 0813 summaries (200 total), all manually judged, including a full adversarial
re-check pass. Continues the project's existing "does the current model still err here" line
(DictaLM3/GLM-4.7-Flash/Ministral-3-14B rounds, not yet written up elsewhere — see
`paper/results/eval/exp_current_model_summaries_targeted.py` and siblings), extended to two newer, more
popular models not in the original evaluated roster: Qwen3.8 Flash and DeepSeek V4 Pro 0813, both
released mid-to-late 2026, picked from an OpenRouter "Most Popular" model list per the user's request.

## Method

**Generation** (`paper/results/eval/exp_current_model_summaries_openrouter_targeted.py`): same targeted
article-selection methodology as the earlier rounds — the N historically most error-prone articles (by
max confirmed severity, then error count, from `thesisdatasets/error_finder/*.csv`'s 198-article pool),
same minimal ("thin") Hebrew summarization prompt. Generated via OpenRouter, in three DeepSeek rounds
(`offset=0`, `offset=50`, `offset=100`, each N=50, covering the 150 historically hardest known articles)
and one Qwen3.8 Flash round (`offset=0`, N=50).

**Pipeline bugs found and fixed during setup** (both models are "reasoning" models by default on
OpenRouter, a failure mode not seen in the project's earlier, non-reasoning-era model rounds):
- Qwen3.8 Flash was silently spending its entire completion-token budget on invisible "thinking"
  tokens, leaving 6/10 outputs empty on an initial smoke test. Fixed with `reasoning: {"enabled": false}`.
- Qwen3.8 Flash also defaulted to a markdown-structured "report" style (bold headers, bullet lists),
  sometimes *expanding* a 762-character source into a 2,181-character report (2.86x) and truncating
  mid-bullet. Fixed with an explicit prompt instruction forbidding markdown and asking for flowing
  narrative prose, matching the style the taxonomy prompt and its confirmed-error examples are built
  around.
- A transient network connection reset crashed the third DeepSeek generation round at 38/50 articles;
  since the script originally only wrote output once at the very end, all 38 completed summaries were
  lost. Fixed by rewriting the script to save each row incrementally (JSONL) and resume from where it
  left off on a rerun, so a future crash can't lose completed work again.

Both model/prompt fixes verified via QC checks (empty/truncation/length-ratio/Hebrew-fraction) on a
10-article smoke test before committing to each full run; see
`exp_current_model_summaries_openrouter_targeted.py`'s QC pass and inline comments for the full diagnosis.

**Judging**: manual, by Claude directly (not a delegated API judge) — the user explicitly required this
("you should be the judge! openrouter use only for the summaries generation"). Every summary was read
and judged against the real production taxonomy prompt
(`summary_augmentation/correctness_errors_prompts.py`'s `correctness_errors_finder_prompt()`), its
mandatory gate (must name one of: בניין / גוף-מגדר-מספר / סמיכות / ה"א הידיעה / כינוי קניין מוצמד /
קליטיק), and its English-contrast test (reject anything an English summarizer could have gotten wrong
the same way), per `docs/error_finder_instructions_for_hebrew_llm.md`. Each round was split across 5
parallel forked Claude instances (10 articles each), with smaller follow-up forks (3-7 articles) to
close context-budget gaps where a fork ran out partway through a batch — a recurring issue on this
task given how long some source articles run (up to ~8,800 characters).

**A discarded first attempt at automated judging**: before the user's instruction to judge manually, a
first pass used an automated judge (Claude Sonnet 5.5 via OpenRouter's API) and hit the identical
reasoning-token bug (2,257 of 2,278 completion tokens spent on invisible thinking, leaving ~21 for the
actual JSON) plus a JSON-formatting bug (the judge sometimes emitted literal unescaped quote characters
inside string values). Both were fixed (`reasoning: {"max_tokens": 4000}` + `max_tokens: 16000`, plus an
explicit no-raw-quotes prompt instruction and a field-based JSON repair fallback — see
`exp_error_finder_openrouter.py`), but the job was stopped mid-run per the user's instruction before
producing a complete, saved result. Its partial live-log counts (Qwen: 6 findings/12 articles judged;
DeepSeek: 2 findings/16 articles judged) were never persisted to a file. The two DeepSeek articles it
had flagged were independently re-verified by hand and the finding was **not reproduced** on careful
re-reading (`manual_judge_batches/rejudge_deepseek_flagged_articles.csv`) — the faster, single-shot API
judge appears to have had a real false-positive problem relative to the more careful manual review.

**Adversarial re-check pass**: after the first manual judging round reported 0 confirmed DeepSeek errors
on both the `offset=0` and `offset=100` rounds, the user explicitly contested that result, having found
real errors independently. Rather than trusting the first pass, all 100 of those summaries were
re-judged completely fresh by new forks explicitly instructed to look harder, actively check every
taxonomy category (not just gender agreement), and report any borderline case rather than silently
dismissing it — including a deliberate, targeted re-examination of every near-miss the first pass had
already found-but-rejected. This reversed one prior rejection into a confirmed finding (see Results) and
confirmed the rest of the "0 findings" conclusions held up under harder scrutiny. This re-check is the
reason the final counts below differ from the first-pass numbers.

## Results

**Qwen3.8 Flash: 3 confirmed Hebrew-morphology errors out of 50 summaries (6%).** All severity 2:

1. `magazine-7774873`: directional clitic flip מ/ל — "removed **from** the city" became "removed **to**
   the city," reversing the territorial direction of a described Jerusalem-boundary change.
2. `celebrities-8932596`: a named female friend (Liron Eini) misgendered via possessive clitic + verb
   agreement — חברו/זיהה (his [male] friend / he identified) instead of חברתו/זיהתה (her feminine forms).
3. `general-14061589`: שלושה (3) vs. שלושים (30) — same consonantal root ש-ל-ש, differing only in the
   number-suffix (ה"/-ים), confused within the same paragraph of a veterans-reunion piece.

**DeepSeek V4 Pro 0813: 3 confirmed Hebrew-morphology errors out of 150 summaries (2%)**, across three
50-article rounds:

- **Round 1 (`offset=0`): 0/50.** Re-checked adversarially; held up. On the same article where Qwen
  misgendered Liron Eini, DeepSeek's independent summary kept her feminine correctly throughout
  (חברה/אמרה/העלתה) — a direct same-source, different-model contrast.
- **Round 2 (`offset=50`): 2/50**, both severity 2, both in the same article (`general-13160920`, a
  dense poetry-anthology piece — much more morphologically demanding than ordinary news prose):
  1. A poem's grammatically-marked-female speaking voice ("וַאֲנִי אַחַת") gets described as male in
     the summary, defaulting to the (male) poet's own gender instead of the text's explicit marking.
  2. "לְדוֹדִי" (to my beloved, masculine possessive clitic) became "לדודה" (to an aunt, an unrelated
     feminine noun sharing the same consonantal root ד-ו-ד) — a root-pattern confusion that replaces a
     love relationship with a family relationship.
- **Round 3 (`offset=100`): 1/50** (reversed from an initial 0 on adversarial re-check), severity 2:
  `culture-10266649`, a linguistics column built entirely around minimal word-pairs distinguished only
  by niqqud (unwritten vowel diacritics) — e.g. הֵכַּרתי (wrong) vs. הִכַּרתי (correct). DeepSeek's
  summary quotes all four such pairs with no niqqud, so "the wrong form" and "the correct form" become
  literally identical text, erasing the article's entire linguistic point. A textbook case of the
  taxonomy's "extreme morphological ambiguity from unwritten vowels" category, impossible in English
  (which encodes vowels directly in spelling). The first-pass judge had actually investigated this exact
  case and wrongly rejected it; the adversarial re-check reversed that call.

**Two other borderline cases were investigated during the re-check and explicitly NOT counted**, for
transparency:
- An ambiguous-gender cat (`animals-6829845`) where the *source itself* mixes a masculine-only noun
  (נמר, tiger) with a feminine demonstrative (הזאת) — too internally inconsistent in the source to
  confirm what the "correct" gender even is.
- A Knesset-quote paraphrase (`general-13165595`) with a mid-sentence person/number shift — flagged as
  likely failing the English-contrast test (a similar slip could plausibly happen when paraphrasing a
  quote in English too).

Full raw data: `paper/results/eval/current_model_summaries_qwen_qwen3_8-flash_targeted.csv`,
`..._deepseek_deepseek-v4-pro-0813_targeted.csv`, `..._offset50.csv`, `..._offset100.csv` (source
article + generated summary for all 200 pairs). Confirmed findings in `manual_judge_all_findings.csv`
(Qwen + DeepSeek round 1), `manual_judge_deepseek_offset50_findings.csv`,
`manual_judge_deepseek_offset100_findings.csv`. Per-batch judging detail in `manual_judge_batches/`,
`manual_judge_batches_deepseek_offset50/`, `manual_judge_batches_deepseek_offset100/`. Adversarial
re-check detail in `recheck_batches_offset0/` and `recheck_batches_offset100/`.

## Honest caveats

- N=50 per round, targeted at the *historically hardest* known articles (not a random sample), so the
  hit rates are not directly comparable to a base rate on typical text.
- Single judge (Claude, manual) per round, though the adversarial re-check functions as a second,
  independent pass on two of the four rounds (and reversed one call), which is some evidence the process
  is self-correcting rather than just confirming its own priors.
- DeepSeek's two confirmed error types (poem-speaker gender default, niqqud-dependent minimal pairs) both
  come from unusually literary/linguistic source material, not ordinary news prose — its 0/50 on purely
  news-style content (round 1) may not generalize to harder literary text, and conversely its 2/50 and
  1/50 rates on rounds 2-3 may be specific to how often this particular 150-article pool contains such
  material rather than a stable rate.
- The adversarial re-check did not reach full independent coverage of every single row with the same
  depth (some forks ran out of context budget partway through a batch and a smaller follow-up fork
  covered the remainder with a fresh read, rather than the original fork re-reading its own full batch
  twice) — see the per-batch fork reports in `recheck_batches_*/` for exactly which rows got a full
  fresh independent re-read versus a quicker pass.

## If this becomes a real chapter

Natural next steps: generate and judge a second and third Qwen3.8 Flash round (offset=50, offset=100)
the same way, for a like-for-like comparison with DeepSeek's N=150; consider whether the same two
pipeline bugs (reasoning-token starvation, markdown-report drift) generalize to other newly-released
reasoning models being considered for this project's evaluated roster; given DeepSeek's two confirmed
error types both involve literary/linguistic source material, a targeted round specifically sampling
poetry/linguistics-column articles (rather than the general "historically hardest" pool, which skews
toward dense news/political content) could test whether that's a real, more general weak point.
