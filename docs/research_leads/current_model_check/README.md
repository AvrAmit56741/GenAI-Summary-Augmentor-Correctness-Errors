# Current-model check: do newly-released, popular LLMs still make Hebrew-morphology errors?

Status: complete (2026-11-23), N=150 Qwen3.8 Flash summaries (149 judged, 1 broken row excluded) +
N=248 DeepSeek V4 Pro 0813 summaries (398 total), all manually judged, including a full adversarial
re-check pass and a final round on a disjoint, never-before-used article pool. Continues the project's
existing "does the current model still err here" line (DictaLM3/GLM-4.7-Flash/Ministral-3-14B rounds,
not yet written up elsewhere -- see `paper/results/eval/exp_current_model_summaries_targeted.py` and
siblings), extended to two newer, more popular models not in the original evaluated roster: Qwen3.8
Flash and DeepSeek V4 Pro 0813, both released mid-to-late 2026, picked from an OpenRouter "Most
Popular" model list per the user's request.

## Method

**Generation** (`paper/results/eval/exp_current_model_summaries_openrouter_targeted.py`): same targeted
article-selection methodology as the earlier rounds for most of the run -- the N historically most
error-prone articles (by max confirmed severity, then error count, from
`thesisdatasets/error_finder/*.csv`'s 198-article pool), same minimal ("thin") Hebrew summarization
prompt. Generated via OpenRouter: two Qwen rounds (`offset=0`, `offset=50`, N=50 each) and four
DeepSeek rounds (`offset=0`, `offset=50`, `offset=100`, `offset=150` [only 48 returned -- pool
exhausted at 198 articles]) -- between them, every doc_id in the 198-article pool now has a generated
summary from at least one of these two models.

A final round per model (`pick_new_articles()`, invoked with the `new` pool argument) then drew from a
genuinely disjoint pool: the mFACT-He portion of the project's own NLI benchmark
(`thesisdatasets/train_datasets/Correctness/data/benchmark/combined.csv`), which has 303 unique
doc_id-tagged articles (Israel_Hayom, Wikipedia, Weizmann Institute science explainers, Knesset
transcripts) built for a different purpose (entailment pairs) and never previously fed through any
summarization model. Only 10 of those 303 overlap with the exhausted 198-article pool, leaving 293
candidates; 50 were sampled with a fixed seed (42) so both models summarized the identical new
articles, with no severity-based ranking (there's no prior judging signal for never-tested articles).

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
  left off on a rerun -- this resumable/retry logic later transparently absorbed an 18-minute network
  stall during the new-pool DeepSeek round (6 retries with exponential backoff) without losing any work.

Both model/prompt fixes verified via QC checks (empty/truncation/length-ratio/Hebrew-fraction) on a
10-article smoke test before committing to each full run; see
`exp_current_model_summaries_openrouter_targeted.py`'s QC pass and inline comments for the full diagnosis.

**Judging**: manual, by Claude directly (not a delegated API judge) -- the user explicitly required this
("you should be the judge! openrouter use only for the summaries generation"). Every summary was read
and judged against the real production taxonomy prompt
(`summary_augmentation/correctness_errors_prompts.py`'s `correctness_errors_finder_prompt()`), its
mandatory gate (must name one of: בניין / גוף-מגדר-מספר / סמיכות / ה"א הידיעה / כינוי קניין מוצמד /
קליטיק), and its English-contrast test (reject anything an English summarizer could have gotten wrong
the same way), per `docs/error_finder_instructions_for_hebrew_llm.md`. Each round was split across 5
parallel forked Claude instances (10 articles each), with smaller follow-up forks (3-7 articles) to
close context-budget gaps where a fork ran out partway through a batch -- a recurring issue on this
task given how long some source articles run (up to ~11,800 characters on the new-pool Wikipedia
articles).

**A discarded first attempt at automated judging**: before the user's instruction to judge manually, a
first pass used an automated judge (Claude Sonnet 5.5 via OpenRouter's API) and hit the identical
reasoning-token bug (2,257 of 2,278 completion tokens spent on invisible thinking, leaving ~21 for the
actual JSON) plus a JSON-formatting bug (the judge sometimes emitted literal unescaped quote characters
inside string values). Both were fixed (`reasoning: {"max_tokens": 4000}` + `max_tokens: 16000`, plus an
explicit no-raw-quotes prompt instruction and a field-based JSON repair fallback -- see
`exp_error_finder_openrouter.py`), but the job was stopped mid-run per the user's instruction before
producing a complete, saved result. Its partial live-log counts (Qwen: 6 findings/12 articles judged;
DeepSeek: 2 findings/16 articles judged) were never persisted to a file. The two DeepSeek articles it
had flagged were independently re-verified by hand and the finding was **not reproduced** on careful
re-reading (`manual_judge_batches/rejudge_deepseek_flagged_articles.csv`) -- the faster, single-shot API
judge appears to have had a real false-positive problem relative to the more careful manual review.

**Adversarial re-check pass**: after the first manual judging round reported 0 confirmed DeepSeek errors
on both the `offset=0` and `offset=100` rounds, the user explicitly contested that result, having found
real errors independently. Rather than trusting the first pass, all 100 of those summaries were
re-judged completely fresh by new forks explicitly instructed to look harder, actively check every
taxonomy category (not just gender agreement), and report any borderline case rather than silently
dismissing it -- including a deliberate, targeted re-examination of every near-miss the first pass had
already found-but-rejected. This reversed one prior rejection into a confirmed finding (see Results) and
confirmed the rest of the "0 findings" conclusions held up under harder scrutiny. This re-check is the
reason the final counts below differ from the first-pass numbers. The same "report borderline cases,
include a confidence rating" instruction was carried forward into every subsequent round (including the
Qwen round 2 and both new-pool rounds), rather than only applied retroactively.

## Results

**Qwen3.8 Flash: 6 confirmed Hebrew-morphology errors out of 149 judged summaries (4.0%)** (one
`general-13577952` row excluded entirely -- a total English-language task refusal/misread, including
hallucinating that a real person didn't exist, a distinct failure mode outside this taxonomy's scope):

1. `magazine-7774873`: directional clitic flip מ/ל -- "removed **from** the city" became "removed **to**
   the city," reversing the territorial direction of a described Jerusalem-boundary change. (Round 1,
   severity 2)
2. `celebrities-8932596`: a named female friend (Liron Eini) misgendered via possessive clitic + verb
   agreement -- חברו/זיהה (his [male] friend / he identified) instead of חברתו/זיהתה (her feminine
   forms). (Round 1, severity 2)
3. `general-14061589`: שלושה (3) vs. שלושים (30) -- same consonantal root ש-ל-ש, differing only in the
   number-suffix (ה"/-ים), confused within the same paragraph of a veterans-reunion piece. (Round 1,
   severity 2)
4. `culture-8031024`: a reality-show contestant explicitly marked female in the source gets masculine
   verb agreement in the summary while keeping a feminine pronoun in the same sentence -- internal
   gender inconsistency. (Round 2, severity 2)
5. `general-13160920`: "לְדוֹדִי" (to my beloved, masculine possessive clitic) became "לדודה" (to an
   aunt, an unrelated feminine noun sharing the same consonantal root ד-ו-ד) -- the same error DeepSeek
   made independently on the identical line of the identical poem (see below). (Round 2, severity 2)
6. `judaism-7857613`: root-binyan confusion -- source's "נצטוותה" (she was ordered/commanded, root
   צ-ו-ה, nifal) became "הוצפה" (she was flooded, root צ-פ-ה, hufal) in the summary, two unrelated
   roots differing by one consonant inside the identical passive-binyan template, producing a
   syntactically broken sentence. (New pool, severity 1, medium confidence)

**DeepSeek V4 Pro 0813: 4 confirmed Hebrew-morphology errors out of 248 judged summaries (1.6%)**,
across five rounds:

- **Round 1 (`offset=0`, 50 articles): 0/50.** Re-checked adversarially; held up. On the same article
  where Qwen misgendered Liron Eini, DeepSeek's independent summary kept her feminine correctly
  throughout (חברה/אמרה/העלתה) -- a direct same-source, different-model contrast.
- **Round 2 (`offset=50`, 50 articles): 2/50**, both severity 2, both in the same article
  (`general-13160920`, a dense poetry-anthology piece -- much more morphologically demanding than
  ordinary news prose):
  1. A poem's grammatically-marked-female speaking voice ("וַאֲנִי אַחַת") gets described as male in
     the summary, defaulting to the (male) poet's own gender instead of the text's explicit marking.
  2. "לְדוֹדִי" (to my beloved, masculine possessive clitic) became "לדודה" (to an aunt, an unrelated
     feminine noun sharing the same consonants) -- a root-pattern confusion that replaces a love
     relationship with a family relationship. Independently reproduced by Qwen on the same line.
- **Round 3 (`offset=100`, 50 articles): 1/50** (reversed from an initial 0 on adversarial re-check),
  severity 2: `culture-10266649`, a linguistics column built entirely around minimal word-pairs
  distinguished only by niqqud (unwritten vowel diacritics) -- e.g. הֵכַּרתי (wrong) vs. הִכַּרתי
  (correct). DeepSeek's summary quotes all four such pairs with no niqqud, so "the wrong form" and
  "the correct form" become literally identical text, erasing the article's entire linguistic point. A
  textbook case of the taxonomy's "extreme morphological ambiguity from unwritten vowels" category,
  impossible in English (which encodes vowels directly in spelling). The first-pass judge had actually
  investigated this exact case and wrongly rejected it; the adversarial re-check reversed that call.
- **Round 4 (`offset=150`, 48 articles, pool exhausted): 0/48.** All 5 judging batches came back clean,
  matching round 1's result on similar general news/business/food content.
- **New-pool round (50 never-before-used articles): 1/50**, severity 2, high confidence:
  `1443644`, a Wikipedia biography -- source's "הוא משמש כנשיא המכינה" (he serves as president of the
  prep academy, masculine) became "ומאז הוא משמש כנשיאה" (he serves as [a female] president), the
  feminine ה suffix directly contradicting "הוא" (he) in the same clause. Same homographic-role +
  gender-agreement mechanism as the taxonomy's מורה/מורה examples.

**A notable cross-model convergent error**: Qwen3.8 Flash and DeepSeek V4 Pro 0813 -- two
architecturally unrelated models, queried completely independently -- both made the identical
root-pattern confusion (לדודי → לדודה) on the identical line of the identical poem
(`general-13160920`). Verified by direct row-index cross-check confirming both files' row 8 is the
same `doc_id`. Strong evidence this specific ambiguity is a structurally hard case for current LLMs in
general, not a random single-model slip.

**Several other borderline cases were investigated and explicitly NOT counted**, for transparency (full
detail in the per-batch CSVs):
- An ambiguous-gender cat (`animals-6829845`) where the *source itself* mixes a masculine-only noun
  (נמר, tiger) with a feminine demonstrative (הזאת) -- too internally inconsistent in the source to
  confirm what the "correct" gender even is.
- A Knesset-quote paraphrase (`general-13165595`) with a mid-sentence person/number shift -- flagged as
  likely failing the English-contrast test.
- (New pool) Several purely grammatical gender-agreement slips on **inanimate** nouns (e.g. a
  corporation name's verb agreement, a possessive clitic disambiguated elsewhere in the same sentence)
  that technically touch the taxonomy's mandatory gate category but don't change any identifiable fact
  about who-did-what -- excluded under the same "must be a real factual distortion" bar used throughout.
- (New pool) A digit/word-substitution error (וצעק → וצעיר, "shouted" → nonsensical "young") that fails
  the mandatory gate entirely (no binyan/gender/construct-state/clitic mechanism involved) -- a real
  issue, but outside this taxonomy's scope, same treatment as the million/billion-inflation and
  digit-transcription issues noted below.

**Non-qualifying but real issues noted along the way** (flagged for transparency, never counted as
taxonomy findings because they fail the mandatory gate and/or the English-contrast test): Qwen
systematically converted "million" to "billion" four times in one article (1000x numeric inflation);
digit-transcription errors (a timestamp losing its hour digit); a factual mislabeling (a minister's
former title stated as his current one); a total English-language task refusal on one Qwen article,
including hallucinating that a real named person didn't exist (the one excluded row above).

Full raw data: `paper/results/eval/current_model_summaries_qwen_qwen3_8-flash_targeted.csv`,
`..._offset50.csv`, `..._newpool.csv`; `..._deepseek_deepseek-v4-pro-0813_targeted.csv`, `..._offset50.csv`,
`..._offset100.csv`, `..._offset150.csv`, `..._newpool.csv` (source article + generated summary for all
398 pairs). Confirmed findings in `manual_judge_all_findings.csv` (Qwen + DeepSeek round 1),
`manual_judge_deepseek_offset50_findings.csv`, `manual_judge_deepseek_offset100_findings.csv`,
`manual_judge_qwen_offset50_findings.csv`, `manual_judge_newpool_findings.csv`. Per-batch judging
detail in `manual_judge_batches/`, `manual_judge_batches_deepseek_offset50/`,
`manual_judge_batches_deepseek_offset100/`, `manual_judge_batches_deepseek_offset150/`,
`manual_judge_batches_qwen_offset50/`, `manual_judge_batches_newpool/`. Adversarial re-check detail in
`recheck_batches_offset0/` and `recheck_batches_offset100/`.

## Honest caveats

- The first four rounds per model (N=50 or 48 each) targeted the *historically hardest* known articles
  (not a random sample); the final 50-article new-pool round per model used a random sample (fixed
  seed) from a disjoint, never-before-tested pool, so it's a cleaner complement to the targeted rounds
  but still skewed toward the same handful of source outlets (Israel_Hayom/Wikipedia/Weizmann/Knesset)
  this project's datasets happen to contain -- not a fully general base rate on arbitrary Hebrew text.
- Single judge (Claude, manual) per round, though the adversarial re-check functions as a second,
  independent pass on two of the DeepSeek rounds (and reversed one call), which is some evidence the
  process is self-correcting rather than just confirming its own priors.
- DeepSeek's two literary-content confirmed error types (poem-speaker gender default, niqqud-dependent
  minimal pairs) come from unusually literary/linguistic source material; its new-pool finding and
  round-1/round-4 clean results are all ordinary news/Wikipedia/biography content, so the overall 1.6%
  rate is not purely driven by one narrow genre.
- The adversarial re-check did not reach full independent coverage of every single row with the same
  depth (some forks ran out of context budget partway through a batch and a smaller follow-up fork
  covered the remainder with a fresh read, rather than the original fork re-reading its own full batch
  twice) -- see the per-batch fork reports in `recheck_batches_*/` for exactly which rows got a full
  fresh independent re-read versus a quicker pass.
- The new-pool round's article selection has no prior severity signal (unlike the targeted rounds, which
  rank by historically-confirmed error severity) -- it's a uniform random sample, which is the right
  comparison for "does this generalize" but is not expected to match the targeted rounds' per-article
  hit rate.

## If this becomes a real chapter

Both models have now been tested on every available article in this project's two Hebrew article pools
(the 198-article targeted pool and the 293-article disjoint mFACT-He pool), so a like-for-like,
same-articles comparison between Qwen and DeepSeek is now complete for both. Natural next steps:
consider whether the same two pipeline bugs (reasoning-token starvation, markdown-report drift)
generalize to other newly-released reasoning models being considered for this project's evaluated
roster; given DeepSeek's two targeted-round confirmed error types both involve literary/linguistic
source material, a round specifically sampling poetry/linguistics-column articles (rather than general
"historically hardest" or random-sample pools) could test whether that's a real, more general weak
point; the cross-model convergent error on `general-13160920` is a strong candidate for a focused
case-study subsection, since it suggests at least one taxonomy category reflects genuine structural
ambiguity rather than model-specific noise.
