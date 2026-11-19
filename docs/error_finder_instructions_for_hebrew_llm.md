# Instructions: Hebrew-Morphology Correctness-Error Finder

**Purpose of this document.** This is a self-contained handoff brief for a *different* language model (ideally one strong in Hebrew) to independently run this project's "Hebrew-morphology correctness-error finder" task — the same task this session ran manually (via Claude acting as the judge) because the project's own API keys for automated error-finding were dead. Everything you need — the real production prompt, the full taxonomy, calibration ground-truth, and where the data lives — is listed below with exact repo paths. All paths are relative to the repo root:

```
/Users/amit.avraham/PycharmProjects/GenAI-Summary-Augmentor-Correctness-Errors
```

---

## 1. What this task is

MorphEvalHeb is a research project (paper in progress, Ben-Gurion University) studying a specific, narrow class of factual error that Hebrew-summarizing LLMs make: errors that occur **only because Hebrew is a morphologically rich language (MRL)** — i.e. an error that could *not* occur in an equivalent English summarizer of the same source facts, because English lacks the grammatical machinery (bound gender/person/number marking, construct-state possession, verb-pattern voice alternation, bound clitic pronouns, etc.) that creates the trap.

The task: given a **source article** (`premise`) and an LLM-generated **summary** (`generative_summary`) of it, find every place where the summary asserts something *factually different* from the source, where that factual change was caused specifically by a Hebrew morphological mechanism — and reject everything else (generic hallucination, wrong dates/numbers/names, lexical substitution, plot confusion), even if it's a genuine factual error, if an English summarizer could have made the identical mistake.

## 2. The real production prompt and schema — READ THESE FIRST, VERBATIM

Do not work from a paraphrase (including this document's summaries below) — read the actual source:

- **`summary_augmentation/correctness_errors_prompts.py`**, function `correctness_errors_finder_prompt()` (lines 104–305). This is the exact Hebrew system+user prompt the project's own pipeline sends to an LLM judge. It defines the taxonomy, the mandatory gate, and the English-contrast test, in Hebrew, in full detail — richer than any English paraphrase.
- **`summary_augmentation/correctness_errors_llm_helper_functions.py`**, function `correctness_error_finder()` and the Pydantic schema classes `CorrectnessError` / `TotalCorrectnessErrors` (around lines 453–476). Your output for each finding must carry these exact fields:
  - `correctness_error_name` — a short, original Hebrew name for the error type
  - `correctness_error_reasoning` — how the Hebrew morphology produced the factual error
  - `mrl_unique_hebrew_features` — which specific Hebrew-only feature(s) are involved (בניין / גוף / מגדר / מספר / סמיכות / ה"א הידיעה / כינוי קניין מוצמד / קליטיק / etc.)
  - `why_its_factual_error` — the actual factual claim that changed
  - `evidence_from_summary` — exact quoted text from the summary
  - `evidence_from_source_article` — exact quoted text from the source
  - `severity` — integer 1–5 (5 = severe, unambiguous, high-stakes factual reversal; 1 = minor/isolated slip)
- **`summary_augmentation/correctness_errors_finder.py`** — the (Windows-path-hardcoded, not directly runnable here) driver script showing how the above two are wired together in production. Useful only as a reference for the call pattern, not for its I/O paths.

### Do NOT use this file as the finder's taxonomy:
- **`summary_augmentation/correctness_errors_examples.py`** is **not** read by the finder prompt above. It is consumed by a *different* script, `correctness_errors_augmentation_v2.py`, which **injects** synthetic errors into clean summaries to build the benchmark's negative examples — the opposite direction (writing errors in, not finding them). HOWEVER: two of its categories describe real Hebrew-morphology mechanisms that are also valid things to search for when judging naturally-generated summaries (see §3 below) — use them as *additional* search categories, not as your primary taxonomy source.

## 3. The full taxonomy to search for

Primary categories (from `correctness_errors_finder_prompt()`):

1. **Binyan (verb-pattern) swaps** — the 7-binyan system (פעל/נפעל/פיעל/פועל/הפעיל/הופעל/התפעל) flips active↔passive↔reflexive↔causative, reversing who did what to whom. Worked example from the prompt itself: "המשטרה עצרה את החשוד" (police arrested the suspect) → "המשטרה נעצרה על ידי החשוד" (police were arrested by the suspect).
2. **Gender/person/number agreement breaks** across a verb/pronoun/adjective chain — a subject's established gender in the source gets the wrong agreeing verb/adjective in the summary.
3. **Possessive clitics (-ו/-ה/-ם/-ן) with antecedent drift** — e.g. "מסקנותיה" (her conclusions) → "מסקנותיו" (his conclusions), silently flipping whose conclusions they are.
4. **Dative/directional clitic pronouns (לו/לה/לי/לך...)** — minimal pairs differing by one letter/vowel, no word boundary — causing referent drift (e.g. "נולדה לו אחות" = a sister was born *to him* → misread as *to her*, reattaching the fact to the wrong person).
5. **Homographic profession/role nouns + gender-agreement chain** — מורה/מרצה/שופט/רופא/מנהל etc. carry no inherent written gender marker; the chain of agreeing verbs/adjectives around them can silently flip.
6. **Directional preposition/clitic stacking** (מ-/ל-/ב-/על/אל) reversing sender/receiver of an action or a source/destination.
7. **Smixut (construct-state) misparsing** and **ה"א הידיעה (definite article)** errors — changing which entity owns/is-identified-with which, or shifting specific↔generic.
8. **Cross-cultural gender mistranslation of foreign names** — a foreign name read as one gender by cultural default, but the real person is the opposite gender; the model "corrects" to the culturally-expected (wrong) gender.
9. **Multi-morphemic word complexity** — a single word packing 7+ morphemes (e.g. "וכשמביתנו" = ו+כש+מ+ה+בית+של+אנחנו) where the wrong segmentation changes a fact.
10. **Extreme morphological ambiguity** — one written word, multiple unrelated valid readings (e.g. "הקפה" = the café / a lap-circuit / "her mouth"; "עצום" = closed [eyes] / huge/immense) — picking the wrong reading inverts a fact.
11. **Proper-noun vs. clitic segmentation ambiguity** — a name that begins with the same letter as a common bound clitic gets misparsed as clitic+shorter-word, inventing a fictitious entity or dropping a real one. *Confirmed real examples this session*: "וליד" (Walid) misparsed as ו- ("and") + ליד; "שהם סמיט" (author name) with the ש stripped as if it were the relative-clause clitic, leaving the nonsense name "הם סמיט".
12. **Verb-tense swaps** (עבר/הווה/עתיד) that contradict other explicit time-markers in the text, creating a chronological contradiction.

Additional categories (from `correctness_errors_examples.py`, dict `hebrew_error_examples` — read lines 944–1142 directly for full detail and worked examples):

13. **`morphological_connective_confusion`** (lines ~1110–1142) — short Hebrew connective words sharing a bound prefix (ש-/כ-/מ-) but **opposite logical meaning** get swapped: שכן ("since/because") ↔ כך ש ("so that") ↔ למרות ש ("despite") ↔ מכיוון ש ("because"); אך ("but/however") ↔ אף ("even/also"). This reverses a cause→result or concession→contrast relationship. (This is the exact mechanism behind the very first error this project's manual review confirmed: a DictaLM 3.0 summary swapped the relativizer clitic ש- for the coordinating clitic ו-, turning "who had been accused" into "and [were] accused," reversing the accusation-before-acquittal order.)
14. **`hebrew_root_pattern_confusion`** (lines ~944–1006) — a word gets swapped for a different one sharing the same binyan/pattern (משקל) or sounding phonetically similar — including a **foreign word that sounds like an unrelated Hebrew verb** (their own worked example: "Performer" → מפרפר, "to flutter/tremble"). Changes the fact, not just style/register.

Treat `idiom_collocation_corruption` (also in that file, lines ~1007–1109) with caution — most of its worked examples are phonetic word-swaps inside a fixed idiom, which could in principle happen in an English idiom too, so apply the English-contrast test extra strictly there.

## 4. Mandatory gate (do not skip)

For **any** finding to count, you must be able to name at least one of these six Hebrew-specific mechanisms as the cause: **בניין (binyan) / גוף-מגדר-מספר (person-gender-number) / סמיכות (smixut) / ה"א הידיעה (definite article) / כינוי קניין מוצמד (attached possessive) / קליטיק (clitic, incl. the connective/root-pattern categories above, which route through these same six)**. If you cannot name one, the finding does not qualify — discard it, no matter how real or dramatic the factual error is.

## 5. Mandatory filter: the English-contrast test

For every candidate finding, ask explicitly: **would this exact error have been possible in an English summarizer working from the same facts?** If yes — reject it. This excludes (even though they are real factual errors): generic entity swaps (wrong person/place/organization name), wrong dates/numbers, hallucinated details, plot/narrative confusion, reference-tracking errors with no morphological mechanism, and spelling errors that don't change meaning. **Only** errors that are *specifically and only possible because Hebrew's morphology works the way it does* count.

Two explicit self-check questions from the real prompt, to ask before reporting anything:
1. Does this change something materially factual (not just style/grammar)?
2. Would a reader come away believing something genuinely different from reality?

Good example (from the prompt itself): "שר החוץ אמר" (the [male] Foreign Minister said) vs. "השרה אמרה" (the [female] minister said) — changes a real identity fact. ✅ counts.
Bad example (from the prompt itself): "הכתבה דנה" vs. "המאמר דן" (the article discussed — feminine "הכתבה" vs masculine "המאמר" word choice) — doesn't change factual content. ❌ doesn't count.

## 6. Calibration — ground-truth confirmed-positive examples (read before judging anything)

**Critical lesson learned this session**: an earlier manual pass by an uncalibrated judge returned a suspiciously-near-zero result on both a test group and a control group, which looked like "models have fixed this" but was actually **judge miscalibration** — the judge was wrongly requiring that the error create *reader-level referent confusion* before counting it, a standard stricter than the project's real one. The fix: before judging anything, read **`thesisdatasets/slides/Amit Avraham_correctness_errors_finder - new open source models_new.md`** — a large markdown file documenting dozens of real, pipeline-confirmed positive examples from the original (gemma3-12b / gpt-oss-20b / qwen3-14b) model runs. At minimum, locate and internalize these three calibration anchors (search the file for the doc_ids):

- **`business-6667074`** (severity 5): a legal claimant's grammatical gender is flipped female→male throughout a summary (בעלת הזכויות / בעלת האגף הדרומי → בעל האגף הדרומי, שבבעלותה → בבעלותו, etc.) — correct even though a human reader could in principle still identify the wing/claim through other means. **Do not require reader-level confusion — a morphological identity-fact reversal counts on its own.**
- **`culture-4533684`** (severity 5): a שמע ("heard"/passive-ish) ↔ השמיע ("caused to be heard"/played) binyan swap in a DJ profile, reversing active-agency (he played music for others) into passive (he merely listened).
- **`food-917568`** (severity 4): a specific named grape varietal, "ענבי שנין בלאן" (Chenin Blanc grapes), is generalized by the summary into plain "ענבים" (grapes) — a specificity loss tied to ה"א הידיעה/definiteness mechanics.
- (If findable) `food-13624004`: a pronoun-referent "cactus spoke" misattribution, severity 3.

## 7. Where the data lives

### 7a. Source articles + historical (old-model) error annotations
`thesisdatasets/error_finder/*.csv` — multiple CSV files, each row an **already-confirmed** correctness error the original production pipeline found in an **old** model's summary (GPT-3.5, Gemini 1.5/2.0, DictaLM 2.0, Gemma-3-12b, gpt-oss-20b, Qwen3-14b), with columns including `doc_id`, `source`, `premise` (the source article), `severity`, and the full error-detail fields from §2's schema. This is the historical ground truth and the source of the calibration anchors in §6. It is also the pool from which "known hard" articles were selected for the targeted current-model tests below (ranked by `(max_severity, n_errors)` per `doc_id`, descending — the hardest, most historically error-prone articles first). The pool has **198 unique articles** total (85 at max-severity 5, 81 at max-severity 4, the rest lower).

### 7b. New summaries generated THIS SESSION from current (2025/2026-era) models — these are what you should error-find on
All under `paper/results/eval/`, one CSV per model/round, columns: `doc_id`, `source`, `orig_max_severity` (the **old** model's historical severity on this article — NOT a property of the new summary, do not confuse the two), `premise`, `generative_summary` (**this is the text to judge**), `model`, `prompt_style`, `gen_seconds`.

| File | Model | Articles (rank range in the 198-pool) | Error-finder status |
|---|---|---|---|
| `current_model_summaries_zai_glm-4_7-flash_targeted.csv` | GLM-4.7-Flash | 1–50 | done, 2/50 found |
| `current_model_summaries_zai_glm-4_7-flash_targeted_round2.csv` | GLM-4.7-Flash | 51–100 | done, 2/50 found |
| `current_model_summaries_zai_glm-4_7-flash_targeted_offset100.csv` | GLM-4.7-Flash | 101–150 | done, 3/50 found |
| `current_model_summaries_zai_glm-4_7-flash_targeted_offset150.csv` | GLM-4.7-Flash | 151–198 (48 articles — pool exhausted) | done, 0/48 found |
| `current_model_summaries_mistral_ministral-3-14b-instruct_targeted.csv` | Ministral-3-14B | 1–50 | done, 0/50 found |
| `current_model_summaries_mistral_ministral-3-14b-instruct_targeted_round2.csv` | Ministral-3-14B | 51–100 | done, 4/50 found |
| `current_model_summaries_mistral_ministral-3-14b-instruct_targeted_round3.csv` | Ministral-3-14B | 101–150 | done, 1/50 found |
| `current_model_summaries_mistral_ministral-3-14b-instruct_targeted_offset150.csv` | Ministral-3-14B | 151–198 (48 articles — pool exhausted) | done, 0/48 found |
| `current_model_summaries_openai_gpt-5_6-luna_targeted.csv` | GPT-5.6-Luna | 1–50 | done, 0/50 found |
| `current_model_summaries_openai_gpt-5_6-luna_targeted_round2.csv` | GPT-5.6-Luna | 51–150 (100 articles) | done, 2/100 found (+1 borderline) |
| `current_model_summaries_qwen_qwen3-32b-v1_0_targeted.csv` | Qwen3-32B | 1–50 | done, 4/50 found |
| `current_model_summaries_dictalm3.csv` + `_round2.csv` | DictaLM 3.0 1.7B-Instruct | **random** (not targeted) sample of 69 MAFAT articles, stratified by source | done, 1/69 found |

A **targeted-regeneration attempt** for DictaLM 3.0 on the hard-article pool was attempted three times at increasing `max_new_tokens` (512→1200→2200) and failed every time (killed by the user for excessive truncation, or died silently from memory pressure) — **DictaLM has zero data on the targeted/hard-article pool.** Only its random-sample result exists.

**If you generate more summaries yourself** (e.g. on the remaining untested rank ranges — Luna has 48 left at 151–198, Qwen3-32B has 148 left at 51–198), use this exact "thin but detailed" Hebrew prompt (the one used for every file above, deliberately minimal in structure but explicit about wanting full detail, per the paper author's request to test natural model behavior without over-engineered scaffolding):

```
סכם את הטקסט הבא בעברית, סיכום מפורט ועשיר בפרטים ובעובדות, לא סיכום תמציתי או מופשט:

{article}
```

### 7c. Record of every confirmed error found this session (worked examples, for pattern-matching)
`docs/experiments.md`, section **§18** ("Do current models still make Hebrew-morphology errors?") documents the random-sample DictaLM result and the judge-miscalibration lesson in full narrative detail, including the exact fix applied. (§19, the targeted-experiment writeup, may or may not exist yet depending on when you're reading this — check the file directly.)

Every individual confirmed finding from this session, with full quoted evidence:

| doc_id | Model | Sev | Mechanism |
|---|---|---|---|
| 218050 | DictaLM 3.0 (random sample) | 2 | ש-/ו- clitic swap (relativizer→conjunction), reverses accusation-before-acquittal order |
| business-6667074 | old-model control | 5 | בעלת→בעל gender reversal (calibration anchor) |
| food-13624004 | old-model control | 3 | pronoun-referent "cactus spoke" misattribution |
| food-14044992 | GLM-4.7-Flash | 3 | pro-drop verb (לימד) misattributed from Alberto Terra ("he") to "Maria" (fem. לימדה) |
| animals-6829845 | GLM-4.7-Flash | 2 | established-masc. dog קירק flipped to fem. throughout |
| culture-10230151 | GLM-4.7-Flash **and** Ministral-3-14B (independently) | 2 | וליד (name) misparsed as ו- + ליד, a real person vanishes from a list |
| general-14268637 | GLM-4.7-Flash **and** Ministral-3-14B (independently) | 3–4 | לו/לה dative-clitic swap fabricates/misattributes a sister's-birth-and-naming story |
| culture-7822389 | Ministral-3-14B | 1 | masc. verb "מספר" for female singer Si Heyman |
| culture-10173038 | Ministral-3-14B | 1 | masc. "התגאה" instead of fem. "התגאתה" for a female band member |
| general-13160920 | GPT-5.6-Luna | 2 | poetry gloss: 1st-person "שהייתי לדודי" → 3rd-person "שהיה לדודה", changes whose dedication it is |
| food-333642 | GLM-4.7-Flash | 4 | masc.-plural clitic חבריו ("his friends") collapses into fem.-singular חברה (also = "company"), swapping identity+number between a burned soldier and the restaurant itself |
| culture-13795813 | GLM-4.7-Flash | 3 | עצום homograph: "closed [eyes]" vs. "huge/immense" — inverts attentive observers into oblivious ones |
| culture-14100072 | GLM-4.7-Flash | 4 | author "שהם סמיט" has the ש stripped (misparsed as relative clitic), inventing fictitious name "הם סמיט" |
| celebrities-8932596 | Qwen3-32B | 2 (×2) | 1st-person gender-neutral "נולדתי"→3rd-person fem. "נולדה" for a male dancer; reflexive clitic עצמי→עצמה (also fem.) on the same man |
| magazine-13224346 | Qwen3-32B | 2–3 | possessive clitic תפקידי→תפקידה (his→her position); separately, root-pattern collision תקנו (ק.נ.ה, "buy/believe") misread as ת.ק.ן ("fix"), turning a flat denial into "refuses to fix" |
| culture-10266649 | GLM-4.7-Flash **and** Ministral-3-14B (independently, different specific error each time — this is a linguistics column about Hebrew errors) | 2–3 | GLM: gendered-numeral agreement (שלוש שקל) left unfixed by the "corrected" example; Ministral: ל+ב+חוץ compound-preposition stack misparsed as ל+מחוץ ("to crush"), an unrelated root |

Note the **two articles that independently broke two different model families the same way** (`culture-10230151`, `general-14268637`) — strong evidence these are genuinely hard Hebrew structures, not random judge noise.

## 8. Output format

For each (premise, generative_summary) pair judged, report: `doc_id | found (yes/no) | severity (1–5 or blank) | category (which of §3's mechanisms) | description (what changed and why) | looks_truncated (yes/no, separate from the error-finding — flag if the summary cuts off mid-sentence/mid-word before reaching a natural end, which limits how much could be checked but is not itself a finding)`.

**Be strict. Do not force findings to hit a quota.** Based on this session's results, the true rate across ~665 current-model summaries tested is **≈2.9%** (19 confirmed). A batch of 25–50 summaries returning 0 findings is a normal, honest, expected result, not a sign you under-searched — many of the batches above genuinely came back clean even under the expanded taxonomy. Conversely, do not reflexively reject something just because it's rare — if it clears the gate and the English-contrast test, report it even at low confidence/severity, with an explicit confidence caveat.
