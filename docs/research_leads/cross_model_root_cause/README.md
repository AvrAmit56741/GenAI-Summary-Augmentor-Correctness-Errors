# Cross-model root-cause analysis: why every model makes the same Hebrew-morphology errors

**Status**: research lead, based on `paper/results/eval/all_models_all_findings_master.json` (501
confirmed findings across 14 models/generations spanning 2023-2026) plus online research into each
model's architecture, tokenizer, and training-data composition. Not yet incorporated into the
paper/thesis.

**Evidence-strength key used throughout**: every claim below is tagged
**[EXTERNAL]** (a published source: paper, model card, official docs),
**[DATASET]** (computed directly from this project's own 501-finding master JSON),
**[REPRODUCED]** (directly re-tested by us against the real artifact — e.g. downloading a model's
actual tokenizer and running it, not just citing a claim about it), or
**[INFERENCE]** (my own reasoning connecting the two, not independently verified by a dedicated
source — flagged explicitly so it isn't mistaken for a citation).

## 1. The pattern: mechanism distribution is nearly identical across every model

**Claim**: gender/person/number agreement and possessive clitics are the top-2 error categories for
every model, regardless of lab, scale, or Hebrew-specialization.

**[DATASET] Evidence**: computed by regex-matching each of the 501 findings' `error_type` +
`mrl_feature` + `summary` text against 7 mechanism keyword sets (Hebrew + English terms per category:
e.g. gender_agreement ← `מגדר|feminine|masculine|gender|נקבה|זכר`), then grouping by model. Script:
`paper/results/eval/consolidate_all_findings.py` output + the ad-hoc aggregation run in this session.
Raw counts:

| Model (n findings) | gender_agreement | possessive_clitic | binyan_voice | smixut/definiteness | number_agreement |
|---|---|---|---|---|---|
| DictaLM 2.0 (157) | 41 | 64 | 54 | 38 | 31 |
| Qwen3-14B (83) | 36 | 23 | 22 | 11 | 18 |
| gpt-oss-20B (76) | 27 | 28 | 30 | 14 | 11 |
| GPT-3.5-Turbo (67) | 14 | 25 | 23 | 17 | 11 |
| Gemma-3-12B (67) | 20 | 22 | 32 | 10 | 9 |
| GLM-4.7-Flash (16) | 7 | 8 | 1 | 3 | 2 |
| Ministral-3-14B (10) | 5 | 3 | 2 | — | 2 |
| Qwen3-32B-v1.0 (6) | 4 | 2 | — | — | 1 |
| Qwen3.8-Flash (6) | 4 | 3 | 1 | — | 2 |
| DeepSeek V4 Pro 0813 (4) | 3 | 1 | — | — | 1 |
| Kimi K2.6 (4) | 2 | 3 | 1 | 1 | 2 |

Gender-agreement and/or possessive-clitic appear as the #1 or #2 category in **every row** above
(11/11 models with ≥4 findings; the remaining 3 models — GPT-5.6-Luna, DictaLM 3.0, Claude Haiku 4.5
— each have only 1-3 total findings, too few to rank categories meaningfully, so they're excluded
from this specific claim rather than force-fit).

**Caveat on this evidence [INFERENCE]**: the keyword-tagging is automated and approximate (a single
finding can match multiple categories; I did not manually re-verify all 501 taggings). Treat the
table as indicative of a strong, consistent skew, not an exact/audited count.

## 2. What's shared across every model's design

### 2a. No model has a morphology-aware tokenizer for Hebrew

**Claim**: every model tokenizes Hebrew with a generic subword scheme with no structural concept of
Hebrew's bound clitics, and none of the 9 labs studied report Hebrew-specific tokenizer engineering
comparable to what they report for other scripts.

**[EXTERNAL] Evidence, model by model**:

- **DictaLM 2.0** — built by Hebrew NLP specialists, Mistral-7B-v0.1 base. Extended the vocabulary
  with **~1,000 Hebrew-specific tokens**, cutting tokenization from **5.78→2.76 tokens/Hebrew word**
  (still >2.5x a single-token ideal). New-token embeddings needed embedding-distillation + LM-head
  calibration to limit damage from the vocabulary surgery — an implicit admission that naive
  tokenizer extension is risky. No morphological annotation/supervision used.
  Source: [DictaLM 2.0 model card](https://huggingface.co/dicta-il/dictalm2.0),
  [arXiv:2407.07080](https://arxiv.org/pdf/2407.07080).
- **DictaLM 3.0 1.7B-Instruct** — built on **Qwen3-1.7B-Base**, i.e. it *inherits Qwen3's generic
  tokenizer* rather than reusing DictaLM 2.0's Hebrew-extended one. Continued pretraining mixed
  SlimPajama (general, English-heavy) with 30% Llama-Nemotron post-training data; no Hebrew-content
  percentage was found published for this specific size.
  Source: [DictaLM 3.0 Technical Report, arXiv:2602.02104](https://arxiv.org/abs/2602.02104),
  [featherless.ai model card](https://featherless.ai/models/dicta-il/DictaLM-3.0-1.7B-Instruct).
- **Qwen3 (14B/32B/3.8-Flash)** — byte-level BPE, ~151,669-152K vocab, built by extending
  OpenAI's cl100k_base with Chinese + "other-language" tokens. Hebrew listed as one of 119
  supported languages (grouped under "Afro-Asiatic" with Arabic/Maltese) sharing 36T training
  tokens — no Hebrew-specific benchmark published.
  Source: [Qwen3 Technical Report, arXiv:2505.09388](https://arxiv.org/pdf/2505.09388),
  [Qwen3 blog](https://qwenlm.github.io/blog/qwen3/).
- **OpenAI (GPT-3.5-Turbo → gpt-oss-20B → GPT-5.6-Luna)** — the one *documented, measured*
  cross-generation tokenizer improvement found in this research. cl100k_base (3.5/4-era) inflates
  Hebrew to ~4x the token count of equivalent English (one measurement: 265 vs. 67 tokens for
  matched text), falling back to raw UTF-8 byte-pairs for nearly all Hebrew. o200k_harmony
  (GPT-4o/gpt-oss/GPT-5 generation) cuts this to ~1.34x and has learned some whole Hebrew words
  (e.g. "שלום") as single tokens.
  Source: [token-cost blog measurement](https://yuv.ai/blog/token-cost-of-shalom-experiment),
  [OpenAI community thread](https://community.openai.com/t/tokens-counting-for-hebrew-response-seems-much-higher/268331),
  [gpt-oss model card/arXiv:2508.10925](https://arxiv.org/pdf/2508.10925).
  **Caveat [INFERENCE]**: the 1.34x figure is one informal blog measurement, not a rigorous
  corpus-wide study — treat as suggestive, not precise.
- **Gemma-3-12B** — SentencePiece, ~256-262K vocab shared with Gemini 2.0. Google's own technical
  report states they explicitly rebalanced training data toward lower-resource languages, and the
  model card claims "140+ languages" — but **Hebrew is not named in any of Gemma 3's own
  multilingual benchmark tables** (MGSM, Global-MMLU-Lite, WMT24++, FLoRes all checked).
  Source: [Gemma 3 Technical Report, arXiv:2503.19786](https://arxiv.org/abs/2503.19786),
  [Gemma 3 model card](https://ai.google.dev/gemma/docs/core/model_card_3).
- **Ministral-3-14B-Instruct** — Tekken tokenizer, 131,072 vocab, 100+ languages claimed. Mistral's
  own documentation specifically highlights large compression-efficiency gains for **Korean and
  Arabic** vs. the older SentencePiece tokenizer — Hebrew is never named in any Mistral language
  list found, despite Arabic (a related Semitic language) getting explicit, named engineering
  attention.
  Source: [Mistral tokenization guide](https://docs.mistral.ai/guides/tokenization/),
  [Ministral-3-14B model page](https://docs.mistral.ai/models/ministral-3-14b-25-12.md).
- **GLM-4.7-Flash** — ~30B total/3B active MoE, vocab ≈151.5K (figure confirmed for sibling
  GLM-4.6, same tokenizer family). GLM's own documentation lists **only English and Chinese** as
  officially supported languages.
  Source: [inference.readthedocs.io GLM-4.6 page](https://inference.readthedocs.io/en/latest/models/builtin/llm/glm-4.6.html),
  [llm-stats.com GLM-4.6 launch notes](https://llm-stats.com/blog/research/glm-4-6-launch).
- **DeepSeek V4 Pro 0813** — 1.6T total/49B active MoE. Predecessor V3's tokenizer (byte-level BPE,
  ~128.8K vocab) was extended from Llama's tokenizer with **Chinese-specific merges only**; no
  Hebrew/Arabic-specific extension found for V3 or V4.
  Source: [DeepSeek-V3 paper, arXiv:2412.19437](https://arxiv.org/pdf/2412.19437),
  [architecture gallery, DeepSeek-V4-Pro](https://sebastianraschka.com/llm-architecture-gallery/models/deepseek-v4-pro/).
- **Kimi K2.6** — ~1T total/32B active MoE, 160K vocab (larger than GLM/DeepSeek, per NVIDIA's
  model card) — but no language-coverage breakdown was found; multilingual claims are
  benchmark-only (e.g. SWE-bench Multilingual), not architectural.
  Source: [Kimi K2 paper, arXiv:2507.20534](https://arxiv.org/html/2507.20534v2),
  [Kimi-K2 GitHub](https://github.com/MoonshotAI/Kimi-K2).

### 2a-verify. Direct empirical verification: running the actual tokenizers

Section 2a's claims above were sourced from papers/docs. To check them directly rather than take
them on citation, I downloaded each model's **actual published tokenizer** (not the full model) via
HuggingFace/tiktoken and ran it against real Hebrew words covering every bound-clitic type in the
taxonomy (prefixes ו-/ש-/ל-/ה-, possessive suffixes -ו/-ה). Script and raw output:
`docs/research_leads/cross_model_root_cause/tokenizer_probe.py` and `tokenizer_probe_results.json`
(re-runnable — downloads each tokenizer via `transformers.AutoTokenizer`/`tiktoken`, encodes each
word set, counts tokens; HF downloads are unauthenticated/public tokenizer files only, no model
weights).

This was run **twice, on two independent word sets**, specifically because the first round's test
words all happened to trace back to GPT-3.5-Turbo findings — a fair objection, since the whole point
is a cross-model claim. Round 2 deliberately pulls one word from each of 8 *different* models
instead. Both rounds are kept below in full; round 2 also overturned one specific claim from round 1
(noted explicitly where it happens), which is reported rather than quietly dropped.

#### Round 1 — 8 words, sourced mostly from GPT-3.5-Turbo findings

**Where each test word came from — traced to this project's own confirmed findings where possible,
constructed only where the sample didn't contain that clitic type**:

| Word | Morphological split | Source |
|---|---|---|
| והמפעל | ו-(and) + ה-(the) + מפעל(enterprise) | **Attested** — `article_id=culture-13217869`, GPT-3.5-Turbo smixut/definiteness finding, literal substring of the model's false summary claim |
| ארוסתו | ארוסה(fiancée,fem.) + -ו(his) | **Attested** — `article_id=celebrities-1896448`, GPT-3.5-Turbo "status-upgrade" finding (also appears there as מארוסתו, with an outer מ-"from" clitic stacked on top — we tested the bare form) |
| כלפיה | כלפי(toward) + -ה(her) | **Attested** — `article_id=business-13488077`, GPT-3.5-Turbo beneficiary-reversal finding |
| לחקור | ל-(to, infinitive) + חקור(research) | **Attested** — `article_id=health-13135433`, GPT-3.5-Turbo + DictaLM 2.0 purpose-substitution finding |
| שיכללו | ש-(that) + יכללו(will include) | **Attested** — `article_id=food-13206727`, GPT-3.5-Turbo over-generalization finding |
| ולמנוע | ו-(and) + ל-(to) + מנוע(prevent) | **Attested** — `article_id=health-13135433`, same finding as לחקור above |
| שלה | של(of) + ה(her) | **Attested** — `article_id=celebrities-9255023`, GPT-3.5-Turbo finding ("הפוסט **שלה** הפך למקור השראה") |
| התזקיקים | ה-(definite article) + תזקיקים(distillates,pl.) | **Constructed** — the root תזקיקים is attested in `food-13206727` but only in its indefinite form; the definite form was built by prepending ה- to specifically test the ה"א הידיעה clitic, since no attested finding happened to contain a definite-article example in this sample |

7 of 8 words are literal substrings of real confirmed-error summary text from this project's own
findings; one (התזקיקים) is a constructed minimal pair isolating the definite-article clitic, built
from a root that is itself attested.

**[REPRODUCED] Result — average tokens for the 8-word test set (lower = less fragmented;
1 token/word = ideal, matching one morpheme or the whole word):**

| Tokenizer | Avg tokens/word | Example: ארוסתו ("his fiancée" = ארוסה+‑ו) | Example: שלה ("hers" = של+ה) |
|---|---|---|---|
| cl100k_base (GPT-3.5/4, pre-o200k) | 6.4 | 6 tokens, raw UTF-8 byte fragments (mojibake) | ש / ל / ה (3 tokens) |
| GLM-4.6 (sibling to 4.7-Flash) | 6.4 | 6 tokens, **byte-for-byte identical fragments to cl100k_base** | ש / ל / ה (3 tokens) |
| o200k_base (GPT-4o/gpt-oss/GPT-5) | 3.0 | אר / וס / תו (3 tokens) | ש / לה (2 tokens) |
| DeepSeek-V3 | 3.75 | א / ר / וס / תו (4 tokens) | ש / לה (2 tokens) |
| Ministral (Tekken) | 3.5 | אר / וס / תו (3 tokens) | ש / לה (2 tokens) |
| Qwen3-14B | 3.1 | אר / וס / תו (3 tokens) | ש / לה (2 tokens) |
| Qwen3-32B | 3.1 | **identical to Qwen3-14B on all 8 words** | ש / לה (2 tokens) |
| DictaLM-3.0-1.7B | 3.1 | **identical to Qwen3-14B/32B on all 8 words** | ש / לה (2 tokens) |
| DictaLM-2.0 (Hebrew-extended) | 2.6 | אר / וס / תו (3 tokens — same split as everyone else) | **שלה (1 token, whole word)** |

**Full raw output — all 8 words × all 9 tokenizers** (pieces shown left-to-right in reading order as
each tokenizer actually emitted them; `�` marks a byte-level fragment that split mid-UTF-8-codepoint
and can't render as a character — itself evidence of falling back to raw bytes, not subwords):

| Word (correct split) | cl100k_base | GLM-4.6 | o200k_base | DeepSeek-V3 | Ministral | Qwen3-14B/32B¹ | DictaLM-3.0¹ | DictaLM-2.0 |
|---|---|---|---|---|---|---|---|---|
| **והמפעל** (ו+ה+מפעל) | ו�/�/מ/�/�/ע/ל (7) | *identical* (7) | וה/מפ/על (3) | וה�/��/�ע/ל (4) | וה�/��/�/על (4) | וה/מ/פעל (3) | *identical* (3) | וה/מפ/על (3) |
| **ארוסתו** (ארוסה+ו) | א/ר/ו�/�/ת/ו (6) | *identical* (6) | אר/וס/תו (3) | א/ר/וס/תו (4) | אר/וס/תו (3) | אר/וס/תו (3) | *identical* (3) | אר/וס/תו (3) |
| **כלפיה** (כלפי+ה) | �/�/ל/�/�/י�/� (7) | *identical* (7) | כל/פ/יה (3) | כ/ל�/�/יה (4) | כל/פ/יה (3) | כל/פ/יה (3) | *identical* (3) | כל/פיה (2) |
| **לחקור** (ל+חקור) | ל/ח/�/�/ו�/� (6) | *identical* (6) | ל/חק/ור (3) | ל�/�ק/ור (3) | ל/חק/ור (3) | ל/חק/ור (3) | *identical* (3) | לח/קור (2) |
| **שיכללו** (ש+יכללו) | ש/י�/�/ל/ל/ו (6) | *identical* (6) | ש/יכ/ל/לו (4) | ש/י�/�ל/לו (4) | ש/יכ/ל/לו (4) | ש/יכ/ל/לו (4) | *identical* (4) | שי/כל/לו (3) |
| **ולמנוע** (ו+ל+מנוע) | ו�/�/מ/נ/ו�/� (6) | *identical* (6) | ול/מ/נוע (3) | ול�/��/�וע (3) | ול/מנ/וע (3) | ול/מנ/וע (3) | *identical* (3) | ול/מנ/וע (3) |
| **שלה** (של+ה) | ש/ל/ה (3) | *identical* (3) | ש/לה (2) | ש/לה (2) | ש/לה (2) | ש/לה (2) | *identical* (2) | **שלה (1)** |
| **התזקיקים** (ה+תזקיקים, constructed) | ה/ת/�/�/�/�/י�/�/י�/� (10) | *identical* (10) | הת/זק/יקים (3) | ה�/��/�/ק/יק/ים (6) | ה�/�/ז/ק/יק/ים (6) | הת/ז/ק/יקים (4) | *identical* (4) | הת/זק/יק/ים (4) |

¹ Qwen3-14B, Qwen3-32B, and DictaLM-3.0-1.7B produced byte-for-byte identical token sequences on
every one of the 8 words — reported once per row to avoid redundancy, not merged for any other
reason. Kimi K2.6 and Gemma-3-12B are excluded from this table (see limitations below); neither was
reproducible.

**What this directly confirms**:
- **GLM-4.6's tokenizer handles Hebrew exactly as badly as OpenAI's old pre-o200k tokenizer** — not
  just similarly, but producing the *identical* byte-level fragment sequence on every one of the 8
  test words. This is hard evidence for the Section 2a claim that GLM's own docs list only
  English/Chinese as supported: its Hebrew handling is empirically indistinguishable from a tokenizer
  with zero Hebrew-specific training.
- **DictaLM-3.0-1.7B produces token-for-token identical output to Qwen3-14B and Qwen3-32B on all 8
  words** — direct proof (not just inference from the technical report) that DictaLM 3.0 discarded
  DictaLM 2.0's Hebrew-extended vocabulary and inherited Qwen3's generic tokenizer unchanged.
- **o200k_base's real improvement over cl100k_base is confirmed** (6.4→3.0 avg tokens/word, a ~2.1x
  cut) — same direction as the previously-cited informal blog figure (4x→1.34x token inflation), on
  an independent word set. The exact ratio differs because the word sets differ; both point the same
  way.
- **DictaLM 2.0 never isolates a bound clitic as its own token, except one lucky whole-word hit.** It
  gets "שלה" as one whole-word token (likely memorized as a frequent word), but for every other test
  word it splits exactly like the generic tokenizers, fusing the possessive suffix into an arbitrary
  chunk (e.g. "ארוסתו" → אר/וס/**תו**, not ארוסה+ו). On this round's average (2.6 tokens/word) it comes
  out ahead of everyone else — see Round 2 below for why that specific ranking does not hold up as a
  general claim.
- A striking illustration: **"שלה" (של="of" + ה="her")** is split as ש/לה by 6 of 7 tokenizers tested
  — cutting through the middle of the word "של" and fusing the possessive ה onto the wrong neighbor.
  That is the literal opposite of the correct morphological boundary, on the single most common
  possessive clitic word in Hebrew.

**Limitations, stated plainly**: Kimi K2.6's tokenizer (`moonshotai/Kimi-K2-Instruct`) ships as a
`tiktoken.model` file loaded through custom Python code in the model repo; loading it would have
required executing untrusted remote code, which was declined for security reasons — so Kimi was
**not** directly tested, only its documented vocab size (160K, no language breakdown) is cited above.
Gemma-3-12B's tokenizer is gated on HuggingFace and could not be downloaded without a granted
license — also **not** directly tested. Both are marked as untested rather than assumed to behave
like the others. (Both limitations repeat in Round 2 below, for the same reasons.)

#### Round 2 — 8 words, one from each of 8 *different* models (zero from GPT-3.5-Turbo)

To answer the fair objection that round 1 leaned on one model's findings, every word below is a
literal substring of a different model's confirmed false-summary text (Qwen3-32B-v1.0's is quoted
from its finding's documented description, since the exact source/false-claim strings weren't
preserved in that particular CSV — noted below, not glossed over):

| Word | Morphological split | Source (model, article, literal quote) |
|---|---|---|
| מייסדה | מייסד(founder) + -ה(her/its) | **DictaLM 2.0**, `magazine-11028025` — "...בזכות חזונו של **מייסדה**, אלי הורביץ..." |
| החניות | ה-(definite article) + חניות(parking spots) | **Gemma-3-12B**, `business-7851243` — "**החניות** מהוות עסק מניב..." |
| וגרנאש | ו-(and) + גרנאש(Grenache) | **gpt-oss-20B**, `food-8480186` — "...ענבי טמפרניו **וגרנאש** לבן..." |
| שהעניק | ש-(that) + העניק(he granted) | **Qwen3-14B**, `food-14044992` — "...ידיד משפחה **שהעניק** לה מגורים..." |
| עורכו | עורך(editor) + -ו(his) | **GLM-4.7-Flash**, `news-3418430` — "**עורכו** של נביל ב׳: דרק ווירסום..." |
| להתאים | ל-(infinitive) + התאים(to suit/match) | **Ministral-3-14B**, `culture-10230151` — "...שונה לשם דיין (כדי **להתאים** לשם ישראלי)" |
| תפקידה | תפקיד(position) + -ה(her) | **Qwen3-32B-v1.0**, `magazine-13224346` — quoted from the finding's error description ("תפקידי → **תפקידה**"), exact false_claim string not preserved in its source CSV |
| התאבדותה | התאבדות(suicide) + -ה(her) | **Kimi K2.6**, doc `9806` — "...ללא ציון **התאבדותה** ומניעיה" (this is literally this project's own confirmed Kimi K2.6 finding) |

**[REPRODUCED] Result — average tokens for this 8-word set:**

| Tokenizer | Avg tokens/word |
|---|---|
| Qwen3-14B / Qwen3-32B / DictaLM-3.0-1.7B (identical) | 3.00 |
| o200k_base (GPT-4o/gpt-oss/GPT-5) | 3.13 |
| DictaLM-2.0 (Hebrew-extended) | 3.25 |
| Ministral (Tekken) | 3.63 |
| DeepSeek-V3 | 4.25 |
| cl100k_base (GPT-3.5/4, pre-o200k) | 6.50 |
| GLM-4.6 (sibling to 4.7-Flash) | 6.50 |

**Full raw output — all 8 words × 6 tokenizer groups** (`�` = raw UTF-8 byte fragment, not a
character):

| Word | cl100k/GLM-4.6¹ | o200k_base | DeepSeek-V3 | Ministral | Qwen-family² | DictaLM-2.0 |
|---|---|---|---|---|---|---|
| **מייסדה** | מ/י/י�/�/ד/ה (6) | מ/ייס/דה (3) | מ/יי�/�ד/ה (4) | מי/יס/דה (3) | מי/יס/דה (3) | מיי/ס/דה (3) |
| **החניות** | ה/ח/נ/י/ו�/� (6) | ה/ח/ניות (3) | ה�/��/�יות (3) | ה�/�/ניות (3) | הח/ניות (2) | הח/ניות (2) |
| **וגרנאש** | ו�/�/ר/נ/א/ש (6) | וגר/נ/אש (3) | וג/ר�/��/�ש (4) | וגר/נ/אש (3) | וג/ר/נא/ש (4) | ו/גר/נ/א/ש (5) |
| **שהעניק** | ש/ה/ע/נ/י�/� (6) | שה/ענ/יק (3) | ש�/��/��/�/יק (5) | ש/ה�/�/נ/יק (5) | שה/ע/ניק (3) | שה/ע/נ/יק (4) |
| **עורכו** | ע/ו�/�/�/�/ו (6) | ע/ור/כו (3) | ע/ור/כ/ו (4) | ע/ור/כו (3) | עור/כו (2) | ∅/עור/כו (3)³ |
| **להתאים** | ל/ה/ת/א/י�/� (6) | לה/ת/אים (3) | ל�/��/��/�ים (4) | ל/ה�/�/אים (4) | לה/ת/אים (3) | להת/אים (2) |
| **תפקידה** | ת/�/�/�/�/י�/�/ה (8) | ת/פק/ידה (3) | ת/�/�ק/ידה (4) | ת/פק/ידה (3) | ת/פק/ידה (3) | ת/פקי/דה (3) |
| **התאבדותה** | ה/ת/א/ב/ד/ו�/�/ה (8) | הת/א/בד/ותה (4) | ה�/��/��/�ד/ות/ה (6) | ה�/�/א/בד/ותה (5) | התא/בד/ות/ה (4) | הת/אב/דות/ה (4) |

¹ cl100k_base and GLM-4.6 are byte-for-byte identical on all 8 words again — the finding replicates
on a fully independent word set. ² Qwen3-14B, Qwen3-32B, and DictaLM-3.0-1.7B are byte-for-byte
identical on all 8 words again — also replicates. ³ DictaLM-2.0 emits a leading empty token before
"עור" (a SentencePiece word-boundary artifact, not a morpheme); counted as reported by the tokenizer.

**What round 2 confirms, and what it overturns from round 1**:
- **Both headline replications hold on a fully independent, 8-model-sourced word set**: GLM-4.6 ≡
  cl100k_base again (byte-for-byte), and Qwen3-14B/32B ≡ DictaLM-3.0 again (byte-for-byte). This is
  now two-for-two, not a one-off.
- **o200k_base's improvement over cl100k_base replicates** (6.50→3.13, a ~2.1x cut — the same ratio as
  round 1's 6.4→3.0, despite a completely different word set).
- **Overturned claim**: round 1 reported DictaLM 2.0 as the single best performer. On this word set
  it is **not** — Qwen3-14B/32B/DictaLM-3.0 (3.00) and o200k_base (3.13) both beat it (3.25). DictaLM
  2.0's ~1,000-token Hebrew extension does not produce a reliable, systematic fragmentation advantage;
  it wins or loses depending on which specific words happen to be in its extended vocabulary. This is
  reported as a correction, not smoothed over — and it strengthens, not weakens, the root-cause
  argument: even the one model built specifically for Hebrew doesn't show a *consistent* edge, which
  is harder to explain by "just needs a bit more data" than a consistent-but-partial edge would be.
- **Still zero clean clitic isolations as a rule.** Across both rounds (16 words, 7 tokenizers each),
  the only cases where a tokenizer's piece boundary happens to land on a real morpheme boundary are
  single-digit exceptions scattered across different tokenizers on different words (e.g. DictaLM-2.0
  isolating the ו- in וגרנאש; several tokenizers' last piece in התאבדותה happening to be the bare
  suffix ה) — consistent with incidental byte/frequency alignment, not a deliberate morphological
  rule, since the same tokenizers fail to isolate the *same* clitic type on other words in the set.

**Limitations, stated plainly (same two models, both rounds)**: Kimi K2.6's tokenizer
(`moonshotai/Kimi-K2-Instruct`) ships as a `tiktoken.model` file loaded through custom Python code in
the model repo; loading it would have required executing untrusted remote code, which was declined
for security reasons — so Kimi was **not** directly tested, only its documented vocab size (160K, no
language breakdown) is cited above. Gemma-3-12B's tokenizer is gated on HuggingFace and could not be
downloaded without a granted license — also **not** directly tested. Both are marked as untested
rather than assumed to behave like the others.

### 2b. The control case that rules out "just needs more Hebrew training data"

**Claim**: DictaLM 2.0's much denser Hebrew training-data share does not translate into a
qualitatively different error profile from data-starved models — pointing at the tokenizer/
architecture layer, not data volume, as the dominant bottleneck.

**[EXTERNAL] Evidence (data density)**: DictaLM 2.0's continued pretraining is reported as
~190-200B tokens at roughly a 50/50 Hebrew/English split (model card/arXiv:2407.07080) — by far the
highest Hebrew-content share of any model in this study. By contrast Qwen3's 36T training tokens
cover 119 languages with no stated Hebrew allocation (arXiv:2505.09388); Gemma 3, Ministral, GLM,
DeepSeek, and Kimi all show the same pattern — Hebrew present only as an unweighted entry in a large
language list, never a stated percentage.

**[DATASET] Evidence (error profile)**: from the Section 1 table, DictaLM 2.0's top categories
(possessive_clitic 64, gender_agreement 41, binyan_voice 54, out of 157) are proportionally the same
shape as e.g. gpt-oss-20B (binyan_voice 30, possessive_clitic 28, gender_agreement 27, out of 76) and
Qwen3-14B (gender_agreement 36, possessive_clitic 23, binyan_voice 22, out of 83) — all three show
gender-agreement and possessive-clitic within the top 2-3 categories and binyan/voice close behind,
despite wildly different Hebrew training-data shares. **[INFERENCE]**: if data volume alone were the
dominant lever, DictaLM's distribution should look meaningfully flatter/different; it doesn't, by
inspection of the proportions above. This is suggestive, not a controlled/statistically-tested
comparison (no significance test was run on these proportions).

### 2c. Hebrew agreement as a long-range dependency — weakest-evidenced claim, flagged explicitly

**Claim**: Hebrew's agreement-propagation (set a referent's gender once, maintain it across every
later pronoun/clitic/verb) is a long-range dependency that transformer attention handles imperfectly,
and this is architecture-independent.

**[INFERENCE] — this is the one claim in this document NOT backed by an external source.** I ran a
dedicated follow-up search specifically to check this claim rather than assert it on priors; see the
"Verification follow-up" subsection below for the result once it lands. Within this project's own
dataset, several confirmed findings do involve a referent established early and misgendered many
sentences later (e.g. the Kimi K2.6 `2221137` Pizza Tower finding — a gender flip sustained across
~6 sentences/clauses of summary; the round-1 Kimi `9806` Chaim Walder finding — a single-token clitic
slip, by contrast, showing the mechanism isn't *only* long-range). This is dataset-internal
illustration, not proof that distance-from-antecedent is the causal variable — a rigorous version of
this claim would need to measure error rate as a function of textual distance from the antecedent
across the full 501-finding set, which was not done here.

### 2d. A separate, non-linguistic shared pattern: reasoning-token starvation

**Claim**: Qwen3.8-Flash, GLM-4.7-Flash, and Kimi K2.6 all silently spend their entire
completion-token budget on hidden "thinking" tokens unless reasoning is explicitly disabled — a
shared side-effect of 2025/2026-era "thinking-by-default" training, not Hebrew-specific.

**[DATASET] Evidence (this project's own pipeline, first-hand)**: Qwen3.8-Flash and Kimi K2.6 both
required `reasoning: {"enabled": false}` added to the OpenRouter API payload in
`paper/results/eval/exp_current_model_summaries_openrouter_targeted.py` after an unconfigured test
batch returned mostly-empty outputs with `reasoning_tokens` consuming the full token budget (Kimi:
4/5 outputs empty, `reasoning_tokens` = 3000/3000/2999/2983 against a `max_tokens=3000` cap — logged
earlier this session).

**[EXTERNAL] Evidence (GLM, independent confirmation)**: an LM Studio bug-tracker report describes
the identical failure mode for GLM-4.7 — thinking consuming the entire token budget, leaving content
empty — and notes `enable_thinking: false` did **not** reliably fix it in that report, meaning the
parameter name/behavior isn't fully standardized across providers.
Source: [lmstudio-ai/lmstudio-bug-tracker#1990](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1990),
[Z.ai thinking-mode docs](https://docs.z.ai/guides/capabilities/thinking-mode),
[Kimi platform docs](https://platform.kimi.com/docs/guide/use-kimi-k2-thinking-model).

## 3. Caveats on the raw counts

- **[DATASET]** The 450 `historical_pool` findings (GPT-3.5-Turbo, DictaLM 2.0, Gemma-3-12B,
  gpt-oss-20B, Qwen3-14B) come from the project's original automated pipeline, run *before* the
  strict MRL-only taxonomy/mandatory-gate/English-contrast-test methodology was calibrated (see
  `docs/error_finder_instructions_for_hebrew_llm.md` section 6). Evaluation-set sizes also differ by
  an order of magnitude between tiers (e.g. Qwen3-14B's 83 findings vs. Qwen3.8-Flash's 6, from a
  ~149-article current-model-check). The mechanism *distribution* (Section 1 table) is the robust
  cross-model signal; the raw counts are not a fair apples-to-apples rate comparison.
- **[EXTERNAL]** Claude Haiku 4.5's "only 1 error, lowest of any model" result is flagged as a
  likely methodological artifact by the dedicated research pass on it: its sample size as a
  *summarizer* in this project is tiny relative to how heavily the same model family was used as the
  *judge* throughout, and the LLM self-critique literature shows no reliable transfer between judging
  skill and generation skill — one cited result even found a 36% accuracy *drop* when a reasoning
  model (DeepSeek-R1) critiqued its own output vs. another model's.
  Source: [self-eval attention study](https://arxiv.org/html/2606.20093),
  [CoT error-detection study](https://arxiv.org/pdf/2502.19361),
  [Claude Haiku 4.5 System Card](https://assets.anthropic.com/m/99128ddd009bdcb/Claude-Haiku-4-5-System-Card.pdf).
- **[EXTERNAL]** No lab in this study publishes a Hebrew-specific training-data percentage or a
  Hebrew-specific benchmark score (checked for all 9 labs above). A 2026 Bar-Ilan paper on Hebrew
  coreference (KibutzR) found LLMs generally perform far worse on Hebrew than English, and worse
  still on unsegmented/raw Hebrew text, with smaller *encoder* models beating decoder LLMs on Hebrew
  coreference specifically — though it does not test any of the 14 models in this study by name.
  Source: [KibutzR, arXiv:2604.17108](https://arxiv.org/pdf/2604.17108).

## 4. Bottom line

**The root cause, stated plainly: this is a representation problem, not a data or scale problem.**
Every model studied — 14 systems, 9 labs, 1.7B to 1.6T parameters, 2023 to 2026, including a model
built specifically for Hebrew — tokenizes Hebrew the same flawed way: generic BPE/SentencePiece with
no concept of Hebrew's bound morphology (the ש-/ו-/ל-/ה- prefixes and -ו/-ה/-ם suffixes that attach
to a word with zero whitespace boundary). None of the 9 labs' own documentation shows morphological
supervision or Hebrew-aware segmentation anywhere in their tokenizer design (Section 2a). So every
model is forced to track a word's gender, person, and definiteness as a side-effect of a subword
split that was never told those boundaries exist — which is why the error signature is identical
everywhere: gender/person/number agreement and possessive-clitic errors rank #1 or #2 for every model
with enough data to rank (Section 1). The DictaLM control (Section 2b) rules out "just needs more
Hebrew data" as the fix: the Hebrew-densest model in the study shows the same error shape as
data-starved models, because the bottleneck sits one layer below the data, at tokenization — the
model never sees "this is a gendered possessive clitic on a feminine noun," only an arbitrary
frequency-learned fragment. The other two findings in this document (long-range dependency, Section
2c; reasoning-token starvation, Section 2d) are secondary: the first is a plausible but unproven
contributing mechanism, the second is an unrelated API/engineering bug — neither is the answer to why
this happens across every lab.

The best-evidenced claim in this document is Section 1 (the mechanism-distribution consistency,
computed directly from the project's own data) combined with Section 2a/2a-verify (no lab, including
the Hebrew-specialist one, has built a morphology-aware Hebrew tokenizer — confirmed both by 9 labs'
own documentation AND by directly downloading and running 7 of those tokenizers ourselves against
real clitic words, which is the strongest evidentiary tier in this document: not a citation, a
reproduction). Section 2b (the DictaLM "data alone doesn't fix it" control) is a fair
but informal comparison, not a statistically controlled one. Section 2c (long-range dependency as
the mechanism) is the weakest-evidenced claim here and should be read as a plausible hypothesis, not
an established finding, pending the verification follow-up below. Section 2d (reasoning-token
starvation) is a solid, independently-confirmed-for-GLM pattern, but is an orthogonal
engineering/API issue, not part of the Hebrew-morphology root-cause story itself.

### Verification follow-up (added after a dedicated check of claim 2c)

A dedicated search was run specifically to check whether claim 2c is literature-backed or
overstated. **Result: partially supported, and the document's phrasing above has been hedged to
match.** What the literature actually supports:

- **[EXTERNAL] Real support, for a narrower claim than "long-range across a summary":**
  Tran et al. (2018) found Transformer LMs underperform RNNs on subject-verb agreement specifically
  as subject-verb *distance* grows ([arXiv:1808.08946](https://arxiv.org/pdf/1808.08946)).
  Attractor-interference studies show agreement accuracy drops toward chance when an intervening
  noun of mismatched number/gender sits near the verb (Zacharopoulos et al., EMNLP 2023,
  [arXiv:2311.16978](https://arxiv.org/pdf/2311.16978); BLiMP,
  [arXiv:1912.00582](https://arxiv.org/pdf/1912.00582)). A TACL probing study found nested
  long-range dependencies go from near-perfect (short-range) to below-chance (long-range) accuracy
  ([arXiv:2212.04523](https://arxiv.org/pdf/2212.04523v2.pdf)).
- **[EXTERNAL] What is NOT supported:** all of the above is *sentence-internal* syntactic agreement
  (a few words to one clause apart) — not document-length referent-gender tracking across multiple
  sentences in generated summaries, which is the actual claim this project needs. "Lost in the
  middle" (Liu et al., [arXiv:2307.03172](https://arxiv.org/abs/2307.03172)) concerns
  retrieval/QA position bias, not morphology, and no connection to agreement degradation was found.
  No paper was found that measures LLM coreference/agreement accuracy as a direct function of
  textual distance during *generation* (two benchmarks name distance as "a difficulty factor" with
  no accuracy-by-distance curve:
  [arXiv:2411.07466](https://arxiv.org/html/2411.07466v2),
  [arXiv:2510.25087](https://arxiv.org/html/2510.25087v1)).

**Revised framing**: "known weak spot for transformer attention" is fair for short-range/attractor-
type agreement (word-to-word or clause-to-clause distance), which is externally well-evidenced.
Extending that mechanism to multi-sentence Hebrew referent-gender tracking during summarization — as
Section 2c does — is a **plausible analogy, not an established, literature-backed fact**. This
document now treats claim 2c as a hypothesis worth stating, not a finding on the same evidentiary
footing as Sections 1, 2a, or 2d.
