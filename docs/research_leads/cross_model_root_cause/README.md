# Cross-model root-cause analysis: why every model makes the same Hebrew-morphology errors

**Status**: research lead, based on `paper/results/eval/all_models_all_findings_master.json` (501
confirmed findings across 14 models/generations spanning 2023-2026) plus online research into each
model's architecture, tokenizer, and training-data composition. Not yet incorporated into the
paper/thesis.

**Evidence-strength key used throughout**: every claim below is tagged
**[EXTERNAL]** (a published source: paper, model card, official docs),
**[DATASET]** (computed directly from this project's own 501-finding master JSON), or
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

The best-evidenced claim in this document is Section 1 (the mechanism-distribution consistency,
computed directly from the project's own data) combined with Section 2a (no lab, including the
Hebrew-specialist one, has built a morphology-aware Hebrew tokenizer — independently confirmed across
9 labs' own documentation). Section 2b (the DictaLM "data alone doesn't fix it" control) is a fair
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
