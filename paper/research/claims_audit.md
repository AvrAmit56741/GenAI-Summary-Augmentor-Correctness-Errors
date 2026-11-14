# MorphEvalHeb: claims audit (2026-11-13)

Every factual claim about the data, pipeline and models, in the original paper and in my later edits, checked against the repo. Paths are relative to the repo root.

Status: ✅ verified · ⚠️ partly true or ambiguous · ❌ contradicted · ❔ not verifiable from the repo (author knowledge needed)

## A. Data

| Claim | Status | Evidence |
|---|---|---|
| Benchmark = mFACT test + MAFAT augmented + rewrites | ✅ | `thesisdatasets/.../benchmark/combination_summary.txt`: `test_mfact.csv` 2,135 + `bk_combined.csv` 430 + `mafat_combined.csv` 1,459 = 4,024 |
| MAFAT reference summaries are human-written | ✅ (author) | You stated this. `docs/augmentation_project_structure_docs.md` §Final Combined Dataset: "Annotation: Human gold", "Source Language: Native Hebrew" |
| mFACT-He = journalist-written XSum summaries with natural errors and silver labels | ✅ | mFACT paper (`docs/parsed_papers/docling/Detecting and Mitigating Hallucinations…`) §2.1: XSum pairs ranked by 4 English metrics, top-k/bottom-k → positive/negative. Benchmark mFACT rows have no `error_types` (0 of 1,058 negatives) |
| mFACT translated into Hebrew with GPT-4o-mini | ✅ | `archive/mfact_translate.py`, `model="gpt-4o-mini"`. `archive/azure_translate.py` uses gpt-4o, so check which script produced `test_mfact.csv` |
| mFACT data was also augmented ("Natural + Augmented") | ⚠️ | Docs §Phase 2 and §Final table say so for training. The benchmark mFACT rows carry no augmentation metadata, so they look like original pairs |
| Article-level split, done before augmentation | ✅ (docs) | Docs §Data Leakage Prevention: "MAFAT: Document-level split enforced", "Augmentation: Performed separately on train/val/test". Code: `main()` reads per-split input folders |
| `combined.csv` is disjoint from the binary verifier's training data | ❔ | Training file is on the server only. Binary test split and `combined.csv` share 8 pairs |
| MAFAT-Aug median lengths: article 1,034 / reference summary 142 words | ✅ | Computed from `combined.csv` |

## B. Augmentation pipeline

| Claim | Status | Evidence |
|---|---|---|
| Sentence-level claim decomposition | ✅ | `config.py` `CLAIM_DECOMPOSITION_METHOD = SENTENCES`; `utils.text_decomposition` regex split |
| 26 families defined, 20 enabled (11 Hebrew LLM + 4 general LLM + 4 NER + number rule) | ✅ | `config.py` `TRANSFORMATION_CONFIG`; `combined.csv` contains exactly 20 families |
| Original paper: "24 active families" | ❌ | 17 + 3 + 6 = 26 defined; 20 enabled |
| Original paper: "6 NER-based entity substitutions" | ❌ | Only person, location, organization and title use NER; event and date are LLM-based (`general_llm_transformations`) |
| NER entities replaced by a same-type entity from the source article, Levenshtein ratio < 0.57 | ✅ | `correctness_errors_augmentation_v2.py` `apply_factuality_transformation_on_claims` |
| Numbers doubled by rule | ✅ | Same function, `NUMBER_SWAP` branch |
| m = max(1, ⌊ρT⌋), allocations capped at 20,000, negation not combined on a claim | ✅ | `utils.control_claims_after_transformation`, `apply_transformation_restrictions` |
| At most one edit per family per claim; LLM edits preferred | ✅ | `utils.recompose_transformed_summary` |
| Diversity filter: max(unigram, bigram Jaccard) ≤ 0.85 + 0.0013·\|E\|^0.75 | ⚠️ | `utils.filter_diverse_summaries`. Bug: the threshold accumulates across candidates. Fallback keeps all candidates if none pass |
| 51 candidates, at most 7 kept | ✅ | `if count > 50: break`; `final_summaries[:7]` |
| ρ = 0.9, θ₀ = 0.85 for the dataset | ⚠️ | `git log` of `config.py`: 0.85/90 from 2025-10-08 to 2025-12-13, then 50 (12-13 to 12-30), then 90 again. Classifiers were trained 2025-12-01 to 12-17, so the data most likely used 0.85/90. README (0.93/95) is outdated |
| Span selection by Gemini; rewriting by Gemini (5 Hebrew families) / GPT for the rest | ⚠️ | Current `config.py`. The per-family mapping was introduced 2025-11-21. Before that (10-08 to 11-21) all stages used OpenAI. Committed defaults: gpt-4o + gemini-2.0-flash until 2025-11-16, then gpt-4.1 + gemini-2.5-pro. The exact model per generation batch is not recorded (no API logs in the repo) |
| Grammar verification step | ❌ | README step [4] claims it. `verify_transformed_summary_grammar` is imported but never called anywhere in the repo |
| Positives: one LLM prompt for back-translation and paraphrase with fact-preservation rules | ✅ | `correctness_errors_prompts.faithfulness_summary_back_translation_and_paraphrasing`; called from `correctness_errors_augmentation_bk_paraph.py` |

## C. Taxonomy origin (author knowledge)

| Claim | Status | Evidence |
|---|---|---|
| About 300 summaries from DictaLM 2.0, GPT-3.5, Gemini 1.5; errors flagged by Claude 4 Sonnet and Gemini 2.5 Pro; severity 4–5 verified by a native speaker | ❔ / ⚠️ | Thesis proposal (supplied by you). `thesisdatasets/error_finder/` contains Gemini and Claude finder runs on GPT-3.5, DictaLM 2.0, Gemma, gpt-oss and Qwen summaries; there are no Gemini 1.5 summary files |
| Each error type evaluated by a Hebrew linguist | ❔ | Original paper text; no artifact in the repo |

## D. Models

| Claim | Status | Evidence |
|---|---|---|
| Binary: NeoDictaBERT, labels [contradiction, entailment] (index 1 = faithful) | ✅ | HF `training_info.txt` |
| Binary split sizes 18,238 / 2,239 / 2,918; 1 epoch; best checkpoint at step 1,000 | ✅ | HF `all_results.json`, `trainer_state.json` |
| Binary test results: acc 96.6, AUC 0.990 | ✅ | Recomputed from HF `predictions.csv` (`paper/results/analyze_hf_results.py`) |
| README/model card "96.78% / 96.20 F1" | ⚠️ | A different run. The released checkpoint's test result is 96.64 / 96.02 |
| Binary trained on MAFAT-Aug + mFACT-He | ✅ (docs) | Docs §Phase 3: "each split combines MAFAT + MFACT samples" |
| Setup 1: article + summary, 4,096 tokens | ✅ | Docs; `training_args.txt`. Not on HF |
| Setup 2 input = article + claim (my wording in the paper) | ❌/⚠️ | Docs (`classifier_models_docs.md`) and the HF model card say premise = **original correct claim**. The server endpoint takes (article, claim); the training file sits in `articleclaimlevel/` with max length 3,072. **Needs the author to confirm** |
| Setup 3 input = top-k evidence + claim (my wording; docs' design) | ❌ for the released checkpoint | HF `script_arguments.json` train = `parse_orgclaim_augclaim/train.csv`; `parse_evidence_claim_level.py` writes premise = original claim; the training script has 0 references to evidence. Docs describe original claim + MiniLM evidence (top-3, ≥ 0.5) as the intended design |
| Setup 3 micro-F1 0.787 | ✅ but oracle | HF `traning_logs`. The test pairs are (original claim, corrupted claim), so this is not a reference-free setting |
| "mFACT" baseline | ⚠️ | It is **NeoDictaBERT fine-tuned on mFACT-He** (metrics page text; server `MFACT_MODEL_PATH`), not the original mFACT model. The paper's Baselines section does not describe it |
| BARTScore baseline | ⚠️ | `inference_api_server/models.py`: `facebook/bart-large-cnn` (English) applied to Hebrew |
| QAFactEval hybrid uses DictaBERT parsing + XLM-R SQuAD2 + GPT-4o-mini | ✅ | `pages/metrics/qa_facteval.py` (`dicta-il/dictabert-syntax`, XLM-RoBERTa-squad, OpenAI) |
| Zero-shot judge: GPT-4o-mini, 1–5 score normalized | ✅ | `pages/metrics/llm_as_a_judge/vanila_zero_shot.py` |

## E. Results in the original paper

| Claim | Status | Evidence |
|---|---|---|
| MorphEvalHeb τ = 0.6996 | ✅ | Re-run: τ_b = 0.689 on `combined.csv` |
| BERTScore τ = 0.0162 | ✅ approx. | Re-run: −0.061 (heBERT) / −0.013 (AlephBERT) |
| ROUGE-1 τ = 0.1391 | ❌ | Re-run with repo code (summary vs article): −0.093 |
| "agreement with human annotations" | ❌ | No human labels; corrected in the paper |
| Intro example: ROUGE-1 ≈ 0.94, BERTScore ≈ 0.96 | ❌ not reproducible | Repo code on the sentence pair: ROUGE-1 F1 0.53, BERTScore 0.81 (heBERT) / 0.82 (AlephBERT) |

## F. Empirical check of the multi-label setups as served (MAFAT-Aug, 1,889 summaries; `paper/results/eval/cache/ml_setup*.csv`)

| Setup (server input) | AUC of max family prob (unfaithful vs faithful) | Mean max prob, faithful / unfaithful | Family recall @0.5 |
|---|---|---|---|
| Setup 2 (article + claim) | 0.844 | 0.986 / 0.995 | 0.860 |
| Setup 3 (top-3 evidence + claim) | **0.517 (chance)** | 0.994 / 0.995 | 0.875 |

Setup 3 at chance with evidence input is consistent with D: the released checkpoint was trained on (original claim, corrupted claim) pairs, so it does not transfer to evidence-based input. Both setups predict at least one error family for nearly every summary, faithful ones included.

## G. Reference-summary provenance (found 2026-11-13)

| Claim | Status | Evidence |
|---|---|---|
| MAFAT references are human-written | ❌ partly | `summarization-7-heb.jsonl` `metadata.type`: 2,373 gold (annotator-written) and 2,995 silver (LLM draft in `metadata.ai_summary`, annotator-reviewed; 1,018 identical to the draft). Benchmark's 209 references: 77 gold, 89 silver edited, 43 silver unedited. Paper corrected (§4.1, §6.1, Labels, Limitations) |
| Figure 1 = errors in LLM-generated summaries | ⚠️ | `summaries_*.jsonl` keep the reference in `summary` and the LLM outputs in `metadata.generative_summaries`; 68 of 254 analyzed summaries (27%) equal the MAFAT reference (gpt-oss 18/44, Qwen3 10/23, Gemma 11/42, DictaLM 27/101, GPT-3.5 2/44). Recompute on LLM outputs only |
