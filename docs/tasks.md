# MorphEvalHeb: tasks

Last updated: 2026-11-13. ACL 2027 (ARR) deadline: **2027-01-04**.

Status: ✅ done · ⏳ running · 🔜 ready to start · ⛔ blocked (needs input) · 💡 optional idea

---

## 0. Original tasks (kept as written)

1. complete lecture review + intro: ✅ Related Work rewritten and new citations added (`paper/research/MorphEvalHeb literature review update.md`); Introduction updated.
2. compare metric:
   - ngram: ✅ ROUGE-1/2/L, BLEU, chrF, Token F1 recomputed on the benchmark (Table 1)
   - neural basic: BERTScore similarity: ✅ heBERT and AlephBERT recomputed (Table 1)
   - SOTA: QGQA: ⛔ QAFactEval on hold (see 1.2)
3. update lecture review with new relevant metrics: ✅ (see section 1 for the metrics still to run)
4. how to extract facts: open. Claims are currently sentence-split (`CLAIM_DECOMPOSITION_METHOD = SENTENCES`).
5. new metric - full LLM-as-a-judge: open. See 1.5 (DictaLM 3.0 judge) and 1.6 (M-Prometheus).

---

## 1. Metrics to run (each fills a row in Table 1 and in the appendix table)

All scores go through `paper/results/eval/run_eval.py` → `scores.csv` → `analyze.py` → tables.
Benchmark: `thesisdatasets/train_datasets/Correctness/data/benchmark/combined.csv` (4,024 pairs).

| # | Metric | Why it matters | Effort / cost | Status |
|---|---|---|---|---|
| 1.1 | **FactCG** (Lei et al., NAACL 2025) | Current best small trained verifier; the closest design to ours | Easiest; local, free, minutes | ⏳ Running. Wired into the shared module `pages/metrics/factcg.py`, the server (`/factcg/predict`) and the GUI (FactCG tab). Table rows pending |
| 1.2 | **QAFactEval** (hybrid) | Described in §6.2; in the original paper | ~55k OpenAI calls, ~$3–5 (hybrid) / ~$15–25 (fully generative) | ⛔ **On hold: the author will give instructions.** Run spec is in the `% TODO` above `\paragraph{QA-Based Factuality.}`. Must pass `openai_model="gpt-4o-mini"` explicitly (factory default is gpt-3.5-turbo) |
| 1.3 | **SummaC-ZS** (mDeBERTa) | In the original Table 1 | Local, free, slow | ⏳ Running (2,372 / 4,024 at 21:27) |
| 1.4 | **LettuceDetect v2** (mmBERT, Kovács et al. 2026) | Multilingual encoder hallucination detector | Medium: span output must be aggregated into a summary score (a design choice to document) | 🔜 Second after FactCG |
| 1.5 | **DictaLM 3.0 as a judge** (Shmidman et al. 2026) | **Highest importance:** the Hebrew-native open LLM; reviewers will ask | Harder: 12–24B model, prompt and score parsing, hours; local | 🔜 Most important, not easy |
| 1.6 | **M-Prometheus** (Pombal et al. 2025) | Open multilingual LLM judge | Harder: 7B judge, hours; local | 🔜 Later |
| 1.7 | **Zero-shot LLM judge** (GPT-4o-mini) | In the original Table 1 (τ 0.5989) | OpenAI, ~$1–2 | ⛔ Needs approval and a rotated `OPENAI_API_KEY` |
| 1.8 | **G-EVAL** (GPT-4o-mini) | In the original Table 1 (τ 0.2418) | OpenAI, ~$1–3 | ⛔ Needs approval and an API key |
| 1.9 | **mFACT-He** (NeoDictaBERT on Hebrew mFACT) | In the original Table 1 (τ 0.5290) | Local once the model is available | ⛔ The model is on the server only (`/home/avramit/classifier/mfact_he/...`); upload to HF |
| 1.10 | **BARTScore** | Described in §6.2 | Local | 💡 Uses English `bart-large-cnn` on Hebrew, so it measures little; decide whether to report it |
| 1.11 | **FactCG translate-test** (Hebrew → English via NLLB, then FactCG) | Answers "why not translate and use an English checker?" | Local MT model (~2.5 GB), free | 🔜 After 1.1 |
| 1.12 | MiniCheck-7B / Granite Guardian 3.3 | Strong English grounding checkers | 7–8B, local | 💡 Optional |
| 1.13 | Few-shot (FaithJudge-style) LLM judge | Stronger LLM-judge baseline | OpenAI | 💡 Optional |

---

## 2. Research leads ("small wins"): only the most promising

Selection rule: keep a lead only if (a) an experiment supports it or it is cheap to verify, and (b) it fills a gap left by the papers in our literature review (`paper/research/MorphEvalHeb literature review update.md`).

| Priority | Lead | Gap vs. existing papers | Evidence so far | Cost | Status |
|---|---|---|---|---|---|
| **1** | **Error localization via attribution:** do gradient attributions of the binary verifier point at the injected edit, per error family? Output: an analysis paragraph plus a figure with highlighted Hebrew examples | FAVA (Mishra et al. 2024), RAGTruth (Niu et al. 2024) and FineSurE (Song et al. 2024) localize errors **with span supervision, in English**. No Hebrew metric, and none of the cited ones, shows unsupervised localization of morphological errors | Pilot n=10: top-attributed word is the edited word 60% vs. 10% chance. Running n=500 (gradient×input) plus 100 (Integrated Gradients); `paper/results/eval/exp_attribution.py` | Free, local | ⏳ Running |
| **2** | **Translation erasure:** translate source claim and corrupted claim to English and measure, per error family, how often the injected error disappears | mFACT (Qiu et al. 2023), Multi-FActScore (Shafayat et al. 2024) and translate-test pipelines rely on translation; Thellmann et al. (2026) argue translation distorts evaluation, but nobody measured it for **morphological** errors. Turns the paper's core argument into a number | Not run | Free, local NLLB (~2.5 GB) | 🔜 Next |
| **3** | **Native Hebrew LLM summaries** (use LLM outputs from `metadata.generative_summaries`, NOT `summary`, which is the MAFAT reference): score the ~290 summaries in `thesisdatasets/error_finder/` (GPT-3.5, DictaLM 2.0, Gemma 3, gpt-oss, Qwen3) with MorphEvalHeb and the baselines; relate to detected errors and severity | TrueTeacher (Gekhman et al. 2023) and Goyal & Durrett (2021) show synthetic errors differ from real ones; mFAVA (ul Islam et al. 2025) finds Hebrew has the highest hallucination rate. Our benchmark has no native-Hebrew output from current LLMs | Data in the repo; not run | Free, local | 🔜 Next |
| **4** | **Hebrew LLM-judge reliability:** agreement across seeds and judges (GPT-4o-mini x3, DictaLM 3.0) | Doğruöz et al. (2026): 33 of 650 LLM-judge papers are multilingual, none on Hebrew factuality; Fu & Liu (2025): judges agree poorly in low-resource languages | Not run | OpenAI ~$5 plus a local DictaLM 3.0 | ⛔ Needs approval (pairs with 1.5, 1.7) |

**Done and in the paper:** difficulty curve (§7.4: 93% detection on single-family negatives, baselines ≤ 43%); robustness to faithful rewrites (§7.4: < 0.01 SD vs ROUGE-2 0.32 SD).

| Priority | Lead | Status | Result |
|---|---|---|---|
| **5** | **Calibration of the binary verifier** (Brier score, reliability, saturation) | ✅ Done, **recommend adding** | Brier 0.0281, ECE 0.0256, 93.2% of predictions outside [0.01,0.99]. Well-calibrated, usable as a continuous risk score with a small (6.8%) flaggable uncertain band. Full log: `docs/experiments.md` §4 |
| **6** | **Severity calibration on real LLM output** (`error_finder`, ~290 summaries, severity 1–5): does MorphEvalHeb's score track real-error severity, not just injected-error labels? | 📋 Proposed, awaiting your prioritization — do not auto-run when GPU frees | Not run. Also completes lead #3 below. `docs/experiments.md` §5 |
| **7** | **Contrastive-embedding feasibility probe**: do faithful/unfaithful embeddings already separate under the current cross-entropy-trained model, before considering a contrastive-loss retrain? | 📋 Proposed, awaiting your prioritization — do not auto-run when GPU frees | Not run. `docs/experiments.md` §6 |
| **8** | **Entity spans vs. rest-of-text fragmentation**: do named entities fragment more than surrounding text (tokenizer-only, no GPU)? | ✅ Done, **recommend adding** | Opposite of the naive hypothesis: entities fragment *less* (1.00 vs. 1.24 subwords/word, n=2,875, $p\approx0$) — fragmentation is a morphology story, not an entity story. `docs/experiments.md` §5 |
| **9** | **Selective prediction / risk-coverage metric**: a new, deployment-relevant statistic beyond Kendall $\tau$/AUC, built on the calibration finding | ✅ Done, **recommend adding** | Abstaining on $\pm0.1$ around the threshold flags 0.25% of pairs (10/4,024) with only 30% accuracy, raising covered accuracy 96.89%→97.06%. `docs/experiments.md` §7 |
| **10** | **Hebrew-morphology error direction in activation space** (user's idea, verified before documenting): extract a diff-of-means direction from faithful vs. injected-morphological-error Hebrew summary pairs in an open-weight generative LLM's hidden states (e.g. DictaLM 3.0, Gemma, Qwen3 — needs white-box access, so *not* usable with the closed GPT-4o-mini/Gemini judges already in the paper); use it (a) as a linear probe to detect from internal activations whether a generation is heading toward this error class, and/or (b) as a steering vector injected during generation to suppress it | Confirmed real, open gap. Activation steering (ActAdd, Turner et al. 2023; CAA, Rimsky et al. ACL 2024; RepE, Zou et al. 2023; ITI, Li et al. NeurIPS 2023) and internal-state hallucination probing (SAPLMA, Azaria & Mitchell, Findings EMNLP 2023; INSIDE, Chen et al. ICLR 2024; CCS, Burns et al. ICLR 2023) are established techniques, but no paper combines either with Hebrew, with morphology-specific (vs. generic entity/factual) errors, or — for probing — with summarization faithfulness specifically. One directly relevant paper (arXiv:2505.24859, steering vectors for summarization) found steering *increases* hallucination there via "attribute-congruent" invented content — a real risk for the steering half, not just an unknown | Not run; needs a new open-weight generative model load (not the existing NeoDictaBERT encoder) plus activation-extraction engineering — this is *not* a quick win | 🔭 Future work, bigger lift — needs its own scoping decision, not a quick-win candidate |

**Note on lead #10:** frame the **probe (detection)** half as the better-supported, lower-risk direction; frame **steering (correction)** as higher-risk given the one directly relevant negative result above. The already-queued contrastive-embedding feasibility probe (lead #7, awaiting prioritization, `docs/experiments.md` §6) is a cheap precursor to the probe half — it already tests whether faithful/unfaithful pairs separate in the *existing* binary verifier's own embedding space, using no new model, before committing to building a probe on a new generative model's internals.

**Considered and rejected:**
- **Hebrew-aware ENFS (prefix stripping):** tested. False alarms fall 66% → 56%, but AUC drops 0.677 → 0.660. Not a contribution; noted in Appendix B.
- **More languages:** needs language experts and a new taxonomy. Future work.
- **RAG summaries:** no RAG data in the repo; needs a new dataset. Future work.
- **Hebrew tokenization optimization (retraining):** needs retraining on the GPU server; cannot be verified locally. Future work.
- **Hebrew subword-fragmentation vs. classifier error rate** (quick, zero-cost diagnostic, not the retraining version above): tested. r=0.02 overall (p=0.20, not significant); weak even on MAFAT-Aug (r=0.075). The fine-tuned model's errors aren't explained by fragmentation. `docs/experiments.md` §2.
- **Late-fusion hybrid metric, MorphEvalHeb ⊕ ENFS:** tested. Hurts AUC (0.992→0.92 avg-fused) — the much weaker ENFS signal (0.68) drags down the near-ceiling binary verifier. `docs/experiments.md` §1.
- **Ensemble of our own two classifiers (binary ⊕ multi-label):** tested. Same pattern: multi-label alone is weaker (0.844) than binary alone (0.995); fusion dilutes rather than helps. Worth one sentence in the paper pre-empting the "why not ensemble your two models" question. `docs/experiments.md` §3.
- **Error-family-group breakdown of verifier error rate** (entity-only / logical-only / morpho-syntactic-only negatives): tested. Confirms single-family negatives are harder (3–4x the error rate of mixed negatives), consistent with the existing §7.4 difficulty finding, but the per-group split (n=14 for entity-only) is too small to support any claim about which group is hardest. `docs/experiments.md` §6.
- **Severity-weighted score from `error_finder`'s free-text error names:** rejected before running. `correctness_error_name` is a free-text linguistic description (not the 26 canonical family identifiers), so mapping it to family severity would need unreliable fuzzy/keyword matching. Superseded by lead #6 (direct severity calibration, no family mapping needed).

---

## 3. Data and model questions (answers needed from the author)

- [ ] **MAFAT silver references:** 132 of the benchmark's 209 references are LLM drafts (43 unedited). Keep them as faithful, or restrict references to gold?

Evidence for each is in `paper/research/claims_audit.md`.

- [ ] **Setup 2 and Setup 3 training inputs:** original claim, article, or evidence? Docs and HF card say original claim. The released Setup 3 was trained on (original claim, corrupted claim) with no evidence; with evidence input it scores AUC 0.517 (chance). Determines §5, Figure 5 and §7.3.
- [ ] **Which LLM versions generated the dataset:** gpt-4o + gemini-2.0-flash (committed until 2025-11-16) or gpt-4.1 + gemini-2.5-pro (after)?
- [ ] **Is `combined.csv` disjoint from the binary verifier's 18,238 training pairs?** (Training file is on the server.)
- [ ] **Which script translated `test_mfact.csv`:** `mfact_translate.py` (gpt-4o-mini) or `azure_translate.py` (gpt-4o)?
- [ ] **Introduction example:** which texts and settings gave ROUGE ≈ 0.94 / BERTScore ≈ 0.96? Repo code on the sentence pair gives 0.53 / 0.81.
- [ ] **Original ROUGE-1 τ = 0.1391** does not reproduce (re-run: −0.093). How was it computed?
- [ ] **Setup 1 model:** not on HF. Upload it so Setup 1 can be evaluated.

---

## 4. Code fixes found during the audit

- [ ] `utils.filter_diverse_summaries`: the threshold accumulates across candidates (`max_same = max_same + ...` inside the loop). Make it per candidate.
- [ ] Diversity-filter fallback keeps all candidates when none pass; decide and document.
- [ ] Candidate cap is 51 (`if count > 50`); intended 50?
- [ ] Binary `predictions.csv` export truncates the premise to 200 characters, so the file can't be re-scored.
- [ ] Setup 3 `predictions.csv` has an empty `ground_truth_error_types` column (all 41,650 rows).
- [ ] Inference server BERTScore truncates by 350 **words**, which exceeds 512 tokens for long Hebrew text and crashes. Truncate by tokens (as in `paper/results/eval/run_eval.py`).
- [ ] `verify_transformed_summary_grammar` is imported but never called (README claims a grammar-check step).
- [ ] `docs/README.md` is outdated: thresholds (0.93/95 vs 0.85/90), grammar step, Setup 2/3 inputs.
- [ ] **Security:** API keys are hardcoded in `summary_augmentation/*.py`. Rotate them and move to `.env`.

---

## 5. Paper tasks

- [ ] **Recompute Figure 1** on LLM outputs only: 27% of the error-finder summaries are MAFAT references (see `paper/research/claims_audit.md` §G). `% CHECK` left at `fig:llm_detected_errors`.

- [ ] Fill pending rows in Table 1 (Kendall τ) and the appendix table as metrics finish (section 1).
- [ ] Main text currently ends past page 8 (Conclusion starts on page 9). Before submission, choose together what moves to the appendix. Nothing gets cut without the author's approval.
- [ ] Human faithfulness judgments of native Hebrew LLM summaries (150–300), the main reviewer objection.
- [ ] Decide on the 3 taxonomy-figure cards that show unused families (impersonal→personal, double-verb dependency, verb-template agency reversal).
- [ ] Responsible NLP checklist, AI-use disclosure (Acknowledgements), anonymized links, `aclpubcheck`.
- [ ] Upload the changed files to Overleaf (`paper/overleaf/latex/acl_latex.tex`, figures).
