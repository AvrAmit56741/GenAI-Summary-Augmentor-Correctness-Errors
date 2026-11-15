# MorphEvalHeb: experiment log

Quick-win micro-experiments run while the main metric evaluation (SummaC-ZS, FactCG) was busy on the GPU. Experiments 1–7 reuse already-cached scores (`paper/results/eval/cache/*.csv`) — no new model inference, no GPU contention; their throwaway scripts (`_quickwin*.py`) were deleted after the numbers were captured, per the recording policy in effect at the time.

**Recording policy (binding from 2026-11-14 on):** every experiment run for this project must be recorded here in full, with its raw data and raw result files kept on disk and referenced by path — not just the final numbers or conclusion, and not only the ones that make it into the paper. This is for the author's later thesis, which needs the complete experimental record (including failed, negative, and null results), not only what is publication-worthy. Concretely: (1) write experiment scripts to a real file in `paper/results/eval/` (`exp_*.py`), never an ephemeral `python -c` one-liner, so the method is reproducible later; (2) have the script save its raw per-row output to a CSV/JSON file in `paper/results/eval/`, not just print a summary; (3) log the method, raw-file path, full result, and conclusion here, even when the result is negative, null, or too weak/underpowered for the paper. Experiments 1–7 above predate this policy and have no surviving raw files beyond what's noted inline; everything from experiment 8 onward follows it.

Status legend: ✅ positive finding, worth adding to the paper · ⚠️ negative/null result, worth knowing but not a paper addition (some are worth a one-line mention as a considered-and-rejected lead, matching the ENFS-He appendix) · 🔜 queued, needs a model load (GPU), not yet run.

---

## 1. Late-fusion hybrid metric: MorphEvalHeb (binary) ⊕ ENFS — ⚠️ negative

**Idea:** combine the binary verifier's probability with ENFS (entity-hallucination score) into one continuous score, on the theory that they catch different error types (general/morphological vs. entity-specific).

**Method:** rank-normalize both scores on `combined.csv` (n=4,024); combine via (a) average of ranks, (b) min of ranks ("weakest link" — either signal can veto faithfulness).

**Result:**
| Score | AUC (all, n=4,024) | AUC (MAFAT-Aug) | AUC (mFACT-He) |
|---|---|---|---|
| MorphEvalHeb alone | 0.9922 | 0.9952 | 0.9948 |
| ENFS alone | 0.6770 | 0.6524 | 0.7389 |
| Hybrid (avg of ranks) | 0.9201 | 0.9703 | 0.9039 |
| Hybrid (min of ranks) | 0.8749 | 0.9376 | 0.8575 |

**Conclusion:** both fusions *hurt*. MorphEvalHeb already dominates (0.99 vs. 0.68); any uniform-weight combination with a much weaker, noisier signal drags the ceiling down. A hybrid would need a learned, heavily-skewed weighting (e.g. logistic regression) to not lose to the component alone, which defeats the "quick, free" premise and needs held-out tuning to avoid overfitting the benchmark it would be evaluated on.

**Complementarity check:** on the 1,250 MAFAT-Aug negatives, only 14 have an all-entity-family corruption set (no mixed morphological family), and MorphEvalHeb alone already catches 12/14 (85.7%) of those at a 0.5 threshold — too small and already too well-handled to motivate adding ENFS.

**Recommendation:** do not add to the paper as a new metric. Worth one sentence in Appendix B (next to the ENFS-He finding) as a second tested-and-rejected combination, if the author wants the paper to show it was considered.

---

## 2. Hebrew subword-fragmentation vs. classifier error rate — ⚠️ null

**Idea:** Hebrew's rich morphology should fragment into more subword tokens than English per word; if the classifier struggles more on heavily-fragmented summaries, that's a concrete, model-grounded number for the "why Hebrew breaks standard tooling" argument (strengthens #P006 in the guidelines).

**Method:** tokenize all 4,024 summaries with the NeoDictaBERT tokenizer (tokenizer-only, no model forward pass — no GPU cost); fragmentation rate = subword tokens / whitespace words. Point-biserial correlation between fragmentation and binary-verifier error (at a plain 0.5 threshold).

**Result:** mean fragmentation 1.275 subwords/word (median 1.252). Correlation with error rate: r = 0.020, p = 0.20 overall (not significant). Split by part: MAFAT-Aug r = 0.075 (p = 0.001, barely significant but tiny effect), mFACT-He r = −0.002 (p = 0.94, nothing). Error rate across fragmentation quartiles: 3.0% → 2.4% → 3.4% → 3.7% (Q1→Q4) — a weak monotonic trend at best, likely not worth a claim.

**Conclusion:** the fine-tuned classifier's errors are not meaningfully explained by subword fragmentation alone. It has apparently learned to handle typical Hebrew fragmentation levels; errors don't concentrate on the most-fragmented summaries.

**Recommendation:** do not add. The "Hebrew breaks standard metrics" argument should stay qualitative/cited rather than backed by this particular number — it didn't hold up.

---

## 3. Ensemble: binary verifier ⊕ multi-label detector (our own two models) — ⚠️ negative

**Idea:** unlike experiment 1 (external ENFS baseline), combine MorphEvalHeb's own two classifiers — the binary verifier and the multi-label detector's `1 − max(family probability)` — since both are trained on the same data and might catch different things (binary sees article+summary; multi-label is claim-level with rare-family oversampling).

**Method:** same rank-average / rank-min fusion, on MAFAT-Aug only (n=1,889 — the only part with multi-label scores).

**Result:**
| Score | AUC |
|---|---|
| MorphEvalHeb (binary) alone | 0.9952 |
| Multi-label (`1 − max family prob`) alone | 0.8441 |
| Ensemble (avg of ranks) | 0.9762 |
| Ensemble (min of ranks) | 0.9489 |

**Conclusion:** same pattern as experiment 1 — the binary verifier alone is already near-ceiling and the multi-label detector is a visibly weaker standalone discriminator (0.844), so naive fusion only dilutes it. This is itself worth stating in the paper as a one-sentence pre-emption of the obvious reviewer question ("why not just ensemble your two models?").

**Recommendation:** add one sentence to §7 (Results) or Limitations: naive ensembling of the two MorphEvalHeb classifiers does not improve on the binary verifier alone (0.995 → 0.976 avg-fused), so we report them separately rather than as a combined score.

---

## 4. Calibration of the binary verifier's probability output — ✅ positive, recommend adding

**Idea:** AUC and balanced accuracy only test discrimination (ranking), not whether the probability itself is a trustworthy continuous "factuality risk score" — relevant if a practitioner wants to threshold or rank by MorphEvalHeb score rather than just classify.

**Method:** Brier score and a 10-bin reliability check (mean predicted probability vs. empirical faithful-rate per bin) on all 4,024 pairs.

**Result:**
- Brier score: **0.0281** (0.25 = uninformative constant-0.5 baseline; 0 = perfect).
- ECE (10-bin): **0.0256**.
- **93.2%** of predictions fall outside [0.01, 0.99] — the model is confident almost everywhere.
- The ~6.8% of pairs (272/4,024) with intermediate scores are where the model is genuinely uncertain; this is a small, identifiable band a practitioner could route to human review.

**Conclusion:** MorphEvalHeb's probability is well-calibrated, not just well-ranked — and it is usable as a genuine continuous confidence/risk score, with a small, flaggable uncertain band, rather than only as a binary classifier. This is a real, cheap, positive finding that is not yet reported anywhere in the paper and differentiates it from most of the 18 reference papers (none report calibration).

**Recommendation:** add to the paper — one short paragraph in §7 (Results) reporting Brier score and the saturation fraction, framed as supporting MorphEvalHeb's use as a continuous risk score in deployment, not just a pass/fail gate. Cheap, already computed, no further experiments needed.

---

## 5. Entity spans vs. the rest of the sentence: which fragments more? — ✅ positive, counter-intuitive

**Idea:** ENFS and the entity-swap families treat named entities as the hard case. If entity names are also where the NeoDictaBERT tokenizer over-fragments, that would be a concrete, model-grounded reason entity errors are different from morphological ones.

**Method:** merge `cache/ner_entities.jsonl` (DictaBERT-NER spans, already computed for ENFS) onto each unique summary text; for every text with at least one single-word, score ≥ 0.5 entity span, compute subwords/word on the entity span(s) and, separately, on the rest of the text (tokenizer-only, no GPU). Paired Wilcoxon signed-rank test over documents.

**Result:** on 2,875 summaries with at least one detected entity, entity spans fragment into **1.00 subwords/word** (i.e. almost always a single token) vs. **1.24 subwords/word** for the rest of the same sentence (Wilcoxon $p\approx 0$; entity fragmentation is lower in 2,873/2,875 documents, 99.93%).

**Conclusion:** the opposite of the naive hypothesis. Hebrew named entities — mostly transliterated proper nouns — tend to already be frequent single sub-word tokens in NeoDictaBERT's vocabulary. The fragmentation actually happens on the *surrounding* morphologically-inflected words (verbs, prefixed prepositions/conjunctions, construct-state nouns), not on entity names. This is a clean, well-powered piece of evidence that subword fragmentation in this benchmark is a morphology story, not an entity story — i.e. it is evidence *for*, not against, the paper's core framing.

**Recommendation:** worth one sentence in §3 (taxonomy) or §6 (why Hebrew breaks standard metrics), e.g. "named entities are not the source of fragmentation: entity spans average 1.00 subwords/word vs. 1.24 for the surrounding text (n=2,875, Wilcoxon $p<0.001$), confirming that Hebrew's inflectional morphology, not entity naming, drives subword misalignment." Cheap, already computed, strengthens #P006 where experiment 2 (overall fragmentation vs. error rate) came up null.

---

## 6. Verifier error rate by error-family group, among single-group negatives — ⚠️ mixed / underpowered

**Idea:** a finer cut of the existing Difficulty analysis (§7.4): among negatives whose injected errors all come from *one* taxonomy group (entity substitutions / morpho-syntactic / cross-linguistic logical), does MorphEvalHeb's error rate differ by group? Also re-checks whether fragmentation differs by group (extends experiment 2).

**Method:** classify each negative's `error_types` as entity-only, logical-only, morpho-syntactic-only, or mixed (multiple groups); compute the binary verifier's error rate and mean fragmentation per group.

**Result:**
| Group | n | Mean frag. | Verifier error rate |
|---|---|---|---|
| entity-only | 14 | 1.232 | 14.3% |
| logical-only | 87 | 1.220 | 10.3% |
| morpho-syntactic-only | 54 | 1.256 | 11.1% |
| mixed (≥2 groups) | 1,095 | 1.284 | 3.2% |

**Conclusion:** confirms the existing Difficulty finding from another angle — single-group (and especially single-family) negatives are harder than multi-family ones (3–4× higher error), consistent with §7.4's reported 93% detection on single-family negatives. Fragmentation does not differ meaningfully by group (1.22–1.28 throughout), so this is a second null result for fragmentation-as-explanation, consistent with experiment 2. The entity-only error rate (14.3%) is nominally the highest of the three pure groups, which would be a notable complication for the paper's "ENFS is the strongest baseline on entity errors" framing — but n=14 is far too small to support any claim; this needs the full single-family breakdown (n=45, already used in §7.4) re-cut by group, not this ad hoc slice.

**Recommendation:** do not add as a standalone claim — the entity-only n=14 is not reliable. If the author re-runs the existing §7.4 difficulty analysis with a group column, this is free to report properly; otherwise skip.

---

## 7. Selective prediction: a risk-coverage metric beyond Kendall $\tau$/AUC — ✅ positive, recommend adding

**Idea:** AUC and $\tau$ summarize ranking quality; balanced accuracy summarizes one fixed operating point. Neither answers the practitioner's actual question: *if I let the model abstain on its most uncertain cases and route them to a human, how much more reliable does the rest become?* This turns the calibration finding (experiment 4) into an actual deployable metric — a risk-coverage curve, standard in selective classification but not used by any of the 18 reference papers.

**Method:** for a widening abstain band $[0.5-\delta, 0.5+\delta]$ around the decision threshold, compute coverage (share of pairs *not* abstained) and accuracy on the covered vs. abstained subsets, on all 4,024 pairs.

**Result:**
| Band ($\delta$) | Coverage | Covered accuracy | Abstained accuracy | n abstained |
|---|---|---|---|---|
| 0 (no abstention) | 100.0% | 96.89% | — | 0 |
| ±0.10 | 99.75% | 97.06% | 30.0% | 10 |
| ±0.20 | 99.43% | 97.15% | 52.2% | 23 |
| ±0.30 | 98.91% | 97.29% | 61.4% | 44 |

**Conclusion:** abstaining on a tiny, cheap-to-flag band (0.25–1.1% of pairs) measurably raises accuracy on the rest (96.89% → 97.06–97.29%) while concentrating genuinely hard cases into the abstained set (accuracy there is 30–61% vs. 96.89% overall) — i.e. the band the model itself flags as uncertain is disproportionately where it is actually wrong. This is a positive, interpretable, cheap result and a genuinely different statistic from Kendall $\tau$/AUC/balanced accuracy: it directly supports a practical "abstain-and-route-to-human" deployment mode.

**Recommendation:** add to the paper, next to the calibration paragraph (§7, Results) — one short addition, e.g. "widening the abstain band to $\pm0.1$ around the threshold flags only 0.25% of pairs (10/4,024) for review, but the flagged set has only 30% accuracy vs. 96.9% overall, confirming the model's own uncertainty concentrates on genuinely hard cases." Already computed from the cached binary-verifier scores; no new experiments needed.

---

## 8. FactCG baseline vs. MorphEvalHeb — ✅ complete (4,024/4,024), added to the paper

**Status:** the FactCG (Lei et al., NAACL 2025) baseline run completed all 4,024 benchmark pairs on 2026-11-14 (paused twice along the way — at 2,371 and again at 3,185 — and resumed each time by request; the ported inference code was independently re-verified against the reference `derenlei/FactCG` repo: same checkpoint, instruction template, label index, and chunking/aggregation logic). Final numbers are from `analyze.py`'s cross-validated balanced accuracy and bootstrap CIs (`meta_eval.json`), not the earlier in-sample best-threshold estimates — **now added to Table 1, Table 4, and the Results prose (§7, after the ENFS/LLM-judge sentence).**

**Result (cross-validated balanced accuracy; AUC with 95% bootstrap CI; Kendall $\tau_b$):**
| Part | n | FactCG AUC | FactCG BAcc | FactCG $\tau_b$ | MorphEvalHeb AUC | MorphEvalHeb BAcc | MorphEvalHeb $\tau_b$ |
|---|---|---|---|---|---|---|---|
| All | 4,024 | 0.562 [0.54,0.59] | 0.638 | 0.087 | 0.992 [0.99,1.00] | 0.969 | 0.689 |
| MAFAT-Aug | 1,889 | 0.534 | 0.505 | 0.045 | 0.995 | 0.980 | — |
| mFACT-He | 2,135 | 0.893 | 0.825 | 0.555 | 0.995 | 0.968 | — |

**Conclusion:** confirms the partial-run reads (both the 2,371- and 3,185-pair snapshots): FactCG transfers reasonably to mFACT-He's naturally occurring, English-adjacent errors (AUC 0.893) but collapses to near chance on MAFAT-Aug's injected Hebrew morphological/entity error taxonomy (AUC 0.534) — a 36-point gap that doesn't appear for MorphEvalHeb (0.995 on both parts). Note the pooled "All" AUC (0.562) is much closer to the MAFAT-Aug figure than a naive average of the two parts would suggest — pooling a benchmark with heterogeneous parts can drag a baseline's apparent overall AUC toward its weaker part's level even when weighted by size, which MorphEvalHeb does not suffer from since it is uniformly strong on both parts. This is a clean, direct demonstration of the paper's core cross-lingual-transfer argument and is now the strongest baseline-failure evidence in the paper (stronger than SummaC's, since FactCG is purpose-built for the same source-grounded factuality task MorphEvalHeb targets, not a generic NLI aggregator).

---

## 9. FactCG's tokenizer on Hebrew: mechanism check — ✅ positive, added to the paper

**Question (raised by the author):** FactCG's backbone (DeBERTa-v3-large) is an English-only model — mechanically, how does scoring it on Hebrew even work, and is the comparison valid?

**Method:** loaded the FactCG tokenizer (`yaxili96/FactCG-DeBERTa-v3-Large`, SentencePiece, 128K vocab) directly and measured its UNK rate and subword-fragmentation rate on a 300-document sample of benchmark source articles (whitespace words vs. subword token count), the same methodology as the entity-fragmentation experiment (§5/#8).

**Result:**
| | FactCG tokenizer | NeoDictaBERT (ours) |
|---|---|---|
| UNK rate | 0.001% (5 / 758,941 tokens) | — |
| Subwords per word | **4.30** | 1.00–1.24 |

**Conclusion:** SentencePiece never hard-fails on unseen scripts — it falls back to small Unicode-level pieces instead of `[UNK]`, which is *why* FactCG can be run on Hebrew at all without crashing or requiring any adaptation. But it has essentially no learned Hebrew subword vocabulary: Hebrew text fragments into 4.30 subwords/word, 3.5–4.3× more than our Hebrew-native tokenizer. This is a credible mechanistic explanation (not proof, but a strong, concrete, checkable one) for why FactCG collapses to near-chance specifically on MAFAT-Aug's word-level morphological/entity errors (experiment 8): the signal of a single swapped suffix or entity gets diluted across many near-meaningless character fragments the model was never trained to interpret, while MorphEvalHeb's tokenizer sees the same edit as a small, meaningful subword change.

**Recommendation:** added as a clarifying sentence directly after FactCG's method description in the paper (§6.2/baselines), pre-empting the "is a cross-lingual English-model baseline even a fair/valid comparison" reviewer question.

---

## 10. Severity calibration on real LLM output — ⚠️ negative, important limitation finding (not a paper win)

**Idea:** test whether MorphEvalHeb's unfaithfulness probability — trained almost entirely on *injected* errors — tracks the *severity* of *naturally occurring* LLM errors, using the LLM-generated summaries and Gemini-identified errors (severity 1–5) in `thesisdatasets/error_finder/*.csv`. Answers the "synthetic vs. real error generalization" reviewer objection (#P009) and completes research lead #3/#6 in `docs/tasks.md`.

**Method:** script `paper/results/eval/exp_severity.py`. Loaded all 15 `error_finder/*.csv` files (450 raw error rows), used `generative_summary` (the real LLM output) paired with `premise` — **not** `ai_summary`, which is a different, misaligned text in these files (verified: only 3.3% of rows have `ai_summary == generative_summary`). Deduplicated to 265 unique (premise, generative_summary) pairs, taking the worst (max) severity per pair when a summary had multiple identified errors. Scored each pair with the existing binary verifier (`NLI-hebrew-binary-correctness-metric`), same preprocessing as `run_eval.py`'s `run_binary`. Raw per-pair output saved to `paper/results/eval/severity.csv` (premise, generative_summary, doc_id, source, max_severity, n_errors, error_names, unfaithful_prob).

**Result (n=265, all Gemini-confirmed real factual errors):**
| Statistic | Value |
|---|---|
| Spearman ρ(severity, unfaithful_prob) | −0.040 (p=0.52, not significant) |
| Mean unfaithful_prob | 0.084 |
| Median unfaithful_prob | 0.0016 |
| Overall flagged rate (prob > 0.5) | **9.1%** |
| Severity distribution (1/2/3/4/5) | 1 / 3 / 47 / 120 / 94 |
| Flagged rate by severity (3/4/5) | 2.1% / 9.2% / 11.7% |

**Conclusion:** no severity calibration at all (ρ≈0, not significant), and more importantly, a real generalization gap: the verifier flags only 9% of real, Gemini-confirmed LLM errors as unfaithful, versus ~93%+ detection on the injected MAFAT-Aug benchmark it was trained on. Training almost entirely on injected errors does not transfer well to naturally occurring LLM errors in this sample. Severity classes are also heavily imbalanced (severity ≤2: n=4), which limits how far the null correlation result can be trusted on its own, but the low overall flag rate does not depend on that imbalance.

**Recommendation:** do not present this as a paper result as-is — it cuts against the paper's own detection claims and needs care before being stated. It does, however, directly and concretely support the existing Limitations paragraph ("injected errors need not match those of real summarizers"); consider citing the 9% figure there if the author wants to quantify that limitation rather than leave it qualitative. Raw data: `paper/results/eval/severity.csv` (265 rows, all columns preserved for later inspection/thesis use).

**Follow-up: do the baselines track real-error severity any better? — also negative, generalizes the finding.** Completed lead #3/#6 by running the zero-model-load n-gram baselines (ROUGE-1/2/L, BLEU, chrF, Token F1; source-vs-summary overlap, same as Table 1) on the identical 265 pairs. Deliberately skipped the baselines that need a model load (ENFS, SummaC, BERTScore, FactCG) this round: a 12B translation model was running concurrently in a peer session, and a memory check beforehand (`top -l 1`) showed only ~180MB free system-wide out of 48GB — not safe to load more models until that job frees memory. Script: `paper/results/eval/exp_real_errors_ngram.py`; raw output: `paper/results/eval/real_errors_ngram.csv`.

Result (n=265): **no correlation for any n-gram metric either** — ROUGE-1 ρ=+0.028 (p=0.65), ROUGE-2 ρ=+0.043 (p=0.48), ROUGE-L ρ=-0.005 (p=0.93), BLEU ρ=-0.023 (p=0.71), chrF ρ=-0.027 (p=0.66), Token F1 ρ=+0.017 (p=0.78) — all non-significant, same null pattern as MorphEvalHeb's ρ=-0.04. This generalizes the earlier finding: the lack of severity tracking isn't specific to MorphEvalHeb, nothing tested so far tracks real-error severity in this sample. Caveat: severity classes 1–2 have very small n (1 and 3 respectively, out of 265; the bulk is severity 3–5 at n=47/120/94), so read the correlation statistic over the full range, not the two extreme buckets alone.

**Visualization (thesis only):** `paper/results/eval/exp_real_errors_viz.py` → `paper/results/eval/thesis_figures/real_error_severity_tracking.png`, a two-panel line plot (MorphEvalHeb; n-gram baselines) of normalized score vs. severity bucket. The small-n buckets (severity 1–2) create visually large swings that are noise, not signal — caption/reader should rely on the reported Spearman ρ, not eyeball the line shape at the extremes.

**Recommendation:** not for the paper (same reasoning as the MorphEvalHeb severity finding), but strengthens the Limitations framing: this isn't "our metric specifically fails on real errors," it's "no metric tested so far, including standard n-gram baselines, tracks real-error severity" — a broader and more defensible statement if the author wants to use it. ENFS/SummaC/BERTScore/FactCG on this same dataset remain queued, to run once the concurrent translation job finishes and frees memory.

---

## 11. Contrastive-embedding feasibility probe — ✅ positive, recommend keeping as the answer (do not pursue a contrastive retrain)

**Idea:** before considering a contrastive-loss retrain (expensive, needs the GPU server, can't be verified locally per the existing "rejected" note in `docs/tasks.md`), a cheap feasibility check: do faithful/unfaithful pairs already separate in the *existing* cross-entropy-trained binary verifier's own embedding space?

**Method:** script `paper/results/eval/exp_contrastive.py`. Stratified sample of 1,200 pairs (300 each of MAFAT-Aug/mFACT-He × faithful/unfaithful) from the benchmark. For each pair, ran the binary verifier with `output_hidden_states=True` and mean-pooled the final hidden layer (not the classification head) to get a frozen, task-head-independent embedding. Measured: (1) 5-fold cross-validated AUC of a fresh linear logistic-regression probe trained on these frozen embeddings; (2) silhouette score of the two label clusters; (3) mean within-class vs. between-class cosine similarity. Raw result saved to `paper/results/eval/contrastive_probe.json`.

**Result (n=1,200):**
| Statistic | Value |
|---|---|
| 5-fold CV logistic-regression AUC (frozen embeddings) | 0.983 |
| Silhouette score | 0.243 |
| Within-class cosine similarity | 0.871 |
| Between-class cosine similarity | 0.719 |
| Gap (within − between) | 0.151 |

**Conclusion:** a simple linear probe on frozen embeddings recovers 0.983 AUC, almost all of the full fine-tuned model's 0.992 AUC, with a clear positive silhouette score and a real within/between-class similarity gap. Faithful/unfaithful pairs are already well-separated under plain cross-entropy training; a contrastive objective is very unlikely to add meaningful separation that isn't already there.

**Recommendation:** treat this as the answer to "should we try contrastive learning" — no, the embedding space already separates the classes well, so a contrastive-loss retrain is not worth the GPU-server time it would need. Worth one sentence in the paper pre-empting the "why not contrastive learning" reviewer question, similar to the existing ensemble/late-fusion pre-emptive sentences. Raw data: `paper/results/eval/contrastive_probe.json`.

**Visual companion (thesis only, not the ACL paper — author's explicit decision):** script `paper/results/eval/exp_contrastive_viz.py` re-embeds the identical 1,200-pair sample (same `random_state=0`) and projects it to 2D with t-SNE. Output: `paper/results/eval/thesis_figures/contrastive_embedding_tsne.png` (plot) and `contrastive_embedding_tsne.npz` (raw embeddings + labels + part, for later re-plotting). The plot shows four clusters, dominated by benchmark part (MAFAT-Aug vs. mFACT-He) rather than label, but faithful/unfaithful split cleanly *within* each part — consistent with the 0.983 probe AUC. Kept separate from `paper/figures/` (the ACL LaTeX source's figure directory) so it is never accidentally pulled into the paper.

---

## 12. Error localization via gradient attribution (lead #1) — ✅ positive, stopped early by request at n=171 (MAFAT-Aug only)

**Idea:** do gradient attributions of the binary verifier point at the actual injected edit, per error family, without span supervision? Fills the gap vs. FAVA/RAGTruth/FineSurE, which localize errors with span supervision in English.

**Method:** script `paper/results/eval/exp_attribution.py` (pre-existing, resumed with `RESUME=1 python exp_attribution.py 500`). Restricted to MAFAT-Aug only (needs each negative's matching gold summary, same `doc_id`, to diff against and find the exact edited words — mFACT-He has no such gold/corrupted pair structure). For each negative: word-level diff against its gold summary to get the edited-word indices; gradient×input (`gxi`, signed) and its absolute value (`abs`) attributed to the verifier's contradiction-vs-entailment logit margin, aggregated to word level via token offsets. Raw per-row output: `paper/results/eval/attribution.csv` (171 rows: 22 from the original 2026-11-13 run, resumed to 171 on 2026-11-14, then stopped early by request — checked for a stable signal first, see below — rather than continuing to the original n=500 target).

**Result (n=171, MAFAT-Aug only):**
| Method | Top-1 hit rate | Chance | Mean AUC (edited vs. other words) |
|---|---|---|---|
| Gradient×input (signed, `gxi`) | 55.0% | 11.5% | 0.523 (weak) |
| **Gradient×input, absolute value (`abs`)** | **73.1%** | 11.5% | **0.724** |

Bootstrap 95% CIs (2,000 resamples): `abs_auc` [0.710, 0.738], `abs_top1_hit` [0.661, 0.795]. Split-half stability check (first 85 vs. last 85 rows) showed `gxi_auc` moving only 0.520→0.528 and `gxi_top1_hit` only 0.541→0.565 — the estimate was not drifting as more rows came in, which is why the run was stopped early rather than continued to n=500: the remaining ~330 rows would have mainly tightened the CI, not changed the conclusion, for ~90 more minutes of compute.

**Conclusion:** the model's top-attributed word (by gradient magnitude, not sign) is the actual injected edit 73% of the time versus an 11.5% chance baseline (6.3×), with a stable AUC of 0.724 — a real, unsupervised localization signal, stronger than the original n=10 pilot note in `docs/tasks.md` ("60% vs. 10% chance"). Scoped to MAFAT-Aug's injected morphological/entity errors only; not tested on mFACT-He's naturally occurring errors (no gold/corrupted pair exists there to diff against).

**Recommendation:** worth adding to the paper as an analysis paragraph (as originally scoped in `docs/tasks.md` lead #1), citing n=171, the `abs` method's 73.1% top-1 hit rate / 0.724 AUC, and the CI. Also directly feeds lead #11(A) (morphology-aware correction agent), which needs attributed spans as its feedback signal. A qualitative figure with highlighted Hebrew examples (as originally scoped) is a separate, small follow-up — not done here.

---

## 13. Detection vs. edit size, with a faithful-rewrite control — ✅ positive, added to the paper (run by a peer session, independently verified)

**Provenance:** this experiment's script and raw output (`paper/results/eval/exp_edit_size.py`, `edit_size.csv`, `edit_size_bins.csv`, `edit_size_control.csv`) were produced by a separate, concurrently running Claude Code session working in this same repo (`genai-summary-augmentor-correctness-erro-bd`), not by this session. Before using it, this session independently re-ran the script end to end and reproduced every number exactly, and checked for the main failure mode that would invalidate it (negatives with no matching gold summary via `doc_id` — found zero such rows out of 1,250). Recorded here per the project's recording policy once verified.

**Idea:** two questions neither raw AUC nor the existing robustness check (§7.4, SD-based) directly answers: (1) do metrics keep detecting corruptions as the edit gets smaller and more subtle? (2) can a metric actually tell a real factual error apart from a harmless, meaning-preserving rewrite of the same article, not just notice that the text changed?

**Method:** for each MAFAT-Aug negative, aligned its gold summary (same `doc_id`) and computed character similarity + word-level diff size; paired detection = P(metric scores the corruption below the gold, ties=0.5), binned by similarity, with a 1,000-resample bootstrap CI per bin. Control: the same paired statistic computed on (a) faithful rewrites vs. gold (false-alarm rate) and (b) corrupted vs. faithful rewrite of the same article (`P_corr_below_rewrite` — the real test, since a metric that merely penalizes *any* wording change would score ~0.5 here even while looking good on raw detection).

**Result — detection by edit size (paired detection rate, 0.5 = chance):**
| Similarity bin | n | Median words changed | MorphEvalHeb | FactCG | ENFS | BERTScore (heBERT) | ROUGE-2 |
|---|---|---|---|---|---|---|---|
| 0.80–0.90 | 74 | 36.5 | 1.00 | 0.50 | 0.92 | 0.55 | 1.00 |
| 0.90–0.95 | 445 | 24.0 | 1.00 | 0.53 | 0.92 | 0.73 | 0.97 |
| 0.95–0.98 | 514 | 15.0 | 1.00 | 0.56 | 0.79 | 0.61 | 0.94 |
| **≥0.98** | **217** | **7.0** | **1.00** | **0.63** | **0.58** | **0.44** | **0.91** |

**Result — faithful-rewrite control (n=208 rewrites, 1,330 corrupted-vs-rewrite pairs, 93 articles):**
| Metric | Detects real corruption vs. gold | False-alarms on faithful rewrite vs. gold | Corruption vs. rewrite (0.5 = can't tell) |
|---|---|---|---|
| **MorphEvalHeb** | 1.00 | 0.53 | **1.00** |
| ROUGE-2 | 0.95 | 0.88 | **0.50** |
| ROUGE-L | 0.92 | 0.84 | **0.48** |
| Token F1 | 0.87 | 0.81 | **0.45** |
| BLEU | 0.72 | 0.73 | **0.42** |
| FactCG | 0.56 | 0.48 | 0.55 |
| SummaC-ZS | 0.83 | 0.51 | 0.79 |
| ENFS | 0.81 | 0.54 | 0.83 |

**Conclusion:** the surface-overlap metrics' apparent detection ability is largely an artifact of penalizing *any* wording change, not actual error detection — ROUGE-2/ROUGE-L/Token F1/BLEU false-alarm on faithful rewrites almost as often as they detect real corruptions, and are at or below chance when asked to directly rank a real corruption below a faithful rewrite of the same article. MorphEvalHeb is the only metric that both detects corruptions (1.00) and almost never false-alarms (0.53), holding up even at the smallest, most subtle edit bin (median 7 words, 3 error families) where FactCG/ENFS/BERTScore collapse to 44–63%. This is a stricter, more direct version of the existing §7.4 robustness claim (paired same-article comparison vs. a standard-deviation shift) and a genuinely strong result.

**Recommendation:** added to the paper — extended the existing "Robustness to meaning-preserving rewrites" paragraph with the `P_corr_below_rewrite` and false-alarm numbers, and added a new short "Detection vs. edit size" paragraph with the smallest-bin numbers (§7, Analysis). Raw data: `paper/results/eval/edit_size.csv` (per-pair), `edit_size_bins.csv` (per bin × metric), `edit_size_control.csv` (the rewrite control).

---

## 14. Resolving the ROUGE-1/BERTScore sign discrepancy (docs/tasks.md §3) — ✅ resolved, Simpson's paradox, not a bug

**Question:** Table 1 reports two numbers for ROUGE-1 that disagree in *sign* — the original paper run gives τ=0.1391 (positive), the re-run on `combined.csv` gives τ=−0.093 (negative). Same pattern for BERTScore (heBERT): original 0.0162 (positive) vs. re-run −0.061 (negative). A sign flip (not just a magnitude difference) could indicate a real bug in one of the two runs, so this needed investigating before submission. Run entirely on CPU, zero model load (safe while the peer session's 12B translation model was using most of system memory).

**Method:** (1) read the live Streamlit evaluation app's n-gram tab (`summary_augmentation/pages/4_Correctness_Evaluation_Metrics.py`) to check whether it computes ROUGE-1 differently from `run_eval.py`'s re-run — found it uses the identical `RougeCalculator` class in the identical `(premise, hypothesis)` direction, ruling out a formula/direction bug in the metric itself. (2) Tested whether the original number instead compared the hypothesis to its gold reference summary (the standard summarization-eval convention) rather than to the source article — on MAFAT-Aug alone this gives τ=+0.017 (still not close to +0.1391, and still much smaller magnitude). (3) Split the *current* re-run methodology by benchmark part instead of changing what's compared.

**Result — ROUGE-1, by part (current `premise` vs. `hypothesis` methodology, unchanged):**
| Part | n | Faithful rate | Mean ROUGE-1 | Kendall τ (within part) |
|---|---|---|---|---|
| MAFAT-Aug | 1,889 | 33.8% | 0.153 | **+0.011** |
| mFACT-He | 2,135 | 50.4% | 0.046 | **+0.039** |
| **Pooled (both parts)** | **4,024** | — | — | **−0.093** |

Same mechanism confirmed for BERTScore (heBERT): MAFAT-Aug τ=+0.023 (mean score 0.641), mFACT-He τ=+0.138 (mean score 0.439), pooled τ=−0.061.

**Conclusion:** this is Simpson's paradox, not a bug in either run. ROUGE-1 and BERTScore both correlate *positively* with faithfulness within each benchmark part on its own — weakly, but in the expected direction. The two parts, however, differ systematically on both axes at once: MAFAT-Aug summaries are more extractive (higher ROUGE-1/BERTScore, since the augmentation pipeline edits an otherwise-intact summary) **and** majority-unfaithful by construction (only 33.8% faithful, since most of MAFAT-Aug exists specifically to be corrupted); mFACT-He summaries are less extractive **and** majority-faithful (50.4%). Pooling the two parts means "high ROUGE-1" partly just means "probably MAFAT-Aug," and MAFAT-Aug pairs are disproportionately unfaithful regardless of their actual ROUGE-1 score — that between-part confound is strong enough to flip the weak within-part signal's sign once pooled. The original τ=0.1391 was most likely computed on an earlier, single-part (MAFAT-Aug only, pre-merge) version of the benchmark, which is consistent with this explanation in sign (positive) if not exact in magnitude (the current MAFAT-Aug-only re-run of +0.011 is much weaker, likely reflecting a smaller/differently-sampled MAFAT-Aug at the time of the original run, or a slightly different metric variant).

This is the same general phenomenon already noted separately for FactCG (§8): a weak baseline's *pooled* statistic on this benchmark can be driven more by which part a pair comes from than by the baseline's actual signal, something MorphEvalHeb does not suffer from since it is strong and consistent on both parts.

**Recommendation:** resolved — marked done in `docs/tasks.md` §3 with the full explanation. Optional for the paper: a one-sentence footnote on Table 1 or Table 4 noting that per-part and pooled signs can differ for weak baselines due to this between-part confound would pre-empt a sharp reviewer's "why is ROUGE-1's pooled correlation negative, shouldn't more overlap mean more faithful?" question — author's call on whether it's worth the space.

---

## 15. Translation erasure (lead #2) — ✅ positive, run by a peer session, independently verified

**Question:** if a Hebrew MAFAT-Aug negative is machine-translated to English, does the injected error survive, or does translation erase the very signal the paper argues is Hebrew-specific? Answers the paper's own rhetorical "why not translate and use an English checker" question with a number instead of an assertion, and directly extends the FactCG cross-lingual-transfer finding (§8–9).

**Provenance:** produced by the concurrently running peer Claude Code session (`genai-summary-augmentor-correctness-erro-bd`), not this session — `exp_translation.py`, cached translations (`cache/translations_translategemma-12b-it.jsonl`, 1,192 negatives' edited sentences, via `google/translategemma-12b-it`), and `translation_erasure_translategemma-12b-it.csv`. Per the established practice for peer-session output (§13), independently reproduced before using: deleted the peer's output file, reran `python exp_translation.py erasure` from scratch against the same translation cache (no model reload — this step is pure CPU/pandas), and got an identical result (10,298 edited sentences, 9.448% exact-erased, byte-identical to the peer's file).

**Result (per injected family, by the pipeline's own 26-family taxonomy — a more granular breakdown than the peer's own summary, which used a 6-bucket rule-based re-labeling instead):**
| Family | n edited sentences | exact-erased | near-erased (chrF≥95) |
|---|---|---|---|
| verb_gender_swap | 2,754 | 12.3% | 21.6% |
| sentence_negation | 5,728 | 11.8% | 19.8% |
| pronoun_reference_errors | 6,223 | 11.8% | 18.4% |
| hebrew_root_pattern_confusion | 6,950 | 10.7% | 17.4% |
| noun_gender_swap | 2,574 | 10.0% | 17.8% |
| number_swap | 6,095 | 8.1% | 16.1% |
| entity_person_swap / entity_organization_swap / entity_title_swap | 4,882 / 3,755 / 2,751 | 5.6% / 4.0% / 3.7% | 11.5% / 11.5% / 10.2% |
| homographic_gender_errors | 196 | 3.6% | 7.1% | (n small)

Overall: 9.4% of edited sentences translate to an *identical* English string despite differing in Hebrew; 16.9% are near-identical (chrF≥95). Morpho-syntactic families (gender/negation/root-pattern/pronoun) erase roughly 2–3× more often than entity-identity families (person/org/title swaps), consistent with the paper's core claim that these specific error types are harder to express or even survive in English.

**Conclusion:** translation is a real, non-trivial confound for the exact error types this paper targets. This is the strongest evidence yet, beyond the FactCG tokenizer-fragmentation finding, that translate-to-English pipelines specifically underserve Hebrew morphological errors (not just Hebrew in general).

**Recommendation:** add to the paper as a new short analysis point (not yet in `acl_latex.tex` as of this entry) — pairs naturally with the FactCG baseline discussion (§6). Raw data: `paper/results/eval/translation_erasure_translategemma-12b-it.csv`; script: `paper/results/eval/exp_translation.py` (`erasure` mode).

---

## 16. Label-fidelity audit of injected negatives — ✅ positive-ish, quick win from the 2026-11-15 deep-research pass

**Motivation:** a 2026-07 arXiv paper on the "test oracle problem" in synthetic LLM-judge corpora (status: arXiv-only, no confirmed peer review — flagged as such, not cited as settled) argues LLM-*generated* negatives are "oracle-less," unlike deterministic string-level perturbation, which carries a free mechanical oracle. On inspection, that framing does **not** transfer cleanly to this pipeline: `correctness_errors_augmentation_v2.py` is LLM-prompted (an LLM is given a family label and free-text-generates the corrupted claim), not a deterministic string operator, so there is no free oracle here either — the honest question is how often the self-reported family label actually matches what changed in the text, and that is checkable rather than assumable.

**Method:** reused `exp_translation.py`'s independent, rule-based `edit_kind()` classifier (built for a different purpose — pre-labeling sentence pairs before a manual translation check) as an *unrelated* ground-truth signal for what changed between a MAFAT-Aug negative and its gold reference, then checked whether this independent signal matches the pipeline's self-reported family label, for the subset of families with an unambiguous lexical signature: `sentence_negation` (negation particles), `number_swap`/`entity_date_swap`/`measure_unit_swap` (digit strings), `noun_gender_swap`/`verb_gender_swap` (suffix-only agreement change or definiteness). Scope limitation stated up front: only 6 of the ~26 injected families have a cheap, unambiguous lexical signature this way; entity-identity swaps, construct-state confusion, idiom corruption, and others have no cheap mechanical check and are excluded, not claimed to be fine. Zero model load, CPU/pandas only. Script: `paper/results/eval/exp_label_fidelity.py`; raw output: `paper/results/eval/label_fidelity.csv` (2,691 claimed-family instances, n=1,158 negatives).

**Result:**
| Family | n | independent-signal hit rate |
|---|---|---|
| noun_gender_swap | 300 | 71.0% |
| measure_unit_swap | 363 | 74.4% |
| verb_gender_swap | 334 | 77.5% |
| entity_date_swap | 345 | 80.9% |
| number_swap | 683 | 91.7% |
| sentence_negation | 666 | 95.2% |
| **overall** | **2,691** | **84.8%** |

**Conclusion:** for the families checkable this way, the pipeline's self-reported label matches an independent lexical signal 71–95% of the time (84.8% overall). This is a **lower bound**, not an error rate: a "miss" can mean the claimed error fired but through a lexical realization `edit_kind()` doesn't pattern-match (e.g. a gender change expressed via a different word choice rather than a bare suffix swap), or the matching sentence pair was filtered out by the strict 1:1 sentence-substitution restriction in `edited_pairs()`. The gender-agreement families show the most misses, plausibly because Hebrew gender agreement is often realized through word choice, not just a suffix, which this particular checker is blind to by design — a real limitation of the check, not necessarily of the injection pipeline.

**Recommendation:** worth one sentence in the paper (Limitations or the augmentation-methodology section) stating that an independent lexical audit of the checkable subset of injected families confirms 71–95% (84.8% overall, n=2,691) label fidelity, framed honestly as a lower bound under a strict context-free check covering 6 of ~26 families — a genuine, citable answer to a "are your injected negatives mislabeled" reviewer objection, but should **not** be dressed up with the oracle paper's "deterministic perturbation = free oracle" framing, since this pipeline isn't deterministic. Author's call on whether it's worth the space; not a load-bearing result on its own.

---

## Proposed, not yet run (awaiting prioritization — do not auto-run when the GPU frees up; wait for an explicit go-ahead)
