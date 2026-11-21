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

**Follow-up 2 (2026-11-15): the queued model-dependent baselines, now run — still negative once corrected for multiple comparisons.** Memory freed up after the peer session's translation job finished; ran ENFS, SummaC-ZS, BERTScore (heBERT), and FactCG on the identical 265 pairs. Script: `paper/results/eval/exp_real_errors_models.py`; raw output: `paper/results/eval/real_errors_models.csv`. (Also fixed, along the way: SummaC and the ENFS/NER pipeline were silently running on CPU — see the MPS device fix noted in `docs/tasks.md` §4 — so this run is faster than it would otherwise have been.)

Result (n=265): BERTScore (heBERT) ρ=+0.137 (p=0.025), FactCG ρ=+0.163 (p=0.008), ENFS ρ=+0.069 (p=0.264, n.s.), SummaC-ZS ρ=−0.007 (p=0.906, n.s.). Two of the four look nominally significant at p<0.05 — but checked this against the full set of 11 metrics now tested for severity correlation across this experiment and its predecessor (MorphEvalHeb + 6 n-gram metrics + these 4) with a Holm-Bonferroni correction: **none survive**, including FactCG's p=0.008 (needs p<0.0045 at its rank). So the honest reading is the same as before: no metric tested so far, model-based or not, shows a severity-tracking signal that holds up once you account for testing 11 metrics. FactCG and BERTScore (heBERT) are the closest to a real signal and would be worth a dedicated, pre-registered follow-up (rather than reading into this post-hoc multiple-comparison result) if the author wants to pursue it further.

**Recommendation:** not for the paper — doesn't change the existing Limitations wording, which already states the finding at the right level of generality. Worth keeping in the thesis as the complete picture (all 11 metrics tested, properly corrected), since it's a stronger, more defensible version of "no metric tracks real-error severity" than the n-gram-only check alone.

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

## 17. Translate-then-check with real commercial LLM translators (lead #2 follow-up) — run by a peer session, independently verified, partially retracted

**Context:** extends §15 (translation erasure via the open-weight `translategemma-12b-it`) with two commercial LLM translators (Claude Haiku 4.5, Amazon Nova Micro via AWS Bedrock) on a larger, stratified 250/260-pair sample, with document-level (not sentence-level) translation, sentence-anchoring to re-align edits after translation, and an LLM-judge calibration step. Produced by the peer session `genai-summary-augmentor-correctness-erro-bd`; API cost reported as ~$2. Scripts: `paper/results/eval/exp_translation_api.py`; raw data: `translate_test_api_results.csv` (250 pairs), `translation_erasure_doclevel.csv` (4,604 edited-sentence rows, 260 pairs × 2 models), `translation_experiments_summary.csv` (final numbers), `cache/api_translations.jsonl` (6MB, every raw translation).

**Verification method:** independently reproduced every headline number directly from the raw CSVs (not trusting the summary file), reading the actual script to check methodology, and inspecting the one file documenting a "manual check."

### What reproduces cleanly and is solid

**Translate-then-check AUC** (`translate_test_api_results.csv`, 250 pairs, re-derived with `roc_auc_score` directly): FactCG natively on Hebrew scores 0.576 [overall], 0.527 [MAFAT-Aug], 0.768 [mFACT-He]; translating first with Haiku 4.5 raises these to 0.804/0.683/0.899, and with Nova Micro to 0.814/0.686/0.908 — exact match to the reported numbers. MorphEvalHeb stays at 0.986/0.988/0.987 throughout, i.e. translation narrows the gap to FactCG but MorphEvalHeb remains far ahead, especially on native-Hebrew text (MAFAT-Aug: 0.98 vs 0.68–0.69). A handful (7/500) of translated texts tripped a Hebrew-leftover QC heuristic; inspected one (uid 2163) and it's benign — the translator kept a parenthetical Hebrew gloss of an English game title, not an actual translation failure.

**Negation-lost-in-translation** (`translation_erasure_doclevel.csv`, deduplicated by (model, gold sentence, corrupted sentence) to match the reported denominators): Nova Micro loses the negation in 19.9% of 206 pairs (reported: 19.4%), Haiku 4.5 in 27.4% of 208 pairs (reported: 27.4%, exact match). The one manual-verification file available (`translation_erasure_doclevel_manual_verification.csv`, 49 hand-read cases) confirms the automated negation-lost flag's precision: 17/19 real for Nova (0.89, matches the reported figure exactly) and 20/20 real for Haiku (1.00, exact match). **Caveat on provenance:** this file's `labeled_by` column reads "Claude (read in session), 2026-11-15" for every row — the "manual check" was the peer Claude session reading the examples itself, not an independent human verification. The actual judgments on inspection look correct, but this should not be described as human-verified in the paper or thesis without saying so.

### What does NOT hold up: "impossible year repaired in 44–63% of cases"

This claim (in both the peer session's report to the author and its draft paper snippet) could not be independently reproduced, for two reasons:
1. **No script on disk defines "impossible."** `grep -rl "impossible" paper/*.py` finds nothing; the 44%/63% figures in `translation_experiments_summary.csv` were computed by an unsaved, ad-hoc analysis, not one of the recorded `exp_*.py` scripts — a direct violation of this project's own recording policy (line 5 above).
2. **The one manual check of the underlying detector shows it mostly fires on non-errors.** Of the 10 manually-read `number_plausible` cases for Haiku (the same `translation_erasure_doclevel_manual_verification.csv`), only 3 are real (`manual_label` starting with `real:`); the other 7 are `false_alarm:` cases — the automated "number erased" check is a literal substring match on digit strings, and it misfires whenever the translator reasonably expands a bare decade/day into a conventional English form ("40"→"1940s", "6"→"June", "4"→"fourth", "90"→"1990s", etc.) that still correctly conveys the (possibly corrupted) value. That is 30% precision on the one sample checked — the true "world-knowledge repair" rate is very likely far below the headline 44–63%, most of which is a formatting-convention artifact of the detector, not the phenomenon it claims to measure.

**Conclusion:** two of the three findings from this follow-up are real and strengthen the paper's existing translation discussion with a larger, better-controlled, two-commercial-translator result; the third ("impossible values get silently repaired from world knowledge") is not supported by the data as currently reported and should not go into the paper without either a real, defined, reproducible detector or being dropped.

**Recommendation:** replace the simpler single-translator paragraph already in §6 (added from §15, TranslateGemma-only) with the translate-then-check AUC table and the manually-verified negation-lost finding from this experiment — it is strictly stronger evidence (two real commercial translators, 250–260 pairs vs. a smaller informal sample, bootstrap CIs, a calibrated precision check). Do not include the "impossible year" claim in its current form. Author's call on whether to invest in properly re-deriving an "impossible value" detector as a follow-up, given the manual check suggests it's mostly noise.

---

## 17b. The "impossible year" claim, redone properly — neither session's framing was quite right, but a real finding survives

**Context:** following up on §17's dropped "impossible values repaired from world knowledge" claim. The peer session (a different Claude Code session) pushed back, correctly pointing out that my §17 precision check (3/10 real) had actually checked a different, mixed bag of cases (impossible days, a physics constant, decade-format artifacts), not specifically impossible *years*, and that it had separately hand-read 8/8 real impossible-year cases without saving the labels. Rather than trust either side, I went back to the already-saved raw data (`translation_erasure_doclevel.csv` — no new API calls) and built a reproducible check.

**Key realization:** the augmentation pipeline's number-injection rule is deterministic doubling (`acl_latex.tex` §4.2). Applied to a year (1900–2024), doubling *always* produces an obviously-impossible 4-digit value (3800–4048) — this isn't a separate "impossible value" injection mechanism, it's a predictable side effect of doubling applied to years specifically. That makes "true original year" directly recoverable for most cases (corrupted ≈ 2 × gold), which lets us check not just *whether* translation masks the impossible year, but *what it replaces it with*.

**Method:** script `exp_translation_impossible_year.py`, output `impossible_year.csv`. Found 208 edited sentences (115 pairs) with an implausible 4-digit number in the corrupted text; 70.2% are exactly explained by the doubling rule (corrupted = 2×gold ±1).

**Result:**
| Model | Masked/changed in English | Of masked cases: recovered the TRUE original year | Of masked cases: explained by a bare leading-digit swap (e.g. 4034→2034) |
|---|---|---|---|
| Nova Micro | 43.4% of 106 | **0.0%** of 27 | 60.0% of 40 |
| Haiku 4.5 | 56.9% of 102 | **10.0%** of 40 | 68.6% of 51 |

**Conclusion — both sessions were partly wrong about the mechanism, both right that something real is happening:**
- The rate (~43–57%, close to but not identical to the original 44%/63%, reflecting a more precise, reproducible definition) is real and reproducible from already-saved data.
- The **"repaired from world knowledge" framing is wrong**: the translator recovers the actual true original year only 0–10% of the time. It is essentially never doing real fact-checking.
- What's actually happening, most of the time (60–69%): the translator does a superficial, context-free fix — swap the implausible leading digit (3→1, 4→2) to produce a different, still-fabricated-but-plausible-looking year. E.g. corrupted "3970" (the true year 1985, doubled) becomes English "1970" — not because anyone knew the true year was 1985, just because "3970" doesn't look like a year and "1970" does.
- This is, if anything, a **stronger and more precise point for the paper** than either session's original framing: translate-then-check doesn't just occasionally miss an error, it can actively launder an obviously-corrupted value into a confident, plausible-looking (but still wrong) one, with no connection to the actual fact — exactly the silent-failure-mode critique the paper is making about cross-lingual transfer.

**Recommendation:** this is now a real, reproducible, saved finding (unlike the original claim) and can go in the paper with careful wording. Suggested sentence for the appendix "Translation erasure" paragraph: *"Of the subset where the pipeline's deterministic number-doubling rule lands on an implausible year (e.g. 3970), translation masks the implausible value 43--57\% of the time -- but almost never (0--10\%) by recovering the true original year; instead it typically swaps only the leading digit to produce a different, equally fabricated but plausible-looking year (e.g. 3970 $\to$ 1970, not the true 1985), masking the corruption without correcting it."* Author's/peer session's call on whether to add this to the paper now that it's properly verified.

---

## 18. Do current models still make Hebrew-morphology errors? (co-author objection follow-up) — ✅ real finding, after fixing a judge-calibration bug

**Context:** a co-author (listed first on the paper) raised a pointed objection in a team chat: the taxonomy and error-finder data were built on GPT-3.5/Gemini-1.5-2.0/DictaLM-2.0-era models, which are "dead" now — so maybe current (Nov 2026) models no longer make these errors at all, undermining the paper's motivation. Rather than argue it in the abstract, this was tested directly.

**Part 1 — generation.** `exp_current_model_summaries.py` + `_round2.py`: 69 fresh Hebrew summaries from `dicta-il/DictaLM-3.0-1.7B-Instruct` (Apache-2.0, Feb 2026 — the current generation of a model family already represented as v2.0 in the original data), on 69 MAFAT articles stratified across all 5 sources, using the exact Hebrew prompt (verbatim, including its original spacing/typo) that generated the original `summaries_dicta_*.jsonl` files, for a fair like-for-like comparison. Smoke-tested first (49s load, 1.58GB RSS, 34 tok/s) given tight system memory. Raw output: `current_model_summaries_dictalm3.csv` (+ `_round2.csv`), 69 rows with full article/summary/generation-time per row.

**Part 2 — error-finding, attempt 1 (flawed).** Rather than wait on broken API credentials (all 3 Gemini keys in `.env` had been auto-revoked by Google as leaked; the direct Anthropic key was also dead, 401), the exact error-finder system+user prompt and Pydantic schema were extracted verbatim from `correctness_errors_prompts.py`/`correctness_errors_llm_helper_functions.py` (confirming the pipeline's *active* code path already calls Claude, not Gemini — the Gemini-named CSVs are from an earlier config), and applied directly by Claude subagents (4 parallel batches, 69 pairs) instead of an API call. **Result: 0/69.** As a control, the identical process was run on 69 pairs of the *original* old-model summaries (via `exp_severity.py`'s `load_error_finder()`, `thesisdatasets/error_finder/*.csv`). **Result: 1/69** (one genuine hit, a pronoun-referent "cactus spoke" error). The two were statistically indistinguishable, suggesting either "models have converged to near-zero on this error class" or "the judge is miscalibrated." Initial read favored the latter concern being secondary; this was wrong.

**The bug, found by cross-checking against real production output.** The user pointed to `thesisdatasets/slides/Amit Avraham_correctness_errors_finder - new open source models_new.md`, which documents dozens of real, production-confirmed qualifying errors (gender reversals, binyan swaps, smixut confusion) from the actual pipeline run on gemma3-12b/gpt-oss-20b/qwen3-14b summaries. Cross-referencing found **7 of these confirmed-positive doc_ids were inside the 69-pair control sample** — and the control judge had rejected all of them, including `business-6667074` (a feminine→masculine identity reversal on a legal claimant, scored **severity 5** by the real pipeline), on the grounds that "the reader could still identify the referent via the wing name, so it's grammar not fact." That reasoning is wrong relative to the project's own standard: the real pipeline treats a person's factual gender being flipped as qualifying regardless of whether the entity stays identifiable through other means. **The subagents' strictness criterion itself was miscalibrated, not just "conservative."**

**Part 2, redone with calibration.** All 8 batches (4 DictaLM-3.0, 4 old-model) were rerun with the identical prompt plus 3 real confirmed-positive examples from the slides file as explicit calibration anchors (the `business-6667074` gender-reversal, a `culture-4533684` binyan swap שמע/השמיע, and a Chenin-Blanc→"grapes" specificity-loss case), with explicit instruction not to require that the error create referent *confusion* — only that it assert something factually different via a Hebrew morphological mechanism. Each agent was told to report its count honestly even if unchanged from the flawed pass, not to force a result either way.

**Validation that the fix worked:** `business-6667074` was correctly re-flagged, independently, at severity 5, by the agent re-checking its batch — an exact match to the real pipeline's own historical verdict on that exact case. This confirms the recalibrated judge is now actually aligned with the project's real standard, not just more lenient in some arbitrary direction.

**Final result (n=69 each, properly calibrated):**
| Group | Qualifying pairs | Errors | Severities |
|---|---|---|---|
| DictaLM 3.0 (current, Feb 2026) | 1/69 (1.4%) | 1 | 2 (relativizer/conjunction clitic swap, ש-→ו-, mild) |
| Old models (GPT-3.5/DictaLM-2.0/Gemma-3/gpt-oss/Qwen3) | 2/69 (2.9%) | 2 | 5 (gender-identity reversal), 3 (pronoun-referent "cactus spoke") |

Raw data: `current_vs_old_model_errorfinder_recalibrated.csv` (138 rows, one per judged pair, full error detail where found). Scripts: `exp_current_model_summaries.py`, `_round2.py`; batch/result JSON files were scratch artifacts under `/tmp`, not preserved as permanent files (the CSV above is the permanent record).

**Conclusion:** 1 vs. 2 out of 69 is not a statistically meaningful difference at this sample size — no claim that either model generation is "better" on this specific error class is supportable from this data. But both numbers are now real and trustworthy (unlike the flawed first pass, which wrongly read as "the taxonomy barely fires on anyone"). The honest, defensible answer to the co-author's objection: **this specific, narrow Hebrew-morphology error class is real, low-frequency (~1–3% of natural summaries in this sample), and still demonstrably occurs in a current, Hebrew-native model** (DictaLM 3.0) — which refutes the strong form of the objection ("current models have fixed this entirely") while confirming the already-known fact that these errors are rare in natural generation (consistent with the severity-calibration finding, §10: only 9.1% of real errors are flagged, a different but related rarity signal). This is also a validated methodology point for the project generally: **when using an LLM as a judge/error-finder, calibrate it against real confirmed-positive examples before trusting a null/near-null result** — a strict-sounding rubric applied by a fresh judge can silently diverge from the actual historical standard, and a "both come back zero" result should raise suspicion of judge miscalibration, not be accepted at face value.

**Recommendation:** worth a short note in the thesis methodology/limitations section about the judge-calibration lesson. The headline numbers (1/69 vs 2/69) are too small and unpowered to cite as a standalone paper claim, but the finding that DictaLM 3.0 (current) still produces this error class at all is a useful, concrete answer if the objection comes up again. Not a paper addition — a thesis/discussion-record item and direct ammunition for the Bracha/Eyal meeting.

**Second caveat, found later (same day): a generation-truncation bug affecting precision, not direction.** Checked after a follow-up power-up experiment (§19) exposed the same bug there: the DictaLM 3.0 summaries here were generated with `max_new_tokens=512`, and 29 of the 69 (42%) are cut off mid-sentence, not fully generated. The one confirmed error (doc_id `218050`, the ש-/ו- clitic swap) occurs in a sentence that *did* fully generate before the cutoff, so that specific finding is unaffected. But the 1/69 and 2/69 counts should now be read as **lower bounds, not precise rates** — truncation can only hide additional errors in the cut-off 42%, it cannot fabricate the one that was found. This does not reverse the conclusion (current models still produce this error class); if anything it means the true rate is probably at or above what's reported. The old-model control (fixed historical data, not generated by this session) is unaffected. See §19 for the properly truncation-checked follow-up.

---

## 19. Prompt-style sensitivity on the co-author objection: thin vs. long-form vs. MAFAT-matched "restricted" prompts, plus a held-out 50-article pool — ✅ real findings, several open items

**Context.** §18 answered the co-author's objection with a single "thin" prompt on 69 articles (1/69 current-model, 2/69 old-model — too small to be a paper claim). This experiment scales that up across all 5 current-generation models available on Bedrock (GLM-4.7-Flash, Ministral-3-14B, Qwen3-32B, Claude Haiku 4.5, GPT-5.6-Luna), across 3 different prompt *styles*, and checks a structural concern the first pass didn't address: does the generation *prompt itself* bias the error rate by producing summaries that don't structurally resemble real MAFAT human references?

**Part A — real-MAFAT structural baseline.** Computed directly from `combined.csv`'s `original_hypothesis` column (human-written reference summaries, n=215 unique premises): mean compression ratio (summary chars / article chars) **16.7%**, median 15.4%; mean words **172**; mean sentences **10.6**. This is the target to match, not just "some length."

**Part B — three prompt regimes compared against that baseline.**
- **"thin"** (`exp_current_model_summaries_targeted.py` and its offset variants): minimal length instruction, highly variable output length across models/rounds.
- **"long-form"** (`exp_length_corrected_val20_bedrock.py` / `_mantle.py`, `LONGFORM_SYSTEM` prompt, "at least 15 sentences, no upper cap" — the real production prompt verbatim from `summary_generator_close_llm.py`/`summary_generator_ollama.py`): mean compression **29%–86.5%** across models/batches — far above the 16.7% MAFAT target, i.e. these summaries are structurally much longer/more detailed than what real human MAFAT summarizers produce.
- **"restricted"** (same files, prompt tightened to "between 8 and 12 sentences, approx. 150-200 words total — a binding, exact range, not shorter and not longer," `maxTokens` cut from 4000 to 1000, output suffix `_restricted_val20...`): mean compression **12.3%–23.0%** — much closer to the 16.7%/172-word/10.6-sentence MAFAT profile, though still variable by model.

This session generated and judged **750 long-form + 650 restricted + 665 thin-prompt summaries** across the 5 current models before moving to the new-pool sub-experiment below.

**Result — error rate drops sharply as the prompt tightens toward MAFAT's real shape.** On the same rank range (the "known hard" article pool from `thesisdatasets/error_finder/*.csv`, selected by original max-severity), the confirmed Hebrew-morphology-error rate was **5.2% under the long-form prompt vs. 0.8% under the restricted prompt**. Separately, the restricted prompt's rate on this original pool was **2.0%** when aggregated across a larger combined sample (see Part D below for the comparison point against the new pool). **IMPORTANT, found later (Part C below): these rates were computed from the original, pre-re-verification 72-finding master list, which Part C shows was ~71% false positives. The true confirmed-error counts behind these percentages are much smaller (16 of 72 survived re-verification), so the 5.2%/0.8%/2.0% figures themselves are very likely overstated and have NOT been recomputed from the re-verified counts — treat the *direction* of this result (long-form ≫ restricted) as still credible, since the false-positive rate does not look prompt-style-specific, but do not cite these exact percentages without redoing this arithmetic from the Part C verdicts.**

**Conclusion (Part A/B).** None of the three prompt styles is a perfect match for real MAFAT structure, but "restricted" is by far the closest, and — importantly for the paper's framing — the taxonomy's error class is **generation-style-dependent**: a verbose, uncapped-length prompt elicits roughly 6x more confirmed errors than a length-matched one on the identical article set. This sharpens, but does not resolve, the original co-author objection: it means "how do you prompt the model" is itself a confound that has to be controlled for before citing any single error rate as *the* rate for a given model family.

**Part C — a 72-finding master list, FULLY re-verified for false positives (completed 2026-11-17).** All new-model-only confirmed findings from this session (19 thin-prompt + 40 long-form + 13 restricted-prompt = 72, excluding the 3 old-model calibration anchors from §18) were compiled into a master list: `paper/results/eval/findings_master_list_72.md`. An initial "clean false positives" pass on the GLM-thin-prompt slice only (8 of 72) found 4/8 did not reproduce. This was then extended to a full BLIND re-verification of every finding with a recoverable doc_id (55 of 72 — 17 could not be checked because the original doc_id was never recorded), reading each premise/summary pair fresh and judging it against the mandatory gate + English-contrast filter + "must positively assert something false" rule, without being primed by what the original judge claimed to have found.

**Result: only 16 of the 55 checkable findings reproduced (29%); 39 did not reproduce and are rejected as false positives.** This is a large downward revision from the original 72. Per-model/prompt breakdown of the 16 confirmed: DictaLM-thin 1 (218050, already known from §18), GLM-thin 4 (food-14044992, animals-6829845, food-333642, culture-14100072, plus general-14268637 reinstated = 5 total for GLM-thin), Qwen-thin 2 (celebrities-8932596's two sub-findings), GLM-longform 3 (business-6667074's own reproduction of the calibration-anchor pattern, business-13302729's "המחבר החליטה" author-gender mismatch, and culture-14100072's "שהם סמיט"→"הם סמיט" dropped-clitic error), Ministral-longform 1 (business-13302729's "המארחת"→"המארח" self-contradiction, weak on the English-contrast test), Qwen-longform 1 (general-6866595's "פרופ' אש דרשה" — fem verb on the real, named, male health official Nachman Ash), GLM-restricted 1 (culture-14100072's "הם סמיט" error reproducing again under the restricted prompt — the single strongest, most-replicated finding in the whole list, confirmed independently under two different prompt styles by the same model on the same article), Ministral-restricted 1 (general-14268637's "בנה"/"יהודית" son/daughter mismatch — a **new** finding not matching the original judge's unrecorded reasoning, discovered fresh on this pass), Qwen-restricted 1 (general-14061589's "הפעילה...לבדה...והפיל" triple-inconsistent verb chain on the real, named, male soldier Yisrael Goldstein).

A notable sub-pattern: general-14061589 and general-6866595 (the Goldstein and Ash articles) were each originally claimed as errors under BOTH the long-form and restricted prompts for Qwen. Re-verification shows each one only reproduces under ONE of the two styles (Goldstein breaks only under restricted; Ash breaks only under long-form) — suggesting prompt style changes *which* specific errors a model makes on a given article, not just how many, which is a genuinely new and interesting methodological observation, not just noise.

**Caveat:** this was one blind read per pair, not an exhaustive sentence-by-sentence parse of every long-form summary (several ran 5,000-9,000 characters) — some false negatives among the 39 rejections are plausible, especially in the longest long-form summaries, and the 17 unrecoverable-doc_id findings remain of genuinely unknown status (should not be assumed confirmed). The headline conclusion is nonetheless clear: **the originally-reported "72 confirmed findings" figure was substantially inflated**, and any future citation of this session's error count should use the re-verified ~16 (plus whatever fraction of the 17 unknowns would survive the same scrutiny), not 72.

**Part D — held-out 50-article pool, selected for likely difficulty.** To test whether the taxonomy's error class only shows up on the specific ~200-article pool this project has reused throughout, 50 brand-new MAFAT articles — never used in any prior round — were selected from `combined.csv` sources this project had never tested (Knesset parliamentary transcripts, Bagatz legal rulings, Wikipedia encyclopedia articles, Weizmann science explainers; the prior work used only journalism sources). Selection method (`new50_candidates.csv`): a keyword-density "hardness" score (KINSHIP terms ×1, ROLE_GENDER terms ×1.5, PASSIVE-binyan markers ×1.2) over each candidate's premise text, taking all 21 available Knesset and all 4 available Bagatz articles, plus the top-scoring 15 Wikipedia and top-scoring 10 Weizmann articles.

Generated with the single, shared "restricted" prompt (the Part B winner) across all 5 models: `exp_new50_restricted_bedrock.py` (GLM/Ministral/Qwen/Haiku) and `exp_new50_restricted_mantle.py` (Luna). 250 summaries total (`current_model_summaries_<tag>_new50_restricted.csv`, one file per model). Mean compression after the 2026-11-17 data-quality fix (below): GLM 10.0%, Ministral 15.5%, Qwen 10.8%, Haiku 14.1%, Luna 18.5%.

Error-finding: 10 parallel subagents (25 rows each) applying the mandatory-gate + English-contrast methodology from `docs/error_finder_instructions_for_hebrew_llm.md`. Full writeups, including one coordinator correction, are in `paper/results/eval/new50_confirmed_findings.md`. Result:

| Model | New pool (50 articles) |
|---|---|
| GLM-4.7-Flash | 1/50 |
| Ministral-3-14B | 0/50 |
| Qwen3-32B | 2/50 (coordinator-corrected down from the agent's reported 4/50 — 2 of the 4 were downgraded for failing the "reader would believe something genuinely different" bar, consistent with a sibling Ministral agent's rejection of a near-identical construction on the same doc_id) |
| Claude Haiku 4.5 | 0/50 |
| GPT-5.6-Luna | 0/50 |
| **Total** | **3/250 (1.2%)** |

**Register-dependence finding (unexpected).** The new pool's 1.2% confirmed-error rate is *lower* than the original journalism-pool's restricted-prompt rate (2.0%, Part B), despite the new pool being deliberately selected for likely difficulty. All 3 confirmed findings are Knesset articles; zero came from Bagatz, Wikipedia, or Weizmann (58% of the pool). Three candidate explanations, none confirmed: (a) Weizmann's impersonal science content has essentially no gendered human entities to flip; (b) formal/legal/parliamentary/encyclopedic registers tend to pair explicit gendered titles with full names, which may reduce genuine reader-facing ambiguity even when an isolated grammar slip occurs; (c) the severe script-contamination problem in this batch (below) may mask or divert attention from genuine morphological errors during judging. Not resolved — flagged as a limitation of this specific test, not a claim about registers in general.

**Data-quality issues found, and fixes applied (2026-11-17).**
- **Luna:** 3 empty + 3 truncated-mid-sentence rows in the new-pool file (`23_ptm_598304`, `632907`, `2009-11-17-...periodic-table` empty; `16_ptm_129101`, `2022-01-31-...lightning...`, `2019-06-03-...new-foundations` truncated). Fixed by regenerating just those 6 rows in place with `max_output_tokens` raised from 1000 to 1500 and a retry-on-empty loop. Pre-fix content backed up to `pre_fix_backups/luna_new50_pre_fix_backup.json`.
- **GLM:** 2 truncated-mid-sentence rows (`16_ptm_533324`, `23_ptv_583794`). Fixed the same way (`maxTokens` 1000→1500). Pre-fix content backed up to `pre_fix_backups/glm_new50_pre_fix_backup.csv`. **Important side effect:** `16_ptm_533324` is the confirmed GLM finding above (the מועמדתו/מועמדת clitic-drop error) — the regenerated (fixed) version of this row no longer contains the error (it omits that sentence of the article entirely). The error-bearing original text is preserved in the backup and in `new50_confirmed_findings.md`; the finding stands as something that genuinely happened in the original run, and its disappearance on a fresh sample at the same temperature (0.2) is itself informative about how fragile/sample-dependent a single confirmed hit can be — a caution against over-reading any single-digit confirmed-error count from a single generation run.
- All 8 regenerated rows (2 GLM + 6 Luna) were re-checked for new morphology errors after the fix; none were found, so the total remains 3/250.
- **Ministral:** newly discovered severe non-Hebrew-script contamination (CJK/Arabic/Cyrillic/Thai fragments spliced mid-word) on 36%+ of rows in this batch specifically — not previously seen for Ministral on the original journalism pool, and apparently triggered by the longer/denser Knesset/Wikipedia/Weizmann source text. **Not fixed** — this is model generation behavior, not a truncation/empty-output bug, and the same prompt/model would likely reproduce it; flagged here as a known caveat on this batch's data quality rather than attempted away.
- **Qwen:** confirmed to have the same pervasive multi-script contamination problem documented throughout this project's history (up to 64% of rows in one new-pool batch) — a known, pre-existing, non-prompt-specific issue, not fixed, consistent with the project's earlier explicit decision to keep using Qwen's output despite this.

**Conclusion.** The co-author's objection ("current models have fixed this") remains refuted, on the strength of the new-50-pool result alone (Part D, independently re-verified and solid: 3/250 = 1.2%, on articles with zero prior exposure in this project): current models (Nov 2026 generation) still produce this exact taxonomy's error class. The *exact* rate to cite, however, is now less certain than it looked mid-session: Part C's full re-verification found that the original pool's 72-finding count (behind the 5.2%/0.8%/2.0% figures above) was ~71% false positives once checked blind, leaving only 16 genuinely confirmed. The error class is real, reproducible, and rare in natural generation — but any headline rate for the *original* pool should be recomputed from the Part C verdicts before being cited; only the new-50-pool's 1.2% (Part D) is currently trustworthy as stated. The paper's framing should present the error class as real and reproducible but rare, disclose the prompt-style sensitivity as a methodological caveat, AND disclose that this session's own error-finding process had a high false-positive rate on first pass — which is itself a relevant methodological finding about LLM-judge reliability for this taxonomy, echoing the judge-miscalibration lesson from §18.

**Recommendation.** Worth the §18-style "real finding, with limitations" treatment in the thesis/methodology section: cite the MAFAT-structural-match result (Part A/B) as a methodology contribution (any future generation-error study on this taxonomy should calibrate its prompt to the target corpus's real length/compression profile first), and cite the new-50-pool result (Part D) as the strongest available evidence against the "current models don't make this error anymore" objection, since it's built on articles and registers with zero prior exposure in this project. The partially-completed false-positive cleanup (Part C) and the unresolved register-dependence finding (Part D) should be disclosed as open items, not silently finished or silently dropped.

---

## 20. Where are error types represented inside MorphEvalHeb, and which directions does it use? (representation learning + activation engineering, 2026-11-21) — ✅ positive, partial (4 of 20 types labeled)

**Literature / novelty:** `paper/research/drl_steering_representation_literature.md` (~30 papers, 9 ranked leads). Gap: error-type probing exists only for decoder LLMs/judges (Orgad et al. 2410.02707; "Inside the Unfair Judge" 2607.11871); no layer-wise probing or concept erasure of a fine-tuned faithfulness cross-encoder, and no Hebrew-encoder morphology probing.

**Data:** 740 sentence-level (faithful, corrupted) pairs from `multisource_units.csv` (Injected part), stratified by listed error type (≤40 per type, ≤2 per article per type) → `paper/results/eval/layers_sample.csv`. Error type per sentence = rule-based `edit_kind()` (exp_translation.py): negation 95, number 141, agreement_suffix 57, definiteness 16, other 410, mixed 21. **LLM labeling into all 20 types is coded (`exp_layers.py label`) but blocked: Bedrock token expired.**

**Method (`paper/results/eval/exp_layers.py`):** each sentence scored with the full article (max 3,072 tokens). Hidden states at all 28 layers of the fine-tuned MorphEvalHeb and the base `dicta-il/neodictabert` (CLS token, and mean over the summary-sentence tokens) → `cache/layers_emb_{ft,base}.npz`. Linear probes (logistic regression, 5-fold grouped by article). On these units MorphEvalHeb's own output: paired detection 0.936, flagged 0.49, false alarms 0.036.

**Results 1 — probing** (`layers_probe.csv`, `layers_probe_by_type.csv`, `layers_type_encoding.csv`; `figures/layers_probe.png`, `figures/layers_tsne_{ft,base}.png`):
- Faithful vs corrupted from the CLS token: fine-tuned 0.59 (layers 1–4) → 0.85 (L12) → **0.96 (L16–20)**; base model 0.60–0.73 at all layers. Fine-tuning builds the decision in layers ~10–16.
- From the summary-sentence tokens, base and fine-tuned both reach ~0.95 from layer 8 (surface difference of the edited words, not a judgment).
- Error type (probe on corrupted − faithful, 4 rule types, majority 0.46): ~0.75–0.86 in both models from the sentence tokens; from CLS only the fine-tuned model encodes it (0.78 vs base 0.45–0.6 at late layers).
- Clustering: types separate most in early layers (cosine silhouette ~0.2 at L1–8, negation a distinct cluster) and merge late (~0.02 at L28) → late layers collapse types into one "unfaithful" signal.
- Per type (fine-tuned CLS): negation and agreement already informative at L4 (0.72/0.63); definiteness and number later; all peak ~L20.

**Results 2 — causal erasure** (`layers_erase_summary.csv`, `layers_erase_summary_orth.csv`, raw `cache/layers_erase*.jsonl`, `figures/layers_erase.png`): mean (corrupted − faithful) direction per type from half of the articles, projected out at all 28 layers, re-scored on the other half (148 units, ≤40 per type). Random-direction control: no change (±0.03).
- **Shared "all types" direction: detection drops to chance** (paired 0.36–0.55 vs 0.85–0.98; flagged −0.28 to −0.85). One shared direction carries the decision causally.
- Full type directions overlap it (cos 0.36–0.71) and shift all scores globally: erasing negation flags everything (false alarms 100%); erasing number/agreement flags nothing. Not selective.
- **Type-specific part (shared component removed):** erasing the negation-specific part cuts flagged negation sentences 0.85 → 0.32 (−0.52) vs −0.22/−0.14/+0.02 on other types; number-specific: number 0.70 → 0.35 (−0.35) vs −0.07 to −0.18. Agreement-specific: no effect (+0.05). Paired ranking barely changes (confidence lowered, not reversed). Definiteness n=6 test → not interpretable.

**Conclusion:** MorphEvalHeb decides in layers ~10–16 along one shared "unfaithful" direction, plus small type-specific components that causally matter for negation and number but not for gender/number agreement suffixes. Novel for faithfulness metrics; supports a paper/thesis section "What does a Hebrew faithfulness metric know and what does it use?".

**Limitations:** only 4 rule-defined types (lexical by construction, so early-layer type separation is partly expected); n=16–40 per type; erasure is rank-1 mean-difference projection (not LEACE); one model. **Next:** LLM labels for all 20 types (refresh Bedrock creds; ~740 Haiku calls), LEACE, activation patching, steering.

---

## 21. Hebrew-morphology tests: word-level probing, morphology attacker, activation patching, single-error sensitivity (2026-11-21) — ⚠️ important, partly negative

All local (no API). Edit types from Dicta's analyzer `dicta-il/dictabert-joint` (lemma, POS, gender/number/tense/person, prefixes, smixut) via `paper/results/eval/exp_morph_labels.py`; parses cached in `cache/morph_parse.jsonl`. Label validity: each morphology label comes from summaries listing the matching injected type 1.2–2.2× more often than chance (`layers_morph_validation.csv`); spot-check of 12 word pairs after two rule fixes ≈ 10/12 correct (labels are rule-based, not human).

**Test 1 — word-level probing** (`exp_layers_words.py`; 1,170 changed words in 1,004 sentences, ≤120 per type; `layers_words_probe.csv`, `layers_words_recall.csv`, `figures/layers_words_probe.png`, `figures/layers_words_tsne_{ft,base}.png`). Probe on (corrupted − faithful) word vector, 11 types (chance 0.09), 5-fold grouped by article. Balanced accuracy peaks ~0.5–0.58 at layers 8–12 in BOTH base and fine-tuned models; fine-tuning adds ≤0.07, and at layer 28 the fine-tuned model loses type structure (silhouette −0.07 vs base 0.02). Per-type recall at layer 8 (fine-tuned): negation 0.93, number 0.77, pronoun suffix 0.75, gender 0.69, definiteness 0.68, tense 0.67, prefix 0.56, entity 0.38, letter swap 0.26, root pattern 0.24, construct 0.23 (n=22). → Hebrew morphology changes ARE encoded (already in the pretrained encoder), root-pattern / construct-state / one-letter changes only weakly.

**Test 2 — morphology attacker** (`exp_morph_attack.py`; 200 faithful reference sentences, proportional by source; 1,723 single-word edits whose new form occurs in the benchmark corpus; ≤3 per type per sentence; `morph_attack_summary.csv`, `morph_attack_scored.csv`, `figures/morph_attack.png`). Faithful originals flagged: 0%. Missed (p(faithful) ≥ 0.5): definiteness 99% [98,100], prefix 98% [97,99], gender 94% [92,96] (verbs 93%, adj 95%), pronoun suffix 93% [87,97]; negation control 14% [9,20]. **Manual reading of 24 missed edits (by Claude, not a human):** gender 0/8 change meaning (agreement errors with the subject explicit, e.g. "וילנאי ציינה"); prefix ~1–2/8 (e.g. זכתה בהצלחה); pronoun ~2–3/8 (e.g. "תעודת הזהות שלה" for a man). Most attacker edits are grammatical, not factual.

**Test 3 — activation patching** (`exp_patching.py`; minimal pairs = faithful sentence with ONE typed word changed to its injected form; 80 detected pairs: entity 20, letter swap 20, root pattern 16, gender 12, number 10; `patching_summary.csv`, `cache/patching.jsonl`, `figures/patching.png`). Restoration from copying the faithful state: edited word 1.00 (L1) → 0.83 (L13) → 0.57 (L22) → 0; decision token (CLS) ~0 until L19, 0.40 at L25, 1.00 at L28; article tokens ~0 everywhere; other summary words ≤0.18 (higher for gender/number in mid layers: agreement spreads to neighbouring words). → The error is carried on the edited word until ~layer 22 and moved to the decision token only in the last ~6 layers; the article's representation never carries the decision.

**Single-error sensitivity — the key finding** (`exp_single_edit_summary.py`, `single_edit_summary.csv`). Of the minimal pairs, the share MorphEvalHeb flags at sentence level: construct 0/15, prefix 0/23, pronoun 0/83, tense 1/52, definiteness 1/58, gender 12%, number 14%, entity 20%, root pattern 20%, letter swap 28%. Missed examples are clear factual errors (rabbi Amital → "הרב נאמן", ממלכת ישראל → ממלכת ארם, 30 → 60 years) scored p = 1.00. Placed into the full faithful reference summary (realistic use; 203 summaries, faithful version not flagged): **flagged 4% overall** (entity 14%, letter swap 13%, number 4%, root 4%, gender/tense/prefix/pronoun/definiteness/construct 0%); only 7% even lower the score by >0.01.

**Conclusions:**
1. MorphEvalHeb's high detection on Injected (~94–100% per family) depends on summaries carrying many errors (~8 edited sentences, ~6 types). **One wrong word in a summary — even a swapped name or number — is almost never flagged (4%).** This is a major limitation; the paper must not claim single-error sensitivity, and the benchmark (multi-error negatives) overstates it.
2. "Knows but doesn't use": morphology changes are encoded in the representations (test 1), but the decision ignores them unless many errors accumulate (tests 2–3).
3. Mechanism (new for faithfulness metrics): error evidence sits on the edited word through ~L22, is moved to CLS in the last layers along one shared direction (§20).

**Open / next:** (a) compare baselines (SummaC, FactCG, LLM judge) on the same 203 single-error summaries — if they also miss them, it is a benchmark-wide finding, not ours alone; (b) detection vs number of errors (1, 2, 4, 8) curve; (c) human check of a sample of single edits; (d) fine-tune on single-error negatives (GPU server).

---

**Baselines on single-error summaries (2026-11-22, stopped at 140/203 by user; `exp_single_edit_baselines.py`, `single_edit_baselines_summary.csv`, `figures/single_vs_multi.png`).** Threshold = 5th pct of each metric's gold scores (5% FA). Flagged, single vs. multi (Injected): MorphEvalHeb 12% [8–19] vs 100%; FactCG 6% [3–12] vs 12%; SummaC-ZS 6% [3–11] vs 17%. Score below gold for single: MorphEvalHeb 84%, FactCG 13% (83% exact ties), SummaC 45% (chance). Per kind n≈14, too small to rank. Conclusion: no metric catches one wrong word in a full summary; MorphEvalHeb registers it most often (score drops) but rarely crosses the threshold. The gap single vs. multi is a benchmark/metric-wide finding, not MorphEvalHeb-specific. Unscored: last 63 items; LLM judge (Bedrock creds expired).

## Proposed, not yet run (awaiting prioritization — do not auto-run when the GPU frees up; wait for an explicit go-ahead)
