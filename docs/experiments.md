# MorphEvalHeb: experiment log

Quick-win micro-experiments run while the main metric evaluation (SummaC-ZS, FactCG) was busy on the GPU. All of these reuse already-cached scores (`paper/results/eval/cache/*.csv`) — no new model inference, no GPU contention. Scripts were throwaway (`_quickwin*.py` in `paper/results/eval/`, deleted after the numbers were captured); this file is the permanent record.

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

## Proposed, not yet run (awaiting prioritization — do not auto-run when the GPU frees up; wait for an explicit go-ahead)

### 5. Severity calibration on real LLM output (`thesisdatasets/error_finder/*.csv`)
Score the ~290 native-Hebrew LLM summaries (already carrying LLM-assigned, partly human-verified severity 1–5 ratings) with the existing binary verifier and multi-label detector. Test whether MorphEvalHeb's unfaithfulness probability — trained almost entirely on *injected* errors — tracks the *severity* of *naturally occurring* LLM errors. This both answers the "synthetic vs. real error generalization" reviewer objection (#P009) and reuses/completes research lead #3 already in `docs/tasks.md` (native Hebrew LLM summaries), using `metadata.generative_summaries` as corrected, not the MAFAT reference. Needs the binary verifier loaded once — run after SummaC (and FactCG, if still pending) finish.

### 6. Contrastive-embedding feasibility probe
Before considering a contrastive-loss retrain (expensive, needs the GPU server, can't be verified locally per the existing "rejected" note in `docs/tasks.md`), a cheap feasibility check: extract the *already fine-tuned* binary verifier's embeddings for faithful vs. unfaithful pairs and measure the separation (e.g. mean cosine-distance gap, or a silhouette score) in the existing embedding space. If faithful/unfaithful pairs are already well-separated under cross-entropy training, a contrastive objective is unlikely to add much; if they're not, that's a concrete, cheap argument *for* trying it. Needs one model load (the binary verifier, already cached) — run after SummaC/FactCG finish; no retraining required for this probe.
