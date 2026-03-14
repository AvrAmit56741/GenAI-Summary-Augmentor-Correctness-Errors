"""
Metric explanation text for use in Streamlit UI expanders.

Each constant is a markdown string describing a metric's definition,
sub-scores, range, and interpretation guidelines.

Add a new constant here whenever a new metric is added to the backend.
"""

ROUGE_EXPLANATION = """
### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

Measures **n-gram overlap** between a generated summary and the source document.

| Variant | What it counts |
|---|---|
| **ROUGE-1** | Unigram (single word) overlap |
| **ROUGE-2** | Bigram (two-word sequence) overlap |
| **ROUGE-L** | Longest Common Subsequence (LCS) — word-order aware |

**Sub-scores returned for each variant:**
- **Precision** — what fraction of the summary's n-grams appear in the source
- **Recall** — what fraction of the source's n-grams are covered by the summary
- **F1** — harmonic mean of precision and recall

**Focus:** Recall — how much of the source content is captured
**Range:** 0.0 to 1.0 (higher is better)
**Interpretation thresholds:**
- ROUGE-1 F1 > 0.4 → good content coverage
- ROUGE-2 F1 > 0.2 → good phrase-level matching
- ROUGE-L F1 > 0.3 → good sequence-level coverage

**איך לפרש את התוצאה (How to interpret the result):**

| Metric | משמעות |
|---|---|
| Precision = 1.0 | הסיכום לא הוסיף מידע שלא קיים במקור |
| Recall = 0.83 | רוב המידע מהמקור נשמר בסיכום |
| F1 = 0.91 | איכות הסיכום גבוהה מאוד |

> This interpretation applies equally to BERTScore (which also returns Precision, Recall, F1).
"""

BLEU_EXPLANATION = """
### BLEU (Bilingual Evaluation Understudy)

Measures **n-gram precision** — how many of the n-grams in the generated summary actually appear in the source.

**Formula:**

BLEU = BP × geometric_mean(precision₁, precision₂, ..., precisionₙ)

**Brevity Penalty (BP):**

- BP = 1.0 — if summary length ≥ source length (no penalty)
- BP = e^(1 − source_length / summary_length) — if summary is shorter than source

**Example** (reference = 6 words, hypothesis = 5 words):

BP = e^(1 − 6/5) = e^(−0.2) ≈ **0.82**

**Sub-scores returned:**
- **BLEU** — final score (BP × geometric mean of all n-gram precisions)
- **Brevity Penalty (BP)** — length penalty only; always > 0 as long as the summary has words
- **BLEU-1 / BLEU-2 / BLEU-3 / BLEU-4** — per-ngram precision breakdown

---

**⚠️ Why can BLEU = 0 while BP > 0?**

BP only measures **length ratio** — it is independent of n-gram matching.
BLEU = BP × geometric_mean(precisions). If **any single n-gram precision is 0**, the geometric mean collapses to 0, and BLEU = 0 — regardless of BP.

This happens when the summary has **zero matching n-grams** at any one order. For example:
- BLEU-1 precision = 0.8 (many unigrams match)
- BLEU-2 precision = 0.5 (some bigrams match)
- BLEU-3 precision = 0.0 ← **no 3-gram matches** (summary too short or paraphrased)
- BLEU-4 precision = 0.0 ← no 4-gram matches

Result: geometric_mean = 0 → **BLEU = 0**, but BP = 0.82 (non-zero because summary has words).

**Fix:** use a smaller maximum n-gram order (e.g., BLEU-2 instead of BLEU-4) for short summaries.

---

**איך לפרש את התוצאה (How to interpret the result):**

| Value | משמעות |
|---|---|
| BLEU > 0, BP = 1.0 | הסיכום ארוך מספיק ומדויק |
| BLEU > 0, BP < 1.0 | הסיכום מדויק אבל קצר מדי — קיבל עונש |
| BLEU = 0, BP > 0 | לפחות סדר n-gram אחד לא נמצא כלל בסיכום |
| BLEU = 0, BP = 0 | הסיכום ריק לחלוטין |

**Focus:** Precision — accuracy of generated text, penalizes hallucinated content
**Range:** 0.0 to 1.0 (higher is better)
**Interpretation thresholds:**
- BLEU > 0.3 → reasonable precision
- BLEU > 0.5 → strong precision
- BP < 0.8 → summary is significantly shorter than source
"""

CHRF_EXPLANATION = """
### ChrF / ChrF++ (Character N-gram F-score)

Measures **character-level n-gram overlap** between the generated text and reference.
Especially useful for morphologically rich languages like **Hebrew and Arabic**, where word boundaries
are less stable than in English.

**Sub-scores returned:**
- **ChrF score** — beta-weighted F-score over character n-grams (0–100 scale)
- **Precision** — fraction of hypothesis character n-grams found in reference
- **Recall** — fraction of reference character n-grams covered by hypothesis

**Focus:** Character-level similarity, beta=2 (favors recall)
**Range:** 0 to 100 (higher is better)
**Interpretation thresholds:**
- ChrF > 40 → good character-level similarity
- ChrF > 60 → strong character-level overlap
"""

BERTSCORE_EXPLANATION = """
### BERTScore (Contextual Embedding Similarity)

Measures **semantic similarity** using contextual token embeddings from a Hebrew BERT model.
Unlike n-gram metrics, BERTScore captures paraphrases and semantic equivalence.

**How it works:**
1. Encode both texts with Hebrew BERT → contextual embeddings per token
2. Compute pairwise cosine similarity between all token pairs
3. Greedy matching — each token finds its best match in the other text
4. Average similarities → Precision, Recall, F1

**Sub-scores returned:**
- **Precision** — avg max similarity: hypothesis tokens → reference tokens
- **Recall** — avg max similarity: reference tokens → hypothesis tokens
- **F1** — harmonic mean of precision and recall

**Focus:** Semantic equivalence
**Range:** 0.0 to 1.0 (higher is better)
**Interpretation thresholds:**
- F1 > 0.85 → very high semantic similarity (near-perfect paraphrase)
- F1 > 0.75 → good semantic overlap
- F1 < 0.60 → significant semantic drift

**Note:** Requires downloading a Hebrew BERT model (~500MB on first run).
BERTScore measures semantic similarity, **not factual correctness**.
"""

KENDALL_TAU_EXPLANATION = """
### Kendall τ (Tau) Correlation

Measures whether a metric score **reliably ranks** correct summaries above incorrect ones.

**How it works:**
- For every pair of samples, check: does the one with a higher metric score also have the correct label?
- τ = (concordant pairs − discordant pairs) / total pairs

**Range:** -1.0 to +1.0

| τ value | Interpretation |
|---|---|
| τ > 0.7 | Excellent — metric strongly predicts correctness |
| τ > 0.5 | Strong — metric reliably reflects faithfulness |
| τ > 0.3 | Moderate — some correlation with correctness |
| τ > 0.1 | Weak — limited correlation |
| -0.1 < τ < 0.1 | No correlation — metric does not reflect correctness |
| τ < -0.1 | Negative — metric is actively misleading |

**P-value:**
- p < 0.05 → statistically significant (reliable result)
- p > 0.05 → not significant (result may be random)

**Why this matters:** High τ means the metric can serve as a proxy for factual correctness.
Low τ means the metric measures surface similarity only, not faithfulness.
"""

OVERLAP_EXPLANATION = """
### Word Overlap

Counts the number of **unique words from the hypothesis (summary) that also appear in the premise (source)**.

**Formula:** overlap = |{words in summary} ∩ {words in source}|

**Example:**
- Source: "ראש הממשלה נפגש עם הנשיא בירושלים"
- Summary: "ראש הממשלה נסע לירושלים"
- Overlap: 3 (ראש, הממשלה, בירושלים/ירושלים — exact match only)

**Notes:**
- Matching is exact and case-sensitive
- Duplicate words in the summary are counted once (set-based)
- High overlap indicates the summary stays close to the source vocabulary
- Low overlap may signal paraphrasing or hallucination

**Range:** 0 to len(summary words) (higher means more shared vocabulary)
"""

COMPRESSION_RATIO_EXPLANATION = """
### Compression Ratio

Measures how much the summary **compresses** the source document.

**Formula:** compression = summary_length / source_length

**Example:**
- Source: 6 words → source_length = 6
- Summary: 5 words → summary_length = 5
- Compression ratio = 5 / 6 = **0.833**

**Interpretation:**
| Value | Meaning |
|---|---|
| ratio ≈ 1.0 | Summary is nearly the same length as the source |
| ratio < 0.5 | Strong compression — summary is less than half the source |
| ratio > 1.0 | Summary is longer than the source (expansion) |

**Typical range for abstractive summarization:** 0.1 – 0.4
**Units:** ratio (not percentage); multiply by 100 to get percentage
"""

METRIC_SELECTION_GUIDE = """
### When to Use Each Metric

| Metric | Best for |
|---|---|
| **ROUGE-1/2** | Standard abstractive summarization evaluation |
| **ROUGE-L** | When word order and sequence structure matters |
| **BLEU** | When precision matters (detecting hallucinated content) |
| **ChrF** | Hebrew text or morphologically variable outputs |
| **BERTScore** | Semantic similarity beyond exact word matches |
| **Kendall τ** | Evaluating whether a metric predicts factual correctness |
"""

ALL_METRICS_EXPLANATION = (
    ROUGE_EXPLANATION
    + "\n---\n"
    + BLEU_EXPLANATION
    + "\n---\n"
    + CHRF_EXPLANATION
    + "\n---\n"
    + BERTSCORE_EXPLANATION
    + "\n---\n"
    + KENDALL_TAU_EXPLANATION
    + "\n---\n"
    + METRIC_SELECTION_GUIDE
)
