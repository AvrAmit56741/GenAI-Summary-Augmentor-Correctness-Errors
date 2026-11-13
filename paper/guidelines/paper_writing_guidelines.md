# MorphEvalHeb Paper Writing Guidelines

Writing rules for the MorphEvalHeb ACL paper. They come from reading the 18 reference papers in `docs/parsed_papers/` for structure, tables, figures, results reporting and tone.

Every rule cites its evidence. "N/18" means how many of the papers follow the practice. Each rule ends with a **Draft** line saying where `MorphHebEval/latex/acl_latex.tex` currently stands. The numbering follows `styling_ai_guidelines.md` (#P001, #P002, …).

**Reference set**
- **Hebrew NLP (6):** AlephBERT (arXiv 2021, ACL 2022), NeoDictaBERT, HeQ, "Getting the ##life out of living" (SIGMORPHON 2020), Beyond N-Grams (ACL 2025).
- **English factuality metrics (7):** FactCC, SummaC, QAFactEval, TRUE, AlignScore, BARTScore, QASemConsistency ("Localizing Factual Inconsistencies").
- **Multilingual factuality (5):** mFACE, mFACT, Multi-FAct, SEAHORSE, STEMF ("Multilingual Self-Taught Faithfulness Evaluators").

The closest models for our paper are **SummaC, TRUE and AlignScore** (metric meta-evaluation), **SEAHORSE and mFACT** (multilingual factuality), **HeQ** (a Hebrew-specific metric) and **Beyond N-Grams** (metric correlation including Hebrew).

---

## Table of Contents

**A. Front matter**
- #P001 Title: "Name: descriptive subtitle"
- #P002 Abstract: 150–210 words, fixed order
- #P003 Name the "English-only" gap in the abstract

**B. Introduction**
- #P004 Intro of about one page, ending in an explicit contribution list
- #P005 Teaser on page 1–2
- #P006 Make the "why Hebrew breaks standard metrics" argument before the method

**C. Related work**
- #P007 Short related work with bold run-in topic heads and one gap sentence per topic

**D. Method, taxonomy and data**
- #P008 Error catalogue table: transformation, original, corrupted
- #P009 Motivate the error types with error analysis of real outputs
- #P010 Dataset statistics table
- #P011 Group train/test splits by source document
- #P012 Confront the translation confound with an explicit check

**E. Meta-evaluation protocol**
- #P013 Primary statistic: balanced accuracy and/or ROC AUC, with correlation secondary
- #P014 Write a paragraph justifying the evaluation protocol
- #P015 Human labels need an annotation protocol and agreement with a number
- #P016 Include trivial and upper-bound rows
- #P017 Significance testing with a bootstrap, defined in the caption
- #P018 Every baseline described in Setup must appear in the results table

**F. Tables and figures**
- #P019 Main table layout: metrics grouped by family, ours last, datasets as columns plus average
- #P020 Bold for the best result, with the convention stated in the caption
- #P021 Self-contained captions
- #P022 Hebrew examples in script, transliteration, gloss and translation
- #P023 Efficiency: parameters and/or throughput

**G. Analysis**
- #P024 Ablations that change one factor at a time
- #P025 Breakdowns by error family
- #P026 Qualitative error analysis with counts

**H. Back matter**
- #P027 Unnumbered Limitations after the Conclusion, with concrete items
- #P028 Ethics statement: data provenance, licensing and annotators
- #P029 Appendix: prompts as listings, hyperparameters, full tables
- #P030 Release links and license on page 1

**I. Tone and language**
- #P031 Confident headline claims with hedged explanations
- #P032 Frame the gap by type, not as "nobody did X"
- #P033 Citation hygiene: cite what the paper actually says

Checklist at the end.

---

# A. Front matter

## #P001 Title: "Name: descriptive subtitle"

Write the system name, a colon, then a descriptive subtitle. A bare acronym tells the reader nothing.

**Evidence**
- 5/6 Hebrew papers and 5/7 metric papers use the "NAME: …" pattern, for example "SummaC: Re-Visiting NLI-based Models…", "AlignScore: Evaluating Factual Consistency with a Unified Alignment Function" and "SEAHORSE: A Multilingual, Multifaceted Dataset for Summarization Evaluation".
- Papers without an acronym use fully descriptive titles: FactCC, mFACE, mFACT, QASemConsistency.
- None of the 18 has a bare-acronym title.

```
❌ MorphEvalHeb
✅ MorphEvalHeb: Morphology-Aware Factual Consistency Evaluation for Hebrew Summarization
```

**Draft:** the title is the bare "MorphEvalHeb". Add a subtitle.

## #P002 Abstract: 150–210 words, fixed order

Order: why it matters → gap → what we propose → how we evaluate → headline result → release.

**Evidence**
- Lengths:
  - metric papers 148–214 words (SummaC ≈148, TRUE ≈186, AlignScore ≈214);
  - multilingual papers 110–195 words;
  - Hebrew papers 168–224 words.
- Headline numbers are optional. 4/7 metric papers give them, such as SummaC "74.4%, +5 points" and BARTScore "16 of 22 settings". Every Hebrew paper keeps the result qualitative.
- AlignScore ends on its strongest hook: "355M … matches or even outperforms … ChatGPT and GPT-4".
- 6/7 metric papers announce the code release in the abstract or on page 1.

**Draft:** about 143 words, so in range. Add the release sentence. Consider ending on a concrete hook such as "a small Hebrew encoder outperforms an LLM judge" (give the exact parameter count), but only once #P013–#P015 are settled.

## #P003 Name the "English-only" gap in the abstract

**Evidence**
- All 5 multilingual papers name the English-centric gap in the abstract.
- AlignScore's own Limitations says it "only covers a single language, English". Cite that sentence directly as motivation.

**Draft:** the gap appears in the intro. Make sure one abstract sentence says it.

---

# B. Introduction

## #P004 Intro of about one page, ending in an explicit contribution list

Write 4–7 paragraphs (about 500–950 words) ending in an explicit contribution list. Each contribution should point to the section that delivers it.

**Evidence**
- Metric intros run about 530–960 words; TRUE's is the longest at about 930.
- Contribution lists are near-universal, but the format varies:
  - inline numbered: QAFactEval "1)–3)", TRUE "(1)–(4)", mFACE "three-fold: (1)(2)(3)";
  - "First… Second…": SummaC, and Beyond N-Grams' "three upshots";
  - bulleted: 4/5 multilingual papers. mFACT points each bullet at its section.
- Multi-FAct poses explicit research questions R1/R2 and phrases its subsection headings as questions.

**Draft:** the contributions are present and numbered inline. Add section pointers ("(§4)").

## #P005 Teaser on page 1–2

Show one Hebrew source/summary pair with the morphological error highlighted, and walk through it in the caption or text.

**Evidence**
- Metric papers: 5/7 have a page 1–2 teaser.
  - FactCC Table 1 shows real errors in green and red.
  - SummaC Figure 1 is an insight teaser.
  - QAFactEval Table 1 is a toy example.
  - TRUE Table 1 gives one example per task.
  - QASemConsistency compares one running example across 4 methods with ✓/✗.
- Multilingual papers: 4/5. SEAHORSE Fig 1 is a Russian summary with a bracketed English gloss and the note "wrong aspect", which is exactly our case. mFACT and Multi-FAct use pipeline teasers.
- Hebrew papers: AlephBERT-ACL's page 2 pipeline figure has a walk-through caption of about 90 words.

**Draft:** the first figures are `figure*` floats in §3–4, so there is no teaser. Promote one row of the error-analysis figure, such as the gender swap or the binyan swap, into a page 1 teaser.

## #P006 Make the "why Hebrew breaks standard metrics" argument before the method

**Evidence**
- 5/6 Hebrew papers argue that morphological rich languages (MRLs) break standard practice before presenting their method.
  - HeQ has a whole §2 "The Challenge" before its method.
  - The word-pieces paper is hypothesis-driven ("We conjecture").
- HeQ §4 lists the metric's desiderata as bullets before defining TLNLS: inflection invariance, span invariance, language independence and speed.

**Draft:** the intro argues this (lines 134–136). Consider a short "Desiderata for a Hebrew factuality metric" list before the classifiers, following HeQ. The word-pieces paper (not cited yet) is direct evidence that subword tokens do not match Hebrew morphemes; there, per-wordpiece tagging drops to 69.47 exact match from 94.44 with gold segments.

---

# C. Related work

## #P007 Short related work with bold run-in topic heads and one gap sentence per topic

Keep related work short. Use bold run-in heads for each topic, and end each topic with one sentence on what it leaves open.

**Evidence**
- The multilingual papers split related work by bold run-in topic heads, each ending in a gap sentence.
- Placement varies: TRUE (§6), mFACT (§9), HeQ and the word-pieces paper put it late.
- BARTScore has no related-work section; a T1–T4 taxonomy of metric paradigms (Figure 1) does that job.

**Draft:** it already uses bold heads and a closing "three gaps" paragraph, which is good. Keep it under about 0.75 pages. The early placement (§2) is acceptable; AlephBERT-ACL does the same.

---

# D. Method, taxonomy and data

## #P008 Error catalogue table: transformation, original, corrupted

Present the synthetic errors as a catalogue table with one row per error type and columns transformation, original, corrupted. Put the generation procedure in an algorithm box or a pipeline figure.

**Evidence**
- FactCC Table 2 has the columns Transformation | Original | Transformed, one paragraph per transformation, and Figure 1 as pseudo-code. It also gives size and balance ("1,003,355 … 50.2% negative").
- STEMF gives one prompt listing per error type in the appendix.
- Beyond N-Grams has a corruption-examples table for its synthetic third of the data.

**Draft:** Figure 2 shows 8 of 24 families. Add a one-line-per-family table of all 24 families (name, rule, short Hebrew example and gloss), in the appendix if space is short. State the corpus size and positive/negative balance.

## #P009 Motivate the error types with error analysis of real outputs

**Evidence**
- FactCC: "errors … most often related to … incorrect entity names, numbers, and pronouns" motivates its 4 transformations.
- FactCC and AlignScore both concede that synthetic errors may not match real error distributions. Reviewers will raise this, so answer it up front.

**Draft:** Figure 1 (errors in real LLM summaries) provides this. Quantify it: how many summaries were inspected, from which LLMs, and how often each family occurred. The `thesisdatasets/error_finder/` outputs contain that data.

## #P010 Dataset statistics table

Include a table with size, % positive, sources, average lengths and agreement.

**Evidence**
- 6/7 metric papers have a benchmark statistics table (size, % positive, agreement, source).
- AlignScore Table 1 lists its training data.
- SEAHORSE front-loads its scope numbers: 96K summaries, 6 dimensions, 6 languages, 9 systems, 4 datasets.
- AlephBERT Tables 1–2 give corpus statistics on page 2.

**Draft:** missing. Required columns:
- Split
- Source (Israel Hayom / Wikipedia / Weizmann / Knesset / Bagatz / translated mFACT)
- #docs and #pairs
- % positive
- Average article and summary length
- Label origin (human or synthetic)

## #P011 Group train/test splits by source document

Say explicitly that the split is grouped by source document.

**Evidence**
- HeQ splits by article to avoid leakage.
- QAFactEval runs ablations on an 80/20 split of the validation set "to preserve the integrity of the test set".
- SummaC discloses how it built its splits and the benchmark changes it made (relabelled Polytope).

**Draft and code:** `ui.py:2473` uses `StratifiedKFold` on the label only. Every article yields 7–10 corrupted variants, so variants of the same article can end up in both train and test. Re-split grouped by `doc_id`, for example with `StratifiedGroupKFold`, and state that in the paper.

## #P012 Confront the translation confound with an explicit check

**Evidence**
- 4/5 multilingual papers run a dedicated translation check.
  - mFACT Appendix A.4: 100 Chinese samples were checked by hand; 13 had errors and only 4 came from translation.
  - Multi-FAct Table 2 validates translation per language (0.80–0.84) and compares Google Translate with GPT-3.5 in Appendix B.
  - STEMF dropped Hindi because of poor translation.

**Draft:** 53% of the benchmark is translated mFACT data, while the paper argues that translation introduces artifacts. Either report results on the translated subset and the native subset separately (#P019), or run an mFACT-style manual audit of the Hebrew translations.

---

# E. Meta-evaluation protocol

## #P013 Primary statistic: balanced accuracy and/or ROC AUC, with correlation secondary

With binary labels, make balanced accuracy and/or ROC AUC the primary statistic. Report correlations (Kendall, Pearson, Spearman) as secondary.

**Evidence**
- 6/7 metric papers evaluate on binary labels with balanced accuracy and/or ROC AUC.
- TRUE recommends "reporting ROC AUC rather than correlations" and tunes thresholds by maximising √(TPR·(1−FPR)).
- SummaC uses balanced accuracy as primary because class imbalance ranges "from 6% to 91%", with ROC AUC secondary. It notes that correlation is "an uncommon choice to measure performance of a classifier due to the discrete and binary label".
- SEAHORSE reports Pearson plus ROC AUC, which is "not sensitive to a thresholding value".
- STEMF uses balanced accuracy only. QASemConsistency pairs balanced accuracy and AUC columns per dataset.
- Correlations go to the appendix: QAFactEval puts Spearman and Kendall there, Beyond N-Grams puts Spearman there.
- Technical note: with a binary label, Kendall τ_b is a rescaling of AUC (Somers' D = 2·AUC − 1) that depends on the class ratio. Reporting AUC directly is cleaner.

**Draft:** Kendall τ is the only statistic. Add ROC AUC and balanced accuracy, with the threshold tuned on dev, and keep τ as a secondary column.

## #P014 Write a paragraph justifying the evaluation protocol

**Evidence**
- 4/7 metric papers justify their statistic in the text: SummaC §4.3, TRUE, SEAHORSE, Beyond N-Grams.
- QASemConsistency states that it uses a fixed 0.5 threshold with no tuning.
- Beyond N-Grams includes a power analysis to justify n≈400.

**Draft:** the "Evaluation Protocol" subsection is empty. It must say:
- which statistic is primary and why;
- the threshold protocol;
- the test-set size and composition;
- where the labels come from.

## #P015 Human labels need an annotation protocol and agreement with a number

**Evidence**
- SEAHORSE reports pairwise agreement per language (82% overall) and Krippendorff's α.
- mFACT honestly reports "fair… 0.28 Fleiss' κ".
- Beyond N-Grams gives α per language and criterion (Hebrew 0.71/0.65) and discloses annotator pay.
- QASemConsistency compares its κ (0.72/0.78/0.67) against FRANK 0.58.
- Avoid what mFACE did: it claims "high inter-annotator agreement" with no coefficient.
- Annotators were native or bilingual speakers who passed a screening set in SEAHORSE, mFACE (Appendix D, 388 raters) and Beyond N-Grams.

**Draft:** the paper says τ is computed against "human factuality annotations" (line 249). In the repo, the benchmark labels (`benchmark/combined.csv`) come from the augmentation pipeline and from translated mFACT; I found no human annotation in the repo. Two options:
- add a human-annotated test set of real LLM summaries, with the protocol and α/κ reported; or
- reword the paper to say "synthetic labels".

This is the most important fix in the paper.

## #P016 Include trivial and upper-bound rows

**Evidence**
- SEAHORSE includes majority-class and ROUGE-L baselines and an "upper-bound" row marked with \*.
- TRUE strikes through contaminated results and leaves them out of the average.
- AlignScore reports **AVG-ZS**, the average over datasets not seen in training.

**Draft:** add a majority-class row. Because MorphEvalHeb is trained on the same pipeline that produced 36% of the benchmark, also report a held-out average on data it never saw during training, in the style of AVG-ZS.

## #P017 Significance testing with a bootstrap, defined in the caption

**Evidence**
- 4/7 metric papers use bootstrap significance against the best prior metric:
  - SummaC uses \*/\*\* with a Bonferroni correction;
  - TRUE uses \*/\*\*;
  - BARTScore uses †/‡;
  - QAFactEval marks 99% confidence intervals with \*.
- Beyond N-Grams defines its significance stars in the caption.
- Missing significance tests are a common weakness (Multi-FAct, SEAHORSE).
- ± std and seeds are rare: 0/7 metric papers report them. mFACT averages 3 seeds and STEMF puts 3-run ± standard error in the appendix. Doing it puts us above the genre norm.

**Draft:** no significance tests. Add a paired bootstrap (MorphEvalHeb vs the vanilla LLM judge, and vs mFACT), and put the 3-seed ± std in the appendix.

## #P018 Every baseline described in Setup must appear in the results table

**Evidence:** the reference papers list every compared metric in the main table, grouped by family (#P019).

**Draft:**
- §5 describes QAFactEval, BARTScore, Token F1, BLEU and ChrF, but Table 1 has only 7 rows. Add them or remove them from Setup.
- Line 221 says "five major paradigms" but lists six.

---

# F. Tables and figures

## #P019 Main table layout: metrics grouped by family, ours last, datasets as columns plus average

Put metrics as rows, grouped by family, with ours last. Put datasets as columns, followed by an average.

**Evidence**
- 6/7 metric papers use this layout.
  - SummaC Table 2 has a "Model Type" column (Baseline / Classifier / Parsing / QAG / NLI) and 6 dataset columns plus Overall.
  - QAFactEval groups Misc / Entailment / QA / Learned.
  - TRUE groups N-gram / Model-based / NLI / QG-QA.
  - AlignScore groups QA / Similarity / Regression / NLI / Misc / Ours.
- Beyond N-Grams Table 5 numbers its rows and refers to them in the text ("lines 1-7"), with group headers and a dashed line separating adapted from plain metrics.
- Hebrew papers list baselines first and "ours" last (AlephBERT: "Previous SOTA" → mBERT → HeBERT → ours).
- STEMF puts languages as columns, with the benchmark as a second header row and an Avg column.

**Draft:** Table 1 is a flat list of 7 rows. Proposed layout:

```
Family        | Metric          | Israel Hayom | Wiki | Weizmann | Knesset | mFACT-he | Avg | Avg-native
N-gram        | ROUGE-1 / BLEU / ChrF / Token-F1
Embedding     | BERTScore (heBERT / AlephBERT / NeoDictaBERT)
Generation    | BARTScore
NLI           | SummaC-ZS (mDeBERTa), mFACT
QA            | QAFactEval
LLM judge     | G-Eval, vanilla zero-shot
Ours          | MorphEvalHeb (binary), MorphEvalHeb (multi-label → binary)
```

## #P020 Bold for the best result, with the convention stated in the caption

**Evidence**
- All 5 multilingual papers bold the best result, and 5/6 Hebrew papers do.
- Underline for second best: Multi-FAct, STEMF.
- BARTScore uses bold for best unsupervised and underline for best overall.
- AlignScore uses dark and light green.
- NeoDictaBERT bolds the best in each column even when it is a baseline. That is the honest choice.
- Beyond N-Grams states "highest per column in bold" in the caption.
- No reference paper uses ↑/↓ arrows except mFACT. They aren't needed when every column is "higher is better".

## #P021 Self-contained captions

A caption should define every abbreviation, state the bold and significance conventions, say where the full results are, and add a one-sentence takeaway for results tables.

**Evidence**
- The multilingual papers have long captions that define abbreviations and the bold rule.
- mFACE and Multi-FAct captions say "full per-language results in Appendix X".
- The word-pieces paper starts each caption with a bold lead phrase.
- AlephBERT-ACL has a walk-through caption of about 90 words.
- Qualitative-table captions explain why a metric failed (4/7 metric papers).

**Draft:** the Table 1 caption is terse and has a typo ("sampels"). Rewrite it.

## #P022 Hebrew examples in script, transliteration, gloss and translation

Show Hebrew examples in script, then transliteration and gloss where morphology matters, then an English translation, with the error span highlighted on both sides.

**Evidence**
- AlephBERT-ACL shows script + transliteration ('lbit hlbn') + per-segment gloss + translation, and notes that the text "goes right to left".
- The word-pieces paper uses Leipzig-style glosses (keep.PRES.MASC.SG).
- mFACT Table 8 shows Chinese script with the English translation underneath and the hallucination highlighted in colour.
- SEAHORSE Fig 1 shows a Russian summary with a bracketed gloss and the note "wrong aspect".
- Counter-example: HeQ example tables have no gloss, and the translations are pushed to the appendix. Beyond N-Grams shows no Hebrew script at all. Most ACL reviewers don't read Hebrew.

**Draft:** the new Figure 2 already uses Hebrew plus an English gloss with blue/orange highlights. In the body text, gloss the binyan, smichut and gender examples in Leipzig style, for example *hitsiga* present.F.SG → *hitsig* present.M.SG.

## #P023 Efficiency: parameters and/or throughput

**Evidence**
- SummaC has a Doc./min column in its main table (about 430 docs/min).
- AlignScore's headline is a parameter count against LLM judges (355M vs ChatGPT/GPT-4).
- AlignScore appears with parameter counts in a main-table column. QASemConsistency puts model sizes in the row names.

**Draft:** add a column for parameters and cost or throughput. A small NeoDictaBERT encoder against an LLM judge is a strong practical argument. Look up and state the exact parameter count.

---

# G. Analysis

## #P024 Ablations that change one factor at a time

**Evidence**
- 5/7 metric papers run one-factor ablations.
  - SummaC varies the NLI model, NLI category and granularity; going from full document vs full summary to two sentences vs one sentence raises the score from 56.4 to 73.5.
  - QAFactEval swaps one component at a time.
  - AlignScore uses leave-one-task-out bar charts.
  - TRUE ablates model size.

**Draft:** candidate ablations:
- input granularity (Setup 1 / 2 / 3);
- with vs without the Hebrew-morphology families;
- with vs without the positive (paraphrase/back-translation) augmentation;
- NeoDictaBERT vs DictaBERT vs AlephBERT backbone.

## #P025 Breakdowns by error family

**Evidence**
- mFACE groups results by resource level, language family and NLI-data availability (Table 4), with per-language tables in the appendix.
- TRUE plots performance by length bin (all metrics degrade above 200 tokens).
- AlignScore reports per-dataset results plus averages.

**Draft:** report per-family F1 for the multi-label detector. Also report how often each baseline catches each family. That shows directly where ROUGE and BERTScore fail on morphological errors.

## #P026 Qualitative error analysis with counts

**Evidence**
- TRUE: of 80 cases where every metric was wrong, 35 were annotation errors.
- HeQ: a table of the largest F1-vs-TLNLS disagreements.
- FactCC §5.2: limitations drawn from inspecting misclassified examples.
- mFACE and mFACT answer likely objections inside the results (summary length, abstractiveness).

**Draft:** add 3–5 examples where MorphEvalHeb and the baselines disagree, plus counts of failure types.

---

# H. Back matter

## #P027 Unnumbered Limitations after the Conclusion, with concrete items

**Evidence**
- All four multilingual papers in the ACL template use an unnumbered `\section*{Limitations}` after the Conclusion, with 2–4 concrete items.
- Beyond N-Grams admits that an earlier annotation scheme reached α≈0.
- QASemConsistency gives concrete counter-examples and admits "N = 16 … cannot conclusively determine".
- AlignScore names its English-only scope.
- ACL requires a Limitations section, and it doesn't count toward the page limit.

**Draft:** Limitations comes *before* the Conclusion and is numbered. Move it. Suggested items:
- synthetic vs real error distribution;
- single domain (news);
- automatic labels in part of the benchmark;
- the 4,096-token limit for long articles.

## #P028 Ethics statement: data provenance, licensing and annotators

**Evidence**
- AlephBERT-ACL's Ethical Statement covers Twitter bias, privacy cleaning and benchmark licenses.
- mFACT and SEAHORSE include ethics statements covering annotator pay and misinformation.
- QAFactEval covers biases, misuse and environmental cost.
- Beyond N-Grams discloses annotator pay.
- Avoid what HeQ did: its Ethics section is a political statement rather than a data-ethics section.

**Draft:** none. Cover:
- licenses of the news sources (Israel Hayom, Wikipedia, Knesset, Bagatz);
- use of commercial LLM APIs for augmentation;
- annotator recruitment and pay, if there is human annotation;
- the dual-use risk of an error-injection tool.

## #P029 Appendix: prompts as listings, hyperparameters, full tables

**Evidence**
- Main body 8–9 pages, appendix 4–15 pages (mFACE: 8-page body, appendix A–G on pp. 14–28).
- STEMF gives all prompts as appendix listings, one per error type.
- QASemConsistency and HeQ quote their prompts verbatim.
- Multi-FAct pins model versions and experiment dates.
- NeoDictaBERT says where baseline numbers come from: "drawn from their respective publications… otherwise… finetuning ourselves".
- AlephBERT reports compute (8 V100 GPUs, 8 days).

**Draft:** the appendix should hold:
- the 24 transformation prompts (from `correctness_errors_prompts.py`);
- the G-Eval and vanilla-judge prompts;
- the hyperparameters from the `training_args.txt` files;
- the LLM versions with dates (gpt-5-mini, GPT-4.1, Gemini, Claude Sonnet 4);
- compute.

## #P030 Release links and license on page 1

**Evidence**
- 5/6 Hebrew papers put GitHub/HF release links in footnotes.
- 6/7 metric papers announce the release on page 1.
- An explicit license is rare: only NeoDictaBERT (CC BY-SA) states one. Stating one is a cheap win.

---

# I. Tone and language

## #P031 Confident headline claims with hedged explanations

**Evidence**
- In all three groups:
  - "we" with present tense for findings;
  - past or passive for procedures;
  - strong verbs for main results ("substantially outperforms");
  - hedged explanations ("likely due to", "we speculate", "we hypothesize", "we conjecture").
- First-of-kind claims use "to the best of our knowledge" (mFACE, mFACT, AlephBERT).
- SEAHORSE scopes its own claim: "our goal is not to train a state-of-the-art metric".
- Avoid promotional tone (AlephBERT preprint: "Stay tuned!"). The ACL version dropped it.

**Draft:** "first" claims should take "to the best of our knowledge". Explanations of why baselines fail ("not morphology-aware") should be hedged unless an ablation supports them.

## #P032 Frame the gap by type, not as "nobody did X"

**Evidence:** the reference papers frame prior-work gaps in a few recurring ways:
- **fragmented evaluation:** TRUE's "in silo", QAFactEval;
- **a negative result revisited with a diagnosis:** SummaC (granularity);
- **training data too narrow:** AlignScore;
- **insufficient granularity:** QASemConsistency;
- **missing data:** FactCC;
- **diversity gaps:** Beyond N-Grams "(i) lack of language diversity (ii) metrics diversity (iii) statistical evidence".

TRUE and SummaC end with explicit recommendations to the community.

**Draft:** the "three gaps" paragraph fits this pattern. Consider closing the Conclusion with 2–3 recommendations for evaluating MRL summarization, in the style of TRUE.

## #P033 Citation hygiene: cite what the paper actually says

**Evidence:** checking the draft against the papers found these misattributions:

| Line | Problem | Fix |
|---|---|---|
| 140 | "SummaC rely on multilingual entailment models" | SummaC uses English NLI (MNLI+VitaminC). mDeBERTa is our adaptation, as line 230 itself says. |
| 154 | AlignScore quote "tends to have difficulty evaluating summaries of long documents…" | The quote describes *earlier NLI-only metrics*. Cite AlignScore's real limitation: "only covers a single language, English". |
| 158 | mFACT "translating English factuality annotations" | The labels are *pseudo-labels from 4 automatic metrics*, not annotations. |
| 158 | Multi-FActScore "adapts atomic-fact verification to nine languages" | It translates outputs *into* English and verifies them against English Wikipedia. Its "English advantage" quote is contradicted by its own Table 3 (es/de/sw > en). |
| 161 | Beyond N-Grams "human judgments" | The ratings are for coherence and completeness, not factuality. Say so. |
| 167 | NeoDictaBERT "suitable for document-level factuality" | Our motivation, not the paper's finding. Its evaluation covers QA, parsing, NER and sentiment only. |
| 224 | Novel n-grams cited to mFACE | mFACE does not use them. Cite See et al. 2017 / Cao et al. 2022. |
| 224 | Token F1 cited to QAFactEval | QAFactEval does not define it as a metric. Cite TRUE. |
| 227 | `neodictabert` cited with `shmidman2023dictabert` | Use `BenDavid2025neodictabert`. |
| bib | `seker-etal-2021-alephbert`: wrong authors (Taub, Shmueli-Scheuer…), title and venue | Replace with ACL 2022 (2022.acl-long.4). |
| bib | QAFactEval, TRUE, AlignScore, mFACT cited as arXiv | Use the published venues: NAACL 2022, NAACL 2022, ACL 2023, EMNLP 2023 (check mFACT). |
| — | TRUE and SEAHORSE are in the bib but never cited | Cite TRUE in the protocol (#P013) and SEAHORSE in related work. |

Rule: before citing a quote or number, open the paper in `docs/parsed_papers/docling/` and check it.

---

## Pre-submission checklist

- [ ] Title has a descriptive subtitle (#P001)
- [ ] Abstract is 150–210 words, names the English-only gap and mentions the release (#P002, #P003)
- [ ] Page 1–2 teaser with a glossed Hebrew error (#P005)
- [ ] Contribution list with section pointers (#P004)
- [ ] Table of all 24 error families (#P008), plus real-error frequencies (#P009)
- [ ] Dataset statistics table with label origin (#P010)
- [ ] Splits grouped by `doc_id`, stated in the paper (#P011)
- [ ] Translation audit, or native vs translated subsets reported separately (#P012)
- [ ] ROC AUC and balanced accuracy primary, τ secondary; protocol paragraph written (#P013, #P014)
- [ ] Human-label protocol with α/κ, or "synthetic labels" stated honestly (#P015)
- [ ] Majority-class row and held-out average (#P016)
- [ ] Bootstrap significance; seeds ± std in the appendix (#P017)
- [ ] Every baseline from Setup appears in the table (#P018)
- [ ] Family-grouped table, conventions in captions, typo "sampels" fixed (#P019–#P021)
- [ ] Hebrew in script, gloss and translation (#P022)
- [ ] Parameters or throughput column (#P023)
- [ ] One-factor ablations and per-family breakdown (#P024, #P025)
- [ ] Qualitative disagreement examples with counts (#P026)
- [ ] `\section*{Limitations}` after the Conclusion; Ethics statement (#P027, #P028)
- [ ] Appendix with prompts, hyperparameters and model versions (#P029)
- [ ] Release links and license (#P030)
- [ ] All rows of the citation table fixed (#P033)
