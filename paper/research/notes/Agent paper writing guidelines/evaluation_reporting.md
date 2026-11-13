# Evaluation & Meta-Evaluation Reporting Guidance for a New-Metric Paper (MorphEvalHeb)

Context: MorphEvalHeb is a Hebrew summary factual-consistency metric. It uses fine-tuned binary and multi-label classifiers trained on synthetic error-injected data, and it is compared with ROUGE, BERTScore, SummaC, mFACT, G-Eval and an LLM judge by correlation with labels. Each source below lists verified bibliographic data, its concrete rules, the checklist items an agent can enforce, and a recommendation: ADOPT (enforce as a rule), ADAPT (enforce a modified version) or REFERENCE (cite only).

Sources published 2024–2026 are marked **[2024+]**.

---

## 1. Significance testing (general, and for metric-correlation differences)

### Takeaway
- **Model vs. model:** use paired bootstrap or permutation tests over test items (Berg-Kirkpatrick 2012; Dror 2018). For distributions over seeds, use almost stochastic dominance (Dror 2019).
- **Metric vs. metric (meta-evaluation):** follow Deutsch, Dror & Roth (TACL 2021).
  - Confidence intervals: Boot-Both bootstrap.
  - Testing whether one metric correlates better than another: Perm-Both permutation test.
  - Do not use the Williams test: it had near-zero power and assumes normality, which summarization data violates.
  - For a summary-level binary-label setup like ours, the analogue is resampling summaries (documents). Boot-Inputs and Perm-Inputs are the natural reductions when there is no "system" axis (see Inferences).

### Cited Findings
- **Dror, Baumer, Shlomov & Reichart (ACL 2018), "The Hitchhiker's Guide to Testing Statistical Significance in NLP"** (pp. 1383–1392, DOI 10.18653/v1/P18-1128).
  - Gives a practical protocol for choosing a significance test based on the task, the experimental setup and the evaluation measure.
  - Briefly surveys the relevant tests.
  - Surveys ACL/TACL 2017 empirical papers and finds that significance testing is "often ignored or misused".
  - [ACL Anthology P18-1128](https://aclanthology.org/P18-1128/)
- **Dror, Peled-Cohen, Shlomov & Reichart, *Statistical Significance Testing for Natural Language Processing*** (book, Synthesis Lectures on HLT, Morgan & Claypool / Springer, 2020; DOI 10.2200/S00994ED1V01Y202002HLT045).
  - Chapters cover hypothesis testing, significance tests, significance in NLP, deep significance, replicability analysis and open questions.
  - [Springer](https://link.springer.com/10.1007/978-3-031-02174-9)
- **Berg-Kirkpatrick, Burkett & Klein (EMNLP-CoNLL 2012), "An Empirical Investigation of Statistical Significance in NLP"** (pp. 995–1005).
  - Uses the paired bootstrap.
  - In-domain, significance levels appear well calibrated. Their predictive value drops sharply when the new data's domain diverges from the test set.
  - [ACL Anthology D12-1091](https://aclanthology.org/D12-1091/)
- **Dror, Shlomov & Reichart (ACL 2019), "Deep Dominance – How to Properly Compare Deep Neural Models"** (pp. 2773–2785).
  - Proposes an almost-stochastic-dominance test with a parameter ε ∈ [0,1] for comparing score distributions across runs (e.g., seeds) rather than single scores.
  - Implemented in the `deepsig` package.
  - [ACL Anthology P19-1266](https://aclanthology.org/P19-1266/); [deep-significance docs](https://deep-significance.readthedocs.io/)
- **Deutsch, Dror & Roth (TACL 2021, vol. 9, pp. 1132–1146), "A Statistical Analysis of Summarization Evaluation Metrics using Resampling Methods"** (DOI 10.1162/tacl_a_00417; arXiv 2104.00054). All of the following come from the full text ([ar5iv full text](https://ar5iv.arxiv.org/html/2104.00054)):
  - **Bootstrap variants:**
    - Boot-Systems resamples systems and treats inputs as fixed.
    - Boot-Inputs resamples inputs and treats systems as fixed.
    - Boot-Both resamples both.
  - **Permutation variants:** each swaps metric X and metric Y scores with probability 0.5.
    - Perm-Systems swaps all of a system's scores.
    - Perm-Inputs swaps all of an input's scores.
    - Perm-Both swaps each individual summary's score.
  - **Recommendations:**
    - "the Boot-Both method provides the most accurate CIs".
    - Perm-Both had the highest power and is recommended "for hypothesis testing the difference between summarization metrics".
    - The Williams test had power near zero in their simulation and "assumes the input variables are normally distributed", which was often violated.
    - Fisher-transformation CIs at summary level were too wide to be useful.
    - A nominal 95% Boot-Inputs CI was "actually only a 68% CI on the held-out data" in one setting.
  - **Definitions:**
    - System-level rSys is the correlation of per-system averages.
    - Summary-level rSum "calculates an average of the correlations between the scores per-input".
  - **Results:**
    - The CIs are "rather wide".
    - Most metrics are not significantly better than ROUGE. QAEval and BERTScore are, in some settings.
  - Code: github.com/CogComp/stat-analysis-experiments.
  - [ACL Anthology 2021.tacl-1.67](https://aclanthology.org/2021.tacl-1.67/)

### Inferences
- **Agent checklist items:**
  - (a) Every headline number comes with a 95% bootstrap CI (≥1,000 resamples; state the count).
  - (b) Every claim that "MorphEvalHeb beats X" is backed by a paired test. Use a paired permutation test for correlation/AUC differences, or a paired bootstrap. Report the p-value and the test name.
  - (c) Correct for multiple comparisons when testing against six or more baselines × several error types (e.g., Holm–Bonferroni). *This is a general statistical practice, not cited from the sources above. Dror et al.'s replicability analysis covers the multi-dataset case.*
  - (d) Never use the Williams test as the only test.
  - (e) Report mean ± std over at least 3–5 seeds for the fine-tuned classifiers. Use ASO/`deepsig` when comparing seed distributions.
  - (f) State whether reported numbers are max, mean or single-run. This mirrors ARR checklist item C3 (Section 4).
- **Reduction for our setup:** our meta-evaluation is mostly per-summary (each summary has a gold factual/non-factual label or error-type labels) with no multi-system axis. Resample summaries (paired across metrics) for both CIs and permutation tests. This is effectively Perm-Inputs/Boot-Inputs at the summary granularity.
  - If summaries come from several summarizers, consider resampling source documents and systems jointly, as Boot-Both does.
  - Deutsch et al. show that ignoring a varying axis under-covers (68% vs 95%).
- **Recommendations:**
  - ADOPT Deutsch 2021 (resampling choice) and Dror 2018 (test protocol).
  - ADOPT Dror 2019 (ASO for seeds).
  - REFERENCE Berg-Kirkpatrick 2012 (domain-shift caveat). This matters because MorphEvalHeb's synthetic train data differs in distribution from the natural test data.

### Gaps
- I did not verify the Hitchhiker's Guide decision table itself (e.g., which test it recommends for which metric type); only the abstract was read.
- No source found that gives a canonical test for differences in ROC AUC between metrics on the same items. DeLong's test is the standard statistical tool but was not researched here.

---

## 2. Which correlation and which level; binary factuality labels (ROC AUC / balanced accuracy)

### Takeaway
The factual-consistency literature reports detection-style metrics, not correlations, for binary labels:
- TRUE uses ROC AUC (threshold-free).
- SummaC and LLM-AggreFact use balanced accuracy (threshold tuned on a dev split).

For a binary + multi-label metric, report both ROC AUC and balanced accuracy, plus per-dataset/per-error-type breakdowns. Use correlations (Kendall/Spearman/Pearson) only where graded human scores exist. State the level explicitly (summary-level vs system-level) and use the Deutsch definitions.

### Cited Findings
- **Honovich et al., "TRUE: Re-evaluating Factual Consistency Evaluation"** (NAACL 2022; DialDoc 2022 version; arXiv 2204.04991).
  - Standardizes 11 factual-consistency datasets.
  - Reports ROC AUC for detecting inconsistent examples for each metric and dataset, over 12 metrics.
  - NLI-based and QG-QA approaches do best.
  - [Google Research page](https://research.google/pubs/true-re-evaluating-factual-consistency-evaluation/); [ar5iv](https://ar5iv.labs.arxiv.org/html/2204.04991)
- **Laban et al., "SummaC: Re-Visiting NLI-based Models for Inconsistency Detection in Summarization"** (TACL 2022).
  - Benchmark of 6 inconsistency-detection datasets.
  - Reports balanced accuracy: SummaCConv scores 74.4%, a 5% improvement.
  - Attributes poor NLI performance to a granularity mismatch between sentence-level NLI and document-level inconsistency detection.
  - [ACL Anthology 2022.tacl-1.10](https://preview.aclanthology.org/setup/2022.tacl-1.10)
- **Tang, Laban & Durrett, "MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents"** (arXiv 2404.10774; EMNLP 2024) **[2024+]**.
  - Introduces LLM-AggreFact: 10 datasets per the paper and the HF card. A third-party page says 11 datasets / 13,122 instances, so treat the official count as authoritative.
  - The primary metric is balanced accuracy for binary classification.
  - [arXiv 2404.10774](https://arxiv.org/html/2404.10774v2); [HF dataset card](https://huggingface.co/datasets/lytang/LLM-AggreFact/blob/main/README.md)
- **Tang et al., "LLMs as Factual Reasoners"** (arXiv 2305.14540).
  - No model reached 80% balanced accuracy on its binary task.
  - Estimates at least 6% of AggreFact samples are mislabeled. This is a label-noise caveat for any benchmark.
  - [arXiv 2305.14540](https://arxiv.org/pdf/2305.14540)
- **Definitions** (Deutsch et al. 2021):
  - System-level correlation is computed on per-system means.
  - Summary-level correlation is the average of per-input correlations.
  - [ar5iv 2104.00054](https://ar5iv.arxiv.org/html/2104.00054)

### Inferences
- **Agent checklist items:**
  - (a) For binary labels, report ROC AUC as the primary threshold-free number and balanced accuracy as the operating-point number.
    - Tune the threshold on a dev split only, never on test.
    - Report the threshold-selection procedure. This follows SummaC/LLM-AggreFact practice.
  - (b) For multi-label error types, report per-label F1 / AUC and macro-averages, plus support counts per label.
  - (c) Report class balance (percentage inconsistent) per test set. This is why balanced accuracy, rather than accuracy, is used.
  - (d) If any correlation is reported, name the coefficient and the level. Prefer Kendall τ (and state which variant, e.g., τ-b/τ-c) or Spearman for ordinal labels. Avoid Pearson on binary labels unless you call it point-biserial.
  - (e) Report per-dataset results, not only an average. TRUE and LLM-AggreFact both do this.
  - (f) Validate a sample of gold labels and report the estimated label noise (cf. the AggreFact ≥6% mislabel estimate).
- **Baseline comparability:** ROUGE and BERTScore are continuous similarity scores. For AUC that is fine. For balanced accuracy, they need the same dev-tuned threshold procedure as every other metric. An agent should enforce identical threshold protocols across all compared metrics.
- **Recommendations:**
  - ADOPT the TRUE (AUC) and SummaC/LLM-AggreFact (BAcc) reporting conventions.
  - ADAPT the Deutsch level definitions for the correlation analyses.

### Gaps
- No authoritative source located that specifically prescribes Kendall vs Spearman vs Pearson for factual-consistency meta-evaluation. Practice varies (e.g., SummEval-style papers report Kendall/Spearman).

---

## 3. Metric meta-evaluation pitfalls (system-level, ties, WMT methodology, survey)

### Takeaway
- **System-level correlations are fragile.** Compute them over the full test set and focus on close system pairs (Deutsch 2022).
- **Kendall τ variants mishandle ties.** Use pairwise accuracy with tie calibration (Deutsch, Foster & Freitag 2023), as WMT24 does at segment level. At system level, WMT24 uses Soft Pairwise Accuracy, with significance clusters from Perm-Both.
- **Gehrmann et al. (JAIR 2023)** is the umbrella survey of obscured evaluation practices, including a 66-paper audit.

### Cited Findings
- **Deutsch, Dror & Roth (NAACL 2022), "Re-Examining System-Level Correlations of Automatic Summarization Evaluation Metrics"** (pp. 6038–6052; arXiv 2204.10216).
  - Standard system-level correlation does not match how metrics are used in practice.
  - Proposes (1) scoring systems on the full test set rather than only the human-judged subset, and (2) restricting to system pairs with close automatic scores.
  - Under this setup, ROUGE's best-estimated correlation with humans is near zero in realistic scenarios (SummEval).
  - [ACL Anthology 2022.naacl-main.442](https://aclanthology.org/2022.naacl-main.442)
- **Deutsch, Foster & Freitag (EMNLP 2023), "Ties Matter: Meta-Evaluating Modern Metrics with Pairwise Accuracy and Tie Calibration"** (pp. 12914–12929; arXiv 2305.14324, originally titled "Modifying Kendall's Tau for Modern Metric Meta-Evaluation").
  - Existing Kendall variants handle ties inconsistently and can be gamed.
  - Proposes a pairwise accuracy that rewards correct tie prediction, plus tie calibration, which introduces ties into metric scores to allow fair comparison.
  - [ACL Anthology 2023.emnlp-main.798](https://aclanthology.org/2023.emnlp-main.798)
- **WMT24 Metrics Shared Task, "Are LLMs Breaking MT Metrics?"** (2024.wmt-1.2) **[2024+]**.
  - Segment level: pairwise accuracy with tie calibration.
  - System level: Soft Pairwise Accuracy (SPA) is the official meta-metric.
  - [Google Research page](https://research.google/pubs/are-llms-breaking-mt-metrics-results-of-the-wmt24-metrics-shared-task/); [WMT24 task page](https://www2.statmt.org/wmt24/metrics-task.html)
- **Thompson et al., "Improving Statistical Significance in Human Evaluation of Automatic Metrics via Soft Pairwise Accuracy"** (WMT 2024; arXiv 2409.09598) **[2024+]**.
  - SPA accounts for the statistical significance of both metric scores and human judgments.
  - Plain pairwise accuracy assigns few distinct values, so many metrics tie artificially. SPA is more stable when the number of systems or segments changes.
  - [ACL Anthology 2024.wmt-1.118](https://aclanthology.org/2024.wmt-1.118)
- **Rank/significance clusters:** follow-up work using the WMT24 meta-evaluation computes significance clusters "following Freitag et al. (2024), using the PERM-BOTH hypothesis test introduced by Deutsch et al. (2021)".
  - [arXiv 2506.19571 (HF summary)](https://huggingface.co/papers/2506.19571.md) **[2024+]**
- **Gehrmann, Clark & Sellam (JAIR 77:103–166, 2023), "Repairing the Cracked Foundation: A Survey of Obstacles in Evaluation Practices for Generated Text."**
  - Surveys problems in human evaluation, automatic metrics and datasets.
  - Proposes concrete steps and audits 66 recent NLG papers for adherence.
  - [researchr record](https://researchr.org/publication/GehrmannCS23); arXiv 2202.06935 (preprint, Feb 2022)

### Inferences
- **Agent checklist items:**
  - (a) If ranking summarization systems, do not report only Pearson over few systems. Report pairwise accuracy (with tie calibration if the metric produces ties) and the number of systems.
  - (b) MorphEvalHeb binary outputs produce massive ties. Any Kendall-type statistic must use a tie-aware variant, or pairwise accuracy with tie calibration, and must name the variant.
  - (c) Group metrics into significance clusters (Perm-Both, or a paired permutation test at summary level) rather than presenting a strict ranking table.
  - (d) Include a "where the metric fails" analysis: per error type, per source domain, and on synthetic vs natural errors. This is in the spirit of Gehrmann et al.
- **Recommendations:**
  - ADOPT Ties Matter for any rank correlation over binary/discrete scores.
  - ADAPT WMT24 practice (significance clusters).
  - REFERENCE Gehrmann et al. as the umbrella justification.

### Gaps
- I did not extract the specific numbered recommendations from Gehrmann et al. (the full text was not fetched). The report writer should not attribute specific rules to it beyond the abstract-level claims above.

---

## 4. Human evaluation reporting (and inter-annotator agreement)

### Takeaway
- If any human labels are collected (e.g., native Hebrew annotators validating gold labels or synthetic errors), document them with HEDS.
- Use standardized quality-criterion names. Howcroft et al. found over 200 different terms for evaluated quality aspects.
- Follow van der Lee et al.'s planning/execution/release guidelines.

### Cited Findings
- **Howcroft et al. (INLG 2020), "Twenty Years of Confusion in Human Evaluation: NLG Needs Evaluation Sheets and Standardised Definitions"** (pp. 169–182, DOI 10.18653/v1/2020.inlg-1.23).
  - Annotated 165 INLG/ENLG papers (2000–2019).
  - Found very confused reporting and wide variation in methods, with over 200 different terms for quality aspects.
  - Calls for standard methods and terminology.
  - [ACL Anthology 2020.inlg-1.23](https://aclanthology.org/2020.inlg-1.23); [Tilburg record](https://research.tilburguniversity.edu/en/publications/twenty-years-of-confusion-in-human-evaluation-nlg-needs-evaluatio/)
- **Shimorina & Belz, "The Human Evaluation Datasheet" (HEDS).**
  - Versions:
    - v1.0: arXiv 2103.09710 (2021).
    - v2.0: HumEval 2022, pp. 54–75, DOI 10.18653/v1/2022.humeval-1.6.
    - HEDS 3.0 later released by Belz & Thomson.
  - A template that uses multiple-choice questions where possible, with an online form and Markdown/LaTeX converters.
  - Designed for comparability, meta-evaluation and reproducibility. It builds on Data Statements, Model Cards and Datasheets.
  - [arXiv 2103.09710](https://arxiv.org/pdf/2103.09710.pdf); [ACL Anthology 2022.humeval-1.6](https://aclanthology.org/2022.humeval-1.6)
- **van der Lee, Gatt, van Miltenburg & Krahmer (Computer Speech & Language 67, 101151, 2021), "Human evaluation of automatically generated text: Current trends and best practice guidelines."**
  - Best practices organized by stage: planning, then execution and release.
  - Earlier version: van der Lee et al., INLG 2019, "Best practices for the human evaluation of automatically generated text" (W19-8643).
  - [Tilburg record](https://research.tilburguniversity.edu/en/publications/human-evaluation-of-automatically-generated-text-current-trends-a/); [ACL Anthology W19-8643](https://aclanthology.org/W19-8643/)

### Inferences
- **Agent checklist items** (if human annotation exists):
  - (a) Fill in a HEDS sheet as an appendix.
  - (b) Report the number of annotators, their expertise (native Hebrew speakers? linguists?), recruitment and pay, and the items per annotator.
  - (c) Report inter-annotator agreement with the coefficient named (e.g., Krippendorff's α or Cohen's/Fleiss' κ), computed on a stated overlap subset.
  - (d) Release the guidelines and interface screenshots.
  - (e) Define each error type with its standard name and give an example. Howcroft et al.'s finding of over 200 terms means MorphEvalHeb's error taxonomy should be mapped to existing taxonomies (e.g., FRANK-style entity/predicate/circumstance errors), not renamed ad hoc.
- **Synthetic-data validation:** an error-injected dataset should include a human audit of a random sample with an agreement statistic. This addresses whether the injected errors are actually errors and are natural. It is the human-eval component reviewers will look for.
- **Recommendations:**
  - ADOPT HEDS (an appendix).
  - ADOPT van der Lee's stage checklist.
  - REFERENCE Howcroft et al. as the motivation.

### Gaps
- I did not fetch a primary source for the IAA coefficient choice. Artstein & Poesio 2008 (Computational Linguistics, "Inter-Coder Agreement for Computational Linguistics") is the standard reference but is unverified here.
- I did not extract van der Lee's itemized guidelines (e.g., minimum annotator counts). Only the structure is verified.

---

## 5. Reproducibility and documentation (seeds, variance, budget, data/model docs)

### Takeaway
- Report hyperparameter search, budget and expected validation performance (Dodge 2019).
- Report descriptive statistics, saying explicitly whether numbers are max, mean or single run (ARR Responsible NLP checklist C2/C3, which is mandatory at ACL venues).
- Document data with Data Statements or Datasheets and the model with a Model Card.
- The NeurIPS ML Reproducibility Checklist (Pineau et al.) is the ML-wide analogue.

### Cited Findings
- **Dodge, Gururangan, Card, Schwartz & Smith (EMNLP-IJCNLP 2019), "Show Your Work: Improved Reporting of Experimental Results"** (pp. 2185–2194).
  - Test-set scores alone are insufficient to decide which model is best.
  - Recommends reporting expected validation performance of the best-found model as a function of computation budget. This is computed from the hyperparameter-search distribution, with no extra compute.
  - Tool: allenai/allentune.
  - Follow-up: Dodge et al., EMNLP 2021, "Expected Validation Performance and Estimation of a Random Variable's Maximum".
  - [ACL Anthology D19-1224](https://aclanthology.org/D19-1224/)
- **ARR Responsible NLP Research checklist.**
  - Items:
    - C1: model size, compute budget, infrastructure.
    - C2: "experimental setup, including hyperparameter search and best-found hyperparameter values".
    - C3: "descriptive statistics about your results (e.g., error bars …), and is it transparent whether you are reporting the max, mean, etc. or just a single run?"
    - B6: dataset statistics and splits.
  - Each "Yes" must cite a section. Systematic failure to do so leads to desk rejection.
  - Wording varies by version, so check the current ARR page.
  - [ARR Responsible NLP page](https://aclrollingreview.org/responsibleNLPresearch); [NAACL 2022 blog](https://2022.naacl.org/blog/responsible-nlp-research-checklist/); [example ACL 2026 filled checklist](https://aclanthology.org/attachments/2026.acl-long.1939.checklist.pdf) **[2024+]**
- **"Is the ACL Responsible NLP Checklist a Box-Ticking Exercise? A Large-Scale Analysis of EMNLP 2025"** (arXiv 2608.09280) **[2024+]**.
  - Analyzes how the checklist is filled in practice.
  - Seen only as a search snippet; contents not verified.
  - [arXiv 2608.09280](https://arxiv.org/pdf/2608.09280)
- **Pineau et al. (JMLR 22(164):1–20, 2021), "Improving Reproducibility in Machine Learning Research (A Report from the NeurIPS 2019 Reproducibility Program)."**
  - The program had three parts: a code submission policy, a community reproducibility challenge, and the ML Reproducibility Checklist in submissions.
  - [JMLR v22/20-303](https://jmlr.org/papers/v22/20-303.html); [arXiv 2003.12206](https://arxiv.org/pdf/2003.12206)
- **Bender & Friedman (TACL 6:587–604, 2018), "Data Statements for Natural Language Processing: Toward Mitigating System Bias and Enabling Better Science."**
  - Documents data from a linguistic perspective: language variety, speaker and annotator demographics, and so on.
  - [Partnership on AI reference list](https://partnershiponai.org/about-ml-process-guide/)
- **Gebru et al., "Datasheets for Datasets"** (arXiv 1803.09010).
  - Proposes a standardized documentation process for ML datasets.
  - [arXiv 1803.09010](https://arxiv.org/pdf/1803.09010.pdf). The CACM venue was not confirmed in this search.
- **Mitchell et al., "Model Cards for Model Reporting"** (FAT* 2019, Atlanta).
  - Documents intended uses and evaluation disaggregated across relevant attributes.
  - [arXiv 1810.03993](https://arxiv.org/pdf/1810.03993v1)

### Inferences
- **Agent checklist items:**
  - (a) A hyperparameter table listing search ranges, the best values, the number of trials and GPU-hours. Optionally, an expected-max-validation curve (Dodge) for MorphEvalHeb vs fine-tuned baselines.
  - (b) Seeds: at least 3 (preferably 5). Report mean ± std and the seed list.
  - (c) Exact train/dev/test sizes and label distribution for both the synthetic training data and the natural evaluation sets. State explicitly that no test documents leak into synthetic generation.
  - (d) A Data Statement for the Hebrew corpus. Cover the variety (Modern Hebrew), source genres, morphology-related preprocessing and the annotator profile. Also document the error-injection procedure (rules, LLM, prompts).
  - (e) A Model Card for the released classifiers, with per-error-type performance disaggregated.
  - (f) Release code, data and checkpoints with licenses (ARR checklist section A/B).
- **Recommendations:**
  - ADOPT ARR C1–C3 / B6 (mandatory).
  - ADOPT Dodge (at minimum report the search budget).
  - ADOPT Data Statement + Model Card.
  - REFERENCE Pineau and Gebru.

### Gaps
- I did not verify the exact current ARR checklist item wording from the official page itself; it was sourced via search snippets.
- I could not verify Ulmer et al. 2022, "Experimental Standards for Deep Learning in NLP Research" (the search returned nothing).

---

## 6. Synthetic data and LLM-as-judge evaluation reporting

### Takeaway
LLM judges must be validated against human labels per task before being trusted. JUDGE-BENCH found large variance across tasks, annotator expertise and text source. Their correlation is sensitive to prompt and process details: explanation-before-score helps, and number-only output is suboptimal (Chiang & Lee 2023). A paper using G-Eval/LLM-judge baselines should therefore report:
- the exact prompts
- the model name, version and date
- the decoding parameters
- the output parsing method

No single authoritative reporting checklist for LLM judges was found.

### Cited Findings
- **Bavaresco et al. (ACL 2025, pp. 238–255; arXiv 2406.18403), "LLMs instead of Human Judges? A Large Scale Empirical Study across 20 NLP Evaluation Tasks"** **[2024+]**.
  - JUDGE-BENCH covers 20 human-annotated datasets and 11 LLMs.
  - Reliability varies widely by evaluated property, annotator expertise, and human- vs model-generated text.
  - Conclusion: validate LLMs against human judgments before using them as evaluators.
  - Code: github.com/dmg-illc/JUDGE-BENCH.
  - [arXiv 2406.18403](https://arxiv.org/abs/2406.18403v2); [MPI record](https://www.mpi.nl/publications/item3650876/llms-instead-human-judges-large-scale-empirical-study-across-20-nlp)
- **Chiang & Lee (ACL 2023), "Can Large Language Models Be an Alternative to Human Evaluations?"** (2023.acl-long.870).
  - LLM ratings track expert ratings on story generation and adversarial attacks.
  - They are stable across instruction formatting and sampling.
  - [arXiv 2305.01937](https://arxiv.org/abs/2305.01937v1)
- **Chiang & Lee (Findings of EMNLP 2023), "A Closer Look into Using Large Language Models for Automatic Evaluation"** (2023.findings-emnlp.599).
  - G-Eval's auto chain-of-thought does not always help.
  - Forcing number-only output is suboptimal.
  - Asking the LLM to explain its rating consistently improves correlation with humans.
  - [ACL Anthology 2023.findings-emnlp.599](https://aclanthology.org/2023.findings-emnlp.599)
- **Surveys** **[2024+]**:
  - Gu et al., "A Survey on LLM-as-a-Judge" (arXiv 2411.15594) proposes reliability-evaluation methodology and a benchmark.
  - "From Generation to Judgment" (arXiv 2411.16594).
  - "LLMs-as-Judges" (arXiv 2412.05579).
  - All note that standardization is lacking.
  - [arXiv 2411.15594](https://arxiv.org/pdf/2411.15594v6); [arXiv 2411.16594](https://arxiv.org/html/2411.16594v7); [arXiv 2412.05579](https://arxiv.org/html/2412.05579v2)

### Inferences
- **Agent checklist items for LLM baselines (G-Eval, LLM-judge):**
  - (a) Put the full prompt verbatim in an appendix, including the Hebrew text and whether the instructions are in Hebrew or English.
  - (b) Give the exact model ID/version and the access date (API models drift).
  - (c) Give the decoding parameters (temperature, top-p, max tokens, number of samples) and the aggregation method (G-Eval probability-weighted vs argmax).
  - (d) Describe output parsing and failure/refusal handling, and report the parse-failure rate.
  - (e) Report the API cost and run count, and the variance across repeated runs if temperature > 0.
  - (f) Validate the judge on the human-labeled subset, with the same metrics as everything else (AUC/BAcc + CIs).
- **Agent checklist items for synthetic error-injected training data:**
  - (a) Describe the injection procedure precisely: rule- or LLM-based, the error types, and the per-type counts.
  - (b) Human-audit a random sample and report the agreement.
  - (c) Evaluate on natural (non-synthetic) errors to show transfer. This connects to Berg-Kirkpatrick's finding that significance degrades under domain shift.
  - (d) If an LLM generated the injected errors, check for and report self-preference / leakage risk when the same model family serves as a judge baseline.
- **Recommendations:**
  - ADOPT Bavaresco (validate the judge per task).
  - ADAPT Chiang & Lee (prompt/process reporting).
  - REFERENCE the surveys.

### Gaps
- I found no authoritative, published reporting checklist dedicated to LLM-as-judge (prompt/version/temperature). The items above are inferred from the cited empirical findings.
- I did not search for guidance specific to synthetic-data evaluation reporting (e.g., papers on evaluating with synthetic factual errors such as FactCC/DAE-style critiques).
- I found no Hebrew- or low-resource-specific meta-evaluation reporting guidance; it was not searched in depth.
