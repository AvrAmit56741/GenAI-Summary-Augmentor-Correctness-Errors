# Verification of generator freshness in MT and code-generation metric/benchmark papers

Verified 2026-11-18 against the first-pass survey `paper/research/model_freshness_mt_code.md`.

**Method.** Each claim was checked against the full text of the primary PDF, extracted locally (ACL Anthology or arXiv) and searched for the exact passage. Two General-task findings papers that the first pass had not read were added: WMT23 General (2023.wmt-1.1) and WMT24 General (2024.wmt-1.1). Venues were checked on ACL Anthology pages. "Age gap" = paper year − release year of the newest system whose outputs are evaluated. LLMs used as metrics or judges are listed separately. Anything not confirmed against a primary source is marked UNCONFIRMED.

## Q1. WMT 2023–2025: which LLM translators organizers added; Hebrew in WMT23; metric conclusions

### Takeaway
All WMT claims in the first pass hold, with two corrections.
1. WMT23 General did **not** publish official human rankings for English↔Hebrew in its proceedings. The "GPT-4 best on he→en" evidence comes only from the WMT23 *Metrics* MQM table. There GPT4-5shot (1.33) is the best MT system but trails the human reference (1.17) and is only narrowly ahead of ONLINE-A (1.38).
2. In WMT23 the *General-task* organizers added GPT4-5shot. The *Metrics* organizers separately added the NLLB-200 outputs.

The WMT24 Metrics answer to "Are LLMs breaking MT metrics?" is "no" for fine-tuned neural metrics. WMT25 finds LLM judges top the system-level rankings but underperform at the segment level.

### Cited Findings

**WMT23**
- GPT-4 was added by the organizers, not submitted: "one dominant commercial LLM (GPT4) was included via our own efforts". It was produced "using 5-shot prompting with fixed random translation examples, using the exact prompt by Hendy et al. (2023)". Title: "LLMs Are Here but Not Quite There Yet". — [WMT23 General findings, Kocmi et al. 2023](https://aclanthology.org/2023.wmt-1.1.pdf)
- English↔Hebrew was new in 2023: "we introduce a new language pair to WMT, namely: English↔Hebrew". Hebrew news came from IAHLT (Israel Hayom 2022). Hebrew user content came from news-article comments. — [WMT23 General](https://aclanthology.org/2023.wmt-1.1.pdf)
- No official en↔he human ranking was published: "we decided not to publish official rankings based on manual evaluation for English↔{Hebrew, Russian, Ukrainian} until the conference". The reason was insufficient high-quality Toloka assessments. — [WMT23 General §5.4](https://aclanthology.org/2023.wmt-1.1.pdf)
- Hebrew reference caveat: "For Hebrew↔English, our references are likely to be post-edited MT, while for Chinese→English, the reference translation is worse than the majority of automatic translations." — [WMT23 General](https://aclanthology.org/2023.wmt-1.1.pdf)
- General finding on GPT-4: "GPT4 was in the first significance cluster for all systems translating towards English, but fell in the second significance cluster (rank 3-5) for English→Czech". Test suites found GPT-4 outputs "not always faithful to the source". — [WMT23 General](https://aclanthology.org/2023.wmt-1.1.pdf)
- WMT23 Metrics MQM, Hebrew→English (lower is better), from the "all" column:
  - refA 1.17
  - **GPT4-5shot 1.33**
  - ONLINE-A 1.38
  - ONLINE-B 1.55
  - GTCOM_DLUT 1.89
  - UvA-LTL 1.92
  - … NLLB_Greedy 3.79
  - Lan-BridgeMT 3.79
  - Other pairs: in en→de, refA 2.96 beats GPT4-5shot 3.72. In zh→en, Lan-BridgeMT 2.10 beats GPT4-5shot 2.31.
  - The MQM pairs were en-de (paragraph level), he-en and zh-en.
  - The Metrics organizers separately added NLLB-200 3.3B (greedy and MBR-BLEU): "We added translations which we expected to differ… by using a massively multilingual model".
  - Source: [WMT23 Metrics findings, Freitag et al. 2023](https://aclanthology.org/2023.wmt-1.51.pdf)

**WMT24**
- LLMs added by the organizers: "we added contrastive translations by 8 LLMs, which included closed commercial products (such as GPT-4) and open models (such as Llama3). This resulted in 95 more submissions". The list: "we collect translations from Aya23, Claude-3.5-Sonnet, Command R+, GPT-4, Gemini-1.5-Pro, Llama3-70B, Mistral-Large, Nvidia-NeMo and Phi-3-Medium". Table 4 prices GPT-4 at $240.4 for the full test set. The paper names no exact GPT-4 snapshot, so the snapshot is UNCONFIRMED. — [WMT24 General findings, Kocmi et al. 2024](https://aclanthology.org/2024.wmt-1.1.pdf)
- Metrics task focus: "a major focus on LLM-based translations that were generated as part of the WMT24 General MT Shared Task". Findings:
  - "Fine-tuned neural metrics continue to be strong in performance and are effective quality estimators, even for LLM-based translations".
  - "Overall, we did not find any issues for neural fine-tuned metrics when evaluating LLM-based translations."
  - Winners were MetricX-24-Hybrid and XCOMET.
  - Source: [WMT24 Metrics findings, Freitag et al. 2024](https://aclanthology.org/2024.wmt-1.2.pdf)
- WMT24 Metrics en→de MQM:
  - Dubformer 1.58 and GPT-4 1.58 tie for first.
  - Unbabel-Tower70B 1.65.
  - refB 1.84 and refA 2.12. These are beaten by several LLMs (GPT-4, Tower70B, Mistral-Large 1.93, CommandR-plus 2.01).
  - Gemini-1.5-Pro 2.20, Claude-3.5 2.28, Aya23 3.09, Llama3-70B 3.62, MSLC 13.46.
  - Source: [WMT24 Metrics](https://aclanthology.org/2024.wmt-1.2.pdf)
- DFKI challenge set: "Metrics are more accurate in detecting linguistic errors among LLM translations than in translations based on the encoder-decoder NMT architecture." — [WMT24 Metrics](https://aclanthology.org/2024.wmt-1.2.pdf)

**WMT25**
- System list: Table 1 lists 24 systems collected by the organizers.
  - Constrained: AyaExpanse-8B, CommandR7B, EuroLLM-9B, Gemma-3-12B, Llama-3.1-8B, Mistral-7B, NLLB-200 3.3B, Qwen2.5-7B, TowerPlus-9B.
  - Unconstrained: AyaExpanse-32B, Claude-4, CommandA 111B, DeepSeek-V3 671B (37B active), EuroLLM-22B, Gemma-3-27B, Gemini-2.5-Pro, GPT-4.1, Llama-4-Maverick, Mistral-Medium, ONLINE-B/G/W, Qwen3-235B, TowerPlus-72B.
  - That makes 20 LLMs, plus NLLB and 3 online engines. This confirms "~20 LLMs".
  - Settings: "for Qwen3-235B reasoning capabilities were disabled", and "for Gemini-2.5-Pro, no reasoning budget was set".
  - Source: [WMT25 preliminary ranking, Kocmi et al., arXiv 2508.14909](https://arxiv.org/pdf/2508.14909)
- The preliminary ranking used GEMBA-ESA with judges "GPT-4.1 … and Command A". So judges belong to the same families as the translators. — [arXiv 2508.14909](https://arxiv.org/pdf/2508.14909)
- WMT25 evaluation task goals and data:
  - Goal: detect "errors generated by increasingly accurate LLM-based translators" and handle "output verbosity and incorrect output language".
  - Data were "intentionally chosen to be challenging for MT".
  - Output selection: "around 20 different MT systems for each language pair", "giving slight prioritization to constrained (small, open-weight) systems".
  - Source: [WMT25 Automated Translation Evaluation findings, Lavie et al. 2025](https://aclanthology.org/2025.wmt-1.24.pdf)
- WMT25 main finding: "Strong LLM-based auto-raters now top the rankings at the system level… At the segment level, LLM-based auto-raters still underperform; this year, unlike in recent years, reference-based baseline metrics (YISI-1, CHRF, and BERTSCORE) fill out the top three rank clusters, outranking recently strong trained metrics." The abstract says: "reference-based baseline metrics outperform LLMs at the segment level." — [WMT25 eval findings](https://aclanthology.org/2025.wmt-1.24.pdf)
- WMT25 Table 5, LLM-as-judge results on pairs without references (cluster, score):
  - Gemini2.5-Pro 1 / 0.650
  - GPT-4.1 3 / 0.616
  - Claude-4 4 / 0.570
  - DeepSeek-V3 5 / 0.561
  - Llama-4-Maverick 12 / 0.462
  - The judge LLMs are the same models as the WMT25 translators.
  - Source: [WMT25 eval findings](https://aclanthology.org/2025.wmt-1.24.pdf)

### Inferences
- The age gap for WMT shared tasks is 0 from 2023 on, and the organizers themselves add the frontier models (GPT-4 in 2023; Claude-3.5, Gemini-1.5-Pro and others in 2024; Gemini-2.5-Pro, GPT-4.1 and Claude-4 in 2025). This institutional practice is the main defense against stale generators.
- For a Hebrew precedent, cite the WMT23 Metrics MQM he→en table rather than the General-task ranking, which was not published in the proceedings. Note that GPT-4 is best among MT systems but not better than the reference, and that its lead over ONLINE-A is small. Significance was not checked.

### Gaps
- The exact GPT-4 snapshot used in WMT23 and WMT24 is UNCONFIRMED; neither paper names it.
- Whether a post-conference en↔he human ranking for WMT23 was later published (for example on statmt.org) is UNCONFIRMED.

## Q2. Classic MT metric papers: BLEURT quality drift, COMET top systems, Agrawal 2024, Mathur 2020

### Takeaway
The first-pass quotes for BLEURT, COMET and Agrawal are exact. COMET's §5.4 statement is confirmed. The general claim that "all metrics drop" with fewer top systems comes from figures only and remains figure-based. Mathur 2020's top-N finding is confirmed at the abstract level only.

### Cited Findings
- **BLEURT:**
  - Motivation: "NLG systems tend to get better over time, and therefore a model trained on ratings data from 2015 may fail to distinguish top performing systems in 2019".
  - "it is easier to discriminate between 'good' and 'bad' systems than to rank 'good' systems".
  - "Training skew has a disastrous effect on BLEURT without pre-training: it is below BERTscore for α = 1.0, and it falls under sentBLEU for α ≥ 1.5."
  - Source: [Sellam et al., ACL 2020](https://aclanthology.org/2020.acl-main.704.pdf)
- **COMET §5.4 "Robustness to High-Quality MT":**
  - Uses WMT19 DARR restricted to "the top 10, 8, 6 and 4 MT systems".
  - "For language pairs where English is the target, our three models are either better or competitive with all others; where English is the source we note that in general our metrics exceed the performance of others".
  - Source: [Rei et al., EMNLP 2020](https://aclanthology.org/2020.emnlp-main.213.pdf)
- **Agrawal, Farinhas, Rei, Martins 2024:**
  - Metrics "are insensitive to nuanced differences in translation quality. This effect is most pronounced when the quality is high and the variance among alternatives is low"; they "severely overestimate (for non-HQ translations) or underestimate (for HQ translations)".
  - GEMBA-MQM "assigns high scores to erroneous GPT-4 translations, suggesting a preferential bias towards the LLM's own outputs"; "GEMBA-MQM frequently assigns zero MQM scores to GPT-4 translations, even when humans identify errors in them."
  - Source: [arXiv 2405.18348](https://arxiv.org/abs/2405.18348)
- **Mathur, Baldwin, Cohn ("Tangled up in BLEU", ACL 2020):** metric–human correlation at the system level drops markedly when only the few best systems are considered, and outlier systems inflate correlations. Confirmed at abstract and search-snippet level only; the exact top-N numbers are UNCONFIRMED. — [arXiv 2006.06264](https://arxiv.org/abs/2006.06264)

### Inferences
- Pre-LLM evidence and 2024 evidence agree. Metric discrimination degrades as the systems being compared become stronger and closer together. Strong generators are therefore exactly the regime in which a metric must be validated.

### Gaps
- Exact per-metric drops in the COMET top-N figure and in Mathur 2020 were not extracted, because the numbers are in figures.

## Q3. Venue and status corrections for rows marked (mem) or (unverified)

### Takeaway
Most remembered venues are correct. CodeJudgeBench now has a confirmed venue (ACL 2026). Two factual corrections apply to code rows: CodeJudge-Eval's judges and CodeJudge's generators.

### Cited Findings
- xCOMET: TACL 2024. — [ACL Anthology 2024.tacl-1.54](https://aclanthology.org/2024.tacl-1.54.pdf)
- CodeBERTScore: EMNLP 2023, pp. 13921–13937. — [2023.emnlp-main.859](https://aclanthology.org/2023.emnlp-main.859)
- ICE-Score (Zhuo): Findings of EACL 2024, pp. 2232–2242. — [2024.findings-eacl.148](https://aclanthology.org/2024.findings-eacl.148)
- CodeJudge-Eval: COLING 2025 (2025.coling-main.7). — [ACL Anthology](https://preview.aclanthology.org/setup/2025.coling-main.7)
- **CodeJudgeBench: ACL 2026 Long Papers**, pp. 19416–19448 (Jiang, Chen, Cao, Lee, Tan). This corrects the first pass's "venue unverified". — [2026.acl-long.888](https://aclanthology.org/2026.acl-long.888/)
- CodeUltraFeedback: only arXiv versions found (2403.09032 v1–v3); the venue is UNCONFIRMED. — [arXiv](https://arxiv.org/html/2403.09032v3)
- Chaplynskyi, Kulynych, Shvedova, Ivashkevych, "Semantic Fidelity Versus Literary Quality: A Construct Validity Study of Neural Machine Translation Metrics", UNLP 2026, pp. 67–79.
  - Generators: GPT-5.2, DeepL and Lapa (Ukrainian), compared against human literary translations.
  - Finding: "Across seven neural metrics… all three AI translations rank at the top."
  - Verified at abstract and search level; full results UNCONFIRMED.
  - Source: [2026.unlp-1.8](https://aclanthology.org/2026.unlp-1.8/)
- **CodeJudge-Eval correction (generators):** the 16 generators are confirmed as "GPT-4, GPT-3.5, Claude-3-Opus, Gemini-1.0-pro, Ernie-3.5, and Qwen-turbo" plus open models (Starchat, Gemma, Mistral-Instruct, Llama2-chat, ChatGLM, CodeLlama-Instruct, WizardCoder, DeepseekCoder-Instruct and others). — [arXiv 2408.10718](https://arxiv.org/abs/2408.10718)
- **CodeJudge-Eval correction (judges):** the 12 evaluated *judges* include **GPT-4o and Claude-3.5-Sonnet**: "proprietary LLMs such as GPT-4o and Claude-3.5-Sonnet outperform open-source models, their macro F1 scores peak at a modest 50". Judges are therefore *newer* than the newest generator (Claude-3-Opus). The first pass's "judges are the evaluated LLMs" is imprecise. — [arXiv 2408.10718](https://arxiv.org/abs/2408.10718)

### Inferences
- CodeJudge-Eval has the same judge-newer-than-generator pattern as ICE-Score.

### Gaps
- The CodeUltraFeedback venue is UNCONFIRMED.
- CometKiwi-22 training-data details remain from memory; they were not re-checked.

## Q4. Code: CodeJudgeBench exact numbers; CodeJudge, ICE-Score and CodeBERTScore data sources; Wang et al. ISSTA 2025

### Takeaway
CodeJudgeBench's frontier-generator claim is confirmed, with numbers. The same judges score 93–98% on JudgeBench's code split and 66–82% on CodeJudgeBench CodeGen.

Judges are also much worse on code generated by Gemini-2.5-Pro than on code generated by Claude-3.7-Sonnet. For example, the Gemini-2.5-Pro judge scores 72.85 on Gemini-generated pairs and 90.46 on Claude-generated pairs; the average over all judges is 62.11 versus 79.49.

ICE-Score and CodeBERTScore validate on Codex (code-davinci-002) outputs and the 2018–2021 CoNaLa systems. CodeJudge's HumanEval-X and APPS generator is not named. Wang et al. use only open models of 34B or less, released up to Sep 2024, with GPT-4o and DeepSeek-V2.5 as judges.

### Cited Findings
- **CodeJudgeBench motivation:**
  - "We use state-of-the-art LLMs like Gemini-2.5-Pro and Claude-3.7-Sonnet to generate high-quality and challenging candidate responses. The errors produced by these advanced LLMs are often subtle… earlier benchmarks often rely on weaker models like GPT-4o, which produce less challenging samples."
  - "all LLM-as-a-Judge models exhibit substantially lower accuracy on CodeJudgeBench compared to prior benchmarks. Frontier models such as Gemini-2.5-Pro achieve near-perfect accuracy on JudgeBench".
  - Source: [Jiang et al., arXiv 2507.10535 / ACL 2026](https://arxiv.org/pdf/2507.10535)
- **CodeJudgeBench Figure 1:** the judges are QwQ, Gemini-2.5-Flash, Claude-3.7-Sonnet, Claude-4-Opus, Claude-4-Sonnet and Gemini-2.5-Pro. The bar labels give JudgeBench-code 93–98%, RM-Bench-code 78–87% and CodeJudgeBench CodeGen 66–82%. The per-model mapping of individual bars is UNCONFIRMED (text extraction from the figure). — [arXiv 2507.10535](https://arxiv.org/pdf/2507.10535)
- **CodeJudgeBench Table 3** (CodeGen overall / 3-task average):
  - Gemini-2.5-Pro 81.98 / 82.12
  - Claude-4-Sonnet 79.15 / 79.93
  - Claude-4-Opus 75.89 / 76.36
  - QwQ-32B 76.89 / 76.40
  - Claude-3.7-Sonnet 66.33 / 67.64
  - Claude-3.5-Sonnet-v2 58.32 / 59.12
  - AceCodeRM-7B 53.97 / 52.92
  - Source: [arXiv 2507.10535](https://arxiv.org/pdf/2507.10535)
- **CodeJudgeBench data and difficulty:**
  - 5,352 pairs built from LiveCodeBench-v6 (problems May 2023 onward), labelled by unit tests.
  - Generators: "Claude-3.7-Sonnet, Gemini-2.5-Flash, and Gemini-2.5-Pro"; for CodeGen also "Qwen3-235B, Claude-4-Sonnet, Claude-4-Opus, Gemini-2.5-Flash-Lite".
  - Caveat: the easy/medium/hard split is *defined* by the fraction of judges that answer correctly.
  - Source: [arXiv 2507.10535](https://arxiv.org/pdf/2507.10535)
- **CodeJudgeBench Table 5, CodeGen accuracy by generator split** (full-code input; Gemini-2.5-Pro split vs Claude-3.7 split vs Gemini-2.5-Flash split):
  - Gemini-2.5-Pro judge: 72.85 / 90.46 / 82.79
  - QwQ: 61.33 / 88.15 / 72.33
  - Claude-4.0-Sonnet: 68.95 / 87.38 / 77.56
  - Claude-3.7-Sonnet: 61.33 / 73.38 / 66.16
  - Mean over all judges: 62.11 / 79.49 / 69.34
  - Source: [arXiv 2507.10535](https://arxiv.org/pdf/2507.10535)
- **CodeJudgeBench authors' reading:** "Ideally, LLM-as-a-Judge performance should remain consistent across outputs from different models, as code correctness is an objective criterion. However… we observe significant variability"; "QwQ is much better at judging responses from Claude-3.7-Sonnet than Gemini-2.5-Pro/Flash"; "even Gemini-2.5-Pro does not exhibit consistent performance across different splits". They attribute this to "coding style or response formatting". — [arXiv 2507.10535](https://arxiv.org/pdf/2507.10535)
- **CodeJudge (Tong & Zhang, EMNLP 2024):**
  - "To generate code snippets for HumanEval-X and APPS, we adopt the code from MultiPL-E. For BigCodeBench, we use the official pre-generated code samples from LLMs." It cites the MultiPL-E and BigCodeBench repos but names **no generator model**, so the generators are UNCONFIRMED.
  - CoNaLa uses the Evtikhiev et al. (2023) human ratings.
  - Judges: GPT-3.5-Turbo-1106, CodeLlama-Instruct-34B, Llama-3-8B/70B-Instruct.
  - Source: [arXiv 2410.02184](https://arxiv.org/pdf/2410.02184)
- **CodeBERTScore:**
  - HumanEval data from Cassano et al. (MultiPL-E), "the predictions of the Codex model (Chen et al., 2021) (code-davinci-002) and their corresponding functional correctness", in Java, C++, Python and JS.
  - Human-preference data: "We use the human annotation of Evtikhiev et al. (2022)" on CoNaLa.
  - Source: [EMNLP 2023](https://aclanthology.org/2023.emnlp-main.859)
- **ICE-Score:**
  - Same data: CoNaLa with Evtikhiev's human annotation "on the generated code from various models trained on the dataset", and HumanEval-X "predictions of code-davinci-002".
  - Judges: GPT-3.5-turbo, and GPT-4 in a later section.
  - Source: [Zhuo, Findings of EACL 2024](https://aclanthology.org/2024.findings-eacl.148) / [arXiv 2304.14317](https://arxiv.org/pdf/2304.14317)
- **Wang et al., "Can LLMs Replace Human Evaluators?", ISSTA 2025:**
  - Generators (Table 1, "sorted by their release date"): CodeLlama-Instruct 7/13/34B (2023.8) through Qwen2.5-Coder 1.5/7B (2024.9). That is 12 open code LLMs from seven families.
  - Judges: output-based methods use "GPT-4o for these methods" plus DeepSeek-V2.5. Also evaluated: Auto-J, Prometheus 2, G-Eval, BatchEval, GPTScore, Mixtral and CodeBERTScore-style embedding methods.
  - Source: [arXiv 2502.06193](https://arxiv.org/abs/2502.06193)

### Inferences
- Within one benchmark, the judge accuracy gap between generator splits (about 17 points on average, Gemini-2.5-Pro split vs Claude-3.7 split) is about as large as the gap between old and new benchmarks.
- However, the splits also differ in which problems are included, because pairs exist only where a generator produced both correct and incorrect code. The split effect is therefore not a clean "generator strength" effect. The authors themselves attribute it partly to style and formatting.
- Code metrics from 2023–2024 (CodeBERTScore, ICE-Score, CodeJudge's CoNaLa part) inherit 2018–2022 generators, giving a gap of 1–6 years at publication.

### Gaps
- Which models produced the MultiPL-E HumanEval-X and APPS samples that CodeJudge used is UNCONFIRMED; the paper does not say.
- The exact per-model values in CodeJudgeBench Figure 1 are UNCONFIRMED (figure only).

## Q5. Missed 2025–2026 MT or code metric papers that test metric quality across system strength or era

### Takeaway
Few papers test this directly. Relevant additions are:
- Proietti et al. (ACL 2025): top metrics now rank on par with or above human annotator baselines.
- Almaghout, Federmann, Gao (arXiv Oct 2026): LLM annotators on WMT23 and WMT25 MQM/ESA data.
- Chaplynskyi et al. (UNLP 2026): neural metrics rank GPT-5.2-era AI literary translations above human ones.
- CodeJudgeBench's ACL 2026 publication.

No 2025–2026 paper was found that systematically re-scores one metric across generator *eras*.

### Cited Findings
- **Proietti, Perrella, Navigli, "Has Machine Translation Evaluation Achieved Human Parity? The Human Reference and the Limits of Progress"**, ACL 2025 (short).
  - Adds human-annotator baselines to WMT Metrics rankings (WMT24 meta-evaluation).
  - Finding: human annotators are not consistently better than automatic metrics, and top metrics often rank on par with or above human baselines.
  - The authors caution against declaring parity.
  - Verified at abstract level only.
  - Source: [ACL Anthology 2025.acl-short.63](https://aclanthology.org/2025.acl-short.63) / [arXiv 2506.19571](https://arxiv.org/abs/2506.19571)
- **Almaghout, Federmann, Gao, "Large Language Models for Machine Translation Quality Annotation: Humans and Models Are Both Challenged"**, arXiv 2610.10918, Oct 2026.
  - Data: WMT23 and WMT25 data plus a 70-language-pair long-context set.
  - Finding: "LLM agreement with human annotators exceeds agreement between human annotators for some evaluation tasks", but agreement is unreliable in most settings, and LLMs struggle with minor errors and wrong language variants.
  - Translator and judge models are not named on the abstract page (UNCONFIRMED).
  - Source: [arXiv 2610.10918](https://arxiv.org/abs/2610.10918)
- **Chaplynskyi et al., UNLP 2026:** see Q3. All three AI translations (GPT-5.2, DeepL, Lapa) rank at the top under seven neural metrics, against human literary translations. — [2026.unlp-1.8](https://aclanthology.org/2026.unlp-1.8/)
- Search results also surfaced several leads that were not read and remain UNCONFIRMED for relevance:
  - an Indian-language metric-reliability study (ITEM, arXiv 2510.07061);
  - "Mind Which Bird You Favour" on adequacy–fluency balance in MT meta-evaluation (arXiv 2609.14795);
  - MCTS-Judge for code correctness (arXiv 2502.12468);
  - an LLM-as-judge-for-SE survey (arXiv 2510.24367).
  - Source: [web search results](https://arxiv.org/pdf/2510.07061)

### Inferences
- In MT, the 2025–2026 debate has shifted from "do metrics break on LLM outputs?" (WMT24: no) to two new problems. First, references and human annotators are now no better than the systems or metrics they serve as gold for (Proietti 2025; WMT23/24 references losing to LLMs). Second, segment-level LLM judging is unreliable (WMT25).
- Both problems mean validation sets need current frontier outputs and careful gold labels.

### Gaps
- No 2025–2026 MT or code paper was found that holds a metric fixed and measures its accuracy separately on outputs from older versus newer generator generations. Such a study appears to be missing from the literature, but given the limited search budget this is UNCONFIRMED.
