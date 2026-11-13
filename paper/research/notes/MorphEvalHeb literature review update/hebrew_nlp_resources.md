# Recent Hebrew NLP Resources, Models, and Benchmarks Relevant to MorphEvalHeb (2024–2026)

Scope note: items already cited by the paper (HeSum, Beyond N-Grams, HeQ, AlephBERT, HeRo, DictaBERT, NeoDictaBERT, DictaLM 2.0, Hebrew LLM Leaderboard) are excluded as "new" items. Status tags: **[peer-reviewed]**, **[arXiv-only]**, **[accepted, arXiv camera-ready]**, **[model/dataset card only]**. Each source page below was opened (WebFetch) unless marked "search-snippet only". Research date: 2026-11-12. Searches were done with about 25 tool calls, so coverage is broad but not exhaustive.

## Q1. Which new Hebrew models could serve as alternative backbones or baselines?

### Takeaway
The two main new options are **Dicta-LM 3.0** (Feb 2026: 24B, 12B and 1.7B, base/chat/thinking variants, Apache-2.0), which is the obvious open Hebrew LLM-as-judge baseline, and **HalleluBERT** (LREC 2026: RoBERTa base and large), which is the only new peer-reviewed Hebrew encoder to compare against NeoDictaBERT. I found no Hebrew ModernBERT model. TafBERTa is a tiny child-speech model and is relevant only as context for minimal pairs.

### Cited Findings
- **Dicta-LM 3.0** [arXiv-only]: Shaltiel Shmidman, Avi Shmidman, Amir DN Cohen, Moshe Koppel. "Dicta-LM 3.0: Advancing The Frontier of Hebrew Sovereign LLMs." arXiv:2602.02104, submitted 2 Feb 2026, cs.CL. The arXiv page lists no venue. — [arXiv](https://arxiv.org/abs/2602.02104)
  - Three sizes: 24B (adapted from Mistral-Small-3.1), 12B (from NVIDIA Nemotron Nano V2) and 1.7B (from Qwen3-1.7B). Native 65k context. Base and chat variants; the chat models support tool calling. — [arXiv](https://arxiv.org/abs/2602.02104)
  - Released checkpoints (dicta-il on HF): DictaLM-3.0-24B-Base, DictaLM-3.0-24B-Thinking, DictaLM-3.0-Nemotron-12B-Base, DictaLM-3.0-Nemotron-12B-Instruct, DictaLM-3.0-1.7B-Base, DictaLM-3.0-1.7B-Instruct, DictaLM-3.0-1.7B-Thinking. — [arXiv HTML](https://arxiv.org/html/2602.02104)
  - Continued pre-training used about 100B Hebrew tokens plus 30B English tokens (75% Hebrew / 25% English); Apache-2.0 license. — search-snippet summary of the report/model cards ([search source: arXiv PDF](https://www.arxiv.org/pdf/2602.02104)). The license and token counts were not re-checked on the HF card.
  - **Recommendation: (ii) baseline.** Use DictaLM-3.0-24B-Thinking or 12B-Instruct as an open Hebrew-native LLM-as-judge baseline next to GPT-4o/Gemini. The 1.7B model could also serve as a cheap decoder classifier backbone to compare with NeoDictaBERT.
- **HalleluBERT** [peer-reviewed]: Raphael Schmitt (sole author). "HalleluBERT: Let Every Token That Has Meaning Bear Its Weight." Proceedings of LREC 2026 (Palma de Mallorca), pp. 3022–3030, May 2026. — [ACL Anthology](https://aclanthology.org/2026.lrec-1.236/)
  - A RoBERTa-style Hebrew encoder in base and large sizes, trained from scratch on about 49.1 GB of deduplicated Hebrew web text and Wikipedia with a Hebrew byte-level BPE vocabulary. It reports the best unweighted mean on NER (BMC, NEMO) and sentiment (SMCD) against monolingual and multilingual baselines. MIT license. — [ACL Anthology](https://aclanthology.org/2026.lrec-1.236/)
  - Caveat: the abstract does not say whether NeoDictaBERT was among its baselines (the PDF did not parse).
  - **Recommendation: (ii) backbone comparison.** It is the most credible new alternative encoder to fine-tune with the same MorphEvalHeb classifier recipe. A large-size RoBERTa is also a natural ablation against NeoDictaBERT. Also (i) cite.
- **TafBERTa** [peer-reviewed workshop]: Anita Gelboim, Elior Sulem. "TafBERTa: Learning Grammatical Rules from Small-Scale Language Acquisition Data in Hebrew." Proceedings of the First BabyLM Workshop, Suzhou, Nov 2025, pp. 76–90. DOI 10.18653/v1/2025.babylm-main.6. — [ACL Anthology](https://aclanthology.org/2025.babylm-main.6/)
  - A compact RoBERTa trained on Hebrew child-directed speech (HTBerman corpus). — [ACL Anthology](https://aclanthology.org/2025.babylm-main.6/)
  - **Recommendation: (i) cite only**, for HeCLiMP (see Q3). It is not a useful backbone.
- **BERURIA (Sefaria)** [model card only, search-snippet only]: a ModernBERT-base model continued on English Jewish texts, with 6,000 Hebrew BPE tokens added. Its authors say it is *not* a Hebrew language model. — [HF](https://huggingface.co/Sefaria/BERURIA). **Recommendation: skip.**

### Inferences
- MorphEvalHeb's encoder comparison could plausibly be NeoDictaBERT vs. HalleluBERT-large vs. DictaBERT (already cited), plus Dicta-LM 3.0 as the judge baseline. This covers the main new 2026 Hebrew models a reviewer would ask about.
- The Dicta-LM 3.0 report itself uses GPT-4o as a summarization judge (see Q2). This means Hebrew-native factuality evaluation is still missing in Dicta's own evaluation pipeline, which strengthens MorphEvalHeb's motivation.

### Gaps
- I found no Hebrew ModernBERT, no Hebrew Gemma/Llama adaptation papers from 2025–26 outside Dicta, and no 2025–26 HeBERT or AlephBERTGimmel successor.
- I could not confirm whether HalleluBERT compares against NeoDictaBERT or reports UD/morphology tasks (the PDF was unreadable).
- I could not verify any independent (third-party) evaluation of Dicta-LM 3.0.

## Q2. Is there concurrent work on Hebrew summary factuality that competes with MorphEvalHeb?

### Takeaway
I found **no concurrent Hebrew-specific summary-factuality or hallucination-detection work** for 2024–Nov 2026. The nearest neighbours are (a) Dicta-LM 3.0's chat summarization benchmark, which uses GPT-4o pairwise win-rate and not factuality; (b) mFAVA (EMNLP 2025), a multilingual hallucination-rate study in which Hebrew had the *highest* estimated hallucination rate, measured on silver data only; (c) an EMNLP Findings 2026 position paper warning against over-trusting LLM judges in non-English settings; and (d) a 2026 English→Hebrew QE dataset that injects synthetic gender/number agreement errors, which is methodologically close to MorphEvalHeb's error injection.

### Cited Findings
- **Dicta-LM 3.0 chat benchmark** (Shmidman et al., 2026) [arXiv-only]. The summarization task uses 1,000 documents ("news, wiki, etc.", source unnamed). The metric is pairwise win-rate against Gemini-2.5-Pro reference summaries, with **GPT-4o as LLM-as-a-judge**. Other tasks: translation (same judge setup), 300-question Israeli Trivia, Hebrew Winograd (Shwartz 2021 translation), and Nikud (expert-diacritized sentences). There is a public leaderboard at huggingface.co/spaces/hebrew-llm-leaderboard/chat-leaderboard. — [arXiv HTML](https://arxiv.org/html/2602.02104)
  - **Recommendation: (i) cite** as the current Hebrew summarization eval practice. It is holistic preference by an English-centric judge with no factuality decomposition and no morphological error analysis, which is the gap MorphEvalHeb fills. It could also serve as (ii) a comparison point: run MorphEvalHeb on the same model outputs.
- **mFAVA** [peer-reviewed, EMNLP 2025]: Saad Obaid ul Islam, Anne Lauscher, Goran Glavaš. "How Much Do LLMs Hallucinate across Languages? On Realistic Multilingual Estimation of LLM Hallucination." arXiv:2502.12769 (v4, Feb 2026), EMNLP 2025. — [arXiv](https://arxiv.org/abs/2502.12769)
  - Hebrew is one of 30 languages, with a **silver test set only**: GPT-4 generated answers to 300 information-seeking prompts, then injected fine-grained hallucinations. Gold human annotations exist only for Arabic, Chinese, German, Russian and Turkish; silver and gold estimates correlate at r = 0.83. — [arXiv HTML v4](https://arxiv.org/html/2502.12769v4)
  - Hebrew had the **largest estimated hallucination rate (16.81%)** of the 30 languages. The detector on the Hebrew silver set reaches P = 82.75, R = 32.97, F1 = 44.19, the lowest recall of the silver languages. — [arXiv HTML v4](https://arxiv.org/html/2502.12769v4)
  - **Recommendation: (i) cite (strongly).** It is direct evidence that multilingual detectors miss Hebrew hallucinations (recall about 33%), which motivates a Hebrew-native detector. Note that it covers long-form QA, not summarization.
- **LLM-as-judge in multilingual/low-resource settings** [accepted, EMNLP Findings 2026 camera-ready on arXiv]: A. Seza Doğruöz, Xixian Liao, Verena Blaschke, Jakob Prange, Senyu Li, David Ifeoluwa Adelani. "Challenges and Recommendations for LLM-as-a-Judge in Multilingual Settings and for Low-Resource Languages." arXiv:2607.02235 (v2, 7 Sep 2026). — [arXiv](https://arxiv.org/abs/2607.02235)
  - Of 650 ACL Anthology papers using LLM-as-a-judge, only 33 focus on multilingual or low-resource settings. The authors report inconsistent outcomes, over-trust in LLM judgments, and reliance on a single judge model. They provide a checklist. The paper does not mention Hebrew. — [arXiv](https://arxiv.org/abs/2607.02235)
  - **Recommendation: (i) cite** to justify not relying on GPT-4o-as-judge for Hebrew, and to justify a multi-judge baseline (GPT-4o + Dicta-LM 3.0).
- **Hebrew QE with synthetic agreement errors** [arXiv-only]: Assaf Siani, Anna Kernerman, Ilan Kernerman. "Semi-Synthetic Parallel Data for Translation Quality Estimation: A Case Study of Dataset Building for an Under-Resourced Language Pair." arXiv:2603.11743, 12 Mar 2026. — [arXiv](https://arxiv.org/abs/2603.11743)
  - An English→Hebrew sentence-level QE dataset built from multiple MT engines, scored by a linguist. It introduces "controlled translation errors", particularly in **gender and number agreement**, and trains BERT and XLM-R QE models. — [arXiv](https://arxiv.org/abs/2603.11743)
  - **Recommendation: (i) cite** as the closest prior work on Hebrew morpho-syntactic synthetic error injection (agreement corruption) for quality classifiers. Its task differs (MT QE, not summary factuality), so it is not a competitor.
- **Multilingual Self-Taught Faithfulness Evaluators** [arXiv-only]: Carlo Alfano, Aymen Al Marjani, Zeno Jonke, Amin Mantrach, Saab Mansour, Marcello Federico. arXiv:2507.20752, 28 Jul 2025. It trains faithfulness evaluators only on synthetic multilingual summarization data with cross-lingual transfer and reports that it beats English evaluators and MT-based approaches. — [arXiv](https://arxiv.org/abs/2507.20752)
  - The abstract page does not list languages, so Hebrew coverage is **unverified**.
  - **Recommendation: (i) cite** as the closest multilingual methodological competitor (synthetic data → faithfulness classifier). If code or models are released, it could be (ii) a baseline.
- **AraHalluEval** [arXiv-only, search-snippet only]: an Arabic fine-grained hallucination evaluation framework covering QA and summarization with 12 indicators (arXiv:2509.04656, Sep 2025). — [arXiv](https://arxiv.org/abs/2509.04656v1). **Recommendation: (i) cite** as the sister Semitic-language analogue. Authors and venue were not verified.
- **JudgeMeNot** [accepted, Findings of ACL 2026]: Itay Razumenko, Arnon Sturm, Nir Grinberg. "JudgeMeNot: Personalizing Large Language Models to Emulate Judicial Reasoning in Hebrew." arXiv:2604.18041, 20 Apr 2026. It evaluates Hebrew generation by lexical, stylistic and semantic similarity. "Judge" here means court judges, not LLM-as-judge. — [arXiv](https://arxiv.org/abs/2604.18041). **Recommendation: optional (i) cite** as recent Hebrew generation-evaluation work. It is not a competitor.

### Inferences
- MorphEvalHeb looks novel as a Hebrew-native factuality evaluator with a morphological error taxonomy. No 2024–2026 paper combines Hebrew, summarization, factuality and morpho-syntactic error types.
- The mFAVA Hebrew finding (highest hallucination rate, about 33% detector recall) together with Dicta-LM 3.0's reliance on GPT-4o judging makes a clean two-sentence motivation for the paper.

### Gaps
- I could not confirm whether arXiv:2507.20752 or any SemEval-2025 Mu-SHROOM split includes Hebrew.
- I did not search Hugging Face exhaustively for small, unpublished Hebrew factuality datasets (for example under HebArabNlpProject or IAHLT).

## Q3. Which Hebrew datasets, benchmarks and tools should be cited?

### Takeaway
New citable resources: **LCHAIM** (Hebrew long-context NLI, Findings ACL 2025) and **HebNLI** (MAFAT/National NLP Program NLI training data) for NLI-based factuality; **HeCLiMP** (Hebrew minimal pairs, agreement) and **MultiBLiMP 1.0** (TACL 2026, agreement minimal pairs including Hebrew) for the morphology taxonomy; the **Dicta-LM 3.0 chat benchmark** for summarization; and **DictaBERT-Parse** ("MRL Parsing Without Tears", Findings ACL 2024) and the **Levi & Tsarfaty joint lattice parser** (EACL 2024) as morphological-analysis tools for error injection and verification.

### Cited Findings
**NLI / entailment (useful for NLI-style factuality training or transfer)**
- **LCHAIM** [peer-reviewed]: Ehud Malul, Oriel Perets, Ziv Mor, Yigal Kassel, Elior Sulem. "LCHAIM - Investigating Long Context Reasoning in Hebrew." Findings of ACL 2025, pp. 7928–7939, DOI 10.18653/v1/2025.findings-acl.413. It has 8,325 context–hypothesis pairs translated and validated from ConTRoL. A fine-tuned LongHero reaches 52% mean accuracy, 35 points below humans; the best few-shot LLMs (Gemma-9B, Dicta-LM-2.0-7B, GPT-4o) reach 60.12%. — [ACL Anthology](https://aclanthology.org/2025.findings-acl.413/)
  - **Recommendation: (i) cite.** Optionally (ii): use it as an intermediate NLI fine-tuning stage for the NeoDictaBERT classifier, or as a transfer test. It also introduces LongHero, a long-context HeRo variant that could be a document-level encoder candidate (unverified as a standalone release).
- **HebNLI** [dataset card only]: HebArabNlpProject on HF, attributed to the "Israel National NLP Program". The schema (premise/hypothesis + Hebrew translations, genres slate/government/fiction/travel/…) indicates machine-translated MultiNLI. Train/validation/test splits, about 189 MB, license listed as "other". No paper or citation is given. — [HF dataset card](https://huggingface.co/datasets/HebArabNlpProject/HebNLI)
  - A search snippet says the translation and quality checks were done by Webiks for MAFAT and the license is CC-BY-3.0. I did not see this on the card I opened, so treat it as unverified.
  - **Recommendation: (i) cite** (as a URL/dataset footnote) and **(ii)** as an NLI pre-fine-tuning source for the classifier.

**Morphology / syntax minimal pairs (supports the error taxonomy)**
- **HeCLiMP** [peer-reviewed workshop]: introduced in Gelboim & Sulem (BabyLM 2025). It is a Hebrew grammaticality benchmark of minimal pairs focused on **determiner–noun agreement**. — [ACL Anthology](https://aclanthology.org/2025.babylm-main.6/)
  - A third-party model page reports TafBERTa at about 69.3% vs. HeRo at 73.5% on HeCLiMP. Search-snippet only; check the paper before quoting.
  - **Recommendation: (i) cite** as the only Hebrew-native minimal-pair benchmark. It covers agreement only and does not cover binyan, smichut or root, which MorphEvalHeb's taxonomy adds.
- **MultiBLiMP 1.0** [peer-reviewed]: Jaap Jumelet, Leonie Weissweiler, Joakim Nivre, Arianna Bisazza. "MultiBLiMP 1.0: A Massively Multilingual Benchmark of Linguistic Minimal Pairs." TACL vol. 14, pp. 193–216, 2026, DOI 10.1162/tacl.a.600. It covers 101 languages and more than 128,000 subject–verb agreement minimal pairs (number, person, gender), auto-generated from UD and UniMorph. — [ACL Anthology](https://aclanthology.org/2026.tacl-1.10/)
  - Hebrew inclusion is indicated by the paper's per-language figure (search snippet of [arXiv v2](https://arxiv.org/html/2504.02768v2)). I did not see this on the TACL abstract page; verify on [HF dataset](https://huggingface.co/datasets/jumelet/multiblimp).
  - **Recommendation: (i) cite.** Its UD/UniMorph-based perturbation is methodologically parallel to MorphEvalHeb's synthetic injection. The Hebrew subset could also be (ii) a sanity probe of backbone sensitivity to gender agreement.

**Summarization / generation evaluation**
- **Dicta-LM 3.0 Hebrew chat benchmark + chat leaderboard** (see Q2). — [arXiv HTML](https://arxiv.org/html/2602.02104). **Recommendation: (i) cite**, separately from the base-model Hebrew LLM Leaderboard that the paper already cites.

**Morphological analysis / parsing tools (for injection and verification pipelines)**
- **DictaBERT-Parse / "MRL Parsing Without Tears: The Case of Hebrew"** [peer-reviewed]: Shaltiel Shmidman, Avi Shmidman, Moshe Koppel, Reut Tsarfaty. Findings of ACL 2024 (arXiv:2403.06970). It is a "flipped pipeline" joint model for segmentation, POS, morphological features and dependencies. The large model reports POS 97.35% and dependency 89.5 UAS / 85.4 LAS, and it is more than 100× faster than Levi & Tsarfaty. — [ACL Anthology PDF](https://aclanthology.org/2024.findings-acl.269.pdf); [arXiv](https://arxiv.org/pdf/2403.06970). Venue and numbers come from a search snippet; the PDF was not opened in this session.
  - **Recommendation: (i) cite** as the tool for morphological analysis in the error-injection pipeline (gender/number features, prefixes, lemma). This is the de facto YAP successor. If MorphEvalHeb already uses dicta-il/dictabert-joint, cite this paper for it.
- **Levi & Tsarfaty (EACL 2024)** [peer-reviewed]: Danit Yshaayahu Levi, Reut Tsarfaty. "A Truly Joint Neural Architecture for Segmentation and Parsing." EACL 2024 (Vol. 1: Long Papers), pp. 1408–1420, St. Julian's, Malta; arXiv:2402.02564. It is a lattice-based joint segmentation and parsing model and reports SOTA on Hebrew UD. — [arXiv](https://arxiv.org/abs/2402.02564); [EACL virtual](https://virtual2024.eacl.org/paper_207-Oral.html). Venue and pages come from a search snippet.
  - **Recommendation: (i) cite** as the ONLP/Tsarfaty-lab state of the art for morphological disambiguation.
- **YAP** (ONLP, GitHub) remains the classic lexicon-based analyzer. — [GitHub](https://github.com/OnlpLab/yap). Cite only if the paper discusses older tooling.

**Multilingual benchmarks including Hebrew (knowledge, not summarization)**
- **Global-MMLU** covers 42 languages including Hebrew (search snippet: [arXiv 2412.03304](https://arxiv.org/pdf/2412.03304v2)). It is relevant only if a general Hebrew-capability ranking of judge models is needed. **Recommendation: optional (i).** Authors and venue were not verified.
- **INCLUDE** (Romanou et al., ICLR 2025; 44 languages, 197,243 MCQA) — [arXiv](https://arxiv.org/pdf/2411.19799). I could **not confirm Hebrew** coverage. Do not cite for Hebrew without checking.

### Inferences
- The strongest new citation set for related work: Dicta-LM 3.0 (models + chat benchmark), HalleluBERT, LCHAIM, HebNLI, HeCLiMP, MultiBLiMP, mFAVA, the multilingual LLM-as-judge critique, Siani et al. (Hebrew synthetic agreement errors), and DictaBERT-Parse / Levi & Tsarfaty for tooling.
- MorphEvalHeb's taxonomy goes beyond every existing Hebrew minimal-pair resource. HeCLiMP covers only determiner–noun agreement and MultiBLiMP only subject–verb agreement. No resource covers binyan swaps, smichut or root confusion, and this is a defensible novelty claim.

### Gaps
- I found no new (2025–26) Hebrew summarization dataset beyond HeSum, and no new Hebrew reading-comprehension dataset beyond HeQ.
- I found no new Hebrew UD treebank release or IAHLT paper in 2025–26 searches. The IAHLT/MAFAT Hugging Face orgs (HebArabNlpProject) may hold more datasets; only HebNLI and LCHAIM were checked.
- Trankit and Stanza Hebrew updates were not investigated. I found no evidence of 2024–26 Hebrew-specific releases.
- The May–Nov 2026 window yielded only HalleluBERT (LREC, May 2026), JudgeMeNot (ACL Findings 2026) and the LLM-as-judge critique (EMNLP Findings 2026). EMNLP 2026 and other late-2026 proceedings may contain Hebrew papers not yet indexed by search.
