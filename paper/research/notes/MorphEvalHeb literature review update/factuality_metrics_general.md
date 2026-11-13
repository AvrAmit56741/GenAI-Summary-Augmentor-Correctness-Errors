# General (English / multilingual) factuality and faithfulness evaluation, 2024–2026: update for MorphEvalHeb

Scope: English and general (not Hebrew-specific) work on fine-tuned factuality verifiers and hallucination detectors, claim decomposition, long-document faithfulness, LLM-as-judge reliability, and meta-evaluation benchmarks. Items already cited in the paper are excluded: SummaC, AlignScore, QAFactEval, FactCC, QAGS, TRUE, FActScore, MiniCheck, FactKB, SelfCheckGPT, G-Eval, SEAHORSE, RAGAS, ARES, BERTScore, BARTScore, BLEURT, MoverScore, Maynez 2020, Goyal & Durrett 2020, and Cao 2022.

Verification method: every arXiv item was checked against the arXiv API record (title, authors, first-posted date, author comment with the venue). ACL Anthology items were checked through their .bib pages. Hugging Face models were checked through the HF model API and README. "Peer-reviewed" means the venue appears in the arXiv author comment or the ACL Anthology record. I did not open the camera-ready proceedings for every item. Today is 2026-11-12.

## Q1. Which 2025–2026 metrics and detectors are strong, open, and multilingual enough to serve as a baseline on Hebrew?

### Takeaway
None of the strong open grounded-factuality checkers declares Hebrew support. FactCG, Bespoke-MiniCheck-7B, Granite Guardian 3.3, HHEM-2.1-open and Lynx are all tagged English-only on their Hugging Face cards. The most realistic new open baselines are:
- (a) FactCG-DeBERTa-L, small and the strongest encoder on LLM-AggreFact, run zero-shot on Hebrew or with translate-then-check;
- (b) the multilingual LettuceDetect v2 family (2026), built on mmBERT (encoder) or Qwen3.5-2B (generative). Its backbones are massively multilingual, but it was trained and tested on 7 languages plus PsiloQA's 14, and Hebrew is not among them;
- (c) Bespoke-MiniCheck-7B or Granite Guardian 3.3 8B, as LLM-based checkers whose Qwen2/Granite backbones have some Hebrew ability. All of these are zero-shot on Hebrew, so treat them as cross-lingual-transfer baselines.

### Cited Findings
- **FactCG** (Deren Lei, Yaxi Li, Siyao Li, Mengya Hu, Rui Xu, Ken Archer, Mingyu Wang, Emily Ching, et al.), "FactCG: Enhancing Fact Checkers with Graph-Based Multi-Hop Data", NAACL 2025 (peer-reviewed).
  - Method: generates synthetic training data (CG2C) by multi-hop reasoning over context graphs extracted from documents. It reports better performance than earlier synthetic-data checkers using the same backbones. — [arXiv 2501.17144](https://arxiv.org/abs/2501.17144)
  - Released model `yaxili96/FactCG-DeBERTa-v3-Large`: deberta-v3-large base, language tag `en`, MIT license. — [HF API](https://huggingface.co/yaxili96/FactCG-DeBERTa-v3-Large)
  - On LLM-AggreFact, FactCG-DeBERTa-L (0.4B) averages 75.6, above MiniCheck-Flan-T5-L (0.8B) at 75.0. — [LLM-AggreFact leaderboard](https://llm-aggrefact.github.io/)
  - Relevance and recommendation: this is the closest analogue to MorphEvalHeb's approach (a small encoder classifier trained on synthetic contrastive or graph data). **(ii) Add as a baseline.** It runs on Hebrew zero-shot, but deberta-v3 has an English vocabulary and will tokenize Hebrew poorly, so expect weak results. Report it zero-shot and with MT-to-English. Cite it in any case.
- **Bespoke-MiniCheck-7B** (Bespoke Labs, Hugging Face model, Aug 2024; no separate paper). It is the top open model listed on the LLM-AggreFact blog ("The best performance on our leaderboard comes from a 7B model") and averages 77.4. — [LLM-AggreFact blog](https://llm-aggrefact.github.io/blog); [leaderboard](https://llm-aggrefact.github.io/)
  - HF card language tag `en`; license not set in the card metadata (check the README before use). — [HF](https://huggingface.co/bespokelabs/Bespoke-MiniCheck-7B)
  - Relevance and recommendation: this is the de facto SOTA open MiniCheck successor. **(ii) Add as a baseline.** It is feasible on Hebrew input because it is a 7B LLM backbone, but it is zero-shot cross-lingual. Cite the HF card plus the MiniCheck paper.
- **Granite Guardian** (Inkit Padhi, Manish Nagireddy, Giandomenico Cornacchia, Subhajit Chaudhury, Tejaswini Pedapati, Pierre Dognin, Keerthiram Murugesan, Erik Miehling, et al.), "Granite Guardian", arXiv 2024 (arXiv-only per the arXiv record). — [arXiv 2412.07724](https://arxiv.org/abs/2412.07724)
  - Granite Guardian 3.3 (8B, created 2025-06-03, Apache-2.0, language tag `en`) scores 76.5 average on LLM-AggreFact. — [HF](https://huggingface.co/ibm-granite/granite-guardian-3.3-8b); [leaderboard](https://llm-aggrefact.github.io/)
  - Relevance and recommendation: an open, permissively licensed groundedness detector at near-SOTA. **(ii) Optional baseline**, zero-shot on Hebrew. **(i) Cite.**
- **HHEM-2.1-open** (Vectara): flan-t5-base backbone, `en`, Apache-2.0. — [HF](https://huggingface.co/vectara/hallucination_evaluation_model)
  - The companion paper is Tamber, Bao, Xu, Luo, Kazi, Bae, Li, Mendelevitch, et al., "Benchmarking LLM Faithfulness in RAG with Evolving Leaderboards", EMNLP 2025 Industry Track (peer-reviewed). It describes the HHEM leaderboard and introduces **FaithJudge**, an LLM-as-judge that uses a pool of human-annotated hallucination examples and improves on HHEM-style detectors for summarization, QA and data-to-text. — [arXiv 2505.04847](https://arxiv.org/abs/2505.04847)
  - Relevance and recommendation: HHEM is the most widely used small summarization-hallucination classifier. **(ii) Cheap baseline** (T5 tokenizer, so Hebrew coverage is poor; use MT). **(i) Cite** FaithJudge as an LLM-judge design.
- **Lynx** (Selvan Sunitha Ravi, Bartosz Mielczarek, Anand Kannappan, Douwe Kiela, Rebecca Qian), "Lynx: An Open Source Hallucination Evaluation Model", arXiv 2024 (arXiv-only). — [arXiv 2407.08488](https://arxiv.org/abs/2407.08488)
  - HF model: 8B, `en`, CC-BY-NC-4.0. — [HF](https://huggingface.co/PatronusAI/Llama-3-Patronus-Lynx-8B-Instruct-v1.1)
  - Recommendation: **(i) cite**. As a baseline it is lower priority than FactCG or Bespoke-MiniCheck because of its non-commercial license and English-only training.
- **LettuceDetect v1** (Ádám Kovács, Gábor Recski), "LettuceDetect: A Hallucination Detection Framework for RAG Applications", arXiv 2025 (arXiv-only; 6 pages). It uses ModernBERT token classification. — [arXiv 2502.17125](https://arxiv.org/abs/2502.17125)
- **LettuceDetect v2** (Ádám Kovács, Bowei He, Xue Liu, István Boros, Szilveszter Tóth, Gábor Recski), "Beyond Document Grounding: Span-Level Hallucination Detection over Code, Tool Output, and Documents", GroundLM 2026 workshop at EMNLP 2026 (peer-reviewed workshop). — [arXiv 2607.00895](https://arxiv.org/abs/2607.00895); [ACL Anthology 2026.groundlm-1.44](https://aclanthology.org/2026.groundlm-1.44/)
  - The fine-tuned Qwen3.5-2B detector reaches 0.689 span-F1 on the unified test set, 81.8 RAGTruth example-F1, and 0.724 English PsiloQA IoU. — [arXiv 2607.00895](https://arxiv.org/abs/2607.00895)
  - `KRLabsOrg/lettucedect-v2-mmbert-base` (mmBERT-base, Apache-2.0, created 2026-06-22) is tagged for en/de/fr/es/it/pl/zh. The card reports RAGTruth example-F1 74.3 and multilingual PsiloQA results. It also says off-the-shelf detectors (HHEM, Lynx, Granite, MiniCheck) are "near chance" on code-agent answers. — [HF card](https://huggingface.co/KRLabsOrg/lettucedect-v2-mmbert-base)
  - `KRLabsOrg/lettucedect-v2-qwen-2b` (Qwen3.5-2B, Apache-2.0) outputs typed hallucinated spans (category plus subcategory) as JSON. — [HF card](https://huggingface.co/KRLabsOrg/lettucedect-v2-qwen-2b)
  - A further variant, `LFM2.5-Encoder-350M-hallucination-detector` (created 2026-08-16, LFM1.0 license), scores span-F1 0.690 and example-F1 0.945 on 14-language PsiloQA. — [HF card](https://huggingface.co/KRLabsOrg/LFM2.5-Encoder-350M-hallucination-detector)
  - Relevance and recommendation: this is the strongest **multilingual open encoder** detector found. Its design (a multilingual BERT-family encoder fine-tuned as a hallucination classifier, with a typed-error head) mirrors MorphEvalHeb's verifier plus error-type detector. **(ii) Strongly consider as a baseline.** The mmBERT backbone covers Hebrew in pre-training, but the detector was not trained or evaluated on Hebrew, and it was trained on RAG/QA rather than summarization. It needs a (context, question, answer) framing, for example "Summarize the document" as the question. **(i) Cite** as the closest multilingual analogue.
- **PsiloQA** (Elisei Rykov, Kseniia Petrushina, Maksim Savkin, Valerii Olisov, Artem Vazhentsev, Kseniia Titova, Alexander Panchenko, Vasily Konovalov, et al.), "When Models Lie, We Learn: Multilingual Span-Level Hallucination Detection with PsiloQA", arXiv 2025 (arXiv-only per the record). — [arXiv 2510.04849](https://arxiv.org/abs/2510.04849)
  - It has span-level annotations in 14 languages and finds that encoder-based models perform best across languages.
  - Its language list is en, fi, zh, hi, ca, sv, cs, fa, es, it, eu, ar, fr, de, so **no Hebrew**. — [HF dataset card](https://huggingface.co/datasets/s-nlp/PsiloQA)
  - Recommendation: **(i) cite** as evidence that multilingual encoder detectors beat LLM tagging. This supports MorphEvalHeb's encoder choice.
- **FaithLens** (Shuzheng Si, Qingyi Wang, Haozhe Zhao, Yuzhuo Bai, Guanqiao Chen, Kangyang Luo, Gang Chen, Fanchao Qi), "FaithLens: Detecting and Explaining Faithfulness Hallucination", ACL 2026 Findings (peer-reviewed). It is trained on LLM-synthesized data with explanations and filtered labels: SFT cold start, then RL. It outputs a binary verdict plus an explanation and is evaluated on 12 tasks from LLM-AggreFact and HoVer. — [arXiv 2512.20182](https://arxiv.org/abs/2512.20182)
  - Recommendation: **(i) cite** as a 2026 synthetic-data-trained detector, conceptually parallel to MorphEvalHeb's synthetic contrastive training. I did not verify a Hugging Face release, model size or language coverage.
- **How Much Do LLMs Hallucinate across Languages?** (Saad Obaid ul Islam, Anne Lauscher, Goran Glavaš), EMNLP 2025 main (peer-reviewed). The authors translate-train a multilingual hallucination detector from English data and study 30 languages. — [arXiv 2502.12769](https://arxiv.org/abs/2502.12769); [ACL Anthology](https://aclanthology.org/2025.emnlp-main.1481/)
  - Recommendation: **(i) cite** as the main multilingual-detector precedent. Whether Hebrew is among the 30 languages was not checked.
- **MultiWikiQHalluA** (Freja Thoresen, Dan Saattrup Smart), "A multilingual hallucination benchmark: MultiWikiQHalluA", RESOURCEFUL 2026 workshop (camera-ready per the arXiv comment). It fine-tunes LettuceDetect-style mmBERT-small classifiers for 30 European languages. — [arXiv 2605.02504](https://arxiv.org/abs/2605.02504)
  - Recommendation: **(i) cite** briefly as a 2026 multilingual encoder detector. It covers European languages only.
- **How Far Can You Get Without a GPU?** (Kriti Faujdar, Smit Kadvani), Findings track of GroundLM 2026, an EMNLP 2026 workshop. It benchmarks ROUGE-L, semantic similarity, BERTScore and a FEVER-trained DeBERTa NLI detector, plus an ensemble, across QA, dialogue and summarisation. — [arXiv 2606.29809](https://arxiv.org/abs/2606.29809)
  - Recommendation: **(i) cite** to justify including cheap baselines (ROUGE, BERTScore, NLI). A multilingual-NLI baseline (for example mDeBERTa-XNLI) would be an easy add-on, but I did not verify that paper's specific numbers.
- MiniCheck's own paper says its models are "trained exclusively on English data" and were not systematically tested on other languages (the search tool quoted this from the MiniCheck PDF). — [MiniCheck arXiv 2404.10774](https://arxiv.org/pdf/2404.10774v2)

### Inferences
- The English-only status of every top LLM-AggreFact checker strengthens MorphEvalHeb's motivation for a Hebrew-native verifier. State it explicitly with these HF-card citations.
- Recommended baseline additions, in priority order:
  1. FactCG-DeBERTa-L (cheap; also a natural "English SOTA encoder under MT" comparison);
  2. lettucedect-v2-mmbert-base (multilingual encoder, closest design analogue);
  3. Bespoke-MiniCheck-7B or Granite Guardian 3.3 8B (strong open LLM checkers, zero-shot Hebrew);
  4. optionally HHEM-2.1-open.

  Running each in two modes, Hebrew-native and translate-to-English, gives a clean cross-lingual-transfer comparison.
- Because Kendall tau is computed against human labels, the span-level detectors (LettuceDetect) need aggregation to a summary-level score, for example the maximum token probability or the fraction of unsupported tokens.

### Gaps
- No open grounded-factuality checker with declared Hebrew training or evaluation was found.
- I could not identify the makers, openness, or release dates of the current LLM-AggreFact top entries (nimble-v4-max 79.5, Clef 79.0, Decider v1.2 78.7, Jev 1.13.0, d1-ultra, nimble-v4 9B, Clef Flash, Drex v1.5). The leaderboard page gives no provenance and the blog does not mention them, so they are probably proprietary or commercial.
- Whether FaithLens weights are released, and its parameter size, was not verified.
- Whether Hebrew is among the 30 languages in ul Islam et al. (2025) was not verified.

## Q2. What is the current state of the art on LLM-AggreFact or similar benchmarks?

### Takeaway
The current LLM-AggreFact leaderboard, which shows 19 of 47 models by default, is topped by unattributed 9B–27B checkers: nimble-v4-max (27B) at 79.5, Clef (27B) at 79.0, and Decider v1.2 (27B) at 78.7. The best clearly open model is Bespoke-MiniCheck-7B at 77.4, ahead of Claude-3.5 Sonnet (77.2), Granite Guardian 3.3 (76.5), GPT-4o (75.9) and FactCG-DeBERTa-L (75.6). Small specialized checkers still match or beat frontier LLM judges, and every system scores roughly 60 on ExpertQA, which looks like a ceiling.

### Cited Findings
- Leaderboard snapshot (fetched 2026-11-12), average over 11 datasets (CNN, XSum, MediaS, MeetB, WiCE, REVEAL, ClaimVerify, FactCheck, ExpertQA, LFQA, RAGTruth). — [LLM-AggreFact leaderboard](https://llm-aggrefact.github.io/)

  | Model | Size | Average |
  |---|---|---|
  | nimble-v4-max | 27B | 79.5 |
  | Clef | 27B | 79.0 |
  | Decider v1.2 | 27B | 78.7 |
  | Jev 1.13.0 | not listed | 78.7 |
  | d1-ultra | not listed | 78.5 |
  | nimble-v4 | 9B | 77.8 |
  | Bespoke-MiniCheck-7B | 7B | 77.4 |
  | Claude-3.5 Sonnet | not listed | 77.2 |
  | Clef Flash | 9B | 76.6 |
  | Drex v1.5 | 9B | 76.5 |
  | Granite Guardian 3.3 | 8B | 76.5 |
  | Mistral-Large 2 | 123B | 76.5 |
  | gpt-4o-2024-05-13 | not listed | 75.9 |
  | FactCG-DeBERTa-L | 0.4B | 75.6 |
  | Qwen2.5-72B-Instruct | 72B | 75.6 |
  | MiniCheck-Flan-T5-L | 0.8B | 75.0 |
  | Llama-3.3-70B | 70B | 74.5 |
  | Llama-3.1-405B | 405B | 74.4 |
  | QwQ-32B-Preview | 32B | 71.8 |

- On the summarization subsets, FactCG-DeBERTa-L has the best CNN score among listed models (70.1). On XSum, nimble-v4-max leads with 78.6 and Bespoke-MiniCheck-7B scores 77.8. — [leaderboard](https://llm-aggrefact.github.io/)
- A third-party mirror (benchmarklist.com) showed results "updated May 29, 2026" with matching top entries. That is an aggregator, not a primary source. — [benchmarklist](https://benchmarklist.com/benchmarks/llm_aggrefact/)
- Vectara's FaithJudge leaderboard is an alternative, summarization-heavy benchmark of more than 160 LLMs' hallucination rates. It replaces HHEM scoring with a few-shot LLM judge grounded in human-annotated examples. — [arXiv 2505.04847](https://arxiv.org/abs/2505.04847)
- **FaithBench** (Forrest Sheng Bao, Miaoran Li, Renyi Qu, Ge Luo, Erana Wan, Yujia Tang, Weisi Fan, Manveer Singh Tamber, et al.), "FaithBench: A Diverse Hallucination Benchmark for Summarization by Modern LLMs", arXiv 2024. The arXiv record shows no venue; I believe it appeared at NAACL 2025 but did not verify this, so cite the arXiv version unless confirmed. — [arXiv 2410.13210](https://arxiv.org/abs/2410.13210)
  - Recommendation: **(i) cite** as the modern-LLM summarization hallucination benchmark (a FRANK successor).
- **The Gray Zone of Faithfulness** (Qiang Ding, Lvzhou Luo, Yixuan Cao, Ping Luo), arXiv 2025 (v4; venue not stated in the record). It proposes a faithfulness annotation framework with an intermediate "Out-Dependent" label for summaries that use external or common-sense knowledge, motivated by annotation ambiguity in existing benchmarks. — [arXiv 2510.21118](https://arxiv.org/abs/2510.21118)
  - Recommendation: **(i) cite**. It is directly relevant to how MorphEvalHeb defines "faithful" for human labels and the error taxonomy.
- **Do Benchmarks Underestimate LLM Performance?** (I. F. Atasoy, B. Mutlu, E. A. Sezer, A. Wahdan), ROMCIR workshop at ECIR 2026. It argues that hallucination-detection benchmarks underestimate LLMs when labels are re-adjudicated with an LLM-first, human-adjudicated process. — [arXiv 2605.08462](https://arxiv.org/abs/2605.08462)
  - Recommendation: **(i) optional cite** on label noise in meta-evaluation. Only the title and abstract-level claim were verified.
- **UniSumEval** (Yuho Lee, Taewon Yun, Jason Cai, Hang Su, Hwanjun Song), EMNLP 2024 Findings: a unified, fine-grained, multi-dimensional summarization evaluation benchmark for LLM summaries. — [arXiv 2409.19898](https://arxiv.org/abs/2409.19898)
  - Recommendation: **(i) cite** among meta-evaluation benchmarks.

### Inferences
- For the paper, the defensible SOTA statement is: "On LLM-AggreFact, small specialized checkers (Bespoke-MiniCheck-7B, 77.4; FactCG-DeBERTa-L, 0.4B, 75.6) match or exceed frontier LLM judges (Claude-3.5 Sonnet, 77.2; GPT-4o, 75.9). All top systems are English-only." Do not cite the unattributed top entries as specific models.
- LLM-AggreFact uses balanced accuracy at a binary threshold, whereas MorphEvalHeb reports Kendall tau. Note this metric difference when positioning the work.

### Gaps
- No 2026 peer-reviewed update to LLM-AggreFact itself (for example, new datasets) was found.
- The leaderboard's last-update date is not shown.
- No FRANK-style error-typology benchmark from 2025–2026 was found, beyond FaithBench and the Gray Zone paper.

## Q3. Which papers should be cited in related work (factuality evaluation, claim decomposition, long-document, LLM-as-judge)?

### Takeaway
Related work should add four groups:
- **Detectors:** FactCG, FaithLens, LettuceDetect v1/v2, Granite Guardian, Lynx, HHEM/FaithJudge, and TrueTeacher (2023, but central).
- **Claim decomposition:** SAFE/long-form factuality, VeriScore, FENICE, FIZZ, Claimify, Decomposition Dilemmas, A Closer Look at Claim Decomposition, DnDScore, VeriFastScore, and ElementCheck.
- **Long-document faithfulness:** FineSurE, LongDocFACTScore, Stress Testing (ACL 2026), and VeriTrail (ICLR 2026).
- **LLM-as-judge reliability:** Prometheus 2, Rating Roulette, How Reliable is Multilingual LLM-as-a-Judge?, the IRT diagnosis paper (ICML 2026), and the GEM 2026 empirical study.

### Cited Findings

#### Detectors and fine-tuned verifiers (beyond Q1)
- **TrueTeacher** (Zorik Gekhman, Jonathan Herzig, Roee Aharoni, Chen Elkind, Idan Szpektor), EMNLP 2023. It trains factual-consistency evaluators on LLM-annotated model-generated summaries, and the paper includes a multilingual (mT5) variant. — [arXiv 2305.11171](https://arxiv.org/abs/2305.11171)
  - Recommendation: **(i) cite.** It is the key precedent for synthetic or LLM-labelled training data for consistency classifiers. It predates 2024 but is not yet cited. The multilingual claim comes from my knowledge of the paper; confirm it in the PDF before relying on it.
- **Lookback Lens** (Yung-Sung Chuang, Linlu Qiu, Cheng-Yu Hsieh, Ranjay Krishna, Yoon Kim, James Glass), EMNLP 2024 main. It detects contextual hallucination from attention maps (lookback ratio) and includes summarization. — [arXiv 2407.07071](https://arxiv.org/abs/2407.07071)
  - Recommendation: **(i) cite** briefly (white-box detection; not a baseline, because it needs the generator's internals).
- **Zero-shot Factual Consistency Evaluation Across Domains** (Raunak Agarwal), arXiv 2024 (arXiv-only). — [arXiv 2408.04114](https://arxiv.org/abs/2408.04114)
  - Recommendation: optional citation.

#### Claim or atomic-fact decomposition and verification
- **Long-form factuality in large language models (SAFE/LongFact)** (Jerry Wei, Chengrun Yang, Xinying Song, Yifeng Lu, Nathan Hu, Jie Huang, Dustin Tran, Daiyi Peng, et al.), NeurIPS 2024. — [arXiv 2403.18802](https://arxiv.org/abs/2403.18802)
  - Recommendation: **(i) cite.**
- **VeriScore** (Yixiao Song, Yekyung Kim, Mohit Iyyer), "VERISCORE: Evaluating the factuality of verifiable claims in long-form text generation". The arXiv record shows no venue; I believe it appeared in Findings of EMNLP 2024 but did not verify this. — [arXiv 2406.19276](https://arxiv.org/abs/2406.19276)
  - Recommendation: **(i) cite.**
- **VeriFastScore** (Rishanth Rajendhran, Amir Zadeh, Matthew Sarte, Chuan Li, Mohit Iyyer), arXiv 2025 (arXiv-only per the record). It makes long-form factuality evaluation faster. — [arXiv 2505.16973](https://arxiv.org/abs/2505.16973)
  - Recommendation: **(i) cite.**
- **FactBench** (Farima Fatahi Bayat, Lechen Zhang, Sheza Munir, Lu Wang), arXiv 2024 (no venue in the record). It is a dynamic benchmark for in-the-wild LM factuality. — [arXiv 2410.22257](https://arxiv.org/abs/2410.22257)
  - Recommendation: **(i) cite** briefly. It covers open-domain (world-knowledge) factuality, not source faithfulness.
- **FENICE** (Alessandro Scirè, Karim Ghonim, Roberto Navigli), ACL 2024. It evaluates summarization factuality with NLI plus claim extraction, and code is released. — [arXiv 2403.02270](https://arxiv.org/abs/2403.02270)
  - Recommendation: **(i) cite.** As a possible **(ii) baseline**, its NLI component is English, so Hebrew would require swapping in a multilingual NLI model.
- **FIZZ** (Joonho Yang, Seunghyun Yoon, Byeongjeong Kim, Hwanhee Lee), EMNLP 2024 main. It detects factual inconsistency using atomic facts from the summary and an expanded document context. — [arXiv 2404.11184](https://arxiv.org/abs/2404.11184)
  - Recommendation: **(i) cite.**
- **A Closer Look at Claim Decomposition** (Miriam Wanner, Seth Ebner, Zhengping Jiang, Mark Dredze, Benjamin Van Durme), arXiv 2024. I believe it appeared at *SEM 2024, but the arXiv record shows no venue. — [arXiv 2403.11903](https://arxiv.org/abs/2403.11903)
  - Recommendation: **(i) cite.**
- **DnDScore** (Miriam Wanner, Benjamin Van Durme, Mark Dredze), arXiv 2024 (no venue in the record). It covers decontextualization and decomposition for long-form verification. — [arXiv 2412.13175](https://arxiv.org/abs/2412.13175)
  - Recommendation: **(i) cite.**
- **Decomposition Dilemmas** (Qisheng Hu, Quanyu Long, Wenya Wang), "Decomposition Dilemmas: Does Claim Decomposition Boost or Burden Fact-Checking Performance?", NAACL 2025 main. — [arXiv 2411.02400](https://arxiv.org/abs/2411.02400)
  - Recommendation: **(i) cite.** It supports MorphEvalHeb's choice of whole-summary classification over decomposition, since decomposition is not uniformly helpful.
- **Claimify** (Dasha Metropolitansky, Jonathan Larson), "Towards Effective Extraction and Evaluation of Factual Claims", ACL 2025 main. — [arXiv 2502.10855](https://arxiv.org/abs/2502.10855)
  - Recommendation: **(i) cite.**
- **DecMetrics** (Minghui Huang), arXiv 2025 (arXiv-only). It defines metrics for decomposition quality. — [arXiv 2509.04483](https://arxiv.org/abs/2509.04483)
  - Recommendation: optional citation.
- **Distill and Align Decomposition for Enhanced Claim Verification** (Jabez Magomere, Elena Kochkina, Samuel Mensah, Simerjot Kaur, Fernando Acero, Arturo Oncevay, Charese H. Smiley, Xiaomo Liu, et al.), EACL 2026 Findings. — [arXiv 2602.21857](https://arxiv.org/abs/2602.21857)
  - Recommendation: **(i) cite** (2026 decomposition work).
- **CREDENCE** (Phuong Huu Vu Tran, Thuan Duc Mai, Bach Xuan Le), "CREDENCE: Claim Reduction for Decomposition & Enhanced Credibility -- Semantic Metrics and Convergence Analysis", arXiv June 2026, submitted to *Language Resources and Evaluation* (not yet peer-reviewed). — [arXiv 2606.19819](https://arxiv.org/abs/2606.19819)
  - Recommendation: optional citation.
- **ElementCheck** (Xinming Wang, Haoran Du, Yi Chen, Jian Xu, Hongming Yang, Han Hu, Yulong Chen, Cheng-Lin Liu, et al.), "ElementCheck: Complexity-Aware Long-Form Text Factuality Evaluation via Sentence Elements", EMNLP 2026 Findings. It replaces uniform atomic decomposition with entity-pair "elements" and an element graph, adapting verification granularity to sentence complexity. It also introduces the FastFact-Sent benchmark. — [arXiv 2608.26118](https://arxiv.org/abs/2608.26118)
  - Recommendation: **(i) cite** (newest decomposition critique, Sep–Nov 2026 venue).
- **Relation Hallucination Index (RHI)** (Praveen Kumar Katwe, Rakesh Chandra Balabantaray, Kali Prasad Vittala, Naman Kabadi), "A Grounded and Decomposed Framework for Relation-Level Hallucination Evaluation in Abstractive Summarization", arXiv Aug 2026 (arXiv-only; 6 pages). It is a dependency-parse relation-triple metric with negation-aware verb modeling. — [arXiv 2608.08180](https://arxiv.org/abs/2608.08180)
  - Recommendation: optional citation. Its relation and negation focus overlaps with MorphEvalHeb's morphology-aware error types, but it is English dependency-based and not a practical Hebrew baseline.

#### Long-document and multi-step summary faithfulness
- **FineSurE** (Hwanjun Song, Hang Su, Igor Shalyminov, Jason Cai, Saab Mansour), "FineSurE: Fine-grained Summarization Evaluation using LLMs", ACL 2024 main. It is an LLM-based, fine-grained evaluation of faithfulness (with error types), completeness and conciseness. — [arXiv 2407.00908](https://arxiv.org/abs/2407.00908)
  - Recommendation: **(i) cite**, and **(ii) add as an LLM-judge baseline**. It is prompt-based and runs on any LLM that handles Hebrew (GPT-4o, Claude, or Gemma/Qwen). Its fact-checking error categories are a natural comparison for MorphEvalHeb's multi-label error-type detector.
- **LongDocFACTScore** (Jennifer A. Bishop, Qianqian Xie, Sophia Ananiadou), LREC-COLING 2024, pp. 10777–10789. — [arXiv 2309.12455](https://arxiv.org/abs/2309.12455)
  - Recommendation: **(i) cite** if long documents are in scope.
- **Stress Testing Factual Consistency Metrics for Long-Document Summarization** (Zain Muhammad Mujahid, Dustin Wright, Isabelle Augenstein), ACL 2026 (peer-reviewed). It tests six reference-free factuality metrics under seven factuality-preserving perturbations: paraphrase, simplification, synonym replacement, logically equivalent negation, vocabulary reduction, compression, and source insertion. Metrics give inconsistent scores for semantically equivalent summaries and degrade on information-dense claims. — [arXiv 2511.07689](https://arxiv.org/abs/2511.07689)
  - Recommendation: **(i) cite prominently.** Its perturbation methodology parallels MorphEvalHeb's contrastive synthetic data. Factuality-preserving paraphrase and negation are exactly where Hebrew morphology could break surface metrics, so this is a possible robustness analysis to add.
- **VeriTrail** (Dasha Metropolitansky, Jonathan Larson), "VeriTrail: Closed-Domain Hallucination Detection with Traceability", ICLR 2026. It detects hallucination in multi-generative-step processes (for example, hierarchical summary-of-summaries) and traces where errors were introduced. — [arXiv 2505.21786](https://arxiv.org/abs/2505.21786)
  - Recommendation: **(i) cite** for summary-of-summaries pipelines.
- **Reexamining zero-shot summarization** (Vasudha Bhatnagar, Purnima Bindal, Vikas Kumar, Raj Kumari Bahl), "Reexamining zero-shot summarization: Empirical investigation of trustworthiness of LLM-summarizers", arXiv July 2026, under journal review. — [arXiv 2607.21010](https://arxiv.org/abs/2607.21010)
  - Recommendation: optional citation; I verified only the title and metadata.

#### LLM-as-judge for factual consistency and judge reliability
- **Prometheus 2** (Seungone Kim, Juyoung Suk, Shayne Longpre, Bill Yuchen Lin, Jamin Shin, Sean Welleck, Graham Neubig, Moontae Lee, et al.), EMNLP 2024 main. — [arXiv 2405.01535](https://arxiv.org/abs/2405.01535)
  - HF `prometheus-7b-v2.0`, Apache-2.0, `en`. — [HF](https://huggingface.co/prometheus-eval/prometheus-7b-v2.0)
  - Recommendation: **(i) cite.** It is an optional open-judge baseline, but it is English-tuned, so a general multilingual instruct model is a better "vanilla LLM judge" for Hebrew.
- **FaithJudge**: see Q1 ([arXiv 2505.04847](https://arxiv.org/abs/2505.04847)). Few-shot judging grounded in human-annotated hallucination examples.
  - Recommendation: **(i) cite.** A FaithJudge-style few-shot judge using Hebrew annotated examples would be a stronger LLM-judge baseline than the vanilla judge.
- **Rating Roulette** (Rajarshi Haldar, Julia Hockenmaier), "Rating Roulette: Self-Inconsistency in LLM-As-A-Judge Frameworks", Findings of EMNLP 2025. LLM judges show low intra-rater reliability (Krippendorff's alpha) on SummEval. — [arXiv 2510.27106](https://arxiv.org/abs/2510.27106); [ACL PDF](https://aclanthology.org/2025.findings-emnlp.1361.pdf)
  - Recommendation: **(i) cite.** It motivates reporting judge variance across seeds.
- **How Reliable is Multilingual LLM-as-a-Judge?** (Xiyan Fu, Wei Liu), Findings of EMNLP 2025. Judge consistency varies significantly across languages, and the tasks include multilingual summarization. — [ACL Anthology bib](https://aclanthology.org/2025.findings-emnlp.587.bib); [PDF](https://aclanthology.org/2025.findings-emnlp.587.pdf)
  - Recommendation: **(i) cite prominently.** It directly supports MorphEvalHeb's argument against relying on LLM judges for Hebrew.
- **Diagnosing the Reliability of LLM-as-a-Judge via Item Response Theory** (Junhyuk Choi, Sohhyung Park, Chanhee Cho, Hyeonchu Park, Bugeun Kim), ICML 2026. — [arXiv 2602.00521](https://arxiv.org/abs/2602.00521)
  - Recommendation: **(i) cite**; findings not read in detail.
- **An Empirical Study of LLM-as-a-Judge: How Design Choices Impact Evaluation Reliability** (Yusuke Yamauchi, Taro Yano, Masafumi Oyamada), GEM 2026 workshop. It studies how evaluation design, decoding and chain-of-thought affect human alignment and consistency. — [ACL Anthology bib](https://aclanthology.org/2026.gem-main.19.bib)
  - Recommendation: **(i) cite** to justify the judge prompt and decoding configuration.
- **A Judge Should Know What Changed: Construct Validity for LLM-as-a-Judge Evaluation** (Jianlin Chen, Wenhui Chen, Ziyao Lin, Chi Man Vong), arXiv Aug 2026 (arXiv-only). — [arXiv 2608.24419](https://arxiv.org/abs/2608.24419)
  - Recommendation: optional citation; I verified only the metadata.
- **HalluLens** (Yejin Bang, Ziwei Ji, Alan Schelten, Anthony Hartshorn, Tara Fowler, Cheng Zhang, Nicola Cancedda, Pascale Fung), "HalluLens: LLM Hallucination Benchmark", arXiv 2025. The record shows no venue; it is reportedly at ACL 2025, but I did not verify that. — [arXiv 2504.17550](https://arxiv.org/abs/2504.17550)
  - Recommendation: **(i) optional cite** for the hallucination taxonomy (extrinsic vs. intrinsic).
- **Excluded:** "Reliability without Validity" (arXiv 2606.19544) **was withdrawn by arXiv** ("does not meet arXiv's research content quality standards"). Do not cite it. — [arXiv 2606.19544](https://arxiv.org/abs/2606.19544)

### Inferences
- **Highest-value new citations:**
  1. FactCG, plus Bespoke-MiniCheck and the LLM-AggreFact SOTA statement;
  2. LettuceDetect v2 and PsiloQA as multilingual encoder detectors;
  3. Stress Testing (ACL 2026) for perturbation-based metric robustness;
  4. How Reliable is Multilingual LLM-as-a-Judge? (EMNLP-F 2025) for multilingual judge inconsistency;
  5. FineSurE for LLM-based error-typed evaluation;
  6. TrueTeacher and FaithLens for synthetic-data-trained detectors;
  7. Decomposition Dilemmas and ElementCheck to justify not decomposing.
- **Most feasible new baselines for Hebrew:**
  - FactCG-DeBERTa-L (zero-shot and MT-to-English);
  - lettucedect-v2-mmbert-base (multilingual encoder; aggregate spans to a summary score);
  - Bespoke-MiniCheck-7B or Granite Guardian 3.3 (zero-shot LLM checker);
  - FineSurE prompts on a Hebrew-capable LLM (an error-typed judge).
- Few items explicitly from May–Nov 2026 are peer-reviewed: ACL 2026 (Stress Testing; FaithLens in Findings), ICLR 2026 (VeriTrail), ICML 2026 (IRT judge reliability), EMNLP 2026 Findings (ElementCheck), the GroundLM 2026 workshop (LettuceDetect v2; How Far Without a GPU), GEM 2026, and RESOURCEFUL 2026.

### Gaps
- Some venues are inferred rather than verified, because the arXiv record shows none: VeriScore (believed EMNLP Findings 2024), FaithBench (believed NAACL 2025), A Closer Look at Claim Decomposition (*SEM 2024), and HalluLens (ACL 2025). Check the ACL Anthology before the camera-ready.
- Granite Guardian, Lynx, LettuceDetect v1, PsiloQA, VeriFastScore and FactBench show no peer-reviewed venue in their arXiv records. Cite them as arXiv unless the Anthology shows otherwise.
- I found no 2026 summarization-specific LLM-judge factual-consistency study beyond the general judge-reliability papers above.
- I did not search exhaustively for EMNLP 2026 main-conference factuality papers. The ACL Anthology listing for EMNLP 2026 may not be complete as of 2026-11-12.
