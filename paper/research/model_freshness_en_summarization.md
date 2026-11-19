# Generator freshness in English summarization factual-consistency metric and benchmark papers

Produced 2026-11-18 by a research subagent. Scope: English summarization faithfulness and factual-consistency
metrics and benchmarks, plus the closely related claim-verification and hallucination papers we already cite.

**How this was checked.** Each paper's arXiv PDF was downloaded and its text searched for the
data-construction sections (37 papers). Model lists marked **(v)** were read in the paper text. Marked
**(mem)**: venue or release dates from memory, not re-checked. Marked **(unverified)**: details the paper
does not state, or that live in a cited source paper we did not open. Release years are each model's
first public release (mem). For papers with an arXiv preprint well before the venue version, the age gap is
given against both.

**Key distinction.**
- **Gen** = the models that produced the summaries or claims that humans labelled, i.e. the data the metric is
  meta-evaluated on.
- **Judge** = the LLMs used as the metric, evaluator or teacher.

Many 2023–2025 papers use frontier LLMs as **judges** but meta-evaluate on **Gen** data from 2017–2020 models.

## Table

Abbreviations:
- **Gen year** = newest evaluated generator.
- **Gap** = paper year − Gen year.
- **Frontier?** = whether the strongest generator of the time is among the **Gen** models.
- **By-gen** = whether metric quality is reported per generator or per model era.
- **SummaC-bench** = CoGenSumm, XSumFaith, Polytope, FactCC, SummEval, FRANK. Their summaries come from
  2017–2020 models: PtGen 2017, BottomUp 2018, TConvS2S 2018, TranS2S, BertSum or BERTS2S 2019, and in
  SummEval and FRANK also BART, PEGASUS and T5 from 2019–2020. (mem, consistent with Tang 2023 Table 2)

### A. Papers we already cite

| # | Paper | Venue, year | Proposes | Gen: summary generators (year, size, open/closed) | Gen year | Gap | Frontier? | Sizes | By-gen? | Data source | Judge or teacher LLMs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | FactCC (Kryscinski et al.) | EMNLP 2020 (mem) | metric + small test set | CNN/DM "summaries output by state-of-the-art summarization models" (v). The paper does not name them. Tang 2023 classes them as OLD (PtGen-era, 2017–18) | ~2018 | ~2 | no (BART and PEGASUS absent) | <0.5B, open | no | new annotation of existing system outputs | none |
| 2 | Maynez et al. | ACL 2020 | benchmark (XSumFaith) + analysis | PtGen 2017, TConvS2S 2018, TranS2S 2018, BERTS2S 2019, GPT-Tuned (GPT-2 2019) (v). All <0.5B, open | 2019 | 1 | yes for its time (BERTS2S); BART was very new | 0.1–0.4B | yes, error rates per system; no metric comparison by system | fresh outputs on XSum | — |
| 3 | FRANK (Pagnoni et al.) | NAACL 2021 | benchmark + typology | CNN/DM: S2S, PGN 2017, BUS 2018, BertSum 2019, BART 2019/20. XSum: TConvS2S, PGN, TransS2S, BERTS2S (v) | 2020 | 1 | yes (BART) | ≤0.4B, open | per-system error proportions (Fig. 2); metric correlations are mostly pooled | fresh outputs | — |
| 4 | Goyal & Durrett | NAACL 2021 | annotations + DAE metric | XSumFaith systems, plus new annotation of BART and PEGASUS XSum outputs (v) | 2020 | 1 | yes | ≤0.6B, open | notes that errors shift with newer systems (quote below) | fresh + reused | — |
| 5 | CLIFF (Cao & Wang) | EMNLP 2021 | training method + error annotation | BART, PEGASUS fine-tuned (v) | 2020 | 1 | yes | ~0.4–0.6B, open | per model | fresh | — |
| 6 | SummaC (Laban et al.) | TACL 2022 | metric + SummaC-bench | SummaC-bench (reused) | 2020 | 2 | no | ≤0.6B | no (pooled) | reused 2017–2020 data | — |
| 7 | QAFactEval (Fabbri et al.) | NAACL 2022 | metric | SummaC-bench (v) | 2020 | 2 | no | ≤0.6B | no | reused | — |
| 8 | TRUE (Honovich et al.) | NAACL 2022 | benchmark of benchmarks | Summarization part: FRANK, SummEval, MNBM (XSumFaith), QAGS-C/X (v). Also dialogue, paraphrase and fact-verification sets | 2020 | 2 | no | ≤0.6B | no | reused | T5-11B NLI, etc. |
| 9 | Tang et al. (AggreFact) | ACL 2023 (arXiv May 2022) | benchmark (AggreFact) + analysis | 9 datasets stratified as OLD (PtGen, BottomUp, ...), EXFORMER (BertSum, GPT-2, ...) and FTSOTA (BART, PEGASUS, T5) (v) | 2020 | 3 (2 vs arXiv) | no LLM summaries; FtSota were the strongest fine-tuned models | ≤0.6B (T5 up to 11B possible, unverified) | **yes, the central finding** (quotes below) | reused | ChatGPT as a metric (v) |
| 10 | AlignScore (Zha et al.) | ACL 2023 | metric | SummaC-bench + TRUE (v) | 2020 | 3 | no | ≤0.6B | no | reused | — |
| 11 | TrueTeacher (Gekhman et al.) | EMNLP 2023 | synthetic data + student metric | Test: TRUE summarization sets (2017–2020). Training: T5 summarizers of several sizes that the authors fine-tuned (v) | 2020 (test) | 3 | no | test ≤0.6B; train T5 small to 11B | no | reused (test) | FLAN-PaLM 540B teacher (v) |
| 12 | G-Eval (Liu et al.) | EMNLP 2023 | LLM-judge metric | SummEval (16 systems, 2017–2020) and QAGS (v). Bias analysis on Zhang et al. human vs GPT-3.5 (text-davinci-003) summaries (v) | 2020 (main); 2022 (bias study) | 3 / 1 | only in the bias side study | ≤0.6B; 175B (bias) | **yes**, judge favours LLM summaries (quote below) | reused | GPT-3.5, GPT-4 |
| 13 | AMRFact (Qiu et al.) | NAACL 2024 (mem) | negative-data method + metric | AggreFact-FtSota (BART, PEGASUS, T5) (v) | 2020 | 4 | no | ≤0.6B | no | reused | ChatGPT, GPT-4 baselines |
| 14 | FineSurE (Song et al.) | ACL 2024 | LLM-judge metric | Meta-eval: FRANK (9 systems) and REALSumm (25 systems, CNNDM ~2017–2020) (v). Separately, 8 LLMs (Phi-2, Llama-2/3-70B, Mixtral, Gemini-1-pro, GPT-3.5-turbo, GPT-4-turbo, GPT-4o) summarized 100 CNNDM docs, scored only by FineSurE, with **no human labels** (v) | 2020 (meta-eval) | 4 | **no** in meta-eval; GPT-4o only as an unvalidated application | ≤0.6B (meta-eval) | no | reused | GPT-4-turbo default; also GPT-4o, Llama-3-70B, Mixtral, Phi-2 |
| 15 | MiniCheck / LLM-AggreFact (Tang et al.) | EMNLP 2024 | metric + benchmark | Summarization subsets: AggreFact-FtSota (BART/PEGASUS/T5, 2020) and TofuEval (GPT-3.5-Turbo, Vicuna-7B, WizardLM-7/13/30B, 2023). Other subsets: LLM or search-engine responses (ExpertQA, LFQA, ClaimVerify, FactCheck-GPT, Reveal) and human claims (WiCE) (v, generically described) | 2023 | 1 (4 for the AggreFact part) | **no GPT-4 summaries** in the summarization subsets | 0.4B–30B open + GPT-3.5 | per dataset only | reused | Gemini-Pro, PaLM2, Mixtral, Mistral-Large, Claude 2.1, Claude 3 Opus, GPT-3.5, GPT-4 (v) |
| 16 | FAVA (Mishra et al.) | COLM 2024 (v) | detector + FavaBench | ChatGPT (2022/23, closed), Llama2-Chat 7B and 70B (2023, open) (v). Information-seeking, **not summarization** | 2023 | 1 | partly (no GPT-4 generator) | 7B–70B + ChatGPT | error distribution per LM (Table 1) | fresh | ChatGPT, GPT-4 baselines |
| 17 | RAGTruth (Niu et al.) | ACL 2024 (mem) | corpus + detector | GPT-3.5-turbo-0613, GPT-4-0613, Mistral-7B-Instruct, Llama-2-7B/13B/70B-chat (v). Tasks: QA, data-to-text and summarization (CNN/DM + recent news) | 2023 | 1 | **yes** (GPT-4) | 7B–70B + closed | hallucination rate per generator; detection reported per task, not per generator (v) | fresh | GPT-3.5/4-turbo prompts, fine-tuned Llama-2-13B |
| 18 | Wanner et al. (claim decomposition) | *SEM 2024 (mem) | analysis of decomposition | FActScore biographies from 12 LMs (InstructGPT, ChatGPT, GPT-4, Alpaca, Vicuna, ...; 2022–23) (v: "12 LMs", reused from Min et al.). Not summarization | 2023 | 1 | yes (GPT-4) | 7B–175B+ | partly | reused | various |
| 19 | FactCG (Lei et al.) | NAACL 2025 | metric + synthetic data | LLM-AggreFact incl. RAGTruth (v) → generators 2019–2023 | 2023 | 2 | GPT-4 (2023) yes; nothing from 2024 | ≤70B + closed | no | reused | GPT-4o baseline |
| 20 | Hu et al. (decomposition dilemmas) | NAACL 2025 (mem) | analysis | WiCE (Wikipedia, human), ClaimDecomp (human), BingChat responses (2023) (v). Not summarization | 2023 | 2 | no | — | no | reused | GPT-4o-mini decomposer, Llama-3.1-8B |
| 21 | LLM-Oasis (Scirè et al.) | Computational Linguistics 52(1), 2026 (arXiv Nov 2024) | resource + benchmark | GPT-4 API rewrites Wikipedia passages into factual and unfactual texts (v). Synthetic, not natural summaries | 2023 | 3 (1 vs arXiv) | GPT-4 yes (as data generator) | — | no | fresh synthetic | GPT-4o, Llama-3.3-70B, Phi-4, Qwen2.5-7B, Mistral-7B, Llama-3.1-8B, ... (v) |
| 22 | Mujahid et al. (stress testing, long doc) | ACL 2026 (arXiv Nov 2025) | robustness study | **No model summaries**: SQuALITY, LexAbSumm and ScholarQABench summaries are all "Human-written" (v, Table 2). The 7 meaning-preserving perturbations are generated by GPT-4o (v) | n/a (GPT-4o 2024 as perturber) | n/a (2) | n/a | — | no | reused human references + GPT-4o edits | metrics: BARTScore, SummaC-ZS/Conv, AlignScore, UniEval, Bespoke-MiniCheck-7B |

### B. Additional 2023–2025 papers found by search

| # | Paper | Venue, year | Proposes | Gen: summary generators | Gen year | Gap | Frontier? | Sizes | By-gen? | Data source | Judge |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 23 | SummEdits (Laban et al., arXiv 2305.14540) | EMNLP 2023 (mem) | benchmark (10 domains) | Seed summaries (human or model), edited by **GPT-3.5-turbo** (v) | 2023 | 0 | edits only | — | judge-side: "larger or more recent LLMs perform better" (v) | fresh synthetic edits | ~20 LLMs incl. GPT-4, PaLM2, Claude v1.3 (v) |
| 24 | TofuEval (Tang et al., arXiv 2402.13249) | NAACL 2024 (mem) | benchmark (dialogue, topic-focused) | GPT-3.5-Turbo (closed), Vicuna-7B, WizardLM-7B/13B/30B (open, 2023) (v) | 2023 | 1 | **no GPT-4 summarizer** (GPT-4 only as judge) | 7–30B + GPT-3.5 | error rates per summarizer (Table 2); motivation quote below | fresh | GPT-4, GPT-3.5, Vicuna-13/33B, WizardLM-13/30B (v) |
| 25 | InstruSum (Liu et al., arXiv 2311.09184) | NAACL Findings 2024 (mem) | benchmark + LLM-judge study | text-davinci-002, text-davinci-003, gpt-3.5-turbo-0301, gpt-4-0314, hybrid LLM-human (v) | 2023 | 1 (0 vs arXiv) | **yes** (GPT-4) | closed | per-system quality; 40 LLM-eval methods | fresh | GPT-4, Llama-2, Mistral, later also Claude-3, GPT-4o, etc. (v) |
| 26 | Zhang et al., "Benchmarking LLMs for news summarization" (arXiv 2301.13848) | TACL 2024 | human-eval benchmark (generation, not a metric) | GPT-3 davinci/curie/ada, InstructGPT davinci/curie/ada (v), plus OPT, GLM, Cohere, Anthropic-LM (mem) and BRIO/PEGASUS baselines, plus freelance writers (v) | 2022 | 2 (1 vs arXiv) | yes (InstructGPT davinci, 175B) | 350M–175B | per system | fresh | — |
| 27 | FENICE (Scirè et al., arXiv 2403.02270) | ACL Findings 2024 (mem) | metric + long-form annotations | AggreFact-FtSota, plus new annotations of book summaries from BART, LongT5 and LED fine-tuned on Echoes (12 systems) (v) | ~2022 | 2 | no | ≤0.6B | no | reused + fresh old-model outputs | T5 claim extractor; ChatGPT baselines |
| 28 | FIZZ (Yang et al., arXiv 2404.11184) | EMNLP 2024 (mem) | metric | AggreFact-FtSota only (v) | 2020 | 4 | no | ≤0.6B | no | reused | Orca-2 and other LLMs for atomic facts (v) |
| 29 | LongDocFACTScore (Bishop et al., arXiv 2309.12455) | LREC-COLING 2024 | metric + LongSciVerify | LongSciVerify: three methods incl. DANCER, an LED-based zoning model and extractive-abstractive LED (v). LongEval PubMed: LongT5-large, BigBird-PEGASUS (v) | ~2022 | 2 | **no** (paper says GPT-4 is "flawed", but no GPT-4 outputs are evaluated) | ≤0.8B | no | fresh old-model + reused | — |
| 30 | FABLES (Kim et al., arXiv 2404.01261) | COLM 2024 (v) | benchmark (book-length) | GPT-3.5-turbo, GPT-4, GPT-4-turbo, Claude-3-Opus (closed), Mixtral (open) (v) | 2024 | 0 | **yes** (Claude-3-Opus, GPT-4-turbo) | closed + 47B MoE | per summarizer (faithfulness rates) | fresh, books published 2023–24 | GPT-4, Claude-3-Opus auto-raters |
| 31 | FactPICO (Joseph et al., arXiv 2402.11456) | ACL 2024 (mem) | benchmark (medical plain-language) | GPT-4, Llama-2-Chat, Alpaca (v) | 2023 | 1 | yes (GPT-4) | 7B(+) + GPT-4 | metric scores per LLM (Table 5) | fresh | LLM-based and classic metrics |
| 32 | Ramprasad et al. (arXiv 2406.03487) | ACL Findings 2024 (mem) | annotation + detection study (dialogue) | GPT-4 (gpt-4-32k-0613), Alpaca-13B, compared with RefMatters fine-tuned models (v) | 2023 | 1 | yes | 13B + GPT-4; ≤0.6B | **yes**, LLM vs older fine-tuned outputs (quote below) | fresh + reused | GPT-4 prompt-based detectors |
| 33 | UniSumEval (Lee et al., arXiv 2409.19898) | EMNLP Findings 2024 (v) | benchmark (9 domains) | BART-Large, Flan-T5-Large (<0.7B); Phi-2, Llama2-13B-Chat, Mistral-7B-Instruct, Mixtral-8x7B (open); GPT-3.5-turbo, GPT-4-turbo, Claude 2.1 (closed) (v) | 2023 | 1 | yes for late 2023 (GPT-4-turbo); not Claude 3 or GPT-4o | 0.4B–47B + closed | **yes**, evidence that evaluation is harder for LLM summaries (quote below) | fresh | UniEval, QAFactEval, SummaC-Conv, AlignScore, MiniCheck, G-Eval, FActScore (v) |
| 34 | FaithBench (Bao et al., arXiv 2410.13210) | NAACL 2025 (short) | benchmark (hard cases) | GPT-4o, GPT-3.5-Turbo, Llama-3.1-70B, Llama-3.1-8B, Gemini-1.5-Flash, Claude-3.5-Sonnet, Qwen2.5-7B, Phi-3-mini-4k, Command-R, Mistral-7B (v). Authors picked "the smallest version in its latest generation" of each family (v) | 2024 | 1 (0 vs arXiv) | **yes** (GPT-4o, Claude-3.5-Sonnet) | 3.8B–70B + closed | hallucination rate per LLM; detectors pooled | fresh (Vectara leaderboard outputs) | HHEM, GPT-4o-as-judge, etc. |
| 35 | Wan et al., positional bias (arXiv 2410.23609) | NAACL 2025 | meta-eval benchmark (LongFormFact) + analysis | LongFormFact = 8 human-annotated long-form sets reused from MultiNews, QMSum, ArXiv, GovReport, ChemSumm, PubMed, SQuALITY and DiverseSumm (v). Their generators are mostly 2020–2023 fine-tuned or early LLMs (unverified per set). New GPT-4o, Llama-3.1, Mixtral, Gemini-1.5-Pro and Claude-3 summaries are **scored by metrics, not humans** (v) | ~2023 (meta-eval) | ~2 | no in meta-eval | mixed | per dataset | reused (meta-eval) | GPT-4o, MiniCheck |
| 36 | MSumBench (Min et al., arXiv 2506.00549) | ACL 2025 | benchmark (EN+ZH, 6 domains) | BART-Large, mT5 (non-LLM); Llama-3.1-70B, Gemma-2-27B, Qwen-2.5-72B (open); GPT-4o, Claude-3.5-Sonnet, Gemini-1.5-pro (closed) (v) | 2024 | 1 | **yes** | 0.4B–72B + closed | per summarizer; judge self-bias per LLM (quote below) | fresh | FineSurE, G-Eval, etc. |
| 37 | VeriGray (Ding et al., arXiv 2510.21118) | arXiv 2025 (venue unverified) | benchmark (ambiguity-aware) | FaithBench's 10 LLM summaries, **plus GPT-5 (2025), DeepSeek-V3-0324 (2025), Qwen3-8B (2025)** (v) | 2025 | 0 | **yes** (GPT-5) | 3.8B–671B MoE + closed | unfaithful rate per generator (Claude-3.5-Sonnet 3%, GPT-5 6%, Qwen3-8B 32%) (v) | fresh + reused | LLM judges, GPT-5 + RAG |

## Patterns

**1. Most metric papers do not evaluate on fresh generators.**
- Every metric paper from SummaC (2022) through AMRFact, FIZZ and FineSurE (2024) meta-evaluates on SummaC-bench,
  TRUE, FRANK, SummEval or AggreFact-FtSota. The newest generators there are BART, PEGASUS and T5 (2019–2020).
- The typical gap is therefore **2–4 years**, and it grows with time because the data stays fixed. Examples:
  - FineSurE (ACL 2024) uses GPT-4-turbo as the judge but is validated on FRANK and REALSumm (gap 4).
  - FIZZ and AMRFact (2024) use AggreFact-FtSota (gap 4).
  - FactCG (2025) uses LLM-AggreFact, whose newest generators are from 2023 (gap 2).
- Papers that are new metrics (not new benchmarks) **rarely include the frontier closed model as a generator**.
  MiniCheck's summarization subsets include GPT-3.5 but no GPT-4 summaries.
- Fresh data appears almost only in papers whose main contribution is a **new benchmark**. Their gap is typically
  **0–1 years**, and from 2024 they usually include closed frontier models: RAGTruth, FABLES, UniSumEval,
  FaithBench, MSumBench, VeriGray, InstruSum, FactPICO.
- One clear exception is the metric paper that ships with a benchmark. **FineSurE** and **Wan et al. 2025**
  generate GPT-4o and Claude summaries but score them **only with their own metric**. Neither collects human
  labels on those summaries.

**2. Counts (37 rows).**
- About 16 rows meta-evaluate only on generators from 2020 or earlier.
- About 9 rows include a closed frontier generator of their time in the human-labelled data: RAGTruth,
  InstruSum, FABLES, FactPICO, Ramprasad, UniSumEval, FaithBench, MSumBench, VeriGray. Wanner and LLM-Oasis use
  GPT-4 for non-summary data.
- **Sizes:** pre-2023 evaluation data is almost entirely 0.1–0.6B fine-tuned encoder-decoders. Post-2023 benchmarks
  span about 3.8B–72B open models plus closed APIs. Several deliberately keep small or "smallest-in-family"
  models: FaithBench, and UniSumEval's Phi-2, Flan-T5 and BART.
- **Judges are much fresher than generators:** GPT-4, GPT-4o and Claude-3 are routinely the metric, while the
  summaries they judge come from 2019 models.

**3. Evidence that metric quality depends on generator age or strength.**
- **Tang et al. 2023 (AggreFact)** ([arXiv 2205.12854](https://arxiv.org/abs/2205.12854), ACL 2023):
  - "much of the recent improvement in the factuality detection space has been on summaries from older
    (pre-Transformer) models instead of more relevant recent summarization models."
  - "There is a significant performance drop when evaluating the CNN/DM summaries generated by models from
    EXFORMER or FTSOTA instead. Approximately a 10% balanced accuracy decrease on average occurs from OLD to
    FTSOTA."
  - "metrics which claim improved performance on SUMMAC do not show similar gains when evaluated on FTSOTA
    summaries. We recommend that future work similarly focuses on the SOTA category."
  - Reason given: OLD summaries "contain obvious errors, such as repetition, that can be more easily detected
    compared to more nuanced errors made by more recent models."
  - XSum results are "more mixed". (all v)
- **UniSumEval** ([arXiv 2409.19898](https://arxiv.org/abs/2409.19898), App. F.1): "human annotators identify
  the highest frequency of errors for summaries generated by the proprietary LLMs, followed by those generated
  by the open-source LLMs, and the least for those generated by the non-LLMs. This trend highlights that
  automated faithfulness evaluation is more challenging for summaries generated by recent LLMs compared to those
  generated by non-LLMs ... factual errors in LLM-generated summaries are more complex and nuanced." (v)
  - Caveat: this measures human corrections of the LLM pre-annotator's labels, not a standard metric's accuracy.
  - The same paper: "contrary to prior findings that NLI-based evaluators perform well (Tang et al., 2024b), they
    show much lower agreement compared to G-Eval when evaluating faithfulness on our hallucinogenic texts." (v)
- **TofuEval** ([arXiv 2402.13249](https://arxiv.org/abs/2402.13249)): "As hallucinations from more recent
  models are harder to detect (Tang et al., 2023a), we re-evaluate non-LLM-based and LLM-based factuality
  metrics using LLM-generated summaries." (v)
- **Ramprasad et al. 2024** ([arXiv 2406.03487](https://arxiv.org/abs/2406.03487)):
  - LLMs "generate plausible inferences, supported by circumstantial evidence ... a pattern that is less
    prevalent in older models."
  - "we systematically assess the efficacy of automatic error detection methods on LLM summaries and find that
    they struggle to detect these nuanced errors." (v)
- **Goyal & Durrett 2021** ([arXiv 2104.04302](https://arxiv.org/abs/2104.04302)): extrinsic hallucinations
  "are even less common in recent systems like PEGASUS", and "synthetic datasets do not reflect the error
  distributions of" real models. (v) Error distributions move with generator generation.
- **FaithBench** ([arXiv 2410.13210](https://arxiv.org/abs/2410.13210)): on summaries from 10 modern LLMs, "even
  the best hallucination detection models have near 50% accuracies" (best balanced accuracy 58%). (v)
- **G-Eval** ([arXiv 2303.16634](https://arxiv.org/abs/2303.16634)): "G-EVAL-4 always gives higher scores to
  GPT-3.5 summaries than human-written summaries, even when human judges prefer human-written summaries." (v)
  - This is a generator-dependent judge bias.
  - MSumBench ([arXiv 2506.00549](https://arxiv.org/abs/2506.00549)) found the self-bias is model-specific: "Llama-3.1-70B rates its own summaries
    less favorably than others, while Claude-3.5-Sonnet favors its own." (v)
- **VeriGray** ([arXiv 2510.21118](https://arxiv.org/abs/2510.21118)) explicitly criticizes stale benchmarks:
  Cao & Wang "relied on old summarization models such as BART and PEGASUS, and their insights have not been
  widely adopted." (v)
  - It also finds GPT-5 is not the most faithful: Claude-3.5-Sonnet has 3% unfaithful sentences against GPT-5's
    6%. Newer is not always more faithful.
- **Judge side:** SummEdits finds "Larger or more recent LLMs perform better on the benchmark ... each model
  generation leading to an improvement" ([arXiv 2305.14540](https://arxiv.org/abs/2305.14540)). (v)

**4. Implication for MorphEvalHeb.**
- The norm for *metric* papers is to evaluate on outputs 2–4 years older than the paper. Benchmark papers since
  2024 increasingly use generators released 0–1 years before the paper, including closed frontier models.
- Since AggreFact, reviewers can point to Tang 2023 and UniSumEval: metric accuracy measured on weaker or older
  generators overstates accuracy on current ones.
- Reporting results per generator or per model era, and including at least one current closed model with human
  labels, would place MorphEvalHeb with FaithBench, UniSumEval and MSumBench rather than with the
  SummaC/AlignScore/FineSurE practice.

## Caveats
- Sizes of the closed models are not public. The T5 sizes inside AggreFact FtSota are unverified.
- LLM-AggreFact has grown over time; the leaderboard now has more datasets, including RAGTruth. The MiniCheck row
  describes the paper version. MiniCheck says TofuEval has "6 LLMs", while TofuEval itself lists five
  summarizers (v). This discrepancy is unresolved.
- The component generators of Wan et al.'s LongFormFact and of FactCC's test set were not opened. They are
  marked unverified.
