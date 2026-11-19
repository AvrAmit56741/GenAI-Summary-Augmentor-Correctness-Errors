# Verification of generator freshness: long-form factuality, RAG, LLM-as-judge / RMs, dialogue/D2T, plus other NLG eval tasks

Compiled 2026-11-18. Verifies `paper/research/model_freshness_qa_rag_judges.md` (the "first pass") against primary sources (arXiv abs/HTML, ACL Anthology, ICLR proceedings). Age gap = paper year minus the release year of the newest model whose OUTPUTS are evaluated. Models used only as metric or judge are listed separately. Generator release dates are general public knowledge (approximate month) and were not re-fetched; they are marked (release: mem).

Caveat on quotes: the fetch tool returns short verbatim spans, typically 125 characters or fewer. Sentences marked "verbatim" were returned as quotations by the tool. Longer passages are paraphrases.

## Q1. Rows tagged (mem)/(unverified) in the first pass: venues, years, generators

### Takeaway
Most first-pass facts hold up. Confirmed venues: SAFE (NeurIPS 2024), MiniCheck (EMNLP 2024), JudgeLM (ICLR 2025), JudgeBench (ICLR 2025), RewardBench (Findings of NAACL 2025), Dorner et al. (ICLR 2025), and Shelf-Life (ICLR 2026, not "venue unverified"). FActScore's human-annotated generators are confirmed as InstructGPT (text-davinci-003), ChatGPT and PerplexityAI. JudgeLM's 11 generator models are NOT enumerated in the paper. LLM-AggreFact per-dataset generator names remain UNCONFIRMED.

### Cited Findings
- **FActScore.** The human-annotated generators are InstructGPT (text-davinci-003), ChatGPT and PerplexityAI; no version is given for ChatGPT or PerplexityAI. The estimator case study covers GPT-4, Alpaca 7B/13B/65B, Vicuna 7B/13B, Dolly 12B, Oasst-pythia 12B, StableLM-tuned-alpha 7B, MPT-Chat 7B, InstructGPT and ChatGPT, plus human DBpedia bios. — [arXiv 2305.14251 HTML](https://arxiv.org/html/2305.14251)
- **FActScore, estimator error by model.** This is a correction to the first pass, which said "(mem) error differs by model". The paper says the best estimator variant depends on the evaluated LM: LLaMA+NP was best for InstructGPT and ChatGPT, and ChatGPT was best for PerplexityAI. Limitations section: "the best variant depends on the degree of how close a generation is to human-written text and its linguistic complexity". — [arXiv 2305.14251 HTML](https://arxiv.org/html/2305.14251)
- **FActScore venue.** The arXiv page (v2, 11 Oct 2023) does not state a venue. EMNLP 2023 is UNCONFIRMED in this pass (from memory only).
- **SAFE/LongFact venue.** NeurIPS 2024, confirmed by the NeurIPS virtual poster page. — [NeurIPS 2024 poster](https://nips.cc/virtual/2024/poster/96675); [mlanthology](https://mlanthology.org/neurips/2024/wei2024neurips-longform)
- **MiniCheck / LLM-AggreFact venue.** EMNLP 2024 main, 2024.emnlp-main.499. — [ACL Anthology](https://aclanthology.org/2024.emnlp-main.499)
- **LLM-AggreFact, generators per dataset.** Only this statement is verified: "The claims to be verified are mostly generated from recent generative models". The per-dataset descriptions are in Appendix C.1 and were not retrieved. Specific model names (Bing Chat/Perplexity/YouChat/NeevaAI for ClaimVerify, ChatGPT for FactCheck-GPT, GPT-4-based systems for ExpertQA) remain UNCONFIRMED. — [arXiv 2404.10774 HTML](https://arxiv.org/html/2404.10774)
- **JudgeLM.** The paper says only "answers are collated from 11 leading open-source LLMs including, but not limited to, LLaMA" (with Alpaca and Vicuna). Appendix A.3 mentions Vicuna-13B, LLaMA-7B and Alpaca-7B for a separate experiment. The full list of 11 is not in the paper, so it is UNCONFIRMED. — [arXiv 2310.17631 HTML](https://arxiv.org/html/2310.17631)
- **JudgeLM venue.** ICLR 2025 (proceedings). Spotlight status rests only on the authors' README. — [ICLR 2025 proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/7f8f73134e253845a8f82983219a8452-Abstract-Conference.html)
- **JudgeBench venue.** ICLR 2025. Authors: Tan, Zhuang, Montgomery, Tang, Cuadron, Wang, Popa, Stoica. — [ICLR 2025 proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/9e720fce64f91114c49cfd640d821da3-Abstract-Conference.html)
- **RewardBench venue.** Findings of NAACL 2025, 2025.findings-naacl.96, pp. 1755–1797. — [ACL Anthology](https://aclanthology.org/2025.findings-naacl.96)
- **RewardBench completions** (Table 1). AlpacaEval-Easy: GPT-4-Turbo vs Alpaca 7B. AlpacaEval-Length: Llama-2-Chat-70B vs Guanaco-13B. AlpacaEval-Hard: Tulu-2-DPO-70B vs davinci-003. The MT-Bench subsets do not name their generators. These details are confirmed. — [arXiv 2403.13787v2 HTML](https://arxiv.org/html/2403.13787v2)
- **RewardBench 2 completion pool** (Appendix H, Table 7), confirmed by subset:
  - Factuality: Llama-3.1-8B/70B, Qwen2.5-7B/72B, Tulu-3-8B/70B, Mistral-7B-v0.3, claude-3-5-sonnet-20241022, gpt-4o-2024-08-06.
  - Math: adds Qwen2.5-Math-7B/72B and deepseek-math-7b-rl.
  - Safety: adds Llama-2-7b-chat, Llama-3.2-1B, OLMoE-1B-7B, dolphin-2.0-mistral-7b, Qwen2-0.5B, Qwen2.5-14B, tulu-2-dpo-70b and zephyr-7b.
  - No o1/o3/R1/Claude-3.7 appears as a generator. Claude 3.7 Sonnet appears only as a judge.
  - The venue is not stated on arXiv and remains UNCONFIRMED.
  - Source: [arXiv 2506.01937 HTML](https://arxiv.org/html/2506.01937)
- **FACTS Grounding generators.** Gemini 1.5 Pro/Flash, Gemini 2.0 Flash Exp, GPT-4o, o1-preview, o1-mini, Claude 3.5 Haiku/Sonnet. GPT-4o-mini appears only in Tables 5–6. arXiv v1 is dated 6 Jan 2025, with no venue stated, so the venue is UNCONFIRMED. — [arXiv 2501.03200 HTML](https://arxiv.org/html/2501.03200)
- **FactBench.** The models are GPT-4o, Gemini 1.5 Pro, Llama 3.1 70B and 405B. The appendix leaderboard adds Claude 3.5 Sonnet, Command R+ and Mistral Large 2. arXiv v2 is dated 8 Jan 2025, with no venue stated (UNCONFIRMED). — [arXiv 2410.22257 HTML](https://arxiv.org/html/2410.22257)
- **HalluLens.** It evaluates 13 models, as listed in the first pass. The closed models are Claude-3-haiku (2024-02-29), Claude-3-sonnet (2024-03-07) and GPT-4o (2024-08-06). arXiv v1 is dated 24 Apr 2025. ACL 2025 is UNCONFIRMED because the arXiv page states no venue. — [arXiv 2504.17550 HTML](https://arxiv.org/html/2504.17550)
- **MT-Bench expert-judgment generators.** GPT-4, GPT-3.5, Claude-v1, Vicuna-13B, Alpaca-13B, LLaMA-13B. Confirmed. — [arXiv 2306.05685 HTML](https://arxiv.org/html/2306.05685)
- **Not re-verified in this pass** (the first-pass tags stand): ARES (NAACL 2024), RAGTruth (ACL 2024), VeriScore (Findings of EMNLP 2024), Prometheus 2 (EMNLP 2024), FaithEval (ICLR 2025), TIGERScore (TMLR 2024), Kasner & Dušek (ACL 2024).

### Inferences
- The first pass should upgrade Shelf-Life to "ICLR 2026", JudgeLM and JudgeBench to "ICLR 2025 (v)", RewardBench to "Findings NAACL 2025 (v)", and SAFE to "NeurIPS 2024 (v)". JudgeLM's gap is confirmed at about 1.5 years from arXiv to venue (Oct 2023 to ICLR 2025). Its generators are early-2023 open models, but they cannot be listed exhaustively.
- The FActScore wording should be "the best estimator variant depends on the evaluated LM", not "error differs by model". This is still evidence that metric behaviour depends on the generator.

### Gaps
- The LLM-AggreFact per-dataset generator names need Appendix C.1 (arXiv HTML offset >100k), which was not read.
- JudgeLM's 11 models: neither the paper nor the HTML enumerates them. The JudgeLM GitHub data card might.
- The venues of RewardBench 2, FACTS Grounding, FactBench, HalluLens and No Free Labels could not be confirmed from arXiv pages.

## Q2. Exact verification of the key judge/staleness claims

### Takeaway
All headline claims are confirmed by the primary sources, with minor corrections. Shelf-Life is an ICLR 2026 paper by Singh, Xu, Zhou, Zhou, Hakkani-Tür and Joty; "FutureProof" is a named metric, and the 68.56→65.36 example appears in Table 4. The JudgeBench 64.3%→44.8% figure is confirmed. The MT-Bench agreement rises "from 70% to nearly 100%" with the performance gap. The Dorner et al. bound of at most half is confirmed. FACTS +3.23% is confirmed. The RewardBench saturation quote and RewardBench 2's "about 20 points" are confirmed. The FactBench and HalluLens refresh quotes are confirmed.

### Cited Findings
- **Shelf-Life: title, authors, venue.** Title: "On the Shelf Life of Fine-Tuned LLM-Judges: Future-Proofing, Backward-Compatibility, and Question Generalization". Authors: Janvijay Singh, Austin Xu, Yilun Zhou, Yefan Zhou, Dilek Hakkani-Tür, Shafiq Joty. v1 was posted 28 Sep 2025 and v2 on 19 Apr 2026; the arXiv comment reads "Updated after ICLR 2026 Acceptance". — [arXiv 2509.23542](https://arxiv.org/abs/2509.23542)
- **Shelf-Life, the "FutureProof" sentence.** "FutureProof" is a defined metric: "FutureProof is defined as the difference in the performance of a weak-response-trained judge…" (Sec. 5.1). Verbatim: "Across all settings, we do not observe any instance where judges generalize to new or stronger responses, with all FutureProof values being negative." — [arXiv 2509.23542 HTML](https://arxiv.org/html/2509.23542)
- **Shelf-Life setup.**
  - Weak/older generators: Gemma-2-9B, Qwen-2-7B, Llama-3.1-8B, Ministral-8B.
  - Strong/newer generators: Gemma-3-12B, Qwen-2.5-7B, Qwen-2.5-32B, Llama-3.3-70B, Mistral-Small-24B.
  - Judge backbones: Llama-3.1-8B, Ministral-8B, Mistral-24B.
  - Datasets: DeepScaleR (math) and MMLU-Pro. These are reasoning tasks, not summarization.
  - Source: [arXiv 2509.23542 HTML](https://arxiv.org/html/2509.23542)
- **Shelf-Life, Table 4.** On DeepScaleR, the Mistral-24B judge trained with DPO on weak responses has a consistent accuracy of 68.56 on weak responses to unseen questions and 65.36 on strong responses to unseen questions. The first pass's 68.6 vs 65.4 is correct. — [arXiv 2509.23542 HTML](https://arxiv.org/html/2509.23542)
- **Shelf-Life, abstract.** The abstract says future-proofing is challenging for most models; backward compatibility is relatively easy, and DPO models consistently improve; and continual learning balances adaptation better. — [arXiv 2509.23542](https://arxiv.org/abs/2509.23542)
- **JudgeBench.** "we use GPT-4o to generate all response pairs". In the ablation, Claude-3.5-Sonnet "achieves 64.3% accuracy on GPT-4o pairs, drops to 44.8% accuracy" on pairs it generated itself. Claude-3.5-Sonnet "is a stronger reasoning model, producing more difficult-to-distinguish response pairs". From the abstract: "As LLMs become more advanced, their responses grow more sophisticated, requiring stronger judges to evaluate them." The best fine-tuned judge is Skywork (70B) at 57.43%. — [arXiv 2410.12784 HTML](https://arxiv.org/html/2410.12784)
- **MT-Bench.** Verbatim: "This suggests that GPT-4 aligns with humans better when significant performance differences exist between the models." Agreement "progressively increases in line with the performance disparity of the model pairs", from "70% to nearly 100%" (Table 7). On self-enhancement, verbatim: "GPT-4 favors itself with a 10% higher win rate; Claude-v1 favors itself with a 25% higher win rate". The authors caveat that they "cannot determine whether the models exhibit a self-enhancement bias". — [arXiv 2306.05685 HTML](https://arxiv.org/html/2306.05685)
- **No Free Labels.** Title: "No Free Labels: Limitations of LLM-as-a-Judge Without Human Grounding". Authors: Krumdick, Lovering, Reddy, Ebner, Tanner. v1 is dated 7 Mar 2025 and v4 28 Sep 2026; no venue is listed.
  - The paper uses BFF-Bench: 160 questions with expert-written responses, and expert correctness judgments on 1,200 LLM responses.
  - The abstract (paraphrased by the tool) says LLM judges agree with experts mainly on questions they can answer correctly themselves, unless they are given an expert-written reference.
  - The first pass's claim that a "Qwen2.5-7B with human refs beats GPT-4o with synthetic refs" is UNCONFIRMED in this pass because it was not in the abstract text retrieved.
  - Source: [arXiv 2503.05061](https://arxiv.org/abs/2503.05061)
- **Dorner, Nastl, Hardt.** Title: "Limits to scalable evaluation at the frontier: LLM as Judge won't beat twice the data". Venue: ICLR 2025, per the arXiv comment. Verbatim: "no debiasing method can decrease the required amount of ground truth labels by more than half", which applies when the judge is no more accurate than the evaluated model. The empirical savings are smaller than the bound. The first pass's "79–84% judge accuracy" detail is UNCONFIRMED in this pass. — [arXiv 2410.13341](https://arxiv.org/abs/2410.13341)
- **FACTS Grounding.** Verbatim: "models generally rate their own outputs higher than those of other models … exhibiting a mean increase of +3.23%" (Sec. 4). Also verbatim: "The leaderboard will be actively maintained and updated to include new models and their variants." — [arXiv 2501.03200 HTML](https://arxiv.org/html/2501.03200)
- **RewardBench saturation.** Verbatim: "Some subsets are solved by small RMs, reaching 100% accuracy" (Intro). Also: "Current reward models can solve some subsets of RewardBench reliably, approaching 100% accuracy" (Sec. 5.2). — [arXiv 2403.13787v2 HTML](https://arxiv.org/html/2403.13787v2)
- **RewardBench 2.** Verbatim: "models score about 20 points on average lower on RewardBench 2 compared to RewardBench". It "sources new human prompts instead of existing prompts from downstream evaluations". The contamination sentence is present in the Background section but was only paraphrased by the tool. — [arXiv 2506.01937 HTML](https://arxiv.org/html/2506.01937)
- **FactBench.** Verbatim: "The static design makes these benchmarks susceptible to data leakage". Also: "we regenerate responses for existing prompts using the latest model versions to ensure they remain challenging". And: "We plan to annually incorporate new prompts from the LMSYS-chat-1M dataset". — [arXiv 2410.22257 HTML](https://arxiv.org/html/2410.22257)
- **HalluLens.** Verbatim: "Static test sets are especially vulnerable to obsolescence as new training datasets continuously update". Also: "we dynamically generate new test questions during evaluation (i.e., no fixed test set)". — [arXiv 2504.17550 HTML](https://arxiv.org/html/2504.17550)

### Inferences
- Shelf-Life is the most direct peer-reviewed evidence (ICLR 2026) that judges fine-tuned on older or weaker generators degrade on newer or stronger ones. Its tasks are math and MMLU-Pro reasoning, so citing it for summarization faithfulness is an extrapolation and should be worded as such.
- The JudgeBench, MT-Bench, No Free Labels and Dorner results share one message: judge reliability falls as the quality of the evaluated generators approaches or exceeds the judge's own competence.

### Gaps
- The numeric FutureProof values in Shelf-Life Fig. 3 were not extracted. The −3.20 for Mistral-24B DPO is my own subtraction from Table 4.
- The detailed numbers in No Free Labels (Qwen2.5-7B vs GPT-4o) were not re-verified.

## Q3. Freshness scan of other NLG evaluation tasks

### Takeaway
Instruction-following and math-step evaluation papers from 2023–2025 use same-year generators (gap 0). Arena-Hard includes closed frontier models. ProcessBench uses only open Qwen and Llama generators, even though its strongest critic is o1-mini. Story and simplification evaluation lag more: a gap of about 1 year, and the generators are open mid-size models or GPT-3.5/GPT-4. Agentic evaluation (Agent-as-a-Judge) evaluates agent frameworks rather than named LLM generators.

### Cited Findings

**Instruction following**
- **IFEval** (Zhou et al., arXiv Nov 2023). It evaluates GPT-4 and PaLM 2 Small only (release: mem, both Mar/May 2023), so the gap is 0. Scoring is programmatic: each instruction is checked by "a simple, interpretable, and deterministic program". The motivation is that LLM auto-evaluation "is potentially biased or limited by the ability of the evaluator LLM". The venue is UNCONFIRMED. — [arXiv 2311.07911 HTML](https://arxiv.org/html/2311.07911)
- **Arena-Hard / BenchBuilder** (Li et al., arXiv Jun 2024).
  - The top-20 correlation set's newest models are gpt-4-turbo-2024-04-09, claude-3-opus-20240229, mistral-large-2402 and dbrx-instruct. The extended leaderboard (Table 13) adds GPT-4o-2024-05-13/08-06, Claude-3.5-Sonnet-20240620 and Llama-3.1-405B. The gap is 0.
  - The default judge is GPT-4-Turbo (gpt-4-1106-preview); the alternatives are Claude-3-Opus, Gemini-1.5-Pro and Llama3-70B.
  - Verbatim: static benchmarks "are quickly becoming saturated and struggle to differentiate state-of-the-art models", and "This necessitates the continuous development of new benchmarks."
  - ICML 2025 is reported only by a third-party source and is UNCONFIRMED.
  - Source: [arXiv 2406.11939 HTML](https://arxiv.org/html/2406.11939)
- **Length-Controlled AlpacaEval** (Dubois et al., COLM 2024 per the arXiv listing, as seen in a search snippet). It raises the Spearman correlation with Chatbot Arena from 0.94 to 0.98. The specific generator list was not checked and is UNCONFIRMED. — [arXiv 2404.04475](https://arxiv.org/abs/2404.04475)

**Math / process (step-level) evaluation**
- **ProcessBench** (Zheng et al., Qwen team; ACL 2025, 2025.acl-long.50; arXiv Dec 2024).
  - Generators: 12 open models, all Qwen or Llama: Llama-3-8B/70B-Instruct, Llama-3.1-8B/70B-Instruct, Qwen2-1.5B/7B/72B, Qwen2.5-1.5B/7B/72B, Qwen2.5-Math-7B/72B. The newest is Qwen2.5 (release: mem, Sep 2024), so the gap is 0. There are no closed generators.
  - Critics evaluated: o1-mini (average F1 87.9), QwQ-32B-Preview (71.5) and GPT-4o-0806 (61.9).
  - Verbatim: "existing PRMs typically fail to generalize to more challenging math problems beyond GSM8K and MATH".
  - PRM labels are "highly 'on-policy'", so they may not transfer to other models' steps.
  - Across generators, "no particular models have notably higher process error rates", but error rates are higher on harder problems.
  - Sources: [arXiv 2412.06559 HTML](https://arxiv.org/html/2412.06559); [ACL Anthology](https://aclanthology.org/2025.acl-long.50)

**Story / creative-writing evaluation**
- **Chhun, Suchanek & Clavel, "Do Language Models Enjoy Their Own Stories?"** (TACL vol. 12, 2024, pp. 1122–1142; arXiv May 2024).
  - Stories: the HANNA systems (BERTGeneration, CTRL, GPT, GPT-2, RoBERTa, XLNet, Fusion, HINT, TD-VAE; roughly 2018–2021), plus 384 new stories from Llama-2-7b-chat, Platypus2-70B-instruct, Llama-30b-instruct-2048, StableBeluga-13B and Mistral-7B-OpenOrca (release: mem, all mid/late 2023).
  - Evaluators: Beluga-13B, Mistral-7B, Llama-2-13b-chat and gpt-3.5-turbo.
  - The gap is about 1 year, with no closed or frontier generators.
  - Verbatim: "Beluga-13B and Mistral-7B both seem to prefer the outputs from larger LLMs (Platypus2-70B, Llama-30B) to their own outputs".
  - Sources: [arXiv 2405.13769 HTML](https://arxiv.org/html/2405.13769v1); [ACL Anthology 2024.tacl-1.62](https://aclanthology.org/2024.tacl-1.62/)
- **LitBench** (Fein et al., arXiv Jul 2025).
  - The test set has 2,480 human-labelled pairwise comparisons drawn from Reddit r/WritingPrompts, which implies human-written stories. The training set has 43,827 pairs.
  - The best zero-shot judge is Claude-3.7-Sonnet at 73%; the trained Bradley-Terry and generative RMs reach 78%.
  - The online study used "newly LLM-generated stories". A secondary summary names GPT-4.1/GPT-4o as their source, which is UNCONFIRMED.
  - The judges are fresh, but the evaluated texts are mostly human-written.
  - The EACL 2026 venue comes from an ACL Anthology preview URL in search results and is UNCONFIRMED.
  - Sources: [arXiv 2507.00769](https://arxiv.org/abs/2507.00769); [search hit, Anthology preview](https://preview.aclanthology.org/credits/2026.eacl-long.362)

**Text simplification evaluation**
- **LENS / SimpEval** (Maddela et al., ACL 2023; arXiv Dec 2022). SimpEval_past has 12K ratings of 2.4K simplifications from 24 past systems. SimpEval_2022 has more than 1K ratings of 360 simplifications, including GPT-3.5 outputs. The gap is 0 for the 2022 subset, but most training data comes from older systems. This is based on a search snippet only; the paper was not fetched. — [arXiv 2212.09739](https://arxiv.org/abs/2212.09739)
- **REFeREE** (arXiv Mar 2024). A reference-free simplification metric that outperforms LENS on overall ratings. The generators of its evaluated outputs are UNCONFIRMED, because only a search snippet was read. — [arXiv 2403.17640](https://arxiv.org/html/2403.17640v1)
- **Wu & Arase, "An In-depth Evaluation of LLMs in Sentence Simplification with Error-based Human Assessment"** (arXiv Mar 2024, v4 Jul 2025). The generators are GPT-4, Qwen2.5-72B and Llama-3.2-3B. The meta-evaluation finds that automatic metrics struggle to separate quality levels, especially for GPT-4 and Qwen2.5-72B, whose outputs are mostly high quality. This is search-snippet level; the venue is UNCONFIRMED. — [arXiv 2403.04963v4](https://arxiv.org/html/2403.04963v4)

**Agentic / long-output evaluation**
- **Agent-as-a-Judge** (Zhuge et al., arXiv Oct 2024; ICML 2025 poster per search results).
  - Evaluated systems: MetaGPT, GPT-Pilot and OpenHands, which are agent frameworks, on 55 DevAI tasks.
  - Agent-as-a-Judge reaches 92.07% (gray-box) and 90.44% (black-box) alignment with human consensus on OpenHands, compared with 70.76% and 60.38% for LLM-as-a-Judge.
  - The backbone LLMs of the evaluated agents are UNCONFIRMED.
  - Sources: [arXiv 2410.10934](https://arxiv.org/html/2410.10934v2); [ICML 2025 text mirror](https://object.cloud.sdsc.edu/v1/AUTH_da4962d3368042ac8337e2dfdd3e7bf3/ml-papers-txt/ICML/2025/agentasajudge_evaluate_agents_with_agents__86bb0565.txt)

### Inferences
- **Gap by task family:**
  - Instruction-following (IFEval, Arena-Hard): gap 0, with closed frontier generators included.
  - Math process evaluation (ProcessBench): gap 0, open generators only.
  - Story evaluation (Chhun et al. 2024): about 1 year, open mid-size 2023 models plus legacy 2018–2021 systems.
  - Simplification (LENS 2023, Wu & Arase 2024–25): gap 0–1. LENS relies mostly on pre-LLM systems; Wu & Arase uses GPT-4 and Qwen2.5.
- **Refresh rationale is shared across families.** Arena-Hard (saturation), IFEval (judge-ability bias), ProcessBench (on-policy PRM labels) and Shelf-Life (judges trained on old responses) all justify fresh generators or ground-truth checks for the same reason that applies to factual-consistency metrics.
- **Implication for a Hebrew factual-consistency metric:**
  - Older generators with a lag of about 1 year are common in story and simplification evaluation.
  - In 2024–2025 judge and metric papers, reviewers increasingly expect at least one current-generation slice of outputs.

### Gaps
- No 2025–2026 simplification metric paper with fully verified generator lists was found.
- The AlpacaEval-LC generator list and the Agent-as-a-Judge backbone LLMs were not verified.
- Venues for IFEval and Arena-Hard (ICML 2025) were not confirmed from primary pages.
- I did not scan dialogue-specific 2024–2026 metric papers beyond the first pass (UniEval, GPTScore, TIGERScore).
