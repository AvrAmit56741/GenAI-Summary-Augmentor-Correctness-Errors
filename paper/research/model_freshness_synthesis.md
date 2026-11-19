# Do evaluation-metric papers test on current models? Cross-task synthesis (2026-11-18)

Sources: four surveys produced this session by research subagents (web + paper full text; facts from memory are tagged
(mem)/(unverified) inside each file — check before citing):
- `model_freshness_en_summarization.md` (37 papers), `model_freshness_multilingual.md` (33),
  `model_freshness_mt_code.md` (WMT22–25 + 21 papers), `model_freshness_qa_rag_judges.md` (26).

**Age gap** = paper year − release year of the newest model whose OUTPUTS are evaluated (not the judge model).

## 1. Summary table

| Task family | Metric papers: age gap | Benchmark papers: age gap | Frontier/closed generators included? | Size range of evaluated generators | Refresh practice | Evidence that metrics degrade on newer/stronger generators |
|---|---|---|---|---|---|---|
| **Summarization, English** | **2–4 y**, growing (SummaC, AlignScore, G-Eval, FineSurE meta-evaluate on SummEval/FRANK/AggreFact: BART/PEGASUS/T5 2019–20, ≤0.6B); MiniCheck/FactCG ~2 y | **0–1 y** (RAGTruth, InstruSum, FABLES, UniSumEval, FaithBench, MSumBench, VeriGray with GPT-5) | Benchmarks since 2024: yes. Metric papers: rarely (no GPT-4 summaries with human labels in MiniCheck) | 0.1B → frontier closed | None; static sets reused | **Strong**: Tang 2023 (~10-pt balanced-acc drop OLD→SOTA on CNN/DM); TofuEval, Ramprasad 2024, UniSumEval; FaithBench detectors 50–58% bal-acc on 10 modern LLMs |
| **Summarization / eval, multilingual** | **1–5 y, median ~2.2** (mFACE ~2.2, mFACT ~2.8, SEAHORSE ~1.1; STEMF 2026 validates on 2020 mT5 + translated FRANK, ~5 y) | **0–1 y** (METAL, HeSum, BASSE, MSumBench, AraHalluEval, MEMERAG, MultiHal) | ~10/26 include the frontier model of their time; ~12 none (mT5/mBART, open ≤70B). Judges almost always newest closed (GPT-family in 79% of judge papers, Doğruöz 2026) | mT5/mBART → GPT-4-class | None | **Missing**: breakdowns are by language resource level, not by generator; no multilingual equivalent of Tang 2023. HeSum: ROUGE ranks mLongT5 first, humans rank GPT-4 first. Heavy reliance on translated data |
| **Machine translation** | **~1 y** (new metrics tested on last year's WMT human data) | **0 y** — WMT refreshes test sets yearly and since 2023 adds frontier LLMs (2025: ~20 LLMs, 7B → 671B, Gemini-2.5-Pro, GPT-4.1, Claude-4) | Yes (by organizers) | 7B → 671B + closed | **Yearly (WMT)** | Mixed: metrics separate high-quality outputs poorly (Agrawal 2024; BLEURT drift; COMET top systems); GPT-4 judge over-rates GPT-4 errors; WMT24 "Are LLMs breaking MT metrics?" → no for trained metrics; human refs sometimes worse than LLM output. **Hebrew precedent:** WMT23 He→En, GPT-4 best system |
| **Code** | **1–3 y** (CodeBLEU, CodeBERTScore, ICE-Score, CodeJudge reuse 2018–21 CoNaLa / Codex outputs) | **0–0.5 y** (CodeUltraFeedback, CodeJudge-Eval, CodeJudgeBench) | Often not (some use only open ≤34B) | ≤34B open → Claude/Gemini | None | **Strong**: CodeJudgeBench — judges near-perfect on GPT-4o-era outputs drop substantially on Gemini-2.5 / Claude-3.7/4 outputs |
| **Long-form factuality** | **~0 y** (FActScore, SAFE, VeriScore, FactBench, HalluLens: newest closed model within 1–2 months + open size ladder 2B→405B) | 0 y | Yes | 2B → 405B + closed | FactBench/HalluLens refresh by design | Some: SAFE's human validation uses 2022–23 outputs; scores depend on response length |
| **RAG / grounded QA** | 0–1 y (RAGAS ChatGPT only; ARES mostly mock systems) | 0–1 y (RAGTruth frozen 2023; FACTS Grounding maintained leaderboard) | Mostly | 7B → closed | FACTS leaderboard | Judge self-preference (+3.2%, FACTS Grounding) |
| **LLM-as-judge / reward models** | 0–1 y at arXiv, up to 2 y by venue | 0–1 y | Frontier often only as judge/teacher; judged responses from 7–13B open | 7B → closed | RewardBench 2 replaced saturated v1 | **Strong**: Shelf-Life 2025 (judges trained on older generators lose accuracy on newer); JudgeBench (Claude-3.5 judge 64% on GPT-4o pairs vs 45% on own pairs); MT-Bench agreement best when strengths differ |
| **Dialogue / data-to-text** | **2–4 y** (UniEval, GPTScore, TIGERScore reuse 2015–20 outputs) | Kasner & Dušek 2024: fresh, contamination-free | Rarely | small seq2seq | None | Little |

## 2. Answers to the questions

1. **Is generator age relevant in metric papers?** Yes, and it is a known, growing concern. Two populations exist:
   **metric papers** mostly meta-evaluate on static human-labeled sets whose outputs are 2–4 years old
   (summarization, code, dialogue), while **benchmark papers** since 2024 almost always include current frontier
   outputs. Only MT (WMT, yearly) and long-form factuality institutionalize freshness.
2. **New vs old, size, open vs closed, frontier:** frontier closed models appear mainly as **judges**, not as the
   **generators** whose errors are evaluated — the same model often plays both roles, causing self-preference.
   Size ladders (2B→400B+) are standard only in long-form factuality and WMT.
3. **Correlation with task:** freshness is highest where a community runs a yearly shared task or leaderboard (MT,
   long-form factuality, reward models) and lowest where meta-evaluation relies on one-off human annotation
   (summarization faithfulness, code, dialogue).
4. **Does metric quality change with generator age?** Yes in English summarization (Tang 2023; FaithBench), code
   (CodeJudgeBench), and judges (Shelf-Life, JudgeBench): errors of stronger models are subtler and harder to
   detect. In MT the picture is mixed (trained metrics survive; discrimination at the top is weak).
5. **Multilingual vs English:** multilingual metric papers lag as much as English ones (~2 y) and lean far more
   on **translated** data; **no multilingual paper tests a fixed metric on old vs new generators** — an open gap.

## 3. Where MorphEvalHeb stands (check against the paper; to be confirmed)

| Part of our evaluation | Generators of evaluated outputs | Approx. age gap at ARR Jan 2027 |
|---|---|---|
| Injected | Human/reference MAFAT summaries (2022 dataset) + errors inserted by GPT/Gemini rewriting | references ~4–5 y; the errors are synthetic, not model errors |
| Natural | mFACT test split (English summaries of 2021–22-era models, translated to Hebrew with GPT-4o-mini) | **~4–5 y**, translated |
| Taxonomy grounding / real errors | ~300 DictaLM 2.0 summaries; five LLMs in Fig. 1; 265 Gemini-identified real errors | ~1–2 y |
| Unpublished: `current_model_summaries_*` (commit a5dde86) | Claude Haiku 4.5, GPT-5.6-luna, Qwen3-32B, GLM-4.7-flash, Ministral-3-14B, DictaLM 3 | ~0 y, but no human labels yet |

## 4. Implications / recommendations

1. **Reviewer risk:** our two headline test sets are old (Natural) or synthetic (Injected) — the same weakness as
   SummaC/AlignScore/STEMF. A reviewer aware of Tang 2023 / FaithBench can ask whether results hold on current-LLM
   Hebrew summaries.
2. **Opportunity (novel for multilingual):** a small, human-labeled set of Hebrew summaries from **current**
   models across eras and sizes (e.g. mT5/DictaLM 2 → DictaLM 3 / Qwen3 / Haiku 4.5 / GPT-5-class), scored by all
   metrics with a **per-generator breakdown**, would be the first multilingual counterpart of Tang 2023 /
   FaithBench, and would make MorphEvalHeb fresher than every multilingual metric paper surveyed (none has a gap < 1 y).
3. Report generator, size, open/closed and release date for every evaluated set (a "freshness" column) — cheap and
   pre-empts the critique.
4. Avoid self-preference: do not let the same model family generate, inject, and judge the same data.
