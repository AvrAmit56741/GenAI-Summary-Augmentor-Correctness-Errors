# Multilingual / non-English factuality evaluation and LLM-as-a-judge reliability in low-resource and morphologically rich languages (2024–2026 update for MorphEvalHeb)

Research date: 2026-11-12. I opened every item below on its arXiv or ACL Anthology page. "Peer-reviewed" means an ACL Anthology entry exists or the arXiv comments name a venue. "arXiv-only" means the page lists no venue. I did not count the papers the authors already cite (mFACE, mFACT, Multi-FActScore, SEAHORSE, G-Eval, Laurer mDeBERTa NLI) as new.

Overall finding: none of the multilingual factuality or hallucination benchmarks I verified includes Hebrew. Where a page gave the language list, Hebrew was absent. Where it did not, Hebrew was not mentioned. This gap supports MorphEvalHeb's novelty claim. Most candidate baselines are released models or evaluators that can be run zero-shot on Hebrew.

## Q1. What multilingual factuality/hallucination detectors and benchmarks exist that include Hebrew or could be applied to Hebrew (candidate baselines)?

### Takeaway
No verified benchmark contains Hebrew. Mu-SHROOM, the largest multilingual span-level hallucination benchmark (SemEval-2025, 14 languages), excludes Hebrew, and its 2026 successor SHROOM-Visions is vision-language only in 4 languages. The most practical new baselines for Hebrew are multilingual evaluator models applied zero-shot: STEMF (multilingual faithfulness evaluator, EACL Findings 2026), M-Prometheus (open multilingual judge), and possibly Hercule. Their feasibility depends on whether weights are released and how well the backbone handles Hebrew.

### Cited Findings
**Shared tasks and benchmarks**
- **SemEval-2025 Task 3: Mu-SHROOM, the Multilingual Shared Task on Hallucinations and Related Observable Overgeneration Mistakes.** Raúl Vázquez, Timothee Mickus, Elaine Zosa, Teemu Vahtola, Jörg Tiedemann, Aman Sinha, Vincent Segonne, Fernando Sánchez-Vega, Alessandro Raganato, Jindřich Libovický, Jussi Karlgren, Shaoxiong Ji, Jindřich Helcl, Liane Guillou, Ona de Gibert, Jaione Bengoetxea, Joseph Attieh, Marianna Apidianaki. Proc. SemEval-2025, pp. 2472–2497 (peer-reviewed; ACL Anthology 2025.semeval-1.322; arXiv 2504.11975). — [arXiv](https://arxiv.org/abs/2504.11975); [ACL Anthology](https://aclanthology.org/2025.semeval-1.322)
  - Task: span-level detection of hallucinations in instruction-tuned LLM outputs, judged against Wikipedia. 2,618 submissions from 43 teams. Annotator disagreement on spans was high. — [arXiv](https://arxiv.org/abs/2504.11975)
  - Languages: AR, CA, CS, DE, EN, ES, EU, FA, FI, FR, HI, IT, SV, ZH. CA, CS, EU and FA were test-only "surprise" languages. **Hebrew is not included.** — [arXiv HTML v2](https://arxiv.org/html/2504.11975v2)
  - Metrics: character-level IoU (primary) and Spearman ρ against annotator marking rates. IT and HI had the highest mean IoU (0.51 and 0.50). ES, ZH and CS were hardest, and the top Spanish system had ρ near zero. RAG-based systems scored significantly higher. Some top systems used translate-test (for example AILS-NTUA translated into EN and ZH). — [arXiv HTML v2](https://arxiv.org/html/2504.11975v2)
  - The ACL Anthology "Cite" field lists only the last authors. The BibTeX and arXiv list Vázquez first. Cite as Vázquez et al. — [search note](https://aclanthology.org/2025.semeval-1.322)
  - Recommendation: **(i) cite in related work** as the main multilingual hallucination-detection shared task. It covers MRLs (Arabic, Finnish, Persian, Czech, Basque) but not Hebrew, and its task is QA-style span labelling, not summary faithfulness. Not a direct baseline. A top system could in principle be reproduced on Hebrew, but systems were tuned per language and target spans rather than summary-level labels.
- **Overview of SHROOM-Visions 2026: A Shared Task on Hallucination Detection in Large Vision-Language Models.** arXiv 2608.25662, held at the UncertaiNLP workshop at EMNLP 2026. The 2026 successor in the SHROOM series covers image-conditioned text in ZH, EN, FR and IT. 27 teams made 600+ submissions. — [arXiv PDF](https://arxiv.org/pdf/2608.25662); [ACL portal CFP](https://www.aclweb.org/portal/node/14502)
  - I only saw this through search snippets and the CFP, not a full fetch of the abstract page. Treat the numbers as provisional. Recommendation: at most a footnote. Its only relevance is showing that the 2026 SHROOM iteration moved to multimodal and still has no Hebrew.
- **Poly-FEVER: A Multilingual Fact Verification Benchmark for Hallucination Detection in Large Language Models.** Hanzhi Zhang, Sumera Anjum, Heng Fan, Weijian Zheng, Yan Huang, Yunhe Feng. arXiv 2503.16541, 2025 (arXiv-only). It has 77,973 labelled claims in 11 languages, drawn from FEVER, Climate-FEVER and SciFact, and is on Hugging Face. The abstract page does not list the languages and Hebrew is not mentioned. — [arXiv](https://arxiv.org/abs/2503.16541)
  - Recommendation: **(i) cite** as a multilingual claim-verification benchmark (open-domain fact checking, not source-grounded summary faithfulness). Probably the "PolyFact-like" item in the brief.
- **HalluVerse25: Fine-grained Multilingual Benchmark Dataset for LLM Hallucinations.** Samir Abdaljalil, Hasan Kurban, Erchin Serpedin. arXiv 2503.07833, 2025 (arXiv-only). Languages are EN, AR and TR. An LLM injects hallucinations into factual biographical sentences and humans review them. Errors are labelled at entity, relation and sentence level. — [arXiv](https://arxiv.org/abs/2503.07833)
  - Recommendation: **(i) cite as close methodological related work**. Like MorphEvalHeb it uses synthetic error injection plus a fine-grained taxonomy, and it covers two MRLs (Arabic, Turkish).
- **Halluverse-M^3: A multitask multilingual benchmark for hallucination in LLMs.** Samir Abdaljalil, Parichit Sharma, Erchin Serpedin, Hasan Kurban. arXiv 2602.06920, Feb 2026 (arXiv-only). It covers EN, AR, HI and TR on QA and **dialogue summarization**, with entity, relation and sentence labels built by controlled editing and human verification. Detection accuracy is highest in English and drops for lower-resource languages, with Hindi lowest. Summarization is harder than QA. Sentence-level hallucinations are hard even for the strongest models. — [arXiv](https://arxiv.org/abs/2602.06920)
  - Recommendation: **(i) cite** as the closest 2026 multilingual benchmark using controlled error injection for summarization. Its result that detection is weaker outside English supports the paper's motivation.

**Multilingual faithfulness evaluators (baseline candidates)**
- **Multilingual Self-Taught Faithfulness Evaluators (STEMF).** Carlo Alfano, Aymen Al Marjani, Zeno Jonke, Amin Mantrach, Saab Mansour, Marcello Federico. Findings of EACL 2026, pp. 5035–5051 (peer-reviewed; arXiv 2507.20752). The evaluator learns only from synthetic multilingual summarization data with cross-lingual transfer. It beats English state-of-the-art evaluators and MT-based approaches. Evaluation performance per language tracks the LLM's general ability in that language. Hebrew is not mentioned in the abstract. — [ACL Anthology](https://aclanthology.org/2026.findings-eacl.266/); [arXiv](https://arxiv.org/abs/2507.20752)
  - Search snippets (not verified on the abstract page) say the final evaluator is a fine-tuned ~9B LLM. — [arXiv PDF](https://arxiv.org/pdf/2507.20752)
  - Recommendation: **(i) cite as the closest related work**: a multilingual faithfulness evaluator built from synthetic data, set against MorphEvalHeb's Hebrew-native synthetic injection plus encoder verifiers. **(ii) Add as a baseline if the weights are public.** Hebrew feasibility: zero-shot via cross-lingual transfer. Its own finding that performance tracks general language ability predicts weaker results on Hebrew. I did not check whether weights are released.
- **M-Prometheus: A Suite of Open Multilingual LLM Judges.** José Pombal, Dongkeun Yoon, Patrick Fernandes, Ian Wu, Seungone Kim, Ricardo Rei, Graham Neubig, André F. T. Martins. arXiv 2504.04953, v1 Apr 2025, v2 Oct 2025. The arXiv page lists no venue, so I treat it as arXiv-only here. Open judges at 3B, 7B and 14B (Qwen2.5-Instruct backbone) for direct assessment and pairwise comparison. They beat open judges on multilingual reward benchmarks covering 20+ languages. Training on natively multilingual feedback beats translated data. Models, data and code are released. — [arXiv](https://arxiv.org/abs/2504.04953)
  - Recommendation: **(ii) add as an open-weight judge baseline** alongside GPT-4o-mini. It is cheap to run on Hebrew zero-shot with a factual-consistency rubric. In BASSE (below) it was one of the judges tested on Basque and Spanish summary consistency, which gives a precedent.
- **Cross-Lingual Auto Evaluation for Assessing Multilingual LLMs (Hercule / CIA suite, Recon test set).** Sumanth Doddapaneni, Mohammed Safi Ur Rahman Khan, Dilip Venkatesh, Raj Dabre, Anoop Kunchukuttan, Mitesh M. Khapra. ACL 2025 (Long), pp. 29297–29329 (peer-reviewed). Hercule is a Llama-3.1-8B-based evaluator. It scores target-language responses using English reference answers and rubrics. It agrees with humans more closely than proprietary models and works zero-shot on unseen languages. Recon has 500 instructions with human scores in 6 languages, which the page does not name. — [ACL Anthology](https://aclanthology.org/2025.acl-long.1419/); [arXiv](https://arxiv.org/abs/2410.13394)
  - Recommendation: **(i) cite**. It is the opposite design choice to MorphEvalHeb (English references for non-English outputs versus Hebrew-native verification). As a baseline it is optional and low-fit: it targets instruction following with reference answers, not source-grounded summary consistency.

### Inferences
- Since Hebrew is missing from Mu-SHROOM, Halluverse-M^3, HalluVerse25 and AraHalluEval, and nothing I found names it in Poly-FEVER, STEMF, M-Prometheus or Hercule, a claim like "first Hebrew-native factual-consistency benchmark and metric" looks defensible. It should still be hedged as "to our knowledge."
- The most persuasive new baselines are (a) M-Prometheus-7B or 14B as an open judge and (b) STEMF if released. Both test the claim that "multilingual" evaluators transfer to Hebrew. Comparing them on the Kendall-τ protocol would strengthen the result beyond GPT-4o-mini, mFACT and G-Eval.

### Gaps
- I found no "mHallu", "X-FACTOR" or "multilingual LLM-AggreFact" paper under those names. They may not exist or may go by other names. I did not include anything unverified.
- I could not confirm the language lists for Poly-FEVER, M-Prometheus's benchmarks, Hercule's Recon or STEMF from the abstract pages. Checking each paper's language table for Hebrew is still needed.
- I did not verify that STEMF weights are released.

## Q2. What evidence shows LLM judges are less reliable in low-resource or morphologically rich languages (supports the paper's motivation)?

### Takeaway
There is strong and growing evidence. One peer-reviewed study (Fu & Liu, Findings of EMNLP 2025) finds average cross-language judge consistency of only Fleiss' κ ≈ 0.3 across 25 languages, worst in low-resource languages, and scale or multilingual training does not fix it. An EMNLP Findings 2026 survey finds only 33 of 650 ACL papers that use LLM-as-a-judge address multilingual or low-resource settings. Several 2026 preprints report language-dependent preference flips and instability that aggregate reporting hides.

### Cited Findings
- **How Reliable is Multilingual LLM-as-a-Judge?** Xiyan Fu, Wei Liu. Findings of EMNLP 2025, pp. 11040–11053 (peer-reviewed; arXiv 2505.12201). Five LLMs from different families were tested on five tasks in 25 languages. Average Fleiss' κ is about 0.3 across languages. Consistency is particularly poor in low-resource languages. Multilingual training and larger scale do not directly improve it. The authors conclude LLMs "are not yet reliable for evaluating multilingual predictions." — [ACL Anthology](https://aclanthology.org/2025.findings-emnlp.587/); [arXiv](https://arxiv.org/abs/2505.12201)
  - Per search snippets of the arXiv HTML (not fully verified): Arabic and Telugu are among the weakest, and Aya-Expanse consistency is near zero for some languages on MGSM. — [arXiv HTML](https://arxiv.org/html/2505.12201v1)
  - Recommendation: **(i) cite as the main motivation evidence**. Arabic, a Semitic MRL close to Hebrew, is among the weakest.
- **Challenges and Recommendations for LLM-as-a-Judge in Multilingual Settings and for Low-Resource Languages.** A. Seza Doğruöz, Xixian Liao, Verena Blaschke, Jakob Prange, Senyu Li, David Ifeoluwa Adelani. arXiv 2607.02235 (v1 Jul 2026, v2 Sep 2026). Comments say it will appear in Findings of EMNLP 2026 (accepted, peer-reviewed). Of 650 ACL Anthology papers that mention LLM-as-a-judge, only 33 (about 5%) address low-resource or multilingual settings. It reports inconsistent outcomes across studies, overtrust of LLM judgments, frequent reliance on a single judge model, and inadequate human validation. It ends with a recommendations checklist. — [arXiv](https://arxiv.org/abs/2607.02235)
  - Recommendation: **(i) cite** (May–Nov 2026 window). MorphEvalHeb's human-label Kendall-τ validation follows its recommendations, so the paper can say it does.
- **MM-Eval: A Multilingual Meta-Evaluation Benchmark for LLM-as-a-Judge and Reward Models.** Guijin Son, Dongkeun Yoon, Juyoung Suk, Javier Aula-Blasco, Mano Aslan, Vu Trong Kim, Shayekh Bin Islam, Jaume Prats-Cristià, Lucía Tormo-Bañuelos, Seungone Kim. arXiv 2410.17578 (v2 Mar 2025; comment "work in progress"; arXiv-only per the page). It covers 18 languages in five core subsets (including "Language Hallucination") and 122 languages in a consistency subset. The abstract says "evaluators are unfair and inconsistent when evaluating lower-resourced languages." — [arXiv](https://arxiv.org/abs/2410.17578)
  - Secondary snippets (not verified) say low-resource evaluators give good responses lower scores and bad responses higher scores. — [arXiv HTML](https://arxiv.org/html/2410.17578v2)
  - Recommendation: **(i) cite** for motivation.
- **Prompting Strategy Effects Depend on the Judge: A Multilingual Study of LLM-as-a-Judge Robustness.** Asal Mehrabi, Mohammad Aliannejadi, Clemencia Siro. arXiv 2610.31172, 22 Oct 2026 (arXiv-only). 9 prompting strategies × 4 open judges × 4 tasks × 13 datasets in 27 languages, giving over 1M scores validated against humans. The judge model matters more than the prompt. Strategy effects reverse between judges. Stability degrades in low-resource languages, "where these effects are concealed by aggregate reporting." The abstract page does not list the languages, so Hebrew is not confirmed. — [arXiv](https://arxiv.org/abs/2610.31172)
  - Recommendation: **(i) cite** (Oct 2026). It supports reporting per-language and per-judge results rather than a single GPT-4o-mini number.
- **When Languages Disagree: Self-Evolving Multilingual LLM Judges (SEMJ).** Xiyan Fu, Wei Lu (as rendered on the fetched page; possibly the same "Wei Liu" as above, so check), arXiv 2606.08092, Jun 2026 (arXiv-only). It uses cross-lingual inconsistency as a signal to trigger reflection and re-judging. It beats voting and reflection baselines in accuracy and consistency. An oracle analysis shows that sampling judgments across languages raises the ceiling. — [arXiv](https://arxiv.org/abs/2606.08092)
  - Recommendation: **(i) cite** as mitigation work. It is not a baseline.
- **Does the Judge Prefer English? Evaluating Language-Switching Invariance in LLM-as-a-Judge.** Shaojie Yin. arXiv 2606.14278, Jun 2026 (arXiv-only). On LLMBar (419 items, 13,408 judgments, 4 API judges), Chinese and code-switched presentations caused 10.7–14.4% preference flips relative to English, and every judge was most accurate in English. — [arXiv](https://arxiv.org/abs/2606.14278)
  - Recommendation: **(i) cite** briefly. The evidence covers EN and ZH only, not an MRL.
- **Evaluating the Evaluator: Summarization Metrics and LLM-Judges beyond English (BASSE).** Jeremy Barnes, Naiara Perez, Alba Bonet-Jover, Begoña Altuna. arXiv 2503.17039 (v3 Sep 2026). Comments say SEPLN 2026 (peer-reviewed venue). BASSE has 2,040 Basque and Spanish summaries rated on 5 criteria, including **consistency**. — [arXiv](https://arxiv.org/abs/2503.17039)
  - Proprietary judges correlate best with humans, then criteria-specific metrics. Open-source judges do poorly. — [arXiv](https://arxiv.org/abs/2503.17039)
  - Consistency was the hardest criterion: mean absolute correlation with humans was 0.201 (Basque) and 0.257 (Spanish), and no metric or judge correlated strongly in both. Inter-annotator agreement on consistency was 0.44 (EU) and 0.19 (ES). Judges tested were Prometheus 2, M-Prometheus 7B/14B, Selene Mini, Qwen2.5 7B/72B, GPT-4o-mini and GPT-4o. The authors excluded English-only metrics. Basque's agglutinative morphology is noted only in token counts. — [arXiv HTML v3](https://arxiv.org/html/2503.17039v3)
  - Recommendation: **(i) cite as the closest related work.** It is the nearest analogue: summary-consistency meta-evaluation of LLM judges in a morphologically rich language (Basque), with GPT-4o-mini among the judges. Its low consistency correlations fit MorphEvalHeb's G-EVAL 0.24 and GPT-4o-mini 0.60 results.

### Inferences
- With Fu & Liu (2025), Doğruöz et al. (2026), MM-Eval, Mehrabi et al. (2026), BASSE and Halluverse-M^3, the paper can make a sourced claim. Judge reliability falls outside English and in low-resource languages, the factual-consistency dimension is the hardest, and these effects hide in aggregate scores.
- No study I found measures judge reliability in Hebrew specifically. The paper's Hebrew Kendall-τ results (GPT-4o-mini 0.60, G-EVAL 0.24) are therefore new evidence for this literature.

### Gaps
- I could not verify per-language numbers (for example Arabic) in Fu & Liu from the full text. Only search-snippet summaries mention Arabic and Telugu.
- For Mehrabi et al. 2026, I could not confirm whether Hebrew is among the 27 languages.

## Q3. Are there analogous "native-language factuality metric" papers for other MRLs that should be cited as closest related work?

### Takeaway
There are few. The clearest analogues are AraHalluEval (Arabic, fine-grained hallucination framework for summarization and GQA, ArabicNLP 2025 per its ACL Anthology PDF), HalluVerse25 and Halluverse-M^3 (Arabic and Turkish, with injection-based fine-grained taxonomies), and BASSE (Basque summary consistency). I found no verified 2024–2026 native factual-consistency metric for Turkish, Finnish, Korean, Hungarian, Czech, Persian or Amharic summarization. That gap supports the paper's contribution.

### Cited Findings
- **AraHalluEval: A Fine-grained Hallucination Evaluation Framework for Arabic LLMs.** Aisha Alansari, Hamzah Luqman. arXiv 2509.04656 (Sep 2025). An ACL Anthology PDF exists at 2025.arabicnlp-main.12, which indicates peer-reviewed ArabicNLP 2025, though the arXiv page lists no venue. It defines 12 fine-grained hallucination indicators for Arabic GQA and **summarization** and evaluates 12 LLMs (Arabic-pretrained, multilingual and reasoning models). Factual hallucinations outnumber faithfulness errors. The Arabic model Allam hallucinates less than multilingual models. — [arXiv](https://arxiv.org/abs/2509.04656); [ACL Anthology PDF](https://aclanthology.org/2025.arabicnlp-main.12.pdf)
  - Recommendation: **(i) cite as the closest Semitic-language analogue.** It is a native fine-grained taxonomy for summarization in a related MRL, but it is an evaluation framework with human annotation, not a trained verifier.
- **HalluVerse25** and **Halluverse-M^3** (details under Q1): Arabic and Turkish fine-grained hallucination data built by injection. — [arXiv 2503.07833](https://arxiv.org/abs/2503.07833); [arXiv 2602.06920](https://arxiv.org/abs/2602.06920)
- **BASSE** (details under Q2): Basque summary-consistency meta-evaluation. — [arXiv](https://arxiv.org/abs/2503.17039)
- **BHRAM-IL: A Benchmark for Hallucination Recognition and Assessment in Multiple Indian Languages.** Bhasha workshop 2025, ACL Anthology 2025.bhasha-1.9. I saw it only in search results. I did not open the page or verify the authors, so verify before citing. — [ACL Anthology](https://aclanthology.org/2025.bhasha-1.9/)
- **Hebrew context: HeSum: a Novel Dataset for Abstractive Text Summarization in Hebrew.** Tzuf Paz-Argaman, Itai Mondshine, Asaf Achi Mordechai, Reut Tsarfaty. Findings of ACL 2024 (peer-reviewed; arXiv 2406.03897). It has 10,000 Hebrew news article–summary pairs and highlights Hebrew's "unique morphological challenges." The abstract does not discuss factual-error evaluation. — [arXiv](https://arxiv.org/abs/2406.03897)
  - Recommendation: likely already cited as a data source. If not, **(i) cite**.
- **Dicta-LM 3.0: Advancing The Frontier of Hebrew Sovereign LLMs.** Shaltiel Shmidman, Avi Shmidman, Amir DN Cohen, Moshe Koppel. arXiv 2602.02104, Feb 2026 (arXiv-only). Its Hebrew chat benchmark suite includes summarization. According to a secondary summary, summarization is scored by LLM-judge win rates (GPT-4 or GPT-4o) without human gold, which I did not verify in the PDF. — [arXiv](https://arxiv.org/abs/2602.02104); [secondary summary](https://www.emergentmind.com/topics/hebrew-llm-benchmark-suite)
  - Recommendation: **(i) cite** as current Hebrew practice. Hebrew summarization quality is evaluated with unvalidated English-centric LLM judges, which is the practice MorphEvalHeb argues against. Check the PDF for the judging setup first.

### Inferences
- The related-work paragraph could read roughly: "Native-language factuality resources for MRLs remain rare: Arabic (AraHalluEval; HalluVerse25), Turkish (HalluVerse25, Halluverse-M^3), and Basque (BASSE) are recent exceptions, and none provides a trained, morphology-aware verifier; for Hebrew, current practice (e.g., Dicta-LM 3.0) relies on LLM-judge win rates."

### Gaps
- My searches for Korean, Turkish, Finnish, Hungarian, Czech, Persian, Amharic and Indic summarization factual-consistency metrics from 2024–2026 found no verifiable dedicated paper. ACL Anthology venue-specific searches (for example SIGTURK, Korean NLP or AfricaNLP workshops) could still find some. I flag this as a gap, not evidence of absence.
- I did not verify BHRAM-IL's authors or contents.

## Q4. What are the pitfalls of translation-based evaluation (translate-test / English pivot)?

### Takeaway
The evidence argues against translating Hebrew outputs into English for evaluation. Translation errors themselves cause measurable accuracy drops on translated benchmarks. English-pivot judging measures agreement with English rather than correctness. Native multilingual training data beats translated data for judges.

### Cited Findings
- **Quantifying the Impact of Translation Errors on Multilingual LLM Evaluation.** Klaudia-Doris Thellmann, Bernhard Stadler, Michael Färber, Jens Lehmann. arXiv 2605.24904, May 2026 (arXiv-only). Target-side translation errors are associated with percentage-point accuracy drops on translated benchmarks, even after controlling for English correctness and source-side anomalies. Automatic error-span detectors (LLM judges, xCOMET-XXL) agree only modestly with expert annotations. — [arXiv](https://arxiv.org/abs/2605.24904)
  - Recommendation: **(i) cite** to justify native-language evaluation over translated benchmarks or translate-then-judge.
- In M-Prometheus, training on natively multilingual feedback beat translated data. — [arXiv](https://arxiv.org/abs/2504.04953)
- STEMF reports gains over machine-translation-based faithfulness evaluation baselines. — [ACL Anthology](https://aclanthology.org/2026.findings-eacl.266/)
- In Mu-SHROOM, some systems used translate-test (for example AILS-NTUA into EN and ZH). Top systems instead relied on retrieval and multi-stage pipelines, and RAG was associated with significantly higher IoU and ρ. — [arXiv HTML v2](https://arxiv.org/html/2504.11975v2)
- Judge preference flips 10.7–14.4% when only the presentation language changes (EN, ZH, code-switched). — [arXiv 2606.14278](https://arxiv.org/abs/2606.14278)

### Inferences
- For Hebrew, translation can collapse or invent morphologically encoded distinctions (gender, number, definiteness, possessive clitics, prefixed prepositions). An English-pivot judge would then miss exactly the error classes in MorphEvalHeb's taxonomy. This is my inference from the general findings above. None of the cited papers tests it for Hebrew. A small ablation (translate summary and source to English, then run an English NLI or judge, and compare Kendall τ) would turn this inference into evidence.

### Gaps
- I found no paper that directly measures translate-then-evaluate loss for factual-consistency judgments in a Semitic language. "Found in Translation" (arXiv 2505.21999) and "Déjà Vu" (arXiv 2504.11829) appeared in search results, but I did not open their pages, so they are excluded as unverified.

## Summary table (verified items)

| Item | Venue / status | Hebrew? | Recommendation |
|---|---|---|---|
| Mu-SHROOM (Vázquez et al. 2025) | SemEval-2025 (peer-reviewed) | No (14 langs listed) | Cite |
| SHROOM-Visions 2026 overview | UncertaiNLP@EMNLP 2026 / arXiv 2608.25662 (snippet-level only) | No | Footnote at most |
| Poly-FEVER (Zhang et al. 2025) | arXiv-only | Not stated | Cite |
| HalluVerse25 (Abdaljalil et al. 2025) | arXiv-only | No (EN/AR/TR) | Cite (close method) |
| Halluverse-M^3 (Abdaljalil et al. 2026) | arXiv-only | No (EN/AR/HI/TR) | Cite (close method) |
| STEMF (Alfano et al. 2026) | Findings EACL 2026 | Not stated | Cite + baseline if weights released |
| M-Prometheus (Pombal et al. 2025) | arXiv (no venue on page) | Not stated | Baseline (open judge) + cite |
| Hercule (Doddapaneni et al. 2025) | ACL 2025 | Not stated | Cite (contrast) |
| Fu & Liu 2025 | Findings EMNLP 2025 | Not stated | Cite (motivation) |
| Doğruöz et al. 2026 | Findings EMNLP 2026 (accepted) | Not stated | Cite (motivation) |
| MM-Eval (Son et al.) | arXiv-only | Not stated | Cite (motivation) |
| Mehrabi et al. 2026 | arXiv-only (Oct 2026) | Not stated | Cite |
| SEMJ (Fu & Lu/Liu 2026) | arXiv-only | Not stated | Cite (mitigation) |
| Judge-LS (Yin 2026) | arXiv-only | No (EN/ZH) | Cite briefly |
| BASSE (Barnes et al.) | SEPLN 2026 (per arXiv comment) | No (EU/ES) | Cite (closest analogue) |
| AraHalluEval (Alansari & Luqman 2025) | ArabicNLP 2025 (Anthology PDF) | No (AR) | Cite (Semitic analogue) |
| HeSum (Paz-Argaman et al. 2024) | Findings ACL 2024 | Yes | Cite (data) |
| Dicta-LM 3.0 (Shmidman et al. 2026) | arXiv-only | Yes | Cite (current Hebrew practice) |
| Thellmann et al. 2026 | arXiv-only | Not stated | Cite (translation pitfalls) |
