# Evidence attribution and claim-level verification: literature review (lead 17)

Produced 2026-11-17 by a research subagent (web search). Items marked (mem) were cited from memory and
not re-verified; items marked unverified could not be confirmed. Check every citation before using it in the paper.

## Methods

| Method | Key paper | What it does | Multi-sentence evidence: + / − |
|---|---|---|---|
| Fusion-aware oracle alignment | Lebanoff et al., ACL 2019, arXiv:1906.00077 | Ranks single sentences and sentence pairs as the source of a summary sentence | + treats fusion as normal; − lexical/BERT only, no verification |
| Proposition alignment (SuperPAL) | Ernst et al., CoNLL 2021, arXiv:2009.00590; follow-up ACL 2024 Findings | Supervised summary-source proposition aligner | + many-to-many; − English, supervised |
| NLI single-sentence max (SummaC) | Laban et al., TACL 2022, arXiv:2111.09525 (mem) | Best NLI score over source sentences | − one premise sentence by construction |
| Chunk alignment (AlignScore, MiniCheck) | Zha et al., ACL 2023, arXiv:2305.16739 (mem); Tang et al., EMNLP 2024, arXiv:2404.10774 (mem) | ~350-token chunks, max per claim | + adjacent sentences; − non-adjacent evidence, no evidence set |
| Multi-hop synthetic training (FactCG) | Lei et al., NAACL 2025, arXiv:2501.17144 | Checker trained on graph-generated multi-hop claims | + built for synthesis; − chunk-level, no evidence pointer |
| Decompose then NLI (FENICE) | Scirè et al., ACL Findings 2024, arXiv:2403.02270 | Claim extraction + NLI alignment + coreference + expanding windows | + explicit evidence; − English claim extractor |
| Zoom-in/zoom-out (FIZZ) | Yang et al., EMNLP 2024, arXiv:2404.11184 | Atomic facts vs expanding window around best sentence | + cheap multi-sentence; − contiguous window |
| LongDocFACTScore | Bishop et al., LREC-COLING 2024, arXiv:2309.12455 | Embedding top-k sentences ±1 window, BARTScore | + no length limit; − generative, English |
| FActScore | Min et al., EMNLP 2023, arXiv:2305.14251 (mem) | LLM decomposition, BM25 retrieval, LM verification | + interpretable; − lexical retrieval, costly |
| HoVer | Jiang et al., Findings EMNLP 2020 (mem) | Multi-hop (2–4 documents) verification benchmark | Wikipedia, not summaries |

## Verifier attention / gradients as evidence finders
- FineGrainFact (Chan, Zeng, Ji, Findings ACL 2023, arXiv:2305.14548): closest prior work; built-in attention highlights supporting facts without span labels (not post-hoc attribution from a cross-encoder).
- FactCC/FactCCX (Kryscinski et al., EMNLP 2020): span extraction, supervised (mem).
- ContextCite (arXiv:2409.00729), AT2 (arXiv:2504.13752): attribution for generative LLMs, not encoder classifiers.
- Lookback Lens (EMNLP 2024, arXiv:2407.07071, mem): attention-ratio hallucination detection.
- Mixed evidence that attribution works: ERASER (arXiv:1911.03429, mem), Atanasova et al. EMNLP 2020 (arXiv:2009.13295, mem), FRESH (ACL 2020, mem), attention-is-(not)-explanation debate (mem).
- **No study found comparing verifier attribution with BM25/dense retrieval for summary-evidence localization.**

## Claim-level vs whole-document verification
Evidence-selected, claim-level verification (FENICE, FIZZ, MiniCheck, AlignScore) generally beats one whole-document pass on benchmarks and handles truncation; weakness: cross-sentence/coreference claims. Comparing it directly with MorphEvalHeb's whole-article design on the same Hebrew data would itself be a contribution.

## Hebrew and other morphologically rich languages
- NeoDictaBERT (arXiv:2510.20386): `dicta-il/neodictabert`, `dicta-il/neodictabert-bilingual` (retrieval-tuned; NDCG@20 0.404 on the Hebrew Semantic Retrieval challenge). `dicta-il/neodictabert-bilingual-embed`: unverified.
- DictaBERT (arXiv:2308.16687): `dicta-il/dictabert-lex` (lemmatizer), `dictabert-seg` — for morphology-aware BM25.
- HeSum (Findings ACL 2024, arXiv:2406.03897): Hebrew summaries more abstractive.
- Hebrew Semantic Retrieval challenge, 1st place (`HebArabNlpProject/Semantic-Retrieval-1st-place`): BM25 + six dense embedders + fine-tuned BGE cross-encoder (retrieval, not alignment).
- AuRED (Arabic, ArabicNLP 2024), Turkish fact-checking (LREC 2024, id from search only). No morphology-aware summary alignment.
- No Hebrew fact-verification or summary-alignment work found.

## Novelty gaps (subagent's confidence)
1. Post-hoc attribution from a faithfulness cross-encoder as an unsupervised evidence retriever, benchmarked against BM25/dense/reranker on summary alignment — medium-high.
2. Morphology-aware lexical + dense + reranker + verifier fusion for Hebrew summary-source alignment — high.
3. Do evidence finders still find the right sentence when the claim carries a morphological error or negation, and does attribution expose the flipped token? — medium-high.
4. A Hebrew gold set annotated for multi-sentence evidence, tested against SummaC's single-sentence assumption — high.
