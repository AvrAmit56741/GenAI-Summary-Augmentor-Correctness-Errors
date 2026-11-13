# Citations: status and what is left to add

Status as of 2026-11-13. Sources are `MorphEvalHeb literature review update.md` (the literature report) and `Agent paper writing guidelines.md` (the statistics and reporting report), both in this folder.

## Done

All 21 papers the literature report marks **Essential** are in `overleaf/latex/custom.bib` and cited in `acl_latex.tex`:
- FactCG, TrueTeacher, Stress Testing, FineSurE, Wanner (claim decomposition);
- mFAVA, STEMF, BASSE;
- Siani (Hebrew injection), TafBERTa/HeCLiMP;
- Fu & Liu, Doğruöz, M-Prometheus;
- Dicta-LM 3.0, HalleluBERT;
- FRANK, Tang et al., LLM-Oasis, AMRFact, FAVA, Goyal & Durrett 2021.

The HeSum key is already `paz-argaman-etal-2024-hesum`.

## Still to add

### 1. Statistics and reporting methods
These support the planned meta-evaluation statistics, so add them when those results go in.

| Citation | Use |
|---|---|
| Deutsch, Dror, Roth. *A Statistical Analysis of Summarization Evaluation Metrics Using Resampling Methods.* TACL 2021 | Bootstrap confidence intervals; permutation test for "metric A beats B" |
| Deutsch, Foster, Freitag. *Ties Matter: Meta-Evaluating Modern Metrics with Pairwise Accuracy and Tie Calibration.* EMNLP 2023 | Kendall τ variant with binary labels; tie handling |
| Dror, Baumer, Shlomov, Reichart. *The Hitchhiker's Guide to Testing Statistical Significance in NLP.* ACL 2018 | Choice of significance test |
| Dror, Shlomov, Reichart. *Deep Dominance (ASO).* ACL 2019 | Comparing models across seeds |
| Shimorina, Belz. *The Human Evaluation Datasheet (HEDS).* HumEval 2022 | Human-annotation reporting |
| Bavaresco et al. *LLMs instead of Human Judges? (JUDGE-BENCH).* ACL 2025 | LLM-judge reliability must be validated per task |

### 2. Recommended (R) from the literature report, not yet in bib
These depend on paper scope. Add a citation only where the text actually uses the paper.

- **New baselines, if you run them:**
  - FaithJudge/HHEM (Tamber et al., arXiv 2505.04847);
  - LettuceDetect v1 (arXiv 2502.17125);
  - Granite Guardian is already in bib.
- **Decomposition:**
  - Molecular Facts (Gunjal & Durrett, Findings of EMNLP 2024);
  - WiCE (EMNLP 2023);
  - FENICE (ACL 2024);
  - FIZZ (EMNLP 2024);
  - Claimify (ACL 2025);
  - ElementCheck (Findings of EMNLP 2026);
  - SAFE (NeurIPS 2024);
  - VeriTrail (ICLR 2026);
  - Gray Zone of Faithfulness (arXiv 2510.21118).
- **Multilingual:**
  - HalluVerse25 (arXiv 2503.07833);
  - Hercule (ACL 2025).
- **LLM judges:**
  - Prometheus 2 (EMNLP 2024);
  - Mehrabi et al. (arXiv 2610.31172);
  - Yamauchi et al. (GEM 2026).
- **Hebrew:** DictaBERT-Parse, "MRL Parsing Without Tears" (Findings of ACL 2024). Cite it if the injection pipeline uses dictabert-joint.
- **Injection and taxonomies:**
  - FactPEGASUS (NAACL 2022);
  - HaluEval (EMNLP 2023);
  - TofuEval (NAACL 2024);
  - LongNovel (EMNLP 2026);
  - TurBLiMP (EMNLP 2025).

### 3. Bib cleanup
- `custom.bib` has 33 entries that the paper never cites. natbib prints only cited entries, so they are harmless, but delete them before release.
- Before adding any entry, fetch it from the ACL Anthology, DBLP or arXiv. Don't write BibTeX by hand.
- See "Verify before citing" in the literature report for entries whose venue or metadata is unconfirmed.
