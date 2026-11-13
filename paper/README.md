# MorphEvalHeb paper

Everything related to writing the MorphEvalHeb paper (ACL format). This folder is the working copy: edit here, then upload changed files to Overleaf.

| Folder | What it holds |
|---|---|
| `overleaf/` | The Overleaf project, with the same layout as on Overleaf. `latex/acl_latex.tex` is the paper; `figures/` holds the images it includes. |
| `figures_src/` | Editable sources for the rebuilt vector figures: `.tex` source, `.pdf` for the paper, `_preview.png` for a quick look. See its README for the design rules and what to verify. |
| `guidelines/paper_writing_guidelines.md` | Our evidence-based style guide (rules #P001–#P033), built from the 18 reference papers in `docs/parsed_papers`. |
| `research/` | Research reports: the ACL/ARR rules and agent tools for writing the paper (`Agent paper writing guidelines.md`), and the literature review update. Source notes are in `research/notes/`. |

## Workflow

1. Edit `overleaf/latex/acl_latex.tex` (and figures) here.
2. When a figure changes, edit its `.tex` in `figures_src/`, recompile with `tectonic <file>.tex` (figs 1–2 need XeLaTeX), then copy the `.pdf` into `overleaf/figures/`.
3. Upload the changed files to Overleaf.
4. If you edit on Overleaf instead, download the changed files back into `overleaf/` so the two don't drift.

## Proposed figures (under review)

`figures_src/figures_review.pdf` shows all proposed figures with draft captions.
- **From real repo data:** the error-family distribution, error families per negative summary, and errors detected in real LLM summaries.
- **Placeholder scores:** per-family detection, score distributions, robustness and multi-label confusions. Regenerate them from real per-summary scores with `charts/make_charts.py --scores scores.csv`; the CSV schema is in the script's docstring.
- **Page-1 teaser** (`fig0_teaser.tex`): needs real metric scores for its example.

## Known open items

- **Results:** `results/analyze_hf_results.py` computes them from the Hugging Face prediction files and writes `results/hf_results.json`.
  - **Binary verifier test set:** AUC 0.990, balanced accuracy 96.7, τ_b 0.684, with bootstrap 95% CIs. This is now in §7.2.
  - **Setup 3:** micro-F1 0.787 (P 0.674 / R 0.945). Now in §7.3.
  - **Still missing:** Setup 1 and Setup 2 test results, which aren't in the HF repos. Setup 3 per-family F1 is also missing, because the `ground_truth_error_types` column in its `predictions.csv` is empty; it needs the server `test.csv`.
  - **Overlap:** Table 1 (`combined.csv`) shares only 8 pairs with the binary test split. Confirm `combined.csv` was not in the training data.
- **Benchmark composition:** `combined.csv` = MAFAT-Aug (1,889 rows) + source-less short pairs (2,135 rows), assumed to be Hebrew-translated mFACT. Confirm, since mFACT is also a baseline.

- **Formulas (Eqs. 1–2, our own algorithm only, per guideline #P037).** `% CHECK` comments in `acl_latex.tex` mark places where the code and the paper must be made to agree:
  - the diversity threshold is updated cumulatively in `utils.filter_diverse_summaries`;
  - if nothing passes the diversity filter, all candidates are kept unfiltered;
  - the candidate cap is 51, from `count > 50`;
  - the bootstrap test, balanced accuracy and AUC are not implemented yet.
- **Taxonomy count fixed:** 26 families defined, 20 used in the pipeline. The 6 unused families are impersonal→personal, incremental temporal state, evidentiality, verb-template agency reversal, double-verb dependency and subject–verb person mismatch. **The taxonomy figure (fig2) still shows 3 of the unused ones**, so swap those cards for used families or say so in the caption.
- **Labels are constructed, not human.** Results text now says "reference labels". If a human-annotated subset exists, report it separately.
- **Grammar verification is never called** in `correctness_errors_augmentation_v2.py`, although the old pipeline PNG implies it.
- Model versions in §4.2 (Gemini 2.5 Pro, GPT-4.1) come from the current code; confirm they match the dataset runs.

- `fig1` and the pipeline figure in `acl_latex.tex` still point at the old PNGs. Swap in `fig1_llm_error_analysis.pdf` (typeset as a table) and `fig3_augmentation_pipeline.pdf`.
- `research/Agent paper writing guidelines.md` §5 lists 11 corrections not yet applied to the guide.
- The draft's meta-evaluation table and results sections are incomplete. The planned tables and charts are: main meta-evaluation table, dataset statistics, binary and multi-label results, per-error-family chart, ablation, and a human-labeled test set of real LLM summaries.
