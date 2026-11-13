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

- **Results still missing (hidden TODOs in the .tex):**
  - binary verifier results: re-run with the article-level split; the documented 96.78% has no recorded test set;
  - multi-label results for Setups 1–3.
- **Benchmark composition:** `combined.csv` = MAFAT-Aug (1,889 rows) + source-less short pairs (2,135 rows), assumed to be Hebrew-translated mFACT. Confirm, since mFACT is also a baseline.

- **Formulas (Eqs. 1–2, our own algorithm only, per guideline #P037).** `% CHECK` comments in `acl_latex.tex` mark places where the code and the paper must be made to agree:
  - the diversity threshold is updated cumulatively in `utils.filter_diverse_summaries`;
  - if nothing passes the diversity filter, all candidates are kept unfiltered;
  - the candidate cap is 51, from `count > 50`;
  - the class index meaning "faithful" differs between the docs and the training script;
  - the bootstrap test, balanced accuracy and AUC are not implemented yet.
- **Taxonomy count fixed:** 26 families defined, 20 used in the pipeline. The 6 unused families are impersonal→personal, incremental temporal state, evidentiality, verb-template agency reversal, double-verb dependency and subject–verb person mismatch. **The taxonomy figure (fig2) still shows 3 of the unused ones**, so swap those cards for used families or say so in the caption.
- **Labels are constructed, not human.** Results text now says "reference labels". If a human-annotated subset exists, report it separately.
- **Grammar verification is never called** in `correctness_errors_augmentation_v2.py`, although the old pipeline PNG implies it.
- Model versions in §4.2 (Gemini 2.5 Pro, GPT-4.1) come from the current code; confirm they match the dataset runs.

- `fig1` and the pipeline figure in `acl_latex.tex` still point at the old PNGs. Swap in `fig1_llm_error_analysis.pdf` (typeset as a table) and `fig3_augmentation_pipeline.pdf`.
- `research/Agent paper writing guidelines.md` §5 lists 11 corrections not yet applied to the guide.
- The draft's meta-evaluation table and results sections are incomplete. The planned tables and charts are: main meta-evaluation table, dataset statistics, binary and multi-label results, per-error-family chart, ablation, and a human-labeled test set of real LLM summaries.
