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

## Known open items

- **Formulas (Eqs. 1–8) are drafted.** `% CHECK` comments in `acl_latex.tex` mark places where the code and the paper must be made to agree:
  - the diversity threshold is updated cumulatively in `utils.filter_diverse_summaries`;
  - if nothing passes the diversity filter, all candidates are kept unfiltered;
  - the candidate cap is 51, from `count > 50`;
  - the class index meaning "faithful" differs between the docs and the training script;
  - the bootstrap test, balanced accuracy and AUC are not implemented yet.
- The taxonomy paragraph says "24 families", but its groups add up to 17 + 3 + 6 = 26.

- `fig1` and the pipeline figure in `acl_latex.tex` still point at the old PNGs. Swap in `fig1_llm_error_analysis.pdf` (typeset as a table) and `fig3_augmentation_pipeline.pdf`.
- `research/Agent paper writing guidelines.md` §5 lists 11 corrections not yet applied to the guide.
- The draft's meta-evaluation table and results sections are incomplete. The planned tables and charts are: main meta-evaluation table, dataset statistics, binary and multi-label results, per-error-family chart, ablation, and a human-labeled test set of real LLM summaries.
