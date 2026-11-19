# MorphEvalHeb thesis

The full-length thesis write-up, kept separate from `../paper/` (the ACL-format paper).
The thesis has its own structure (more background, related work, and chapters than
the paper allows) and will get its own Overleaf project once drafting starts.

## Relationship to paper/

- Don't duplicate content. The thesis **references** `../paper/`'s results, figures,
  and experiment log (`../docs/experiments.md`) rather than recomputing or re-deriving
  them.
- Figures/tables shared between the two should stay authored once (in
  `../paper/figures_src/` or `../paper/results/`) and be pulled into the thesis
  Overleaf project as copies, the same way the paper pulls them into its own
  `overleaf/figures/`.
- If a result or claim changes, update it at its source (`../paper/` or `../docs/`)
  first, then propagate to the thesis — not the other way around.

## Folders

| Folder | What it holds |
|---|---|
| `overleaf/` | The thesis's Overleaf project. `overleaf/latex/how_to_work_with_template.tex` documents the template this thesis is built on (see below); the actual `main.tex` and the rest of the template have not been pulled in from Overleaf yet. |
| `Example_thesis.pdf` | Compiled example output of the template (cover pages, front matter, chapter list, bibliography, Hebrew TOC), for reference while filling in `thesis_metadata.tex` / `thesis_chapters.tex` / `content/`. |
| `figures_src/` | Thesis-only figure sources (background/related-work diagrams, etc. that don't belong in the paper). |
| `research/` | Thesis-specific research notes (e.g. related-work survey, chapter outlines) that go beyond what the paper's `research/` covers. |

## The thesis template

The thesis is built on a template centered on `main.tex`, which pulls in the cover
pages, front matter, chapter list, bibliography, and Hebrew table of contents.
Source: https://www.overleaf.com/read/fxhpwgkznxdx#ca782c

Key files once the template is pulled in:
- `thesis_metadata.tex` — title (English + Hebrew), author, advisors, faculty,
  program, submission month/year. Edited once; reused across cover/abstract pages.
- `thesis_chapters.tex` — chapter and appendix order, via `\thesischapter{...}` /
  `\thesisappendix{...}` declarations pointing at files under `content/`.
- `content/.../*.tex` — the actual chapter text; `content/appendices/` for appendices;
  `content/images/` for figures.
- `Main_pages/abstract.tex`, `Main_pages/abstract_heb.tex`, `Main_pages/acknowledgements.tex`.

Workflow: edit `thesis_metadata.tex` → edit `thesis_chapters.tex` → fill in
`content/` chapter files → compile `main.tex` until TOC/bibliography/Hebrew TOC
settle. If stale TOC/Hebrew-TOC entries persist after structural changes, delete
the auxiliary build files and recompile.

## Status

As of 2026-11-19: only the template's usage instructions
(`overleaf/latex/how_to_work_with_template.tex`) have been added. `main.tex` and
the rest of the template (metadata/chapters/content files) still need to be pulled
in from the Overleaf project above before any chapter writing can start.
