# MorphEvalHeb: optimized figures

Vector rebuilds of the three PNG figures in `MorphHebEval/figures/`. The originals and the paper are unchanged.

| File | Replaces | Size (true print size) |
|---|---|---|
| `fig1_llm_error_analysis.pdf` | `llm_error_analysis_hebrew.png` | 16.1 × 20.1 cm |
| `fig2_error_taxonomy_examples.pdf` | `hebrew_error_taxonomy.png` | 16.1 × 11.6 cm |
| `fig3_augmentation_pipeline.pdf` | `dataset_creation.png` | 15.7 × 5.6 cm |

Each figure has three files: `.tex` (editable source), `.pdf` (use in the paper), and `_preview.png` (quick look only).

All three are built for ACL's 16 cm text width. Included with `width=\textwidth` they print at 1:1, so the text comes out at 8–9 pt, the same size as captions. That is the main legibility rule for paper figures.

---

## Best practices applied

1. **Vector, not raster.** PNG text blurs when zoomed or printed, and a reader can't search or copy it. The PDFs stay sharp at any zoom, and the Hebrew is real, selectable text.
2. **Figure text no smaller than the caption.** The original PNGs were shrunk to fit `\textwidth`, which put their text at roughly 4–6 pt. Figure 1 was the worst case.
3. **Font consistent with the paper.** FreeSerif is a Times clone with Hebrew glyphs, so it matches ACL's Times body text.
4. **Self-explanatory for readers who don't read Hebrew.** Most ACL reviewers don't read Hebrew, and the originals had no English for the examples. Every Hebrew example now has an English gloss of what changed.
5. **Color carries meaning, with a backup.** Blue is the source wording and orange is the changed wording; this is the Okabe-Ito colour-blind-safe pair. Both are also **bold**, so the distinction survives greyscale printing.
6. **Text-heavy content goes in tables.** Figure 1 is a table, so it should be typeset and captioned as one (see below).
7. **No vendor logos.** The pipeline diagram used OpenAI, Anthropic, LangChain and Dicta logos. Papers name tools in text; logos add visual noise and look like branding.
8. **One visual encoding per meaning.** In the pipeline, step numbers run 1–6 with no repeats, color marks a step's role, and dashed frames group the two augmentation lanes.

---

## Figure 1: LLM error analysis

**Problems in the original**
- It's a 6-column table pasted as a 1916 px image. At `\textwidth` the text is about 4 pt, which is unreadable in print.
- Everything is in Hebrew, including the error names and descriptions, so most reviewers can't read it.
- It's mostly prose. The "Error Description" and "Hebrew Unique Features" columns repeat information that belongs in the text or in an appendix.
- It's a table but is captioned as a figure.

**What changed**
- It's condensed to 5 columns: error type (English), source article text (Hebrew), LLM summary text (Hebrew), what changed (English), and LLM.
- The Hebrew text is copied word for word from the original table, with the changed words highlighted. The remaining "…" marks are in your original table: the PNG doesn't contain the rest of those sentences.
- Model names are made consistent: Gemini, DictaLM 2.0, GPT-3.5-turbo.

**Further suggestions**
- In the paper, use it as `table*` with the caption above it, and call it "Table N" in the text.
- Move the full original version, with the description and Hebrew-feature columns, to an appendix.
- Row 1 (look-alike letter: Orhan Pamuk spelled with ח instead of ה) is arguably a spelling or transliteration error rather than a factual one. Consider whether it should be in the table.

## Figure 2: taxonomy examples

**Problems in the original**
- Hebrew only, with no translation of the rules or examples.
- Raster, with small text and uneven card sizes and alignment.
- The changed word was only bolded and underlined, and in some cards only one of two changed words was marked. In the gender card, both הציגה→הציג and מחקרה→מחקרו change.
- The caption implies the full taxonomy, but it shows 8 of the 24 families.

**What changed**
- 8 equal-height cards, each containing an English rule, a source claim, a corrupted claim, and an English gloss.
- All changed words are highlighted on both sides.

**Further suggestions**
- Say "8 of the 24 families" in the caption, and add a full one-line-per-family table of all 24 families to an appendix.
- Once group membership is fixed, color-code each card header by taxonomy group (morpho-syntactic, logical, entity). I didn't do this, because the figure doesn't say which group each card belongs to.

## Figure 3: augmentation pipeline

**Problems in the original**
- Vendor logos.
- A typo: "Claim-ErrorType **Mappaing**".
- Step numbering is ambiguous: "(4)" is used three times and "(5)" twice.
- The flow zig-zags: up, right, then down through a dashed box.
- Colors have no legend or consistent meaning.
- Mixed fonts, raster output, and a lot of empty space.

**What changed**
- Flow reads left to right in two lanes:
  - negative lane: steps 1–5;
  - positive lane: 6a back-translation and 6b paraphrasing.
- Both lanes feed into the labeled contrastive set, with + and − marking which kind of example each produces.
- There's a color legend, every step has a unique number, and the tools are named in text (LLMs, DictaBERT-NER).
- The config box (GenAI model, error types, corruption level) was moved out of the diagram. Mention it in the caption or the text.
- This file needs no Hebrew, so it compiles with pdfLaTeX too. You can paste the `tikzpicture` straight into the paper instead of including the PDF.

**Please check:** I kept your step names and only reworded them. If step 4 ("combine errors & recompose") or step 5 (filtering) works differently from how the boxes describe it, edit the `.tex`.

---

## Using them in the paper

Copy the three PDFs into `MorphHebEval/figures/`, then:

```latex
% Figure 1 is a table: use table*, caption above
\begin{table*}[t]
\caption{Factual errors observed in LLM-generated Hebrew summaries. Blue: source wording; orange: the corresponding erroneous summary wording.}
\label{tab:llm_error_analysis}
\centering
\includegraphics[width=\textwidth]{fig1_llm_error_analysis.pdf}
\end{table*}

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{fig2_error_taxonomy_examples.pdf}
\caption{Eight of the 24 error families in the taxonomy. Each card gives the transformation rule, a faithful source claim (blue) and the corrupted claim (orange), with an English gloss.}
\label{fig:taxonomy}
\end{figure*}

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{fig3_augmentation_pipeline.pdf}
\caption{Data augmentation pipeline. Negative lane (1--5): claims are decomposed, mapped to error types, corrupted, recomposed and filtered. Positive lane (6): meaning-preserving rewrites. Both feed the labeled contrastive training set.}
\label{fig:pipeline}
\end{figure*}
```

If Figure 1 becomes a table, change `Figure~\ref{fig:llm_error_analysis}` in the text to `Table~\ref{tab:llm_error_analysis}`.

**Overleaf:** the paper itself still compiles with pdfLaTeX, because the PDFs are included as images. Only if you edit `fig1` or `fig2` on Overleaf do they need the XeLaTeX compiler (Menu → Compiler). Locally, run `tectonic <file>.tex`.

---

## Verify before using

- **Hebrew transcription.** The Hebrew in figs 1–2 was transcribed word for word from the PNGs. Compare it against your source data.
- **Incomplete sentences in Figure 1.** Seven rows still contain "…" because your original table cut those sentences short: rows 3, 4, 5, 7, 10, 11 and 12 (counting rows from the top). To show full sentences, paste the complete sentences from your data into the `.tex` and recompile, or send them to me. I didn't fill them in myself, because changing quoted article text would misrepresent the source.
- **English glosses.** I wrote the "What changed" column and the card glosses; check that the translations say what you mean.
- **Row 4 of Figure 1** (ומתרפק): my reading is that the summary moved a trait from one dog (Yigal) to another (Uzi). Confirm this against the article.
- **Technical note.** Highlighted Hebrew phrases are joined with non-breaking spaces (`~`). Without them, XeTeX lets the highlight color spill onto neighboring words when a phrase wraps across a line. Keep the `~` if you edit them.
