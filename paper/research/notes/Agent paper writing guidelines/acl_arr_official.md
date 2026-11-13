# Official ACL / ACL Rolling Review (ARR) Author and Reviewer Guidelines (as of 13 Nov 2026)

Scope: the rules that apply to a long paper in ACL format submitted to ARR in late 2026 or 2027, for example to the ACL 2027 ARR deadline of 4 Jan 2027. Target paper: MorphEvalHeb, a Hebrew summary factual-consistency metric. It uses synthetic training data, fine-tuned NeoDictaBERT classifiers, LLM-judge baselines and meta-evaluation.

Resource inventory (for converting into agent instructions):

| Resource | URL | Version / date | What it is | License / reuse | Agent usability | Recommendation |
|---|---|---|---|---|---|---|
| ARR Call for Papers | https://aclrollingreview.org/cfp | page (c) 2026; "new in Oct 2026" sustainable-reviewing block | Page limits, Limitations, ethics, appendix, anonymity, resubmission, desk-reject, AI policy summary | No license stated (site content); paraphrase + link | High: rules are enumerable; convert to a pre-submit checklist | ADOPT as the authoritative rules source |
| ARR author checklist ("common submission problems") | https://aclrollingreview.org/authorchecklist | Mentions Jan 2025 formatting updates and an Apr 2025 appendix change | About 38 concrete pre-submission checks | No license stated | Very high: already a checklist | ADOPT nearly verbatim as an agent pre-flight list |
| ARR Authors page | https://aclrollingreview.org/authors | Mentions Oct 2026 policy and EMNLP 2025 checklist publication | Process, author response rules, preprints, consent | No license stated | Medium | REFERENCE |
| Responsible NLP Research Checklist | https://aclrollingreview.org/responsibleNLPresearch/ | Updated for ARR Oct 2024 cycle (Anna Rogers); page (c) 2026 | Items A1–A2, B1–B6, C1–C4, D1–D5, E1 | No license stated | Very high: fixed item IDs; filled in OpenReview form (not a PDF since Feb 2024) | ADOPT: make the agent produce a section-pointer answer per item |
| ARR Reviewer Guidelines | https://aclrollingreview.org/reviewerguidelines | Changelog last entry 22.10.2025; "updated for Oct '26 cycle" | Score meanings, lazy heuristics H1–H17, reviewer AI rules | No license stated | High: use to pre-empt reviewer objections | ADAPT into a "self-review" prompt |
| ARR Review Form | https://aclrollingreview.org/reviewform | No version stated; (c) 2026 | All fields and 1–5 scales | No license stated | High | ADAPT for a simulated review |
| ACL formatting guidelines | https://acl-org.github.io/ACLPUB/formatting.html (mirrors formatting.md in the style repo) | No date on page | Physical layout, fonts, captions, citations, Limitations/Ethics/Appendix placement, non-English text | No license stated | High | ADOPT |
| acl-org/acl-style-files | https://github.com/acl-org/acl-style-files | Last push 2026-06-29 (bst fixes, XeLaTeX README note) | acl.sty, acl_latex.tex, acl_lualatex.tex, acl_natbib.bst, anthology.bib.txt, custom.bib, formatting.md, tests | **GitHub API reports `license: null` (no LICENSE file)**; may not be modified for submissions | Direct: the agent should use it unmodified | ADOPT (use as-is; do not modify) |
| aclpubcheck | https://github.com/acl-org/aclpubcheck | Not checked | PDF format checker (margins, fonts, citations) | See repo | Very high: CLI `aclpubcheck --paper_type long paper.pdf` | ADOPT as an automated gate |
| ACL Policy on Publication Ethics (incl. "Guidelines for Generative Assistance in Authorship") | https://www.aclweb.org/adminwiki/index.php/ACL_Policy_on_Publication_Ethics | Exec approvals 2024-06-15, 2025-04-25, 2025-07-19, 2025-09-26; last edited 12 Nov 2026 | AI-use categories a–f, authorship, 10% text reuse, prompt-injection ban | No license stated; canonical PDF linked | High | ADOPT for the AI-disclosure logic |
| ACL 2023 Policy on AI Writing Assistance | https://2023.aclweb.org/blog/ACL-2023-policy/ | 2023 | Older AI policy (still linked from the ACL 2027 CFP) | n/a | Low (superseded in substance) | REFERENCE only; follow the 2025 Ethics Policy |
| ACL 2027 main CFP | https://2027.aclweb.org/calls/main/ | Current | Deadline 4 Jan 2027, theme, topics, Findings | n/a | Medium | REFERENCE |
| EACL 2027 Paper Integrity Policy | https://2027.eacl.org/calls/paper-integrity-policy/ | 2026 | GPTZero hallucinated-reference screening; desk reject + 2-year ban | n/a | Signals a trend | REFERENCE |

---

## 1. ACL formatting and style rules (style files and formatting guidelines)

### Takeaway
Use the unmodified acl-org style files with `\usepackage[review]{acl}`. Main content is limited to 8 pages for a long paper (9 in the camera-ready version), A4, two-column, Times. Limitations, Ethical Considerations, Acknowledgements, references and appendices do not count toward the limit. Hebrew text needs the LuaLaTeX/XeLaTeX template (babel with `bidi=default`), and every non-English example needs a transliteration plus an English gloss.

### Cited Findings
**Page limits**
- Long papers get up to 8 pages of content for review and 9 for the final version. Short papers get 4 for review and 5 for the final version. References are unlimited. Figures and tables in the main text count toward the limit. — [ACL formatting](https://acl-org.github.io/ACLPUB/formatting.html); [ARR CFP](https://aclrollingreview.org/cfp); [ACL 2027 CFP](https://2027.aclweb.org/calls/main/)
- A short paper is "not a shortened long paper"; it should make a small, focused contribution. Exceeding the length limit leads to desk rejection. — [ARR CFP](https://aclrollingreview.org/cfp)
- Not counted toward the limit: Limitations (required), an optional Ethics section, references, Acknowledgements and appendices. — [ARR CFP](https://aclrollingreview.org/cfp); [ACL formatting](https://acl-org.github.io/ACLPUB/formatting.html)
- The ARR author checklist says an ethical or broader-impact discussion "may extend to page 5 (short) or page 9 (long)". — [ARR author checklist](https://aclrollingreview.org/authorchecklist)

**Physical layout**
- A4 only (595.276 × 841.89 pt). Margins are 2.5 cm, with two columns of 7.7 cm and a 0.6 cm gutter. All fonts must be embedded (check with `pdffonts`; "Asian fonts must be embedded"). Submit PDF only. — [ACL formatting](https://acl-org.github.io/ACLPUB/formatting.html)
- Font sizes: Times body text at 11 pt; title 15 pt bold; section headings 12 pt bold; subsection headings 11 pt bold; captions, abstract and bibliography 10 pt; footnotes 9 pt. Non-Latin scripts and math are exempt from the font requirement. — [ACL formatting](https://acl-org.github.io/ACLPUB/formatting.html)
- The abstract is limited to 200 words of plain text. The title is plain text in title case with no LaTeX markup. Metadata entered in OpenReview or the Anthology must be plain Unicode, with no LaTeX or HTML. — [ACL formatting](https://acl-org.github.io/ACLPUB/formatting.html)
- The review version has line numbers (the ruler) and page numbers. The final version has neither. Acknowledgements are excluded from the review version. — [ACL formatting](https://acl-org.github.io/ACLPUB/formatting.html)
- **Conflict:** the ARR checklist says the review version's Acknowledgements should contain **only** generative-AI disclosures, and should be omitted entirely if no AI was used. — [ARR author checklist](https://aclrollingreview.org/authorchecklist). The formatting page says Acknowledgements are omitted in review versions. The ARR rule is newer and specific, so follow it.

**Style file options**
- `acl.sty` has three options. `review` gives anonymized output with line and page numbers. `final` is the default, with no line or page numbers. `preprint` is non-anonymous and has page numbers. hyperref is loaded by default; `nohyperref` disables it. — [acl.sty source](https://raw.githubusercontent.com/acl-org/acl-style-files/master/acl.sty)
- Authors may not modify the style files or use another conference's template. — [acl-style-files README](https://github.com/acl-org/acl-style-files)
- Follow the template's standard vertical spacing for figures, tables and titles. "Vertical space manipulation" counts as length abuse and leads to desk rejection. — [ARR author checklist](https://aclrollingreview.org/authorchecklist); [ARR CFP](https://aclrollingreview.org/cfp)
- Use pubcheck and the Overleaf template with `[review]`. The January 2025 rule updates cover figure and table font size, the placement of Limitations and Ethics, and two-column appendices. — [ARR author checklist](https://aclrollingreview.org/authorchecklist)
- aclpubcheck is the official PDF checker: `aclpubcheck --paper_type PAPER_TYPE paper.pdf`. Running it on a line-numbered review PDF produces spurious margin errors. — [acl-org/aclpubcheck](https://github.com/acl-org/aclpubcheck)

**Citations and references**
- The style is natbib-based. `\citet` gives "Author (Year)", `\citep` gives "(Author, Year)", and `\citealp` gives the same without parentheses. `\citeyearpar` is supported. `\citeposs` is ACL-specific. — [acl_latex.tex](https://raw.githubusercontent.com/acl-org/acl-style-files/master/acl_latex.tex)
- Do not use full citations as sentence constituents. With more than two authors, use "et al." Combine multiple citations with semicolons. Cite the refereed archival version actually used, and cite code and data too. — [ACL formatting](https://acl-org.github.io/ACLPUB/formatting.html)
- References go in an unnumbered "References" section placed before the appendices, ordered alphabetically. Provide DOIs where possible, otherwise ACL Anthology links. "Do not rely on automated citation indices for accuracy." — [ACL formatting](https://acl-org.github.io/ACLPUB/formatting.html)
- BibTeX entries cannot contain Unicode; use LaTeX accent commands instead. A `url` or `doi` field hyperlinks the title. — [acl_latex.tex](https://raw.githubusercontent.com/acl-org/acl-style-files/master/acl_latex.tex)
- `anthology.bib.txt` (the full ACL Anthology bib) ships in the repo. The June 2026 bst fixes changed how "and others" renders and truncate author lists longer than 20. — [acl-style-files commits](https://github.com/acl-org/acl-style-files)

**Figures and tables**
- Each figure or table must be at least one column wide, with no text wrapping around it. Use vector formats where possible. Captions go *below*, in 10 pt, as "Figure N:" or "Table N:". Text in figures and tables should be close to body size. "Abusing font size or spacing in figures/tables, or using invisible text, may lead to desk rejection." Grayscale readability is strongly encouraged, and distinctions must not rely on color alone. — [ACL formatting](https://acl-org.github.io/ACLPUB/formatting.html)
- Hyperlinks are dark blue (#000099), with no underline or box. — [ACL formatting](https://acl-org.github.io/ACLPUB/formatting.html)

**Unicode and non-Latin scripts (Hebrew)**
- "Provide an English translation for all non-English text." For non-Latin scripts, also give a Latin-script transliteration followed by an English gloss. — [ACL formatting](https://acl-org.github.io/ACLPUB/formatting.html)
- pdfLaTeX with `[T1]{fontenc}` and `[utf8]{inputenc}` is the default. LuaLaTeX and XeLaTeX are described as especially suitable for non-Latin scripts. `acl_lualatex.tex` uses `\usepackage[english,bidi=default]{babel}`, `\babelfont{rm}{TeXGyreTermesX}`, and `\babelprovide[import]{arabic}` / `{hindi}` with script-specific `\babelfont` lines. That pattern carries over directly to Hebrew. — [acl_lualatex.tex](https://raw.githubusercontent.com/acl-org/acl-style-files/master/acl_lualatex.tex); [acl_latex.tex](https://raw.githubusercontent.com/acl-org/acl-style-files/master/acl_latex.tex)

**Anonymity (paper)**
- No author names or affiliations. Cite your own work in the third person ("Smith previously showed..."). Do not point reviewers to resources they cannot access. — [ARR CFP](https://aclrollingreview.org/cfp); [ACL formatting](https://acl-org.github.io/ACLPUB/formatting.html)

**License of the style files**
- The GitHub API returns `"license": null` for acl-org/acl-style-files, and no LICENSE file is listed in the repo root. None of acl.sty, acl_latex.tex or the README states a license. — [GitHub API repo metadata](https://api.github.com/repos/acl-org/acl-style-files); [acl.sty](https://raw.githubusercontent.com/acl-org/acl-style-files/master/acl.sty)

### Inferences
- For MorphEvalHeb, use `acl_lualatex.tex` with `\babelprovide[import]{hebrew}` and a Hebrew font such as a Noto Hebrew or David CLM substitute. Wrap Hebrew examples with transliteration and gloss, for example in a gloss table in the appendix. Check embedded fonts with `pdffonts`, because Hebrew fonts count as non-Latin fonts that must be embedded.
- Because the style files have no license, an agent can use them to write the paper but should not redistribute modified copies. Link to the repo or Overleaf template instead of vendoring changed copies.
- Run aclpubcheck on a `[final]` or `[preprint]` build to avoid false margin errors, while still submitting the `[review]` build.

### Gaps
- The formatting page carries no date. The ARR checklist cites "updated January 2025" rules, but I could not find a changelog for formatting.html itself.
- I did not verify the exact Hebrew babel or font configuration against the template. Inferring it from the Arabic and Hindi examples is reasonable but untested.
- I did not open the Word template.

---

## 2. ARR CFP and author guidelines (current cycle)

### Takeaway
ARR rules are strict and enforced by desk rejection. Every paper needs a "Limitations" section, a correctly completed Responsible NLP checklist, two-column appendices that are not essential reading, anonymized supplementary material, disclosed preprint status, and checked, real references. From October 2026, a "sustainable reviewing" policy also requires a qualified service contributor; without one, the paper goes into a lottery.

### Cited Findings
**Limitations and Ethics**
- A section titled "Limitations" is mandatory for long and short papers and is placed after the conclusion and before the references. Omitting it means desk rejection. It has been mandatory since the December 2023 cycle. It may not introduce new methods, analyses or results. — [ARR CFP](https://aclrollingreview.org/cfp); [ACL formatting](https://acl-org.github.io/ACLPUB/formatting.html)
- The Ethics section is optional but encouraged. Its suggested title is "Ethical considerations", and it may contain no new experiments, figures or analysis. — [ARR CFP](https://aclrollingreview.org/cfp); [ARR author checklist](https://aclrollingreview.org/authorchecklist)
- Limitations, Ethics, references and appendices must all be in the main PDF, not in separate files. — [ARR author checklist](https://aclrollingreview.org/authorchecklist)

**Appendices and supplementary material**
- Appendices go after the references, lettered with informative titles, in two-column format; only math-heavy sections may use one column (updated April 2025). Material needed to judge novelty, claims, correctness or related work must be in the main paper. Related work placed only in an appendix counts as length evasion and can be desk rejected. — [ARR CFP](https://aclrollingreview.org/cfp); [ARR author checklist](https://aclrollingreview.org/authorchecklist); [ARR authors](https://aclrollingreview.org/authors)
- Software goes in one .tgz or .zip and data in a separate archive. Everything must be anonymized, using anonymous repositories; Google Drive and Dropbox links are not allowed. — [ARR CFP](https://aclrollingreview.org/cfp)
- Datasheets and model cards are encouraged as supplementary material. — [ARR author checklist](https://aclrollingreview.org/authorchecklist)

**Anonymity and preprints**
- There has been no anonymity period since 15 Feb 2024, and non-anonymous preprints are allowed but must be declared. Choosing "we do not intend to release" is binding until meta-reviews are out, and breaking it means desk rejection. Anonymous submissions get priority for awards and borderline decisions. — [ARR CFP](https://aclrollingreview.org/cfp); [ARR authors](https://aclrollingreview.org/authors)

**Originality, resubmission and withdrawal**
- Text reuse is limited to 10% of tokens, with no overlap in stated contributions. Concurrent related submissions must cite each other ("Anonymous (2026). Paper title. Under review.") and be uploaded anonymized. The paper may not be under review elsewhere during ARR review. — [ARR CFP](https://aclrollingreview.org/cfp); [ACL Ethics Policy](https://www.aclweb.org/adminwiki/index.php/ACL_Policy_on_Publication_Ethics)
- Thinly sliced contributions are discouraged, following ICML 2026 (new in 2025–26). — [ARR CFP](https://aclrollingreview.org/cfp)
- Resubmissions must link the previous version (including desk-rejected ones) and include a summary of changes and a point-by-point response. A meta-review score of 1.5 means skipping a cycle; a score of 1 means never resubmitting. Reframing, tweaks or added baselines do not make a paper "new". — [ARR CFP](https://aclrollingreview.org/cfp); [ARR author checklist](https://aclrollingreview.org/authorchecklist)
- Withdrawing more than 48 hours after the deadline blocks the next cycle. A paper with at least one review cannot be withdrawn and resubmitted as new. — [ARR CFP](https://aclrollingreview.org/cfp)
- Author response: one general response plus one per review, each up to 5K characters. Text only, with no links; violations mean desk rejection. Minor new experiments are allowed only when they answer a reviewer's question. — [ARR authors](https://aclrollingreview.org/authors)

**Desk-reject criteria (enumerated)**
- Violations of format, length or anonymity. Dual submission or self-plagiarism. An incorrect or misleading Responsible NLP checklist. Resubmission-policy violations. Non-two-column appendices. Length abuse (results in Limitations, related work only in the appendix, vertical-space tricks). "Hallucitations" (fabricated references). Papers clearly below the quality bar when two chairs agree. Prompt injections aimed at automated reviewers. These can happen at any point. — [ARR CFP](https://aclrollingreview.org/cfp)
- The checklist states: "Submissions attempting to manipulate machine readers will be desk rejected." It also requires removing all comments and leftover text from earlier revisions. — [ARR author checklist](https://aclrollingreview.org/authorchecklist)
- At EACL 2027, GPTZero checks every bibliography entry. Any unverifiable reference can lead to desk rejection, and all authors involved may be barred from committing to EACL 2027 and 2028. — [EACL 2027 integrity policy](https://2027.eacl.org/calls/paper-integrity-policy/)

**New in October 2026: sustainable reviewing**
- Review capacity is capped. Guaranteed review requires a qualified service contributor who vouches for the paper. Each contributor can back at most 2 submissions, which means up to 10 reviews or 16 meta-reviews. Submissions without one go into a lottery, and those not drawn are desk rejected. A registration form must be completed within 48 hours of the deadline. — [ARR CFP](https://aclrollingreview.org/cfp); [ARR author checklist](https://aclrollingreview.org/authorchecklist)
- Limits: at most 20 submissions per author and 5 as first or joint-first author. Every author needs a complete OpenReview profile (ORCID, affiliation history, DBLP/Anthology links) within 1 week of the deadline. The author list and order are frozen at submission. — [ARR CFP](https://aclrollingreview.org/cfp); [ARR author checklist](https://aclrollingreview.org/authorchecklist)
- Metadata and the checklist have a 48-hour grace period after the deadline; the PDF does not. — [ARR authors](https://aclrollingreview.org/authors)
- Scope: a paper using an LM is not automatically in scope; it must contribute to computational language processing. — [ARR author checklist](https://aclrollingreview.org/authorchecklist)

**ACL 2027 specifics**
- The ARR deadline is 4 Jan 2027 (AoE). The conference runs 20–22 Aug 2027, with workshops 18–19 Aug. Papers reviewed in the January 2027 cycle or earlier can be committed, and the author list cannot change at commitment. Rejected main-conference papers are considered for Findings. Listed topics include Summarization, "Language Diversity and NLP for Low-resourced Languages" and "Multilingualism and Cross-Lingual NLP". The theme is "Homogenization and Knowledge Collapse in LLMs". — [ACL 2027 CFP](https://2027.aclweb.org/calls/main/)
- The location is Japan, according to the ACL workshop call. — [ACL workshop call 2027](https://www.aclweb.org/portal/node/14548) (via search snippet; not fetched)

### Inferences
- For MorphEvalHeb, the agent should keep core meta-evaluation results, the synthetic-data generation method and related work in the 8 main pages. Prompts, hyperparameters, perturbation examples and per-model tables can go in two-column appendices.
- A reference-verification step, checking every bib entry against the Anthology, DOI or Semantic Scholar, is effectively mandatory now that both ARR and EACL 2027 treat hallucinated references as grounds for desk rejection.
- The agent must never insert hidden text or instructions in the PDF, including white text or "ignore previous instructions" tricks.

### Gaps
- I did not fetch the ARR dates page for the exact January 2027 cycle; the ACL 2027 CFP gives 4 Jan 2027.
- I could not confirm whether ARR itself (rather than EACL) runs GPTZero on all cycles.

---

## 3. Responsible NLP Research checklist and the AI-writing policy

### Takeaway
The checklist has 18 items in five blocks (A1–A2, B1–B6, C1–C4, D1–D5, E1). It is answered in the OpenReview form. A "Yes" must point to a section; a "No" needs a justification. Answering "yes" to everything or giving misleading answers can mean desk rejection, and checklists are published with accepted papers (resumed at ACL'25 and in the Anthology since EMNLP 2025). AI use is governed by the ACL Publication Ethics Policy (2025 revisions), which defines categories a–f. Generated text and ideas must be disclosed in the Acknowledgements; polishing and proofreading need no disclosure.

### Cited Findings
**Checklist items** (all from [Responsible NLP checklist](https://aclrollingreview.org/responsibleNLPresearch/))
- **A1** Limitations described: assumptions, scope of claims, and the definition of bias if bias is analyzed. **A2** Potential risks discussed: misuse, environmental impact, fairness, privacy, security, dual use, stakeholders.
- **B (artifacts used or created).** **B1** Cite the creators of artifacts, with original source, version and URL. **B2** State the license or terms of each artifact, and the legal basis if data was used without consent. **B3** Use must be consistent with the intended use and access conditions, including for derived artifacts. **B4** Check for PII and offensive content, and describe anonymization. **B5** Documentation covering domain, languages, demographics and linguistic phenomena (data statements, model cards, dataset cards, datasheets suggested). **B6** Statistics: counts and train/validation/test splits, even for standard benchmarks.
- **C (computational experiments).** **C1** Parameter counts, total compute budget (GPU hours) and infrastructure, for *all* experiments; estimates are acceptable. **C2** Setup, hyperparameter search and best values, including failed searches, plus the tuning set (not test). **C3** Descriptive statistics such as error bars, whether a number is a single run, max or mean, and validation results alongside test results. **C4** Implementations, versions and settings of existing packages.
- **D (human annotators or participants).** **D1** Full instructions, with screenshots and risk disclaimers. **D2** Recruitment, pay, and whether the pay is adequate for the demographic. **D3** Consent. **D4** IRB approval or exemption, without anonymity-breaking details. **D5** Annotator demographics and geography, self-reported or inferred.
- **E1** Information on any AI-assistant use. If none, justify.
- Usage rules: a "Yes" points to a section, a "No" needs a justification, and a non-"Yes" answer is not by itself grounds for rejection. Systematically missing sections or justifications can mean desk rejection. Reviewers are told not to penalize honesty. The checklist was updated for the ARR Oct 2024 cycle. It derives from NAACL 2022 and the NeurIPS 2021 checklist.
- Do not answer "yes" to everything, including irrelevant questions. — [ARR author checklist](https://aclrollingreview.org/authorchecklist)
- Checklists are published in ACL Anthology proceedings since EMNLP 2025. — [ARR authors](https://aclrollingreview.org/authors); the CFP says ACL'25 resumed publishing them — [ARR CFP](https://aclrollingreview.org/cfp)

**AI writing and assistance policy** ([ACL Policy on Publication Ethics, Guidelines for Generative Assistance in Authorship](https://www.aclweb.org/adminwiki/index.php/ACL_Policy_on_Publication_Ethics); Exec approvals 2024-06-15, 2025-04-25, 2025-07-19, 2025-09-26; last edited 12 Nov 2026)
- Authors are responsible for all content. Generative tools cannot be authors. Any generative use that *creates content*, including in rebuttals, must be disclosed in the Acknowledgements, for example "Section 3 was written with inputs from ChatGPT". Ethically sourced, open models are encouraged. "Any use not listed below is considered inappropriate use."
- (a) Language assistance (polishing your own text) and (b) short-form input assistance need no disclosure. (c) Literature search is allowed, but citation-accuracy duties still apply. (d) Low-novelty generated text must be specified, checked and cited, crediting both the text source and the idea source. (e) Model-suggested ideas need a literature search, citations and disclosure. (f) New ideas plus new text: a model cannot meet co-authorship requirements.
- Coding assistance should be disclosed in the checklist or Acknowledgements and may also be noted in a README. — [ARR CFP](https://aclrollingreview.org/cfp)
- Section 4.3.1 bans "hidden or obscured prompts or cues" meant to manipulate automated processing. The policy also covers authorship (four criteria; the first author cannot change after the deadline), and its coordinated-disclosure rules are effective from 25 Apr 2025. — [ACL Ethics Policy](https://www.aclweb.org/adminwiki/index.php/ACL_Policy_on_Publication_Ethics)
- **Version conflict:** the ACL 2027 CFP still links the 2023 AI writing policy — [ACL 2027 CFP](https://2027.aclweb.org/calls/main/); [ACL 2023 policy](https://2023.aclweb.org/blog/ACL-2023-policy/). The ARR CFP says ACL "adopted a new AI writing assistance policy" and summarizes categories a–f — [ARR CFP](https://aclrollingreview.org/cfp). The 2025 Ethics Policy is the current one.

### Inferences
- For MorphEvalHeb, sections B, C, D and E all apply:
  - B1/B2/B3: list the licenses of the source summarization corpora, NeoDictaBERT and the LLM judges, and their terms of use for generating synthetic training data. LLM output-use clauses matter here.
  - B5: write a data statement for the Hebrew synthetic dataset.
  - B6: give split statistics.
  - C1: report parameter counts and GPU hours, including API cost and LLM-judge calls.
  - C2: give the fine-tuning hyperparameter search.
  - C3: report seeds, error bars and bootstrap CIs on the meta-evaluation correlations.
  - D1–D5: needed if any human factual-consistency annotation or validation of synthetic data was done.
  - E1: disclose agent or LLM assistance in writing and coding.
- An AI agent helping write the paper falls under category (d) or (e) whenever it drafts text or ideas. Those uses must be disclosed in the review-version Acknowledgements, which may hold only AI disclosures.
- LLMs used as *research objects* (LLM-judge baselines, synthetic-data generators) are methodology, not writing assistance. Describe them in the paper; they still trigger B1–B3 and C4.

### Gaps
- I did not retrieve the exact wording of the OpenReview answer fields (Yes/No/N/A plus free text). The checklist page describes them only in prose.
- I did not read the canonical policy PDF (ACL_Publication_Ethics_Policy.pdf) beyond the wiki version.

---

## 4. ARR reviewer guidelines and the review form

### Takeaway
Reviewers score Soundness, Excitement and Overall Assessment (1–5 in half steps), plus Confidence, Reproducibility, Datasets and Software (1–5). They also guess author identity and can flag a paper for ethics review. The guidelines list 17 "lazy" heuristics (H1–H17) that are not valid grounds for rejection on their own. These include "not SOTA", "too simple", "too niche", "only tested on non-English", "language errors", "should compare to closed model X", and treating acknowledged limitations as weaknesses. Resource papers and monolingual work on any language are explicitly in scope.

### Cited Findings
- **Review form** ([ARR review form](https://aclrollingreview.org/reviewform)): free-text Paper Summary, Strengths, Weaknesses and Comments/Suggestions/Typos.
  - Confidence 1–5.
  - Soundness 1–5 (5 Excellent, 4 Strong, 3 Acceptable, 2 Poor, 1 Major Issues).
  - Excitement 1–5 (5 Highly Exciting down to 1 Not Exciting).
  - Overall Assessment: 5 Consider for Award, 4.5 Borderline Award, 4 Conference, 3.5 Borderline Conference, 3 Findings, 2.5 Borderline Findings, 2 Resubmit next cycle, 1.5 Resubmit after next cycle, 1 Do not resubmit.
  - Best paper justification.
  - Free text on Limitations and Societal Impact, Ethical Concerns and Needs Ethics Review.
  - Reproducibility 1–5 (5 "could easily reproduce" down to 1 "would not be able to").
  - Datasets 1–5 and Software 1–5 (5 Enabling, 4 Useful, 3 Potentially useful, 2 Documentary, 1 none released).
  - Author-identity knowledge 1–5.
- A low Soundness score must be justified by faults identified in the review. Excitement reflects personal taste. Overall Assessment combines soundness, excitement, novelty and impact. Since May 2026 the form has a "Declare a secondary reviewer" option. — [ARR reviewer guidelines](https://aclrollingreview.org/reviewerguidelines)
- **Lazy heuristics H1–H17:**
  - H1 not surprising; H2 contradicts my expectation; H3 not novel (a reference is required); H4 no precedent.
  - H5 doesn't beat SOTA ("neither necessary nor sufficient"; efficiency, generalizability, interpretability and fairness gains count).
  - H6 negative results; H7 too simple; H8 not my preferred methodology; H9 too niche.
  - H10 "tested only on [not English]" (monolingual work on any language has practical and theoretical value).
  - H11 language errors.
  - H12 missing reference X (only counts if the work was published 3 or more months before the deadline).
  - H13 could also do experiment X; H14 should compare to closed model X (only if it bears on the claim, given contamination and transparency issues); H15 should have done X instead.
  - H16 limitations ≠ weaknesses; H17 citation count ≠ validity.
  - — [ARR reviewer guidelines](https://aclrollingreview.org/reviewerguidelines)
- Contribution types named: models and engineering, resources (datasets and corpora), surveys, data/linguistic/social analysis, position papers, theory, reproductions and reimplementations. "A resource may not sound novel or exciting but can still be a significant contribution for its community." — [ARR reviewer guidelines](https://aclrollingreview.org/reviewerguidelines)
- Reviewers do not have to read appendices. Material that belongs in the main paper but sits in an appendix is a suggestion, not a serious weakness. Reviewers use the checklist for reproducibility (M2), ethics (T1) and licenses (T2). — [ARR reviewer guidelines](https://aclrollingreview.org/reviewerguidelines)
- Reviewer AI rules: reviewers must read the paper and write the content themselves, and may not use generative AI for a first draft. Paraphrasing help is acceptable. Manuscripts may not be uploaded to non-privacy-preserving tools. Prompt injections must be reported. Violations can bring sanctions, including temporary bans. — [ARR reviewer guidelines](https://aclrollingreview.org/reviewerguidelines); [ARR CFP](https://aclrollingreview.org/cfp)
- Authors can report reviews for lack of due diligence using issue codes I1–I12, but not for scientific disagreement. — [ARR authors](https://aclrollingreview.org/authors)
- Changelog: 22.10.2025 (sustainable reviewing), 01.05.2025 (peer-review incentives), 21.10.2024 (major update: publication ethics, closed-LLM guidance, common NLP paper problems, issue codes). — [ARR reviewer guidelines](https://aclrollingreview.org/reviewerguidelines)

### Inferences
- H10, H5, H7, H14 and H16 directly protect a Hebrew-only metric paper that is not SOTA-chasing and uses a small encoder. The agent can frame the contribution as "resource + metric + meta-evaluation" and cite the value of monolingual work. It should still avoid overclaiming generality beyond Hebrew.
- The agent should release code and data (anonymized) to target Datasets and Software scores of 4–5 ("Useful" or "Enabling") and Reproducibility of 4–5.
- To pre-empt the H14 risk, justify which closed LLM judges are compared and why, and report contamination considerations.
- A self-review pass should simulate the form: score Soundness with explicit faults, check every claim against evidence, and make sure no new results appear in Limitations.

### Gaps
- The guidelines' "common NLP paper problems" section (M- and T-codes beyond M2, T1 and T2) was not fully extracted. It is worth a follow-up fetch to get the full M/T list for a self-review rubric.
- The page has no guidance specific to evaluation-metric papers.

---

## 5. ACL guidance specific to evaluation/metric papers and dataset documentation

### Takeaway
ARR has no separate official rubric for metric or evaluation papers. The relevant obligations come from the Responsible NLP checklist: B5 documentation (data statements, datasheets, model and dataset cards), B6 statistics and C3 error bars. The author checklist "encourages" datasheets and model cards in the supplementary material.

### Cited Findings
- Checklist B5 suggests data statements, model cards, dataset cards and datasheets to document domain, languages, demographics and linguistic phenomena. — [Responsible NLP checklist](https://aclrollingreview.org/responsibleNLPresearch/)
- "Datasheets and model cards are encouraged in supplementary material." — [ARR author checklist](https://aclrollingreview.org/authorchecklist)
- Resource papers are in scope and judged on their own terms. The review form scores Datasets and Software releases separately. — [ARR reviewer guidelines](https://aclrollingreview.org/reviewerguidelines); [ARR review form](https://aclrollingreview.org/reviewform)
- C3 requires error bars or variance, and a statement of whether a number is a single run, max or mean. — [Responsible NLP checklist](https://aclrollingreview.org/responsibleNLPresearch/)

### Inferences
- For MorphEvalHeb, include a data statement for the synthetic Hebrew consistency dataset: curation rationale, language variety (Modern Hebrew, he-IL), speaker and annotator demographics if any, text characteristics and source licenses. Add a model card for the released NeoDictaBERT classifiers. For meta-evaluation, report correlation CIs (bootstrap) and system- versus summary-level results.

### Gaps
- I did not fetch the primary sources for Bender & Friedman (2018), "Data Statements for NLP" (TACL), or Gebru et al., "Datasheets for Datasets". These need a follow-up for exact template fields.
- I found no official ACL or ARR guidance specific to metric papers, for example required meta-evaluation protocols. If such guidance exists it is likely community-level (WMT Metrics shared-task conventions), not ARR policy.
