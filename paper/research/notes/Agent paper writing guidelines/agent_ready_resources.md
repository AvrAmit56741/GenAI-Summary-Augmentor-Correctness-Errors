# Agent-Ready Resources for AI-Assisted Academic Paper Writing (for MorphEvalHeb, ACL LaTeX via Claude Code + tectonic + Overleaf)

Research date: 2026-11-13. Star counts are as displayed on the GitHub page at fetch time. Commit dates were mostly NOT shown by the fetch tool. Where a date is missing, the notes say so.

## Q1. Claude Agent Skills (SKILL.md) for scientific and academic writing

### Takeaway
Anthropic's official skills repo has no academic, LaTeX or paper skill. Its `doc-coauthoring` skill offers a reusable staged workflow (context, then section-by-section drafting, then "reader testing" by a fresh sub-agent), but it covers no academic conventions. The best paper-specific skill is Orchestra Research's MIT-licensed `ml-paper-writing`. It explicitly covers ACL and has a 6-step programmatic citation-verification workflow. K-Dense's `claude-scientific-skills` (MIT, very popular) adds scientific-writing, peer-review and citation-management skills, but its writing skill is aimed more at the life sciences.

### Cited Findings
**anthropics/skills (official)**
- About 194k stars and 22.8k forks. Top-level folders: `skills`, `spec`, `template`, `.claude-plugin`. Plugins are `document-skills` and `example-skills`. Many skills, including docx, pdf, pptx and xlsx, are Apache 2.0, and the repo has a THIRD_PARTY_NOTICES.md. — [anthropics/skills](https://github.com/anthropics/skills)
- The SKILL.md format is YAML frontmatter with only `name` (lowercase-hyphen) and `description` (what the skill does and when to use it) required, followed by Markdown instructions, examples and guidelines. A `template/` folder is provided. — [anthropics/skills](https://github.com/anthropics/skills)
- The `skills/` directory holds 19 skills: academy-guide, algorithmic-art, brand-guidelines, canvas-design, claude-api, discernment-nudge, doc-coauthoring, docx, frontend-design, internal-comms, mcp-builder, pdf, pptx, skill-creator, slack-gif-creator, theme-factory, web-artifacts-builder, webapp-testing, xlsx. None of them targets LaTeX or research papers. — [GitHub API listing](https://api.github.com/repos/anthropics/skills/contents/skills)
- `doc-coauthoring` has 3 stages:
  - (1) Context Gathering: Claude asks 5–10 clarifying questions about audience, impact and constraints.
  - (2) Refinement & Structure: the doc is built one section at a time. For each section Claude asks clarifying questions, brainstorms 5–20 options that the user keeps, removes or combines, drafts, then makes targeted edits.
  - (3) Reader Testing: Claude predicts 5–10 reader questions, and a fresh Claude instance with no context answers them from the document alone, checking for ambiguity, false assumptions and contradictions.
  - Rules: edit with `str_replace` and never reprint the whole doc; ask the user to describe changes so Claude learns their style; after 3 iterations with no substantial change, ask whether anything can be cut.
  - It names no academic doc types, only proposals, specs, PRDs, design docs and RFCs. No license line appears in the frontmatter.
  - Source: [doc-coauthoring SKILL.md](https://raw.githubusercontent.com/anthropics/skills/main/skills/doc-coauthoring/SKILL.md)

**Orchestra-Research/AI-Research-SKILLs: `20-ml-paper-writing`**
- About 14.5k stars and 1.0k forks. MIT license. The latest listed update is v1.7.1 (June 2026). Related skills: Academic Plotting (venue-styled matplotlib/seaborn), ARA Rigor Reviewer (scores rigor on 6 dimensions and outputs a severity-ranked report), and ARA Research Manager. — [Orchestra-Research/AI-Research-SKILLs](https://github.com/Orchestra-Research/AI-Research-SKILLs)
- SKILL.md frontmatter: `name: ml-paper-writing`, `version: 1.2.0`, `license: MIT`, with dependencies semanticscholar, arxiv, habanero and requests. It targets NeurIPS, ICML, ICLR, ACL, AAAI and COLM, and sends systems venues to a separate skill. — [SKILL.md](https://raw.githubusercontent.com/Orchestra-Research/AI-Research-SKILLs/main/20-ml-paper-writing/ml-paper-writing/SKILL.md)
- Structure: Workflow 0 starts from a research repo (explore, identify the contribution, find citations, draft). Then come a proactivity/confidence table, the What/Why/So-What narrative principle, and a paper-structure workflow (one-sentence contribution → Figure 1 → abstract → intro → methods → experiments → related work → limitations → checklist). Writing philosophy is drawn from Nanda, Farquhar, Gopen & Swan, Lipton, Steinhardt, Perez and Karpathy. The rest covers a conference-requirements table, LaTeX template handling, the citation workflow, and reviewer criteria. — [SKILL.md](https://raw.githubusercontent.com/Orchestra-Research/AI-Research-SKILLs/main/20-ml-paper-writing/ml-paper-writing/SKILL.md)
- Citation workflow, the "most important rule": (1) search Semantic Scholar; (2) confirm the paper exists in at least 2 sources (S2 plus CrossRef or arXiv); (3) fetch BibTeX from `https://doi.org/{doi}` with the `Accept: application/x-bibtex` header, never from memory; (4) confirm the attributed claim actually appears in the paper; (5) add the entry to the .bib; (6) if anything fails, write `[CITATION NEEDED]`, `[PLACEHOLDER - VERIFY]` or `\cite{PLACEHOLDER_smith2023_verify}` with a TODO, and tell the scientist. The skill states an AI-citation error rate of about 40% but gives no primary source for it. — [SKILL.md](https://raw.githubusercontent.com/Orchestra-Research/AI-Research-SKILLs/main/20-ml-paper-writing/ml-paper-writing/SKILL.md)
- Rules:
  - Farquhar's 5-sentence abstract: what you achieved, why it is hard and important, how, evidence, most remarkable result. Delete generic openings.
  - Intro of at most 1–1.5 pages with 2–4 contribution bullets.
  - Each experiment states the claim it supports and reports error bars, number of runs, hyperparameter ranges and compute.
  - A Limitations section is required, and honest limitations are not penalized.
  - Reviewers read the abstract 100% of the time, the intro 90%+, figures before methods.
  - Reference files: `writing-guide.md`, `citation-workflow.md`, `checklists.md` (includes ACL), `reviewer-guidelines.md`, `sources.md`, plus `templates/`.
  - There is no explicit "never invent numbers" rule.
  - Source: [SKILL.md](https://raw.githubusercontent.com/Orchestra-Research/AI-Research-SKILLs/main/20-ml-paper-writing/ml-paper-writing/SKILL.md)
- Forks and copies of this skill circulate: davila7/claude-code-templates (Smithery listing shows 12 installs), synthetic-sciences/openscience, OpenLAIR/dr-claw and OpenRaiser/NanoResearch. Galaxy-Dawn/claude-scholar's `ml-paper-writing` is flagged on Tessl with a Critical Snyk security finding ("do not install without review"), a 1123-line SKILL.md and broken template links. — [Tessl: claude-scholar](https://tessl.io/registry/skills/github/Galaxy-Dawn/claude-scholar/ml-paper-writing); [Smithery: davila7](https://smithery.ai/skills/davila7/ml-paper-writing); [Playbooks: orchestra 20-ml-paper-writing](https://playbooks.com/skills/orchestra-research/ai-research-skills/20-ml-paper-writing)

**K-Dense-AI/claude-scientific-skills**
- About 57.0k stars and 5.0k forks. 764 commits, latest release v2.72.0. The repo is MIT, but each skill declares its own license in its SKILL.md metadata, so check per skill. — [K-Dense-AI/claude-scientific-skills](https://github.com/K-Dense-AI/claude-scientific-skills)
- Relevant skills: Scientific Writing ("evidence-traceable writing"), Peer Review (local, confidential), Literature Review, Paper Lookup (10 academic databases), Paperclip (full-text papers with line-pinned citations), Citation Management, pyzotero, Venue Templates, Scientific Schematics, LaTeX Posters and Scientific Slides. There is no dedicated LaTeX paper-writing skill. Skills with `scripts/` have tests under `tests/<skill-name>/`. — [K-Dense-AI/claude-scientific-skills](https://github.com/K-Dense-AI/claude-scientific-skills)

### Inferences
- **Adopt with adaptation:** Orchestra `ml-paper-writing`. It is MIT, names ACL explicitly, and its citation workflow is the most concrete agent-executable one found. Vendor it into `.claude/skills/` in the paper repo. Add a "no number not traceable to a results file" rule, which it lacks, plus Hebrew/ACL-specific items such as ARR checklist and Limitations placement.
- **Adapt:** the staged workflow and the fresh-context "reader testing" sub-agent from Anthropic's `doc-coauthoring`. A context-free sub-agent answering "what is the metric, how was it validated, what is the main result?" from the PDF/tex alone works well as a clarity check for a metric paper.
- **Reference or cherry-pick:** K-Dense Citation Management, Peer Review and Paper Lookup. Their file layout was not inspected (raw paths returned 404), and the collection leans toward the life sciences.
- **Avoid:** the Galaxy-Dawn/claude-scholar copy unless you audit it, because of its security flag.

### Gaps
- Could not fetch K-Dense's `scientific-writing/SKILL.md` (path 404, layout unknown), so its exact rules (e.g., prose versus bullets, IMRAD) are unverified.
- Last-commit dates for anthropics/skills and K-Dense were not visible.
- Did not verify the contents of Orchestra's `templates/` folder, i.e., whether the ACL style files are current (acl.sty 2025/2026).

## Q2. CLAUDE.md / AGENTS.md / rules-file templates for LaTeX research papers or Overleaf

### Takeaway
The most complete, actively maintained template is `pedrohcgs/claude-code-my-workflow` (MIT, about 1.7k stars). It ships a CLAUDE.md, path-scoped `.claude/rules` for `.tex` files, skills (`/verify-claims`, `/validate-bib`, `/humanize`, `/review-paper --peer`, `/compile-latex`) and referee/verifier agents. It is built for economics and Beamer, so it needs adaptation for an ACL paper compiled with tectonic. No authoritative ACL-specific CLAUDE.md template was found.

### Cited Findings
- `pedrohcgs/claude-code-my-workflow`: about 1.7k stars, 3.2k forks, MIT, 318 commits, described as "Actively maintained". The README notes official guidance to keep CLAUDE.md under about 200 lines because adherence degrades as instruction files grow. — [pedrohcgs/claude-code-my-workflow](https://github.com/pedrohcgs/claude-code-my-workflow)
- Path-scoped rules that load when matching files are edited:
  - `verification-protocol` (.tex): task-completion checklist.
  - `quality-gates`: 0–100 scoring; 80 to commit, 90 for a PR, 95 aspirational.
  - `proofreading-protocol`: propose first, apply only with approval.
  - `post-flight-verification`: fresh-context check for factual claims.
  - `cross-artifact-review`: links paper claims to code.
  - Source: [pedrohcgs/claude-code-my-workflow](https://github.com/pedrohcgs/claude-code-my-workflow)
- Skills: `/lit-review`, `/review-paper` (`--peer` runs an editor plus domain-referee plus methods-referee pipeline), `/verify-claims` (separate verifier), `/audit-reproducibility` (matches paper numbers to code output), `/humanize` (read-only detection of AI-sounding prose), `/validate-bib`, `/respond-to-referees`, `/compile-latex` (3-pass XeLaTeX plus bibtex). Agents: proofreader, verifier, claim-verifier and a customizable `domain-reviewer`. A pre-commit hook runs 11 gates. — [pedrohcgs/claude-code-my-workflow](https://github.com/pedrohcgs/claude-code-my-workflow)
- Other CLAUDE.md sources found but not opened:
  - TakatoPhy/awesome-claude-md (curated list).
  - claudecodehq "LaTeX Conference Template Organizer" (cleans a conference template zip so it compiles on Overleaf for anonymous review).
  - "claude-academic-workflow-guide" (wentorai/research-plugins; sets compiler, natbib, build commands).
  - a-green-hand-jack/ml-research-skills `init-latex-project`.
  - One third-party guide reports Claude Code inserting wrong LaTeX escapes and recommends a CLAUDE.md line forbidding HTML-escaping of `&` and `%`. This is a low-authority source.
  - Sources: [awesome-claude-md](https://github.com/TakatoPhy/awesome-claude-md); [claudecodehq LaTeX organizer](https://www.claudecodehq.com/playbooks/latex-conference-template-organizer); [skills.sh academic workflow guide](https://skills.sh/wentorai/research-plugins/claude-academic-workflow-guide); [init-latex-project](https://www.skills.sh/a-green-hand-jack/ml-research-skills/init-latex-project); [LaTeX workflow guide](https://claude-skills-guide-3lc.pages.dev/claude-code-latex-document-workflow-2026/)

### Inferences
- **Adapt:** pedrohcgs's architecture into a short (<200 lines) paper-repo CLAUDE.md, plus path-scoped `.tex` rules, plus 3–4 skills or agents: `/verify-claims` (every number maps to a results file), `/validate-bib`, `/humanize`, `/review-paper` with an ACL/ARR-styled referee. Replace `/compile-latex` with `tectonic main.tex` and add a check that every `\cite` key resolves.
- The proofreading rule "propose first, apply with approval" fits the user's workflow of local Claude Code edits followed by Overleaf upload.

### Gaps
- No Cursor `.cursorrules`, Copilot instructions or AGENTS.md template for ACL papers was verified in this pass (search budget).
- The pedrohcgs last-commit date was not shown.

## Q3. Open-source paper-writing agents and reviewers: reusable prompts and checklists

### Takeaway
Sakana's AI Scientist v1/v2 supplies the most directly reusable prompts. v2 has a per-section writeup guide, a reflection checklist (chktex, unused or missing figures, page fit, "no results not in the logs") and Semantic-Scholar-only citation insertion. v1 has a NeurIPS-form JSON reviewer with reflection and an area-chair ensemble. The license, however, requires prominent AI disclosure in any resulting manuscript. Agent Laboratory, data-to-paper, PaperQA2 and STORM are full pipelines and less suited to section-level reuse.

### Cited Findings
- **AI Scientist v2 writeup prompt** (`perform_writeup.py`):
  - Per-section tips cover Title, Abstract (TL;DR, one paragraph), Introduction (state negative or inconclusive results frankly), Related Work ("academic siblings", contrasted), Background, Method, Experimental Setup, Experiments ("only data-supported analyses"), Conclusion and Appendix.
  - Global rules: "Accurately report the experimental results, even if they are negative or inconclusive"; "Do not change the overall style which is mandated by the conference"; "Do not hallucinate new citations or any results not in the logs."
  - Reflection loop checks: chktex output, clarity and rigor, completeness "without hallucinating", figures in the folder not used, figure refs with no file, and page-length fit.
  - Citation step: "only add citations that you have found using the [Semantic Scholar] API", with no verbatim copying and no duplicates. The model emits JSON `Description`/`Query`, then selects from the results.
  - Source: [perform_writeup.py](https://raw.githubusercontent.com/SakanaAI/AI-Scientist-v2/main/ai_scientist/perform_writeup.py)
- AI Scientist v2: about 7.5k stars. "AI Scientist Source Code License" (a Responsible-AI-License derivative) requires users to "clearly and prominently disclose the use of AI in any resulting scientific manuscripts or papers." — [SakanaAI/AI-Scientist-v2](https://github.com/SakanaAI/AI-Scientist-v2)
- **AI Scientist v1 reviewer** (`perform_review.py`):
  - Persona: "reviewing a paper submitted to a prestigious ML venue... Be critical and cautious." The default negative-bias prompt says "If a paper is bad or you are unsure, give it bad scores and reject it."
  - Output is THOUGHT plus REVIEW JSON: Summary, Strengths, Weaknesses, Questions, Limitations; Originality, Quality, Clarity, Significance (1–4); Ethical Concerns (bool); Soundness, Presentation, Contribution (1–4); Overall (1–10); Confidence (1–5); Decision (Accept/Reject).
  - Built on a `neurips_form` reviewer guide, with optional reflection rounds and an ensemble (temperature 0.75) plus an Area-Chair meta-review.
  - Up to 3 few-shot paper/review pairs.
  - Code quirks: the reflection prompt's `{current_round}` placeholders are never formatted.
  - Source: [perform_review.py](https://raw.githubusercontent.com/SakanaAI/AI-Scientist/main/ai_scientist/perform_review.py)
- **Agent Laboratory** (Schmidgall et al.): three phases (literature review, experimentation, report writing). The literature step uses the arXiv API, and human feedback is possible at each stage. — [AgentLaboratory GitHub](https://github.com/SamuelSchmidgall/AgentLaboratory); [arXiv 2501.04227](https://arxiv.org/pdf/2501.04227)
- **data-to-paper** (Kishony lab): multiple LLM and rule-based agents follow predefined research steps and aim for "transparent, traceable, verifiable" output, with paper claims traceable to the analysis. — [arXiv 2404.17605](https://web3.arxiv.org/pdf/2404.17605)
- **PaperQA2**: refactored into an Aviary environment in paper-qa v5. Reported human-level accuracy on LitQA2 literature QA, which makes it relevant for grounded related-work and citation checking. — [Aviary, arXiv 2412.21154](https://arxiv.org/pdf/2412.21154)
- **STORM** (stanford-oval/storm, NAACL 2024): generates Wikipedia-like cited articles via multi-perspective question asking; Co-STORM adds a human in the loop. MIT and about 29.9k stars per an aggregator snapshot (last updated 2025-09-30), not verified on GitHub. — [STORM paper arXiv 2402.14207](https://arxiv.org/html/2402.14207v2); [ecosyste.ms listing](https://awesome.ecosyste.ms/projects/github.com%2Fstanford-oval%2Fstorm)
- Google "PaperOrchestra" (lab notes to paper) was reported by press. — [Decrypt](https://decrypt.co/363837/googles-paperorchestra-ai-converts-lab-notes-into-publication-ready-research-papers)

### Inferences
- **Reference and copy text:** the AI Scientist v2 per-section tips, the reflection checklist and the "no results not in the logs" rule. Re-express them in your own CLAUDE.md or skill wording instead of copying the code. The license's disclosure requirement applies to "using the code"; borrowing ideas is lower-risk, but keep the ACL AI-use disclosure regardless (see Q4).
- **Adapt:** the v1 reviewer JSON schema for an ACL/ARR-styled self-review sub-agent. Swap NeurIPS fields for ARR ones (Soundness, Excitement, Overall Assessment, Reproducibility, Confidence; verify against the current ARR form). Expect a harsh-reviewer bias from the "reject if unsure" prompt, and treat it as a stress test, not a score.
- **Reference only:** Agent Laboratory, data-to-paper and STORM. They are end-to-end generators, a poor fit for a human-led paper. data-to-paper's traceability principle is worth stating as a rule.
- Possibly useful as tools: PaperQA2 (paper-qa) for verifying that each cited paper actually supports the attributed claim, which is step 4 of the Orchestra workflow.

### Gaps
- MARG (multi-agent review generation) and ReviewerGPT were not located or verified in this pass.
- License and stars for Agent Laboratory, data-to-paper and paper-qa were not fetched.
- The current ARR review form fields were not verified.

## Q4. Documented failure modes of LLM-written papers and mitigations for agent instructions

### Takeaway
Fabricated citations are the best-documented failure. 100 fabricated citations in 53 accepted NeurIPS 2025 papers were missed by 3–5 reviewers each, and ICLR 2026 desk-rejected and banned offenders. "LLM-style" vocabulary ("delve", "intricate") is measurable at corpus scale. Mitigations that agent instructions should include: programmatic citation verification, claim-support verification, numbers traceable to result files, and AI-use disclosure per ACL policy.

### Cited Findings
- Ansari (arXiv 2602.05930, Feb 2026) found 100 fabricated citations in 53 NeurIPS 2025 accepted papers (about 1%), undetected by 3–5 reviewers per paper.
  - Taxonomy: Total Fabrication 66%, Partial Attribute Corruption 27%, Identifier Hijacking 4%, Placeholder Hallucination 2%, Semantic Hallucination 1%.
  - All 100 were compound: a real-looking DOI or ID plus plausible metadata defeats heuristic checks.
  - Distribution is bimodal: 92% of affected papers had 1–2 such citations, 8% had 4–13.
  - The author recommends mandatory automated citation verification at submission.
  - Source: [arXiv 2602.05930](https://arxiv.org/abs/2602.05930)
- ICLR 2026's official retrospective: reviews likely written by an LLM were flagged, references to nonexistent publications were checked for, offending authors were banned and their submissions desk-rejected. It reports 779 desk rejections and 5,042 withdrawals. — [ICLR blog retrospective](https://blog.iclr.cc/2026/03/31/a-retrospective-on-the)
- GPTZero (a vendor with a commercial interest) reported 50 of 300 sampled ICLR 2026 submissions with at least one obvious fabricated citation. Pangram (also a vendor) estimated about 21% of ICLR 2026 reviews were fully AI-generated. A secondary blog misattributes the 50/300 figure to reviews. — [letsdatascience summary](https://letsdatascience.com/news/gptzero-finds-hallucinated-citations-in-iclr-35671028); [Pebblous blog, conflicting](https://blog.pebblous.ai/report/iclr-2026-ai-peer-review-crisis/en/)
- Liang et al. analyzed 950,965 papers (Jan 2020 to Feb 2024) using word-frequency shifts. LLM-modified content was highest in CS (up to 17.5%) and lowest in Math and Nature journals (up to 6.3%). — [arXiv 2404.01268](https://arxiv.org/html/2404.01268v1)
- Use of "delve" and similar flagged words dropped soon after they were publicized in early 2024 (peak Jan–Mar 2024). Authors adapt, so blacklisting words alone does not fix generic style. — [arXiv 2502.09606](https://arxiv.org/html/2502.09606v1)
- "delve", "intricate" and "nuanced" appear far more often in 2023–2024 scientific abstracts. — [arXiv 2412.11385](https://web3.arxiv.org/pdf/2412.11385)
- The skill encodings of mitigations:
  - Orchestra: never write BibTeX from memory; 2-source existence check; DOI BibTeX fetch; claim-in-paper check; placeholders.
  - AI Scientist v2: "Do not hallucinate new citations or any results not in the logs"; "report results even if negative."
  - pedrohcgs: `/audit-reproducibility` matches paper numbers to code output; `/humanize` flags AI-sounding prose.
  - Sources: [Orchestra SKILL.md](https://raw.githubusercontent.com/Orchestra-Research/AI-Research-SKILLs/main/20-ml-paper-writing/ml-paper-writing/SKILL.md); [perform_writeup.py](https://raw.githubusercontent.com/SakanaAI/AI-Scientist-v2/main/ai_scientist/perform_writeup.py); [pedrohcgs](https://github.com/pedrohcgs/claude-code-my-workflow)
- ACL 2023 Policy on AI Writing Assistance:
  - Content must be the original work of the named authors.
  - Purely language-level assistance (e.g., spell-checking) needs no disclosure; generative use beyond that should be described in scope and nature.
  - Use for literature search was discouraged.
  - Authors stay responsible for verifying methods and results.
  - A writing-assistant question was added to the Responsible NLP Checklist.
  - Source: [ACL 2023 policy](https://2023.aclweb.org/blog/ACL-2023-policy/)

### Inferences
- Rules recommended for the MorphEvalHeb CLAUDE.md, synthesized from the above:
  - (1) Never add a `\cite` or .bib entry from memory. Resolve it via Semantic Scholar, ACL Anthology, DOI or arXiv, and check that the claimed finding is in the source. Otherwise insert `\cite{TODO_verify_...}`.
  - (2) Every number in the text or tables must be traceable to a named results file or script in the repo. Never round, estimate or invent; if unknown, write `\todo{}`.
  - (3) No claims beyond the evidence: phrase correlation-with-human-judgment results with exact statistics and CIs, and avoid "significantly" without a test.
  - (4) Run a banned-phrase and style pass ("delve", "intricate", "pivotal", "showcase", "in the realm of", generic openers), and treat it as weak signal only.
  - (5) Fill the ACL Responsible NLP checklist AI-assistance item honestly.
- ACL Anthology BibTeX export is a natural verified source for NLP citations, a step most generic skills miss. (This is the author's inference; not checked against a source in this pass.)

### Gaps
- The current (2025–2026) ACL/ARR AI-assistance policy text was not verified. Only the 2023 policy was found, and it may have been updated.
- No primary study was found quantifying overclaiming in LLM-written papers specifically.
- The "~40% AI citation error rate" in the Orchestra skill has no traced primary source.

## Q5. Anthropic/OpenAI guidance on LLMs for long-form technical writing

### Takeaway
No dedicated Anthropic or OpenAI doc on academic or long-form paper writing was found in this pass. The closest official artifacts are Anthropic's `doc-coauthoring` skill (a staged, section-wise, reader-tested workflow) and the SKILL.md format spec. Community guidance repeats the "keep CLAUDE.md short (<200 lines)" principle.

### Cited Findings
- Anthropic's `doc-coauthoring` skill encodes Anthropic's own long-doc workflow: context interview, section-by-section brainstorm/draft/edit with `str_replace` (never reprinting the whole doc), fresh-context reader testing and a final user fact/link check. — [doc-coauthoring SKILL.md](https://raw.githubusercontent.com/anthropics/skills/main/skills/doc-coauthoring/SKILL.md)
- SKILL.md format: frontmatter with `name` and `description`, then instruction body; `template/` in the repo. — [anthropics/skills](https://github.com/anthropics/skills)
- The guidance to keep CLAUDE.md under about 200 lines because adherence degrades is relayed via the pedrohcgs README as "official guidance". — [pedrohcgs/claude-code-my-workflow](https://github.com/pedrohcgs/claude-code-my-workflow)

### Inferences
- The practical Anthropic-aligned pattern: a short CLAUDE.md for invariant rules (citations, numbers, tectonic build, ACL style untouched), skills for workflows (writing, citation verification, review), and sub-agents for fresh-context checks (reader test, claim verifier, ARR reviewer).

### Gaps
- Did not fetch docs.claude.com Agent Skills pages, Anthropic prompt-engineering docs on long-form writing, or OpenAI writing guidance (tool-call budget). The report writer should not attribute specific long-form-writing advice to Anthropic or OpenAI docs on the basis of these notes.

---
**Adoption summary for MorphEvalHeb (author's recommendation)**
| Resource | URL | License | Recommendation |
|---|---|---|---|
| Orchestra `ml-paper-writing` | https://github.com/Orchestra-Research/AI-Research-SKILLs | MIT | ADOPT and adapt (citation workflow, ACL checklist, abstract/intro rules; add numbers-traceability rule) |
| pedrohcgs claude-code-my-workflow | https://github.com/pedrohcgs/claude-code-my-workflow | MIT | ADAPT (short CLAUDE.md, .tex path rules, verify-claims, validate-bib, humanize, referee agents; swap XeLaTeX for tectonic) |
| Anthropic `doc-coauthoring` | https://github.com/anthropics/skills | not stated in skill | ADAPT (staged drafting plus fresh-context reader test) |
| AI Scientist v2 writeup and v1 reviewer prompts | https://github.com/SakanaAI/AI-Scientist-v2 , https://github.com/SakanaAI/AI-Scientist | AI Scientist Source Code License (disclosure clause) | REFERENCE (rewrite the section tips, reflection checklist and reviewer JSON in your own words, ARR-styled) |
| K-Dense claude-scientific-skills | https://github.com/K-Dense-AI/claude-scientific-skills | MIT (per-skill licenses vary) | REFERENCE / cherry-pick (Citation Management, Peer Review) |
| STORM, Agent Laboratory, data-to-paper, PaperQA2 | see Q3 | various | REFERENCE (PaperQA2 optional for claim-support checks) |
| Galaxy-Dawn claude-scholar | Tessl listing | ? | AVOID without audit (Critical Snyk flag) |
