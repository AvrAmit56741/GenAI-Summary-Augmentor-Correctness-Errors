# Turn ACL's rules into agent guardrails

The most useful guidelines an AI agent can follow for the MorphEvalHeb paper are not writing manuals. They are the **official ARR rules, which are enforced by desk rejection**, together with a small number of MIT-licensed agent skills that make those rules executable. The ARR rules cover the 8-page limit, a mandatory "Limitations" section, two-column appendices, an 18-item Responsible NLP checklist, the 2025 ACL generative-AI policy, and a ban on fabricated references and on hidden prompts ([ARR CFP](https://aclrollingreview.org/cfp)). No reusable resource covers an ACL metric paper end to end. The practical stack has four layers. The first is the ACL/ARR pages, adopted almost verbatim as a pre-flight checklist. The second is Orchestra Research's `ml-paper-writing` skill for its citation-verification workflow, plus the CLAUDE.md, rules and sub-agent architecture of `pedrohcgs/claude-code-my-workflow` ([Orchestra](https://github.com/Orchestra-Research/AI-Research-SKILLs); [pedrohcgs](https://github.com/pedrohcgs/claude-code-my-workflow)). The third is a paraphrased set of rules from the classic guides (Peyton Jones, Widom, Lipton & Steinhardt). The fourth is meta-evaluation statistics taken from Deutsch, Dror & Roth (TACL 2021). Our own guide (`paper_writing_guidelines.md`) is strong on genre conventions, because it was mined from 18 reference papers. It is weak on venue compliance and statistics. It sets an abstract range that breaks the **200-word ACL limit**. It treats Hebrew glossing as optional, although ACL makes it mandatory. It puts release links on page 1, which risks breaking anonymity. It is silent on AI-use disclosure, on the Responsible NLP checklist, and on which significance test to use. The ACL 2027 ARR deadline is **4 January 2027**, roughly seven weeks from now ([ACL 2027 CFP](https://2027.aclweb.org/calls/main/)), so these fixes should be packaged now as a short CLAUDE.md plus a paper-writing skill inside `~/Downloads/MorphHebEval`.

## ARR rules are the non-negotiable layer, and most are checkable

A long paper gets **8 pages of content for review and 9 for the camera-ready version**. Figures and tables count toward the limit. References, Limitations, an optional Ethics section and appendices do not ([ACL formatting](https://acl-org.github.io/ACLPUB/formatting.html); [ARR CFP](https://aclrollingreview.org/cfp)). The formatting rules are precise enough to enforce mechanically:

- A4 paper, two columns, Times 11 pt, captions placed *below* figures and tables at 10 pt.
- Hyperlinks in dark blue.
- All fonts embedded, including non-Latin fonts.
- An abstract of **at most 200 words**.
- A plain-text title in title case.
- Every figure and table at least one column wide.
- Distinctions that do not rely on color alone ([ACL formatting](https://acl-org.github.io/ACLPUB/formatting.html)).

The style files must be used unmodified. "Vertical space manipulation" and shrinking fonts in figures count as length abuse, which is grounds for desk rejection ([acl-style-files](https://github.com/acl-org/acl-style-files); [ARR author checklist](https://aclrollingreview.org/authorchecklist)). The checker for these rules is `aclpubcheck --paper_type long paper.pdf`. Run it on a `final` or `preprint` build, because the review build's line-number ruler produces false margin errors ([aclpubcheck](https://github.com/acl-org/aclpubcheck)).

The **"Limitations" section is mandatory**. It is unnumbered in practice, sits after the conclusion and before the references, and **may not introduce new results**. The Ethics section is optional; its suggested title is "Ethical considerations" ([ARR CFP](https://aclrollingreview.org/cfp)). Appendices follow the references, are lettered, and use **two columns**. Anything needed to judge novelty or correctness, including related work, must stay in the main body. Related work placed only in an appendix counts as length evasion ([ARR author checklist](https://aclrollingreview.org/authorchecklist)). Supplementary code and data must be anonymized, and Google Drive or Dropbox links are not allowed ([ARR CFP](https://aclrollingreview.org/cfp)).

The listed desk-reject triggers are:

- format violations;
- an incorrect or misleading Responsible NLP checklist;
- **"hallucitations"** (fabricated references);
- **prompts hidden in the PDF to manipulate machine readers**;
- leftover comments from earlier revisions ([ARR CFP](https://aclrollingreview.org/cfp); [ARR author checklist](https://aclrollingreview.org/authorchecklist)).

EACL 2027 goes further. It screens every bibliography entry with GPTZero and can bar the authors from the next two EACL cycles ([EACL 2027 integrity policy](https://2027.eacl.org/calls/paper-integrity-policy/)). One rule is new in October 2026 and matters for planning rather than writing. Guaranteed review requires a **qualified service contributor**. Papers without one enter a lottery, and those not drawn are desk-rejected ([ARR CFP](https://aclrollingreview.org/cfp)).

The **Responsible NLP checklist** has 18 items: A1–A2, B1–B6, C1–C4, D1–D5 and E1. They are answered in the OpenReview form. Each "Yes" must point to a section, and each "No" needs a justification. Answering "yes" across the board is itself flagged as a problem ([Responsible NLP checklist](https://aclrollingreview.org/responsibleNLPresearch/); [ARR author checklist](https://aclrollingreview.org/authorchecklist)). For MorphEvalHeb the items that apply are:

- **B1–B3:** licenses and terms for Israel Hayom, Wikipedia, Knesset and Bagatz text, NeoDictaBERT, and the output terms of every LLM used to generate training data.
- **B5:** a data statement.
- **B6:** split statistics.
- **C1:** parameter counts and GPU hours, plus API calls.
- **C2:** the hyperparameter search, including searches that failed.
- **C3:** error bars, and whether each number is a single run, a max or a mean.
- **C4:** package versions.
- **D1–D5:** only if humans annotate.
- **E1:** AI-assistant use.

The checklists of accepted papers are now published in the Anthology ([ARR authors](https://aclrollingreview.org/authors)), so the answers become part of the public record.

The **AI-writing policy** that governs the paper is the 2025 ACL Publication Ethics Policy, not the 2023 blog post that the ACL 2027 CFP still links ([ACL Ethics Policy](https://www.aclweb.org/adminwiki/index.php/ACL_Policy_on_Publication_Ethics); [ACL 2023 policy](https://2023.aclweb.org/blog/ACL-2023-policy/)). The policy sorts AI use into categories:

| Category | Use | Disclosure |
|---|---|---|
| (a) | Polishing the authors' own text | None |
| (b) | Short-form input assistance | None |
| (c) | Literature search | None, but the authors remain responsible for citation accuracy |
| (d) | Generated low-novelty text | Must be checked, cited and disclosed |
| (e) | Model-suggested ideas | Must be checked, cited and disclosed |

Disclosures go in the Acknowledgements. In the review version, the Acknowledgements section should contain **only the AI disclosure**, and it should be omitted entirely if no AI was used. This ARR rule is newer than the formatting page's "omit Acknowledgements in review" ([ARR author checklist](https://aclrollingreview.org/authorchecklist)). Coding assistance should also be disclosed ([ARR CFP](https://aclrollingreview.org/cfp)). An agent that drafts sections therefore falls under (d) or (e), and the agent itself should keep a log of what it drafted.

The **reviewer guidelines** work best as a defensive tool. Reviewers score Soundness, Excitement and Overall (1–5 in half steps), plus Reproducibility, Datasets and Software ([ARR review form](https://aclrollingreview.org/reviewform)). The guidelines list 17 "lazy heuristics" that are not on their own grounds for rejection. Four of them protect this paper: **H10 "tested only on [not English]"**, H5 "doesn't beat SOTA", H14 "should compare to closed model X", and H16 "limitations ≠ weaknesses". H12 limits "missing reference" complaints to work published at least three months before the deadline ([ARR reviewer guidelines](https://aclrollingreview.org/reviewerguidelines)).

| Resource | URL | Contents | License / reuse | Agent usability | Recommendation |
|---|---|---|---|---|---|
| ARR CFP | https://aclrollingreview.org/cfp | Limits, Limitations/Ethics, appendices, anonymity, desk-reject list, AI policy summary, sustainable reviewing | No license stated; paraphrase and link | High (enumerable) | **Adopt** as the rules source |
| ARR author checklist | https://aclrollingreview.org/authorchecklist | About 38 pre-submission checks | No license stated | Very high (already a checklist) | **Adopt** almost verbatim as a pre-flight list |
| Responsible NLP checklist | https://aclrollingreview.org/responsibleNLPresearch/ | A1–E1 items with fixed IDs | No license stated | Very high | **Adopt**: the agent drafts a section pointer for each item |
| ACL formatting guidelines | https://acl-org.github.io/ACLPUB/formatting.html | Layout, fonts, citations, non-English text, abstract ≤200 words | No license stated | High | **Adopt** |
| acl-style-files | https://github.com/acl-org/acl-style-files | acl.sty, acl_latex.tex, acl_lualatex.tex, bst | `license: null`; use as-is, never modify | Direct | **Adopt unmodified** |
| aclpubcheck | https://github.com/acl-org/aclpubcheck | PDF format checker CLI | See repo | Very high (scriptable gate) | **Adopt** |
| ACL Publication Ethics Policy | https://www.aclweb.org/adminwiki/index.php/ACL_Policy_on_Publication_Ethics | AI-use categories a–f, authorship, 10% text-reuse limit, prompt-injection ban | No license stated | High | **Adopt** for disclosure logic |
| ARR reviewer guidelines + review form | https://aclrollingreview.org/reviewerguidelines , https://aclrollingreview.org/reviewform | Score scales, H1–H17 heuristics | No license stated | High | **Adapt** into a self-review sub-agent |
| ACL 2023 AI policy | https://2023.aclweb.org/blog/ACL-2023-policy/ | Older policy | n/a | Low | **Reference** only (superseded) |
| EACL 2027 integrity policy | https://2027.eacl.org/calls/paper-integrity-policy/ | GPTZero screening of references; bans | n/a | Signal | **Reference** |

## Two MIT-licensed agent projects cover citations and claim tracing

Anthropic's official `anthropics/skills` repository has **no LaTeX or research-paper skill** among its 19 skills ([GitHub API listing](https://api.github.com/repos/anthropics/skills/contents/skills)). Its `doc-coauthoring` skill still offers a workflow worth borrowing, in three stages:

1. A context interview.
2. Drafting one section at a time, using targeted `str_replace` edits instead of reprinting the document.
3. **"Reader testing"**: a fresh Claude instance with no prior context answers predicted reader questions from the document alone, to surface ambiguity ([doc-coauthoring SKILL.md](https://raw.githubusercontent.com/anthropics/skills/main/skills/doc-coauthoring/SKILL.md)).

For a metric paper this becomes a clean clarity test. Can a context-free agent say, from the PDF alone, what MorphEvalHeb measures, how it was validated, and what the headline number is?

The best paper-specific skill is Orchestra Research's **`ml-paper-writing`**. It is **MIT-licensed** and names ACL explicitly. It encodes a six-step citation workflow:

1. Search Semantic Scholar.
2. Confirm that the paper exists in at least two sources.
3. Fetch the BibTeX from `doi.org` with `Accept: application/x-bibtex`, **never from memory**.
4. Check that the attributed claim actually appears in the source.
5. Add the entry to the `.bib` file.
6. If any step fails, leave a visible placeholder such as `\cite{PLACEHOLDER_..._verify}`.

The skill also covers Farquhar's five-sentence abstract, an introduction of at most 1–1.5 pages with 2–4 contribution bullets, and "every experiment states the claim it supports" ([Orchestra SKILL.md](https://raw.githubusercontent.com/Orchestra-Research/AI-Research-SKILLs/main/20-ml-paper-writing/ml-paper-writing/SKILL.md)). It lacks a rule that every number must trace to a results file. That gap matters most for a paper whose substance is a results table.

`pedrohcgs/claude-code-my-workflow` (MIT, about 1.7k stars) supplies the architecture rather than the content:

- a CLAUDE.md kept **under about 200 lines**, because instruction adherence degrades as the file grows;
- path-scoped `.claude/rules` for `.tex` files;
- the skills `/verify-claims`, `/validate-bib`, `/audit-reproducibility` (paper numbers checked against code output), `/humanize` (read-only detection of AI-sounding prose) and `/review-paper --peer`;
- a "propose first, apply with approval" proofreading rule ([pedrohcgs](https://github.com/pedrohcgs/claude-code-my-workflow)).

It targets economics and Beamer, so the referee persona and the build command need replacing.

Sakana's AI Scientist v2 writeup prompt contributes three compact rules worth restating in our own words:

- "Do not hallucinate new citations or any results not in the logs."
- "Accurately report the experimental results, even if they are negative."
- A reflection checklist: chktex warnings, figures that are never referenced, references to figure files that do not exist, and page fit.

Its v1 reviewer outputs NeurIPS-style JSON and runs as an area-chair ensemble ([perform_writeup.py](https://raw.githubusercontent.com/SakanaAI/AI-Scientist-v2/main/ai_scientist/perform_writeup.py); [perform_review.py](https://raw.githubusercontent.com/SakanaAI/AI-Scientist/main/ai_scientist/perform_review.py)). Its license **requires prominent AI disclosure in any manuscript produced with the code** ([AI-Scientist-v2](https://github.com/SakanaAI/AI-Scientist-v2)). Borrow its ideas, not its code. Its reviewer prompt says "if unsure, reject", so treat it as a stress test rather than a calibrated score. A copy of `ml-paper-writing` circulating as Galaxy-Dawn/claude-scholar carries a **Critical Snyk flag** and should not be installed without an audit ([Tessl listing](https://tessl.io/registry/skills/github/Galaxy-Dawn/claude-scholar/ml-paper-writing)).

The failure modes these tools guard against are documented. Ansari found **100 fabricated citations across 53 accepted NeurIPS 2025 papers**, each missed by 3–5 reviewers. **66%** were total fabrications, and all of them combined a real-looking identifier with plausible metadata, which defeats heuristic checks ([arXiv 2602.05930](https://arxiv.org/abs/2602.05930)). ICLR 2026 desk-rejected submissions that cited nonexistent publications and banned the offending authors ([ICLR retrospective](https://blog.iclr.cc/2026/03/31/a-retrospective-on-the)). LLM-style vocabulary ("delve", "intricate") is measurable at corpus scale ([arXiv 2412.11385](https://web3.arxiv.org/pdf/2412.11385)). Its frequency dropped soon after being publicized, though ([arXiv 2502.09606](https://arxiv.org/html/2502.09606v1)). A banned-word list is therefore only a weak lint. The defenses that actually work are verification steps: citation existence, claim support, and number provenance. Style policing does not substitute for them.

| Resource | URL | Contents | License | Agent usability | Recommendation |
|---|---|---|---|---|---|
| Orchestra `ml-paper-writing` | https://github.com/Orchestra-Research/AI-Research-SKILLs | Paper workflow, ACL requirements table, 6-step citation verification, reviewer criteria | MIT | Very high (drop-in SKILL.md) | **Adopt + adapt**: add a number-traceability rule and ARR/Hebrew items |
| pedrohcgs claude-code-my-workflow | https://github.com/pedrohcgs/claude-code-my-workflow | CLAUDE.md, `.tex` rules, verify-claims, validate-bib, humanize, referee agents, pre-commit gates | MIT | High (needs domain swap) | **Adapt** the architecture |
| Anthropic `doc-coauthoring` | https://github.com/anthropics/skills | Staged drafting + fresh-context reader test | Repo mixes Apache-2.0 skills; none stated in this skill | High | **Adapt** the reader-test step |
| AI Scientist v2 writeup / v1 reviewer | https://github.com/SakanaAI/AI-Scientist-v2 , https://github.com/SakanaAI/AI-Scientist | Section tips, reflection checklist, JSON reviewer | AI Scientist license with a disclosure clause | Medium | **Reference**: rewrite in our own words, ARR-styled |
| K-Dense claude-scientific-skills | https://github.com/K-Dense-AI/claude-scientific-skills | Citation Management, Peer Review, Paper Lookup | MIT (licenses vary per skill) | Medium (life-science lean) | **Reference / cherry-pick** |
| PaperQA2, STORM, Agent Laboratory, data-to-paper | https://arxiv.org/pdf/2412.21154 , https://arxiv.org/html/2402.14207v2 , https://github.com/SamuelSchmidgall/AgentLaboratory , https://web3.arxiv.org/pdf/2404.17605 | End-to-end pipelines; PaperQA2 for grounded claim checks | Various | Low for a human-led paper | **Reference** (PaperQA2 optional for claim-support checks) |
| Galaxy-Dawn claude-scholar | https://tessl.io/registry/skills/github/Galaxy-Dawn/claude-scholar/ml-paper-writing | Copy of ml-paper-writing | Unclear | n/a | **Avoid** without an audit |

## Classic guides agree on one introduction shape and on honesty rules

Peyton Jones, Widom, Karpathy, Neubig and Freeman agree on a single introduction template:

1. The problem.
2. Why it matters.
3. Why it is hard or still unsolved.
4. Our idea.
5. A **refutable contribution list that forward-references the supporting section**.

The guides also rule out a "rest of this paper is organized as follows" roadmap and a personal story of discovery ([SPJ slides](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/07/How-to-write-a-great-research-paper.pdf); [Widom](https://cs.stanford.edu/people/widom/paper-writing.html)). Two rules are agent-checkable as written. Peyton Jones: "check each claim in the introduction, identify the evidence, and forward-reference it." Neubig: "make sure every claim promised in the introduction is fulfilled" ([Neubig mirror](https://api.sambuz.com/doc/how-to-read-write-an-international-conference-paper-ppt-presentation-705699)).

Lipton & Steinhardt supply the honesty rules:

- Separate speculation from explanation, ideally quarantining speculation in a motivation section.
- Use ablations to attribute where the gains come from.
- Avoid suggestive or overloaded terminology ([arXiv 1807.03341](https://arxiv.org/pdf/1807.03341)).

This applies directly to MorphEvalHeb's phrase "baselines are not morphology-aware". Without an ablation it is a hypothesis and should be worded as one.

Steinhardt's blog and Widom provide sentence-level lints:

- name the metric instead of writing "performance";
- keep terminology consistent;
- no "for various reasons";
- no nonreferential "this";
- put `\label` after `\caption` ([Steinhardt](https://jsteinhardt.stat.berkeley.edu/blog/advice-for-authors); [Widom](https://cs.stanford.edu/people/widom/paper-writing.html)).

Gopen & Swan contribute a revision pass, not a drafting rule: keep the subject next to its verb, put old information in the topic position and new information in the stress position ([Gopen & Swan](https://cseweb.ucsd.edu/~swanson/papers/science-of-writing.pdf)). Freeman adds captions that "tell the reader what to notice" and advises against ending on future work ([Freeman slides](https://www.cs.torontomu.ca/~wangcs/resources\How-to-write-a-good-CVPR-submission.pdf)). The **Bender Rule** ("always name the language") is now built into the ARR checklist ([The Gradient](https://thegradient.pub/the-benderrule-on-naming-the-languages-we-study-and-why-it-matters/)).

The guides disagree on one point, where related work goes. Peyton Jones puts it late. Widom makes it conditional. Karpathy puts it right after the introduction. Our current §2 placement is defensible as long as the section stays short.

None of these guides is Creative Commons licensed. Peyton Jones explicitly permits reuse with acknowledgement ([MSR page](https://www.microsoft.com/en-us/research/academic-program/write-great-research-paper/)). Gopen & Swan is © Sigma Xi. Lipton & Steinhardt carries arXiv's distribution-only license. The safe pattern is to **paraphrase each rule in our own words with a link**, not to vendor the text.

| Guide | URL | License | Agent usability | Recommendation |
|---|---|---|---|---|
| Peyton Jones | https://www.microsoft.com/en-us/research/academic-program/write-great-research-paper/ | Reuse with acknowledgement | High | **Adopt** (paraphrase) |
| Lipton & Steinhardt | https://arxiv.org/abs/1807.03341 | arXiv non-exclusive | High (honesty lints) | **Adopt** (paraphrase, cite) |
| Widom | https://cs.stanford.edu/people/widom/paper-writing.html | None stated | High (style lints) | **Adopt** (paraphrase) |
| Steinhardt blog | https://jsteinhardt.stat.berkeley.edu/blog/advice-for-authors | © author | High | **Adopt** selected rules |
| Gopen & Swan | https://cseweb.ucsd.edu/~swanson/papers/science-of-writing.pdf | © Sigma Xi | Medium (revision pass) | **Adapt** |
| Freeman (Kajiya/Adelson) | https://www.cs.torontomu.ca/~wangcs/resources\How-to-write-a-good-CVPR-submission.pdf | Mirror only | Medium-high | **Adapt** |
| Neubig 2015 | https://api.sambuz.com/doc/how-to-read-write-an-international-conference-paper-ppt-presentation-705699 | Mirror only, possibly garbled | High | **Adapt** |
| Karpathy | http://karpathy.github.io/2016/09/07/phd/ | None stated | Medium | **Adapt** selected rules |
| Eisner | https://www.cs.jhu.edu/~jason/advice/write-the-paper-first.html | None stated | Process-level | **Reference** |
| Bender Rule | https://thegradient.pub/the-benderrule-on-naming-the-languages-we-study-and-why-it-matters/ | None stated | High | **Adopt** (already enforced by ARR) |

## Meta-evaluation statistics should follow Deutsch, not Williams

The authoritative reference for comparing summarization metrics is **Deutsch, Dror & Roth (TACL 2021)**. They recommend:

- **Boot-Both** bootstrap for confidence intervals;
- **Perm-Both** permutation test for hypothesis tests on the difference between two metrics.

They found that the **Williams test had near-zero power and assumes normality**, an assumption summarization data often violates. In one setting, a nominal 95% Boot-Inputs interval was effectively **only a 68% interval** on held-out data ([ar5iv 2104.00054](https://ar5iv.arxiv.org/html/2104.00054)).

MorphEvalHeb's benchmark labels individual summaries and has no multi-system ranking axis. That setup reduces to resampling summaries, paired across metrics, for both the interval and the permutation test. If the test summaries come from several summarizers or translation pipelines, the resampling should respect that grouping, because ignoring an axis that varies is exactly what produced the under-coverage. For seed variance in the fine-tuned classifiers, Dror et al.'s **almost-stochastic-dominance** test (`deepsig`) compares score distributions across runs, not single scores ([ACL Anthology P19-1266](https://aclanthology.org/P19-1266/)). Dror et al.'s Hitchhiker's Guide is the general protocol reference ([P18-1128](https://aclanthology.org/P18-1128/)). Correcting for multiple comparisons across about ten baselines and 24 error families (e.g., Holm) is standard statistical practice. None of the sources reviewed here prescribes it.

For binary labels, the factuality literature reports **ROC AUC** (TRUE, threshold-free) and **balanced accuracy** with a threshold tuned on dev (SummaC, LLM-AggreFact), broken down per dataset ([TRUE](https://ar5iv.labs.arxiv.org/html/2204.04991); [SummaC](https://preview.aclanthology.org/setup/2022.tacl-1.10); [MiniCheck](https://arxiv.org/html/2404.10774v2)). Our guide already reached this conclusion independently (#P013). It should add two rules: every metric uses the **same** threshold protocol, and no threshold is ever tuned on test.

Kendall τ deserves care. A binary metric output produces massive ties, and Kendall variants handle ties inconsistently and can be gamed. "Ties Matter" proposes pairwise accuracy with tie calibration, and WMT24 adopted it at segment level ([Deutsch, Foster & Freitag 2023](https://aclanthology.org/2023.emnlp-main.798); [WMT24](https://www2.statmt.org/wmt24/metrics-task.html)). Any τ that stays in the paper must name its variant.

Three further sources shape the reporting:

- **Human labels.** HEDS (the Human Evaluation Datasheet) is the documentation template for any human labeling ([HEDS](https://aclanthology.org/2022.humeval-1.6)). Howcroft et al. found over 200 different terms for quality criteria, which argues for mapping our error families onto established names such as FRANK's rather than inventing new ones ([Howcroft et al.](https://aclanthology.org/2020.inlg-1.23)).
- **Label noise.** Tang et al. estimate that **at least 6% of AggreFact is mislabeled** ([arXiv 2305.14540](https://arxiv.org/pdf/2305.14540)), so even our gold labels need an audited sample.
- **LLM judges.** JUDGE-BENCH found that judge reliability varies widely by task and text source and must be validated against human labels per task ([Bavaresco et al.](https://arxiv.org/abs/2406.18403v2)). Chiang & Lee showed that number-only judge outputs are suboptimal and that explanations help ([2023.findings-emnlp.599](https://aclanthology.org/2023.findings-emnlp.599)).

The paper's GPT-4o-mini G-Eval and vanilla-judge baselines therefore need the following reported:

- the verbatim prompts;
- the exact model ID and access date;
- the decoding parameters;
- the parsing method and parse-failure rate.

They also need a check for **same-family leakage**: the LLMs that generated the error-injected data overlap in vendor with the judge. No published reporting checklist exists for LLM judges. These items are inferred from the empirical findings.

On reproducibility, ARR C1–C3 and B6 are mandatory, and Dodge et al. add reporting the search budget ([Show Your Work](https://aclanthology.org/D19-1224/)). Pair them with a Data Statement for the Hebrew corpus and a Model Card for the released classifiers ([Model Cards](https://arxiv.org/pdf/1810.03993v1)).

| Resource | URL | License | Agent usability | Recommendation |
|---|---|---|---|---|
| Deutsch, Dror & Roth 2021 | https://aclanthology.org/2021.tacl-1.67/ | ACL Anthology (open access) | High (concrete test choice; code at CogComp/stat-analysis-experiments) | **Adopt** |
| Dror et al. 2018 / 2019 | https://aclanthology.org/P18-1128/ , https://aclanthology.org/P19-1266/ | ACL Anthology | High (`deepsig`) | **Adopt** |
| TRUE / SummaC / MiniCheck conventions | see links above | Papers | High | **Adopt** AUC + BAcc |
| Ties Matter; WMT24 | https://aclanthology.org/2023.emnlp-main.798 , https://www2.statmt.org/wmt24/metrics-task.html | Papers | Medium | **Adopt** for any rank correlation |
| HEDS; van der Lee et al. | https://aclanthology.org/2022.humeval-1.6 , https://aclanthology.org/W19-8643/ | Papers / templates | High (fill-in form) | **Adopt** if humans annotate |
| JUDGE-BENCH; Chiang & Lee | https://arxiv.org/abs/2406.18403v2 , https://aclanthology.org/2023.findings-emnlp.599 | Papers | Medium | **Adapt** into a judge-reporting checklist |
| Dodge 2019; Model Cards; Data Statements | https://aclanthology.org/D19-1224/ , https://arxiv.org/pdf/1810.03993v1 | Papers | High | **Adopt** |
| Gehrmann et al. 2023; Berg-Kirkpatrick 2012 | https://researchr.org/publication/GehrmannCS23 , https://aclanthology.org/D12-1091/ | Papers | Low (motivation) | **Reference** |

## Our guide needs eleven corrections and a compliance layer

Every #P rule in `paper_writing_guidelines.md` is backed by counts from 18 papers. That makes it the best available source on *what accepted factuality papers look like*. It never asks *what ARR will desk-reject*, though, and in a few places the genre evidence contradicts the venue rules. Checking the draft (`MorphHebEval/latex/acl_latex.tex`) against the research turned up several concrete issues the guide misses:

- **`\cite` is used where `\citep` is needed.** In the ACL natbib style `\cite` renders as a textual citation, so "ROUGE-1 \cite{lin2004rouge}" prints "ROUGE-1 Lin (2004)".
- **Limitations is numbered and placed before the Conclusion.**
- **Hebrew is set with `cjhebrew` under pdfLaTeX**, not with the `acl_lualatex.tex` babel pattern that ACL describes as suited to non-Latin scripts.
- **`anthology.bib.txt` is only a stub** that points to aclanthology.org.
- **`references.bib` is unused** because only `custom.bib` is loaded.

| # | Change | Type | Basis |
|---|---|---|---|
| 1 | **#P002:** abstract "150–210 words" → **at most 200 words**, plain text. The 143-word draft is fine; AlignScore's 214 is not a model to copy. | Correction | [ACL formatting](https://acl-org.github.io/ACLPUB/formatting.html) |
| 2 | **#P022:** glossing "where morphology matters" → **every** Hebrew string gets an English translation, and in-text examples also get a Latin transliteration and a gloss. HeQ's untranslated tables are an anti-pattern, not an option. Check that fonts are embedded with `pdffonts`. | Correction | [ACL formatting](https://acl-org.github.io/ACLPUB/formatting.html) |
| 3 | **#P030:** release links on page 1 → in the review build, use **anonymized repositories only** (no GitHub org, HF user, Drive or Dropbox) and cite self-work in the third person. Real links go in the camera-ready version. The license statement stays (it feeds B2). | Correction | [ARR CFP](https://aclrollingreview.org/cfp) |
| 4 | **#P017:** "paired bootstrap" → Boot-Both-style (or summary-resampling) CIs with ≥1,000 resamples stated; **paired permutation (Perm-Both analogue)** for "MorphEvalHeb beats X"; **never the Williams test alone**; Holm correction across baselines × families; ≥3 seeds with mean ± std, compared with ASO. Promote seed variance from "appendix nicety" to C3 compliance. | Correction + extension | [Deutsch 2021](https://ar5iv.arxiv.org/html/2104.00054); [P19-1266](https://aclanthology.org/P19-1266/); [checklist C3](https://aclrollingreview.org/responsibleNLPresearch/) |
| 5 | **#P013:** add "same threshold protocol for every metric, tuned on dev only", and "any τ must name its tie-handling variant, or be replaced by tie-calibrated pairwise accuracy". | Extension | [Ties Matter](https://aclanthology.org/2023.emnlp-main.798); [SummaC](https://preview.aclanthology.org/setup/2022.tacl-1.10) |
| 6 | **#P027:** add "Limitations may contain **no new results or analyses**", and keep the Limitations rule in the pre-flight check, since the draft still violates it. | Extension | [ARR CFP](https://aclrollingreview.org/cfp) |
| 7 | **#P029:** appendices must be **two-column** and must not hold anything needed to judge the claims; related work cannot move there to save space. Add the LLM-judge block: verbatim prompts, model ID and date, decoding parameters, parse-failure rate, cost, and same-family leakage between generator and judge. | Extension | [ARR author checklist](https://aclrollingreview.org/authorchecklist); [Bavaresco](https://arxiv.org/abs/2406.18403v2) |
| 8 | **#P033:** "open the paper and check" → a **programmatic** rule: no BibTeX from memory; resolve each entry via ACL Anthology, DOI or Semantic Scholar against two sources; verify the attributed claim; otherwise `\cite{TODO_verify_key}`. Fix the `\cite`→`\citep`/`\citet` misuse. No Unicode in `.bib` (use LaTeX accents). Prefer the archival venue with a DOI or Anthology URL. | Upgrade | [Orchestra](https://raw.githubusercontent.com/Orchestra-Research/AI-Research-SKILLs/main/20-ml-paper-writing/ml-paper-writing/SKILL.md); [ACL formatting](https://acl-org.github.io/ACLPUB/formatting.html); [arXiv 2602.05930](https://arxiv.org/abs/2602.05930) |
| 9 | **New #P034, AI-use disclosure:** keep an agent-authored log of the drafted sections. The review-version Acknowledgements holds **only** the AI disclosure (category d/e text, ideas, coding help). Answer checklist E1 to match. LLMs used as research objects (augmentation, judges) are methodology and belong in §3–5 and B1–B3/C4, not in the disclosure. | New rule | [ACL Ethics Policy](https://www.aclweb.org/adminwiki/index.php/ACL_Policy_on_Publication_Ethics); [ARR author checklist](https://aclrollingreview.org/authorchecklist) |
| 10 | **New #P035, Responsible NLP checklist map:** a maintained table from item to section (A1→Limitations; B1–B3→Data + Ethics; B5→Data Statement appendix; B6→stats table #P010; C1–C4→Setup + appendix; D1–D5→only if #P015 adds human labels; E1→#P034). No blanket "Yes". | New rule | [Responsible NLP checklist](https://aclrollingreview.org/responsibleNLPresearch/) |
| 11 | **New #P036, integrity and format gates:** no hidden or white text and no machine-reader instructions; no leftover comments or `\todo` in the submitted PDF; unmodified `acl.sty`; no vertical-space hacks; caption font ≥ body-adjacent size; aclpubcheck passes on a `final` build; ≤8 content pages. | New rule | [ARR CFP](https://aclrollingreview.org/cfp); [aclpubcheck](https://github.com/acl-org/aclpubcheck) |

Four smaller additions round out the guide:

- **Bender Rule in the title and abstract.** "Hebrew" should appear explicitly in both. #P001 already does this.
- **Honesty lint (Lipton & Steinhardt).** Every causal explanation of a baseline's failure needs an ablation or a hedge. This sharpens #P031.
- **Contribution bullets.** They must be refutable and forward-referenced, which tightens #P004.
- **Reviewer-defense notes.** Pre-empt H10 by framing the paper as monolingual work of practical value. Pre-empt H14 by justifying the choice of closed judges and discussing contamination.

The #P015 finding stays the single most consequential issue: the benchmark labels appear synthetic, but the paper calls them human annotations. It now carries compliance weight as well, because a C3/D-block answer that implies human annotation would be a "misleading checklist" desk-reject risk. The HEDS appendix and an agreement statistic are the remedy if annotation is added.

## Package it as a short CLAUDE.md plus one skill and four checkers

Anthropic's own pattern, as relayed by the community templates, is to keep invariants in a CLAUDE.md of under about 200 lines, put workflows in skills, and run fresh-context checks in sub-agents ([pedrohcgs](https://github.com/pedrohcgs/claude-code-my-workflow); [anthropics/skills](https://github.com/anthropics/skills)). For the LaTeX project, the recommended layout is:

```
~/Downloads/MorphHebEval/
├── CLAUDE.md                         # ≤150 lines: invariants only
└── .claude/
    ├── rules/tex.md                  # path-scoped to latex/**/*.tex
    ├── skills/acl-paper-writing/
    │   ├── SKILL.md                  # workflow: draft → verify → self-review → pre-flight
    │   ├── guidelines.md             # our #P001–#P036 (moved from the augmentor repo)
    │   ├── arr-preflight.md          # ARR author checklist + Responsible NLP map (#P035)
    │   ├── metaeval-stats.md         # Deutsch/Dror/Ties-Matter/LLM-judge reporting
    │   ├── style-lints.md            # paraphrased SPJ/Widom/Steinhardt/Gopen-Swan rules
    │   └── scripts/                  # abstract word count, cite-key check, \todo grep, pdffonts
    └── agents/
        ├── citation-verifier.md      # Orchestra 6-step workflow, Anthology/DOI/S2
        ├── claim-verifier.md         # every number ↔ results file in the augmentor repo
        ├── arr-reviewer.md           # ARR form scores + H1–H17 awareness, stress-test only
        └── reader-tester.md          # doc-coauthoring "fresh reader" test on the PDF
```

The **CLAUDE.md** should state only invariants:

- Never modify `acl.sty` or `acl_natbib.bst`; build with the `[review]` option.
- Never write BibTeX from memory.
- Every number must trace to a named results file in `GenAI-Summary-Augmentor-Correctness-Errors-updatev1`; write `\todo{}` rather than estimate.
- Use `\citep` or `\citet`, never bare `\cite`.
- Translate every Hebrew string, and transliterate and gloss in-text examples.
- No hidden text and no new results in Limitations.
- Propose edits for approval before applying them to prose, since the Overleaf upload is manual.
- Append to an AI-assistance log for the disclosure.
- Run the pre-flight before declaring any section done.

The build line must match the toolchain actually in use. Tectonic is XeTeX-based, so it works with the babel/fontspec Hebrew route. If the project stays on pdfLaTeX with `cjhebrew`, test it with `pdffonts` instead of switching blindly. The Hebrew babel configuration was inferred from the Arabic and Hindi examples in `acl_lualatex.tex` and has not been tested ([acl_lualatex.tex](https://raw.githubusercontent.com/acl-org/acl-style-files/master/acl_lualatex.tex)).

The **skill** should vendor Orchestra's MIT-licensed citation workflow with attribution. It should paraphrase, not copy, the Sakana and classic-guide rules, and link to the ARR pages rather than mirroring them, because those pages carry no license and change every cycle. The two verifier agents carry most of the risk reduction: fabricated references and untraceable numbers are the failures that end papers. The `arr-reviewer` should emit ARR fields (Soundness, Excitement, Overall, Reproducibility, Datasets, Software). Treat its harshness as a list of objections to pre-empt, not as a forecast.

## Conclusion

The research overturns an implicit assumption, namely that a "paper-writing guide for an agent" is mainly about prose. The prose rules are well established and cheap to encode. What actually ends ACL papers in 2026 is compliance and provenance: fabricated references, misleading checklists, de-anonymizing links, results placed in Limitations, and statistics reviewers no longer accept. These are exactly the places where an LLM agent is most likely to fail, and also where it is easiest to gate mechanically. Our 18-paper guide should therefore stay as the *style* authority, while a smaller, testable compliance layer (#P034–#P036, verifier sub-agents, aclpubcheck) becomes the *submission* authority.

Two uncertainties remain. First, the ARR pages are revised every cycle (the Acknowledgements rule already conflicts across two official pages), so the skill should link and re-check them rather than freeze a copy. Second, no source prescribes the significance test for the summary-level, single-system, binary-label setup we have. The Perm-Both/Boot-Both reduction is a well-grounded inference, not a citation, and the paper should say so in its protocol paragraph (#P014).
