# Classic human-written guides on writing ML/NLP research papers

Scope: guides that the ML, NLP, vision and systems communities cite widely, read for concrete rules that can become agent instructions. Every source below was fetched and read on 2026-11-13, unless the Gaps section says otherwise. Quotes are short. "Agent-usability" means how directly a rule can become a checkable instruction.

## Q1. What concrete rules does each guide give?

### Takeaway
The guides largely agree on one core set of rules:
- **One idea:** the paper sells one sharp idea.
- **Contributions:** state them up front as an explicit, refutable list that forward-references the evidence.
- **Examples first:** give intuition and examples before the general or formal case.
- **No journey or roadmap:** skip the "how we discovered it" narrative and the "rest of this paper is organized as" paragraph.
- **Fair related work:** be generous to prior work.
- **Endings:** finish on a conclusion, not on a list of future work.

Gopen & Swan add sentence-level rules on where information goes. Lipton & Steinhardt add the "scientific honesty" rules: separate speculation from explanation, ablate, avoid mathiness and misleading terms.

### Cited Findings

**Gopen & Swan, "The Science of Scientific Writing" (American Scientist, 1990)**
- URLs: https://www.americanscientist.org/blog/the-long-view/the-science-of-scientific-writing (returned HTTP 503 when fetched). Reprint PDF that was read: https://cseweb.ucsd.edu/~swanson/papers/science-of-writing.pdf.
- Copyright statement in the reprint: "copyright © 1990 by Sigma Xi, The Scientific Research Society. Reprinted with the permission of American Scientist." — [reprint PDF](https://cseweb.ucsd.edu/~swanson/papers/science-of-writing.pdf)
- Three core principles: "grammatical subjects should be followed as soon as possible by their verbs"; "every unit of discourse, no matter the size, should serve a single function or make a single point"; "information intended to be emphasized should appear at points of syntactic closure" (the **stress position**, at the end of the sentence) — [Gopen & Swan](https://cseweb.ucsd.edu/~swanson/papers/science-of-writing.pdf)
- Definition of length: "A sentence is too long when it has more viable candidates for stress positions than there are stress positions available." — [Gopen & Swan](https://cseweb.ucsd.edu/~swanson/papers/science-of-writing.pdf)
- Topic position (start of the sentence). Put there "the backward-linking old information" and "the person, thing or concept whose story it is". Put "the new, emphasis-worthy information" in the stress position. — [Gopen & Swan](https://cseweb.ucsd.edu/~swanson/papers/science-of-writing.pdf)
- The authors call this a principle, not a rule: "Put in the topic position the old information that links backward; put in the stress position the new information you want the reader to emphasize." — [Gopen & Swan](https://cseweb.ucsd.edu/~swanson/papers/science-of-writing.pdf)
- Further rules: "Articulate the action of every clause or sentence in its verb", and make logical connections explicit instead of leaving them "unarticulated". — [Gopen & Swan](https://cseweb.ucsd.edu/~swanson/papers/science-of-writing.pdf)

**Simon Peyton Jones, "How to write a great research paper" (Microsoft Research talk; slides and video)**
- Landing page: https://www.microsoft.com/en-us/research/academic-program/write-great-research-paper/
- Slides that were read: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/07/How-to-write-a-great-research-paper.pdf
- Seven suggestions: "1. Don't wait: write 2. Identify your key idea 3. Tell a story 4. Nail your contributions 5. Related work: later 6. Put your readers first (examples) 7. Listen to your reviewers" — [SPJ slides](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/07/How-to-write-a-great-research-paper.pdf)
- One "ping": the paper should have "just one 'ping': one clear, sharp idea", and "If you have lots of ideas, write lots of papers". — [SPJ slides](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/07/How-to-write-a-great-research-paper.pdf)
- Narrative order: problem → it is interesting → it is unsolved → my idea → my idea works (details, data) → comparison with other approaches. — [SPJ slides](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/07/How-to-write-a-great-research-paper.pdf)
- Structure: abstract of 4 sentences; introduction of 1 page; problem 1 page; idea 2 pages; details 5 pages; related work 1–2 pages, placed **after** the details; conclusions 0.5 page. — [SPJ slides](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/07/How-to-write-a-great-research-paper.pdf)
- The introduction does two things: "Describe the problem / State your contributions ...and that is all". Open with a concrete example, not with "It is very important to eliminate these bugs [1,2]... Yawn!" — [SPJ slides](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/07/How-to-write-a-great-research-paper.pdf)
- Contributions:
  - "Write the list of contributions first"
  - Use a bulleted list
  - "Contributions should be refutable". "We study its properties" is a bad example; "We prove that the type system is sound... (Section 4)" is a good one.
  — [SPJ slides](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/07/How-to-write-a-great-research-paper.pdf)
- Evidence: "Check each claim in the introduction, identify the evidence, and forward-reference it from the claim." — [SPJ slides](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/07/How-to-write-a-great-research-paper.pdf)
- No roadmap: do not write "The rest of this paper is structured as follows". Use forward references from the narrative instead. — [SPJ slides](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/07/How-to-write-a-great-research-paper.pdf)
- Credit: "Be generous to the competition", "Acknowledge weaknesses in your approach", and "Giving credit to others does not diminish the credit you get from your paper". — [SPJ slides](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/07/How-to-write-a-great-research-paper.pdf)
- Examples before generality: introduce the problem and idea "using EXAMPLES and only then present the general case". "Conveying the intuition is primary". "Do not recapitulate your personal journey of discovery." — [SPJ slides](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/07/How-to-write-a-great-research-paper.pdf)
- Reviews: read every criticism as a sign that something needs a clearer explanation, and fix the paper "so that X is apparent even to the stupidest reader". — [SPJ slides](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/07/How-to-write-a-great-research-paper.pdf)

**Lipton & Steinhardt, "Troubling Trends in Machine Learning Scholarship" (arXiv:1807.03341, ICML 2018 Debates)**
- URLs: https://arxiv.org/abs/1807.03341 and the full text at https://arxiv.org/pdf/1807.03341. The ACM Queue version (https://queue.acm.org/detail.cfm?id=3328534) returned HTTP 403.
- Four trends: (1) "Failure to distinguish between explanation and speculation"; (2) failure to identify the sources of empirical gains, e.g. architecture credited when tuning drives the gain; (3) mathiness; (4) misuse of language. — [arXiv abs](https://arxiv.org/abs/1807.03341)
- Ideal papers "(i) provide intuition... but clearly distinguish it from stronger conclusions supported by evidence; (ii) describe empirical investigations that consider and rule out alternative hypotheses; (iii) make clear the relationship between theoretical analysis and intuitive or empirical claims; (iv) ... choosing terminology to avoid misleading or unproven connotations". — [arXiv PDF](https://arxiv.org/pdf/1807.03341)
- Positive pattern: a designated "Motivation" section "clearly quarantines" speculation. — [arXiv PDF](https://arxiv.org/pdf/1807.03341)
- Ablations: proposing "many tweaks absent proper ablation studies" obscures where the gains come from. The paper cites Melis et al., where gains attributed to architecture came from hyperparameter tuning. — [arXiv PDF](https://arxiv.org/pdf/1807.03341)
- Advice for authors:
  - Ask "'what worked?' and 'why?', rather than just 'how well?'"
  - Use "error analysis, ablation studies, and robustness checks (to e.g. choice of hyper-parameters, as well as ideally to choice of dataset)"
  - Test each explanation with "Would I rely on this explanation for making predictions or for getting a system to work?"
  - Be clear "about which problems are open and which are solved"
  — [arXiv PDF](https://arxiv.org/pdf/1807.03341)
- Misuse of language comes in three forms:
  - **Suggestive definitions**, i.e. anthropomorphic names such as "thought vectors", "curiosity" or "reading comprehension"
  - **Overloaded terminology**, e.g. "deconvolution", "generative model", "covariate shift"
  - **Suitcase words**
  — [arXiv PDF](https://arxiv.org/pdf/1807.03341)
- Test for reviewers: "Might I have accepted this paper if the authors had done a worse job?" A simple idea plus two negative results should be judged more favorably than three combined ideas without ablations that give the same gain. — [arXiv PDF](https://arxiv.org/pdf/1807.03341)
- License: arXiv's non-exclusive distribution license 1.0, not a Creative Commons license. — [arXiv abs](https://arxiv.org/abs/1807.03341)

**Jacob Steinhardt, "Advice for Authors" (blog post, 28 Feb 2017)**
- URL: https://jsteinhardt.stat.berkeley.edu/blog/advice-for-authors
- General writing: be precise ("say what you mean"), e.g. name the metric instead of writing "performance"; be concise; keep sentences simple; use consistent terms for the same concept. — [Steinhardt](https://jsteinhardt.stat.berkeley.edu/blog/advice-for-authors)
- Abstract: "The first sentence / phrase should be something that all readers will agree with". The second sentence then makes the surprising claim. Leading with the punchy claim "will tend to go over reader's heads". — [Steinhardt](https://jsteinhardt.stat.berkeley.edu/blog/advice-for-authors)
- Introduction: avoid vague hedges such as "much recent interest". Explain context before introducing a concept. Write "A, therefore B". State the payoff early. — [Steinhardt](https://jsteinhardt.stat.berkeley.edu/blog/advice-for-authors)
- Conclusion: hold the points that need the whole paper as context for the conclusion, put open questions there, and do not rehash the introduction. — [Steinhardt](https://jsteinhardt.stat.berkeley.edu/blog/advice-for-authors)
- Math: remind readers of notation they have not seen for a while. Use the hierarchy Theorem / Proposition / Lemma / Corollary. "There shouldn't be more than 3 theorems in your paper." Display only the equations worth emphasizing. — [Steinhardt](https://jsteinhardt.stat.berkeley.edu/blog/advice-for-authors)
- LaTeX: put \label after \caption; write "Smith (2001) shows…" with the citation as subject; do not use the fullpage package. — [Steinhardt](https://jsteinhardt.stat.berkeley.edu/blog/advice-for-authors)

**Jennifer Widom, "Tips for Writing Technical Papers" (2006, revised 2009 and 2012)**
- URL: https://cs.stanford.edu/people/widom/paper-writing.html
- Abstract: state the problem, approach, solution and main contributions, with little or no background. Do not repeat the abstract verbatim later in the paper. — [Widom](https://cs.stanford.edu/people/widom/paper-writing.html)
- Introduction: five paragraphs answering:
  1. What is the problem?
  2. Why is it interesting and important?
  3. Why is it hard?
  4. Why has it not been solved before?
  5. What are the key components of the approach and its results, including specific limitations?

  End the introduction with a bulleted "Summary of Contributions" that names the section for each contribution. — [Widom](https://cs.stanford.edu/people/widom/paper-writing.html)
- Related work: place it at the start if it is short or needs a defensive stance; place it at the end if comparisons need the technical content. — [Widom](https://cs.stanford.edu/people/widom/paper-writing.html)
- Body:
  - Introduce a clear new technical contribution "by the end of page 3"
  - Tell a linear story of the results, not how you found them
  - Describe top-down, use a running example, and keep preliminaries concise
  — [Widom](https://cs.stanford.edu/people/widom/paper-writing.html)
- Experiments: avoid "hokey" experiments or tests designed only to show your work at its best. Report sensitivity to key parameters, scalability, and both absolute and relative performance against naive and previous approaches. — [Widom](https://cs.stanford.edu/people/widom/paper-writing.html)
- Conclusions: a short paragraph that does not repeat the abstract or introduction and makes earlier claims concrete with numbers. — [Widom](https://cs.stanford.edu/people/widom/paper-writing.html)
- Style:
  - Define each term or notation before its first use
  - Avoid "etc." unless the remaining items are obvious
  - Never write "for various reasons"; state the reasons
  - Avoid nonreferential "this", "that", "these" and "it"
  - Use italics for definitions, not emphasis
  - Use "that" for defining clauses and "which" for nondefining ones
  — [Widom](https://cs.stanford.edu/people/widom/paper-writing.html)
- Mechanics:
  - Figure fonts about the same size as body text
  - Floats at the top of a page, on the same page as their first reference or the next page
  - Bibliography complete and consistent; do not copy BibTeX blindly
  - Appendix holds only detailed proofs and algorithms, nothing needed to understand the contributions
  — [Widom](https://cs.stanford.edu/people/widom/paper-writing.html)

**Jason Eisner, "Write the paper first"**
- URL: https://www.cs.jhu.edu/~jason/advice/write-the-paper-first.html
- Write the paper before the experiments are finished. "It is better to have a great story with fewer experiments." Show the first 2–3 pages to colleagues early, and let the claims in the introduction determine which experiments are needed. — [Eisner](https://www.cs.jhu.edu/~jason/advice/write-the-paper-first.html)
- Create stub section headers, plus empty tables and graphs with captions. Generate tables and graphs automatically, e.g. with a Makefile, so results stay current. Record ideas in the draft with todo macros. — [Eisner](https://www.cs.jhu.edu/~jason/advice/write-the-paper-first.html)
- "Your code usually dies; your writeup needs to live." — [Eisner](https://www.cs.jhu.edu/~jason/advice/write-the-paper-first.html)

**Andrej Karpathy, "A Survival Guide to a PhD" (2016; the "Writing papers" section)**
- URL: http://karpathy.github.io/2016/09/07/phd/
- Default outline: Intro, Related Work, Model, Experiments, Conclusions. Each paragraph makes one concrete point, "stated on the first sentence". — [Karpathy](http://karpathy.github.io/2016/09/07/phd/)
- Introduction flow: why the problem matters → core challenges → prior work and its shortcomings → what you do → its properties and evidence. — [Karpathy](http://karpathy.github.io/2016/09/07/phd/)
- A paper "sells a single thing that was not obvious or present before". Avoid secondary contributions, and list only the challenges you address. — [Karpathy](http://karpathy.github.io/2016/09/07/phd/)
- Justify each step and name the alternatives considered, not a laundry list of what you did. Include a "pull figure (on page 1 or 2) and system figure (on page 3)". Related work runs about one page with a good density of citations. — [Karpathy](http://karpathy.github.io/2016/09/07/phd/)
- Word choice: prefer "develop" or "propose" over "study" or "investigate". Avoid "combine", "modify" and "expand", which signal incremental work. — [Karpathy](http://karpathy.github.io/2016/09/07/phd/)
- Set an internal deadline two weeks early, with a 5-page draft reviewed internally. — [Karpathy](http://karpathy.github.io/2016/09/07/phd/)

**Bill Freeman, "How to write a good CVPR submission" (MIT CSAIL slides, 6 Nov 2014)**
- The original was not found on an MIT page. Copy that was read: https://www.cs.torontomu.ca/~wangcs/resources\How-to-write-a-good-CVPR-submission.pdf. Another copy: https://sambuz.com/doc/how-to-write-a-good-cvpr-submission-ppt-presentation-794287.
- Quoting Kajiya: "The most dangerous mistake you can make when writing your paper is assuming that the reviewer will understand the point of your paper." State the problem, its context, and "what is really new in your paper (and what isn't)" up front: "write a dynamite introduction". — [Freeman slides](https://www.cs.torontomu.ca/~wangcs/resources\How-to-write-a-good-CVPR-submission.pdf)
- Ted Adelson's formula:
  1. The problem, and why the reader should care
  2. Other solutions and why they are unsatisfactory
  3. Your solution, compared, and why it is better
  4. Related work on different problems at the end
  — [Freeman slides](https://www.cs.torontomu.ca/~wangcs/resources\How-to-write-a-good-CVPR-submission.pdf)
- Explain the main idea with a simple toy example, which Freeman calls an "underutilized technique". — [Freeman slides](https://www.cs.torontomu.ca/~wangcs/resources\How-to-write-a-good-CVPR-submission.pdf)
- Figures: "the figure captions should be self-contained and the caption should tell the reader what to notice about the figure". "It should be easy to read the paper in a big hurry and still learn the main points." — [Freeman slides](https://www.cs.torontomu.ca/~wangcs/resources\How-to-write-a-good-CVPR-submission.pdf)
- Ending: do not end on future work ("I can't stand 'future work' sections... hard to think of a weaker way to end a paper"; there is no "partial credit" for things you did not do). End with conclusions or "what this opens up". — [Freeman slides](https://www.cs.torontomu.ca/~wangcs/resources\How-to-write-a-good-CVPR-submission.pdf)
- Rewriting exercise: concrete before/after edits that cut wordy, hedged sentences, e.g. "We assume local influence—that nodes only depend on other nodes within a patch." — [Freeman slides](https://www.cs.torontomu.ca/~wangcs/resources\How-to-write-a-good-CVPR-submission.pdf)
- Strategy: avoid the "puppy with 6 toes", a delightful paper with "some easy-to-point-to flaw" that makes it easy to reject. — [Freeman slides](https://www.cs.torontomu.ca/~wangcs/resources\How-to-write-a-good-CVPR-submission.pdf)

**Graham Neubig, "How to Read/Write an International Conference Paper" (NLP2015 tutorial, NAIST)**
- Only a third-party mirror was found: https://api.sambuz.com/doc/how-to-read-write-an-international-conference-paper-ppt-presentation-705699
- Abstract: about ¼ page in four sentences (problem, method overview, merits or details, experimental results). — [Neubig mirror](https://api.sambuz.com/doc/how-to-read-write-an-international-conference-paper-ppt-presentation-705699)
- Introduction: about 1 page. State the problem, why it matters and why it is unsolved; list the contributions as bullets; state differences from prior work. Avoid "In recent years…" and "The structure of this paper is…". — [Neubig mirror](https://api.sambuz.com/doc/how-to-read-write-an-international-conference-paper-ppt-presentation-705699)
- Method: intuition first, then formulas. "Make sure every claim promised in the introduction is fulfilled." — [Neubig mirror](https://api.sambuz.com/doc/how-to-read-write-an-international-conference-paper-ppt-presentation-705699)
- Experiments: standard data and metrics, recent baselines, ablations removing one component at a time, and evidence that the examples shown are not flukes. — [Neubig mirror](https://api.sambuz.com/doc/how-to-read-write-an-international-conference-paper-ppt-presentation-705699)
- Related work: cover highly related work, since omitting it signals ignorance or hiding. "Don't attack other authors"; describe their work neutrally. — [Neubig mirror](https://api.sambuz.com/doc/how-to-read-write-an-international-conference-paper-ppt-presentation-705699)
- Conclusion: about ½ page with a 3-sentence summary, future work, and the incomplete parts acknowledged. — [Neubig mirror](https://api.sambuz.com/doc/how-to-read-write-an-international-conference-paper-ppt-presentation-705699)
- Proofreading: get readers who do not know the work ("Each person can only read for the first time once!"). — [Neubig mirror](https://api.sambuz.com/doc/how-to-read-write-an-international-conference-paper-ppt-presentation-705699)
- English: prefer the active voice; use "However," not "But,"; avoid "a lot" and "really"; keep BibTeX consistent and protect capitalization with braces. — [Neubig mirror](https://api.sambuz.com/doc/how-to-read-write-an-international-conference-paper-ppt-presentation-705699)

**Kayvon Fatahalian, "What Makes a (Graphics) Systems Paper Beautiful"**
- URL: https://graphics.stanford.edu/~kayvonf/notes/systemspaper/. Reviewer addendum: https://graphics.stanford.edu/~kayvonf/notes/systemspaper/reviewer_tips.html.
- State goals, non-goals and constraints before describing the system: "We are looking to achieve the goals A, B, C, under the constraints X, Y, Z." — [Fatahalian](https://graphics.stanford.edu/~kayvonf/notes/systemspaper/)
- Name the key design decisions and the alternatives that were rejected. Separate central decisions from incidental ones. — [Fatahalian](https://graphics.stanford.edu/~kayvonf/notes/systemspaper/)
- Evaluation must show that the key decisions caused the goals to be met; ablate the important optimizations. "Failing to perform this evaluation leaves open the possibility that the success of the system is due to other factors." — [Fatahalian](https://graphics.stanford.edu/~kayvonf/notes/systemspaper/)
- The contribution is the organizing insight, not the novelty of individual components. — [Fatahalian](https://graphics.stanford.edu/~kayvonf/notes/systemspaper/)

**Emily M. Bender, "The #BenderRule: On Naming the Languages We Study and Why It Matters" (The Gradient, 14 Sep 2019)**
- URL: https://thegradient.pub/the-benderrule-on-naming-the-languages-we-study-and-why-it-matters/
- The rule: "Always name the language(s) you're working on." It was first tweeted on 3 Jun 2019. — [The Gradient](https://thegradient.pub/the-benderrule-on-naming-the-languages-we-study-and-why-it-matters/)
- ACL's checklist makes it a requirement: "report the language of any language data, even if it is commonly-used benchmarks". — [ARR Responsible NLP checklist](https://aclrollingreview.org/responsibleNLPresearch/)

**ACL Rolling Review Responsible NLP Research checklist** (an official checklist, not a personal guide, but it is the operative standard for ACL venues)
- URL: https://aclrollingreview.org/responsibleNLPresearch/
- Limitations: "Point out any strong assumptions and how robust your results are to violations of these assumptions". Reflect on the scope of claims. "Reviewers will be specifically instructed to not penalize honesty concerning limitations." — [ARR checklist](https://aclrollingreview.org/responsibleNLPresearch/)
- Results:
  - The source of any error bars must be clear: "if it is from a single run, the max across N random seeds, the average, etc."
  - Report validation results as well as test results
  - Report the best hyperparameter values and the search, including searches that were stopped early
  - Give train/validation/test counts, even for standard benchmarks
  — [ARR checklist](https://aclrollingreview.org/responsibleNLPresearch/)
- Licenses: name the license of each asset; for scraped data, state the source's copyright and terms of service. — [ARR checklist](https://aclrollingreview.org/responsibleNLPresearch/)

### Inferences
- SPJ, Widom, Karpathy, Neubig, Freeman/Kajiya and Fatahalian all converge on the same five-part introduction (problem → why it matters → why it is hard or unsolved → our idea → contributions with evidence). This consensus is strong enough to adopt as a hard template.
- Placement of related work is the one real disagreement:
  - **SPJ:** always late.
  - **Widom:** start or end, depending on the case.
  - **Karpathy and Freeman's example:** right after the introduction.

  An agent rule should be conditional: put it early only if it is short and the reader needs it to place the contribution.
- Two forms of the "every claim maps to evidence" rule are mechanically checkable by an agent. SPJ: forward-reference the section that supports each contribution bullet. Neubig: "every claim promised in the introduction is fulfilled".
- Lipton & Steinhardt and the ARR checklist are the best sources for rules on overclaiming and honesty. Steinhardt's blog and Widom are best for sentence and LaTeX hygiene. Gopen & Swan is best for sentence-level revision passes.

### Gaps
- **Jason Eisner, "How to write a paper":** only "Write the paper first" was verified. The fetched page links to no other Eisner writing page, and I did not confirm a separate "how to write a paper" page.
- **Graham Neubig:** I found no CS11-711 lecture specifically on paper writing, and no official phontron.com or neulab copy of the writing slides. The rules above come from a third-party mirror of his 2015 NAIST tutorial and may be garbled in places.
- **Sebastian Ruder, Yoav Goldberg, Noah Smith:** I found no specific, verifiable writing guides in the time budget. The Noah Smith / Bender advice is represented only through the Bender rule.
- **ACL Year-Round Mentorship, Black in AI:** I found no writing guides for these programs (not searched in depth).
- **Fatahalian:** the request mentioned a "what makes a good figure" guide; I found none by him. Only the systems-paper essay was verified.

## Q2. Specific guidance on tables and figures, limitations, overclaiming, mathiness, and speculation vs. explanation

### Takeaway
- **Figures:** the classic guides say little, but it is precise: captions must be self-contained and say what to notice (Freeman); figure fonts match body text and floats sit near their first reference (Widom); include a pull figure on page 1–2 and a system figure on page 3 (Karpathy); add \label after \caption (Steinhardt).
- **Not found:** none of the verified sources give explicit rules on bolding the best result or on significant digits.
- **Limitations and overclaiming:** these rules come mainly from Lipton & Steinhardt and the ARR checklist.

### Cited Findings
- Captions are self-contained and "tell the reader what to notice". The paper must be readable "in a big hurry" from the figures and captions. — [Freeman slides](https://www.cs.torontomu.ca/~wangcs/resources\How-to-write-a-good-CVPR-submission.pdf)
- Figure fonts are about the same size as the text. Tables and figures go at the top of a page or column, on the page of their first reference or the next one. — [Widom](https://cs.stanford.edu/people/widom/paper-writing.html)
- Include a well-designed pull figure on page 1–2 and a system figure on page 3, not ones "made in MS Paint". — [Karpathy](http://karpathy.github.io/2016/09/07/phd/)
- Stub empty tables and graphs with captions early, and generate them from code so the numbers stay in sync. — [Eisner](https://www.cs.jhu.edu/~jason/advice/write-the-paper-first.html)
- Results reporting: say whether a number is a single run, the max over N seeds, or a mean; report validation results; report the hyperparameters found and the search. — [ARR checklist](https://aclrollingreview.org/responsibleNLPresearch/)
- Limitations:
  - State strong assumptions and robustness to their violation; reviewers are told not to penalize honesty — [ARR checklist](https://aclrollingreview.org/responsibleNLPresearch/)
  - Name "specific limitations" in the introduction — [Widom](https://cs.stanford.edu/people/widom/paper-writing.html)
  - "Acknowledge weaknesses in your approach" — [SPJ slides](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/07/How-to-write-a-great-research-paper.pdf)
  - Acknowledge the incomplete parts in the conclusion — [Neubig mirror](https://api.sambuz.com/doc/how-to-read-write-an-international-conference-paper-ppt-presentation-705699)
- Overclaiming: say "what is really new in your paper (and what isn't)". — [Freeman/Kajiya](https://www.cs.torontomu.ca/~wangcs/resources\How-to-write-a-good-CVPR-submission.pdf)
- Contributions must be "refutable". — [SPJ](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/07/How-to-write-a-great-research-paper.pdf)
- No "hokey" experiments or experiments designed only to show your method at its best. — [Widom](https://cs.stanford.edu/people/widom/paper-writing.html)
- Speculation vs. explanation: separate them, with intuition clearly marked and ideally confined to a "Motivation" section. Be clear about which problems are open and which are solved. — [Lipton & Steinhardt](https://arxiv.org/pdf/1807.03341)
- Mathiness: mathematics should clarify, not impress. Test each theorem or explanation with "Would I rely on this explanation for making predictions or for getting a system to work?" — [Lipton & Steinhardt](https://arxiv.org/pdf/1807.03341)
- Theorem count: "no more than 3 theorems"; display only the equations worth emphasizing. — [Steinhardt](https://jsteinhardt.stat.berkeley.edu/blog/advice-for-authors)
- Terminology: avoid anthropomorphic or suggestive names and terms that overload established meanings. — [Lipton & Steinhardt](https://arxiv.org/pdf/1807.03341)
- Use consistent terms for one concept. — [Steinhardt](https://jsteinhardt.stat.berkeley.edu/blog/advice-for-authors)

### Inferences
- Agent rules on tables (bold the best result, use consistent significant digits, include ±std, mark which numbers are cited and which re-run) need sources outside this set, e.g. venue style guides or ML reproducibility checklists. The classic guides support only captions, placement, font size and auto-generation.
- "Speculation must be labeled" and "every claim needs evidence or a hedge" are the two honesty rules most worth encoding. Both are sourced: Lipton & Steinhardt plus SPJ.

### Gaps
- No verified classic source covers bolding the best results, significant digits, or statistical significance testing in tables.
- Fatahalian's figure-specific advice was not found.

## Q3. Licensing and reusability: adopt, adapt, or reference?

### Takeaway
None of the verified guides carries a Creative Commons license. SPJ explicitly allows the slides to be repurposed with acknowledgement. arXiv's license allows only distribution by arXiv. The rest have no license or are copyrighted. Recommendation: **paraphrase the rules into our own wording with attribution and links. Do not copy text in bulk.**

### Cited Findings
- **SPJ:** "you should feel free to repurpose these slides for your own use as long as you acknowledge ownership". — [MSR page](https://www.microsoft.com/en-us/research/academic-program/write-great-research-paper/)
- **Gopen & Swan:** "copyright © 1990 by Sigma Xi"; reprinted by permission. — [reprint](https://cseweb.ucsd.edu/~swanson/papers/science-of-writing.pdf)
- **Lipton & Steinhardt:** arXiv non-exclusive distribution license 1.0. — [arXiv](https://arxiv.org/abs/1807.03341)
- **Steinhardt blog:** "© 2026 Jacob Steinhardt", no license. — [Steinhardt](https://jsteinhardt.stat.berkeley.edu/blog/advice-for-authors)
- **Widom, Eisner and Fatahalian:** no license stated. — [Widom](https://cs.stanford.edu/people/widom/paper-writing.html); [Eisner](https://www.cs.jhu.edu/~jason/advice/write-the-paper-first.html); [Fatahalian](https://graphics.stanford.edu/~kayvonf/notes/systemspaper/)
- **Bender:** no license, but the page gives a citation format for attribution. — [The Gradient](https://thegradient.pub/the-benderrule-on-naming-the-languages-we-study-and-why-it-matters/)
- **ARR checklist:** "© 2026 ACL Rolling Review", no explicit content license. — [ARR](https://aclrollingreview.org/responsibleNLPresearch/)
- **Freeman and Neubig:** available only through third-party mirrors; no license found. — [Freeman copy](https://www.cs.torontomu.ca/~wangcs/resources\How-to-write-a-good-CVPR-submission.pdf); [Neubig mirror](https://api.sambuz.com/doc/how-to-read-write-an-international-conference-paper-ppt-presentation-705699)

### Inferences (per guide: agent-usability → recommendation)

| Guide | Agent-usability | Recommendation |
|---|---|---|
| Peyton Jones | High: 7 rules, checkable (one idea, refutable contribution bullets with § refs, no roadmap, related work late, example first) | **Adopt** (paraphrase with credit; reuse is permitted with acknowledgement) |
| Lipton & Steinhardt | High: honesty lints (label speculation, ablate, avoid anthropomorphic or overloaded terms, "would I rely on it" test) | **Adopt** as an honesty and overclaiming checklist (paraphrase, cite the arXiv paper) |
| Widom | High: introduction in 5 questions, contribution bullets with sections, style lints ("for various reasons", nonreferential "this", "etc.") | **Adopt** (paraphrase) |
| Gopen & Swan | Medium: a sentence-level revision pass (subject next to verb, old→new, stress position, action in the verb) | **Adapt** into a "revision pass" instruction; reference only (copyrighted) |
| Steinhardt blog | High for abstract shape, consistent terms, theorem hygiene, LaTeX lints | **Adopt** selected rules (paraphrase) |
| Freeman (Kajiya/Adelson) | Medium-high: self-contained captions saying what to notice, toy example, no closing "future work" | **Adapt** (mirror only; cite) |
| Neubig 2015 | High: per-section page budgets, banned openers, ablations, neutral related work | **Adapt** (mirror only; verify against an official copy if one becomes available) |
| Karpathy | Medium: single contribution, pull figure on p.1–2, word choice | **Adapt** selected rules |
| Eisner | Process-level (write first, stubbed tables, auto-generated results) | **Reference** / workflow guidance, not prose rules |
| Fatahalian | Medium: goals, non-goals and constraints; design alternatives; evaluation that attributes the effect to each decision | **Adapt** for systems or pipeline papers |
| Bender rule + ARR checklist | High: name the language; report seeds and runs, validation and test results, hyperparameters, licenses, and a limitations section | **Adopt** as mandatory NLP reporting items |

### Gaps
- I did not inspect Microsoft's site-wide "Terms of use" for the SPJ PDF; the permission rests only on the author's statement on the page.
- The license status of the Freeman and Neubig originals is unknown because only mirrors were found.
