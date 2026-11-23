# Pilot: is the verb_gender_swap difference vector causally steerable?

Status: pilot complete (2026-11-22), N=6 held-out sentences. Direct causal follow-on to
`../representation_geometry_pilot/`, reusing its layer-24 hidden states and difference vectors at
zero extraction cost -- this only needed a generation script and a forward hook.

## What this tests

The representation-geometry pilot showed the `verb_gender_swap` mean difference vector
(`d = mean(h(erroneous) - h(faithful))` over the 10 verb_gender_swap examples, at layer 24 of 28)
is a *descriptive* regularity: same-family errors cluster in that direction. That leaves open
whether the direction is *causal* -- whether adding it back into the model during generation
actually makes the model commit that error, rather than just being a passive correlate of it. This
is the cheap activation-steering follow-on flagged (but not run) in that pilot's "next steps."

## Method (`steer.py`)

Register a forward hook on `model.model.layers[23]` of DictaLM-3.0-1.7B-Instruct (0-indexed decoder
layer 23, whose output is `hidden_states[24]` in the extraction pipeline -- same layer the geometry
pilot found the peak within/between-family gap). The hook adds `scale * d` to every position's
hidden state, active for the whole `generate()` call (every decoding step, not just the prompt).

Task: ask the model (via its chat template) to repeat a sentence verbatim --
*"חזור בדיוק על המשפט הבא, מילה במילה, ללא שום שינוי: \<sentence\>"*. Six held-out sentences
(three male-subject, three female-subject), each with unambiguous single-subject gender, on topics
deliberately different from the geometry pilot's examples (engineering, cooking, sports -- away
from its dog-shelter-listing cluster) so a gender flip is unambiguous and not a repeat of training
data. Five conditions per sentence: baseline (no steering), `d` at scale 1/2/3, and a **random
vector norm-matched to `d`** at scale 2, as a control for "any big perturbation breaks grammar."

## Results

**The vector is causally steerable, and selectively so.** At scale 2, every one of the three
male-subject sentences flips its verb to the feminine form while leaving everything else
unchanged -- e.g. `בדק` (he checked) -> `בדקה` (she checked), `הכין` -> `הכינה`, `קלט` -> `קלטה` --
exactly the target error type, with nothing else in the sentence disturbed. **The norm-matched
random control, at the same scale, reproduces every sentence verbatim with no effect at all** --
ruling out "a sufficiently large perturbation at this layer breaks grammar generically" as the
explanation. At scale 3, the real vector starts additionally corrupting fluency (e.g. the subject
noun itself gets replaced, `המהנדס` -> `ההיא`), so scale ~2 is the clean causal regime and scale 3
overshoots.

**Important asymmetry, not a bug**: the vector reliably flips male-correct sentences to
female-incorrect, but mostly leaves already-female-correct sentences unchanged (one exception
produced a different, unrelated morphological distortion, not a gender flip). This traces directly
to the training data: 9 of the 10 `verb_gender_swap` examples in the geometry pilot's dataset are
themselves male-correct-to-female-error flips (only `nova_shoshana` goes the other way), so the
*mean* difference vector inherits that directional imbalance -- it encodes "push toward feminine
verb marking," not a symmetric "flip whichever gender is present" operation. This is itself an
interesting, honest finding about what the averaged vector actually represents, not a failure of
the method.

Full raw transcripts (all 6 sentences x 5 conditions): `steering_results.json`.

## Honest caveats

- N=6 held-out test sentences, one steering direction (verb_gender_swap only), one model
  (DictaLM-3.0-1.7B-Instruct), one task framing ("repeat verbatim"). A pilot, not a powered study.
- The directional asymmetry means this specific vector is better described as "push toward
  feminine marking" than "induce verb_gender_swap errors in general" -- a vector built from a
  gender-balanced training set (or two separate vectors, M-to-F and F-to-M) would be needed to
  test the fully symmetric claim.
- Only tested on a "repeat the sentence" task, not on a realistic generation/summarization task
  where the effect might be weaker or interact differently with content planning.
- No automatic/quantitative scoring (e.g. an error-detector run over many more sentences) -- this
  pilot is qualitative, by-eye verification of a clean, legible effect on 6 examples.

## If this becomes a real chapter

Cheap next steps: (1) build a separate noun_gender_swap vector and test it the same way; (2) build
a gender-balanced version of the verb_gender_swap vector (equal M->F and F->M training examples)
to test whether a truly bidirectional "flip whatever gender is present" vector is achievable; (3)
scale up to more held-out sentences and add an automatic detector (e.g. a simple morphological
checker or MorphEvalHeb itself) to get a quantitative flip-rate instead of by-eye inspection; (4)
test on a realistic summarization-generation task instead of verbatim repetition.

## Follow-up: quantitative test with balanced vectors (2026-11-23, `steer_eval.py`)

Vectors built from real benchmark edits instead of the 10 pilot examples: VERB gender edits from
`paper/results/eval/morph_attack_candidates.csv` (Dicta-parser gender of the original verb), split by article
(train 84 masc->fem, 44 fem->masc; test articles held out). cosine(m2f, f2m) = -0.89: one shared gender axis.
Vector norm ~115 (pilot vector 453), so scales here are ~4x the pilot's. 20 held-out sentences (10 with a
masculine target verb, 10 feminine), "repeat verbatim", greedy decoding, scored automatically
(`steer_eval.jsonl`, `steer_eval_summary.csv`).

| condition | target verb flipped (fem sentences / masc sentences) | output verbatim |
|---|---|---|
| baseline | 0/10, 0/10 | 90% |
| f2m x8 | **6/10** (fem->masc), 0/10 | 30-60% |
| m2f x8 | 0/10, **4/10** (masc->fem) | 10-50% |
| random x8 (norm-matched) | 0/10, 0/10 | 80-90% |

- Both directions work and are selective: f2m never makes a word feminine, m2f never masculine; the random
  vector never flips gender (f2m vs random p=0.011, m2f vs random p=0.087, Fisher, n=10 each).
  Balanced training data removes the pilot's one-way asymmetry.
- But flips are mostly not single-word: only 3/10 flips at x8 changed only the target word (median 3.5 words
  changed). The vector tends to re-gender the whole referent (pronoun, verb, adjectives together), e.g.
  "ד"ר גב מציעה" -> "ד"ר גב מציע", "הוא גם השתתף" -> "היא גם השתתפה". So it behaves more like
  "the subject is female/male" (a referent-gender / entity error) than a pure agreement error.
- x12 starts breaking fluency for all vectors, random included. Small n; a pilot, not a powered result.
