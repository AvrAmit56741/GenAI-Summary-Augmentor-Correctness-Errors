"""Label-fidelity audit of injected negatives (quick win from the 2026-11-15 deep-research pass).

A recent paper on the "test oracle problem" in synthetic LLM-judge corpora argues that negatives from
LLM-prompted corruption are "oracle-less" (nothing mechanically confirms the claimed error actually
fired), unlike deterministic string-level perturbation. MorphEvalHeb's injection pipeline
(correctness_errors_augmentation_v2.py) is NOT a deterministic string operator: an LLM is given a
family label (e.g. "sentence_negation", "number_swap") and free-text-generates a corrupted claim, so
the "free mechanical oracle" framing does not transfer cleanly -- this script checks, rather than
assumes, how often the self-reported family label matches an INDEPENDENT rule-based signal of what
actually changed in the text, reusing the same diff machinery as exp_translation.py's edit_kind().

Scope (be honest about this): the independent rule-based check only covers 4 of the ~20 injected
families, the ones with a cheap, unambiguous lexical signature (negation particles, digit strings,
Hebrew definite-article prefix, suffix-only agreement changes). Entity swaps, construct-state
confusion, idiom corruption, etc. have no cheap mechanical check and are out of scope here.

    python exp_label_fidelity.py   -> label_fidelity.csv, printed per-family hit rate
"""
import difflib
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eval as R  # noqa: E402
from exp_translation import mafat, edited_pairs, edit_kind  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "label_fidelity.csv")

# family -> independent edit_kind(s) that should show up if the family really fired.
# Only families with an unambiguous lexical signature in edit_kind() are included.
FAMILY_TO_KIND = {
    "sentence_negation": {"negation"},
    "number_swap": {"number"},
    "entity_date_swap": {"number"},
    "measure_unit_swap": {"number"},
    "noun_gender_swap": {"agreement_suffix", "definiteness"},
    "verb_gender_swap": {"agreement_suffix"},
}


def main():
    m, gold = mafat()
    rows = []
    for r in m[m.y == 0].itertuples():
        fams = {f.strip() for f in str(r.error_types).split(",")}
        target_fams = fams & FAMILY_TO_KIND.keys()
        if not target_fams:
            continue
        pairs = edited_pairs(gold[r.doc_id], r.hypothesis)
        kinds_seen = set()
        for g, c in pairs:
            kinds_seen |= set(edit_kind(g, c).split("+"))
        for fam in target_fams:
            expected = FAMILY_TO_KIND[fam]
            rows.append({
                "uid": r.uid, "family": fam, "n_edited_sentences": len(pairs),
                "kinds_seen": "+".join(sorted(kinds_seen)),
                "hit": bool(expected & kinds_seen),
            })
    t = pd.DataFrame(rows)
    t.to_csv(OUT, index=False)

    print(f"{len(t)} (negative, claimed-family) pairs covered by the independent check, "
          f"from {t.uid.nunique()} negatives\n")
    print("Independent-check hit rate per family (does an unrelated, rule-based diff checker see the\n"
          "expected signature anywhere in the negative's edited sentences?):")
    g = t.groupby("family").agg(n=("hit", "size"), hit_rate=("hit", "mean")).sort_values("hit_rate")
    print(g.round(3).to_string())
    print(f"\noverall: {t.hit.mean():.1%} of claimed-family instances show the matching independent "
          f"signature ({t.hit.sum()}/{len(t)})")
    print("\nNote: a miss does not necessarily mean the error didn't fire -- it may show up in a way "
          "edit_kind() doesn't catch (e.g. a gender change realized through a different word choice, "
          "not a bare suffix swap), or the matching sentence-pair may have been filtered out by the "
          "1:1 sentence-substitution restriction in edited_pairs(). This measures a lower bound on "
          "label fidelity under a strict, context-free lexical check, not an upper bound on error rate.")


if __name__ == "__main__":
    main()
