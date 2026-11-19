"""Per-chunk verification of translation erasure (run after each chunk of exp_translation.py erasure_stream).

Exact erasure (identical English) misses errors whose *meaning* is lost while the wording differs.
Second check: an English fact-checker (FactCG, DeBERTa-v3-large, English-native) scores whether the
English corrupted sentence is supported by the English gold sentence. support >= 0.5 -> the error is
semantically erased in English (an English checker would see no inconsistency).

Validation built in: changed numbers survive translation verbatim, so FactCG should give them LOW
support; that rate is printed per chunk as a sanity check of the checker itself.

    python exp_translation_verify.py   -> translation_erasure_verified_<model>.csv, printed chunk report
"""
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eval as R  # noqa: E402

TAG = os.environ.get("MT_TAG", "translategemma-12b-it")
SRC = os.path.join(R.HERE, f"translation_erasure_{TAG}.csv")
OUT = os.path.join(R.HERE, f"translation_erasure_verified_{TAG}.csv")
KINDS = ["negation", "number", "definiteness", "agreement_suffix", "other"]


def main():
    from summary_augmentation.pages.metrics.factcg import FactCGScorer
    import exp_translation as T
    e = pd.read_csv(SRC)
    e["row"] = range(len(e))
    e["edit_kind"] = [T.edit_kind(g, c) for g, c in zip(e.gold_he, e.corr_he)]   # current rules, applied to all rows
    done = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame(columns=["row"])
    new = e[~e.row.isin(done.row)]
    if new.empty:
        print("verify: nothing new"); return
    sc = FactCGScorer(device=R.device())
    sc.load()
    new = new.copy()
    new["en_support"] = sc._support_probabilities(list(new.gold_en), list(new.corr_en))
    # NOT erasure: FactCG gives <0.5 to ~13% of identical pairs and misses surviving tense changes, so this
    # measures "an English checker would miss it", a separate end-to-end number.
    new["en_checker_miss"] = new.en_support >= 0.5
    # checker-free negation test: did the English negation count change along with the Hebrew one?
    neg = lambda t: len(re.findall(r"\b(not|no|never|none|nothing|without|neither|nor)\b|n't", str(t).lower()))
    new["neg_survived_en"] = [(neg(g) != neg(c)) if "negation" in k else None
                              for g, c, k in zip(new.gold_en, new.corr_en, new.edit_kind)]
    new.to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
    a = pd.read_csv(OUT)
    a["edit_kind"] = a.row.map(e.set_index("row").edit_kind)          # relabel old rows with current rules
    a = a.drop_duplicates(["gold_he", "corr_he"])                        # each unique sentence pair counts once
    new = new.drop_duplicates(["gold_he", "corr_he"])
    new = new[~new.set_index(["gold_he", "corr_he"]).index.isin(
        pd.read_csv(OUT).loc[lambda d: ~d.row.isin(new.row)].set_index(["gold_he", "corr_he"]).index)]

    def by_kind(df):
        k = df.assign(k=df.edit_kind.str.split("+")).explode("k")
        return ", ".join(f"{x} {int(k[k.k == x].erased.sum())}/{int((k.k == x).sum())}" for x in KINDS)

    print(f"VERIFY chunk: {len(new)} new edited sentences from {new.uid.nunique()} summaries")
    print(f"  this chunk: exact erased {new.erased.mean():.1%}, English checker misses {new.en_checker_miss.mean():.1%}")
    print(f"  cumulative ({a.uid.nunique()} summaries, {len(a)} UNIQUE sentence pairs): exact erased {a.erased.mean():.1%}, "
          f"English checker misses {a.en_checker_miss.mean():.1%}")
    print(f"  exact erasure by kind (erased/total): {by_kind(a)}")
    ng = a[a.edit_kind.str.contains("negation")]
    ng = ng.assign(neg_survived_en=[neg(g) != neg(c) for g, c in zip(ng.gold_en, ng.corr_en)])
    print(f"  negation edits: negation lost in English {(~ng.neg_survived_en).mean():.0%} of {len(ng)} "
          f"(exact-erased {ng.erased.mean():.0%})")
    num = a[a.edit_kind.str.contains("number") & ~a.erased]
    print(f"  checker sanity: surviving number edits flagged by FactCG (support<0.5): "
          f"{(num.en_support < 0.5).mean():.0%} of {len(num)}")
    same = a[a.erased]
    print(f"  checker sanity: exact-erased pairs get support>=0.5: {(same.en_support >= 0.5).mean():.0%} of {len(same)}")
    print("  erased this chunk (for review):")
    show = new[new.erased | (new.edit_kind.str.contains("negation") & (new.neg_survived_en == False))]
    for _, x in show.iterrows():
        tag = "EXACT" if x.erased else "NEG-LOST"
        print(f"   [{tag}|{x.edit_kind}] G «{x.gold_he[:80]}» | C «{x.corr_he[:80]}»\n      EN-G «{x.gold_en[:100]}»\n      EN-C «{x.corr_en[:100]}»")


if __name__ == "__main__":
    main()
