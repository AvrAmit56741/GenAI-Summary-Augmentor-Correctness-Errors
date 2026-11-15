"""Does translation mask obviously-impossible injected years? (follow-up to docs/experiments.md #17)

The augmentation pipeline's number rule is deterministic doubling (acl_latex.tex Sec 4.2, "(iii) Numbers
are changed by a deterministic rule (doubling)"). Applied to a Hebrew year (1900-2024), doubling always
produces an obviously-impossible 4-digit value (3800-4048) -- there is no separate "impossible value"
injection mechanism; it's a predictable side effect of doubling applied to years specifically.

Question: when such a doubled, impossible-looking year is translated to English, does the translator
(a) pass it through literally (the injected error survives, visible in English), or (b) silently turn
it into a plausible-looking year, masking the error? And if (b), is it recovering the TRUE original
value (real world-knowledge repair) or just pattern-matching the leading digit?

Uses only already-saved data (translation_erasure_doclevel.csv, from exp_translation_api.py) -- no new
API calls.

    python exp_translation_impossible_year.py   -> impossible_year.csv, printed summary
"""
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eval as R  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "impossible_year.csv")
DIG4 = re.compile(r"\b\d{4}\b")


def true_original_year(gold_he, corr_he):
    """If gold has exactly one 4-digit number and corrupted has exactly one (different) 4-digit
    number, and corrupted ~= 2x gold (the doubling rule), return the gold (true original) year."""
    g4, c4 = DIG4.findall(gold_he), DIG4.findall(corr_he)
    if len(g4) == 1 and len(c4) == 1 and g4[0] != c4[0]:
        g, c = int(g4[0]), int(c4[0])
        if abs(c - 2 * g) <= 1:   # allow off-by-one from rounding elsewhere in the sentence
            return g
    return None


def leading_digit_swap(corr_year, en_year):
    """True if en_year is exactly corr_year with only the leading digit changed (3->1 or 4->2),
    i.e. a superficial plausibility fix, not a recovery of a different, unrelated value."""
    c, e = str(corr_year), str(en_year)
    return len(c) == len(e) == 4 and c[1:] == e[1:] and c[0] != e[0]


def main():
    a = pd.read_csv(os.path.join(HERE, "translation_erasure_doclevel.csv"))
    a = a[a.aligned & a.num_erased.notna()].drop_duplicates(["model", "gold_he", "corr_he"])

    rows = []
    for r in a.itertuples():
        c4 = DIG4.findall(r.corr_he)
        impossible = [y for y in c4 if not (1500 <= int(y) <= 2035)]
        if not impossible:
            continue
        true_year = true_original_year(r.gold_he, r.corr_he)
        e4 = DIG4.findall(r.corr_en)
        en_year = next((int(y) for y in e4 if 1500 <= int(y) <= 2035), None)
        rows.append({
            "model": r.model, "corr_impossible_year": impossible[0],
            "is_doubling_injection": true_year is not None, "true_original_year": true_year,
            "erased_or_masked": bool(r.num_erased), "en_year_shown": en_year,
            "recovered_true_value": en_year == true_year if (en_year and true_year) else None,
            "leading_digit_swap": leading_digit_swap(int(impossible[0]), en_year) if en_year else None,
            "gold_he": r.gold_he, "corr_he": r.corr_he, "corr_en": r.corr_en,
        })
    t = pd.DataFrame(rows)
    t.to_csv(OUT, index=False)

    print(f"{len(t)} edited sentences with an implausible 4-digit number in the corrupted text "
          f"(from {a.uid.nunique() if hasattr(a, 'uid') else '?'} pairs)")
    print(f"  of these, {t.is_doubling_injection.mean():.1%} are exactly explained by the paper's "
          f"number-doubling rule applied to a year (true_original_year found)")
    for mdl, g in t.groupby("model"):
        tag = mdl.split(".")[1][:9]
        masked = g[g.erased_or_masked]
        print(f"\n  {tag}: masked/changed in English {g.erased_or_masked.mean():.1%} of {len(g)}")
        if len(masked):
            rec = masked[masked.recovered_true_value.notna()]
            print(f"    of masked cases with a determinable true value (n={len(rec)}): "
                  f"recovered the TRUE original year {rec.recovered_true_value.mean():.1%} of the time")
            swap = masked[masked.leading_digit_swap.notna()]
            print(f"    of masked cases (n={len(swap)}): a bare leading-digit swap "
                  f"(e.g. 4034->2034, not the true value) explains {swap.leading_digit_swap.mean():.1%}")

    print("\nexample rows (masked, doubling-injection confirmed):")
    ex = t[t.erased_or_masked & t.is_doubling_injection].head(5)
    for _, r in ex.iterrows():
        print(f"  true={r.true_original_year} corrupted={r.corr_impossible_year} "
              f"EN_shows={r.en_year_shown} recovered_true={r.recovered_true_value} "
              f"digit_swap={r.leading_digit_swap}")
        print(f"    G: {r.gold_he[:70]}\n    C: {r.corr_he[:70]}\n    EN: {r.corr_en[:90]}")


if __name__ == "__main__":
    main()
