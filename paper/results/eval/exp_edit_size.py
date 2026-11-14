"""Edit size vs. detection: do metrics catch small Hebrew edits?

For each MAFAT-Aug negative, the gold summary of the same article is in the benchmark (a faithful pair
with the same premise). Paired detection = P(metric scores the corrupted summary below the gold one),
ties count 1/2. We bin negatives by how much the Hebrew text changed (character similarity to gold,
and number of changed words) and report each metric's paired detection rate per bin, with a 95%
bootstrap CI over pairs. No model is loaded; uses scores.csv only.

    python exp_edit_size.py   -> edit_size.csv (per pair), edit_size_bins.csv (per bin x metric), printed table
"""
import difflib
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eval as R  # noqa: E402

METRICS = ["MorphEvalHeb", "FactCG", "SummaC-ZS", "ENFS", "BERTScore (heBERT)", "BERTScore (AlephBERT)",
           "ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU", "chrF", "Token F1"]


def changed_words(a, b):
    sm = difflib.SequenceMatcher(None, a.split(), b.split(), autojunk=False)
    return sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal")


def main():
    d = R.load_bench()
    s = pd.read_csv(os.path.join(R.HERE, "scores.csv")).set_index("uid")
    m = d[d.part == "MAFAT-Aug"]
    gold = m[(m.y == 1) & m.method.isna()].drop_duplicates("doc_id").set_index("doc_id")
    neg = m[m.y == 0].copy()
    neg["gold_uid"] = neg.doc_id.map(gold.uid)
    neg["char_sim"] = [difflib.SequenceMatcher(None, gold.hypothesis[r.doc_id], r.hypothesis, autojunk=False).ratio()
                       for r in neg.itertuples()]
    neg["changed_words"] = [changed_words(gold.hypothesis[r.doc_id], r.hypothesis) for r in neg.itertuples()]
    for mt in METRICS:
        g, c = s.loc[neg.gold_uid, mt].to_numpy(), s.loc[neg.uid, mt].to_numpy()
        neg[f"det:{mt}"] = np.where(c < g, 1.0, np.where(c == g, 0.5, 0.0))
    neg["bin"] = pd.cut(neg.char_sim, [0, .6, .8, .9, .95, .98, 1.0001],
                        labels=["<0.60", "0.60-0.80", "0.80-0.90", "0.90-0.95", "0.95-0.98", ">=0.98"])
    neg.drop(columns=["premise"]).to_csv(os.path.join(R.HERE, "edit_size.csv"), index=False)

    rng = np.random.default_rng(0)
    rows = []
    for b, g in neg.groupby("bin", observed=True):
        for mt in METRICS:
            x = g[f"det:{mt}"].to_numpy()
            boots = [rng.choice(x, len(x)).mean() for _ in range(1000)]
            rows.append({"bin": b, "n": len(x), "median_changed_words": g.changed_words.median(), "metric": mt,
                         "detection": x.mean(), "lo": np.percentile(boots, 2.5), "hi": np.percentile(boots, 97.5)})
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(R.HERE, "edit_size_bins.csv"), index=False)

    print(f"{len(neg)} MAFAT-Aug negatives; char-sim to gold quantiles (10/50/90%): "
          f"{neg.char_sim.quantile([.1, .5, .9]).round(2).tolist()}")
    print(neg.groupby("bin", observed=True).agg(n=("uid", "size"), med_changed_words=("changed_words", "median"),
                                                 med_families=("error_types", lambda e: e.str.count(",").add(1).median())).to_string())
    print("\nPaired detection rate (corrupted scored below gold; 0.5 = chance):")
    print(t.pivot(index="metric", columns="bin", values="detection").loc[METRICS].round(2).to_string())
    print("\nOverall:", {mt: round(neg[f'det:{mt}'].mean(), 3) for mt in METRICS})

    # Control: faithful rewrites (paraphrasing / back-translation) of the same article. A metric that
    # merely detects "text changed" also scores rewrites below gold (false alarm), and cannot rank a
    # corrupted summary below a faithful rewrite (P(corrupted < rewrite) ~ 0.5).
    rw = m[(m.y == 1) & m.method.notna() & m.doc_id.isin(gold.index)]
    pairs = neg.merge(rw[["doc_id", "uid"]], on="doc_id", suffixes=("", "_rw"))
    sgn = lambda a, b: np.mean(np.sign(a - b) * 0.5 + 0.5)    # P(b < a), ties = 1/2
    ctl = pd.DataFrame([{"metric": mt,
                         "detect_corr_below_gold": sgn(s.loc[neg.gold_uid, mt].to_numpy(), s.loc[neg.uid, mt].to_numpy()),
                         "false_alarm_rewrite_below_gold": sgn(s.loc[rw.doc_id.map(gold.uid), mt].to_numpy(), s.loc[rw.uid, mt].to_numpy()),
                         "P_corr_below_rewrite": sgn(s.loc[pairs.uid_rw, mt].to_numpy(), s.loc[pairs.uid, mt].to_numpy())}
                        for mt in METRICS])
    ctl.to_csv(os.path.join(R.HERE, "edit_size_control.csv"), index=False)
    print(f"\nControl: {len(rw)} faithful rewrites with gold; {len(pairs)} corrupted-vs-rewrite pairs "
          f"from {pairs.doc_id.nunique()} articles")
    print(ctl.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
