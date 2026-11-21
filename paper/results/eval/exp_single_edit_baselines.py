"""Do OTHER metrics also miss a single error in a full summary? (follow-up of exp_single_edit_summary.py)

For each of the single-error summaries in single_edit_summary.csv (one injected word placed into the faithful reference
summary), score with MorphEvalHeb, FactCG and SummaC-ZS:
  gold   = faithful reference summary
  single = gold with ONE wrong word
  multi  = an Injected corrupted summary of the same article from the benchmark (many errors)
Threshold per metric = 5th percentile of its gold scores (5% false alarms), so metrics are compared at equal FA.

    python exp_single_edit_baselines.py score     -> cache/single_edit_baselines.jsonl (chunks of 20 summaries)
    python exp_single_edit_baselines.py analyze   -> single_edit_baselines_summary.csv, figures/single_vs_multi.png
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eval as R  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(HERE, "single_edit_summary.csv")
CACHE = os.path.join(HERE, "cache", "single_edit_baselines.jsonl")
METRICS = ["MorphEvalHeb", "FactCG", "SummaC-ZS"]


def items():
    d = pd.read_csv(IN)
    d = d[d.p_summ_gold >= .5].reset_index(drop=True)
    w = pd.read_csv(os.path.join(HERE, "layers_words_sample.csv")).drop_duplicates("unit_id").set_index("unit_id")
    pj = {j["unit_id"]: j for j in map(json.loads, open(os.path.join(HERE, "cache", "patching.jsonl")))}
    b = R.load_bench()
    m = b[b.part == "MAFAT-Aug"]
    g = m[m.y == 1].drop_duplicates("doc_id").set_index("doc_id")
    neg = m[m.y == 0].sample(frac=1, random_state=0).drop_duplicates("doc_id").set_index("doc_id")
    rows = []
    for r in d.itertuples():
        doc = w.doc_id[r.unit_id]
        gold = g.hypothesis[doc]
        single = gold.replace(w.gold_sent[r.unit_id], pj[r.unit_id]["minimal_sent"], 1)
        rows.append({"unit_id": r.unit_id, "kind": r.kind, "doc_id": doc, "premise": g.premise[doc], "gold": gold,
                     "single": single, "multi": neg.hypothesis.get(doc, np.nan),
                     "n_types_multi": len(str(neg.error_types.get(doc, "")).split(","))})
    return pd.DataFrame(rows)


def score(chunk=20):
    from exp_multisource import Scorers
    it = items()
    done = set()
    if os.path.exists(CACHE):
        done = {(j["unit_id"], j["version"], j["metric"]) for j in map(json.loads, open(CACHE))}
    f = None
    t0 = time.time()
    for i in range(0, len(it), chunk):
        buf = []
        for r in it.iloc[i:i + chunk].itertuples():
            for ver in ["gold", "single", "multi"]:
                txt = getattr(r, ver)
                if not isinstance(txt, str):
                    continue
                for met in METRICS:
                    if (r.unit_id, ver, met) in done:
                        continue
                    f = f or Scorers()
                    buf.append({"unit_id": int(r.unit_id), "kind": r.kind, "version": ver, "metric": met,
                                "v": f(met, r.premise, txt)})
        with open(CACHE, "a") as fh:
            for b in buf:
                fh.write(json.dumps(b) + "\n")
        if buf:
            x = pd.DataFrame(buf).pivot_table(index=["unit_id", "metric"], columns="version", values="v").reset_index()
            msg = " | ".join(f"{m}: single<gold {(g.single < g.gold).mean():.2f}, ties {(g.single == g.gold).mean():.2f}"
                             for m, g in x.groupby("metric") if "single" in g)
            print(f"  {min(i + chunk, len(it))}/{len(it)} ({time.time() - t0:.0f}s) {msg}", flush=True)


def analyze():
    from exp_multisource import wilson
    s = pd.DataFrame([json.loads(x) for x in open(CACHE)]).drop_duplicates(["unit_id", "version", "metric"], keep="last")
    p = s.pivot_table(index=["unit_id", "kind", "metric"], columns="version", values="v").reset_index()
    rows = []
    for met, g in p.groupby("metric"):
        thr = np.quantile(g.gold, 0.05)
        for ver in ["single", "multi"]:
            x = g.dropna(subset=[ver])
            k = int((x[ver] < thr).sum())
            lo, hi = wilson(k, len(x))
            rows.append({"metric": met, "version": ver, "n": len(x), "threshold": thr, "flagged": k / len(x),
                         "ci_lo": lo, "ci_hi": hi, "scored_below_gold": (x[ver] < x.gold).mean(),
                         "tie_with_gold": (x[ver] == x.gold).mean()})
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(HERE, "single_edit_baselines_summary.csv"), index=False)
    print(t.round(3).to_string(index=False))
    kk = []
    for met, g in p.groupby("metric"):
        thr = np.quantile(g.gold, 0.05)
        kk.append(g.assign(flag=g.single < thr).groupby("kind").flag.mean().rename(met))
    print("single-error flagged by kind:\n", pd.concat(kk, axis=1).round(2))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.3))
    x = np.arange(len(METRICS))
    for j, (ver, lab) in enumerate([("single", "one wrong word"), ("multi", "Injected (many errors)")]):
        v = t[t.version == ver].set_index("metric").loc[METRICS]
        ax.bar(x + (j - .5) * .38, v.flagged, .38, yerr=[v.flagged - v.ci_lo, v.ci_hi - v.flagged], capsize=3, label=lab)
    ax.set_xticks(x, ["Ours" if m == "MorphEvalHeb" else m for m in METRICS])
    ax.set(ylabel="flagged at 5% false alarms", ylim=(0, 1), title="One error vs. many errors in the same summary")
    ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(HERE, "figures", "single_vs_multi.png"), dpi=160); plt.close()


if __name__ == "__main__":
    {"score": score, "analyze": analyze}[sys.argv[1]]()
