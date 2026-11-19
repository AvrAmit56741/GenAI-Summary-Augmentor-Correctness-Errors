"""Lead 17, option 3: do metrics behave differently on single-source vs multi-source summary claims?

Unit = one summary SENTENCE (claim). Injected corruptions edit ~8 sentences per summary, so a whole-summary score
cannot be attributed to one sentence; we therefore re-score each (article, sentence) pair with every metric.

Source structure of a claim = best1 = largest share of the claim's (morphology-normalized) content words covered
by ONE article sentence. best1 < 0.5 -> "multi". Validated on the 39 hand-labeled claims (evidence_gold_labels.csv,
same population: Injected gold-summary sentences): accuracy 0.77, AUC 0.80. Grouping is computed on the GOLD
sentence (faithful text) so that the inserted error cannot itself lower the coverage.
Natural (mFACT) is excluded: article and summary were translated separately, so lexical coverage is not valid there.

Analyses on a stratified subset (N per group):
  paired detection: metric scores the gold sentence above its corrupted version
  false alarm:      metric flags the faithful gold sentence (MorphEvalHeb/FactCG p<0.5; SummaC < threshold that
                    catches 80% of the corrupted sentences)

    python exp_multisource.py build [N]     -> multisource_units.csv (all units) + multisource_subset.csv
    python exp_multisource.py score         -> cache/multisource_scores.jsonl (chunks of 20, resumable)
    python exp_multisource.py analyze       -> multisource_summary.csv
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eval as R  # noqa: E402
import exp_translation as T  # noqa: E402
import exp_evidence as E  # noqa: E402

THR = 0.5
OUT = R.HERE
UNITS = os.path.join(OUT, "multisource_units.csv")
SUBSET = os.path.join(OUT, "multisource_subset.csv")
SCORES = os.path.join(R.CACHE, "multisource_scores.jsonl")
METRICS = ["MorphEvalHeb", "FactCG", "SummaC-ZS"]


def best1(claim, art_toks):
    cw = set(E.toks(claim))
    if not cw:
        return np.nan
    return max(len(cw & s) for s in art_toks) / len(cw)


def build(n_per_group=250, seed=7):
    bench = R.load_bench().set_index("uid")
    es = pd.read_csv(os.path.join(OUT, "edit_size.csv"))
    art_toks = {}
    rows = []
    for x in es.itertuples():
        g = bench.at[x.gold_uid, "hypothesis"]
        prem = bench.at[x.uid, "premise"]
        gs, cs = T.split_sents(g), T.split_sents(x.hypothesis)
        if len(gs) != len(cs):          # alignment unsafe (6% of pairs) -> skip
            continue
        if x.doc_id not in art_toks:
            art_toks[x.doc_id] = [set(E.toks(s)) for s in T.split_sents(prem)]
        for j, (a, c) in enumerate(zip(gs, cs)):
            if a.split() == c.split() or len(E.toks(a)) < 5:
                continue
            b = best1(a, art_toks[x.doc_id])
            rows.append({"doc_id": x.doc_id, "source": x.source, "pair_uid": x.uid, "sent_idx": j,
                         "error_types": x.error_types, "gold_sent": a, "corr_sent": c, "best1": b,
                         "group": "multi" if b < THR else "single", "words": len(a.split())})
    U = pd.DataFrame(rows).drop_duplicates(["gold_sent", "corr_sent"]).reset_index(drop=True)
    U.to_csv(UNITS, index=False)
    gu = U.drop_duplicates(["doc_id", "gold_sent"])
    print(f"units {len(U)} | unique gold sentences {len(gu)} | multi share among gold sentences "
          f"{(gu.group == 'multi').mean():.2f} | by source:\n{gu.groupby('source').group.apply(lambda s: (s == 'multi').mean()).round(2)}")
    # stratified subset: per group, proportional to source, max 2 units per (doc, group), one unit per gold sentence
    rng = np.random.default_rng(seed)
    pick = []
    for grp in ("single", "multi"):
        q = U[U.group == grp].sample(frac=1, random_state=seed).drop_duplicates(["doc_id", "gold_sent"])
        q = q.groupby("doc_id").head(2)
        share = q.source.value_counts(normalize=True)
        for src, s in share.items():
            k = int(round(s * n_per_group))
            pick.append(q[q.source == src].head(k))
    S = pd.concat(pick).reset_index(drop=True)
    S["unit_id"] = np.arange(len(S))
    S.to_csv(SUBSET, index=False)
    print("subset", S.groupby(["group", "source"]).size().unstack(0).to_string(),
          "\nwords by group", S.groupby("group").words.mean().round(1).to_dict(),
          "| best1 by group", S.groupby("group").best1.mean().round(2).to_dict())


# ---------------------------------------------------------------- scoring (cached per metric x text)

def key(metric, prem, hyp):
    return metric + ":" + hashlib.sha1((prem + "\x00" + hyp).encode()).hexdigest()


def load_cache():
    out = {}
    if os.path.exists(SCORES):
        for line in open(SCORES, encoding="utf-8"):
            r = json.loads(line); out[r["k"]] = r["v"]
    return out


class Scorers:
    def __init__(self):
        import torch
        from summary_augmentation.pages.metrics.factcg import FactCGScorer
        from summary_augmentation.pages.metrics.summac import SummacFactory
        self.torch = torch
        self.tok, self.mod = R.load_cls("NLI-hebrew-binary-correctness-metric")
        self.fc = FactCGScorer(device=R.device())
        self.sc = SummacFactory.zero_shot("MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")

    def __call__(self, metric, prem, hyp):
        if metric == "MorphEvalHeb":
            enc = self.tok(prem, hyp, return_tensors="pt", truncation=True, max_length=3072).to(R.device())
            with self.torch.no_grad():
                return float(self.torch.softmax(self.mod(**enc).logits.float(), -1)[0, 1])
        if metric == "FactCG":
            return float(self.fc.score(prem, hyp).score)
        if metric == "SummaC-ZS":
            return float(self.sc.score(prem, hyp).summary_score)


def score(chunk=20, group=None):
    S = pd.read_csv(SUBSET)
    if group:
        S = S[S.group == group]
    S["_r"] = S.groupby("group").cumcount()
    S = S.sort_values(["_r", "group"])        # alternate single/multi: a partial run is still balanced
    prem = R.load_bench().drop_duplicates("doc_id").set_index("doc_id").premise
    cache = load_cache()
    todo = [(m, d, h) for r in S.itertuples() for h in (r.gold_sent, r.corr_sent) for m in METRICS
            for d in [r.doc_id] if key(m, prem[d], h) not in cache]
    todo = list(dict.fromkeys(todo))
    print(f"to score: {len(todo)} (cached {len(cache)})", flush=True)
    if not todo:
        return
    f = Scorers()
    t0 = time.time()
    with open(SCORES, "a", encoding="utf-8") as fh:
        for i in range(0, len(todo), chunk):
            for m, d, h in todo[i:i + chunk]:
                v = f(m, prem[d], h)
                fh.write(json.dumps({"k": key(m, prem[d], h), "metric": m, "doc_id": d, "hyp": h, "v": v},
                                    ensure_ascii=False) + "\n")
            fh.flush()
            n = min(i + chunk, len(todo))
            print(f"  chunk {n}/{len(todo)}  {(time.time() - t0) / n:.2f}s/score", flush=True)


# ---------------------------------------------------------------- analysis

def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def boot_diff(a, b, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return np.percentile(d, [2.5, 97.5])


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return ((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean())


def analyze():
    S = pd.read_csv(SUBSET)
    prem = R.load_bench().drop_duplicates("doc_id").set_index("doc_id").premise
    cache = load_cache()
    for m in METRICS:
        S[f"gold:{m}"] = [cache.get(key(m, prem[r.doc_id], r.gold_sent)) for r in S.itertuples()]
        S[f"corr:{m}"] = [cache.get(key(m, prem[r.doc_id], r.corr_sent)) for r in S.itertuples()]
    S = S.dropna(subset=[f"gold:{m}" for m in METRICS] + [f"corr:{m}" for m in METRICS])
    rows = []
    for m in METRICS:
        S[f"det:{m}"] = (S[f"gold:{m}"] > S[f"corr:{m}"]).astype(float)
        S[f"tie:{m}"] = (S[f"gold:{m}"] == S[f"corr:{m}"]).astype(float)
        t = 0.5 if m != "SummaC-ZS" else np.percentile(S[f"corr:{m}"], 80)
        S[f"fa:{m}"] = (S[f"gold:{m}"] < t).astype(float)
        S[f"caught:{m}"] = (S[f"corr:{m}"] < t).astype(float)
        for what in ("det", "tie", "fa", "caught"):
            a = {g: S[S.group == g][f"{what}:{m}"] for g in ("single", "multi")}
            lo, hi = boot_diff(a["multi"], a["single"]) if min(len(a["multi"]), len(a["single"])) else (np.nan, np.nan)
            rows.append({"metric": m, "measure": {"det": "paired_detection", "fa": "false_alarm_on_gold", "tie": "same_score_gold_and_corrupted",
                                                  "caught": "corrupted_flagged"}[what], "threshold": t,
                         **{f"{g}_n": len(a[g]) for g in a}, **{f"{g}_rate": a[g].mean() for g in a},
                         **{f"{g}_ci": "[%.2f,%.2f]" % wilson(a[g].sum(), len(a[g])) for g in a},
                         "diff_multi_minus_single": a["multi"].mean() - a["single"].mean(),
                         "diff_ci": f"[{lo:.3f},{hi:.3f}]"})
        for g in ("single", "multi"):
            q = S[S.group == g]
            if q.empty:
                continue
            rows.append({"metric": m, "measure": f"auc_{g}", f"{g}_n": len(q),
                         f"{g}_rate": auc(q[f"gold:{m}"], q[f"corr:{m}"])})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "multisource_summary.csv"), index=False)
    S.to_csv(os.path.join(OUT, "multisource_subset_scored.csv"), index=False)
    # length control: within word-count terciles
    S["len_bin"] = pd.qcut(S.words, 3, labels=["short", "mid", "long"], duplicates="drop")
    lc = S.groupby(["len_bin", "group"], observed=True)[[f"fa:{m}" for m in METRICS] + [f"det:{m}" for m in METRICS]].mean()
    lc["n"] = S.groupby(["len_bin", "group"], observed=True).size()
    lc.round(3).to_csv(os.path.join(OUT, "multisource_length_control.csv"))
    pd.set_option("display.width", 250)
    print(f"scored units: {len(S)}", S.group.value_counts().to_dict())
    print(out.round(3).to_string(index=False))
    print("\nlength control\n", lc.round(2).to_string())


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build(int(sys.argv[2]) if len(sys.argv) > 2 else 250)
    elif cmd == "score":
        score(group=sys.argv[2] if len(sys.argv) > 2 else None)
    elif cmd == "analyze":
        analyze()


def figure():
    """figures/multisource_partial.png: prevalence by source + detection / false alarm, single vs multi (with Wilson CIs)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S = pd.read_csv(os.path.join(OUT, "multisource_summary.csv"))
    U = pd.read_csv(UNITS)
    gu = U.drop_duplicates(["doc_id", "gold_sent"])
    prev = gu.groupby("source").group.apply(lambda s: (s == "multi").mean()).sort_values()
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    ax[0].barh(prev.index, prev.values, color="#4a6fa5")
    ax[0].axvline((gu.group == "multi").mean(), ls="--", c="k", lw=1); ax[0].set_xlim(0, 1)
    ax[0].set_title(f"Multi-source share of gold sentences (n={len(gu)}, all={(gu.group == 'multi').mean():.2f})", fontsize=9)
    for i, v in enumerate(prev.values):
        ax[0].text(v + .02, i, f"{v:.2f}", va="center", fontsize=8)
    ms = ["MorphEvalHeb", "SummaC-ZS", "FactCG"]; x = np.arange(3); w = .38
    ci = lambda s: tuple(map(float, s.strip("[]").split(",")))
    for k, (meas, title) in enumerate([("paired_detection", "Catches the error (gold > corrupted)"),
                                       ("false_alarm_on_gold", "False alarm on the correct sentence")]):
        a = ax[k + 1]
        for j, g in enumerate(["single", "multi"]):
            r = [S[(S.metric == m) & (S.measure == meas)].iloc[0] for m in ms]
            v = np.array([q[f"{g}_rate"] for q in r])
            lo = np.array([ci(q[f"{g}_ci"])[0] for q in r]); hi = np.array([ci(q[f"{g}_ci"])[1] for q in r])
            a.bar(x + (j - .5) * w, v, w, yerr=[v - lo, hi - v], capsize=3, color=["#7fa7d9", "#d9822b"][j],
                  label=f"{g}-source (n={int(r[0][f'{g}_n'])})")
            for xx, vv in zip(x + (j - .5) * w, v):
                a.text(xx + .1, vv + .03, f"{vv:.2f}", ha="center", fontsize=7)
        a.set_xticks(x, ms); a.set_ylim(0, 1.12); a.set_title(title, fontsize=9); a.legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    plt.savefig(os.path.join(OUT, "figures", "multisource_partial.png"), dpi=150)
    print("saved figure")


if __name__ == "__main__" and sys.argv[1] == "figure":
    figure()
