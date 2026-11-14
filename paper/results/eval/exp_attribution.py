"""Does the binary verifier attribute its 'unfaithful' decision to the injected edit?

For MAFAT-Aug negatives, the gold summary of the same article is in the benchmark, so a word-level
diff gives the exact edited words. We attribute the contradiction-vs-entailment logit margin to the
summary tokens and test whether attributions point at the edited words.

    python exp_attribution.py [N] [ig]   -> results/eval/attribution.csv, printed summary
Methods: gradient x input (Captum's InputXGradient); with 'ig', also Integrated Gradients (16 steps, zero baseline).
"""
import difflib
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eval as R  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
USE_IG = len(sys.argv) > 2 and sys.argv[2] == "ig"
OUT = os.path.join(R.HERE, "attribution_ig.csv" if USE_IG else "attribution.csv")


def words_with_spans(text):
    return [(m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def edited_word_idx(gold, hyp):
    gw = [w for w, _, _ in words_with_spans(gold)]
    hw = [w for w, _, _ in words_with_spans(hyp)]
    sm = difflib.SequenceMatcher(None, gw, hw, autojunk=False)
    idx = set()
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "insert"):
            idx.update(range(j1, j2))
    return idx, len(hw)


def main():
    d = R.load_bench()
    mafat = d[d.part == "MAFAT-Aug"]
    gold = mafat[(mafat.y == 1) & mafat.method.isna()].drop_duplicates("doc_id").set_index("doc_id").hypothesis
    neg = mafat[(mafat.y == 0) & mafat.doc_id.isin(gold.index)].sample(frac=1, random_state=7)
    tok, model = R.load_cls("NLI-hebrew-binary-correctness-metric")
    dev = R.device()
    emb_layer = model.model.encoder
    captured = {}

    def hook(_, __, out):
        out.retain_grad()
        captured["emb"] = out
    emb_layer.register_forward_hook(hook)

    rows = pd.read_csv(OUT).to_dict("records") if os.path.exists(OUT) and os.environ.get("RESUME") else []
    done_uids = {x["uid"] for x in rows}
    for r in neg.itertuples():
        if len(rows) >= N:
            break
        if r.uid in done_uids:
            continue
        edited, n_words = edited_word_idx(gold[r.doc_id], r.hypothesis)
        if not edited or n_words < 5 or len(edited) == n_words:
            continue
        enc = tok(r.premise, r.hypothesis, return_tensors="pt", truncation="only_first", max_length=3072,
                  return_offsets_mapping=True)
        seq = enc.sequence_ids(0)
        offsets = enc.pop("offset_mapping")[0].tolist()
        enc = enc.to(dev)
        model.zero_grad()
        logits = model(**enc).logits[0]
        margin = logits[0] - logits[1]                       # contradiction - entailment
        margin.backward()
        e = captured["emb"]
        gxi = (e.grad[0] * e[0]).sum(-1).detach().float().cpu().numpy()

        ig = None
        if USE_IG:
            base = torch.zeros_like(e[0])
            total = torch.zeros(e.shape[1], device=dev)
            steps = 16
            for k in range(1, steps + 1):
                alpha = k / steps
                x = (base + alpha * (e[0].detach() - base)).unsqueeze(0).requires_grad_(True)
                # re-run encoder stack from interpolated embeddings by patching the embedding output
                h = emb_layer.register_forward_hook(lambda m_, i_, o_, x=x: x)
                lg = model(**enc).logits[0]
                (lg[0] - lg[1]).backward()
                h.remove()
                total += (x.grad[0] * (e[0].detach() - base)).sum(-1) / steps
                model.zero_grad()
            ig = total.float().cpu().numpy()

        spans = words_with_spans(r.hypothesis)
        word_gxi = np.full(len(spans), -np.inf)
        word_ig = np.full(len(spans), -np.inf)
        word_abs = np.full(len(spans), -np.inf)
        wi = 0
        for t, (sid, (a, b)) in enumerate(zip(seq, offsets)):
            if sid != 1 or a == b:
                continue
            while wi < len(spans) and spans[wi][2] <= a:
                wi += 1
            if wi < len(spans) and spans[wi][1] <= a < spans[wi][2]:
                word_gxi[wi] = max(word_gxi[wi], gxi[t])
                word_abs[wi] = max(word_abs[wi], abs(gxi[t]))
                if ig is not None:
                    word_ig[wi] = max(word_ig[wi], ig[t])
        ok = np.isfinite(word_gxi)
        if ok.sum() < 5:
            continue
        lab = np.array([i in edited for i in range(len(spans))])[ok]
        row = {"uid": r.uid, "error_types": r.error_types, "n_words": int(ok.sum()), "n_edited": int(lab.sum()),
               "pred_unfaithful": bool(margin.item() > 0), "chance_top1": float(lab.mean()),
               "gxi_top1_hit": bool(lab[np.argmax(word_gxi[ok])]),
               "gxi_auc": float(roc_auc_score(lab, word_gxi[ok])) if 0 < lab.sum() < len(lab) else np.nan,
               "abs_top1_hit": bool(lab[np.argmax(word_abs[ok])]),
               "abs_auc": float(roc_auc_score(lab, word_abs[ok])) if 0 < lab.sum() < len(lab) else np.nan}
        if ig is not None:
            wig = word_ig[ok]
            row.update({"ig_top1_hit": bool(lab[np.argmax(wig)]),
                        "ig_auc": float(roc_auc_score(lab, wig)) if 0 < lab.sum() < len(lab) else np.nan})
        rows.append(row)
        pd.DataFrame([row]).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
        if len(rows) % 25 == 0:
            df = pd.DataFrame(rows)
            print(f"{len(rows)}  top1 hit {df.gxi_top1_hit.mean():.3f} (chance {df.chance_top1.mean():.3f})  "
                  f"AUC {df.gxi_auc.mean():.3f}", flush=True)
    df = pd.DataFrame(rows)
    print("\nFINAL n=%d  predicted unfaithful %.3f" % (len(df), df.pred_unfaithful.mean()))
    for m in ["gxi", "abs"] + (["ig"] if USE_IG else []):
        print(f"{m}: top-1 word is an edited word {df[m + '_top1_hit'].mean():.3f}  (chance {df.chance_top1.mean():.3f})  "
              f"mean AUC edited vs other words {df[m + '_auc'].mean():.3f}")


if __name__ == "__main__":
    main()
