"""Extract one qualitative example for the error-localization figure (docs/tasks.md lead #1).

Re-runs exp_attribution.py's per-example computation for a single uid (chosen from attribution.csv:
abs_top1_hit=True, small n_edited, for a readable figure) and prints the full texts, the edited
word(s), and the top-3 words by attribution magnitude, so the figure can be built by hand with
verified, real model output rather than a plausible-looking fabrication.

    python exp_attribution_example.py <uid>
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eval as R  # noqa: E402
from exp_attribution import edited_word_idx, words_with_spans  # noqa: E402

UID = int(sys.argv[1])


def main():
    d = R.load_bench()
    mafat = d[d.part == "MAFAT-Aug"]
    gold = mafat[(mafat.y == 1) & mafat.method.isna()].drop_duplicates("doc_id").set_index("doc_id").hypothesis
    r = mafat[mafat.uid == UID].iloc[0]

    print("PREMISE (source):", r.premise[:300])
    print()
    print("GOLD (faithful):", gold[r.doc_id])
    print()
    print("HYPOTHESIS (corrupted):", r.hypothesis)
    print()
    print("error_types:", r.error_types)

    edited, n_words = edited_word_idx(gold[r.doc_id], r.hypothesis)
    spans = words_with_spans(r.hypothesis)
    words = [w for w, _, _ in spans]
    print("\nedited word indices:", sorted(edited), "->", [words[i] for i in sorted(edited)])

    tok, model = R.load_cls("NLI-hebrew-binary-correctness-metric")
    dev = R.device()
    emb_layer = model.model.encoder
    captured = {}

    def hook(_, __, out):
        out.retain_grad()
        captured["emb"] = out
    emb_layer.register_forward_hook(hook)

    enc = tok(r.premise, r.hypothesis, return_tensors="pt", truncation="only_first", max_length=3072,
              return_offsets_mapping=True)
    seq = enc.sequence_ids(0)
    offsets = enc.pop("offset_mapping")[0].tolist()
    enc = enc.to(dev)
    model.zero_grad()
    logits = model(**enc).logits[0]
    margin = logits[0] - logits[1]
    margin.backward()
    e = captured["emb"]
    gxi = (e.grad[0] * e[0]).sum(-1).detach().float().cpu().numpy()

    word_abs = np.full(len(spans), -np.inf)
    wi = 0
    for t, (sid, (a, b)) in enumerate(zip(seq, offsets)):
        if sid != 1 or a == b:
            continue
        while wi < len(spans) and spans[wi][2] <= a:
            wi += 1
        if wi < len(spans) and spans[wi][1] <= a < spans[wi][2]:
            word_abs[wi] = max(word_abs[wi], abs(gxi[t]))

    ok = np.isfinite(word_abs)
    order = np.argsort(-word_abs[ok])
    idxs = np.arange(len(spans))[ok]
    print("\ntop-5 words by |gradient x input|:")
    for rank, j in enumerate(order[:5]):
        wi2 = idxs[j]
        print(f"  #{rank+1}  word='{words[wi2]}'  idx={wi2}  score={word_abs[wi2]:.4f}  is_edited={wi2 in edited}")
    print(f"\npredicted unfaithful: {margin.item() > 0}  margin={margin.item():.3f}")


if __name__ == "__main__":
    main()
