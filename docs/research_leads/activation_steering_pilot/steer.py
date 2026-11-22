"""
Cheap causal follow-on to the representation-geometry pilot: is the layer-24 verb_gender_swap
mean difference vector (erroneous - faithful, averaged over the 10 verb_gender_swap examples)
actually STEERABLE -- does adding it back into the residual stream during generation cause the
model to produce gender-agreement errors it wouldn't otherwise make?

The geometry pilot only showed the vector is a *descriptive* regularity (same-family errors
cluster). This tests whether it's *causal* (reusable as an actual intervention), which is a much
stronger and more useful claim if it holds.

Method: register a forward hook on decoder layer 24 of DictaLM-3.0-1.7B-Instruct (0-indexed
layer 23, i.e. the layer whose OUTPUT is hidden_states[24] in the extraction pipeline) that adds
scale * v to every position's hidden state during generation. Task: ask the model to repeat a
held-out Hebrew sentence verbatim (new sentences, not from the 27-example geometry-pilot set,
diverse topics away from the pilot's dog-shelter cluster). Baseline (no steering) should reproduce
the sentence close to verbatim; if the steering vector is causal, roughly the same reproduction
should flip the subject's verb/noun gender marking under steering.

A random, norm-matched control vector is included to rule out "any big perturbation breaks
grammar" as a null explanation for whatever effect is observed.

    python steer.py smoke   # 2 sentences x all conditions, prints everything raw
    python steer.py full    # all test sentences x all conditions, saves steering_results.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "dicta-il/DictaLM-3.0-1.7B-Instruct"
HERE = os.path.dirname(os.path.abspath(__file__))
GEOM_DIR = os.path.join(os.path.dirname(HERE), "representation_geometry_pilot")
LAYER = 24  # hidden_states index used throughout the geometry pilot (peak within/between gap)
LAYER_MODULE_IDX = LAYER - 1  # model.model.layers is 0-indexed; hidden_states[24] = output of layers[23]

# Held-out sentences: NOT from the 27-example geometry-pilot dataset, diverse topics deliberately
# away from that set's dog-shelter cluster, each an unambiguous single-subject-gender sentence so
# a gender flip is easy to spot by eye. Three male-subject, three female-subject.
TEST_SENTENCES = [
    ("M", "המהנדס בדק את המערכת החדשה בזהירות רבה."),
    ("M", "הטבח הכין את המנה המיוחדת למסעדה."),
    ("M", "השחקן קלט את הכדור ורץ לעבר השער."),
    ("F", "המהנדסת בדקה את המערכת החדשה בזהירות רבה."),
    ("F", "הטבחית הכינה את המנה המיוחדת למסעדה."),
    ("F", "השחקנית קלטה את הכדור ורצה לעבר השער."),
]

REPEAT_PROMPT = "חזור בדיוק על המשפט הבא, מילה במילה, ללא שום שינוי:\n\n{sentence}"


def load_vectors():
    d = np.load(os.path.join(GEOM_DIR, "hidden_states.npz"))
    Xf, Xe = d["X_faithful"], d["X_erroneous"]
    records = json.load(open(os.path.join(GEOM_DIR, "records.json"), encoding="utf-8"))
    types = np.array([r["error_type"] for r in records])
    diff = Xe[:, LAYER, :] - Xf[:, LAYER, :]
    mask = types == "verb_gender_swap"
    v_verb = diff[mask].mean(axis=0)
    rng = np.random.default_rng(0)
    v_random = rng.normal(size=v_verb.shape)
    v_random = v_random / np.linalg.norm(v_random) * np.linalg.norm(v_verb)  # norm-matched control
    return v_verb.astype(np.float32), v_random.astype(np.float32)


class Steerer:
    """Adds `scale * vector` to every position's hidden state at the output of one decoder layer.
    Active for the whole generate() call (every decoding step), not just the prompt."""

    def __init__(self, model, layer_idx):
        self.layer = model.model.layers[layer_idx]
        self.vector = None
        self.scale = 0.0
        self.handle = None

    def _hook(self, module, inputs, output):
        if self.vector is None or self.scale == 0.0:
            return output
        hs = output[0] if isinstance(output, tuple) else output
        hs = hs + self.scale * self.vector.to(hs.dtype).to(hs.device)
        if isinstance(output, tuple):
            return (hs,) + output[1:]
        return hs

    def set(self, vector_np, scale):
        self.vector = None if vector_np is None else torch.from_numpy(vector_np)
        self.scale = scale

    def __enter__(self):
        self.handle = self.layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, *a):
        self.handle.remove()


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"

    print(f"Loading {MODEL_ID} ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    model.to(device)
    model.eval()
    print(f"Loaded in {time.time()-t0:.1f}s on {device}")

    v_verb, v_random = load_vectors()
    print(f"verb_gender_swap vector norm={np.linalg.norm(v_verb):.1f}  "
          f"random control vector norm={np.linalg.norm(v_random):.1f}")

    sentences = TEST_SENTENCES[:2] if mode == "smoke" else TEST_SENTENCES
    conditions = [
        ("baseline", None, 0.0),
        ("verb_gender_swap x1", v_verb, 1.0),
        ("verb_gender_swap x2", v_verb, 2.0),
        ("verb_gender_swap x3", v_verb, 3.0),
        ("random_control x2", v_random, 2.0),
    ]

    steerer = Steerer(model, LAYER_MODULE_IDX)
    results = []
    with steerer:
        for gender, sent in sentences:
            print(f"\n{'='*70}\nSOURCE ({gender}-subject): {sent}")
            row = {"gender": gender, "sentence": sent, "outputs": {}}
            for cond_name, vec, scale in conditions:
                steerer.set(vec, scale)
                messages = [{"role": "user", "content": REPEAT_PROMPT.format(sentence=sent)}]
                prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tok(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=60, do_sample=False,
                                          pad_token_id=tok.eos_token_id)
                text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                row["outputs"][cond_name] = text
                print(f"  [{cond_name:22s}] {text}")
            results.append(row)
    steerer.set(None, 0.0)

    if mode == "full":
        out_path = os.path.join(HERE, "steering_results.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nSaved {len(results)} rows to {out_path}")


if __name__ == "__main__":
    main()
