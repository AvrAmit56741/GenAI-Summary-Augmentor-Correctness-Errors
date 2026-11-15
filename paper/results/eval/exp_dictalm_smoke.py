"""Smoke test: does DictaLM 3.0 load and run at all on this machine? (lead 11(A) prerequisite check)

Starts with the smallest official DictaLM 3.0 size (1.7B-Instruct, Apache-2.0, dicta-il/DictaLM-3.0-
1.7B-Instruct) given tight system memory (~8.7GB free, shared machine) -- NOT the 12B/24B sizes docs/
tasks.md originally targeted for the judge baseline (lead 1.5); this is a different, lighter-weight use
(correction-agent candidate for lead 11(A)), where a small model may be adequate.

    python exp_dictalm_smoke.py   -> prints load time, one Hebrew generation, peak memory
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eval as R  # noqa: E402

MODEL = "dicta-il/DictaLM-3.0-1.7B-Instruct"


def mem_gb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # bytes on macOS


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"device={R.device()}  loading {MODEL} ...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(R.device())
    print(f"loaded in {time.time() - t0:.1f}s, peak RSS so far {mem_gb():.2f} GB", flush=True)

    prompt = "סכם בשתי משפטים את החדשות הבאות: ראש הממשלה הודיע היום על תוכנית כלכלית חדשה."
    msgs = [{"role": "user", "content": prompt}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(R.device())
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=100, do_sample=False)
    dt = time.time() - t0
    text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    print(f"\ngenerated in {dt:.1f}s ({(out.shape[1]-ids.shape[1])/dt:.1f} tok/s)")
    print(f"PROMPT: {prompt}")
    print(f"OUTPUT: {text}")
    print(f"\npeak RSS: {mem_gb():.2f} GB")


if __name__ == "__main__":
    main()
