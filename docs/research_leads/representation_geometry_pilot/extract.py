import json
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset import EXAMPLES

MODEL_ID = "dicta-il/DictaLM-3.0-1.7B-Instruct"

print(f"Loading {MODEL_ID} ...")
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL_ID)
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, output_hidden_states=True)
model.to(device)
model.eval()
print(f"Loaded in {time.time()-t0:.1f}s on {device}. n_layers={model.config.num_hidden_layers} hidden={model.config.hidden_size}")


def last_token_hidden_states(text):
    inputs = tok(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    # hidden_states: tuple of (n_layers+1) tensors, each (1, seq_len, hidden)
    hs = torch.stack(out.hidden_states, dim=0)  # (n_layers+1, 1, seq_len, hidden)
    last = hs[:, 0, -1, :].float().cpu().numpy()  # (n_layers+1, hidden)
    return last


records = []
X_faithful = []
X_erroneous = []
t0 = time.time()
for i, ex in enumerate(EXAMPLES):
    hf = last_token_hidden_states(ex["faithful"])
    he = last_token_hidden_states(ex["erroneous"])
    X_faithful.append(hf)
    X_erroneous.append(he)
    records.append({k: v for k, v in ex.items()})
    print(f"  [{i+1}/{len(EXAMPLES)}] {ex['id']:28s} ({ex['error_type']})")

print(f"Extraction done in {time.time()-t0:.1f}s")

X_faithful = np.stack(X_faithful)   # (n_examples, n_layers+1, hidden)
X_erroneous = np.stack(X_erroneous)  # (n_examples, n_layers+1, hidden)

np.savez(
    "hidden_states.npz",
    X_faithful=X_faithful,
    X_erroneous=X_erroneous,
)
with open("records.json", "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print("Saved hidden_states.npz and records.json")
print("X_faithful shape:", X_faithful.shape)
