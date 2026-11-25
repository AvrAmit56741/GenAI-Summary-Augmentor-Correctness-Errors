import os, sys, json
os.environ["HF_HOME"] = "/tmp/tok_cache"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
import truststore
truststore.inject_into_ssl()

TEST_WORDS = [
    ("והמפעל", "ו- (and) + ה- (the) + מפעל (enterprise) = 'and the enterprise'"),
    ("ארוסתו", "ארוסה (fiancee, fem.) + -ו (his) possessive suffix = 'his fiancee'"),
    ("כלפיה", "כלפי (toward) + -ה (her) possessive/pronominal suffix = 'toward her'"),
    ("לחקור", "ל- (to, infinitive marker) + חקור (research) = 'to research'"),
    ("שיכללו", "ש- (that) + יכללו (will include, 3pl fut.) = 'that will include'"),
    ("התזקיקים", "ה- (definite article) + תזקיקים (distillates, pl.) = 'the distillates'"),
    ("ולמנוע", "ו- (and) + ל- (to) + מנוע (prevent) = 'and to prevent'"),
    ("שלה", "של (possessive 'of') + ה (her) = 'hers / her [object]' -- single orthographic word"),
]

RESULTS = {}

def try_tiktoken():
    import tiktoken
    for name in ["cl100k_base", "o200k_base"]:
        enc = tiktoken.get_encoding(name)
        rows = []
        for w, gloss in TEST_WORDS:
            ids = enc.encode(w)
            pieces = [enc.decode([i]) for i in ids]
            rows.append({"word": w, "gloss": gloss, "n_tokens": len(ids), "pieces": pieces})
        RESULTS[f"tiktoken:{name}"] = rows

def try_hf(model_id, label):
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(model_id)
    except Exception as e:
        RESULTS[label] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
        return
    rows = []
    for w, gloss in TEST_WORDS:
        ids = tok.encode(w, add_special_tokens=False)
        pieces = [tok.decode([i]) for i in ids]
        rows.append({"word": w, "gloss": gloss, "n_tokens": len(ids), "pieces": pieces})
    RESULTS[label] = rows

try_tiktoken()

HF_MODELS = [
    ("dicta-il/dictalm2.0", "DictaLM-2.0 (Hebrew-extended)"),
    ("dicta-il/DictaLM-3.0-1.7B-Instruct", "DictaLM-3.0-1.7B (Qwen3-based)"),
    ("Qwen/Qwen3-14B", "Qwen3-14B"),
    ("Qwen/Qwen3-32B", "Qwen3-32B"),
    ("zai-org/GLM-4.6", "GLM-4.6 (sibling tokenizer to 4.7-Flash)"),
    ("deepseek-ai/DeepSeek-V3", "DeepSeek-V3 (predecessor tokenizer)"),
    ("moonshotai/Kimi-K2-Instruct", "Kimi-K2-Instruct"),
    ("google/gemma-3-12b-it", "Gemma-3-12B"),
    ("mistralai/Ministral-8B-Instruct-2410", "Ministral (Tekken tokenizer family)"),
]

for model_id, label in HF_MODELS:
    try_hf(model_id, label)

out_path = "/private/tmp/claude-502/-Users-amit-avraham-PycharmProjects-urlf-classification/8d93004e-ad4a-48df-9a33-31c792ea0385/scratchpad/tokenizer_probe_results.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(RESULTS, f, ensure_ascii=False, indent=2)

print("DONE")
for k in RESULTS:
    v = RESULTS[k]
    if isinstance(v, dict) and "error" in v:
        print(k, "-> ERROR:", v["error"][:150])
    else:
        print(k, "-> OK,", len(v), "words tokenized")
