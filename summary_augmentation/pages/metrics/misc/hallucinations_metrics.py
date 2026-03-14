"""
ENFS% (Entity Non-Factual Source) hallucination metric backend.

Measures entity-level hallucinations by comparing named entities in the
summary against the source document.

Reference: Cao et al. (2022) — "Hallucinated but Factual!"

Public API:
    ENFS_EXPLANATION   : markdown string for UI display
    ENTITY_LABELS      : {model_name: [label, ...]} mapping
    load_ner_pipeline  : load (and cache) a HF NER pipeline by model name
    extract_entities   : extract deduplicated entities from text (with chunking)
    calculate_enfs     : compute ENFS% for one (premise, hypothesis) pair
    build_download_rows: flatten results list into a pandas-ready list of dicts
"""

from __future__ import annotations
from typing import List, Optional

# ─── Explanation text ──────────────────────────────────────────────────────────

ENFS_EXPLANATION = """
### ENFS% — Entity Non-Factual Source

Measures entity-level hallucinations by comparing entities in the summary against
the source document.  Based on Cao et al. (2022) — *"Hallucinated but Factual!"*

**Formula:**

$$
ENFS\\% = \\frac{\\text{Entities in summary NOT found in source}}{\\text{Total entities in summary}} \\times 100
$$

**Example:**
- Source: "דני הלך למכולת ביום שלישי"
- Summary: "דני קנה לחם בסופרמרקט שפע"
- Entities in summary: דני (PER), שפע (ORG)
- Entities not in source: שפע
- **ENFS% = 50%**

Higher ENFS% → more hallucinated entities → lower faithfulness.

**Supported models:**

| Model | Entity types |
|---|---|
| `dicta-il/dictabert-ner` | PER, LOC, GPE, ORG, TTL, EVE |
| `avichr/heBERT_NER` | PERS, LOC, ORG, TIME, DATE, MONEY, PERCENT |

**Notes:**
- Long texts are split into overlapping character chunks (1 800 chars, 200 overlap)
  to stay within the 512-token model limit.
- Entity matching is case-insensitive exact match.
- Duplicates within a text are removed before comparison.

> This implementation is fully automatic — no human annotation validation as in
> the original paper.
"""

# ─── Entity label sets ─────────────────────────────────────────────────────────

ENTITY_LABELS: dict = {
    "dicta-il/dictabert-ner": ["PER", "LOC", "GPE", "ORG", "TTL", "EVE"],
    "avichr/heBERT_NER":      ["PERS", "LOC", "ORG", "TIME", "DATE", "MONEY", "PERCENT"],
}

# ─── NER pipeline loader ───────────────────────────────────────────────────────

_PIPELINE_CACHE: dict = {}


def load_ner_pipeline(model_name: str):
    """Load and cache a HuggingFace NER pipeline by model name."""
    if model_name not in _PIPELINE_CACHE:
        from transformers import pipeline
        _PIPELINE_CACHE[model_name] = pipeline(
            "ner", model=model_name, aggregation_strategy="simple"
        )
    return _PIPELINE_CACHE[model_name]


# ─── Entity extraction ─────────────────────────────────────────────────────────

_CHUNK_SIZE = 1800
_OVERLAP    = 200


def _build_chunks(text: str) -> List[str]:
    if len(text) <= _CHUNK_SIZE:
        return [text]
    chunks = []
    pos = 0
    while pos < len(text):
        chunks.append(text[pos: pos + _CHUNK_SIZE])
        pos += _CHUNK_SIZE - _OVERLAP
    return chunks


def extract_entities(text: str, ner_pipeline, threshold: float) -> List[dict]:
    """
    Extract deduplicated NER entities from text.

    Handles long texts via overlapping character chunks.
    Returns list of {'text', 'label', 'score'} dicts.
    """
    if not text or not text.strip():
        return []

    seen: set = set()
    entities: List[dict] = []

    for chunk in _build_chunks(text):
        for ent in ner_pipeline(chunk):
            if ent.get("score", 0) < threshold:
                continue
            word = ent["word"].replace("#", "").strip()
            if len(word) < 2:
                continue
            key = (word.lower(), ent["entity_group"])
            if key not in seen:
                seen.add(key)
                entities.append({"text": word, "label": ent["entity_group"], "score": ent["score"]})

    return entities


# ─── ENFS% calculation ─────────────────────────────────────────────────────────

def calculate_enfs(
    premise_entities: List[dict],
    hypothesis_entities: List[dict],
) -> tuple:
    """
    Compute ENFS% for one sample.

    Returns:
        (enfs_percent, total_hyp_entities, hallucinated_count, hallucinated_list)
        enfs_percent is None when the summary contains no entities.
    """
    if not hypothesis_entities:
        return None, 0, 0, []

    premise_texts = {e["text"].lower() for e in premise_entities}
    hallucinated = [e for e in hypothesis_entities if e["text"].lower() not in premise_texts]
    enfs = (len(hallucinated) / len(hypothesis_entities)) * 100
    return enfs, len(hypothesis_entities), len(hallucinated), hallucinated


# ─── Download helper ───────────────────────────────────────────────────────────

def build_download_rows(results_list: List[dict], df_eval) -> List[dict]:
    """Flatten results list into rows suitable for pd.DataFrame / CSV download."""
    rows = []
    for r in results_list:
        row = df_eval.iloc[r["index"]].to_dict()
        row["enfs_percent"]           = r["enfs_percent"]
        row["total_entities_summary"] = r["total_entities"]
        row["total_entities_premise"] = len(r["premise_entities"])
        row["hallucinated_count"]     = r["hallucinated_entities"]
        row["hallucinated_entities"]  = ", ".join(e["text"] for e in r["hallucinated_list"])
        row["premise_entities"]       = ", ".join(f"{e['text']}({e['label']})" for e in r["premise_entities"])
        row["hypothesis_entities"]    = ", ".join(f"{e['text']}({e['label']})" for e in r["hypothesis_entities"])
        rows.append(row)
    return rows
