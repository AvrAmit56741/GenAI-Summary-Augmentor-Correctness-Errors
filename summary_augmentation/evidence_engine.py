import json
import re
from difflib import SequenceMatcher
from typing import List, Dict, Tuple, Optional, Any


def split_sentences(text: str) -> List[str]:
    """Naive sentence splitter supporting common punctuation. Keeps non-empty trimmed sentences."""
    if not text:
        return []
    # Split on ., !, ?, Hebrew sof-pasuk and similar, while preserving minimalism
    parts = re.split(r"(?<=[\.!?\u05C3])\s+", text)
    return [p.strip() for p in parts if p and p.strip()]


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower()) if text else []


def jaccard_similarity(a: str, b: str) -> float:
    ta = set(_tokenize(a))
    tb = set(_tokenize(b))
    if not ta and not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def sequence_matcher_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a or "", b or "").ratio()


try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # Optional dependency

_ST_MODEL: Optional[Any] = None


def has_mini_lm() -> bool:
    return SentenceTransformer is not None


def _get_st_model() -> Optional[Any]:
    global _ST_MODEL
    if _ST_MODEL is None and SentenceTransformer is not None:
        try:
            _ST_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        except Exception:
            _ST_MODEL = None
    return _ST_MODEL


def similarity(a: str, b: str, metric: str = "sequence_matcher") -> float:
    if metric == "jaccard":
        return jaccard_similarity(a, b)
    return sequence_matcher_similarity(a, b)


# Replace rank_evidence implementation to support MiniLM batching
def rank_evidence(claim: str, sentences: List[str], top_k: int = 5, metric: str = "sequence_matcher") -> List[Tuple[str, float]]:
    if metric in ("mini_lm", "minilm", "all-MiniLM-L6-v2"):
        model = _get_st_model()
        if model is not None and sentences:
            try:
                # Normalize embeddings so dot product equals cosine similarity
                qv = model.encode([claim], convert_to_numpy=True, normalize_embeddings=True)[0]
                sv = model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
                scores = (sv @ qv)
                scored = list(zip(sentences, [float(s) for s in scores]))
                scored.sort(key=lambda x: x[1], reverse=True)
                return scored[: max(1, top_k)]
            except Exception:
                pass  # Fallback to default below
    # Default path (SequenceMatcher/Jaccard)
    scored = [(s, similarity(claim, s, metric)) for s in sentences]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[: max(1, top_k)]


def filter_by_threshold(ranked: List[Tuple[str, float]], min_score: float = 0.7) -> List[Tuple[str, float]]:
    return [pair for pair in ranked if pair[1] >= min_score]


def analyze_claim(
    original_claim: str,
    transformed_claim: Optional[str],
    source_article: str,
    top_k: int = 5,
    min_score: float = 0.7,
    metric: str = "sequence_matcher",
    delta_min: float = 0.2,
) -> Dict[str, Any]:
    """Return evidence for original claim; if transformed is provided, also compute support delta."""
    sentences = split_sentences(source_article)
    ranked_orig = rank_evidence(original_claim, sentences, top_k=top_k, metric=metric)
    kept_orig = filter_by_threshold(ranked_orig, min_score=min_score)

    result: Dict[str, Any] = {
        "original_claim": original_claim,
        "transformed_claim": transformed_claim,
        "evidence": [{"sentence": s, "score": float(sc)} for s, sc in kept_orig] or [{"sentence": "", "score": 0.0}],
        "metric": metric,
        "top_k": top_k,
        "min_score": min_score,
    }

    if transformed_claim:
        ranked_trans = rank_evidence(transformed_claim, sentences, top_k=top_k, metric=metric)
        kept_trans = filter_by_threshold(ranked_trans, min_score=min_score)
        mean_orig = sum(sc for _, sc in kept_orig) / len(kept_orig) if kept_orig else 0.0
        mean_trans = sum(sc for _, sc in kept_trans) / len(kept_trans) if kept_trans else 0.0
        delta = mean_orig - mean_trans
        result["transformed_support_delta"] = float(delta)
        result["delta_flagged"] = bool(delta >= delta_min)
        result["transformed_evidence"] = [{"sentence": s, "score": float(sc)} for s, sc in kept_trans] or [{"sentence": "", "score": 0.0}]
        result["delta_min"] = delta_min

    return result


def _iter_summary_claims(summary_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract claims from a single corrupted summary object using its transformation_list."""
    claims: List[Dict[str, Any]] = []
    transformation_list = summary_obj.get("transformation_list", [])
    for t in transformation_list:
        if isinstance(t, dict):
            claims.append(
                {
                    "original_claim": t.get("original_claim", ""),
                    "transformed_claim": t.get("transformed_claim", ""),
                }
            )
    return claims


def analyze_summary(doc: Dict[str, Any], summary_idx: int, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    source_article = doc.get("source_article", "")
    summaries = doc.get("correctness_corrupted_summaries", [])
    if summary_idx < 0 or summary_idx >= len(summaries):
        return results

    summary_obj = summaries[summary_idx]
    for claim_idx, claim in enumerate(_iter_summary_claims(summary_obj)):
        res = analyze_claim(
            claim.get("original_claim", ""),
            claim.get("transformed_claim", ""),
            source_article,
            top_k=params.get("top_k", 5),
            min_score=params.get("min_score", 0.7),
            metric=params.get("metric", "sequence_matcher"),
            delta_min=params.get("delta_min", 0.2),
        )
        res.update({
            "summary_idx": summary_idx,
            "claim_idx": claim_idx,
        })
        results.append(res)
    return results


def analyze_all(docs: List[Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any]:
    all_results: Dict[str, Any] = {}
    for d_idx, doc in enumerate(docs):
        key = str(doc.get("doc_id", f"doc_{d_idx}"))
        doc_results: List[Dict[str, Any]] = []
        summaries = doc.get("correctness_corrupted_summaries", [])
        for s_idx in range(len(summaries)):
            doc_results.extend(analyze_summary(doc, s_idx, params))
        all_results[key] = doc_results
    return all_results


def load_augmentation(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    # Run with in-file arguments (no CLI). Adjust these variables as needed.
    input_path = "augmentation.json"  # set your path here
    output_path = "evidence_results.json"  # set your path here

    try:
        docs = load_augmentation(input_path)
    except Exception as e:
        print(f"Could not load input '{input_path}': {e}")
        docs = []

    default_params = {
        "top_k": 5,
        "min_score": 0.7,
        "metric": "sequence_matcher",  # or "jaccard"
        "delta_min": 0.2,
    }

    results = analyze_all(docs, default_params) if docs else {}

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Wrote evidence results to {output_path}")
    except Exception as e:
        print(f"Could not write output '{output_path}': {e}") 