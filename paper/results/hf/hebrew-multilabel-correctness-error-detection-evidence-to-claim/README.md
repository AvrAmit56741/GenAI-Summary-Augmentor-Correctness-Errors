---
license: mit
---

# Hebrew Multi-Label Error Type Classifier (Evidence-Based)

## Model Description

Fine-tuned `dicta-il/neodictabert` for multi-label error type classification in Hebrew. Identifies specific factual error types in corrupted claims using evidence sentences extracted from source articles via semantic similarity.

**Task:** Multi-label error type classification (15-20 error types)  
**Language:** Hebrew  
**Max Context:** 512 tokens (evidence + claim)  
**Granularity:** Sentence-level with article context

## Key Features

- **Evidence Extraction:** Sentence-transformer (multilingual-MiniLM-L6-v2) for semantic similarity
- **Similarity Threshold:** 0.5 (minimum for evidence inclusion)
- **Top-K Evidence:** 2-3 most similar sentences per claim
- **Error Types:** Covers Hebrew-specific morphological errors (verb_gender_swap, hebrew_root_pattern_confusion) and cross-language errors (entity_swap, sentence_negation, temporal)

## Architecture

- **Base Model:** `dicta-il/neodictabert`
- **Classification Head:** Multi-label (sigmoid activation per error type)
- **Input Format:** `[CLS] evidence_sentences [SEP] corrupted_claim [SEP]`
- **Output:** Binary vector (length = number of error types) with independent sigmoid probabilities

## Training Configuration

- **Learning Rate:** 2e-5
- **Epochs:** 2.0
- **Batch Size:** 4 per device (effective: 32 with gradient accumulation)
- **Max Sequence Length:** 512 tokens
- **Learning Rate Scheduler:** Cosine
- **Class Balancing:** BCEWithLogitsLoss with pos_weight + WeightedRandomSampler

## Usage

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sentence_transformers import SentenceTransformer
import numpy as np

# Load model
model_name = "Amit5674/hebrew-multilabel-evidence-correctness-error-detection"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    trust_remote_code=True,
    problem_type="multi_label_classification"
)
model.eval()

# Load evidence extractor
evidence_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Extract evidence from article
def extract_evidence(article: str, claim: str, top_k: int = 3, threshold: float = 0.5):
    """Extract top-k most similar sentences from article to the claim"""
    article_sentences = article.split('.')
    claim_embedding = evidence_model.encode(claim)
    article_embeddings = evidence_model.encode(article_sentences)
    
    similarities = np.dot(article_embeddings, claim_embedding) / (
        np.linalg.norm(article_embeddings, axis=1) * np.linalg.norm(claim_embedding)
    )
    
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    evidence_sentences = [article_sentences[i] for i in top_indices if similarities[i] >= threshold]
    return ' '.join(evidence_sentences)

# Example usage
article = "ישראל התחילה בהרעשה רגע אחרי הפסקת האש. הממשלה הודיעה על צעדים חדשים."
corrupted_claim = "ישראל התחילה להתרגש רגע אחרי הפסקת האש"

# Extract evidence
evidence = extract_evidence(article, corrupted_claim, top_k=3, threshold=0.5)

# Tokenize and predict
inputs = tokenizer(
    evidence,
    corrupted_claim,
    return_tensors="pt",
    padding="max_length",
    max_length=512,
    truncation=True
)

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits[0]
    probs = torch.sigmoid(logits)
    
    # Get predicted error types (threshold = 0.5)
    predicted_indices = (probs > 0.5).nonzero(as_tuple=True)[0].tolist()
    predicted_labels = [model.config.id2label[i] for i in predicted_indices]
    
    # Get all scores
    all_scores = {
        model.config.id2label[i]: float(probs[i].item())
        for i in range(model.config.num_labels)
    }

print(f"Predicted error types: {predicted_labels}")
print(f"All scores: {all_scores}")
## Input Format

- **Premise:** Original correct claim + top 2-3 evidence sentences from article (concatenated)
- **Hypothesis:** Corrupted claim with factual errors
- **Evidence Extraction:** Automatic via sentence-transformer similarity matching

## Output Format

- **Predicted Labels:** List of error type strings (e.g., `["hebrew_root_pattern_confusion", "verb_gender_swap"]`)
- **All Scores:** Dictionary mapping error type → probability (0.0 to 1.0)
- **Threshold:** 0.5 (configurable)

## Supported Error Types

**Hebrew-Specific:**
- `hebrew_root_pattern_confusion`, `verb_gender_swap`, `noun_gender_swap`, `morphological_connective_confusion`, `construct_state_confusion`, `specificity_shift_errors`, `verb_tense_swap`

**Cross-Language:**
- `entity_person_swap`, `entity_location_swap`, `entity_organization_swap`, `sentence_negation`, `number_swap`, `measure_unit_swap`, `temporal`

## Limitations

- Max sequence length: 512 tokens (evidence + claim)
- Evidence extraction depends on sentence-transformer quality
- Performance may degrade if evidence sentences are not semantically relevant
- Class imbalance: rare error types may have lower recall

## Citation

@misc{hebrew_evidence_multilabel_classifier,
  title={Hebrew Multi-Label Error Type Classifier with Evidence-Based Context},
  author={Your Name},
  year={2025},
  publisher={Hugging Face}
}