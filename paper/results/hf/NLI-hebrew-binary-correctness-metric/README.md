---
license: apache-2.0
language:
- he
base_model:
- dicta-il/neodictabert
tags:
- nli
- natural-language-inference
- hebrew
- fact-checking
- contradiction-detection
pipeline_tag: text-classification
library_name: transformers
metrics:
- accuracy
- f1
---

# Hebrew Binary NLI Classifier for Factuality Checking

## Model Description

Fine-tuned [dicta-il/neodictabert](https://huggingface.co/dicta-il/neodictabert) for binary Natural Language Inference in Hebrew. Detects whether a summary claim contradicts a source article.

**Task:** Entailment vs Contradiction Detection  
**Language:** Hebrew  
**Max Context:** 4,096 tokens

## Performance

- **Accuracy:** 96.78%
- **F1 Score:** 96.20%

## Architecture

- **Base Model:** `dicta-il/neodictabert`
- **Classification Head:** Binary (softmax over 2 classes)
- **Input Format:** `[CLS] source_article [SEP] summary_claim [SEP]`
- **Output:** Probability distribution over [contradiction, entailment]

## Training Configuration

- **Learning Rate:** 2e-5
- **Epochs:** 2
- **Batch Size:** 2 per device (effective: 16 with gradient accumulation)
- **Max Sequence Length:** 4,096 tokens
- **Learning Rate Scheduler:** Linear
- **Warmup Steps:** 500
- **Best Model Selection:** Based on eval_f1

## Usage

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "Amit5674/NLI-hebrew-binary-correctness-metric"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True)
model.eval()

# Example usage
article = "ישראל התחילה בהרעשה רגע אחרי הפסקת האש. הממשלה הודיעה על צעדים חדשים..."
summary = "ישראל התחילה להתרגש רגע אחרי הפסקת האש"

# Tokenize
inputs = tokenizer(
    article,
    summary,
    return_tensors="pt",
    padding="max_length",
    max_length=4096,
    truncation=True
)

# Predict
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits[0]
    probs = torch.softmax(logits, dim=-1)
    predicted_class_idx = torch.argmax(probs).item()
    predicted_class = model.config.id2label[predicted_class_idx]
    confidence = probs[predicted_class_idx].item()

    probabilities = {
        model.config.id2label[i]: float(probs[i].item())
        for i in range(model.config.num_labels)
    }

print(f"Prediction: {predicted_class}")
print(f"Confidence: {confidence:.4f}")
print(f"Probabilities: {probabilities}")For detailed inference examples, see the inference scripts and server API documentation.

## Input Format

- **Premise:** Source article text (full document)
- **Hypothesis:** Summary claim (can be full summary or individual claim)
- **Processing:** Binary classification (entailment vs contradiction)

## Output Format

- **Prediction:** String label (`"entailment"` or `"contradiction"`)
- **Confidence:** Probability of predicted class (0.0 to 1.0)
- **Probabilities:** Dictionary with probabilities for both classes:
  - `{"entailment": 0.9678, "contradiction": 0.0322}`

## Use Cases

- **Production Fact-Checking:** Fast yes/no contradiction detection for Hebrew summaries
- **Quality Control:** Automated validation of summary factuality
- **Batch Processing:** Efficient processing of large document-summary pairs
- **Real-Time Validation:** Low-latency factuality checking in summary generation pipelines

## Limitations

- Max sequence length: 4,096 tokens (may truncate very long articles)
- Binary classification: Cannot identify specific error types (use multi-label models for detailed error analysis)
- Context dependency: Performance may vary with article length and complexity
- Hebrew-specific: Optimized for Hebrew text; may not generalize to other languages

## Citation

@misc{hebrew_binary_nli_classifier,
  title={Hebrew Binary NLI Classifier for Factuality Checking},
  author={Your Name},
  year={2025},
  publisher={Hugging Face}
}