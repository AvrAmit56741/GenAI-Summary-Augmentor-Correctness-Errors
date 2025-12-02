# Models Documentation

## Overview
This document describes the three types of models trained for Hebrew factuality checking and error detection.

---

## 1. Binary NLI Classifier

### Model
- **Architecture**: DictaBERT (`dicta-il/neodictabert`)
- **Task**: Binary classification (Entailment vs Contradiction)
- **Purpose**: Detect if a summary claim contradicts the source article

### Dataset Structure
```csv
premise,hypothesis,label
"[Source article text]","[Summary claim]","entailment" or "contradiction"
```

### Training Configuration
- **Max sequence length**: 4096 tokens
- **Batch size**: 2 (effective: 16 with gradient accumulation)
- **Learning rate**: 2e-5
- **Epochs**: 4
- **Best model selection**: Based on eval_f1

### Performance
- **Accuracy**: 96.78%
- **F1 Score**: 96.20%

### Key Features
- ✅ CPU/GPU compatible inference
- ✅ Handles very long Hebrew articles
- ✅ Saves best checkpoint automatically

---

## 2. Multi-Label Error Type Classifiers

All three setups use **DictaBERT** for multi-label classification of error types.

### Setup 1: Summary-Level Classification
**Input Structure:**
```csv
premise,hypothesis,error_types,label
"[Full article]","[Full corrupted summary]","entity_swap,temporal","contradiction"
```

**Characteristics:**
- Granularity: Document-level
- Max sequence: 4096 tokens
- Batch size: 1 (effective: 32)
- Context: Full source article

**Use Case**: Classify all error types in entire summary at once

---

### Setup 2: Claim-Level Classification
**Input Structure:**
```csv
premise,hypothesis,error_types,label
"[Original correct claim]","[Corrupted claim]","entity_swap","contradiction"
```

**Characteristics:**
- Granularity: Sentence-level
- Max sequence: 3072 tokens
- Batch size: 2 (effective: 32)
- Context: Only original correct claim

**Use Case**: Fine-grained error detection per claim

---

### Setup 3: Evidence-Based Claim Classification
**Input Structure:**
```csv
premise,hypothesis,error_types,label
"[Original claim + evidence sentences from article]","[Corrupted claim]","entity_swap","contradiction"
```

**Characteristics:**
- Granularity: Sentence-level
- Max sequence: TBD
- Context: Original claim + semantically similar article evidence
- Evidence extraction: Uses sentence transformer for similarity matching

**Use Case**: Error detection with supporting article evidence

---

### Common Multi-Label Settings
- **Learning rate**: 2e-5
- **Epochs**: 2
- **Optimizer**: AdamW with cosine LR scheduler
- **Warmup steps**: 300
- **Loss**: BCEWithLogitsLoss with class-balanced pos_weight
- **Evaluation metric**: F1 micro-averaged

### Error Types Detected
- entity_swap
- temporal
- negation
- number_swap
- paraphrasing_error
- hallucination
- etc.

---

## 3. LLM Classifier (Gemma-2 9B)

### Model
- **Architecture**: Gemma-2 9B (`unsloth/gemma-2-9b`)
- **Quantization**: 4-bit (QLoRA)
- **LoRA Configuration**:
  - r=16, alpha=16
  - Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

### Task
Given context and a claim, identify error type or verify correctness.

**Input:**
```
CONTEXT: [Original claim + similar article sentences]
CLAIM: [Corrupted claim or correct claim]
```

**Output (JSON):**
```json
{
  "error_type": "entity_swap" or "no_error",
  "claim": "...",
  "context": ["...", "..."]
}
```

---

### Dataset Structure & Parsing

#### Parsing Configuration
```python
# Context extraction
SIMILARITY_THRESHOLD = 0.5      # Min similarity for evidence sentences
MAX_CONTEXT_SENTENCES = 3       # Max evidence sentences per claim
DUPLICATE_THRESHOLD = 0.95      # Avoid near-duplicate sentences

# Prompt formats
PROMPT_FORMAT = "multi"         # Mix of Gemma (50%), Alpaca (35%), ChatML (15%)

# Positive samples (correct claims)
POSITIVE_SAMPLES_MODE = "all_untouched"  # Options: "none", "all_untouched", "ratio"
POSITIVE_RATIO = 2              # For ratio mode: N corruptions → 1 positive
```

#### Evidence Extraction Process
1. For each claim (original or corrupted):
   - Encode claim using sentence transformer (multilingual-MiniLM)
   - Find semantically similar sentences from source article
   - Keep top 2-3 most similar (above threshold, not duplicates)
   - Create context window: [claim + evidence sentences]

#### Prompt Format Distribution
**Gemma Format (50%):**
```
<bos><start_of_turn>user
You are a Hebrew factuality checker...
CONTEXT: [...]
CLAIM: [...]
<end_of_turn>
<start_of_turn>model
{"error_type": "...", "claim": "..."}<end_of_turn><eos>
```

**Alpaca Format (35%):**
```
### Instruction:
Given context and claim, identify errors...
### Input:
CONTEXT: [...] CLAIM: [...]
### Response:
{"error_type": "..."}
```

**ChatML Format (15%):**
```
<|im_start|>system
You are a Hebrew factuality checker...<|im_end|>
<|im_start|>user
CONTEXT: [...] CLAIM: [...]<|im_end|>
<|im_start|>assistant
{"error_type": "..."}<|im_end|>
```

#### Dataset Balance
- **Negative samples**: Claims with errors (entity_swap, temporal, etc.)
- **Positive samples**: Correct claims from gold summaries (not corrupted)
- **Ratio**: Depends on configuration (typically ~1-5% positive in all_untouched mode)

---

### Training Configuration
- **Max sequence length**: 4096 tokens
- **Batch size**: 2 (effective: 16 with gradient_accumulation=8)
- **Learning rate**: 2e-4
- **Epochs**: 1
- **Optimizer**: AdamW 8-bit
- **Strategy**: Save best model by eval_loss
- **Dataset percentage**: 85%

### Memory Optimization
- ✅ 4-bit quantization
- ✅ Gradient checkpointing
- ✅ LoRA (trains only 0.58% of parameters)
- ✅ Dynamic padding (no fixed padding to max_seq_length)

---

## Model Comparison

| Model | Type | Max Tokens | Training Time | Inference Speed | Best For |
|-------|------|------------|---------------|-----------------|----------|
| **Binary BERT** | Encoder | 4096 | ~4 hours | Fast (GPU/CPU) | Production NLI |
| **Multi-label BERT** | Encoder | 3072-4096 | ~2-6 hours | Fast (GPU/CPU) | Error classification |
| **Gemma-2 9B** | Decoder | 4096 | ~12+ hours | Slow (GPU only) | Research, complex reasoning |

---

## File Locations

### Binary Classifier
- Training script: `bertencoderclassifiers/binaryclassifiers/mfact_original_binary_classifier.py`
- Training args: `bertencoderclassifiers/binaryclassifiers/traning_args.txt`
- Inference: `bertencoderclassifiers/binaryclassifiers/inference_binary_nli.py`

### Multi-Label Classifiers
- Setup 1: `bertencoderclassifiers/multillabelclassifier/article_summary_errortypes/`
- Setup 2: `bertencoderclassifiers/multillabelclassifier/article_summaryClaim_errortypes/`
- Setup 3: `bertencoderclassifiers/multillabelclassifier/articleEvidence_summaryClaim_errorTypes/`

### LLM Classifier
- Parser: `llmclassifiers/shared/parse_llm_finetune_dataset.py`
- Training: `llmclassifiers/finetune_gemma2_9b_evidence.py`

---

## Quick Start Commands

### Binary Classifier
```bash
python mfact_original_binary_classifier.py @traning_args.txt
```

### Multi-Label Classifier
```bash
python mfact_train_multilabel_claim_level_neo.py @training_args.txt
```

### LLM Parser & Training
```bash
# 1. Parse dataset
python parse_llm_finetune_dataset.py

# 2. Train
python finetune_gemma2_9b_evidence.py
```

---

## Notes

- All models use **trust_remote_code=True** for DictaBERT
- Best model checkpoints saved automatically using `load_best_model_at_end`
- Training args stored in `training_args.txt` files for reproducibility
- Parse config saved in `parse_config.txt` for LLM datasets

