# Hebrew Summary Augmentation & Factuality Checking

A comprehensive Hebrew NLP system for generating corrupted summaries with linguistic errors and training factuality detection models.

## Overview

This project provides an end-to-end pipeline for:
1. **Data Augmentation:** Generating corrupted Hebrew summaries with 20+ error types (morphological, semantic, factual)
2. **Model Training:** Fine-tuning Hebrew BERT models for factuality checking
3. **Evaluation Tools:** Streamlit UI for dataset analysis, model evaluation, and evidence extraction
4. **API Server:** FastAPI-based inference server for model deployment

## Key Components

### 1. Augmentation Pipeline

Automatically generates corrupted Hebrew summaries by introducing controlled factual and linguistic errors while maintaining grammatical fluency.

**Pipeline Workflow:**
```
Input: Gold Summary + Source Article
↓
[1] Claim Decomposition (sentence/fact/paragraph-based)
↓
[2] Transformation Candidate Selection (LLM-based: GPT-4/Gemini/Claude)
↓
[3] Error Application
├── NER-Based: Entity swaps (person/location/organization)
├── Static Rules: Number/date/measure unit changes
└── LLM-Generated: Hebrew morphological errors
↓
[4] Grammar Verification (LLM validates fluency)
↓
[5] Diversity Filtering (Jaccard similarity threshold: 0.93)
↓
[6] Summary Reconstruction
↓
Output: JSON with 7-10 corrupted variants per document
```

**Key Features:**
- **Multi-LLM Support:** OpenAI GPT-4, Google Gemini, Anthropic Claude
- **Hebrew NER:** DictaBERT-based named entity recognition
- **Quality Control:** Grammar verification, diversity filtering, transformation validation
- **Metadata Tracking:** Character-level diffs, transformation types, claim mappings

### Data Generation Pipeline Details

The augmentation pipeline follows a structured workflow to generate corrupted summaries:

**Data Flow:**
```
Input Summary
↓
Claim Decomposition
↓
Transformation Candidate Identification
↓
Error Application
├── NER-Based Swaps
├── Static Transformations
└── LLM-Generated Errors
↓
Grammar Verification
↓
Summary Reconstruction
↓
Diversity Filtering
↓
Output: Corrupted Summaries (JSON format)
```

**Hebrew Language Specialization:**
- Morphologically-aware error generation
- Hebrew NER with DictaBERT integration
- Grammar verification for Hebrew text
- Cultural and linguistic context preservation

**Multi-Modal Error Generation:**
- Rule-based transformations (entity swaps, number changes)
- LLM-generated morphological errors
- Hybrid approaches combining multiple error types

**Quality Control:**
- Similarity-based filtering to ensure diversity
- Grammar verification to maintain fluency
- Transformation success validation
- Configurable error density control

**Scalability:**
- Multi-threaded processing support
- Batch processing capabilities
- Configurable transformation intensity
- Memory-efficient claim processing

**Configuration & Customization:**

The pipeline provides extensive configuration options through `config.py`:

**Claim Decomposition Method:**
- **Options**: `SENTENCES`, `FACTS`, `PARAGRAPH`
- **Purpose**: Controls how summaries are broken into claims for transformation

**Transformation Configuration:**
- Enable/disable specific error transformation types
- Three categories: Hebrew LLM transformations, General LLM transformations, NER-based transformations
- Accessed via `get_enabled_transformations_as_lists()` and `get_enabled_transformations_as_dicts()`

**Similarity & Quality Control:**
- `MAXIMUM_ALLOWED_NEGATIVE_SUMMARIES_SIMILARITY_THRESHOLD = 0.93` - Controls diversity filtering
- `PERCENTAGE_TRANSFORMATIONS_PERFORM_ON_NEGATIVE_SUMMARY = 95` - Controls corruption intensity (percentage of claims to transform)

**LLM Model Selection:**
- Configurable models for claim selection and transformation generation
- Options: `"openai"`, `"gemini"`, `"claude"`

**Diversity Filtering Algorithm:**

The pipeline uses a sophisticated diversity filtering system to ensure corrupted summaries are sufficiently different:

1. **Similarity Metric**: Combines unigram and bigram Jaccard similarity
   - Uses maximum of unigram/bigram similarity scores
   - Formula: `Jaccard(A, B) = |A ∩ B| / |A ∪ B|`

2. **Adaptive Threshold**: Base threshold increases with error complexity
   - `max_same = max_same + 0.0013 * (len(error_types) ** 0.75)`

3. **Dual Filtering**:
   - **vs. Golden**: Corrupted summary must be ≤ `max_same` similar to original
   - **vs. Kept**: Corrupted summary must be ≤ `max_same` similar to already-accepted summaries

4. **N-gram Analysis**:
   - **Unigrams**: Word-level overlap detection
   - **Bigrams**: Phrase-level similarity detection

**Workflow Customization Points:**
- Claim decomposition method: Switch between sentence-based, fact-based, or paragraph-based segmentation
- Error type selection: Enable/disable specific transformation families
- Model selection: Choose different LLMs for claim selection vs transformation generation
- Similarity thresholds: Adjust diversity filtering sensitivity
- Corruption intensity: Control what percentage of claims get transformed
- Multi-threading: Configurable parallelization for batch processing

### 2. Factuality Detection Models

#### Binary NLI Classifier
- **Task:** Entailment vs Contradiction detection
- **Model:** DictaBERT (dicta-il/neodictabert)
- **Performance:** 96.78% accuracy, 96.20% F1
- **Context:** 4,096 tokens (full Hebrew articles)
- **Use Case:** Production systems, fast yes/no contradiction detection

#### Multi-Label Error Type Classifiers (3 Setups)

**Setup 1: Summary-Level**
- **Input:** Full article + full summary
- **Granularity:** Document-level
- **Use Case:** Batch processing, holistic quality assessment

**Setup 2: Claim-Level**
- **Input:** Original claim + corrupted claim
- **Granularity:** Sentence-level
- **Use Case:** Precise error localization for editors

**Setup 3: Evidence-Based Claim-Level**
- **Input:** Original claim + article evidence + corrupted claim
- **Evidence Extraction:** Sentence-transformer similarity (threshold: 0.5)
- **Use Case:** Context-aware fact-checking with citations

## Supported Error Types

### Cross-Language Errors (9 types)
- `entity_person_swap` - Swapping person names
- `entity_location_swap` - Swapping place names
- `entity_organization_swap` - Swapping organization names
- `entity_event_swap` - Swapping event names
- `entity_title_swap` - Swapping job titles
- `entity_date_swap` - Changing dates
- `number_swap` - Altering numerical values
- `measure_unit_swap` - Changing units (meters ↔ kilometers)
- `sentence_negation` - Adding/removing negation

### Hebrew-Specific Linguistic Errors (15+ types)

#### Morphological Errors
- `hebrew_root_pattern_confusion` - Root/binyan confusions (ר.ע.ש ↔ ר.ג.ש)
- `verb_gender_swap` - Verb gender disagreement (הציג ↔ הציגה)
- `noun_gender_swap` - Noun/pronoun gender errors (הוא ↔ היא)
- `homographic_gender_errors` - Gender mismatches in homographic words (מורה, מרצה)
- `verb_tense_swap` - Tense inconsistencies (הולך ↔ הלך ↔ ילך)

#### Syntactic Errors
- `morphological_connective_confusion` - Logical connector errors (שכן ↔ כך ש, למרות ש ↔ מכיוון ש)
- `construct_state_confusion` - Smichut (סמיכות) errors
- `directional_preposition_swap` - Direction reversals (מ-X ל-Y ↔ מ-Y ל-X)
- `subject_verb_person_mismatch` - Person agreement errors (1st/2nd/3rd)

#### Semantic Errors
- `specificity_shift_errors` - Definite article changes (ה-)
- `evidentiality_source_attribution_collapse` - Attribution removal (לפי X → removed)
- `impersonal_to_personal_verb_errors` - Voice changes (הוחלט → הוא החליט)
- `verb_template_agency_reversal` - Agency reversals (הלביש ↔ התלבש)
- `incremental_temporal_state_mutations` - Timeline/order changes
- `double_verb_dependency_errors` - Verb dependency errors
- `paraphrasing_error` - Semantic distortions
- `hallucination` - Non-factual additions

## Model Training

### Binary Classifier
```bash
python models_finetune/bertencoderclassifiers/binaryclassifiers/mfact_original_binary_classifier.py @training_args.txt
```

### Multi-Label Classifiers
```bash
# Setup 1: Summary-level
python models_finetune/bertencoderclassifiers/multillabelclassifier/article_summary_errortypes/mfact_train_multilabel_neo.py @training_args.txt

# Setup 2: Claim-level
python models_finetune/bertencoderclassifiers/multillabelclassifier/article_summaryClaim_errortypes/mfact_train_multilabel_claim_level_neo.py @training_args.txt

# Setup 3: Evidence-based claim-level
python models_finetune/bertencoderclassifiers/multillabelclassifier/articleEvidence_summaryClaim_errorTypes/mfact_train_multilabel_evidence_claim_level.py @training_args.txt
```

## Inference API Server

The project includes a FastAPI-based inference server (`inference_api_server/server_api.py`) that provides REST endpoints for model inference. The server supports binary and multi-label classifiers with optional attention visualization. Endpoints include `/binary/predict`, `/binary/v2/predict`, `/multilabel/setup1/predict`, `/multilabel/setup2/predict`, `/multilabel/setup2/v2/predict`, and `/multilabel/setup3/predict`. The server loads models at startup and handles requests with premise-hypothesis pairs, returning predictions, confidence scores, and optional attention weights.

## Streamlit UI

The project includes a comprehensive Streamlit-based web interface (`summary_augmentation/ui.py`) with 8 main tabs for dataset analysis and quality control: **JSON Analyzer** (browse corrupted summaries with visual diffs), **CSV Reviewer** (filter and analyze augmented datasets), **Utils** (dataset splitting and K-fold setup), **Analysis** (statistical distributions), **Error Analyzer** (error type deep-dives), **API Logs** (LLM request/response inspection), **Dataset Analysis** (multi-dataset comparison), and **Dataset Modification** (in-place editing). Additional pages include **Model Analytics** (F1/Precision/Recall per error type), **Evidence Analysis** (claim-evidence validation), **Mafat Dataset Analysis** (JSONL browser), and **Correctness Evaluation Metrics** (ROUGE/BLEU/ChrF computation).

## Project Structure

```
pythonProject3/
├── summary_augmentation/              # Main augmentation pipeline
│   ├── correctness_errors_augmentation_v2.py    # Main pipeline orchestration
│   ├── correctness_errors_examples.py           # Error type examples & rules
│   ├── correctness_errors_llm_helper_functions.py  # LLM integration (GPT-4/Gemini/Claude)
│   ├── correctness_errors_prompts.py            # Prompt templates & Pydantic schemas
│   ├── correctness_errors_finder.py            # Error detection & analysis
│   ├── utils.py                                  # Core utilities (NER, filtering, decomposition)
│   ├── config.py                                 # Configuration management
│   ├── logger_config.py                         # Centralized logging
│   ├── ui.py                                     # Streamlit main app (8 tabs)
│   ├── model_analytics.py                        # Model evaluation backend
│   ├── evidence_engine.py                        # Evidence extraction backend
│   ├── summaries_generator.py                    # Summary generation utilities
│   ├── summary_generator_ollama.py               # Open-source LLM summaries
│   ├── pages/                                    # Streamlit pages
│   │   ├── 1_🔬_Model_Analytics.py              # Model evaluation page
│   │   ├── 2_🧾_Evidence_Analysis.py            # Evidence validation page
│   │   ├── 3_📊_Mafat_Dataset_Analysis.py       # JSONL browser page
│   │   └── 4_Correctness_Evaluation_Metrics.py  # ROUGE/BLEU/ChrF page
│   └── llm_api_logs/                             # LLM request/response logs
│
├── models_finetune/                              # Model training scripts
│   ├── bertencoderclassifiers/
│   │   ├── binaryclassifiers/                    # Binary NLI classifier
│   │   │   ├── mfact_original_binary_classifier.py
│   │   │   ├── inference_binary_nli.py
│   │   │   └── unified_classifier_evaluate_with_attention_binary_nli.py
│   │   └── multillabelclassifier/                # Multi-label error classifiers
│   │       ├── article_summary_errortypes/       # Setup 1: Summary-level
│   │       ├── article_summaryClaim_errortypes/  # Setup 2: Claim-level
│   │       └── articleEvidence_summaryClaim_errorTypes/  # Setup 3: Evidence-based
│   └── llmclassifiers/                           # LLM classifier (Gemma-2 9B)
│       ├── finetune_gemma2_9b_evidence.py
│       └── shared/
│           └── parse_llm_finetune_dataset.py
│
├── inference_api_server/                          # FastAPI inference server
│   ├── server_api.py                             # Main API server (BERT models)
│   └── server_api_vllm.py                        # vLLM server (LLM models)
│
├── docs/                                         # Documentation
│   ├── README.md                                 # This file
│   ├── augmentation_project_structure_docs.md     # Detailed pipeline docs
│   └── classifier_models_docs.md                 # Model architecture docs
│
└── thesisdatasets/                               # Datasets
    ├── train_datasets/                           # Training data
    └── error_finder/                              # Error analysis outputs
```     
