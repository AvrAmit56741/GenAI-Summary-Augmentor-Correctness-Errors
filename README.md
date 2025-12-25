# Hebrew Summary Augmentation & Factuality Checking

A comprehensive Hebrew NLP system for generating corrupted summaries with linguistic errors and training factuality detection models.

## Overview

This project provides an end-to-end pipeline for:
1. **Data Augmentation:** Generating corrupted Hebrew summaries with 20+ error types (morphological, semantic, factual)
2. **Model Training:** Fine-tuning Hebrew BERT models for factuality checking
3. **Evaluation Tools:** Streamlit UI for dataset analysis, model evaluation, and evidence extraction

## Key Components

### 1. Augmentation Pipeline

Automatically generates corrupted Hebrew summaries by introducing controlled factual and linguistic errors while maintaining grammatical fluency.

**Pipeline Workflow:**
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
**Key Features:**
- **Multi-LLM Support:** OpenAI GPT-4, Google Gemini, Anthropic Claude
- **Hebrew NER:** DictaBERT-based named entity recognition
- **Quality Control:** Grammar verification, diversity filtering, transformation validation
- **Metadata Tracking:** Character-level diffs, transformation types, claim mappings

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

## Architecture

### Data Generation Pipeline

### Model Training
# Binary classifier
python mfact_original_binary_classifier.py @training_args.txt

# Multi-label classifier (Setup 2)
python mfact_train_multilabel_claim_level_neo.py @training_args.txt

# Multi-label classifier (Setup 3 with evidence)
python mfact_train_multilabel_evidence_claim_level.py @training_args.txt

Project Structure
GenAI-Summary-Augmentor-Correctness-Errors/
├── summary_augmentation/
│   ├── correctness_errors_augmentation_v2.py    # Main pipeline
│   ├── correctness_errors_examples.py           # Error type examples
│   ├── correctness_errors_llm_helper_functions.py
│   ├── correctness_errors_prompts.py            # LLM prompts
│   ├── utils.py                                  # NER, similarity, filtering
│   ├── config.py                                 # Configuration
│   └── ui.py                                     # Streamlit main app
├── models_finetune/
│   └── bertencoderclassifiers/
│       ├── binaryclassifiers/
│       │   ├── mfact_original_binary_classifier.py
│       │   └── inference_binary_nli.py
│       └── multillabelclassifier/
│           ├── article_summary_errortypes/       # Setup 1
│           ├── article_summaryClaim_errortypes/  # Setup 2
│           └── articleEvidence_summaryClaim_errorTypes/  # Setup 3
└── thesisdatasets/     
