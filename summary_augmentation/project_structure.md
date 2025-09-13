# Hebrew Summary Augmentation Project Structure

## Overview
This project focuses on Hebrew text processing and summary augmentation, specifically designed to generate corrupted summaries with deliberate linguistic errors for training and evaluating Hebrew Natural Language Inference (NLI) models. The system introduces morphologically-aware errors that exploit unique characteristics of Hebrew grammar and morphology.

## Project Architecture
summary_augmentation/
├── config.py                                          # Configuration management
├── correctness_errors_transformation_workflow.py      # Main orchestration logic
├── correctness_errors_llm_helper_functions.py        # LLM interaction utilities
├── correctness_errors_prompts.py                     # Prompt templates and schemas
├── correctness_errors_examples.py                    # Hebrew linguistic error examples
├── utils.py                                          # Core utilities and data structures
├── logger_config.py                                  # Logging configuration
├── ui.py                                             # Streamlit main app (multipage hub)
├── model_analytics.py                                # Model Analytics page (internal tabs)
├── evidence_engine.py                                # Evidence extraction backend (sentences + similarity)
├── pages/1_🔬_Model_Analytics.py                     # Streamlit multipage entry for Model Analytics
└── pages/2_🧾_Evidence_Analysis.py                   # Streamlit multipage entry for Evidence Analysis

## Core Components

### 1. **Configuration Layer** (`config.py`)
- **Purpose**: Centralized configuration management for transformation types
- **Key Features**:
  - Toggle-based transformation control (Hebrew-specific, general, NER-based)
  - Similarity thresholds and processing parameters
  - Claim decomposition method selection
- **Configuration Categories**:
  - Hebrew LLM transformations (morphological errors)
  - General LLM transformations (semantic errors)
  - NER-based transformations (entity swaps)

### 2. **Workflow Orchestration** (`correctness_errors_transformation_workflow.py`)
- **Purpose**: Main processing pipeline for summary corruption
- **Key Functions**:
  - `correctness_errors_transformation_workflow()` - Main entry point
  - `all_suitable_transformation_possible()` - Identifies transformable claims
  - `apply_factuality_transformation_on_claims()` - Applies transformations
  - `recompose_transformed_summary()` - Reconstructs corrupted summaries
- **Processing Pipeline**:
  1. Claim decomposition
  2. Transformation candidate identification
  3. Error application (NER, static, LLM-based)
  4. Grammar verification
  5. Summary reconstruction
  6. Diversity filtering

### 3. **LLM Integration** (`correctness_errors_llm_helper_functions.py`)
- **Purpose**: Multi-model LLM interaction layer
- **Supported Models**: OpenAI GPT-4, Google Gemini, Anthropic Claude
- **Key Functions**:
  - `choose_suitable_claim()` - Identifies transformation candidates
  - `summary_negative_augmentation()` - Applies error transformations
  - `verify_transformed_summary_grammar()` - Grammar validation
  - `correctness_error_finder()` - Error detection and analysis
- **Logging**: LLM API request/response logs are written to `llm_api_logs/` as JSONL files.

### 4. **Prompt Engineering** (`correctness_errors_prompts.py`)
- **Purpose**: Structured prompt templates and response schemas
- **Key Components**:
  - Pydantic models for structured LLM outputs
  - Hebrew-aware prompt templates
  - Grammar verification prompts
  - Error-specific transformation instructions

### 5. **Linguistic Error Examples** (`correctness_errors_examples.py`)
- **Purpose**: Curated examples of Hebrew morphological errors
- **Error Categories**:
  - Gender agreement errors
  - Verb conjugation errors
  - Root pattern confusions
  - Morphological connectivity issues
- **Structure**: Detailed examples with transformation rules and explanations

#### Examples Format (Schema)
- Organized as a dictionary keyed by error family (e.g., `measure_unit_swap`, `entity_person_swap`). Each family contains:
  - **Transformation rules / Transformation_rules**: List of prescriptive constraints the transformation must follow (naming varies by family).
  - **Description**: One-paragraph explanation of the error family in Hebrew (and/or English).
  - **Examples (typical fields)**:
    - `Original_LLM_Generative_Summary`: Source (gold/original) clause/sentence.
    - `Error_transformation_LLM_Generative_Summary`: Transformed (corrupted) version.
    - `Error_explanation`: Short rationale describing the factual/morphological change.
  - **Optional metadata fields (family-dependent)**:
    - `Linguistic_patterns`, `Morphological_Note`, `Hebrew_Linguistic_Note`: linguistic cues, patterns, or notes.
    - `Linguistic_metadata`: structured breakdown of original/transformed elements (e.g., name, title, verb, gender, number, binyan, tense), and inconsistencies.
    - `Morphological_analysis`, `Target_words_examples`: analysis notes and target lexemes.
    - `Transformation_type`, `Rule_word_של`, `Critical_note`: subtype, diagnostic checks (e.g., סמיכות via "של"), and caveats.
    - `Critical_instruction` / `CRITICAL_INSTRUCTION`: must-have guardrails for valid transformations.
    - `EXPLICIT_EXAMPLES_FOR_LLM_INTRO`, `EXPLICIT_EXAMPLES_FOR_LLM`: guided patterns/examples for LLM prompting.
    - `Specific_change` / `Specific_changes`: minimal token-level edits performed.
    - `NON_VALID_TRANSFORMATION_FALSE_POSITIVE` with `Pairs`: counter-examples to exclude (synonyms, stylistic only, invalid constructs).
- Notes:
  - Field names are mostly stable but sometimes vary in casing/underscores (e.g., `Transformation rules` vs `Transformation_rules`).
  - Families may include both valid examples and explicit non-valid examples to define decision boundaries.

### 6. **Core Utilities** (`utils.py`)
- **Purpose**: Foundational data structures and processing functions
- **Key Components**:
  - `CorrectnessErrorTypes` enum - Comprehensive error taxonomy
  - `ClaimDecompositionMethod` enum - Text segmentation strategies
  - Hebrew NER processing pipeline
  - Text similarity and filtering algorithms
  - Entity extraction and classification

### 7. **UI (Streamlit)** (`ui.py`, `model_analytics.py`, `pages/1_🔬_Model_Analytics.py`)
- **Purpose**: Interactive analysis tools and reports
- **Main App (`ui.py`) Tabs**:
  - JSON Analyzer, CSV Reviewer, Utils, Analysis, Error Analyzer, API Logs, Dataset Analysis, Dataset Modification
- **Model Analytics Page** (`model_analytics.py` via `pages/1_🔬_Model_Analytics.py`):
  - Internal tabs: Multilabel Classifier, Binary Classifier
  - Multilabel analysis computes per-error-type F1/Precision/Recall, confidence on correct/wrong, and overall weighted metrics
- **Logs**: API Logs Analyzer reads JSONL logs from `llm_api_logs/`
- **JSON Analyzer Enhancements**: Can export selected claim-level transformations as JSON for chosen error types.
- **Evidence Analysis Page** (`pages/2_🧾_Evidence_Analysis.py`):
  - Upload augmentation JSON; choose scope: specific claim, one summary, or all summaries
  - Parameters: Top-K, similarity threshold, metric (`sequence_matcher`, `jaccard`, optional `mini_lm`), delta threshold
  - Backend (`evidence_engine.py`) splits source into sentences, ranks by similarity, filters by threshold, and contrasts original vs transformed claim support

## Error Taxonomy

### Hebrew-Specific Errors
- **Morphological**: Root pattern confusion, gender agreement, verb conjugation
- **Syntactic**: Construct state confusion, pronoun reference errors
- **Semantic**: Specificity shifts, temporal mutations

### General Errors  
- **Logical**: Sentence negation, measure unit swaps
- **Factual**: Entity swaps, date modifications, number alterations

### NER-Based Errors
- **Entity Replacements**: Person, location, organization, title swaps
- **Contextual Substitutions**: Maintaining semantic plausibility

### Supported Factual Error Families (examples)
Derived from `correctness_errors_examples.py`, the pipeline supports generation and detection of the following factual error families (non-exhaustive):
- **Measure/Number changes**: `measure_unit_swap`, `number_swap` – unit/value alterations that change quantitative facts.
- **Entity swaps**: `entity_person_swap`, `entity_location_swap`, `entity_organization_swap`, `entity_event_swap`, `entity_title_swap`, `entity_date_swap` – substitutions that change the who/where/what/when.
- **Negation/logical flips**: `sentence_negation`, `morphological_connective_confusion` – flips of polarity/logic via negation or Hebrew connectives.
- **Hebrew morphological/phonological confusions**: `hebrew_root_pattern_confusion`, `homographic_gender_errors`, `verb_gender_swap`, `noun_gender_swap` – root/pattern/gender mix-ups that alter factual content.
- **Attribution/source shifts**: `evidentiality_source_attribution_collapse`, `impersonal_to_personal_verb_errors` – changing the source/attribution of claims.
- **Agency/role reversals**: `verb_template_agency_reversal`, `directional_preposition_swap`, `double_verb_dependency_errors`, `subject_verb_person_mismatch` – who-did-what-to-whom reversals (active↔passive, direction, argument roles).
- **Temporal mutations**: `incremental_temporal_state_mutations`, `entity_date_swap` – timeline/order/historicity changes.
- **Specificity changes**: `specificity_shift_errors` – generic↔specific swaps that alter the referenced entity/scope.

### Hebrew-Specific Error Details
- **`hebrew_root_pattern_confusion`**: Replaces a word with another that shares the same root but has a different pattern, altering the meaning. (בלבול בין משקלים שונים לאותו שורש)
- **`homographic_gender_errors`**: For homographs (same spelling, different meaning/pronunciation), swaps the gender to one appropriate for the *wrong* meaning. (טעות מין דקדוקי במילים הומוגרפיות)
- **`verb_gender_swap`**: Changes the gender of a verb, creating a mismatch with its subject. (החלפת מין הפועל כך שלא יתאים לנושא)
- **`noun_gender_swap`**: Changes the gender of a noun, requiring adjustments to adjectives/verbs. (החלפת מין של שם עצם)
- **`morphological_connective_confusion`**: Replaces a morphological connective (e.g., 'ש' - that, 'ו' - and) with another, altering the relationship between clauses. (החלפת מילת קישור מורפולוגית המשנה את הקשר הלוגי)
- **`evidentiality_source_attribution_collapse`**: Removes attribution (e.g., "according to X"), presenting a claim as objective fact. (השמטת ייחוס, הצגת טענה כעובדה)
- **`impersonal_to_personal_verb_errors`**: Changes an impersonal verb form (e.g., "it was decided") to a personal one ("he decided"). (הפיכת פועל סביל לאישי)
- **`verb_template_agency_reversal`**: Swaps verb templates (binyanim) to reverse agency (e.g., active to passive). (היפוך סוכֵן-מושא על ידי שינוי בניין הפועל)
- **`directional_preposition_swap`**: Replaces a preposition to reverse directionality ("from X to Y" becomes "from Y to X"). (החלפת מילת יחס המשנה כיווניות)
- **`double_verb_dependency_errors`**: In sentences with two verbs, creates a mismatch in the dependency (e.g., tense, subject). (יצירת אי-התאמה בין שני פעלים במשפט)
- **`subject_verb_person_mismatch`**: Changes the verb's person (1st/2nd/3rd) to mismatch the subject. (שינוי גוף הפועל כך שלא יתאים לנושא)

These families are used across the pipeline for augmentation (negative transformations), correctness finding, and evaluation.

## Data Flow
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
Output: Corrupted Summaries

## Key Features

### Hebrew Language Specialization
- Morphologically-aware error generation
- Hebrew NER with DictaBERT integration
- Grammar verification for Hebrew text
- Cultural and linguistic context preservation

### Multi-Modal Error Generation
- Rule-based transformations (entity swaps, number changes)
- LLM-generated morphological errors
- Hybrid approaches combining multiple error types

### Quality Control
- Similarity-based filtering to ensure diversity
- Grammar verification to maintain fluency
- Transformation success validation
- Configurable error density control

### Scalability
- Multi-threaded processing support
- Batch processing capabilities
- Configurable transformation intensity
- Memory-efficient claim processing

## Configuration & Customization

The pipeline provides extensive configuration options through `config.py` and various workflow parameters:

### Core Configuration Parameters (`config.py`)

#### **Claim Decomposition Method**
```python
CLAIM_DECOMPOSITION_METHOD = ClaimDecompositionMethod.SENTENCES
```
- **Options**: `SENTENCES`, `FACTS`, `PARAGRAPH`
- **Usage**: `text_decomposition(text=summary, decomposition_method=CLAIM_DECOMPOSITION_METHOD)`
- **Purpose**: Controls how summaries are broken into claims for transformation

#### **Transformation Configuration**
```python
TRANSFORMATION_CONFIG = {
    "hebrew_llm_transformations": {
        CorrectnessErrorTypes.HEBREW_ROOT_PATTEN_CONFUSION: True,
        CorrectnessErrorTypes.MORPHOLOGICAL_CONNECTIVE_CONFUSION: True,
        # ... other Hebrew-specific errors
    },
    "general_llm_transformations": {
        CorrectnessErrorTypes.SENTENCE_NEGATION: True,
        CorrectnessErrorTypes.MEASURE_UNIT_SWAP: True,
        # ... other general errors
    },
    "ner_based_transformation": {
        CorrectnessErrorTypes.ENTITY_PERSON_SWAP: True,
        # ... other NER-based errors
    }
}
```
- **Purpose**: Enable/disable specific error transformation types
- **Usage**: Accessed via `get_enabled_transformations_as_lists()` and `get_enabled_transformations_as_dicts()`

#### **Similarity & Quality Control**
```python
MAXIMUM_ALLOWED_NEGATIVE_SUMMARIES_SIMILARITY_THRESHOLD = 0.93
```
- **Purpose**: Controls diversity filtering to prevent overly similar corrupted summaries
- **Usage**: `filter_diverse_summaries(golden=summary, cands=candidates, max_same=THRESHOLD)`

```python
PERCENTAGE_TRANSFORMATIONS_PERFORM_ON_NEGATIVE_SUMMARY = 95
```
- **Purpose**: Controls corruption intensity (percentage of claims to transform)
- **Usage**: `control_claims_after_transformation(claims, percentage_of_correctness_errors_transformation=PERCENTAGE)`

#### **Evidence Analysis Parameters** (`evidence_engine.py` / UI controls)
- `top_k` (default 5), `min_score` (default 0.7), `metric` (`sequence_matcher` | `jaccard` | optional `mini_lm`), `delta_min` (default 0.2)
- Configured via the Evidence Analysis page or directly in `evidence_engine.py` main.

### LLM Model Selection

#### **Claim Selection Model**
```python
hebrew_claim_results = choose_suitable_claim(
    claim_decomposition, 
    hebrew_transformation_types_list,
    hebrew_filtered_examples, 
    model="gemini"  # Configurable: "openai", "gemini", "claude"
)
```

#### **Transformation Generation Model**
```python
llm_result = summary_negative_augmentation(
    apply_transformation_types=[error_type],
    transformation_examples=hebrew_filtered_examples.get(error_type, []),
    suitable_claims_decomposition_list=suitable_claims_decomposition_list,
    unique_transformation_instructions="",
    model="gemini"  # Configurable: "openai", "gemini", "claude"
)
```

### Diversity Filtering Algorithm (`filter_diverse_summaries`)

The pipeline uses a sophisticated diversity filtering system to ensure corrupted summaries are sufficiently different:

#### **Algorithm Details:**
1. **Similarity Metric**: Combines unigram and bigram Jaccard similarity
   ```python
   def overlap(a: str, b: str) -> float:
       t1 = a.split()
       t2 = b.split()
       uni = jaccard(t1, t2)  # Jaccard: |A ∩ B| / |A ∪ B|
       bi = jaccard(ngram(t1, 2), ngram(t2, 2))  # Bigram Jaccard
       return max(uni, bi)  # Take maximum of unigram/bigram similarity
   ```

2. **Adaptive Threshold**: Base threshold increases with error complexity
   ```python
   max_same = max_same + 0.0013 * (len(error_types) ** 0.75)
   ```

3. **Dual Filtering**:
   - **vs. Golden**: Corrupted summary must be ≤ `max_same` similar to original
   - **vs. Kept**: Corrupted summary must be ≤ `max_same` similar to already-accepted summaries

4. **N-gram Analysis**:
   - **Unigrams**: Word-level overlap detection
   - **Bigrams**: Phrase-level similarity detection
   - **Selection**: Uses maximum of unigram/bigram similarity scores

#### **Jaccard Similarity Formula:**
```
Jaccard(A, B) = |A ∩ B| / |A ∪ B|
```
Where A and B are sets of tokens (unigrams) or token pairs (bigrams).

### Workflow Customization Points

- **Claim decomposition method**: Switch between sentence-based, fact-based, or paragraph-based segmentation
- **Error type selection**: Enable/disable specific transformation families
- **Model selection**: Choose different LLMs for claim selection vs transformation generation
- **Similarity thresholds**: Adjust diversity filtering sensitivity
- **Corruption intensity**: Control what percentage of claims get transformed
- **Multi-threading**: Configurable parallelization for batch processing

## Dependencies

### Core Libraries
- `transformers` - Hebrew NER processing
- `pydantic` - Structured data validation
- `langchain` - LLM integration framework
- `pandas` - Data manipulation
- `Levenshtein` - String similarity metrics
- `scikit-learn` - Splits (KFold/Train-Test) and metrics (F1/Precision/Recall)
- `streamlit` - Web UI framework

### Optional for Evidence Analysis (MiniLM)
- `sentence-transformers` (provides `all-MiniLM-L6-v2`)
- `torch`

### LLM APIs
- OpenAI GPT-4
- Google Gemini
- Anthropic Claude

### Hebrew NLP
- `dicta-il/dictabert-ner` - Hebrew named entity recognition
- Custom Hebrew text normalization utilities

This architecture provides a comprehensive framework for generating high-quality Hebrew summary corruptions that maintain linguistic naturalness while introducing specific morphological and semantic errors for model evaluation and training.