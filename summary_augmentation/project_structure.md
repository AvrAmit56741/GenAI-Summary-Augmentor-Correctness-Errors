# Hebrew Summary Augmentation Project Structure

## Overview
This project focuses on Hebrew text processing and summary augmentation, specifically designed to generate corrupted summaries with deliberate linguistic errors for training and evaluating Hebrew Natural Language Inference (NLI) models. The system introduces morphologically-aware errors that exploit unique characteristics of Hebrew grammar and morphology.

## Project Architecture
summary_augmentation/
├── config.py                                          # Configuration management
├── correctness_errors_augmentation_v2.py              # Main augmentation workflow (pipeline orchestration)
├── correctness_errors_llm_helper_functions.py        # LLM interaction utilities (OpenAI, Gemini, Claude)
├── correctness_errors_prompts.py                     # Prompt templates and Pydantic schemas
├── correctness_errors_examples.py                    # Hebrew linguistic error examples and transformation rules
├── correctness_errors_finder.py                      # Error detection and analysis tool
├── utils.py                                          # Core utilities (NER, filtering, decomposition, word validation)
├── logger_config.py                                  # Centralized logging configuration (LoggerFactory)
├── ui.py                                             # Streamlit main app (8 tabs: JSON, CSV, Utils, Analysis, etc.)
├── model_analytics.py                                # Model Analytics page backend (multilabel/binary analysis)
├── evidence_engine.py                                # Evidence extraction backend (similarity + ranking)
├── summary_generator_ollama.py                       # Open-source LLM summary generation via Ollama
├── summaries_generator.py                            # Summary generation utilities (closed LLM)
├── pages/
│   ├── 1_🔬_Model_Analytics.py                       # Streamlit page: Model evaluation (F1/P/R per error type)
│   ├── 2_🧾_Evidence_Analysis.py                     # Streamlit page: Claim-evidence validation
│   └── 3_📊_Mafat_Dataset_Analysis.py                # Streamlit page: JSONL browser and analysis
│   └── 4_Correctness_Evaluation_Metrics.py          # Streamlit page: Enter source+summary, compute ROUGE/BLEU/ChrF
├── llm_api_logs/                                     # LLM request/response logs (JSONL format)
├── word_validity_logs/                               # Word validation fix logs (dated files)
└── finetuning_scripts/
    ├── run_xnli.py                                   # Binary NLI classifier training (HuggingFace)
    └── run_xnli_multilabel_v2.py                     # Multi-label error type classifier (with class balancing)

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

### 7. **UI (Streamlit)** - Multi-Page Interactive Platform

#### **Main Application Hub** (`ui.py`)
**Purpose**: Centralized interface for dataset analysis, augmentation inspection, and quality control

**Main Tabs (8 tabs total)**:

1. **JSON Analyzer** 📋
   - Upload augmentation JSON files (full pipeline output)
   - Browse corrupted summaries with source articles and gold summaries
   - Filter by error types with multi-select
   - Visual diff highlighting (word-level changes with color coding)
   - **Export Feature**: Download claim-level transformations as JSON for selected error types (excludes source_article/gold_summary for lighter files)
   - Navigate between summaries with index selection
   - Display transformation details per claim

2. **CSV Reviewer** 📊
   - Load augmented datasets in CSV format
   - Filter rows by error types
   - View premise-hypothesis pairs with corruption details
   - Extract and display claim transformations from `transformation_list` column
   - Support for metadata (doc_id, len_group, source)

3. **Utils** 🛠️
   - Dataset splitting utilities (Train/Test splits)
   - K-Fold cross-validation setup
   - Stratified sampling for balanced error types
   - Configuration helpers

4. **Analysis** 🔍
   - Statistical analysis of CSV datasets
   - Distribution of error types across samples
   - Summary statistics and metadata exploration

5. **Error Analyzer** 🐛
   - Deep-dive into specific error families
   - Cross-reference claims by error type
   - Examine transformation patterns and quality

6. **API Logs** 📝
   - Read and parse LLM API logs from `llm_api_logs/` directory
   - Filter by model (OpenAI, Gemini, Claude)
   - Inspect request/response pairs
   - Debug LLM transformation quality
   - Search logs by timestamp, prompt content, or error type

7. **Dataset Analysis** 📈
   - Compare multiple datasets side-by-side
   - Aggregate error type distributions
   - Identify coverage gaps in error families
   - Generate summary reports

8. **Dataset Modification** ✏️
   - In-place editing of corrupted summaries
   - Quality rating system for transformations
   - Batch operations on selected samples
   - Export modified datasets

#### **Model Analytics Page** (`pages/1_🔬_Model_Analytics.py`)
**Purpose**: Comprehensive model evaluation for factual consistency detection

**Internal Tabs**:

1. **Multilabel Classifier Analysis**
   - **Input Format**: CSV with columns: `index, premise, hypothesis, label, len_group, doc_id, source, error_types, prob_X, pred_X, label_X` (one triplet per error type)
   - **Per-Error-Type Metrics**: F1, Precision, Recall, Avg Confidence on Correct, Avg Confidence on Wrong
   - **Overall Metrics**: Weighted F1, Weighted Precision, Weighted Recall (across all error types)
   - **Visualizations**: Confusion matrices, confidence distributions, error type heatmaps
   - **Export**: Download metrics as CSV/JSON

2. **Binary Classifier Analysis** (Placeholder for future implementation)

#### **Evidence Analysis Page** (`pages/2_🧾_Evidence_Analysis.py`)
**Purpose**: Validate claim transformations against source article evidence

**Features**:
- **Upload**: Augmentation JSON with source articles and transformed claims
- **Modes**:
  - Specific Claim: Analyze one claim transformation with candidate claim preview
  - One Summary: Analyze all claims in a single summary
  - All Summaries: Batch analysis across the entire dataset
- **Parameters**:
  - `top_k`: Number of most similar sentences to retrieve (default 5)
  - `min_score`: Similarity threshold for evidence (default 0.7)
  - `metric`: Similarity function
    - `sequence_matcher` (SequenceMatcher diff-based)
    - `jaccard` (Set-based token overlap)
    - `mini_lm` (Sentence-transformers semantic similarity with `all-MiniLM-L6-v2`)
  - `delta_min`: Minimum difference between original and transformed claim support (default 0.2)
    - **Delta = (Original Claim Evidence Score) - (Transformed Claim Evidence Score)**
    - High delta → Strong evidence that transformation is factually inconsistent
    - Used to filter out weak/stylistic transformations
- **Output**:
  - Top-K evidence sentences ranked by similarity
  - Mean evidence score for original vs. transformed claim
  - Delta flagging (True if delta ≥ threshold)
  - JSON export of all results
- **Backend**: `evidence_engine.py` with sentence splitting, similarity ranking, threshold filtering

#### **Mafat Dataset Analysis Page** (`pages/3_📊_Mafat_Dataset_Analysis.py`)
**Purpose**: Specialized JSONL browser for summary generation datasets

**Features**:
- **Format Support**: 
  - `summary_generator_ollama` output (with `generative_summaries`, `model`, `prompt` fields)
  - `summarization-7-heb_2.jsonl` (standard MAFAT format)
- **Filters**:
  - Source (Israel_Hayom, etc.)
  - Annotator (kfir, etc.)
  - Type (silver, gold)
  - User
  - Model (llama3.1, gemma3n, etc.)
  - Text search (in text_raw, summary, metadata)
  - Word count ranges (min/max)
- **Tabs**:
  - **Overview**: Dataset statistics, sample counts, format detection
  - **Browse**: Navigate records with RTL Hebrew support, pagination
  - **Compare**: Side-by-side comparison of summaries from different models
  - **Export**: Filtered subset export as JSONL
- **Robust Parsing**: Handles malformed JSONL, detects format automatically, caches files

**Navigation Between Pages**:
- Sidebar info messages guide users to separate pages
- Streamlit multipage architecture with automatic page discovery

#### **Correctness Evaluation Metrics Page** (`pages/4_Correctness_Evaluation_Metrics.py`)
**Purpose**: Quick, model-agnostic factuality approximation by comparing a user-provided summary to its source using standard text-overlap metrics.

**Inputs**:
- Source document (reference)
- Candidate summary (hypothesis)

**Supported Metrics**:
- **ROUGE-N** (ROUGE-1, ROUGE-2) and **ROUGE-L**: Recall-oriented n-gram and LCS overlaps
- **BLEU**: n-gram precision with brevity penalty (originally for MT)
- **ChrF / ChrF++**: Character n-gram F-score (robust for morphologically rich languages like Hebrew)

**Usage**:
- Paste or upload source and summary, select metrics, compute scores
- Display per-metric scores with brief explanations and downloadable JSON of results

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

### Hebrew-Specific Error Details (With Examples)

#### **1. `hebrew_root_pattern_confusion` (בלבול בין שורש ומשקל)**
**Definition**: Morphologically-aware confusion exploiting Hebrew's root-pattern system. Replaces words with similar phonetic/morphological structure but different roots or binyanim, causing factual changes.

**Sub-types**:
- Same binyan/pattern, different roots (פ.ר.פ.ר vs פ.ח.ת in pi'el)
- Phonetic similarity (שורשים דומים פונטית: ר.ע.ש ↔ ר.ג.ש)
- Foreign word → similar Hebrew word (Performer → מפרפר)
- Same root, different binyan with semantic shift (פיעל ↔ התפעל)

**Examples**:
```
מקור: "ישראל התחילה בהרעשה רגע אחרי הפסקת האש"
שגוי: "ישראל התחילה להתרגש רגע אחרי הפסקת האש"
הסבר: שורשים קרובים פונטית - ר.ע.ש (military action) ↔ ר.ג.ש (emotion)

מקור: "החברה מפתחת מוצרים חדשניים"
שגוי: "החברה מתפתחת למוצרים חדשניים"
הסבר: אותו שורש פ.ת.ח, בניינים שונים - פיעל (develops/creates) ↔ התפעל (evolves)
```

---

#### **2. `morphological_connective_confusion` (בלבול מילות קישור מורפולוגיות)**
**Definition**: Uniquely Hebrew error exploiting morphological similarity between short connectives (2-4 letters) with opposite logical meanings. Confuses cause/effect, concession/contrast due to shared prefixes (ש-, כ-, מ-) or letter overlap.

**Key Confusion Pairs**:
- `שכן` (because) ↔ `כך ש` (so that) — shared ש+כ
- `למרות ש` (despite) ↔ `מכיוון ש` (because) — both מ-+word+ש-
- `כי` (because) ↔ `כך` (thus) — phonetic overlap

**Examples**:
```
מקור: "יצטרכו למצוא פתרונות יצירתיים, שכן משרד התיירות נאבק"
שגוי: "יצטרכו למצוא פתרונות יצירתיים, כך שמשרד התיירות נאבק"
הסבר: היפוך סיבה→תוצאה; דמיון מורפולוגי ש-+כ-

מקור: "למד בהצלחה למרות הקשיים הכלכליים"
שגוי: "למד בהצלחה מכיוון שהיו קשיים כלכליים"
הסבר: ויתור (despite) → סיבה (because); תבנית מ-+מילה+ש-
```

---

#### **3. `verb_gender_swap` (החלפת מין הפועל)**
**Definition**: Changes only the verb's gender (masculine ↔ feminine) without changing the subject, creating subject-verb gender disagreement.

**Examples**:
```
מקור: "המנהל הציג את התוכנית"
שגוי: "המנהל הציגה את התוכנית"
הסבר: Subject (המנהל) masculine, verb (הציגה) feminine → mismatch

מקור: "התלמידות כתבו עבודות מרשימות"
שגוי: "התלמידות כתבו עבודות מרשים"
הסבר: Noun (עבודות) feminine, adjective (מרשים) masculine
```

---

#### **4. `noun_gender_swap` (החלפת מין שם עצם)**
**Definition**: Changes the gender of nouns or pronouns without adjusting dependent verbs/adjectives, creating grammatical gender mismatches.

**Examples**:
```
מקור: "הן עבדו על הפרויקט"
שגוי: "הם עבדו על הפרויקט"
הסבר: Pronoun switched feminine plural → masculine plural

מקור: "המורה החדשה העבירה שיעור"
שגוי: "המורה החדש העבירה שיעור"
הסבר: Adjective (החדש) masculine, verb (העבירה) feminine → mixed gender
```

---

#### **5. `homographic_gender_errors` (טעויות מין במילים הומוגרפיות)**
**Definition**: Exploits Hebrew words written identically for both genders (מורה, מרצה, רופא, שופט). Changes contextual gender markers (verbs/possessives) to mismatch the actual person's gender.

**Target Words**: מורה, מרצה, שופט, רופא, מנהל, יועץ, עיתונאי, אמן

**Examples**:
```
מקור: "המרצה ד״ר רחל כהן הציגה את מחקרה"
שגוי: "המרצה ד״ר רחל כהן הציג את מחקרו"
הסבר: Name (רחל) clearly feminine, but verb/possessive switched to masculine

מקור: "השופט מירב לוי פסקה לטובת הנתבע"
שגוי: "השופט מירב לוי פסק לטובת הנתבע"
הסבר: Name (מירב) feminine, verb masculine → factual error about judge's identity
```

---

#### **6. `specificity_shift_errors` (שינוי רמת ספציפיות)**
**Definition**: Changes specificity level via definite article (ה-), proper names, or generic references, altering the factual scope or entity identification.

**Transformation Types**:
- Generic → Specific: Add ה-, proper nouns, identifying details
- Specific → Generic: Remove ה-, replace names with "מישהו", "אומרים", "ארגון כלשהו"

**Critical Constraint**: Only specificity changes; no alterations to dates, numbers, locations, or other factual details.

**Examples**:
```
מקור: "מומחים ממליצים לקרוא מאמרים מדעיים"
שגוי: "מומחים ממליצים לקרוא את המאמרים המדעיים"
הסבר: Generic articles → specific articles (implies previously mentioned)

מקור: "פרופ׳ כהן הציג את הממצאים"
שגוי: "מישהו הציג את הממצאים"
הסבר: Specific identity (פרופ׳ כהן) → anonymous (מישהו)

מקור: "אומרים שהרפורמה תיכנס לתוקף"
שגוי: "הפקידים אומרים שהרפורמה תיכנס לתוקף"
הסבר: Anonymous source → specific group (הפקידים)
```

**Non-Valid Example** (Rejected):
```
מקור: "הוא שתה וויסקי קלאסי עם ליקר דבש איכותי"
שגוי: "הוא שתה וויסקי קלאסי עם ליקר דבש"
הסבר: 'איכותי' is subjective adjective, not specificity marker → INVALID
```

---

#### **7. `verb_tense_swap` (החלפת זמן הפועל)**
**Definition**: Changes verb tense (past ↔ present ↔ future) while preserving person, gender, number, and binyan. Must create chronological inconsistency or contradict time markers.

**Examples**:
```
מקור: "הוא הולך הביתה אחרי יום עבודה"
שגוי: "הוא הלך הביתה אחרי יום עבודה"
הסבר: Present (הולך) → Past (הלך)

מקור: "היא קוטפת תפוזים בפרדס כל שבוע"
שגוי: "היא תקטוף תפוזים בפרדס כל שבוע"
הסבר: Present (קוטפת) → Future (תקטוף)
```

---

#### **8. Additional Hebrew-Specific Errors**

**`evidentiality_source_attribution_collapse`** (השמטת ייחוס):
- Removes attribution markers ("לפי X", "על פי מקורות"), presenting claims as objective facts

**`impersonal_to_personal_verb_errors`** (הפיכת פועל סביל לאישי):
- Changes impersonal/passive constructions to personal active ("הוחלט" → "הוא החליט")

**`verb_template_agency_reversal`** (היפוך סוכן-מושא):
- Swaps binyanim to reverse agency (active ↔ passive): "הלביש" (dressed someone) ↔ "התלבש" (dressed oneself)

**`directional_preposition_swap`** (החלפת מילת יחס כיוונית):
- Reverses directionality: "מ-X ל-Y" → "מ-Y ל-X"

**`construct_state_confusion`** (בלבול במצב סמיכות):
- Errors in construct state (smichut) noun chains, breaking possessive/attributive relationships

**`subject_verb_person_mismatch`** (אי-התאמת גוף בין נושא לפועל):
- Changes verb person (1st/2nd/3rd) to mismatch subject

These errors leverage Hebrew's morphological richness, compactness, and ambiguity to create factual inconsistencies that remain grammatically fluent.

These families are used across the pipeline for augmentation (negative transformations), correctness finding, and evaluation.

## Output Format: Augmentation JSON Structure

The pipeline produces a JSON file containing the original source article and multiple corrupted summaries with detailed transformation metadata. Each record follows this schema:

### Top-Level Structure
```json
{
  "source_article": "string",              // Original Hebrew source article
  "gold_summary": "string",                // Original (correct) summary
  "correctness_corrupted_summaries": [     // Array of corrupted variants
    { /* Corrupted Summary Object */ }
  ]
}
```

### Corrupted Summary Object
Each corrupted summary contains:

```json
{
  "corrupted_summary": "string",           // Fully reconstructed corrupted summary text
  "transformations_performed": integer,    // Count of successful transformations applied
  "transformation_list": [                 // Detailed claim-level transformation records
    {
      "original_claim": "string",          // Original claim/sentence from decomposition
      "transformed_claim": "string",       // Corrupted version of the claim
      "applied_transformations": [         // Array of error transformations applied
        {
          "type": "string",                // Error type enum (e.g., "entity_person_swap")
          "diffs": [                       // Token-level diff information
            {
              "tag": "string",             // "insert", "delete", "replace", or "equal"
              "orig_start": integer,       // Character start index in original_claim
              "orig_end": integer,         // Character end index in original_claim
              "trans_start": integer,      // Character start index in transformed_claim
              "trans_end": integer,        // Character end index in transformed_claim
              "orig_text": "string",       // Original text span
              "trans_text": "string"       // Transformed text span
            }
          ]
        }
      ]
    }
  ]
}
```

### Field Descriptions

#### **`source_article`** (string)
- The original Hebrew source document from which the gold summary was generated
- Used as the premise in NLI tasks

#### **`gold_summary`** (string, optional)
- The original (correct) summary before corruption
- May be omitted in some output variants to reduce file size

#### **`corrupted_summary`** (string)
- The final recomposed summary after applying all transformations
- Serves as the hypothesis in NLI contradiction pairs

#### **`transformations_performed`** (integer)
- Total count of error transformations successfully applied
- Controlled by `PERCENTAGE_TRANSFORMATIONS_PERFORM_ON_NEGATIVE_SUMMARY` config

#### **`transformation_list`** (array)
- Detailed record of each claim transformation
- Maps original claims to their corrupted versions
- Includes granular diff information for interpretability

#### **`applied_transformations`** (array)
- Lists all error types applied to a single claim
- Multiple error types can be applied to the same claim (e.g., `entity_person_swap` + `verb_gender_swap`)

#### **`type`** (string)
- Error family from `CorrectnessErrorTypes` enum
- Examples: `entity_person_swap`, `hebrew_root_pattern_confusion`, `sentence_negation`, `construct_state_confusion`

#### **`diffs`** (array)
- Character-level diff operations (similar to Python's `difflib.SequenceMatcher`)
- Enables precise visualization of changes in the UI
- **Tags**:
  - `"equal"`: Unchanged text span
  - `"delete"`: Text removed from original
  - `"insert"`: Text added in transformation
  - `"replace"`: Text substituted

### Example Record (Abbreviated)

```json
{
  "source_article": "מאוהל ועד תרמיל. בסוף שבוע קיצי אין כמו לצאת לקמפינג...",
  "gold_summary": "הכתבה עוסקת בקמפינג למשפחות, וכוללת המלצות לציוד חיוני...",
  "correctness_corrupted_summaries": [
    {
      "corrupted_summary": "הכתבה עוסקת בקמפינג למשפחות, וכוללת המלצות לציוד חיוני, טיפים לבחירה נכונה...",
      "transformations_performed": 10,
      "transformation_list": [
        {
          "original_claim": "כדאי לבחור ציוד נוח וקומפקטי, כגון מזרנים מתקפלים, צידנית רכה עם תחתית קשיחה וכיסאות נוחים.",
          "transformed_claim": "כדאי לבחור ציוד נוח, כגון מזרנים מתקפלים, תחתית קשיחה וכיסאות נוחים.",
          "applied_transformations": [
            {
              "type": "construct_state_confusion",
              "diffs": [
                {
                  "tag": "delete",
                  "orig_start": 19,
                  "orig_end": 28,
                  "trans_start": 19,
                  "trans_end": 19,
                  "orig_text": " וקומפקטי",
                  "trans_text": ""
                },
                {
                  "tag": "delete",
                  "orig_start": 50,
                  "orig_end": 64,
                  "trans_start": 41,
                  "trans_end": 41,
                  "orig_text": " צידנית רכה עם",
                  "trans_text": ""
                }
              ]
            }
          ]
        },
        {
          "original_claim": "מומלץ לרכוש תאורת לד, שהיא חסכונית ובטוחה לשימוש בסביבת ילדים.",
          "transformed_claim": "מומלץ לרכוש תאורת לד, שהיא חסכונית לשימוש.",
          "applied_transformations": [
            {
              "type": "construct_state_confusion",
              "diffs": [
                {
                  "tag": "delete",
                  "orig_start": 35,
                  "orig_end": 42,
                  "trans_start": 35,
                  "trans_end": 35,
                  "orig_text": "ובטוחה ",
                  "trans_text": ""
                },
                {
                  "tag": "delete",
                  "orig_start": 48,
                  "orig_end": 61,
                  "trans_start": 41,
                  "trans_end": 41,
                  "orig_text": " בסביבת ילדים",
                  "trans_text": ""
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}
```

### Usage in Downstream Tasks

**For Binary NLI Training** (`run_xnli.py`):
```python
# Convert JSON to CSV format
premise = record["source_article"]
hypothesis_positive = record.get("gold_summary", "")  # Label: 1 (entailment)
for corrupted in record["correctness_corrupted_summaries"]:
    hypothesis_negative = corrupted["corrupted_summary"]  # Label: 0 (contradiction)
```

**For Multi-Label Error Classification** (`run_xnli_multilabel_v2.py`):
```python
premise = record["source_article"]
hypothesis = corrupted["corrupted_summary"]
error_types = ",".join(set(
    trans["type"] 
    for claim in corrupted["transformation_list"]
    for trans in claim["applied_transformations"]
))
```

**For UI Visualization** (`ui.py`, JSON Analyzer tab):
- `diffs` array enables word-level highlighting (green=added, red=removed)
- `transformation_list` displayed in expandable claim-by-claim view
- Filter by `type` field to focus on specific error families

### File Naming Convention
```
augmentation_output_{timestamp}.json
augmentation_output_20250104_143022.json
```

Files are typically stored in the workspace root or a dedicated `outputs/` directory.

---

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
Output: Corrupted Summaries (JSON format detailed above)

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

---

## Downstream Tasks: Model Training and Evaluation

### Overview

The augmented datasets generated by this pipeline serve as training data for two primary Natural Language Inference (NLI) tasks focused on Hebrew factual consistency detection:

1. **Binary NLI Classification**: Determines if a summary is factually consistent (entailment) or inconsistent (contradiction) with the source article
2. **Multi-Label Error Type Classification**: Identifies which specific error types are present in an inconsistent summary

Both tasks use a **premise-hypothesis** format where:
- **Premise**: Source article (document)
- **Hypothesis**: Generated summary (gold or corrupted)
- **Label**: Binary (0=contradiction, 1=entailment) or Multi-hot vector of error types

---

### Task 1: Binary NLI Classification

**Script**: `run_xnli.py` (adapted from HuggingFace's XNLI example)

#### Purpose
Train transformer models to detect whether a Hebrew summary is factually consistent with its source article, treating factual consistency as a binary entailment problem.

#### Architecture
```python
AutoModelForSequenceClassification.from_pretrained(
    model_name_or_path,  # e.g., "xlm-roberta-base", "alephbert-base"
    num_labels=2,        # entailment (1) vs contradiction (0)
)
```

#### Data Format
**Input CSV/JSONL columns**:
- `premise`: Source article text
- `hypothesis`: Summary text (gold or corrupted)
- `label`: 0 (contradiction/corrupted) or 1 (entailment/gold)

#### Training Process
1. **Tokenization**: Pairs `(premise, hypothesis)` tokenized with `[CLS] premise [SEP] hypothesis [SEP]` format
2. **Sequence Length**: Default `max_seq_length=128`, with warnings for truncated examples
3. **Loss Function**: Cross-entropy loss for binary classification
4. **Metrics**: Accuracy, F1, Precision, Recall (via `evaluate.combine`)
5. **Optimizer**: AdamW (default HuggingFace Trainer settings)

#### Key Features
- **Multi-lingual Support**: Works with XLM-RoBERTa, mBERT, AlephBERT, DictaBERT
- **Dynamic Padding**: Optionally pads to max length or dynamically per batch
- **Checkpoint Resume**: Automatic detection and resumption from last checkpoint
- **Prediction Output**: Can output either labels or entailment probability scores (`get_predict_scores` flag)

#### Output
- **Training Metrics**: Saved to `{output_dir}/train_results.json`
- **Eval Metrics**: `{output_dir}/eval_results.json`
- **Predictions**: `{output_dir}/predictions.txt` (format: `index \t label` or `index \t probability`)

#### Example Command
```bash
python run_xnli.py \
  --model_name_or_path xlm-roberta-base \
  --train_file train.csv \
  --validation_file dev.csv \
  --test_file test.csv \
  --do_train --do_eval --do_predict \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir ./results/binary_nli \
  --overwrite_output_dir
```

---

### Task 2: Multi-Label Error Type Classification

**Script**: `run_xnli_multilabel_v2.py` (custom multi-label adaptation)

#### Purpose
Given a corrupted summary, identify **all** specific error types present (e.g., `entity_person_swap`, `verb_gender_swap`, `sentence_negation`). This is a **multi-label** problem where each summary can have 0 or more error tags.

#### Architecture
```python
AutoModelForSequenceClassification.from_pretrained(
    model_name_or_path,
    num_labels=len(error_types),         # e.g., 15-20 error types
    problem_type="multi_label_classification",
)
```

#### Data Format
**Input CSV/JSONL columns**:
- `premise`: Source article text
- `hypothesis`: Corrupted summary text
- `error_types`: Comma-separated error tags (e.g., `"entity_person_swap,verb_gender_swap"`)
- `label`: Ignored (replaced by multi-hot encoding)

**Encoded Format** (after preprocessing):
- `labels`: Multi-hot vector `[0.0, 1.0, 0.0, 1.0, ...]` (length = num error types)

#### Training Process

**1. Label Discovery**
```python
# Automatically gather all unique error types from train/val/test
label_list = sorted(all_unique_error_types)
label2id = {error: idx for idx, error in enumerate(label_list)}
```

**2. Multi-Hot Encoding**
```python
def encode_error_types(example):
    multi_hot = [0.0] * num_labels
    for tag in example["error_types"].split(","):
        if tag.strip() in label2id:
            multi_hot[label2id[tag.strip()]] = 1.0
    example["labels"] = multi_hot
    return example
```

**3. Class Imbalance Handling** (Critical Enhancement)

**Problem**: Some error types (e.g., `entity_person_swap`) are frequent, while others (e.g., `hebrew_root_pattern_confusion`) are rare, causing the model to ignore minority classes.

**Solution**: Two-pronged approach implemented in `v2`:

**a) Class-Balanced Loss (`BCEWithLogitsLoss` with `pos_weight`)**
```python
# Compute pos_weight for each error type
pos_counts = all_labels.sum(axis=0)  # per-class positive counts
neg_counts = len(all_labels) - pos_counts
pos_weight = neg_counts / pos_counts  # higher weight for rare classes

# Custom loss function
loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

**b) Stratified Oversampling (`WeightedRandomSampler`)**
```python
# Weight examples by rarity of their error types
example_weights = []
for label_vector in all_labels:
    positive_error_indices = np.where(label_vector == 1)[0]
    weight = np.mean(pos_weight[positive_error_indices])
    example_weights.append(weight)

sampler = WeightedRandomSampler(
    weights=example_weights,
    num_samples=len(example_weights),
    replacement=True  # oversample rare classes
)
```

**4. Custom Trainer**
```python
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").float()
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        loss = self.loss_fct(outputs.logits, labels)  # BCEWithLogitsLoss
        return (loss, outputs) if return_outputs else loss
    
    def get_train_dataloader(self):
        # Use custom sampler for stratified oversampling
        return DataLoader(
            self.train_dataset,
            sampler=self.sampler,
            batch_size=self.args.train_batch_size,
            collate_fn=self.data_collator
        )
```

#### Metrics
**Per-Label Metrics** (Micro-averaged):
```python
from sklearn.metrics import precision_recall_fscore_support

def compute_metrics(p: EvalPrediction):
    probs = torch.sigmoid(torch.tensor(p.predictions)).numpy()
    y_pred = (probs >= 0.5).astype(int)  # threshold at 0.5
    y_true = p.label_ids
    
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    return {"precision": prec, "recall": rec, "f1": f1}
```

**Note**: Micro-averaging treats each (example, label) pair independently, giving equal weight to all predictions.

#### Output
- **Training Metrics**: Per-epoch F1/Precision/Recall
- **Predictions**: `{output_dir}/predictions.txt` with per-label probabilities
  ```
  index   entity_person_swap   verb_gender_swap   sentence_negation   ...
  0       0.8234              0.1234             0.0123              ...
  1       0.0234              0.7823             0.9123              ...
  ```

#### Example Command
```bash
python run_xnli_multilabel_v2.py \
  --model_name_or_path alephbert-base \
  --train_file train.csv \
  --validation_file dev.csv \
  --test_file test.csv \
  --do_train --do_eval --do_predict \
  --max_seq_length 128 \
  --per_device_train_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 5 \
  --output_dir ./results/multilabel_errors \
  --save_strategy epoch \
  --evaluation_strategy epoch
```

---

### Model Recommendations

#### Primary Models Used in This Project

**For BERT-Based Classifiers** (Binary NLI & Multi-Label Error Type):

1. **LongHeRo** (`HeNLP/LongHeRo`)
   - **Architecture**: Hebrew RoBERTa with extended context window
   - **Context Length**: Up to 4096 tokens (vs standard 512)
   - **Use Case**: Ideal for long premise-hypothesis pairs (full articles + summaries)
   - **Advantage**: Handles Hebrew article-summary pairs without aggressive truncation
   - **Recommended for**: Both binary NLI and multi-label error classification

2. **NeoDictaBERT** (`dicta-il/neodictabert`)
   - **Architecture**: Modern Hebrew BERT variant (2023+)
   - **Training Data**: Large-scale Hebrew corpora with morphological annotations
   - **Strength**: Superior morphological awareness for Hebrew-specific errors
   - **Use Case**: Especially effective for detecting `verb_gender_swap`, `noun_gender_swap`, `hebrew_root_pattern_confusion`
   - **Recommended for**: Multi-label error type classification

**For LLM-Based Classifiers** (Alternative/Experimental):

3. **Dicta 2.0** (`dicta-il/dicta2.0` or similar)
   - **Architecture**: Large Language Model (likely 7B-13B parameters)
   - **Approach**: Few-shot or instruction-tuned classification
   - **Use Case**: Can handle nuanced reasoning about factual errors
   - **Trade-off**: Higher computational cost vs BERT models

4. **Gemma 3 12B** (`google/gemma-3-12b`)
   - **Architecture**: 12 billion parameter instruction-tuned LLM
   - **Multilingual**: Supports Hebrew with strong multilingual transfer
   - **Approach**: Prompt-based classification with chain-of-thought reasoning
   - **Use Case**: Experimental alternative for complex error detection

#### Model Selection Guidelines

**Use LongHeRo when**:
- Premise (article) exceeds 512 tokens
- You need full context without truncation
- Training on article-summary pairs with long premises

**Use NeoDictaBERT when**:
- Detecting Hebrew morphological errors (gender, root patterns, construct state)
- Multi-label error type classification
- You have shorter texts or can afford aggressive truncation

**Use LLM classifiers (Dicta 2.0, Gemma 3) when**:
- Experimenting with few-shot learning
- Need interpretable reasoning about errors
- Have sufficient compute resources (A100/H100 GPUs)

#### Comparison Table

| Model | Type | Params | Max Tokens | Hebrew-Specific | Best For |
|-------|------|--------|------------|-----------------|----------|
| **LongHeRo** | BERT | ~125M | 4096 | ✓ | Long contexts, both tasks |
| **NeoDictaBERT** | BERT | ~110M | 512 | ✓✓ | Morphological errors |
| **Dicta 2.0** | LLM | ~7-13B | 4096+ | ✓✓ | Few-shot, reasoning |
| **Gemma 3 12B** | LLM | 12B | 8192+ | ✓ | Multilingual, experimental |

#### Legacy/Baseline Models (for comparison)

- **AlephBERT** (`onlplab/alephbert-base`): Earlier Hebrew BERT, good baseline
- **DictaBERT v1** (`dicta-il/dictabert`): Predecessor to NeoDictaBERT
- **XLM-RoBERTa** (`xlm-roberta-base`): Multilingual baseline
- **mBERT** (`bert-base-multilingual-cased`): Cross-lingual comparison

---

### Training Best Practices

#### Hyperparameters
- **Learning Rate**: 2e-5 to 5e-5 (lower for large models)
- **Batch Size**: 16-32 (adjust based on GPU memory)
- **Epochs**: 3-5 for binary, 5-10 for multi-label
- **Warmup Ratio**: 0.1 (10% warmup steps)
- **Weight Decay**: 0.01 (L2 regularization)

#### Data Split Strategy
- **Train**: 70-80% of augmented dataset
- **Validation**: 10-15% (for hyperparameter tuning)
- **Test**: 10-15% (held-out, report final metrics)

**Important**: Use the UI's **Utils** tab for stratified splits that balance error type distributions.

#### Evaluation Protocol
1. **Binary NLI**:
   - Primary Metric: **F1-Score** (harmonic mean of precision/recall)
   - Secondary: Accuracy (if classes balanced)
   
2. **Multi-Label**:
   - Primary: **Micro-F1** (treats all labels equally)
   - Per-Error-Type: F1, Precision, Recall (use Model Analytics UI)
   - Confidence Analysis: Avg confidence on correct vs. wrong predictions

---

### Integration with UI

The **Model Analytics** page (`pages/1_🔬_Model_Analytics.py`) is specifically designed to analyze outputs from these finetuning scripts:

**Workflow**:
1. Train model using `run_xnli_multilabel_v2.py`
2. Generate predictions on test set (outputs `predictions.txt`)
3. Format predictions into UI-compatible CSV with `prob_X`, `pred_X`, `label_X` columns
4. Upload to Model Analytics → Multilabel Classifier tab
5. View per-error-type metrics, confidence distributions, and weighted aggregates

This closed-loop workflow enables iterative model improvement by identifying which error types are difficult to detect, informing further augmentation refinements.

---

This architecture provides a comprehensive framework for generating high-quality Hebrew summary corruptions that maintain linguistic naturalness while introducing specific morphological and semantic errors for model evaluation and training.