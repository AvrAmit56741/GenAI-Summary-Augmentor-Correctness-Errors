# Streamlit Application UI Structure

## Overview

**Hebrew NLI Analysis Hub** — a multi-page Streamlit application for managing, inspecting, and evaluating Hebrew NLP summarization datasets and factuality detection models.

The app is launched with:
```bash
streamlit run summary_augmentation/ui.py
```

---

## Directory Structure

```
summary_augmentation/
├── ui.py                                    ← Home page (entry point)
├── config.py                                ← Error types, model routing, WorkflowConfig
├── utils.py                                 ← Similarity metrics, sentence splitting
├── evidence_engine.py                       ← Evidence extraction logic
├── model_analytics.py                       ← Analytics helpers for model performance
├── logger_config.py                         ← Centralized logging (LoggerFactory)
└── pages/
    ├── 1_🔬_Model_Analytics.py             ← Sidebar page 1
    ├── 2_🧾_Evidence_Analysis.py           ← Sidebar page 2
    ├── 3_📊_Mafat_Dataset_Analysis.py      ← Sidebar page 3
    ├── 4_Correctness_Evaluation_Metrics.py ← Sidebar page 4
    └── metrics/                             ← Metrics calculation sub-package
        ├── __init__.py
        ├── ngram_metrics.py
        ├── semantic.py
        ├── summac.py
        ├── qa_facteval.py
        ├── compare_metrics_eval.py
        ├── metrics_explained.py
        └── misc/
            └── hallucinations_metrics.py
```

---

## Home Page — `ui.py`

The home page is a **10-tab application** covering all dataset management and inspection workflows.

### Tab Overview

| # | Tab Name | Key Capability |
|---|----------|---------------|
| 1 | JSON Analyzer | Browse and inspect augmented JSON datasets |
| 2 | CSV Reviewer | Review NLI CSV files with error type filtering |
| 3 | Utils | Combine, convert, and split datasets |
| 4 | Analysis | Statistical analysis of CSV premise/hypothesis pairs |
| 5 | Error Analyzer | Inspect JSON correctness finder reports |
| 6 | API Logs | Browse LLM API call logs (`.jsonl`) |
| 7 | Dataset Analysis | Compare two datasets side-by-side |
| 8 | Dataset Modification | SQL-like categorical filtering of CSV data |
| 9 | LLM Parse Data | Analyze `.arrow` HuggingFace training datasets |
| 10 | Corrupted Level | Analyze and reduce corruption intensity in JSON datasets |

---

### Tab 1 — JSON Analyzer

**Input:** Augmentation JSON file (`correctness_corrupted_summaries` structure)

**Features:**
- **Sidebar controls:** error type multiselect filter, document index navigator, raw JSON toggle
- **Corruption statistics:** actual transformation count vs stored `transformations_performed` value; inconsistency detection; distribution bar chart
- **Interactive error type buttons:** 3-column grid of clickable buttons to toggle filters; bar chart of error distribution
- **Claim-level analysis** (when error types are selected):
  - Side-by-side original vs transformed claim display (RTL Hebrew text)
  - Word-level diff visualization (dark red = deletion, dark green = insertion)
  - Transformation details (character positions or original→replacement)
  - Quality rating buttons: 👍 Good / 👎 Poor / ❓ Unclear (stored in session state)
  - Download Claims JSON button
- **Document browser:** source article, gold summary, each corrupted summary with diff
- **Export:** filtered JSON, quality ratings CSV

---

### Tab 2 — CSV Reviewer

**Input:** CSV with columns `premise`, `hypothesis`, `label`, optionally `error_types`, `transformation_list`, `doc_id`, `source`, `len_group`

**Features:**
- **Sidebar controls:** error type multiselect filter, clear filters button
- Data preview table with filter applied
- **Interactive error type buttons** (same pattern as JSON tab)
- **Claim-level analysis** (when `transformation_list` column present): diff view per claim, quality ratings
- **Document-level comparison:** groups rows by `doc_id`; identifies gold summary (label=`entailment`) vs corrupted; side-by-side diff
- **Custom text comparison:** free-text paste area for arbitrary diff visualization
- **Row inspector:** RTL premise/hypothesis display, error type badges, claim-level diffs
- **Statistics:** label distribution, error type distribution, source distribution, length group distribution
- **Text length analysis:** premise and hypothesis word count stats and histograms
- **Export:** filtered CSV, quality ratings CSV

---

### Tab 3 — Utils

**Four independent sub-sections:**

#### Combine CSV Files
- Upload multiple CSV files
- Modes: Default (all rows) or Custom Sample Size (per-file or per-label limits)
- Per-label sampling: set separate entailment / contradiction counts per file
- Debug info: per-file contribution counts, duplicate detection, label distribution
- Download: ZIP containing `combined.csv` + `combination_summary.txt`

#### Combine JSON Files
- Upload multiple JSON files with optional per-file sample limits
- Error-tolerant (skips invalid files, reports errors)
- Download: ZIP containing `combined.json` + `combination_summary.txt`

#### Convert JSON → CSV (Claim-Level)
Extracts individual claim transformations from augmentation JSON into NLI training rows.

| Mode | Output |
|------|--------|
| **Balance** | N contradictions + configurable ratio of entailments (e.g. N=1 → 50/50) |
| **Contradiction-only** | Only corrupted claims, 100% contradiction label |

Output columns: `premise`, `hypothesis`, `error_types`, `label`, `doc_id`, `original_claim`

#### Convert JSON ↔ CSV (Summary-Level)
Bidirectional conversion between augmentation JSON and NLI CSV at summary level.
- `JSON → CSV`: one row per summary (gold = entailment, corrupted = contradiction)
- `CSV → JSON`: groups by `doc_id`, reconstructs nested JSON structure

#### Split Data
Upload a CSV and split it with stratification on the `label` column.

| Mode | Description |
|------|-------------|
| **K-Fold** | `StratifiedKFold` with configurable K; per-fold quality analysis table; error type mapping table with counts and ratios; ZIP download |
| **Naive** | Train / Validation / Test split with configurable sliders; same quality analysis; ZIP download |

---

### Tab 4 — Analysis

**Input:** CSV file

**Features:**
- Unique counts: premises, hypotheses, premise-hypothesis pairs, exact matches
- Length statistics: mean/min/max for premise and hypothesis
- Sample viewer by category: Unique Premises, Unique Hypotheses, Unique Pairs, Exact Matches, Non-Matches

---

### Tab 5 — Error Analyzer

**Input:** JSON "finder" report (output from `correctness_errors_finder.py`)

**Features:**
- Data preview table
- Error name distribution bar chart
- Source distribution bar chart
- Single-record inspector showing: `severity`, `doc_id`, `premise`, `ai_summary`, `generative_summary`, `correctness_error_name`, `correctness_error_reasoning`, `mrl_unique_hebrew_features`, `why_its_factual_error`, `evidence_from_summary`, `evidence_from_source_article`

---

### Tab 6 — API Logs

**Input:** `.jsonl` LLM API log file

**Log format parsed:**
```
timestamp - logger_name - thread_name - LEVEL - REQUEST: {...}
timestamp - logger_name - thread_name - LEVEL - RESPONSE: {...}
timestamp - logger_name - thread_name - LEVEL - ERROR: {...}
```

**Features:**
- Statistics: total entries, error count, models used
- Log type distribution bar chart, model usage bar chart
- Log entry browser with filters (by type, errors-only)
- **Format-aware entry display:**
  - `REQUEST`: system message, user message, function/model info
  - `RESPONSE`: response content, token usage (prompt/completion/total), finish reason
  - `ERROR`: error type, function, message
  - `LOG` / `RAW`: plain message display
- Navigation buttons: First / Previous / Next
- Error analysis section with recent error summary
- **Export:** summary JSON, filtered logs JSON

---

### Tab 7 — Dataset Analysis

**Input:** 1 or 2 datasets (JSON or CSV)

**Single dataset features:**
- Error type distribution
- Label distribution (entailment / contradiction)
- Source distribution
- Text length statistics and histogram

**Comparison features (2 datasets):**
- Shared `doc_id` count, overlap percentage
- Expandable list of exact shared document IDs
- Side-by-side metric comparison
- Export: JSON comparison report

---

### Tab 8 — Dataset Modification

**Input:** CSV file

**Features:**
- Auto-detects categorical columns
- Filter mode: **Filter Out** (remove matching rows) or **Filter In** (keep only matching rows)
- Multi-column selection with AND logic between columns
- OR logic within each column's values
- Value options shown with counts (e.g. `entailment (1200)`)
- Filter preview: shows exact impact before applying
- Before/after comparison table per column
- **Export:** filtered CSV with timestamp in filename

**Filter logic example:**
```
(len_group IN ('b1', 'b2')) AND (source IN ('news', 'wiki'))
```

---

### Tab 9 — LLM Parse Data

**Input:** `.arrow` HuggingFace `Dataset` file

**Features:**
- Dataset overview: total samples, columns
- **Prompt format detection** by substring matching:
  - `### Instruction:` → Alpaca
  - `<start_of_turn>` → Gemma
  - `<|im_start|>` → ChatML
- Counts null/empty samples with warnings
- Format distribution bar chart
- **Sample browser:** format-aware display; Alpaca shows structured instruction/input/output fields; Gemma/ChatML shows raw formatted text
- Text length statistics: avg/min/max token count, valid vs null count, histogram
- **Export:** first 10 samples as JSON preview

---

### Tab 10 — Corrupted Level

**Input:** Augmentation JSON file

**Features:**
- Corruption statistics: actual count (from `transformation_list`) vs stored `transformations_performed`; inconsistency detection (⚠️ vs ✓)
- Distribution charts: absolute corruption count and percentage
- Detailed table per summary with doc/summary index, counts, percentage
- Sample reviewer: source article, gold summary, corrupted summary with diff

**Corruption Level Reduction (control panel):**

| Mode | Description |
|------|-------------|
| **Single** | Apply one target % uniformly to N summaries |
| **Plan** | Define N parts, each with its own sample count, target %, and strategy |

**Selection strategies:**

| Strategy | Logic |
|----------|-------|
| `random` | Random subset of transformations kept |
| `max_transformation` | Keep claims with the most error types per claim |
| `hebrew_focus` | Prioritize Hebrew-specific linguistic error types (17 types) |

After reduction:
- Download: ZIP containing `reduced_*.json` + `metadata.txt`
- `metadata.txt` includes: configuration, avg/min/max corruption stats, error type distribution breakdown
- Post-reduction statistics table with sample reviewer

---

## Shared Helper Functions (defined in `ui.py`)

| Function | Purpose |
|----------|---------|
| `highlight_diff(old, new)` | Word-level SequenceMatcher diff → HTML spans (dark red deletions, dark green insertions) |
| `parse_error_types(s)` | Split comma-separated error type string into list |
| `get_all_error_types(df)` | Extract sorted unique error types from a CSV DataFrame |
| `filter_by_error_types(df, types)` | Filter DataFrame rows by selected error types |
| `extract_claim_transformations(str)` | Parse `transformation_list` JSON → list of claim dicts |
| `get_claims_by_error_type(df, type)` | Get all claim transformations of a given type from CSV |
| `count_claims_sentences(text)` | Count sentences by splitting on `.` `!` `?` |
| `reduce_corruption_level(summary, %, strategy)` | Revert selected corrupted claims back to originals |
| `generate_metadata_text(data, mode, cfg)` | Produce text metadata report for exported reduced datasets |
| `get_claims_by_error_type_json(data, type)` | Extract claim transformations by type from nested JSON |
| `get_all_error_types_json(data)` | Extract unique error types from nested JSON structure |
| `filter_json_by_error_types(data, types)` | Filter JSON docs/summaries by error types |
| `analyze_dataset_structure(data, fmt, name)` | Generic JSON/CSV analyzer for Dataset Analysis tab |
| `find_shared_doc_ids(a1, a2)` | Compute intersection/difference/overlap of two `doc_id` sets |
| `display_dataset_analysis(analysis, prefix)` | Render error/label/source/length charts for one dataset |

---

## Session State Reference

| Key | Type | Owner | Purpose |
|-----|------|-------|---------|
| `selected_error_types` | `list[str]` | CSV tab | Active error type filter |
| `quality_ratings` | `dict` | CSV tab | Per-claim quality ratings (Good/Poor/Unclear) |
| `json_selected_error_types` | `list[str]` | JSON tab | Active error type filter |
| `json_quality_ratings` | `dict` | JSON tab | Per-claim quality ratings |
| `nav_entry_idx` | `int` | API Logs tab | Current log entry position |
| `kfold_split_data` | `dict[str, DataFrame]` | Utils/K-Fold | Split DataFrames for ZIP download |
| `kfold_k_value` | `int` | Utils/K-Fold | K value used in last K-Fold run |
| `naive_split_data` | `dict[str, DataFrame]` | Utils/Naive | Train/Val/Test DataFrames for ZIP download |
| `filter_config` | `dict` | Modification tab | Persists column/value selections across reruns |
| `filtered_data` | `DataFrame` | Modification tab | Result of applied filter |
| `modified_data` | `list` | Corruption Level | JSON dataset after reduction |
| `reduction_applied` | `bool` | Corruption Level | Whether a reduction has been run |
| `processed_count` | `int` | Corruption Level | Number of summaries processed |
| `applied_reduction_mode` | `str` | Corruption Level | `'single'` or `'plan'` |
| `config_params` | `dict` | Corruption Level | Reduction config for metadata generation |

---

## Sidebar Pages

Each page in `pages/` is automatically shown in the Streamlit sidebar.

### Page 1 — Model Analytics (`1_🔬_Model_Analytics.py`)

**Two tabs:**
- **Multilabel Classifier:** Upload CSV with `prob_X`, `pred_X`, `label_X` columns → per-error-type F1, Precision, Recall, Accuracy, confusion matrix
- **Binary Classifier:** Binary NLI classification performance

Backed by `model_analytics.py`.

---

### Page 2 — Evidence Analysis (`2_🧾_Evidence_Analysis.py`)

**Two tabs:**
- **Evidence Analysis:** Upload augmentation JSON → extract sentences that support or contradict each claim using similarity metrics
  - Similarity backends: `sequence_matcher`, `jaccard`, `mini_lm` (sentence-transformers)
  - Modes: Specific claim, One summary, All summaries
  - Parameters: Top-K, similarity threshold, delta filtering
- **Model Inference API:** Send inference requests to the FastAPI server (`inference_api_server/`)

Backed by `evidence_engine.py` and `utils.py`.

---

### Page 3 — Mafat Dataset Analysis (`3_📊_Mafat_Dataset_Analysis.py`)

**Input:** JSONL files

**Features:**
- Line-by-line JSONL parsing with source file and line number tracking
- Auto-detects format: original (`text_raw`/`summary`) vs Ollama-enhanced (`generative_summaries`)
- DataFrame display and basic statistics

---

### Page 4 — Correctness Evaluation Metrics (`4_Correctness_Evaluation_Metrics.py`)

**Seven tabs:**

| Tab | Metric | Category |
|-----|--------|----------|
| Single Evaluation | ROUGE, BLEU, ChrF, BERTScore | Overlap + Semantic |
| Batch Evaluation | Same metrics over many documents | Batch |
| Classifier Predictions | Model output analysis | Classification |
| Hallucination Metrics | ENFS% (entity-level hallucination) | Entity coverage |
| Compare Metrics | Kendall Tau, Point-Biserial correlation | Statistical |
| QAFactEval | QA-based factual consistency | QA pipeline |
| SummaC | NLI-based factual consistency | NLI |

---

## Metrics Sub-package (`pages/metrics/`)

| Module | Contents |
|--------|---------|
| `ngram_metrics.py` | `RougeCalculator`, `BleuCalculator`, `ChrfCalculator` — character n-grams handle Hebrew morphology |
| `semantic.py` | `BertScoreCalculator` — contextual embedding similarity via Hebrew BERT |
| `summac.py` | `SummacZS` (zero-shot col-max→mean), `SummacConv` (trained 1D conv over NLI histogram) |
| `qa_facteval.py` | QA pipeline: NP extraction → question generation → answer comparison; HuggingFace or OpenAI backends |
| `compare_metrics_eval.py` | `KendallTauCalculator`, `PointBiserialCalculator`, `ComparisonResult` |
| `metrics_explained.py` | Markdown strings for UI expanders; includes Hebrew interpretation notes |
| `misc/hallucinations_metrics.py` | ENFS% = entities in summary not in source / total entities; supports `dictabert-ner` and `heBERT_NER` |

---

## Data Flow

```
ui.py (home page)
│
│  Session state: selected_error_types, quality_ratings, filter_config, ...
│
├── Tab 1 (JSON Analyzer)
│     upload JSON → filter by error type → browse docs → rate claim quality → export
│
├── Tab 2 (CSV Reviewer)
│     upload CSV → filter by error type → inspect rows → compare summaries → export
│
├── Tab 3 (Utils)
│     combine CSVs/JSONs → convert JSON↔CSV → split train/val/test or K-Fold → download ZIP
│
├── Tab 6 (API Logs)
│     upload .jsonl → parse REQUEST/RESPONSE/ERROR → browse/filter → export
│
├── Tab 10 (Corrupted Level)
│     upload JSON → analyze corruption stats → reduce corruption (single/plan) → download ZIP
│
pages/
│
├── Page 1 (Model Analytics)
│     upload predictions CSV → per-error-type classifier metrics
│
├── Page 2 (Evidence Analysis)
│     upload JSON → extract supporting evidence sentences (similarity-ranked)
│
├── Page 3 (Mafat Dataset)
│     upload JSONL → detect format → DataFrame display
│
└── Page 4 (Correctness Metrics)
      source + summary text → ROUGE/BLEU/ChrF/BERTScore/SummaC/QAFactEval/ENFS% → statistical comparison
```

---

## Known Issues

| Issue | Location | Notes |
|-------|----------|-------|
| `st.experimental_rerun()` is deprecated | Multiple tabs | Replace with `st.rerun()` |
| `corrupted` variable used outside its `if` block | `ui.py` line 831 | Can raise `NameError` if no data loaded |
| `df` referenced in Dataset Modification tab | `ui.py` tab 8 | Falls back to `if 'df' in locals()` — fragile |
| Imports (`zipfile`, `io`, `datetime`, `random`) inside tab blocks | `ui.py` tabs 3, 10 | Should be top-level imports |
