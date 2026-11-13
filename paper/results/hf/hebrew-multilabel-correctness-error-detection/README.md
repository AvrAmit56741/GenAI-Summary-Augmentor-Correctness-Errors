---
license: apache-2.0
base_model:
- dicta-il/neodictabert
---
------
license: apache-2.0
language:
- he
base_model:
- dicta-il/neodictabert
tags:
- multi-label-classification
- hebrew
- fact-checking
- error-detection
pipeline_tag: text-classification
library_name: transformers
---

# Hebrew Multi-Label Error Type Classifier (Setup 2: Claim-Level)

## Model Description

Fine-tuned [dicta-il/neodictabert](https://huggingface.co/dicta-il/neodictabert) for multi-label classification of factual error types in Hebrew text at the claim/sentence level.

**Task:** Identify specific error types in corrupted Hebrew claims  
**Language:** Hebrew  
**Max Context:** 3,072 tokens  
**Granularity:** Sentence-level (individual claims)

## Input/Output

### Input
- **Premise:** Original correct claim (Hebrew sentence)
- **Hypothesis:** Summary claim (Hebrew sentence)

### Output
Multi-label classification with probability scores for each error type. Threshold: 0.5 (probabilities ≥ 0.5 indicate presence of that error type).

## Supported Error Types

The model detects both **cross-language** and **Hebrew-specific** error types:

### Cross-Language Errors
- `entity_person_swap` - Swapping person names
- `entity_location_swap` - Swapping place names
- `entity_organization_swap` - Swapping organization names
- `entity_date_swap` - Changing dates
- `number_swap` - Altering numerical values
- `measure_unit_swap` - Changing units of measurement
- `sentence_negation` - Adding/removing negation


### Hebrew-Specific Errors
- `hebrew_root_pattern_confusion` - Morphological root/binyan errors
- `morphological_connective_confusion` - Logical connector errors (שכן ↔ כך ש)
- `verb_gender_swap` - Verb gender disagreement
- `noun_gender_swap` - Noun/pronoun gender errors
- `homographic_gender_errors` - Gender mismatches in homographic words (מורה, מרצה)
- `specificity_shift_errors` - Definite article changes (ה-)
- `construct_state_confusion` - Smichut (סמיכות) errors
- `subject_verb_person_mismatch` - Person agreement errors
- `verb_tense_swap` - Tense inconsistencies
- `evidentiality_source_attribution_collapse` - Attribution removal
- `impersonal_to_personal_verb_errors` - Voice changes (passive ↔ active)
- `verb_template_agency_reversal` - Agency reversals (הלביש ↔ התלבש)
- `directional_preposition_swap` - Directional errors (מ-X ל-Y ↔ מ-Y ל-X)