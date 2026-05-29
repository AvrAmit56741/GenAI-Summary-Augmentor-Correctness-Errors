# Models Documentation

## Overview
This document describes the models trained for Hebrew factuality checking and error detection, including detailed architecture analysis derived from the fine-tuning scripts.

---

## 1. Binary NLI Classifier

### Architecture Diagram

```
INPUT (up to 4096 tokens)
┌─────────────────────────────────────────────────────────┐
│  [CLS]  article sentence₁ ... sentenceₙ  [SEP]          │
│         summary sentence₁ ... sentenceₘ  [SEP]          │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│              DictaBERT Encoder                           │
│         (dicta-il/neodictabert)                          │
│                                                          │
│   Layer 1  ──►  Layer 2  ──►  ...  ──►  Layer 12        │
│                                                          │
│   Hidden size: 768   |   12 attention heads              │
└─────────────────────────────────────────────────────────┘
         │                       │
    [CLS] token             other tokens
    hidden state             (discarded)
    shape: [B, 768]
         │
         ▼
   Dropout(p=hidden_dropout_prob)
         │
         ▼
   Linear(768 → 2)
         │
         ▼
   logits  [B, 2]
         │
    ┌────┴────┐
  Training  Inference
    │            │
CrossEntropy   Softmax → Argmax
   Loss          │
              class index
              0 = consistent
              1 = inconsistent
```

### Base Model
- **Architecture**: `dicta-il/neodictabert` — a Hebrew BERT-style encoder, 12 transformer layers, hidden size 768
- **Loaded via**: `AutoModelForSequenceClassification.from_pretrained(..., num_labels=2)`

### Classification Head
A **single linear layer** added on top of the BERT encoder, operating on the `[CLS]` token representation:

```
[CLS] token hidden state   →   Dropout(p=hidden_dropout_prob)   →   Linear(768 → 2)   →   logits
```

- **Input to head**: `[CLS]` representation from last transformer layer — shape `[batch, 768]`
- **Output of head**: raw logits — shape `[batch, 2]`
- **No intermediate layers**: single linear projection, no activation before logits
- **One head only**: no multi-head setup

From the training script:
```python
model = AutoModelForSequenceClassification.from_pretrained(
    model_args.model_name_or_path,
    config=config,       # config.num_labels = 2
    ...
)
```

### Input / Output
- **Input**: tokenized `[CLS] premise [SEP] hypothesis [SEP]` — max 4096 tokens
- **Output (training)**: raw logits `[batch, 2]`
- **Output (inference)**: `softmax(logits)` → argmax → predicted class index

From inference script:
```python
probs = torch.softmax(torch.tensor(all_logits), dim=1).numpy()
y_pred = np.argmax(probs, axis=1)
```

### Fine-tuning Scope
**Whole model fine-tuned** — both the DictaBERT encoder backbone and the classification head. No freezing of any layers.

### Loss Function
**Default HuggingFace cross-entropy loss** (no custom `compute_loss` override):
```
CrossEntropyLoss(logits, labels)   →   scalar loss
```

The standard HF `Trainer` applies cross-entropy when `problem_type` is single-label.

#### Cross-Entropy Loss — Worked Example

**Scenario**: classify whether a Hebrew summary sentence is factually consistent with the article.

```
Input pair:
  premise  (article):  "הדוח פורסם ב-2021 על ידי משרד הבריאות"
  hypothesis (summary): "הדוח פורסם ב-2019 על ידי משרד הבריאות"
  true label: 1  (inconsistent — year is wrong)

Step 1 — model outputs raw logits (before any activation):
  logits = [-1.2,  2.4]
             │      │
             │      └── logit for class 1 (inconsistent)
             └────────── logit for class 0 (consistent)

Step 2 — softmax turns logits into probabilities (must sum to 1.0):
  softmax([-1.2, 2.4]):
    exp(-1.2) = 0.30
    exp( 2.4) = 11.02
    sum        = 11.32

    P(class=0) = 0.30 / 11.32 = 0.027   ← 2.7% chance "consistent"
    P(class=1) = 11.02 / 11.32 = 0.973  ← 97.3% chance "inconsistent"

Step 3 — cross-entropy picks the log-prob of the TRUE class (class 1):
  loss = -log(P(true class)) = -log(0.973) = 0.027

  ← small loss → model was confident and correct ✓

Bad prediction example (model confused):
  logits = [1.8, -0.5]     ← model thinks "consistent"
  P(class=0) = 0.878
  P(class=1) = 0.122
  loss = -log(0.122) = 2.10   ← large loss → big gradient update
```

**Intuition**: cross-entropy = "how surprised are you by the true label?"
- Confident + correct → low loss → small weight update
- Confident + wrong → very high loss → large weight update
- Works here because the two classes (consistent / not) are **mutually exclusive** — a sentence is either factual or not, never both at once, so `softmax` (which forces probabilities to sum to 1) is the right activation.

### Training Configuration
| Parameter | Value |
|---|---|
| Max seq length | 4096 tokens |
| Batch size | 2 (effective 16 via gradient accumulation ×8) |
| Learning rate | 2e-5 |
| Epochs | 4 |
| Best model criterion | `eval_f1` |
| Optimizer | AdamW (HF default) |

### Performance
- **Accuracy**: 96.78%
- **F1 Score**: 96.20%

---

## 2. Multi-Label Error Type Classifiers

All three setups use the **same base model and head architecture**. They differ only in what goes into `premise` and `hypothesis`.

### Shared Architecture Diagram (all 3 setups)

```
INPUT (varies per setup — see below)
┌─────────────────────────────────────────────────────────┐
│  [CLS]  premise tokens  [SEP]  hypothesis tokens  [SEP] │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│              DictaBERT Encoder                           │
│         (dicta-il/neodictabert)                          │
│                                                          │
│   Layer 1  ──►  Layer 2  ──►  ...  ──►  Layer 12        │
│                                                          │
│   Hidden size: 768   |   12 attention heads              │
└─────────────────────────────────────────────────────────┘
         │
    [CLS] token
    shape: [B, 768]
         │
         ▼
   Dropout(p=hidden_dropout_prob)
         │
         ▼
   Linear(768 → N)       ← N = number of error types (detected from data)
         │
         ▼
   logits  [B, N]         ← one logit per error type (NOT mutually exclusive)
         │
    ┌────┴────────────────────────────────────┐
  Training                                 Inference
    │                                          │
BCEWithLogitsLoss                          Sigmoid(logits)   [B, N]  ← per-label prob
+ class-balanced pos_weight                    │
                                           threshold @ 0.5
                                               │
                                           multi-hot vector  [B, N]
                                           e.g. [0,1,0,0,1,0,...]
                                           ← multiple errors can be 1
```

### WeightedTrainer (custom HF Trainer subclass)

```
WeightedRandomSampler           WeightedTrainer
┌────────────────┐              ┌─────────────────────────────┐
│ Per-example    │  overrides   │ compute_loss():             │
│ weight = mean  │──────────── ▶│   labels = inputs.pop()     │
│ pos_weight of  │  dataloader  │   logits = model(inputs)    │
│ its active     │              │   loss = BCEWithLogits(      │
│ error types    │              │       logits, labels,        │
└────────────────┘              │       pos_weight=...)        │
                                └─────────────────────────────┘
```

### Base Model
- **Architecture**: `dicta-il/neodictabert` — same 12-layer Hebrew BERT encoder, hidden size 768
- **Loaded via**: `AutoModelForSequenceClassification.from_pretrained(..., num_labels=N, problem_type="multi_label_classification")`

### Classification Head
Same structure as Binary — **one single linear layer** on top of `[CLS]`, but with N outputs (one per error type):

```
[CLS] token hidden state   →   Dropout(p=hidden_dropout_prob)   →   Linear(768 → N)   →   logits
```

- **Input to head**: `[CLS]` representation — shape `[batch, 768]`
- **Output of head**: raw logits — shape `[batch, N]`
- **N** is determined dynamically by scanning all error types in the dataset
- **One head only**: no separate head per error type — a single linear layer produces all N logits simultaneously

From the training scripts:
```python
config = AutoConfig.from_pretrained(
    ...,
    num_labels=num_labels,               # N error types, detected from data
    problem_type="multi_label_classification",
    id2label={i: l for i, l in enumerate(label_list)},
    label2id=label2id,
)
model = AutoModelForSequenceClassification.from_pretrained(
    model_args.model_name_or_path,
    config=config,
    trust_remote_code=True,
)
```

### Output Encoding
Multi-hot vector — each position is independent:
```python
def encode_error_types(example):
    multi_hot = [0.0] * num_labels
    for tag in example["error_types"].split(","):
        multi_hot[label2id[tag]] = 1.0   # can have multiple 1s
    example["labels"] = multi_hot
    return example
```

### Input / Output
- **Input**: tokenized `[CLS] premise [SEP] hypothesis [SEP]`
- **Output (training)**: raw logits `[batch, N]`
- **Output (inference)**: `sigmoid(logits)` → threshold 0.5 → multi-hot binary vector

From inference script:
```python
probs = torch.sigmoid(torch.tensor(all_logits)).numpy()
y_pred = (probs >= THRESHOLD).astype(int)   # THRESHOLD = 0.5
```

**Key difference from binary**: sigmoid (independent per label) instead of softmax (mutually exclusive).

### Fine-tuning Scope
**Whole model fine-tuned** — backbone + classification head. No layer freezing.

### Loss Function
**Custom `BCEWithLogitsLoss` with class-balanced `pos_weight`** — because error types are rare (highly imbalanced):

```python
# Compute pos_weight per label
pos_counts = all_labels_arr.sum(axis=0)           # how many positives per label
neg_counts = len(all_labels_arr) - pos_counts
pos_weight = torch.tensor([
    neg_counts[i] / pos_counts[i]                  # upweight rare positives
    for i in range(num_labels)
])

# Custom loss in WeightedTrainer
self.loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
    labels = inputs.pop("labels").float()
    model_inputs = {k: v for k, v in inputs.items()
                    if k in ['input_ids', 'attention_mask', 'token_type_ids']}
    outputs = model(**model_inputs)
    loss = self.loss_fct(outputs.logits, labels)
    return (loss, outputs) if return_outputs else loss
```

`BCEWithLogitsLoss` = fused sigmoid + binary cross-entropy per label, summed over all N labels.

#### Why Not Cross-Entropy Here?

Cross-entropy uses `softmax` which forces all class probabilities to sum to 1. That would mean the model can only predict one error type. But a single corrupted summary can have `entity_person_swap` AND `number_swap` at the same time. Each error type is an independent yes/no question, so we need **independent binary decisions** — one sigmoid per label.

#### BCEWithLogitsLoss — Worked Example (without pos_weight first)

**Scenario**: 4 error types for simplicity: `[number_swap, entity_person_swap, verb_tense_swap, noun_gender_swap]`

```
Input (corrupted summary sentence):
  "ראש הממשלה ביקר ב-1995 את שרה כהן, שזכתה בפרס בשנה שעברה"
   ↑ year is wrong (number_swap)  ↑ name is wrong (entity_person_swap)

True label (multi-hot vector):
  label = [1,    1,    0,           0          ]
           ↑     ↑     ↑            ↑
        number entity verb_tense  noun_gender
        _swap  _person  _swap       _swap

Step 1 — model outputs 4 raw logits (one per error type):
  logits = [2.1,  1.8,  -0.9,  -2.3]

Step 2 — sigmoid applied independently to each logit:
  sigmoid(x) = 1 / (1 + exp(-x))

  sigmoid(2.1)  = 0.89   → 89% chance of number_swap        label=1 ✓ confident correct
  sigmoid(1.8)  = 0.86   → 86% chance of entity_person_swap label=1 ✓ confident correct
  sigmoid(-0.9) = 0.29   → 29% chance of verb_tense_swap    label=0 ✓ correctly rejected
  sigmoid(-2.3) = 0.09   →  9% chance of noun_gender_swap   label=0 ✓ correctly rejected

Step 3 — binary cross-entropy computed INDEPENDENTLY per label:
  BCE(p, y) = -[ y * log(p) + (1-y) * log(1-p) ]

  label 0 (number_swap):      BCE(0.89, 1) = -log(0.89)       = 0.116
  label 1 (entity_person):    BCE(0.86, 1) = -log(0.86)       = 0.151
  label 2 (verb_tense):       BCE(0.29, 0) = -log(1 - 0.29)   = 0.149
  label 3 (noun_gender):      BCE(0.09, 0) = -log(1 - 0.09)   = 0.094

Step 4 — total loss = sum (or mean) across all 4 labels:
  loss = 0.116 + 0.151 + 0.149 + 0.094 = 0.510   ← low, model mostly correct
```

#### pos_weight — Why It's Needed

**The class imbalance problem**: In a dataset of 10,000 examples, suppose:
```
  number_swap:       appears in 2,000 examples  → common
  entity_date_swap:  appears in    80 examples  → very rare
  noun_gender_swap:  appears in   150 examples  → rare
```

Without pos_weight, the model learns to always predict 0 for `entity_date_swap` (80 positives vs 9920 negatives) — it gets 99.2% accuracy on that label just by always saying "no" — but it never detects the actual error.

**pos_weight = neg_count / pos_count per label**:

```
  pos_weight computation (from training set):

  number_swap:      pos=2000, neg=8000  → pos_weight = 8000/2000   =  4.0
  entity_date_swap: pos=  80, neg=9920  → pos_weight = 9920/80     = 124.0
  noun_gender_swap: pos= 150, neg=9850  → pos_weight = 9850/150    =  65.7

  pos_weight = [4.0, 124.0, 65.7, ...]
```

**What pos_weight does to the loss formula**:

```
  Without pos_weight:
    BCE = -[ y*log(p) + (1-y)*log(1-p) ]

  With pos_weight w:
    BCE = -[ w * y*log(p) + (1-y)*log(1-p) ]
                ↑
          only the positive term (y=1) gets multiplied

  Effect: a wrong prediction on a rare positive class costs 124× more than
          a wrong prediction on the negative class
```

**Concrete example for entity_date_swap (pos_weight=124)**:

```
  True label = 1 (this sentence has a date swap error)
  Model predicts sigmoid(-1.5) = 0.18   ← model says only 18% chance

  Without pos_weight:
    loss = -log(0.18) = 1.71

  With pos_weight=124:
    loss = -(124 * log(0.18) + 0) = 124 × 1.71 = 212.0

  ← 124× bigger gradient → model is forced to pay attention to this rare class
```

**And when label=0 (negative example)**, pos_weight has zero effect:
```
  True label = 0, model predicts 0.18 (correctly low)
  loss = -(124 × 0×log(0.18)  +  1×log(1-0.18))
       = -(0 + log(0.82)) = 0.198   ← normal, unmodified
```

**Summary**:
| | Cross-Entropy (Binary NLI) | BCEWithLogitsLoss (Multi-label) |
|---|---|---|
| Activation | Softmax (probabilities sum to 1) | Sigmoid (each label independent) |
| Labels | Exactly one class is true | Multiple classes can be true |
| Loss per example | One term for the true class | N independent binary terms |
| Imbalance handling | N/A (2 balanced classes) | `pos_weight` upweights rare positives |

### Stratified Oversampling
To address class imbalance further, a `WeightedRandomSampler` oversamples rare error-type examples:

```python
example_weights = []
for vec in all_labels_arr:
    pos_idxs = np.where(vec == 1)[0]
    w = np.mean(neg_counts[pos_idxs] / pos_counts[pos_idxs])
    example_weights.append(w)

sampler = data_utils.WeightedRandomSampler(
    weights=example_weights,
    num_samples=len(example_weights),
    replacement=True,
)
```

### Common Training Configuration
| Parameter | Value |
|---|---|
| Learning rate | 2e-5 |
| Epochs | 2 |
| Optimizer | AdamW with cosine LR scheduler |
| Warmup steps | 300 |
| Loss | `BCEWithLogitsLoss` + class-balanced `pos_weight` |
| Evaluation metric | micro-averaged F1 |

### Error Types Detected (from inference script)
```
Morphological/Syntactic: idiom_collocation_corruption, morphological_connective_confusion,
  noun_gender_swap, verb_gender_swap, verb_tense_swap, directional_preposition_swap

Linguistic Phenomena: pronoun_reference_errors, specificity_shift_errors,
  homographic_gender_errors, hebrew_root_pattern_confusion,
  impersonal_to_personal_verb_errors, incremental_temporal_state_mutations,
  evidentiality_source_attribution_collapse, verb_template_agency_reversal,
  double_verb_dependency_errors, construct_state_confusion, subject_verb_person_mismatch

Misc: measure_unit_swap, number_swap, sentence_negation

Entity Swaps: entity_title_swap, entity_event_swap, entity_location_swap,
  entity_person_swap, entity_organization_swap, entity_date_swap
```

---

### Setup 1: Summary-Level Classification

**Granularity: document-level — the whole article vs. the whole summary in one shot.**

```
PREMISE  =  Full source article (hundreds of tokens)
────────────────────────────────────────────────────────────────────
  "שר הביטחון יואב גלנט נפגש אתמול עם עמיתו האמריקאי בפנטגון.
   הפגישה התקיימה בוושינגטון ועסקה בסיוע הצבאי לאוקראינה.
   גלנט הגיע לוושינגטון לאחר ביקור בבריסל בפני שרי ביטחון נאט"ו.
   ... (hundreds more tokens) ..."

HYPOTHESIS  =  Full corrupted summary (multiple sentences, multiple errors)
────────────────────────────────────────────────────────────────────
  "שר החוץ יואב גלנט נפגש עם עמיתו האמריקאי בפנטגון."    ← entity_title_swap (ביטחון→חוץ)
  "הפגישה התקיימה בבריסל ועסקה בסיוע הצבאי לאוקראינה."   ← entity_location_swap (וושינגטון→בריסל)
  "גלנט הגיע לוושינגטון לפני ביקור בבריסל."              ← verb_tense_swap (לאחר→לפני)

INPUT TOKEN SEQUENCE (up to 4096 tokens):
  [CLS] שר הביטחון יואב... (full article) [SEP] שר החוץ יואב... (full summary) [SEP]
  ◄──────────────────────────── max 4096 tokens ───────────────────────────────────►
```

```
                    DictaBERT encoder
                          │
                    [CLS] → Linear(768 → N)
                          │
                          ▼
Output — multi-hot vector [N=24 error types]:

  Index  Error type               Logit   Sigmoid  Pred  True
  ─────  ───────────────────────  ──────  ───────  ────  ────
    0    number_swap               -2.1    0.11      0     0
    1    entity_person_swap        -1.8    0.14      0     0
    2    entity_title_swap          2.4    0.92      1     1  ✓
    3    entity_location_swap       2.1    0.89      1     1  ✓
    4    entity_date_swap          -1.5    0.18      0     0
    5    verb_tense_swap            1.9    0.87      1     1  ✓
    6    noun_gender_swap          -2.3    0.09      0     0
    ...  ...                       ...    ...      ...   ...

  pred  = [0, 0, 1, 1, 0, 1, 0, ...]   ← 3 errors detected across whole summary
  label = [0, 0, 1, 1, 0, 1, 0, ...]   ← ground truth
```

**Key property**: the model sees the entire article + entire summary at once. It detects **all** error types scattered across **all** sentences of the summary in a single forward pass. It cannot tell *which sentence* each error is in — only that the error type exists somewhere in the summary.

| Field | Content |
|---|---|
| `premise` | Full source article |
| `hypothesis` | Full corrupted summary (may contain multiple errors) |
| Max seq length | 4096 tokens |
| Batch size | 1 (effective 32 via gradient accumulation ×32) |
| Granularity | Document-level — all errors in one prediction |

**Limitation**: no sentence-level localization. If two sentences each have a different error, the model predicts both error types but cannot tell which error belongs to which sentence.

---

### Setup 2: Claim-Level Classification

**Granularity: sentence-level — one claim at a time, no article context.**

```
PREMISE  =  Original correct claim (single sentence)
────────────────────────────────────────────────────────────────────
  "שר הביטחון יואב גלנט הגיע לוושינגטון לאחר ביקור בבריסל"

HYPOTHESIS  =  Corrupted claim (single sentence, one error injected)
────────────────────────────────────────────────────────────────────
  "שר הביטחון יואב גלנט הגיע לוושינגטון לפני ביקור בבריסל"
                                              ↑
                                    לאחר → לפני  (verb_tense_swap)

INPUT TOKEN SEQUENCE:
  [CLS] שר הביטחון יואב גלנט הגיע לוושינגטון לאחר ביקור בבריסל [SEP]
        שר הביטחון יואב גלנט הגיע לוושינגטון לפני ביקור בבריסל [SEP]
  ◄───────────────────── max 3072 tokens ────────────────────────────►

  The model sees both sentences side by side — it must detect the error
  purely from the contrast (לאחר vs לפני), with NO article to cross-check.
```

```
                    DictaBERT encoder
                          │
                    [CLS] → Linear(768 → N)
                          │
                          ▼
Output — multi-hot vector [N=24 error types]:

  Index  Error type               Logit   Sigmoid  Pred  True
  ─────  ───────────────────────  ──────  ───────  ────  ────
    0    number_swap               -2.4    0.08      0     0
    1    entity_person_swap        -1.9    0.13      0     0
    2    entity_title_swap         -2.1    0.11      0     0
    3    entity_location_swap      -1.7    0.15      0     0
    4    verb_tense_swap            2.6    0.93      1     1  ✓
    5    noun_gender_swap          -2.2    0.10      0     0
    ...  ...                       ...    ...      ...   ...

  pred  = [0, 0, 0, 0, 1, 0, ...]   ← one error detected
  label = [0, 0, 0, 0, 1, 0, ...]
```

**Key property**: one example = one claim pair. Each claim is processed separately. The model learns to detect error types purely from linguistic differences between the original and corrupted sentence — word choice, morphology, syntax — without needing external factual knowledge.

**Example of what the model learns**:
- `לאחר` (after) → `לפני` (before) = temporal direction swap → `verb_tense_swap`
- `ירושלים` → `תל אביב` alone in this setup is ambiguous: is it `entity_location_swap` or just a valid paraphrase? Without the article, the model must rely purely on the contrast between the sentences.

| Field | Content |
|---|---|
| `premise` | Original correct claim (single sentence) |
| `hypothesis` | Corrupted claim (single sentence) |
| Max seq length | 3072 tokens |
| Batch size | 2 (effective 32 via gradient accumulation ×16) |
| Granularity | Sentence-level, no article context |

**Limitation**: no factual grounding. The model cannot verify whether a location/entity swap is actually wrong — it can only detect the structural/linguistic pattern of the corruption.

---

### Setup 3: Evidence-Based Claim-Level Classification

**Granularity: sentence-level + retrieved article evidence — best of both worlds.**

```
STEP 1 — Evidence Retrieval
────────────────────────────────────────────────────────────────────
Article sentences (all of them):
  sent₁: "שר הביטחון יואב גלנט נפגש אתמול עם עמיתו האמריקאי בפנטגון."
  sent₂: "הפגישה התקיימה בוושינגטון ועסקה בסיוע הצבאי לאוקראינה."
  sent₃: "גלנט הגיע לוושינגטון לאחר ביקור בבריסל בפני שרי ביטחון נאט"ו."
  sent₄: "ישראל ביקשה להגדיל את חבילת הנשק האמריקאית."
  sent₅: "שר החוץ קאץ' ייפגש עם בלינקן מחר בניו יורק."

Corrupted claim (the hypothesis):
  "שר הביטחון יואב גלנט הגיע לוושינגטון לפני ביקור בבריסל"
                                              ↑ לאחר→לפני

multilingual-MiniLM encodes the corrupted claim → cosine similarity vs all sentences:
  sent₁: 0.71  ✓ (mentions גלנט + location)
  sent₂: 0.58  ✓ (mentions וושינגטון)
  sent₃: 0.89  ✓ (mentions גלנט + וושינגטון + בריסל + לאחר)  ← most relevant!
  sent₄: 0.31  ✗ below threshold 0.5
  sent₅: 0.22  ✗ below threshold 0.5

Top-3 retrieved (similarity ≥ 0.5):  sent₃, sent₁, sent₂


STEP 2 — Build Input
────────────────────────────────────────────────────────────────────
PREMISE  =  Original claim  +  evidence sentences:
  "גלנט הגיע לוושינגטון לאחר ביקור בבריסל"        ← original claim
  "גלנט הגיע לוושינגטון לאחר ביקור בבריסל..."     ← evidence sent₃  (contains לאחר = ground truth!)
  "שר הביטחון יואב גלנט נפגש אתמול..."            ← evidence sent₁
  "הפגישה התקיימה בוושינגטון..."                   ← evidence sent₂

HYPOTHESIS  =  Corrupted claim:
  "שר הביטחון יואב גלנט הגיע לוושינגטון לפני ביקור בבריסל"

INPUT TOKEN SEQUENCE:
  [CLS] orig_claim | evidence₁ | evidence₂ | evidence₃ [SEP] corrupted_claim [SEP]
  ◄────────────────────────────── max seq length (TBD) ───────────────────────────►

  Now the model sees:
    - evidence₃ contains "לאחר" (after)  → corrupted claim says "לפני" (before)
    - factual contradiction confirmed from article
```

```
                    DictaBERT encoder
                          │
                    [CLS] → Linear(768 → N)
                          │
                          ▼
Output — multi-hot vector [N=24 error types]:

  Index  Error type               Logit   Sigmoid  Pred  True
  ─────  ───────────────────────  ──────  ───────  ────  ────
    0    number_swap               -2.5    0.08      0     0
    1    entity_location_swap      -2.0    0.12      0     0   ← Setup 2 might confuse
    2    verb_tense_swap            2.9    0.95      1     1  ✓  this because לפני/לאחר
    3    entity_person_swap        -2.1    0.11      0     0      could look like location
    ...  ...                       ...    ...      ...   ...      without the article

  pred  = [0, 0, 1, 0, ...]
  label = [0, 0, 1, 0, ...]
```

**Why Setup 3 is better than Setup 2 for entity swaps**:

```
Corrupted claim: "גלנט נפגש עם בלינקן בתל אביב"    (original: "בוושינגטון")

Setup 2 sees:    original: "...בוושינגטון"
                 corrupted: "...בתל אביב"
                 → detects location change, but is it entity_location_swap
                   or just a valid paraphrase? Uncertain without article.

Setup 3 sees:    original + evidence: "הפגישה התקיימה בוושינגטון"  ← article confirms
                 corrupted: "...בתל אביב"
                 → article contradicts "בתל אביב" → confident: entity_location_swap ✓
```

| Field | Content |
|---|---|
| `premise` | Original claim + top-K semantically similar article sentences |
| `hypothesis` | Corrupted claim |
| Max seq length | TBD |
| Granularity | Sentence-level + retrieved evidence |

Evidence extraction uses a sentence transformer (multilingual-MiniLM) to find the most semantically similar article sentences to the claim. Configuration:
- `SIMILARITY_THRESHOLD = 0.5`
- `MAX_CONTEXT_SENTENCES = 3`
- `DUPLICATE_THRESHOLD = 0.95`

**Trade-off vs Setup 1**: Setup 1 gets the whole article but loses sentence granularity. Setup 3 gets targeted evidence and per-sentence predictions, but retrieval quality depends on the sentence transformer — if the wrong evidence is retrieved, the model gets misleading context.

---

## 3. LLM Classifier — Gemma-2 9B

### Architecture Diagram

```
INPUT — formatted text prompt (one of 3 templates: Gemma/Alpaca/ChatML)
┌──────────────────────────────────────────────────────────────────┐
│  <start_of_turn>user                                             │
│  You are a Hebrew factuality checker...                          │
│  CONTEXT: [original claim + evidence sentences]                  │
│  CLAIM:   [corrupted claim]                                      │
│  <end_of_turn>                                                   │
│  <start_of_turn>model                                            │
└──────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│              Gemma-2 9B  (causal decoder)                        │
│              4-bit quantized — weights FROZEN                    │
│                                                                  │
│  Token₁ → Token₂ → ... → TokenN                                  │
│    │         │               │                                   │
│  Each layer has LoRA adapters injected:                          │
│                                                                  │
│   Attention:  q_proj, k_proj, v_proj, o_proj                     │
│   MLP:        gate_proj, up_proj, down_proj                      │
│                                                                  │
│   W_adapted = W_frozen(int4) + B × A × (alpha/r)                │
│                              ▲                                   │
│                    only A, B are trained                         │
│                    r=16, alpha=16  →  scale=1.0                  │
│                    ~0.58% of total parameters                    │
└──────────────────────────────────────────────────────────────────┘
                          │
                    token-by-token generation
                    (greedy decode)
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│  Raw generated text  →  JSON parsing                             │
│  {"error_type": "entity_location_swap", "claim": "...",          │
│   "context": ["..."]}                                            │
└──────────────────────────────────────────────────────────────────┘

LoRA adapter detail (per adapted weight matrix):
┌──────────────────────────────────────────┐
│  Input x  [d_in]                         │
│     │                                    │
│     ├──► W_frozen (int4, not updated)    │
│     │         │                          │
│     └──► A [d_in × 16]  (trained)        │
│               │                          │
│               ▼                          │
│          B [16 × d_out]  (trained)       │
│               │                          │
│    LoRA output = x @ A @ B × (16/16)    │
│               │                          │
│    Final = W_frozen(x) + LoRA output     │
└──────────────────────────────────────────┘
```

### Base Model
- **Architecture**: `unsloth/gemma-2-9b` — causal decoder-only LLM, 9 billion parameters
- **Quantization**: 4-bit (QLoRA via `unsloth.FastLanguageModel`)
- **No classification head** — this is a text generation model, not a sequence classifier

### LoRA Adapter
Only LoRA weights are trained — **~0.58% of total parameters**:

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                   # LoRA rank — 16 trainable dimensions per adapted weight
    lora_alpha=16,          # scaling factor = alpha/r = 1.0
    lora_dropout=0,
    bias="none",
    target_modules=[        # applied to all attention + MLP projection layers
        "q_proj", "k_proj", "v_proj", "o_proj",    # attention
        "gate_proj", "up_proj", "down_proj",        # MLP (SwiGLU)
    ],
    use_gradient_checkpointing="unsloth",
    use_rslora=False,
)
```

LoRA inserts trainable rank-16 decomposition matrices `A` and `B` into each target layer:
```
W_adapted = W_frozen + (B × A) × (alpha / r)
```
`W_frozen` is never updated. Only `A` and `B` matrices (shape `[d, 16]` and `[16, d]`) are trained.

### Input / Output
- **Input**: formatted text prompt (one of three formats below)
- **Output**: generated text token-by-token, parsed as JSON

**Gemma format (50% of training data):**
```
<bos><start_of_turn>user
You are a Hebrew factuality checker...
CONTEXT: [original claim + evidence sentences]
CLAIM: [corrupted claim]
<end_of_turn>
<start_of_turn>model
{"error_type": "entity_swap", "claim": "...", "context": ["..."]}<end_of_turn><eos>
```

**Alpaca format (35%):**
```
### Instruction:
Given context and claim, identify errors...
### Input:
CONTEXT: [...] CLAIM: [...]
### Response:
{"error_type": "..."}
```

**ChatML format (15%):**
```
<|im_start|>system
You are a Hebrew factuality checker...<|im_end|>
<|im_start|>user
CONTEXT: [...] CLAIM: [...]<|im_end|>
<|im_start|>assistant
{"error_type": "..."}<|im_end|>
```

### Loss Function
**Standard causal LM cross-entropy** via HuggingFace SFTTrainer:
```
CrossEntropyLoss over next-token predictions on the response portion of the prompt
```

No custom loss — the SFT (Supervised Fine-Tuning) objective is simply: predict the next token given all previous tokens, with loss computed only on the response tokens.

### Fine-tuning Scope
**Only LoRA weights trained** — the 9B base model weights are entirely frozen (4-bit quantized, not updated). Only the low-rank adapter matrices A and B are updated.

### Training Configuration
| Parameter | Value |
|---|---|
| Max seq length | 4096 tokens |
| Batch size | 2 (effective 16 via gradient accumulation ×8) |
| Learning rate | 2e-4 |
| Epochs | 1 |
| Optimizer | AdamW 8-bit |
| LR scheduler | Linear |
| Warmup steps | 50 |
| Best model criterion | `eval_loss` |
| Gradient checkpointing | `unsloth` (memory-optimized) |

### Memory Optimization
- ✅ 4-bit quantization (base model weights stored as int4)
- ✅ Gradient checkpointing (recompute activations instead of storing)
- ✅ LoRA (only 0.58% of params have gradients)
- ✅ AdamW 8-bit (optimizer states in 8-bit)
- ✅ Dynamic padding (`packing=False`, no fixed padding to max_seq_length)

---

## Architecture Comparison

### Side-by-Side Diagram

```
BINARY NLI               MULTI-LABEL (×3 setups)        GEMMA-2 9B (LLM)
────────────────────     ────────────────────────        ──────────────────────────

[CLS] art [SEP]          [CLS] premise [SEP]             <start_of_turn>user
      sum [SEP]                hypothesis [SEP]           CONTEXT: ...
      │                        │                          CLAIM: ...
      │                        │                          <end_of_turn>
      ▼                        ▼                          │
DictaBERT (768)          DictaBERT (768)                  ▼
12 transformer           12 transformer            Gemma-2 9B decoder
layers                   layers                    (4-bit frozen)
FULLY fine-tuned         FULLY fine-tuned          + LoRA A,B matrices
      │                        │                   (only trained, 0.58%)
  [CLS] 768              [CLS] 768                        │
      │                        │                          ▼
  Dropout                  Dropout                  token-by-token
      │                        │                    generation
  Linear(768→2)           Linear(768→N)                   │
      │                        │                          ▼
  logits [B,2]            logits [B,N]              JSON output
      │                        │                    parsed as
   Softmax                  Sigmoid                 error_type string
      │                        │
   Argmax                  ≥ 0.5 ?
      │                   per label
      ▼                        ▼
  0 or 1                 multi-hot [N]
(consistent/             (multiple errors
 not)                     per example)

Loss: CrossEntropy       Loss: BCEWithLogits        Loss: Causal LM
                         + pos_weight               CrossEntropy
                         (class-balanced)           (response tokens only)
```

| | Binary NLI | Multi-label (all setups) | Gemma-2 9B |
|---|---|---|---|
| Base model | DictaBERT (BERT encoder) | DictaBERT (BERT encoder) | Gemma-2 9B (causal decoder) |
| Hidden size | 768 | 768 | 3584 |
| Classification head | Linear(768→2) | Linear(768→N) | None |
| Head count | 1 | 1 | N/A |
| Fine-tuning scope | **Full model** | **Full model** | **LoRA only (0.58%)** |
| Activation before logits | None (raw logits) | None (raw logits) | N/A |
| Inference activation | Softmax | Sigmoid | Greedy/beam decode |
| Loss function | CrossEntropy | BCEWithLogitsLoss + pos_weight | Causal LM CrossEntropy |
| Output type | Single class (argmax) | Multi-hot (threshold 0.5) | JSON text |
| Max tokens | 4096 | 3072–4096 | 4096 |

---

## File Locations

### Binary Classifier
- Training: `bertencoderclassifiers/binaryclassifiers/mfact_original_binary_classifier.py`
- Inference: `bertencoderclassifiers/binaryclassifiers/inference_binary_nli.py`
- Training args: `bertencoderclassifiers/binaryclassifiers/traning_args.txt`

### Multi-Label Classifiers
- Setup 1 (article+summary): `bertencoderclassifiers/multillabelclassifier/article_summary_errortypes/mfact_train_multilabel_neo.py`
- Setup 2 (claim-level): `bertencoderclassifiers/multillabelclassifier/article_summaryClaim_errortypes/mfact_train_multilabel_claim_level_neo.py`
- Setup 3 (evidence-based): `bertencoderclassifiers/multillabelclassifier/articleEvidence_summaryClaim_errorTypes/mfact_train_multilabel_evidence_claim_level.py`

### LLM Classifier
- Dataset parser: `llmclassifiers/shared/parse_llm_finetune_dataset.py`
- Training: `llmclassifiers/finetune_gemma2_9b_evidence.py`
- Inference: `llmclassifiers/shared/inference_model.py`
- vLLM inference: `llmclassifiers/shared/inference_model_vllm.py`

---

## Notes

- All DictaBERT models use `trust_remote_code=True` (required for DictaBERT's custom code)
- Best model checkpoints saved automatically using `load_best_model_at_end=True`
- Multi-label setups use a custom `WeightedTrainer` subclass that overrides both `compute_loss` and `get_train_dataloader`
- The `pos_weight` tensor is computed per-label from training set statistics before training begins
- For Gemma-2, the `dataset_num_proc=1` and `dataloader_num_workers=0` settings are critical to prevent tokenizer process spawning issues
