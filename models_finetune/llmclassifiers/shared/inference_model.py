import json
import torch
from datasets import load_from_disk
from tqdm import tqdm
from unsloth import FastLanguageModel
from collections import defaultdict

# Error type constants
class ErrorTypes:
    # Morphological & Syntactic
    IDIOM_COLLOCATION_CORRUPTION = "idiom_collocation_corruption"
    MORPHOLOGICAL_CONNECTIVE_CONFUSION = "morphological_connective_confusion"
    NOUN_GENDER_SWAP = "noun_gender_swap"
    VERB_GENDER_SWAP = "verb_gender_swap"
    VERB_TENSE_SWAP = "verb_tense_swap"
    DIRECTIONAL_PREPOSITION_SWAP = "directional_preposition_swap"
    
    # Linguistic Phenomena
    PRONOUN_REFERENCE_ERRORS = "pronoun_reference_errors"
    SPECIFICITY_SHIFT_ERRORS = "specificity_shift_errors"
    HOMOGRAPHIC_GENDER_ERRORS = "homographic_gender_errors"
    HEBREW_ROOT_PATTEN_CONFUSION = "hebrew_root_pattern_confusion"
    IMPERSONAL_TO_PERSONAL_VERB_ERRORS = "impersonal_to_personal_verb_errors"
    INCREMENTAL_TEMPORAL_STATE_MUTATIONS = "incremental_temporal_state_mutations"
    EVIDENTIALITY_SOURCE_ATTRIBUTION_COLLAPSE = "evidentiality_source_attribution_collapse"
    VERB_TEMPLATE_AGENCY_REVERSAL = "verb_template_agency_reversal"
    DOUBLE_VERB_DEPENDENCY_ERRORS = "double_verb_dependency_errors"
    CONSTRUCT_STATE_CONFUSION = "construct_state_confusion"
    SUBJECT_VERB_PERSON_MISMATCH = "subject_verb_person_mismatch"
    
    # Misc
    MEASURE_UNIT_SWAP = "measure_unit_swap"
    NUMBER_SWAP = "number_swap"
    SENTENCE_NEGATION = "sentence_negation"
    
    # Entity Swaps
    ENTITY_TITLE_SWAP = "entity_title_swap"
    ENTITY_EVENT_SWAP = "entity_event_swap"
    ENTITY_LOCATION_SWAP = "entity_location_swap"
    ENTITY_PERSON_SWAP = "entity_person_swap"
    ENTITY_ORGANIZATION_SWAP = "entity_organization_swap"
    ENTITY_DATE_SWAP = "entity_date_swap"

# ========================
# CONFIGURATION (Edit these values)
# ========================
MODEL_PATH = "/home/avramit/llm_finetune/hebrew_evidence_llm_lora_70p"
TEST_DATASET_PATH = "/home/avramit/llm_finetune/llm_finetune_big_parse_dataset/llm_evidence_dataset_test/"
OUTPUT_RESULTS_PATH = "inference_results.json"
OUTPUT_METRICS_PATH = "inference_metrics.json"
MAX_SEQ_LENGTH = 4096

# Test filtering configuration
# Options: 
#   - None or [] = Test all samples
#   - [ErrorTypes.NUMBER_SWAP, ErrorTypes.SENTENCE_NEGATION] = Test only specific error types
FILTER_ERROR_TYPES = None
#FILTER_ERROR_TYPES = [ErrorTypes.IDIOM_COLLOCATION_CORRUPTION]
# Test sample limit configuration
# Options:
#   - None = Test all samples (after filtering)
#   - N (integer) = Test only first N samples (after filtering)
MAX_TEST_SAMPLES = 10

# Print mode configuration
# Options:
#   - True = Print each sample's classification (detailed output)
#   - False = Silent mode (faster, only show progress bar and final summary)
PRINT_EACH_SAMPLE = False

# Alpaca prompt template (same as training)
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


def load_model():
    """Load fine-tuned model"""
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)  # Enable inference mode
    
    # Check and print device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"✓ Model running on CUDA: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ WARNING: Model running on CPU (inference will be slow)")
    
    return model, tokenizer


def generate_prediction(model, tokenizer, instruction, input_text):
    """Generate prediction for a single sample"""
    # Format prompt (without the response part)
    prompt = alpaca_prompt.format(instruction, input_text, "")

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Generate
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.3,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the response part (after "### Response:")
    response = generated_text.split("### Response:")[-1].strip()

    return response


def parse_json_output(output_text):
    """Try to parse JSON from model output"""
    try:
        # Try direct parsing
        return json.loads(output_text)
    except:
        # Try to extract JSON from text (sometimes model adds extra text)
        import re
        json_match = re.search(r'\{.*\}', output_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass
    return None


def calculate_metrics(results):
    """Calculate accuracy metrics"""
    metrics = {
        "total_samples": len(results),
        "valid_json_outputs": 0,
        "error_type_correct": 0,
        "error_type_accuracy": 0.0,
        "error_type_distribution": defaultdict(int),
        "error_type_correct_distribution": defaultdict(int)
    }

    for result in results:
        expected_error_type = result.get("expected_error_type", "")
        parsed_pred = result["parsed_prediction"]

        if expected_error_type:
            metrics["error_type_distribution"][expected_error_type] += 1

        # Check if prediction is valid JSON
        if parsed_pred:
            metrics["valid_json_outputs"] += 1

            # Check error type accuracy
            predicted_error_type = parsed_pred.get("error_type", "")
            if predicted_error_type == expected_error_type:
                metrics["error_type_correct"] += 1
                metrics["error_type_correct_distribution"][expected_error_type] += 1

    # Calculate percentages
    if metrics["total_samples"] > 0:
        metrics["valid_json_percentage"] = (metrics["valid_json_outputs"] / metrics["total_samples"]) * 100
        metrics["error_type_accuracy"] = (metrics["error_type_correct"] / metrics["total_samples"]) * 100

    # Per-class accuracy
    metrics["per_class_accuracy"] = {}
    for error_type in metrics["error_type_distribution"]:
        total = metrics["error_type_distribution"][error_type]
        correct = metrics["error_type_correct_distribution"][error_type]
        metrics["per_class_accuracy"][error_type] = (correct / total * 100) if total > 0 else 0.0

    return metrics


def filter_dataset_by_error_types(dataset, error_types):
    """Filter dataset to only include samples with specific error types"""
    if not error_types:
        return dataset
    
    filtered_indices = []
    for i, sample in enumerate(dataset):
        try:
            expected_json = json.loads(sample["output"])
            sample_error_type = expected_json.get("error_type", "")
            if sample_error_type in error_types:
                filtered_indices.append(i)
        except:
            continue
    
    return dataset.select(filtered_indices)


def run_inference():
    """Main inference loop"""
    # Load model
    model, tokenizer = load_model()

    # Load test dataset (your separate test file)
    print(f"Loading test dataset from {TEST_DATASET_PATH}...")
    test_dataset = load_from_disk(TEST_DATASET_PATH)

    print(f"Original test dataset size: {len(test_dataset)} samples")
    
    # Filter by error types if specified
    if FILTER_ERROR_TYPES:
        print(f"\nFiltering for error types: {FILTER_ERROR_TYPES}")
        test_dataset = filter_dataset_by_error_types(test_dataset, FILTER_ERROR_TYPES)
        print(f"Filtered test dataset size: {len(test_dataset)} samples")
    else:
        print("Testing on ALL error types")
    
    # Limit number of samples if specified
    if MAX_TEST_SAMPLES is not None and MAX_TEST_SAMPLES < len(test_dataset):
        print(f"Limiting to first {MAX_TEST_SAMPLES} samples")
        test_dataset = test_dataset.select(range(MAX_TEST_SAMPLES))
        print(f"Limited test dataset size: {len(test_dataset)} samples")
    
    print(f"\nRunning inference on {len(test_dataset)} test samples...\n")

    results = []

    for i, sample in enumerate(tqdm(test_dataset, disable=PRINT_EACH_SAMPLE)):
        instruction = sample["instruction"]
        input_text = sample["input"]
        expected_output = sample["output"]

        # Print prompt (only if PRINT_EACH_SAMPLE is True)
        if PRINT_EACH_SAMPLE:
            print(f"\n{'=' * 80}")
            print(f"SAMPLE {i + 1}/{len(test_dataset)}")
            print(f"{'=' * 80}")
            print(f"INSTRUCTION:\n{instruction}\n")
            print(f"INPUT:\n{input_text}\n")
            print(f"EXPECTED OUTPUT:\n{expected_output}\n")

        # Generate prediction
        predicted_output = generate_prediction(model, tokenizer, instruction, input_text)

        # Print response (only if PRINT_EACH_SAMPLE is True)
        if PRINT_EACH_SAMPLE:
            print(f"PREDICTED OUTPUT:\n{predicted_output}")
            print(f"{'=' * 80}\n")

        # Parse prediction
        parsed_prediction = parse_json_output(predicted_output)

        # Extract expected error type
        expected_error_type = ""
        try:
            expected_json = json.loads(expected_output)
            expected_error_type = expected_json.get("error_type", "")
        except:
            pass

        # Store result
        result = {
            "sample_id": i,
            "instruction": instruction,
            "input": input_text[:500] + "..." if len(input_text) > 500 else input_text,  # Truncate for readability
            "expected_output": expected_output,
            "expected_error_type": expected_error_type,
            "predicted_output": predicted_output,
            "parsed_prediction": parsed_prediction,
            "is_valid_json": parsed_prediction is not None
        }
        results.append(result)

    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(results)

    # Save results
    print(f"\nSaving results to {OUTPUT_RESULTS_PATH}...")
    with open(OUTPUT_RESULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saving metrics to {OUTPUT_METRICS_PATH}...")
    with open(OUTPUT_METRICS_PATH, 'w', encoding='utf-8') as f:
        # Convert defaultdict to dict for JSON serialization
        metrics_serializable = {
            k: dict(v) if isinstance(v, defaultdict) else v
            for k, v in metrics.items()
        }
        json.dump(metrics_serializable, f, ensure_ascii=False, indent=2)

    # Print summary
    print("\n" + "=" * 50)
    print("INFERENCE SUMMARY")
    print("=" * 50)
    print(f"Total samples: {metrics['total_samples']}")
    print(f"Valid JSON outputs: {metrics['valid_json_outputs']} ({metrics.get('valid_json_percentage', 0):.2f}%)")
    print(
        f"Error type accuracy: {metrics['error_type_correct']}/{metrics['total_samples']} ({metrics['error_type_accuracy']:.2f}%)")
    print("\nPer-class accuracy:")
    for error_type, accuracy in metrics['per_class_accuracy'].items():
        total = metrics['error_type_distribution'][error_type]
        print(f"  {error_type}: {accuracy:.2f}% ({metrics['error_type_correct_distribution'][error_type]}/{total})")
    print("=" * 50)

    return results, metrics


if __name__ == "__main__":
    # Print configuration
    print("\n" + "="*50)
    print("INFERENCE CONFIGURATION")
    print("="*50)
    print(f"Model: {MODEL_PATH}")
    print(f"Test dataset: {TEST_DATASET_PATH}")
    print(f"Output results: {OUTPUT_RESULTS_PATH}")
    print(f"Output metrics: {OUTPUT_METRICS_PATH}")
    print(f"Max seq length: {MAX_SEQ_LENGTH}")
    print(f"Filter error types: {FILTER_ERROR_TYPES if FILTER_ERROR_TYPES else 'None (all types)'}")
    print(f"Max test samples: {MAX_TEST_SAMPLES if MAX_TEST_SAMPLES else 'None (all samples)'}")
    print(f"Print each sample: {PRINT_EACH_SAMPLE}")
    print("="*50 + "\n")
    
    results, metrics = run_inference()
    print("\nInference complete!")
