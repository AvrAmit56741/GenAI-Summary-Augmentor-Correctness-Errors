import json
import torch
from datasets import load_from_disk
from tqdm import tqdm
from vllm import LLM, SamplingParams
from collections import defaultdict
import unsloth


print("v3")
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
# NOTE: vLLM requires a MERGED model, not a LoRA adapter
# If you have a LoRA adapter, merge it first (see instructions below)
MODEL_PATH = "/home/avramit/llm_finetune/hebrew_evidence_llm_lora_70p_merged"  # Path to MERGED model
LORA_ADAPTER_PATH = "/home/avramit/llm_finetune/hebrew_evidence_llm_lora_70p"  # Original LoRA adapter (for merging)
BASE_MODEL = "unsloth/gemma-2-9b-it-bnb-4bit"  # Base model used for training

TEST_DATASET_PATH = "/home/avramit/llm_finetune/llm_finetune_big_parse_dataset/llm_evidence_dataset_test/"
OUTPUT_RESULTS_PATH = "inference_results_vllm.json"
OUTPUT_METRICS_PATH = "inference_metrics_vllm.json"
MAX_SEQ_LENGTH = 4096

# vLLM Configuration
TENSOR_PARALLEL_SIZE = 1  # Number of GPUs to use (1 for single GPU)
GPU_MEMORY_UTILIZATION = 0.9  # Use 90% of GPU memory (adjust if OOM)
MAX_NUM_SEQS = 256  # Maximum number of sequences to process in parallel

# Merge LoRA before inference (set to True for first run)
MERGE_LORA_BEFORE_INFERENCE = False  # Set to True to merge LoRA adapter once

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
MAX_TEST_SAMPLES = 6000

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


def merge_lora_adapter():
    """Merge LoRA adapter with base model for vLLM compatibility"""
    import os
    from unsloth import FastLanguageModel
    
    print("\n" + "="*80)
    print("MERGING LoRA ADAPTER WITH BASE MODEL")
    print("="*80)
    print(f"Base model: {BASE_MODEL}")
    print(f"LoRA adapter: {LORA_ADAPTER_PATH}")
    print(f"Output path: {MODEL_PATH}")
    print("="*80 + "\n")
    
    # Check if merged model already exists
    if os.path.exists(MODEL_PATH) and os.path.exists(os.path.join(MODEL_PATH, "config.json")):
        print(f"⚠ Merged model already exists at {MODEL_PATH}")
        response = input("Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Using existing merged model.")
            return
    
    print("Loading base model and LoRA adapter...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=LORA_ADAPTER_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=False,  # Must be False for merging
    )
    
    print("Merging LoRA weights into base model...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Should match your training config
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    
    # Merge and save
    print(f"Saving merged model to {MODEL_PATH}...")
    os.makedirs(MODEL_PATH, exist_ok=True)
    model.save_pretrained_merged(MODEL_PATH, tokenizer, save_method="merged_16bit")
    
    print("✓ LoRA adapter merged successfully!")
    print(f"✓ Merged model saved to: {MODEL_PATH}")
    print("="*80 + "\n")


def load_model():
    """Load fine-tuned model with vLLM"""
    print("Loading model with vLLM...")
    
    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"✓ Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ WARNING: CUDA not available, vLLM requires GPU")
        raise RuntimeError("vLLM requires GPU for inference")
    
    # Check if model path exists and has config.json
    import os
    config_path = os.path.join(MODEL_PATH, "config.json")
    if not os.path.exists(config_path):
        print(f"\n⚠ ERROR: Model path does not contain config.json")
        print(f"   Path: {MODEL_PATH}")
        print(f"\n💡 Solution: Your model appears to be a LoRA adapter.")
        print(f"   vLLM requires a merged model. Please:")
        print(f"   1. Set MERGE_LORA_BEFORE_INFERENCE = True")
        print(f"   2. Run this script once to merge the adapter")
        print(f"   3. Set MERGE_LORA_BEFORE_INFERENCE = False")
        print(f"   4. Run inference normally\n")
        raise RuntimeError("Model not found or not merged. See instructions above.")
    
    # Initialize vLLM with optimized settings
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_num_seqs=MAX_NUM_SEQS,
        max_model_len=MAX_SEQ_LENGTH,
        trust_remote_code=True,
    )
    
    print(f"✓ vLLM model loaded successfully")
    print(f"  - Max sequences in parallel: {MAX_NUM_SEQS}")
    print(f"  - GPU memory utilization: {GPU_MEMORY_UTILIZATION * 100}%")
    
    return llm


def generate_predictions_batch(llm, prompts, sampling_params):
    """Generate predictions for a batch of prompts using vLLM"""
    # vLLM automatically handles batching and optimization
    outputs = llm.generate(prompts, sampling_params)
    
    # Extract responses
    responses = []
    for output in outputs:
        generated_text = output.outputs[0].text
    # Extract only the response part (after "### Response:")
        response = generated_text.split("### Response:")[-1].strip()
        responses.append(response)

    return responses


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
    """Main inference loop with vLLM batch processing"""
    # Load model
    llm = load_model()
    
    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=0.3,
        top_p=0.9,
        max_tokens=512,
    )
    print(f"Sampling params: temperature={sampling_params.temperature}, top_p={sampling_params.top_p}, max_tokens={sampling_params.max_tokens}")

    # Load test dataset (your separate test file)
    print(f"\nLoading test dataset from {TEST_DATASET_PATH}...")
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
    
    print(f"\n{'='*80}")
    print(f"Running vLLM batch inference on {len(test_dataset)} test samples...")
    print(f"{'='*80}\n")

    # Prepare all prompts for batch processing
    print("Preparing prompts for batch processing...")
    prompts = []
    samples_data = []
    
    for i, sample in enumerate(test_dataset):
        instruction = sample["instruction"]
        input_text = sample["input"]
        expected_output = sample["output"]
        
        # Format prompt (without the response part)
        prompt = alpaca_prompt.format(instruction, input_text, "")
        prompts.append(prompt)
        
        # Extract expected error type
        expected_error_type = ""
        try:
            expected_json = json.loads(expected_output)
            expected_error_type = expected_json.get("error_type", "")
        except:
            pass
        
        samples_data.append({
            "sample_id": i,
            "instruction": instruction,
            "input_text": input_text,
            "expected_output": expected_output,
            "expected_error_type": expected_error_type
        })
    
    print(f"✓ Prepared {len(prompts)} prompts")
    
    # Generate predictions in batch (vLLM handles batching automatically)
    print(f"\n🚀 Generating predictions with vLLM (this will be fast!)...")
    predicted_outputs = generate_predictions_batch(llm, prompts, sampling_params)
    print(f"✓ Generated {len(predicted_outputs)} predictions")
    
    # Process results
    print("\nProcessing results...")
    results = []
    
    for i, (predicted_output, sample_data) in enumerate(tqdm(zip(predicted_outputs, samples_data), total=len(predicted_outputs))):
        # Print sample details (only if PRINT_EACH_SAMPLE is True)
        if PRINT_EACH_SAMPLE:
            print(f"\n{'=' * 80}")
            print(f"SAMPLE {i + 1}/{len(predicted_outputs)}")
            print(f"{'=' * 80}")
            print(f"INSTRUCTION:\n{sample_data['instruction']}\n")
            print(f"INPUT:\n{sample_data['input_text']}\n")
            print(f"EXPECTED OUTPUT:\n{sample_data['expected_output']}\n")
            print(f"PREDICTED OUTPUT:\n{predicted_output}")
            print(f"{'=' * 80}\n")

        # Parse prediction
        parsed_prediction = parse_json_output(predicted_output)

        # Store result
        result = {
            "sample_id": sample_data["sample_id"],
            "instruction": sample_data["instruction"],
            "input": sample_data["input_text"][:500] + "..." if len(sample_data["input_text"]) > 500 else sample_data["input_text"],
            "expected_output": sample_data["expected_output"],
            "expected_error_type": sample_data["expected_error_type"],
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
    import time
    
    # Merge LoRA adapter if requested
    if MERGE_LORA_BEFORE_INFERENCE:
        merge_lora_adapter()
        print("\n✓ Merge complete! Now set MERGE_LORA_BEFORE_INFERENCE = False and run again for inference.\n")
        exit(0)
    
    # Print configuration
    print("\n" + "="*80)
    print("vLLM INFERENCE CONFIGURATION")
    print("="*80)
    print(f"Model: {MODEL_PATH}")
    print(f"Test dataset: {TEST_DATASET_PATH}")
    print(f"Output results: {OUTPUT_RESULTS_PATH}")
    print(f"Output metrics: {OUTPUT_METRICS_PATH}")
    print(f"\nvLLM Settings:")
    print(f"  Max seq length: {MAX_SEQ_LENGTH}")
    print(f"  Tensor parallel size: {TENSOR_PARALLEL_SIZE}")
    print(f"  GPU memory utilization: {GPU_MEMORY_UTILIZATION * 100}%")
    print(f"  Max sequences in parallel: {MAX_NUM_SEQS}")
    print(f"\nFiltering:")
    print(f"  Filter error types: {FILTER_ERROR_TYPES if FILTER_ERROR_TYPES else 'None (all types)'}")
    print(f"  Max test samples: {MAX_TEST_SAMPLES if MAX_TEST_SAMPLES else 'None (all samples)'}")
    print(f"  Print each sample: {PRINT_EACH_SAMPLE}")
    print("="*80 + "\n")
    
    start_time = time.time()
    results, metrics = run_inference()
    end_time = time.time()
    
    duration = end_time - start_time
    samples_per_second = len(results) / duration if duration > 0 else 0
    
    print("\n" + "="*80)
    print("✓ Inference complete!")
    print("="*80)
    print(f"Total time: {duration:.2f} seconds")
    print(f"Throughput: {samples_per_second:.2f} samples/second")
    print(f"Avg time per sample: {duration / len(results):.3f} seconds")
    print("="*80)
