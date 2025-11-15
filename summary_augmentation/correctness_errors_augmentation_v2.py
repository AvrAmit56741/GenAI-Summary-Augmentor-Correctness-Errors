# -*- coding: utf-8 -*-
import concurrent.futures
import threading
import datetime
import pandas as pd
import logging
import sys
import os
from typing import List, Dict, Any
import logging
from typing import List
import random
import Levenshtein
import datetime
from summary_augmentation.config import get_workflow_config
from typing import Dict, List, Any, Tuple
from collections import Counter, defaultdict

# Configure the logger (only once, typically at the top of your script or entry point)
#logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
import json
import summary_augmentation.correctness_errors_examples as correctness_errors_examples
from openai import OpenAI
# Define OpenAI API credentials
# import google.generativeai as genai

import threading
from summary_augmentation.logger_config import setup_logger
logger = setup_logger(__name__)
workflow_config = get_workflow_config()

from summary_augmentation.correctness_errors_prompts import suitable_claim_decomposition_prompt, \
    claims_decomposition_prompt, apply_transformation_on_claims_prompt
from summary_augmentation.utils import filter_diverse_summaries, save_dataset_json, save_dataset_csv, \
    control_claims_after_transformation, get_entities_by_type, is_same_entity, extract_named_entities, \
    CorrectnessErrorTypes, text_decomposition,suitable_number_swap_transformation, recompose_transformed_summary, compute_all_diffs, ner_instructions
# Define the main data structure
from summary_augmentation.correctness_errors_prompts import claims_decomposition_prompt
from summary_augmentation.correctness_errors_llm_helper_functions import verify_transformed_summary_grammar, \
    choose_suitable_claim, summary_negative_augmentation, summary_halluzaition_negative_augmentation
import re
from summary_augmentation.config import get_enabled_transformations_as_dicts, get_enabled_transformations_as_lists, \
    MAXIMUM_ALLOWED_NEGATIVE_SUMMARIES_SIMILARITY_THRESHOLD, PERCENTAGE_TRANSFORMATIONS_PERFORM_ON_NEGATIVE_SUMMARY, CLAIM_DECOMPOSITION_METHOD

general_error_examples = correctness_errors_examples.general_error_examples
hebrew_error_examples = correctness_errors_examples.hebrew_error_examples

NER_TO_ERROR_TYPE_MAP = {
    'LOC': CorrectnessErrorTypes.ENTITY_LOCATION_SWAP,
    'GPE': CorrectnessErrorTypes.ENTITY_LOCATION_SWAP,
    'PER': CorrectnessErrorTypes.ENTITY_PERSON_SWAP,
    'ORG': CorrectnessErrorTypes.ENTITY_ORGANIZATION_SWAP,
    'EVE': CorrectnessErrorTypes.ENTITY_EVENT_SWAP,
    'TTL': CorrectnessErrorTypes.ENTITY_TITLE_SWAP
}


# hebrew_llm_transformations = [
#     # CorrectnessErrorTypes.NOUN_GENDER_SWAP,
#     # CorrectnessErrorTypes.VERB_GENDER_SWAP,
#     # CorrectnessErrorTypes.VERB_TENSE_SWAP,
#     CorrectnessErrorTypes.PRONOUN_REFERENCE_ERRORS,
#     CorrectnessErrorTypes.SPECIFICITY_SHIFT_ERRORS,
#     CorrectnessErrorTypes.HOMOGRAPHIC_GENDER_ERRORS,
#     CorrectnessErrorTypes.IMPERSONAL_TO_PERSONAL_VERB_ERRORS
#     #CorrectnessErrorTypes.DIRECTIONAL_PREPOSITION_SWAP
# ]
#
# general_llm_transformations = [
#     CorrectnessErrorTypes.SENTENCE_NEGATION,
#     CorrectnessErrorTypes.MEASURE_UNIT_SWAP,
#     CorrectnessErrorTypes.ENTITY_EVENT_SWAP,
#     CorrectnessErrorTypes.ENTITY_DATE_SWAP
# ]
#
# ner_based_transformation = [
#     CorrectnessErrorTypes.ENTITY_PERSON_SWAP,
#     CorrectnessErrorTypes.ENTITY_LOCATION_SWAP,
#     CorrectnessErrorTypes.ENTITY_TITLE_SWAP,
#     CorrectnessErrorTypes.ENTITY_ORGANIZATION_SWAP
# ]
def map_ner_to_error_types(
        ner_results: Dict[str, List[str]],
        desired_error_types: List[CorrectnessErrorTypes]
) -> Dict[CorrectnessErrorTypes, List[str]]:
    mapped_results = {}

    for ner_label, entities in ner_results.items():
        error_type = NER_TO_ERROR_TYPE_MAP.get(ner_label)
        if error_type in desired_error_types:
            mapped_results[error_type] = entities
    return mapped_results


def all_suitable_transformation_possible(text: str, claim_decomposition):
    # static_based_transformation = [CorrectnessErrorTypes.NUMBER_SWAP]
    hebrew_llm_transformations, general_llm_transformations, ner_based_transformation = get_enabled_transformations_as_lists()

    # Create separate transformation type lists
    hebrew_transformation_types_list = [t.strip() for t in hebrew_llm_transformations]
    general_transformation_types_list = [t.strip() for t in general_llm_transformations]

    # Filter examples using the appropriate transformation types
    general_filtered_examples = {key: general_error_examples[key] for key in general_llm_transformations if
                                 key in general_error_examples}
    hebrew_filtered_examples = {key: hebrew_error_examples[key] for key in hebrew_llm_transformations if
                                key in hebrew_error_examples}
    transformation_results = []

    # LLM-based transformation - First query with Hebrew examples and Hebrew transformation types
    try:
        logger.info(
            f"From all_suitable_transformation_possible - Activate LLM function of choose_suitable_claim - Hebrew Based Errors - Error types {hebrew_transformation_types_list}")
        hebrew_claim_results = choose_suitable_claim(claim_decomposition, hebrew_transformation_types_list,
                                                     hebrew_filtered_examples, model=workflow_config.choose_suitable_claim_model)
        # Check if the result has the attribute 'suitable_claims' and process each entry.
        if hebrew_claim_results and hasattr(hebrew_claim_results, "suitable_claims"):
            # Line 118 - ADD this line:
            logger.info(f"HEB-SUITABLE | types={len(hebrew_transformation_types_list)} | claims={len(getattr(hebrew_claim_results, 'suitable_claims', []))}")
            for suitable_claim in hebrew_claim_results.suitable_claims:
                transformation_results.append({
                    "claim": suitable_claim.original_claim,
                    "error_type": suitable_claim.correctness_error_type,
                    "word": suitable_claim.applicable_transformation_text_within_claim
                })
    except Exception as e:
        print(f"Error in LLM choose suitable claims for Hebrew transformation: {e}")
    # LLM-based transformation - Second query with general examples and general transformation types
    # try:
    #     general_claim_results = choose_suitable_claim(claim_decomposition, general_transformation_types_list,general_filtered_examples)
    #     # Check if the result has the attribute 'suitable_claims' and process each entry.
    #     if general_claim_results and hasattr(general_claim_results, "suitable_claims"):
    #         for suitable_claim in general_claim_results.suitable_claims:
    #             transformation_results.append({
    #                 "claim": suitable_claim.original_claim,
    #                 "error_type": suitable_claim.correctness_error_type,
    #                 "word": suitable_claim.applicable_transformation_text_within_claim
    #             })
    # except Exception as e:
    #     print(f"Error in LLM choose suitable claims for general transformation: {e}")
    try:
        # Parallel execution for general error types
        def process_general_error_type(err_type, claim_decomposition, examples):
            try:
                logger.info(f"From all_suitable_transformation_possible - Activate LLM function of choose_suitable_claim - General Based Errors - Error type {[err_type]}")
                result = choose_suitable_claim(
                    claim_decomposition,
                    [err_type],
                    examples,
                    model=workflow_config.choose_suitable_claim_model
                )
                return result
            except Exception as e:
                logger.error(f"Error processing general error type {err_type}: {e}")
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for err_type in general_llm_transformations:
                if err_type in general_filtered_examples:
                    examples = {err_type: general_filtered_examples[err_type]}
                    future = executor.submit(process_general_error_type, err_type, claim_decomposition, examples)
                    futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result and hasattr(result, "suitable_claims"):
                    for sc in result.suitable_claims:
                        transformation_results.append({
                            "claim": sc.original_claim,
                            "error_type": sc.correctness_error_type,
                            "word": sc.applicable_transformation_text_within_claim
                        })

    except Exception as e:
        print(f"Error in LLM choose suitable claims for general transformation: {e}")

    # NER-based transformation
    # try:
    #     # Get NER entities with their positions
    #     summary_ner = extract_named_entities(text)
    #     if summary_ner:
    #         # Process each claim to find its span in the full text
    #         for claim_item in claim_decomposition.claims_decomposition:
    #             # Extract the claim text based on the data structure
    #             # It could be a dictionary with a numeric key, or a tuple with (index, text)
    #             if isinstance(claim_item, dict):
    #                 # Handle dictionary format
    #                 for _, claim_text in claim_item.items():
    #                     break  # Just get the first (and only) value
    #             elif isinstance(claim_item, tuple):
    #                 # Handle tuple format (index, text)
    #                 claim_text = claim_item[1]
    #             elif isinstance(claim_item, str):
    #                 # Handle string format directly
    #                 claim_text = claim_item
    #             else:
    #                 # Skip unknown formats
    #                 continue
    #
    #             # Find where this claim appears in the full text
    #             claim_start = text.find(claim_text)
    #             if claim_start == -1:
    #                 # Skip claims not found within the full text
    #                 continue
    #
    #             claim_end = claim_start + len(claim_text)
    #
    #             # Process each NER entity that falls within this claim's span
    #             for entity in summary_ner:
    #                 # Check if entity is within the claim's boundaries
    #                 if entity["start"] >= claim_start and entity["end"] <= claim_end:
    #                     # Map the entity type to our error type classification
    #                     entity_type = entity["entity_group"].upper()
    #
    #                     # Direct mappings for standard entity types
    #                     if entity_type == "PER":
    #                         mapped_type = CorrectnessErrorTypes.ENTITY_PERSON_SWAP
    #                     elif entity_type in ["LOC", "GPE"]:
    #                         mapped_type = CorrectnessErrorTypes.ENTITY_LOCATION_SWAP
    #                     elif entity_type == "ORG":
    #                         mapped_type = CorrectnessErrorTypes.ENTITY_ORGANIZATION_SWAP
    #                     elif entity_type == "TTL":
    #                         mapped_type = CorrectnessErrorTypes.ENTITY_TITLE_SWAP
    #                     elif entity_type == "EVE":
    #                         mapped_type = CorrectnessErrorTypes.ENTITY_EVENT_SWAP
    #                     else:
    #                         mapped_type = None
    #
    #                     if mapped_type and mapped_type in ner_based_transformation:
    #                         transformation_results.append({
    #                             "claim": claim_text,
    #                             "error_type": mapped_type.value,
    #                             "word": entity["word"]
    #                         })
    #
    # except Exception as e:
    #     print(f"Error in NER processing: {str(e)}")

    try:
        # Group NER entities by our error types
        entities_by_type = get_entities_by_type(text)

        # Process each claim to find matching entity swaps
        for claim_item in claim_decomposition.claims_decomposition:
            # Extract claim_text from whatever format claim_item is
            if isinstance(claim_item, dict):
                claim_text = next(iter(claim_item.values()))
            elif isinstance(claim_item, tuple):
                claim_text = claim_item[1]
            elif isinstance(claim_item, str):
                claim_text = claim_item
            else:
                continue

            # Only add swaps for entities that actually occur in this claim
            for mapped_type, words in entities_by_type.items():
                if mapped_type in ner_based_transformation:
                    for word in words:
                        if word in claim_text:
                            transformation_results.append({
                                "claim": claim_text,
                                "error_type": mapped_type.value,
                                "word": word
                            })

    except Exception as e:
        print(f"Error in NER processing: {str(e)}")

    # Static Number swap
    try:
        # Use the simplified number_swap_transformation function
        number_swap_results = suitable_number_swap_transformation(claim_decomposition)
        transformation_results.extend(number_swap_results)
    except Exception as e:
        print(f"Error in Number swap processing: {str(e)}")

    grouped_results = {}
    for result in transformation_results:
        claim = result["claim"]
        error_type = result["error_type"]
        word = result["word"]

        if claim not in grouped_results:
            grouped_results[claim] = []

        grouped_results[claim].append({
            "error_type": error_type,
            "word": word
        })

    # De-duplicate per‐claim by ‘word’: if a word appears multiple times, drop the first occurrence
    for claim, errors in grouped_results.items():
        counts = Counter(err["word"] for err in errors)
        seen = defaultdict(int)
        filtered = []
        for err in errors:
            w = err["word"]
            seen[w] += 1
            # keep if it’s the only one, or if it’s the 2nd+ occurrence
            if counts[w] == 1 or seen[w] > 1:
                filtered.append(err)
        grouped_results[claim] = filtered

    return grouped_results
    # Static


def apply_factuality_transformation_on_claims(source_article: str,
                                              transformation_results: Dict[str, List[Dict[str, Any]]]):
    """
    Apply transformations to claims based on the identified transformation types.

    Args:
        source_article: The original source article text
        transformation_results: The grouped results from all_suitable_transformation_possible

    Returns:
        A dictionary mapping original claims to their transformed versions
    """
    # hebrew_llm_transformations = {
    #     # CorrectnessErrorTypes.NOUN_GENDER_SWAP: [],
    #     # CorrectnessErrorTypes.VERB_GENDER_SWAP: [],
    #     # CorrectnessErrorTypes.VERB_TENSE_SWAP: [],
    #     # CorrectnessErrorTypes.DIRECTIONAL_PREPOSITION_SWAP: [],
    #     CorrectnessErrorTypes.PRONOUN_REFERENCE_ERRORS: [],
    #     CorrectnessErrorTypes.SPECIFICITY_SHIFT_ERRORS: [],
    #     CorrectnessErrorTypes.HOMOGRAPHIC_GENDER_ERRORS: [],
    #     CorrectnessErrorTypes.IMPERSONAL_TO_PERSONAL_VERB_ERRORS: []
    # }
    # general_llm_transformations = {
    #     CorrectnessErrorTypes.SENTENCE_NEGATION: [],
    #     CorrectnessErrorTypes.MEASURE_UNIT_SWAP: [],
    #     #CorrectnessErrorTypes.ENTITY_EVENT_SWAP: [],
    #     CorrectnessErrorTypes.ENTITY_DATE_SWAP: [],
    # }
    # # Extract named entities from the source article
    logger.info("Enter function apply_factuality_transformation_on_claims: perform factuality transformations on suitable claims founs")
    hebrew_llm_transformations_dict, general_llm_transformations_dict = get_enabled_transformations_as_dicts()
    source_entities_by_type = get_entities_by_type(source_article)

    # Convert sets back to lists for easier access
    for key in source_entities_by_type:
        source_entities_by_type[key] = list(source_entities_by_type[key])
    logger.info("Source entities by type: %s", {et.name: len(lst) for et, lst in source_entities_by_type.items()})

    transformed_claims = {}
    current_claim_versions = {}

    hebrew_transformation_types_list = list(hebrew_llm_transformations_dict.keys())
    general_transformation_types_list = list(general_llm_transformations_dict.keys())

    # Filter examples using the appropriate transformation types
    general_filtered_examples = {key: general_error_examples[key] for key in general_transformation_types_list if
                                 key in general_error_examples}
    hebrew_filtered_examples = {key: hebrew_error_examples[key] for key in hebrew_transformation_types_list if
                                key in hebrew_error_examples}

    # Process each claim and its transformations
    logger.info("Processing %d claims for transformation", len(transformation_results))
    logger.info("Phase 1 - map each claim to it's founded error type transformations")
    for original_claim, transformations in transformation_results.items():
        logger.info(f"Start transformation for the claim: {original_claim}")
        transformed_claims[original_claim] = []

        for transformation in transformations:

            error_type = transformation["error_type"]
            word_to_replace = transformation["word"]
            logger.info(f"Perform the transformation type of: {error_type} on word: {word_to_replace}")
            # Handle NER-based transformations
            if error_type in [
                CorrectnessErrorTypes.ENTITY_PERSON_SWAP,
                CorrectnessErrorTypes.ENTITY_LOCATION_SWAP,
                CorrectnessErrorTypes.ENTITY_ORGANIZATION_SWAP,
                CorrectnessErrorTypes.ENTITY_TITLE_SWAP,
                CorrectnessErrorTypes.ENTITY_EVENT_SWAP,
            ]:
                # Check if we have entities of this type in the source article
                if error_type in source_entities_by_type and source_entities_by_type[error_type]:
                    # Find entities in the source that are different from the one in the claim
                    replacement_candidates = [
                        e for e in source_entities_by_type[error_type]
                        if not is_same_entity(e, word_to_replace)
                    ]
                    logger.info(f"NER replacement candidate for error_type {error_type}: {replacement_candidates}")
                    if replacement_candidates:
                        # Choose a replacement entity (using first one for deterministic behavior)
                        # Could use random.choice(replacement_candidates) for variety
                        # replacement_entity = replacement_candidates[0]
                        max_attempts = 5
                        replacement_entity = None

                        for _ in range(max_attempts):
                            candidate = random.choice(replacement_candidates)
                            similarity = Levenshtein.ratio(candidate, word_to_replace)
                            if similarity < 0.57:
                                replacement_entity = candidate
                                break

                        if replacement_entity is None:
                            logger.info(f"NER error type of: {error_type}: didnt find replacement_entity with lower similarity, choose default {replacement_candidates[0]}")
                            replacement_entity = replacement_candidates[0]

                        current_claim = current_claim_versions.get(original_claim, original_claim)

                        pattern = r'\b' + re.escape(word_to_replace) + r'\b'
                        if re.search(pattern, current_claim):
                            transformed_claim = re.sub(pattern, replacement_entity, current_claim)
                            logger.info(f"Replace NER. replacement_entity: {replacement_entity}, word_to_replace: {word_to_replace}, error_type: {error_type}")
                            # Update the current version of this claim for future transformations
                            current_claim_versions[original_claim] = transformed_claim
                            transformed_claims[original_claim].append({
                                "transformed_claim": transformed_claim,
                                "error_type": error_type,
                                "original_entity": word_to_replace,
                                "replacement_entity": replacement_entity,
                                "original_claim": original_claim
                            })

            elif error_type == CorrectnessErrorTypes.NUMBER_SWAP:
                # Simple number transformation - replace with a different number
                original_number = word_to_replace

                # Extract numeric part and any suffix
                numeric_match = re.match(r'(\d+(?:,\d+)*(?:\.\d+)?)(.*)$', original_number)
                if numeric_match:
                    numeric_part = numeric_match.group(1)
                    suffix = numeric_match.group(2) or ""

                    # Remove commas for processing
                    clean_number = numeric_part.replace(',', '')

                    try:
                        if '.' in clean_number:
                            num_value = float(clean_number)
                            # For decimal numbers, multiply by 2
                            new_value = round(num_value * 2, 2)
                        else:
                            num_value = int(clean_number)
                            # For integers, multiply by 2
                            new_value = int(num_value * 2)

                        # Format the new number
                        if ',' in numeric_part:
                            new_number_str = f"{new_value:,}{suffix}"
                        else:
                            new_number_str = f"{new_value}{suffix}"

                        # Replace the number
                        pattern = r'\b' + re.escape(original_number) + r'\b'
                        if re.search(pattern, original_claim):
                            transformed_claim = re.sub(pattern, new_number_str, original_claim)

                            transformed_claims[original_claim].append({
                                "transformed_claim": transformed_claim,
                                "error_type": error_type,
                                "original_entity": original_number,
                                "replacement_entity": new_number_str,
                                "original_claim": original_claim
                            })
                    except ValueError:
                        # Skip if we can't convert to a number
                        pass
            # Collect LLM-based transformations to process in batches
            elif error_type in hebrew_llm_transformations_dict:
                hebrew_llm_transformations_dict[error_type].append({
                    "claim": original_claim,
                    "word": word_to_replace
                })

            elif error_type in general_llm_transformations_dict:
                general_llm_transformations_dict[error_type].append({
                    "claim": original_claim,
                    "word": word_to_replace
                })
    # Process Hebrew LLM transformations
    logger.info("Phase 2 - perform hebrew based transformation on claims")
    logger.info(f"HEB-QUEUE | types={len(hebrew_llm_transformations_dict)} | items={sum(len(v) for v in hebrew_llm_transformations_dict.values())}")
    
    # Parallel execution for Hebrew error types
    def process_hebrew_error_type(error_type, claims_list, hebrew_filtered_examples):
        try:
            suitable_claims_decomposition_list = []
            for claim_data in claims_list:
                suitable_claims_decomposition_list.append({
                    "original_claim": claim_data["claim"],
                    "correctness_error_type": error_type,
                    "applicable_transformation_text_within_claim": claim_data["word"]
                })
            
            logger.info(f"Perform summary_negative_augmentation on hebrew - error type of: {[error_type]}")
            logger.debug(f"HEB-LLM-CALL | type={error_type} | model=gemini | claims={len(suitable_claims_decomposition_list)} | examples={len(hebrew_filtered_examples.get(error_type, []))}")
            llm_result = summary_negative_augmentation(
                apply_transformation_types=[error_type],
                transformation_examples=hebrew_filtered_examples.get(error_type, []),
                suitable_claims_decomposition_list=suitable_claims_decomposition_list,
                unique_transformation_instructions="",
                model=workflow_config.apply_factuality_transformation_model_hebrew
            )
            logger.debug(f"HEB-LLM-RESULT | type={error_type} | ok={llm_result is not None} | pairs={len(getattr(llm_result, 'original_and_corrupted_transformed_claims', []))}")
            return error_type, llm_result
        except Exception as e:
            logger.error(f"Error in Hebrew LLM transformation {error_type}: {e}")
            return error_type, None

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for error_type, claims_list in hebrew_llm_transformations_dict.items():
            if claims_list:
                future = executor.submit(process_hebrew_error_type, error_type, claims_list, hebrew_filtered_examples)
                futures.append(future)
        
        for future in concurrent.futures.as_completed(futures):
            error_type, llm_result = future.result()
            if llm_result and hasattr(llm_result, "original_and_corrupted_transformed_claims"):
                for transformed_claim, original_claim in llm_result.original_and_corrupted_transformed_claims:
                    if original_claim in transformed_claims:
                        diffs = compute_all_diffs(original_claim, transformed_claim)
                        transformed_claims[original_claim].append({
                            "transformed_claim": transformed_claim,
                            "error_type": error_type,
                            "llm_generated": True,
                            "original_claim": original_claim,
                            "diffs": diffs
                        })
    # Process General LLM transformations
    logger.info("Phase 2 - perform general llm based transformation on claims")
    
    # Parallel execution for General error types
    def process_general_error_type_transform(error_type, claims_list, general_filtered_examples):
        try:
            suitable_claims_decomposition_list = []
            for claim_data in claims_list:
                suitable_claims_decomposition_list.append({
                    "original_claim": claim_data["claim"],
                    "correctness_error_type": error_type,
                    "applicable_transformation_text_within_claim": claim_data["word"]
                })
            
            logger.info(f"Perform summary_negative_augmentation on general - error type of: {[error_type]}")
            llm_result = summary_negative_augmentation(
                apply_transformation_types=[error_type],
                transformation_examples=general_filtered_examples.get(error_type, []),
                suitable_claims_decomposition_list=suitable_claims_decomposition_list,
                unique_transformation_instructions="",
                model=workflow_config.apply_factuality_transformation_model_general
            )
            return error_type, llm_result
        except Exception as e:
            logger.error(f"Error in General LLM transformation {error_type}: {e}")
            return error_type, None

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for error_type, claims_list in general_llm_transformations_dict.items():
            if claims_list:
                future = executor.submit(process_general_error_type_transform, error_type, claims_list, general_filtered_examples)
                futures.append(future)
        
        for future in concurrent.futures.as_completed(futures):
            error_type, llm_result = future.result()
            if llm_result and hasattr(llm_result, "original_and_corrupted_transformed_claims"):
                for transformed_claim, original_claim in llm_result.original_and_corrupted_transformed_claims:
                    if original_claim in transformed_claims:
                        diffs = compute_all_diffs(original_claim, transformed_claim)
                        transformed_claims[original_claim].append({
                            "transformed_claim": transformed_claim,
                            "error_type": error_type,
                            "llm_generated": True,
                            "original_claim": original_claim,
                            "diffs": diffs
                        })

    # Remove claims that the transformation failed
    for k, v in transformed_claims.items():
        new_v = []
        for t in v:
            if t.get("original_claim") != t.get("transformed_claim"):
                new_v.append(t)
            else:
                logger.info(f"Removed transformation for claim: '{k}' with error_type: {t.get('error_type')}")
        transformed_claims[k] = new_v
    # Remove claims with no transformations
    transformed_claims = {k: v for k, v in transformed_claims.items() if v}
    logger.info("Returning %d claims with transformations applied", len(transformed_claims))

    return transformed_claims


def apply_hallucinations_transformation_on_claims(claims_list: List[str],
                                                  transformation_results: Dict[str, List[Dict[str, Any]]]):
    """
    Apply hallucination transformations to claims - creates factually incorrect information
    that still sounds plausible in context.

    Args:
        claims_list: List of all claims from the summary
        transformation_results: The grouped results from all_suitable_transformation_possible

    Returns:
        A dictionary mapping original claims to their hallucinated versions
    """
    transformed_claims = {}

    # Define hallucination transformation types
    hallucination_ner_types = [
        CorrectnessErrorTypes.ENTITY_LOCATION_SWAP,
        CorrectnessErrorTypes.ENTITY_PERSON_SWAP,
        CorrectnessErrorTypes.ENTITY_ORGANIZATION_SWAP,
        CorrectnessErrorTypes.ENTITY_DATE_SWAP
    ]

    # Special case for number addition
    number_addition_type = CorrectnessErrorTypes.NUMBER_SWAP

    # Create dictionaries to collect claims for each type
    hallucination_ner_transformations = {error_type: [] for error_type in hallucination_ner_types}
    number_add_claims = []

    # Process each claim and its transformations
    for original_claim, transformations in transformation_results.items():
        transformed_claims[original_claim] = []

        for transformation in transformations:
            error_type = transformation["error_type"]
            word_to_replace = transformation["word"]

            # Collect NER-based hallucination transformations
            if error_type in hallucination_ner_types:
                hallucination_ner_transformations[error_type].append({
                    "claim": original_claim,
                    "word": word_to_replace
                })

            # Collect claims for potential number addition
            elif error_type == number_addition_type:
                if original_claim not in number_add_claims:
                    number_add_claims.append(original_claim)

    # Process each NER hallucination type
    for error_type, claims_list in hallucination_ner_transformations.items():
        if claims_list:
            try:
                # Prepare data for LLM transformation
                suitable_claims_decomposition_list = []

                for claim_data in claims_list:
                    suitable_claims_decomposition_list.append({
                        "original_claim": claim_data["claim"],
                        "correctness_error_type": error_type,
                        "applicable_transformation_text_within_claim": claim_data["word"]
                    })

                # Get transformation instructions based on error type
                unique_transformation_instructions = get_hallucination_instructions(error_type)

                # Call LLM for transformation
                llm_result = summary_halluzaition_negative_augmentation(
                    apply_transformation_types=[error_type],
                    transformation_examples={},  # Empty examples - hallucination is creative
                    suitable_claims_decomposition_list=suitable_claims_decomposition_list,
                    unique_transformation_instructions=unique_transformation_instructions
                )

                # Process results
                if hasattr(llm_result, "original_and_corrupted_transformed_claims"):
                    for transformed_claim, original_claim in llm_result.original_and_corrupted_transformed_claims:
                        if original_claim in transformed_claims:
                            transformed_claims[original_claim].append({
                                "transformed_claim": transformed_claim,
                                "error_type": error_type,
                                "llm_generated": True,
                                "original_claim": original_claim
                            })
            except Exception as e:
                print(f"Error in hallucination transformation {error_type}: {e}")

    # Process number addition if we have claims
    if number_add_claims:
        try:
            # Get transformation instructions for number addition
            unique_transformation_instructions = get_hallucination_instructions(number_addition_type)

            # Call LLM for number addition
            llm_result = summary_halluzaition_negative_augmentation(
                apply_transformation_types=[number_addition_type],
                transformation_examples={},
                suitable_claims_decomposition_list=[{"original_claim": claim} for claim in number_add_claims],
                unique_transformation_instructions=unique_transformation_instructions
            )

            # Process results
            if hasattr(llm_result, "original_and_corrupted_transformed_claims"):
                for transformed_claim, original_claim in llm_result.original_and_corrupted_transformed_claims:
                    if original_claim in transformed_claims:
                        transformed_claims[original_claim].append({
                            "transformed_claim": transformed_claim,
                            "error_type": number_addition_type,
                            "llm_generated": True,
                            "original_claim": original_claim
                        })
        except Exception as e:
            print(f"Error in number addition transformation: {e}")

    # Remove claims with no transformations
    transformed_claims = {k: v for k, v in transformed_claims.items() if v}

    return transformed_claims


# def apply_transformation_on_claims(source_article: str, transformation_results: Dict[str, List[Dict[str, Any]]]):
#     """
#     Apply transformations to claims based on the identified transformation types.
#
#     Args:
#         source_article: The original source article text
#         transformation_results: The grouped results from all_suitable_transformation_possible
#
#     Returns:
#         A dictionary mapping original claims to their transformed versions
#     """
#     # Extract named entities from the source article
#     source_ner = extract_named_entities(source_article)
#
#     # Group source entities by type and remove duplicates
#     source_entities_by_type = {}
#     for entity in source_ner:
#         entity_type = entity["entity_group"].upper()
#
#         # Map entity types to the CorrectnessErrorTypes
#         if entity_type == "PER":
#             mapped_type = CorrectnessErrorTypes.ENTITY_PERSON_SWAP
#         elif entity_type in ["LOC", "GPE"]:
#             mapped_type = CorrectnessErrorTypes.ENTITY_LOCATION_SWAP
#         elif entity_type == "ORG":
#             mapped_type = CorrectnessErrorTypes.ENTITY_ORGANIZATION_SWAP
#         elif entity_type == "TTL":
#             mapped_type = CorrectnessErrorTypes.ENTITY_TITLE_SWAP
#         elif entity_type == "EVE":
#             mapped_type = CorrectnessErrorTypes.ENTITY_EVENT_SWAP
#         else:
#             continue  # Skip unmapped entity types
#
#         if mapped_type not in source_entities_by_type:
#             source_entities_by_type[mapped_type] = set()
#
#         # Add to set to remove duplicates
#         source_entities_by_type[mapped_type].add(entity["word"])
#
#     # Convert sets back to lists for easier access
#     for key in source_entities_by_type:
#         source_entities_by_type[key] = list(source_entities_by_type[key])
#
#     transformed_claims = {}
#
#     # Process each claim and its transformations
#     for original_claim, transformations in transformation_results.items():
#         transformed_claims[original_claim] = []
#
#         for transformation in transformations:
#             error_type = transformation["error_type"]
#             word_to_replace = transformation["word"]
#
#             # Handle NER-based transformations
#             if error_type in [
#                 CorrectnessErrorTypes.ENTITY_PERSON_SWAP,
#                 CorrectnessErrorTypes.ENTITY_LOCATION_SWAP,
#                 CorrectnessErrorTypes.ENTITY_ORGANIZATION_SWAP,
#                 CorrectnessErrorTypes.ENTITY_TITLE_SWAP,
#                 CorrectnessErrorTypes.ENTITY_EVENT_SWAP,
#             ]:
#                 # Check if we have entities of this type in the source article
#                 if error_type in source_entities_by_type and source_entities_by_type[error_type]:
#                     # Find entities in the source that are different from the one in the claim
#                     replacement_candidates = [
#                         e for e in source_entities_by_type[error_type]
#                         if e.lower() != word_to_replace.lower()
#                     ]
#
#                     if replacement_candidates:
#                         # Choose a replacement entity (using first one for deterministic behavior)
#                         # Could use random.choice(replacement_candidates) for variety
#                         replacement_entity = replacement_candidates[0]
#
#                         # Create a transformed claim by replacing the entity
#                         # Use word boundaries in regex for more precise replacement
#                         pattern = r'\b' + re.escape(word_to_replace) + r'\b'
#                         if re.search(pattern, original_claim):
#                             transformed_claim = re.sub(pattern, replacement_entity, original_claim)
#
#                             transformed_claims[original_claim].append({
#                                 "transformed_claim": transformed_claim,
#                                 "error_type": error_type,
#                                 "original_entity": word_to_replace,
#                                 "replacement_entity": replacement_entity
#                             })
#
#     # Remove claims with no transformations
#     transformed_claims = {k: v for k, v in transformed_claims.items() if v}
#
#     return transformed_claims

def correctness_errors_transformation_workflow(summary: str, source_article: str,
                                               transformation_types: List[CorrectnessErrorTypes],
                                               amount_claims_to_apply_transformation=1):
    candidate_corrupted_augmented_summaries = []
    claim_decomposition = text_decomposition(text=summary, decomposition_method=CLAIM_DECOMPOSITION_METHOD)
    logger.info(f"Performed claim decomposition: {claim_decomposition}")

    logger.info(f"Start suitable_transformation_claims - find suitable transformation claims from the summary")
    suitable_transformation_claims = all_suitable_transformation_possible(summary, claim_decomposition)
    logger.info(f"Performed detecting on suitable transformation claims: {suitable_transformation_claims}")
    logger.info(f"Detected {len(suitable_transformation_claims)} suitable transformation allocations")
    if not suitable_transformation_claims:
        logger.warning("No suitable transformation claims found — nothing to transform.")

    claims_after_transformation = apply_factuality_transformation_on_claims(source_article, suitable_transformation_claims)
    logger.info(f"Claims after transformation: {len(claims_after_transformation)} claims modified")
    
    logger.info(f"About to call control_claims_after_transformation with {len(claims_after_transformation)} claims")
    logger.info(f"Percentage to transform: {PERCENTAGE_TRANSFORMATIONS_PERFORM_ON_NEGATIVE_SUMMARY}%")
    # Count total transformation variants
    total_variants = sum(len(transforms) for transforms in claims_after_transformation.values())
    logger.info(f"Total transformation variants across all claims: {total_variants}")
    logger.warning(f"This may generate up to 2^{len(claims_after_transformation)} combinations - could take a while!")
    
    transformed_claims_variations = control_claims_after_transformation(
        claims_after_transformation,
        percentage_of_correctness_errors_transformation=PERCENTAGE_TRANSFORMATIONS_PERFORM_ON_NEGATIVE_SUMMARY
    )
    logger.info(f"Generated {len(transformed_claims_variations)} transformed claim variations")
    if not transformed_claims_variations:
        logger.warning("No transformed claim variations were produced.")

    count = 0
    for s in transformed_claims_variations:
        logger.info(f"there are {len(transformed_claims_variations)} of transformed_claims_variations")
        logger.info(f"enter recompose_transformed_summary for claim variation {s}")

        summary_aug = recompose_transformed_summary(claim_decomposition, s)
        logger.info(f"complete recompose_transformed_summary for claim variation: {s} and summary_aug: {summary_aug}")
        logger.info(f"Appended summary_aug #{count + 1} to candidate_corrupted_augmented_summaries")
        candidate_corrupted_augmented_summaries.append(summary_aug)
        count += 1
        if count > 50:
            logger.warning("Reached 100 summary augmentations, breaking loop early.")
            break

    if len(claims_after_transformation) > 1 and len(candidate_corrupted_augmented_summaries) == 0:
        logger.info(
            f"Summary recompose return 0 corrupted summaries  but there is {len(claims_after_transformation)} amount of claims after transformation")
        return

    # # NEW: deterministic word validity pass (add-only)
    # try:
    #     from summary_augmentation.utils import validate_and_fix_candidate_summaries
    #     candidate_corrupted_augmented_summaries = validate_and_fix_candidate_summaries(
    #         candidate_corrupted_augmented_summaries,
    #         vocab_texts=[summary, source_article],
    #         enable_file_logging=True
    #     )
    #     logger.info(f"Applied word validity pass on {len(candidate_corrupted_augmented_summaries)} candidates")
    # except Exception as e:
    #     logger.error(f"Word validity pass failed: {e}")

    diverse = filter_diverse_summaries(
        golden=summary,
        cands=candidate_corrupted_augmented_summaries,
        max_same=MAXIMUM_ALLOWED_NEGATIVE_SUMMARIES_SIMILARITY_THRESHOLD
    )
    logger.info(f"After diversity filtering, {len(diverse)} summaries remain")

    if len(diverse) == 0:
        logger.info(f"After diversity filtering, {len(diverse)} summaries remain.")
        logger.info(
            f"Set the total summary without filtering, about {len(candidate_corrupted_augmented_summaries)} summaries.")
        final_summaries = candidate_corrupted_augmented_summaries
    else:
        logger.info(f"After diversity filtering, {len(diverse)} summaries remain.")
        final_summaries = diverse

    if len(final_summaries) > 7:
        return final_summaries[:7]
    else:
        logger.info(f"Total final augmented corrupted summaries {len(final_summaries)}")
        return final_summaries
    # return candidate_corrupted_augmented_summaries
    # sum = verify_transformed_summary_grammar(summary_after_augmentation['corrupted_summary'])

    # claims_after_transformation_hallu=apply_hallucinations_transformation_on_claims(claim_decomposition,suitable_transformation_claims)
    # logger.info(f"Performed transformation on suitable: {claims_after_transformation}")

    # number_swap = number_swap_transformation(summary_decomposition_claims=claim_decomposition,number_swap_transformation_amount=1)
    # claim_decomposition = summary_claim_decomposition(text=summary)
    # logger.info(f"Perform claim decomposition: {claim_decomposition}")


# def main():
#     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#     input_csv_path = r"C:\Users\amita\Downloads\mafat_800_article_150_summary.csv"
#     output_json = rf"C:\Users\amita\Downloads\augmented_dataset_{timestamp}.json"
#     output_csv = rf"C:\Users\amita\Downloads\augmented_dataset_{timestamp}.csv"
#     num_rows = 13
#     transform_types = [
#         CorrectnessErrorTypes.ENTITY_PERSON_SWAP,
#         CorrectnessErrorTypes.ENTITY_ORGANIZATION_SWAP,
#         CorrectnessErrorTypes.ENTITY_LOCATION_SWAP,
#         CorrectnessErrorTypes.ENTITY_DATE_SWAP,
#         CorrectnessErrorTypes.ENTITY_EVENT_SWAP,
#         CorrectnessErrorTypes.ENTITY_TITLE_SWAP
#     ]
#
#     df = pd.read_csv(input_csv_path)
#     sample = df.sample(n=min(num_rows, len(df)))
#
#     # Prepare thread-safe storage
#     all_records = []
#     lock = threading.Lock()
#
#     # JSON file (we'll write a JSON array manually)
#     json_file = open(output_json, 'w', encoding='utf-8')
#     json_file.write('[\n')
#
#     def worker(idx, row):
#         source = row['hypothesis']
#         gold = row['premise']
#         corrupted_list = correctness_errors_transformation_workflow(
#             summary=gold,
#             source_article=source,
#             transformation_types=transform_types
#         )
#         logger.info(f"After full correctness_errors_transformation_workflow returned {len(corrupted_list)} summaries")
#         record = {
#             'source_article': source,
#             'gold_summary': gold,
#             'correctness_corrupted_summaries': corrupted_list
#         }
#
#         text = json.dumps(record, ensure_ascii=False)
#         with lock:
#             # Add to our record list for CSV processing later
#             all_records.append(record)
#
#             # JSON: comma-prefix after the first item
#             if idx > 0:
#                 json_file.write(',\n')
#             json_file.write(text)
#
#     # Launch one thread per row
#     with ThreadPoolExecutor() as exec:
#         futures = []
#         for idx, (_, row) in enumerate(sample.iterrows()):
#             futures.append(exec.submit(worker, idx, row))
#
#         # Wait for all threads to complete
#         for future in futures:
#             future.result()
#
#     # Finish JSON array and close file
#     json_file.write('\n]\n')
#     json_file.close()
#
#     # Now process the CSV with our specialized function after all threads are done
#     save_dataset_csv(all_records, output_csv)
#
#     print(f"Written {len(sample)} examples to:\n  JSON: {output_json}\n  CSV:  {output_csv}")

#### good main !!
# def main():
#     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#     input_csv_path  = r"C:\Users\amita\Downloads\mafat_800_article_150_summary.csv"
#     output_json = rf"C:\Users\amita\Downloads\augmented_dataset_{timestamp}.json"
#     output_csv = rf"C:\Users\amita\Downloads\augmented_dataset_{timestamp}.csv"
#     num_rows = 13
#     transform_types = [
#         CorrectnessErrorTypes.ENTITY_PERSON_SWAP,
#         CorrectnessErrorTypes.ENTITY_ORGANIZATION_SWAP,
#         CorrectnessErrorTypes.ENTITY_LOCATION_SWAP,
#         CorrectnessErrorTypes.ENTITY_DATE_SWAP,
#         CorrectnessErrorTypes.ENTITY_EVENT_SWAP,
#         CorrectnessErrorTypes.ENTITY_TITLE_SWAP
#     ]
#
#     df = pd.read_csv(input_csv_path)
#     sample = df.sample(n=min(num_rows, len(df)))
#
#     all_records = []
#     for _, row in sample.iterrows():
#         # note: you had premise/hypothesis reversed
#         source = row['hypothesis']
#         gold   = row['premise']
#
#         corrupted_list = correctness_errors_transformation_workflow(
#             summary=gold,
#             source_article=source,
#             transformation_types=transform_types
#         )
#
#         logger.info(f"After full correctness_errors_transformation_workflow returned {len(corrupted_list)} summaries")
#         all_records.append({
#             'source_article': source,
#             'gold_summary': gold,
#             'correctness_corrupted_summaries': corrupted_list
#         })
#
#     # save both files
#     save_dataset_json(all_records, output_json)
#     save_dataset_csv(all_records,  output_csv)
#
#     print(f"Written {len(all_records)} examples to:\n  JSON: {output_json}\n  CSV:  {output_csv}")
#### good main !!
#
# if __name__ == "__main__":
#     main()

def process_row(row, transform_types):
    """Worker function to process a single row."""
    thread_id = threading.current_thread().name
    try:
        # note: premise/hypothesis as in original code
        source = row['premise']
        gold = row['hypothesis']

        corrupted_list = correctness_errors_transformation_workflow(
            summary=gold,
            source_article=source,
            transformation_types=transform_types
        )

        logger.info(f"Thread {thread_id}: generated {len(corrupted_list)} corrupted summaries")
        return {
            'source_article': source,
            'gold_summary': gold,
            'correctness_corrupted_summaries': corrupted_list,
            'len_group': row['len_group'],
            'doc_id': row['doc_id'],
            'source': row['source']

        }
    except Exception as e:
        logger.error(f"Thread {thread_id}: Error processing row: {str(e)}")
        return None


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s'
    )
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"augmentation_log_{timestamp}.log"

    # Reconfigure logger with file output
    global logger
    logger = setup_logger(__name__, log_level="INFO", log_file=log_file)
    # Hardcoded configuration
    max_workers = 3
    start_row = 51
    end_row = 65

    logger.info(f"Starting processing with {max_workers} worker threads")
    logger.info(f"Processing rows {start_row} to {end_row-1}")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_type = "validation"
    # input_csv_path = r"C:\Users\amita\Downloads\correctness metric data\All dataset split\mafat\train\train.csv"
    # output_json = rf"C:\Users\amita\Downloads\correctness metric data\All dataset split\mafat\train\mafat_train_{timestamp}_start_row_{start_row}_end_row_{end_row}.json"
    # output_csv = rf"C:\Users\amita\Downloads\correctness metric data\All dataset split\mafat\train\mafat_train_{timestamp}_start_row_{start_row}_end_row_{end_row}.csv"
    input_csv_path = rf"C:\Users\amita\Downloads\correctness metric data\All dataset split\mafat\{dataset_type}\{dataset_type}.csv"
    output_json = rf"C:\Users\amita\Downloads\correctness metric data\All dataset split\mafat\{dataset_type}\mafat_{dataset_type}_{timestamp}_start_row_{start_row}_end_row_{end_row}.json"
    output_csv = rf"C:\Users\amita\Downloads\correctness metric data\All dataset split\mafat\{dataset_type}\mafat_{dataset_type}_{timestamp}_start_row_{start_row}_end_row_{end_row}.csv"



    transform_types = [
        CorrectnessErrorTypes.ENTITY_PERSON_SWAP,
        CorrectnessErrorTypes.ENTITY_ORGANIZATION_SWAP,
        CorrectnessErrorTypes.ENTITY_LOCATION_SWAP,
        CorrectnessErrorTypes.ENTITY_DATE_SWAP,
        CorrectnessErrorTypes.ENTITY_EVENT_SWAP,
        CorrectnessErrorTypes.ENTITY_TITLE_SWAP
    ]

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s'
    )

    logger.info(f"Starting processing with {max_workers} worker threads")

    try:
        # Read the CSV and slice rows [start_row:end_row)
        df = pd.read_csv(input_csv_path)
        slice_df = df.iloc[start_row:end_row]
        total_rows = len(slice_df)

        # Thread-safe file writing locks
        json_lock = threading.Lock()
        csv_lock = threading.Lock()

        # Progress tracking
        completed = 0
        failed = 0
        progress_lock = threading.Lock()

        def update_progress(success=True):
            nonlocal completed, failed
            with progress_lock:
                completed += 1
                if not success:
                    failed += 1
                progress = (completed / total_rows) * 100
                sys.stdout.write(f"\rProgress: {progress:.2f}% ({completed}/{total_rows}) - Failures: {failed}")
                sys.stdout.flush()

        print(f"Processing rows {start_row} to {end_row - 1} ({total_rows} total) with {max_workers} worker threads")

        # Initialize JSON file with opening bracket
        with open(output_json, 'w', encoding='utf-8') as json_file:
            json_file.write('[\n')

        # Initialize CSV file with headers
        csv_initialized = False

        # Use ThreadPoolExecutor to manage threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_row, row, transform_types)
                for _, row in slice_df.iterrows()
            ]

            record_count = 0
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        logger.info(f"✓ Sample augmentation completed for doc_id={result.get('doc_id', 'unknown')} | corrupted_summaries={len(result.get('correctness_corrupted_summaries', []))}")
                        # Write JSON incrementally (thread-safe)
                        with json_lock:
                            with open(output_json, 'a', encoding='utf-8') as json_file:
                                if record_count > 0:
                                    json_file.write(',\n')
                                json_text = json.dumps(result, ensure_ascii=False, indent=2)
                                json_file.write(json_text)
                            record_count += 1

                        # Write CSV incrementally (thread-safe)
                        with csv_lock:
                            # Convert record to CSV rows
                            rows = []
                            # Entailment pair
                            rows.append({
                                'premise': result['source_article'],
                                'hypothesis': result['gold_summary'],
                                'label': 'entailment',
                                'len_group': result['len_group'],
                                'doc_id': result['doc_id'],
                                'source': result['source'],
                                'error_types': ''
                            })
                            # Contradiction pairs
                            for corrupt in result.get('correctness_corrupted_summaries', []):
                                ets = []
                                for t in corrupt.get('transformation_list', []):
                                    for at in t.get('applied_transformations', []):
                                        ty = at.get('type')
                                        if isinstance(ty, CorrectnessErrorTypes):
                                            ets.append(ty.value)
                                        elif isinstance(ty, str):
                                            ets.append(ty)
                                rows.append({
                                    'premise': result['source_article'],
                                    'hypothesis': corrupt['corrupted_summary'],
                                    'label': 'contradiction',
                                    'len_group': result['len_group'],
                                    'doc_id': result['doc_id'],
                                    'source': result['source'],
                                    'error_types': ','.join(set(ets))
                                })
                            
                            # Write to CSV (append mode after first write)
                            df_chunk = pd.DataFrame(rows)
                            if not csv_initialized:
                                df_chunk.to_csv(output_csv, mode='w', index=False, encoding='utf-8')
                                csv_initialized = True
                            else:
                                df_chunk.to_csv(output_csv, mode='a', index=False, header=False, encoding='utf-8')

                        update_progress(success=True)
                        logger.info(f"Saved record {record_count} to disk")
                    else:
                        update_progress(success=False)
                except Exception as e:
                    logger.error(f"Error getting task result: {e}")
                    update_progress(success=False)

        # Close JSON array
        with open(output_json, 'a', encoding='utf-8') as json_file:
            json_file.write('\n]\n')

        print("\nProcessing completed")
        print(f"Written {record_count} examples to:\n  JSON: {output_json}\n  CSV: {output_csv}")

    except Exception as e:
        logger.error(f"Error in main processing: {e}")
        raise


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"Unhandled exception: {str(e)}")
# def main():
#     # input_csv_path = r"/sise/home/avramit/dataset_augmentation/output_mafat_less_500_test_new.csv"
#     #input_csv_path = r"/sise/home/avramit/mfact-summ/mfact/datasets/heb/mafat_800_article_150_summary.csv"
#     input_csv_path = r"C:\Users\amita\Downloads\mafat_800_article_150_summary.csv"
#     output_csv_path = r"/sise/home/avramit/dataset_augmentation/output_mafat_less_500_test_negative.csv"
#     num_rows_to_transform = 20
#     transformation_types_1 = [CorrectnessErrorTypes.ENTITY_PERSON_SWAP, CorrectnessErrorTypes.ENTITY_ORGANIZATION_SWAP,
#                               CorrectnessErrorTypes.ENTITY_LOCATION_SWAP,CorrectnessErrorTypes.ENTITY_DATE_SWAP, CorrectnessErrorTypes.ENTITY_EVENT_SWAP,
#                               CorrectnessErrorTypes.ENTITY_TITLE_SWAP]
#
#     # Load the CSV data
#     df = pd.read_csv(input_csv_path)
#     # Take the first n rows to transform
#     if num_rows_to_transform < len(df):
#         rows_to_transform = df.sample(n=num_rows_to_transform)
#     else:
#         rows_to_transform = df  # Use all rows if requested more than available
#     # Create a list to store new rows
#     new_rows = []
#
#     # Process each selected row
#     for _, row in rows_to_transform.iterrows():
#         # summary = row['hypothesis']
#         # premise = row['premise']
#         premise = row['hypothesis']
#         summary = row['premise']
#         new_sum=correctness_errors_transformation_workflow(summary=summary, source_article=premise, transformation_types=transformation_types_1)
#
#     # Create DataFrame with new rows
#     new_df = pd.DataFrame(new_rows)
#
#     # Append new rows to original DataFrame and save
#     result_df = pd.concat([df, new_df], ignore_index=True)
#     result_df.to_csv(output_csv_path, index=False)
#
#     print(f"Successfully processed {len(new_rows)} rows and saved to {output_csv_path}")
#     return result_df


# Example of how to call the main function:
# if __name__ == "__main__":
#     print("hello")
#     main()
