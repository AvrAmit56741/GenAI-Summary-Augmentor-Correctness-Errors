# -*- coding: utf-8 -*-

from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers.retry import RetryWithErrorOutputParser
from typing import List, Tuple
from typing import List, Dict
import logging
from typing import List
import pandas as pd
from pydantic import BaseModel, PrivateAttr, Field
from typing import List, Any
from enum import Enum
import re
import random
from typing import Dict, List, Any, Tuple

# Configure the logger (only once, typically at the top of your script or entry point)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import json
import summary_augmentation.correctness_errors_examples as correctness_errors_examples
from openai import OpenAI
# Define OpenAI API credentials
#import google.generativeai as genai
from langchain_openai import ChatOpenAI


from summary_augmentation.correctness_errors_prompts import suitable_claim_decomposition_prompt, \
    claims_decomposition_prompt, apply_transformation_on_claims_prompt
from summary_augmentation.utils import extract_named_entities, CorrectnessErrorTypes, text_decomposition, ClaimDecompositionMethod, number_swap_transformation, suitable_number_swap_transformation, recompose_transformed_summary
# Define the main data structure
from summary_augmentation.correctness_errors_prompts import claims_decomposition_prompt
from summary_augmentation.correctness_errors_llm_helper_functions import choose_suitable_claim, summary_negative_augmentation, summary_halluzaition_negative_augmentation
import re


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

INSTRUCTION_TEMPLATES = {
    CorrectnessErrorTypes.ENTITY_PERSON_SWAP: (
        "For the 'entity_person_swap' error transformation type, you can perform word swaps "
        "only from the given person list: {entities}."
    ),
    CorrectnessErrorTypes.ENTITY_LOCATION_SWAP: (
        "For the 'entity_location_swap' error transformation type, consider swapping locations "
        "from the following list: {entities}."
    ),
    CorrectnessErrorTypes.ENTITY_ORGANIZATION_SWAP: (
        "For the 'entity_organization_swap' error transformation type, you can replace organization "
        "names using the provided list: {entities}."
    ),
    # Add other instruction templates as needed
}


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

def ner_instructions(mapped_ner_results: Dict[CorrectnessErrorTypes, List[str]]) -> str:
    # Instruction templates for each error type
    instructions = []

    for error_type, entities in mapped_ner_results.items():
        template = INSTRUCTION_TEMPLATES.get(error_type)
        if template:
            instruction = template.format(entities=", ".join(entities))
            instructions.append(instruction)

    return "\n".join(instructions)

def all_suitable_transformation_possible(text:str, claim_decomposition):
    hebrew_llm_transformations = [
        CorrectnessErrorTypes.NOUN_GENDER_SWAP,
        CorrectnessErrorTypes.VERB_GENDER_SWAP,
        CorrectnessErrorTypes.VERB_TENSE_SWAP,
        CorrectnessErrorTypes.DIRECTIONAL_PREPOSITION_SWAP
    ]

    general_llm_transformations = [
        CorrectnessErrorTypes.SENTENCE_NEGATION,
        CorrectnessErrorTypes.MEASURE_UNIT_SWAP,
        CorrectnessErrorTypes.ENTITY_EVENT_SWAP,
        CorrectnessErrorTypes.ENTITY_DATE_SWAP
    ]

    ner_based_transformation = [
        CorrectnessErrorTypes.ENTITY_PERSON_SWAP,
        CorrectnessErrorTypes.ENTITY_LOCATION_SWAP,
        CorrectnessErrorTypes.ENTITY_TITLE_SWAP,
        CorrectnessErrorTypes.ENTITY_ORGANIZATION_SWAP
    ]

    static_based_transformation = [CorrectnessErrorTypes.NUMBER_SWAP]

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
        hebrew_claim_results = choose_suitable_claim(claim_decomposition, hebrew_transformation_types_list,
                                                     hebrew_filtered_examples)
        # Check if the result has the attribute 'suitable_claims' and process each entry.
        if hebrew_claim_results and hasattr(hebrew_claim_results, "suitable_claims"):
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
        for err_type in general_llm_transformations:
            # build a single‐type examples dict (skip if no examples)
            examples = {}
            if err_type in general_filtered_examples:
                examples = { err_type: general_filtered_examples[err_type]}
            else:
                continue

            # run the prompt for just this one error type
            result = choose_suitable_claim(
                claim_decomposition,
                [err_type],         # only this type
                examples            # only its examples
            )

            # collect any suitable_claims
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
    try:
        # Get NER entities with their positions
        summary_ner = extract_named_entities(text)
        if summary_ner:
            # Process each claim to find its span in the full text
            for claim_item in claim_decomposition.claims_decomposition:
                # Extract the claim text based on the data structure
                # It could be a dictionary with a numeric key, or a tuple with (index, text)
                if isinstance(claim_item, dict):
                    # Handle dictionary format
                    for _, claim_text in claim_item.items():
                        break  # Just get the first (and only) value
                elif isinstance(claim_item, tuple):
                    # Handle tuple format (index, text)
                    claim_text = claim_item[1]
                elif isinstance(claim_item, str):
                    # Handle string format directly
                    claim_text = claim_item
                else:
                    # Skip unknown formats
                    continue

                # Find where this claim appears in the full text
                claim_start = text.find(claim_text)
                if claim_start == -1:
                    # Skip claims not found within the full text
                    continue

                claim_end = claim_start + len(claim_text)

                # Process each NER entity that falls within this claim's span
                for entity in summary_ner:
                    # Check if entity is within the claim's boundaries
                    if entity["start"] >= claim_start and entity["end"] <= claim_end:
                        # Map the entity type to our error type classification
                        entity_type = entity["entity_group"].upper()

                        # Direct mappings for standard entity types
                        if entity_type == "PER":
                            mapped_type = CorrectnessErrorTypes.ENTITY_PERSON_SWAP
                        elif entity_type in ["LOC", "GPE"]:
                            mapped_type = CorrectnessErrorTypes.ENTITY_LOCATION_SWAP
                        elif entity_type == "ORG":
                            mapped_type = CorrectnessErrorTypes.ENTITY_ORGANIZATION_SWAP
                        elif entity_type == "TTL":
                            mapped_type = CorrectnessErrorTypes.ENTITY_TITLE_SWAP
                        elif entity_type == "EVE":
                            mapped_type = CorrectnessErrorTypes.ENTITY_EVENT_SWAP
                        else:
                            mapped_type = None

                        if mapped_type and mapped_type in ner_based_transformation:
                            transformation_results.append({
                                "claim": claim_text,
                                "error_type": mapped_type.value,
                                "word": entity["word"]
                            })

    except Exception as e:
        print(f"Error in NER processing: {str(e)}")

    # Static-based transformation (Number swap)
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

    return grouped_results
    # Static


def apply_factuality_transformation_on_claims(source_article: str,
                                   transformation_results: Dict[str, List[Dict[str, Any]]]):
    """
    Apply transformations to claims based on the identified transformation types.

    Args:
        source_article: The original source article text
        transformation_results: The grouped results from all_suitable_transformation_possible
        general_error_examples: Dictionary of general error examples
        hebrew_error_examples: Dictionary of Hebrew-specific error examples

    Returns:
        A dictionary mapping original claims to their transformed versions
    """
    # Extract named entities from the source article
    source_ner = extract_named_entities(source_article)

    # Group source entities by type and remove duplicates
    source_entities_by_type = {}
    for entity in source_ner:
        entity_type = entity["entity_group"].upper()

        # Map entity types to the CorrectnessErrorTypes
        if entity_type == "PER":
            mapped_type = CorrectnessErrorTypes.ENTITY_PERSON_SWAP
        elif entity_type in ["LOC", "GPE"]:
            mapped_type = CorrectnessErrorTypes.ENTITY_LOCATION_SWAP
        elif entity_type == "ORG":
            mapped_type = CorrectnessErrorTypes.ENTITY_ORGANIZATION_SWAP
        elif entity_type == "TTL":
            mapped_type = CorrectnessErrorTypes.ENTITY_TITLE_SWAP
        elif entity_type == "EVE":
            mapped_type = CorrectnessErrorTypes.ENTITY_EVENT_SWAP
        else:
            continue  # Skip unmapped entity types

        if mapped_type not in source_entities_by_type:
            source_entities_by_type[mapped_type] = set()

        # Add to set to remove duplicates
        source_entities_by_type[mapped_type].add(entity["word"])

    # Convert sets back to lists for easier access
    for key in source_entities_by_type:
        source_entities_by_type[key] = list(source_entities_by_type[key])

    transformed_claims = {}
    current_claim_versions = {}

    # Organize claims by transformation type for LLM processing
    hebrew_llm_transformations = {
        CorrectnessErrorTypes.NOUN_GENDER_SWAP: [],
        CorrectnessErrorTypes.VERB_GENDER_SWAP: [],
        CorrectnessErrorTypes.VERB_TENSE_SWAP: []
    }

    general_llm_transformations = {
        CorrectnessErrorTypes.SENTENCE_NEGATION: [],
        CorrectnessErrorTypes.MEASURE_UNIT_SWAP: [],
        CorrectnessErrorTypes.ENTITY_EVENT_SWAP: [],
        CorrectnessErrorTypes.ENTITY_DATE_SWAP: []
    }

    hebrew_transformation_types_list = list(hebrew_llm_transformations.keys())
    general_transformation_types_list = list(general_llm_transformations.keys())

    # Filter examples using the appropriate transformation types
    general_filtered_examples = {key: general_error_examples[key] for key in general_transformation_types_list if
                                 key in general_error_examples}
    hebrew_filtered_examples = {key: hebrew_error_examples[key] for key in hebrew_transformation_types_list if
                                key in hebrew_error_examples}
    # Process each claim and its transformations
    for original_claim, transformations in transformation_results.items():
        transformed_claims[original_claim] = []

        for transformation in transformations:
            error_type = transformation["error_type"]
            word_to_replace = transformation["word"]

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
                        if e.lower() != word_to_replace.lower()
                    ]

                    if replacement_candidates:
                        # Choose a replacement entity (using first one for deterministic behavior)
                        # Could use random.choice(replacement_candidates) for variety
                        replacement_entity = replacement_candidates[0]

                        current_claim = current_claim_versions.get(original_claim, original_claim)

                        pattern = r'\b' + re.escape(word_to_replace) + r'\b'
                        if re.search(pattern, current_claim):
                            transformed_claim = re.sub(pattern, replacement_entity, current_claim)
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
            elif error_type in hebrew_llm_transformations:
                hebrew_llm_transformations[error_type].append({
                    "claim": original_claim,
                    "word": word_to_replace
                })

            elif error_type in general_llm_transformations:
                general_llm_transformations[error_type].append({
                    "claim": original_claim,
                    "word": word_to_replace
                })

    # Process Hebrew LLM transformations
    for error_type, claims_list in hebrew_llm_transformations.items():
        if claims_list:
            try:
                # Prepare data for LLM transformation
                suitable_claims_decomposition_list = []
                claims_decomposition_list = []

                for claim_data in claims_list:
                    suitable_claims_decomposition_list.append({
                        "original_claim": claim_data["claim"],
                        "correctness_error_type": error_type,
                        "applicable_transformation_text_within_claim": claim_data["word"]
                    })
                    #claims_decomposition_list.append(claim_data["claim"])

                # Get transformation instructions based on error type
                unique_transformation_instructions = get_transformation_instructions(error_type)

                # Call LLM for transformation
                llm_result = summary_negative_augmentation(
                    apply_transformation_types=[error_type],
                    transformation_examples=hebrew_filtered_examples,
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
                print(f"Error in Hebrew LLM transformation {error_type}: {e}")

    # Process General LLM transformations
    for error_type, claims_list in general_llm_transformations.items():
        if claims_list:
            try:
                # Prepare data for LLM transformation
                suitable_claims_decomposition_list = []
                claims_decomposition_list = []

                for claim_data in claims_list:
                    suitable_claims_decomposition_list.append({
                        "original_claim": claim_data["claim"],
                        "correctness_error_type": error_type,
                        "applicable_transformation_text_within_claim": claim_data["word"]
                    })
                    #claims_decomposition_list.append(claim_data["claim"])


                # Get transformation instructions based on error type
                unique_transformation_instructions = get_transformation_instructions(error_type)

                # Call LLM for transformation
                llm_result = summary_negative_augmentation(
                    apply_transformation_types=[error_type],
                    transformation_examples=general_filtered_examples,
                    suitable_claims_decomposition_list=suitable_claims_decomposition_list,
                    #claims_decomposition_list=claims_decomposition_list,
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
                print(f"Error in General LLM transformation {error_type}: {e}")

    # Remove claims with no transformations
    transformed_claims = {k: v for k, v in transformed_claims.items() if v}

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


def get_hallucination_instructions(error_type):
    """Helper function to get hallucination instructions based on error type"""
    # Define instructions for each hallucination type
    instructions = {
        CorrectnessErrorTypes.ENTITY_PERSON_REPLACE: "Replace the person entity with a different plausible person that would make sense in context but is factually incorrect",
        CorrectnessErrorTypes.ENTITY_LOCATION_REPLACE: "Replace the location entity with a different plausible location that would make sense in context but is factually incorrect",
        CorrectnessErrorTypes.ENTITY_ORGANIZATION_REPLACE: "Replace the organization entity with a different plausible organization that would make sense in context but is factually incorrect",
        CorrectnessErrorTypes.ENTITY_DATE_REPLACE: "Replace the date entity with a different plausible date that would make sense in context but is factually incorrect",
        CorrectnessErrorTypes.NUMBER_ADD: "Add a specific, plausible number to the claim where it would naturally fit (like quantities, measurements, or statistics)"
    }

    return instructions.get(error_type,
                            "Apply a hallucination transformation that is contextually plausible but factually incorrect")

def get_transformation_instructions(error_type):
    """Helper function to get transformation instructions based on error type"""
    # Define instructions for each error type
    instructions = {
        CorrectnessErrorTypes.SENTENCE_NEGATION: "Negate the meaning of the sentence",
        CorrectnessErrorTypes.MEASURE_UNIT_SWAP: "Replace the measurement unit with a different one",
        CorrectnessErrorTypes.NOUN_GENDER_SWAP: "Change the gender of the noun",
        CorrectnessErrorTypes.VERB_TENSE_SWAP: "Change the tense of the verb",
        CorrectnessErrorTypes.VERB_GENDER_SWAP: "Change the gender of the verb",
        CorrectnessErrorTypes.ENTITY_EVENT_SWAP: "Replace the event with a different event",
        CorrectnessErrorTypes.ENTITY_DATE_SWAP: "Replace the date with a different date"
    }

    return instructions.get(error_type, "Apply the transformation")
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
def correctness_errors_transformation_workflow(summary: str, source_article: str, transformation_types: List[CorrectnessErrorTypes], amount_claims_to_apply_transformation=1):

    claim_decomposition = text_decomposition(text=summary,decomposition_method=ClaimDecompositionMethod.SENTENCES)
    logger.info(f"Performed claim decomposition: {claim_decomposition}")


    suitable_transformation_claims = all_suitable_transformation_possible(summary,claim_decomposition)
    logger.info(f"Performed detecting on suitable transformation claims: {suitable_transformation_claims}")

    claims_after_transformation = apply_factuality_transformation_on_claims(source_article,suitable_transformation_claims)
    #claims_after_transformation_hallu=apply_hallucinations_transformation_on_claims(claim_decomposition,suitable_transformation_claims)
    #logger.info(f"Performed transformation on suitable: {claims_after_transformation}")

    #number_swap = number_swap_transformation(summary_decomposition_claims=claim_decomposition,number_swap_transformation_amount=1)
    summary_after_augmentation=recompose_transformed_summary(claim_decomposition,claims_after_transformation)
    print(summary_after_augmentation)
    return summary_after_augmentation
    #claim_decomposition = summary_claim_decomposition(text=summary)
    # logger.info(f"Perform claim decomposition: {claim_decomposition}")
    # choose_suitable_claims = choose_suitable_claim(claim_decomposition, transformation_types_list, filtered_examples)
    # summary_ner = extract_named_entities(text=summary)
    # logger.info(f"Chosen suitable claims: {choose_suitable_claims}")
    #
    # if choose_suitable_claims.suitable_claims_decomposition:
    #     article_ner = extract_named_entities(text=source_article)
    #     summary_ner = extract_named_entities(text=summary)
    #     logger.info(f"Extracted NER: {article_ner}")
    #
    #     mapped_error_types = map_ner_to_error_types(article_ner, transformation_types)
    #
    #     # Generate the prompt instructions based on the mapped NER results
    #     prompt_instructions = ner_instructions(mapped_error_types)
    #
    #     transformed_claims = summary_negative_augmentation(transformation_types_list, filtered_examples,
    #                                                        choose_suitable_claims, claim_decomposition,
    #                                                        prompt_instructions)
    #     logger.info(f"Transformed claims: {transformed_claims}")
    #
    #     #Apply negative augmentation
    #     # if len(choose_suitable_claims.suitable_claims_decomposition) < amount_claims_to_apply_transformation:
    #     #     logger.info(f"Suitable  claims: {transformed_claims}")
    #     # result = summary_negative_augmentation(
    #     #     apply_transformation_types=transformation_types,
    #     #     transformation_examples=filtered_examples,
    #     #     amount_claims_to_apply_transformation=5,
    #     #     original_summary=summary,
    #     #     summary_claims=claim_decomposition.claims_decomposition
    #     # )

    # else:
    #     return


# Add this as the main function at the end of your file

def main():
    # input_csv_path = r"/sise/home/avramit/dataset_augmentation/output_mafat_less_500_test_new.csv"
    input_csv_path = r"/sise/home/avramit/mfact-summ/mfact/datasets/heb/mafat_800_article_150_summary.csv"
    #input_csv_path = r"C:\Users\amita\Downloads\mafat_800_article_150_summary.csv"
    output_csv_path = r"/sise/home/avramit/dataset_augmentation/output_mafat_less_500_test_negative.csv"
    num_rows_to_transform = 20
    transformation_types_1 = [CorrectnessErrorTypes.ENTITY_PERSON_SWAP, CorrectnessErrorTypes.ENTITY_ORGANIZATION_SWAP,
                              CorrectnessErrorTypes.ENTITY_LOCATION_SWAP,CorrectnessErrorTypes.ENTITY_DATE_SWAP, CorrectnessErrorTypes.ENTITY_EVENT_SWAP,
                              CorrectnessErrorTypes.ENTITY_TITLE_SWAP]

    # Load the CSV data
    df = pd.read_csv(input_csv_path)
    # Take the first n rows to transform
    if num_rows_to_transform < len(df):
        rows_to_transform = df.sample(n=num_rows_to_transform)
    else:
        rows_to_transform = df  # Use all rows if requested more than available
    # Create a list to store new rows
    new_rows = []

    # Process each selected row
    for _, row in rows_to_transform.iterrows():
        # summary = row['hypothesis']
        # premise = row['premise']
        premise = row['hypothesis']
        summary = row['premise']
        new_sum=correctness_errors_transformation_workflow(summary=summary, source_article=premise, transformation_types=transformation_types_1)
        # correctness_errors_transformation_workflow(summary=summary, source_article=premise,
        #                                            transformation_types=transformation_types_2)
        # correctness_errors_transformation_workflow(summary=summary, source_article=premise,
        #                                            transformation_types=transformation_types_3)


        # Apply negative augmentation
        # result = summary_negative_augmentation(
        #     apply_transformation_types=transformation_types,
        #     transformation_examples=filtered_examples,
        #     amount_claims_to_apply_transformation=5,
        #     original_summary=summary,
        #     summary_claims=claim_decomposition.claims_decomposition
        # )

        # Create new row with the transformed data
        # new_row = {
        #     'index': len(df) + len(new_rows) + 1,  # Generate new index
        #     'premise': premise,  # Keep premise unchanged
        #     'hypothesis': result.generative_negative_augmentation_summary,  # Use the generated negative summary
        #     'label': 'contradiction'  # Set label to contradiction
        # }
        #
        # new_rows.append(new_row)
        # print(f"Processed row {len(new_rows)}/{num_rows_to_transform}")

    # Create DataFrame with new rows
    new_df = pd.DataFrame(new_rows)

    # Append new rows to original DataFrame and save
    result_df = pd.concat([df, new_df], ignore_index=True)
    result_df.to_csv(output_csv_path, index=False)

    print(f"Successfully processed {len(new_rows)} rows and saved to {output_csv_path}")
    return result_df


# def main():
#     #input_csv_path = r"/sise/home/avramit/dataset_augmentation/output_mafat_less_500_test_new.csv"
#     input_csv_path = r"/sise/home/avramit/mfact-summ/mfact/datasets/heb/mafat_800_article_150_summary.csv"
#     output_csv_path = r"/sise/home/avramit/dataset_augmentation/output_mafat_less_500_test_negative.csv"
#     num_rows_to_transform = 20
#     transformation_types = [CorrectnessErrorTypes.ENTITY_ORGANIZATION_SWAP]
#
#     #transformation_types = [CorrectnessErrorTypes.NOUN_GENDER_SWAP, CorrectnessErrorTypes.VERB_GENDER_SWAP]
#
#     transformation_types_list = [t.strip() for t in transformation_types]
#     # Load the CSV data
#     df = pd.read_csv(input_csv_path)
#
#     # Take the first n rows to transform
#     if num_rows_to_transform < len(df):
#         rows_to_transform = df.sample(n=num_rows_to_transform)
#     else:
#         rows_to_transform = df  # Use all rows if requested more than available
#
#     # Prepare transformation examples
#     general_filtered_examples = {key: general_error_examples[key] for key in transformation_types if key in general_error_examples}
#     hebrew_filtered_examples = {key: hebrew_error_examples[key] for key in transformation_types if key in hebrew_error_examples}
#     filtered_examples = {**general_filtered_examples, **hebrew_filtered_examples}
#     print(filtered_examples)
#
#     # Create a list to store new rows
#     new_rows = []
#
#     # Process each selected row
#     for _, row in rows_to_transform.iterrows():
#         summary = row['hypothesis']
#         premise = row['premise']
#         article_ner = extract_named_entities(text=premise)
#
#         mapped_error_types = map_ner_to_error_types(article_ner, transformation_types)
#
#         # Generate the prompt instructions based on the mapped NER results
#         prompt_instructions = ner_instructions(mapped_error_types)
#
#         claim_decomposition = summary_claim_decomposition(text=summary)
#         choose_suitable_claims = choose_suitable_claim(claim_decomposition, transformation_types_list, filtered_examples)
#         print(choose_suitable_claims)
#         transformed_claims = summary_negative_augmentation(transformation_types_list, filtered_examples,
#                                                           choose_suitable_claims, claim_decomposition, prompt_instructions)
#         print(transformed_claims)
#         # Apply negative augmentation
#         # result = summary_negative_augmentation(
#         #     apply_transformation_types=transformation_types,
#         #     transformation_examples=filtered_examples,
#         #     amount_claims_to_apply_transformation=5,
#         #     original_summary=summary,
#         #     summary_claims=claim_decomposition.claims_decomposition
#         # )
#
#         # Create new row with the transformed data
#         new_row = {
#             'index': len(df) + len(new_rows) + 1,  # Generate new index
#             'premise': premise,  # Keep premise unchanged
#             'hypothesis': result.generative_negative_augmentation_summary,  # Use the generated negative summary
#             'label': 'contradiction'  # Set label to contradiction
#         }
#
#         new_rows.append(new_row)
#         print(f"Processed row {len(new_rows)}/{num_rows_to_transform}")
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
if __name__ == "__main__":
    print("hello")
    main()
