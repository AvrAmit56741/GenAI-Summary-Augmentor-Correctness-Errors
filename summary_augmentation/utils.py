# -*- coding: utf-8 -*-
import json
import pandas as pd
from typing import List, Dict
from difflib import SequenceMatcher
from typing import List, Dict, Type
from enum import Enum
from pydantic import BaseModel, PrivateAttr, Field
from openai import OpenAI
# Define OpenAI API credentials
import re
import random
from typing import List, Dict, Tuple, Optional, Any
from pydantic import BaseModel, Field, PrivateAttr
import re
from typing import Optional
from summary_augmentation.correctness_errors_llm_helper_functions import summary_claim_decomposition
from summary_augmentation.correctness_errors_prompts import TransformationDetail, SummaryDecompositionClaims, NegativeAugmentationSummary

from difflib import SequenceMatcher


import re
from Levenshtein import ratio
import itertools
from typing import Dict, List, Any
import itertools
from typing import Dict, List, Any
from transformers import pipeline, AutoTokenizer
import re
from tokenizers import Tokenizer
from tokenizers.decoders import WordPiece as WPDecoder
import logging

logger = logging.getLogger(__name__)

class CorrectnessErrorTypes(str, Enum):
    NOUN_GENDER_SWAP = "noun_gender_swap"
    VERB_GENDER_SWAP = "verb_gender_swap"
    VERB_TENSE_SWAP = "verb_tense_swap"
    DIRECTIONAL_PREPOSITION_SWAP = "directional_preposition_swap"

    PRONOUN_REFERENCE_ERRORS = "pronoun_reference_errors"
    SPECIFICITY_SHIFT_ERRORS = "specificity_shift_errors"
    HOMOGRAPHIC_GENDER_ERRORS = "homographic_gender_errors"
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

    ENTITY_TITLE_SWAP="entity_title_swap"
    ENTITY_EVENT_SWAP="entity_event_swap"
    ENTITY_LOCATION_SWAP = "entity_location_swap"
    ENTITY_PERSON_SWAP = "entity_person_swap"
    ENTITY_ORGANIZATION_SWAP = "entity_organization_swap"
    ENTITY_DATE_SWAP = "entity_date_swap"

    # hallucinations
    NUMBER_ADD = "number_add"
    ENTITY_LOCATION_REPLACE = "entity_location_replace"
    ENTITY_PERSON_REPLACE = "entity_person_replace"
    ENTITY_ORGANIZATION_REPLACE = "entity_organization_replace"
    ENTITY_DATE_REPLACE = "entity_date_replace"

class ClaimDecompositionMethod(str, Enum):
    FACTS = "facts"
    SENTENCES = "sentences"
    PARAGRAPH = "paragraphs"

# Define OpenAI API credentials
# EYAL
OPENAI_API_KEY = "***REMOVED-API-KEY***"

#AMIT
#OPENAI_API_KEY = "***REMOVED-API-KEY***"
client = OpenAI(api_key=OPENAI_API_KEY)



# Module-scope initialization
tokenizer = AutoTokenizer.from_pretrained('dicta-il/dictabert-ner')
ner_pipeline = pipeline(
    'ner',
    model='dicta-il/dictabert-ner',
    aggregation_strategy='simple'
)
ner_pipeline.tokenizer.backend_tokenizer.decoder = WPDecoder()

def detach_hebrew_clitics(txt: str) -> str:
    """
    Inserts a space after a standalone Hebrew clitic letter
    (ו־, ב־, ל־, כ־, מ־, ש־) when it's stuck onto a word.
    E.g. "שכדור הארץ" → "ש כדור הארץ"
    """
    return re.sub(
        r'\b([ובלכס])(?=[\u05D0-\u05EA])',
        r'\1 ',
        txt
    )
def normalize_hebrew(txt: str) -> str:
    txt = detach_hebrew_clitics(txt)
    # Remove Hebrew diacritics, unify punctuation, strip HTML and collapse whitespace
    txt = re.sub(r'[\u0591-\u05BD\u05BF\u05C1\u05C2\u05C4-\u05C7]', '', txt)
    txt = txt.replace('“', '"').replace('”', '"').replace('–', '-')
    txt = re.sub(r'<[^>]+>', ' ', txt)
    return re.sub(r'\s+', ' ', txt).strip()

def extract_named_entities(text: str, min_score: float = 0.6):
    """
    Splits the input into overlapping chunks to respect the 512-token limit,
    runs NER on each chunk, and aggregates results.
    Returns a list of dicts with keys: entity_group, word, start, end, score.
    """
    text = normalize_hebrew(text)
    tokens = tokenizer.tokenize(text)

    max_len = tokenizer.model_max_length      # e.g. 512
    chunk_size = max_len - 2                  # reserve for [CLS] & [SEP]
    overlap = 50                              # overlap to catch boundary entities

    entities_list = []
    for start_idx in range(0, len(tokens), chunk_size - overlap):
        chunk_tokens = tokens[start_idx:start_idx + chunk_size]
        chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
        ner_results = ner_pipeline(chunk_text)

        for entity in ner_results:
            score = entity.get('score', 0)
            if score < min_score:
                continue
            if "#" in entity['word'] or len(entity['word']) < 2:
                continue
            entities_list.append({
                "entity_group": entity['entity_group'],
                "word": entity['word'].replace("#", "").strip(),
                "start": entity['start'],
                "end": entity['end'],
                "score": score
            })

    return entities_list

# def extract_named_entities(text: str):
#     """
#     Extracts named entities from the provided Hebrew text using the 'dicta-il/dictabert-ner' model.
#
#     Args:
#         text (str): The Hebrew text from which to extract named entities.
#
#     Returns:
#         list: A list of dictionaries with each entity's type, text, start and end positions.
#     """
#     # Initialize the NER pipeline with the specified model and aggregation strategy
#     ner_pipeline = pipeline('ner', model='dicta-il/dictabert-ner', aggregation_strategy='simple')
#
#     # Set the decoder for the tokenizer to handle wordpieces correctly
#     ner_pipeline.tokenizer.backend_tokenizer.decoder = WordPiece()
#
#     # Perform NER on the input text
#     ner_results = ner_pipeline(text)
#
#     # Convert results to a list of entity dictionaries with positions
#     entities_list = []
#     for entity in ner_results:
#         entities_list.append({
#             "entity_group": entity['entity_group'],
#             "word": entity['word'],
#             "start": entity['start'],
#             "end": entity['end'],
#             "score": entity.get('score', 0)
#         })
#
#     return entities_list



def text_decomposition(
    text: str,
    decomposition_method: ClaimDecompositionMethod = ClaimDecompositionMethod.SENTENCES
) -> Optional[SummaryDecompositionClaims]:
    if decomposition_method == ClaimDecompositionMethod.SENTENCES:
        # Split the text into sentences using regular expressions
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Remove any empty strings and strip whitespace
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        # Create a list of dictionaries with sentence indices and texts
        claims_decomposition = [{idx: sentence} for idx, sentence in enumerate(sentences)]
        # Create and return the SummaryDecompositionClaims object
        return SummaryDecompositionClaims(
            claims_decomposition=claims_decomposition,
            total_amount_of_claims=len(claims_decomposition),
            original_text=text
        )

    if decomposition_method == ClaimDecompositionMethod.FACTS:
        return summary_claim_decomposition(text)
    # if decomposition_method==ClaimDecompositionMethod.PARAGRAPH:
    #     return text


def suitable_number_swap_transformation(claim_decomposition) -> List[Dict[str, Any]]:
    """
    Simplified function to identify claims containing numbers for potential number swap transformations.

    Args:
        claim_decomposition: Object containing decomposed claims

    Returns:
        A list of dictionaries, each containing claim, error_type, and the specific number to swap
    """
    results = []

    # Improved regex to catch various number formats:
    # - Integers: 42
    # - Decimals: 3.14
    # - Numbers with commas: 1,000
    # - Numbers with units: 50kg, $100, 5m
    # - Years: 2023
    # - Fractions: 1/2
    # - Percentages: 75%
    number_pattern = r'\b\d+(?:,\d+)*(?:\.\d+)?(?:\s*(?:[a-zA-Z]+|%|\$|€|£|\d+/\d+))?\b'

    # Process each claim to find numbers
    for claim_item in claim_decomposition.claims_decomposition:
        claim_text = extract_claim_text(claim_item)
        if not claim_text:
            continue

        # Find all numbers in the claim
        numbers = re.findall(number_pattern, claim_text)

        # Add each number as a potential transformation
        for number in numbers:
            results.append({
                "claim": claim_text,
                "error_type": CorrectnessErrorTypes.NUMBER_SWAP,
                "word": number.strip()
            })

    return results

def extract_claim_text(claim_item):
    """Helper function to extract text from different claim formats"""
    if isinstance(claim_item, dict):
        # Handle dictionary format
        for _, claim_text in claim_item.items():
            return claim_text  # Just get the first (and only) value
    elif isinstance(claim_item, tuple):
        # Handle tuple format (index, text)
        return claim_item[1]
    elif isinstance(claim_item, str):
        # Handle string format directly
        return claim_item
    else:
        # Return None for unknown formats
        return None

def number_swap_transformation(
        summary_decomposition_claims: SummaryDecompositionClaims,
        number_swap_transformation_amount: Optional[int] = None
) -> List[Tuple[str, str]]:
    """
    Perform number swaps between claims in the decomposition.

    Args:
        summary_decomposition_claims: A SummaryDecompositionClaims object containing the claims.
        number_swap_transformation_amount: Optional number of swaps to perform. If None, perform all possible swaps.

    Returns:
        A list of tuples, each containing [original claim, number swap claim].
    """
    # Extract claims as (id, text) pairs
    claims = []
    for claim_dict in summary_decomposition_claims.claims_decomposition:
        for idx, claim in claim_dict.items():
            claims.append((idx, claim))

    # Store original claims for reference
    original_claims = {idx: claim for idx, claim in claims}

    # Find all numbers in each claim, including those with commas like "2,000"
    numbers_in_claims = {}
    for idx, claim in claims:
        # Updated regex to match numbers with commas and optional decimal places
        numbers = re.findall(r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b', claim)
        if numbers:
            numbers_in_claims[idx] = numbers

    # Create a list of all possible number swaps
    possible_swaps = []
    for idx1, numbers1 in numbers_in_claims.items():
        for num1 in numbers1:
            for idx2, numbers2 in numbers_in_claims.items():
                for num2 in numbers2:
                    # Skip if the claims are the same or the numbers are the same
                    if idx1 == idx2 or num1 == num2:
                        continue
                    possible_swaps.append((idx1, num1, idx2, num2))

    # Determine how many swaps to perform
    if number_swap_transformation_amount is None or number_swap_transformation_amount > len(possible_swaps):
        number_of_swaps = len(possible_swaps)
    else:
        number_of_swaps = number_swap_transformation_amount

    # Randomly select swaps to perform
    random.shuffle(possible_swaps)

    # Perform the swaps and record the results
    modified_claims = {idx: claim for idx, claim in claims}
    processed_claims = set()
    results = []

    for swap in possible_swaps:
        idx1, num1, idx2, num2 = swap

        # Skip if either claim has already been processed
        if idx1 in processed_claims or idx2 in processed_claims:
            continue

        # Get the current state of the claims
        claim1 = modified_claims[idx1]
        claim2 = modified_claims[idx2]

        # Create new claims with swapped numbers
        # Use word boundaries to ensure we're replacing the whole number
        pattern1 = r'\b' + re.escape(num1) + r'\b'
        pattern2 = r'\b' + re.escape(num2) + r'\b'

        new_claim1 = re.sub(pattern1, num2, claim1, count=1)
        new_claim2 = re.sub(pattern2, num1, claim2, count=1)

        # Update the modified claims
        modified_claims[idx1] = new_claim1
        modified_claims[idx2] = new_claim2

        # Mark these claims as processed
        processed_claims.add(idx1)
        processed_claims.add(idx2)

        # Add to results
        results.append((original_claims[idx1], new_claim1))
        results.append((original_claims[idx2], new_claim2))

        # Stop if we've performed enough swaps
        if len(processed_claims) // 2 >= number_of_swaps:
            break

    return results

# def recompose_transformed_summary(claim_decomposition, transformed_claims: Dict[str, List[Dict]]):
#     """
#     Rebuilds the summary by replacing original claims with their transformed versions.
#
#     Args:
#         claim_decomposition: Object containing decomposed claims
#         transformed_claims: Dictionary mapping original claims to their transformed versions
#
#     Returns:
#         Dictionary with corrupted summary and transformation statistics
#     """
#     # Extract all claims from claim_decomposition
#     all_claims = []
#     for claim_item in claim_decomposition.claims_decomposition:
#         claim_text = extract_claim_text(claim_item)
#         if claim_text:
#             all_claims.append(claim_text)
#
#     # Create a copy of all claims to modify
#     modified_claims = all_claims.copy()
#
#     # Track transformations performed
#     transformations_performed = []
#     total_transformations = 0
#
#     # For each transformed claim, find and replace in modified_claims
#     for original_claim, transformations in transformed_claims.items():
#         # Find the best matching claim in all_claims
#         best_match = None
#         best_similarity = 0
#
#         for i, claim in enumerate(all_claims):
#             similarity = SequenceMatcher(None, original_claim, claim).ratio()
#             if similarity > 0.9 and similarity > best_similarity:
#                 best_similarity = similarity
#                 best_match = i
#
#         # If we found a good match, replace it with a transformed version
#         if best_match is not None and transformations:
#             # Use the first transformation (could be extended to choose randomly or by specific criteria)
#             # takes only the first transformation
#             transformation = transformations[0]
#
#             # Replace the claim
#             modified_claims[best_match] = transformation["transformed_claim"]
#
#             # Track this transformation
#             transformation_info = {
#                 "original_claim": original_claim,
#                 "transformed_claim": transformation["transformed_claim"],
#                 "error_type": transformation["error_type"]
#             }
#
#             # Add additional fields if they exist
#             if "original_entity" in transformation:
#                 transformation_info["original_entity"] = transformation["original_entity"]
#             if "replacement_entity" in transformation:
#                 transformation_info["replacement_entity"] = transformation["replacement_entity"]
#
#             transformations_performed.append(transformation_info)
#             total_transformations += 1
#
#     # Rebuild the summary by joining the modified claims
#     corrupted_summary = " ".join(modified_claims)
#
#     # Calculate the transformation score
#     total_claims = len(all_claims)
#     transformation_score = total_transformations / total_claims if total_claims > 0 else 0
#
#     # Create the result object
#     result = {
#         "corrupted_summary": corrupted_summary,
#         "transformations_performed": total_transformations,
#         "transformation_list": transformations_performed,
#         "transformation_score": transformation_score
#     }
#
#     return result

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

def ner_instructions(mapped_ner_results: Dict[CorrectnessErrorTypes, List[str]]) -> str:
    # Instruction templates for each error type
    instructions = []

    for error_type, entities in mapped_ner_results.items():
        template = INSTRUCTION_TEMPLATES.get(error_type)
        if template:
            instruction = template.format(entities=", ".join(entities))
            instructions.append(instruction)

    return "\n".join(instructions)
def recompose_transformed_summary(claim_decomposition, transformed_claims: Dict[str, List[Dict]]):
    """
    Rebuilds the summary by replacing original claims with their transformed versions,
    applying at most one transformation per claim.
    """
    # 1) Extract all claims
    all_claims = []
    for claim_item in claim_decomposition.claims_decomposition:
        claim_text = extract_claim_text(claim_item)
        if claim_text:
            all_claims.append(claim_text)
    modified_claims = all_claims.copy()

    total_transformations = 0
    transformations_performed = []
    error_type_names = set()
    logger.info("Starting recompose: %d claims to check", len(transformed_claims))

    # 2) For each claim with transformations
    for original_claim, transformations in transformed_claims.items():
        logger.debug("Processing claim: %s", original_claim)
        if not transformations:
            logger.debug(" -> No transformations for this claim, skipping")
            continue

        # 2a) Find the best matching sentence index
        best_match = None
        best_sim = 0.0
        for i, claim in enumerate(all_claims):
            sim = SequenceMatcher(None, original_claim, claim).ratio()
            if sim > 0.9 and sim > best_sim:
                best_sim = sim
                best_match = i
        logger.debug(" Best match index=%s, similarity=%.3f", best_match, best_sim)
        if best_match is None:
            logger.error(f"!! No suitable match found for original claim: '{original_claim}' (max_sim={best_sim:.3f}), skipping")
            continue

        # 2b) Prepare to patch this claim
        current_claim = all_claims[best_match]
        applied = []

        # 2c) Group by error type
        transformations_by_type = {}
        for t in transformations:
            key = str(t["error_type"])
            transformations_by_type.setdefault(key, []).append(t)
        logger.debug(f"   Transformations found for error types: {list(transformations_by_type.keys())}")

        # 2d) Sort so LLM-generated patches run first
        items = list(transformations_by_type.items())
        items.sort(
            key=lambda kv: any(x.get("llm_generated") for x in kv[1]),
            reverse=True
        )

        # 3) Apply at most one transformation
        for t_type, group in items:
            logger.debug(f"Attempting group error_type={t_type}, {len(group)} candidates (llm_generated={any(x.get('llm_generated') for x in group)})")
            llm_opts = [x for x in group if x.get("llm_generated")]
            t = llm_opts[0] if llm_opts else group[0]
            error_type_names.add(t["error_type"])
            # 3a) Static replace (entity or number)
            if "original_entity" in t and "replacement_entity" in t:
                orig = t["original_entity"]
                repl = t["replacement_entity"]
                if orig in current_claim:
                    current_claim = current_claim.replace(orig, repl)
                    applied.append({
                        "type": t["error_type"],
                        "original": orig,
                        "replacement": repl
                    })
                    total_transformations += 1

            # 3b) LLM-generated minimal patch
            elif t.get("llm_generated") and "diffs" in t:
                diffs = t["diffs"]
                current_claim = apply_diffs(current_claim, diffs)
                applied.append({
                    "type": t["error_type"],
                    "diffs": diffs
                })
                total_transformations += len(diffs)

            # elif t.get("llm_generated"):
            #     o = t["original_claim"]
            #     n = t["transformed_claim"]
            #     # find common prefix
            #     i = 0
            #     while i < len(o) and i < len(n) and o[i] == n[i]:
            #         i += 1
            #     # find common suffix
            #     j_o, j_n = len(o), len(n)
            #     while j_o > i and j_n > i and o[j_o - 1] == n[j_n - 1]:
            #         j_o -= 1
            #         j_n -= 1
            #     o_mid = o[i:j_o]
            #     n_mid = n[i:j_n]
            #
            #     if o_mid and o_mid in current_claim:
            #         current_claim = current_claim.replace(o_mid, n_mid)
            #         applied.append({
            #             "type": t["error_type"],
            #             "original": o_mid,
            #             "replacement": n_mid
            #         })
            #         total_transformations += 1
            #         break
            #
            #     elif not o_mid and n_mid:
            #         current_claim = current_claim[:i] + n_mid + current_claim[i:]
            #         applied.append({
            #             "type": t["error_type"],
            #             "original": "",
            #             "replacement": n_mid
            #         })
            #         total_transformations += 1
            #         break

        # 4) Save patched claim
        modified_claims[best_match] = current_claim
        transformations_performed.append({
            "original_claim": original_claim,
            "transformed_claim": current_claim,
            "applied_transformations": applied
        })
        logger.info("recompose_transformed_summary - Finished claim. Applied total %d transforms", len(applied))
        logger.info(f"recompose_transformed_summary: \noriginal claim: {original_claim}\ntransformed_claim: {current_claim}")

    # 5) Rebuild the full summary and score
    corrupted_summary = " ".join(modified_claims)
    total_claims = len(all_claims)
    transformation_score = total_transformations / total_claims if total_claims else 0
    logger.info(f"Recompose complete: {total_transformations} transformations applied across {total_claims} claims (score={transformation_score:.3f})")
    logger.info(f"Error types covered: {list(error_type_names)}")
    return {
        "corrupted_summary": corrupted_summary,
        "transformations_performed": total_transformations,
        "transformation_list": transformations_performed,
        "transformation_score": transformation_score,
        "distinct_errors_type_names": list(error_type_names)
    }

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
# !!! worked !!!
# def recompose_transformed_summary(claim_decomposition, transformed_claims: Dict[str, List[Dict]]):
#     """
#     Rebuilds the summary by replacing original claims with their transformed versions,
#     applying multiple transformations to each claim where available.
#     """
#     # 1) Extract all claims
#     all_claims = []
#     for claim_item in claim_decomposition.claims_decomposition:
#         claim_text = extract_claim_text(claim_item)
#         if claim_text:
#             all_claims.append(claim_text)
#     modified_claims = all_claims.copy()
#
#     total_transformations = 0
#     transformations_performed = []
#     logger.info("Starting recompose: %d claims to check", len(transformed_claims))
#
#     # 2) For each claim with transformations
#     for original_claim, transformations in transformed_claims.items():
#         logger.debug("Processing claim: %s", original_claim)
#         if not transformations:
#             logger.debug(" -> No transformations for this claim, skipping")
#             continue
#
#         # 2a) Find the best matching sentence index
#         best_match = None
#         best_sim = 0.0
#         for i, claim in enumerate(all_claims):
#             sim = SequenceMatcher(None, original_claim, claim).ratio()
#             if sim > 0.9 and sim > best_sim:
#                 best_sim = sim
#                 best_match = i
#         logger.debug(" Best match index=%s, similarity=%.3f", best_match, best_sim)
#         if best_match is None:
#             continue
#
#         # 2b) Prepare to patch this claim
#         current_claim = all_claims[best_match]
#         applied = []
#
#         # 2c) Group by error type
#         transformations_by_type = {}
#         for t in transformations:
#             key = str(t["error_type"])
#             transformations_by_type.setdefault(key, []).append(t)
#             logger.debug(" Grouped into %d types", len(transformations_by_type))
#
#         # 2d) Sort so LLM-generated patches run first
#         items = list(transformations_by_type.items())
#         items.sort(
#             key=lambda kv: any(x.get("llm_generated") for x in kv[1]),
#             reverse=True
#         )
#
#         # 3) Apply each transformation
#         for t_type, group in items:
#             logger.debug("  Applying type: %s (%d candidates)", t_type, len(group))
#             # pick LLM if available, else static
#             llm_opts = [x for x in group if x.get("llm_generated")]
#             t = llm_opts[0] if llm_opts else group[0]
#
#             # 3a) Static replace (entity or number)
#             if "original_entity" in t and "replacement_entity" in t:
#                 orig = t["original_entity"]
#                 repl = t["replacement_entity"]
#                 if orig in current_claim:
#                     current_claim = current_claim.replace(orig, repl)
#                     applied.append({
#                         "type": t["error_type"],
#                         "original": orig,
#                         "replacement": repl
#                     })
#                     logger.debug("   Static replace: %s -> %s", orig, repl)
#
#             # 3b) LLM-generated minimal patch
#             elif t.get("llm_generated"):
#                 o = t["original_claim"]
#                 n = t["transformed_claim"]
#                 # find common prefix
#                 i = 0
#                 while i < len(o) and i < len(n) and o[i] == n[i]:
#                     i += 1
#                 # find common suffix
#                 j_o, j_n = len(o), len(n)
#                 while j_o > i and j_n > i and o[j_o-1] == n[j_n-1]:
#                     j_o -= 1
#                     j_n -= 1
#                 o_mid = o[i:j_o]
#                 n_mid = n[i:j_n]
#                 if o_mid and o_mid in current_claim:
#                     # replacement case
#                     current_claim = current_claim.replace(o_mid, n_mid)
#                     applied.append({
#                         "type": t["error_type"],
#                         "original": o_mid,
#                         "replacement": n_mid
#                     })
#                     logger.debug("   LLM patch replace: %s -> %s", o_mid, n_mid)
#                 elif not o_mid and n_mid:
#                     # pure insertion case
#                     current_claim = current_claim[:i] + n_mid + current_claim[i:]
#                     applied.append({
#                         "type": t["error_type"],
#                         "original": "",
#                         "replacement": n_mid
#                     })
#                     logger.debug("   LLM patch insert: '%s' at pos %d", n_mid, i)
#
#             total_transformations += 1
#
#         # 4) Save patched claim
#         modified_claims[best_match] = current_claim
#         transformations_performed.append({
#             "original_claim": original_claim,
#             "transformed_claim": current_claim,
#             "applied_transformations": applied
#         })
#         logger.info(" Finished claim. Applied %d transforms", len(applied))
#
#     # 5) Rebuild the full summary and score
#     corrupted_summary = " ".join(modified_claims)
#     total_claims = len(all_claims)
#     transformation_score = total_transformations / total_claims if total_claims else 0
#
#     return {
#         "corrupted_summary": corrupted_summary,
#         "transformations_performed": total_transformations,
#         "transformation_list": transformations_performed,
#         "transformation_score": transformation_score
#     }

# !!! worked !!!
# def recompose_transformed_summary(claim_decomposition, transformed_claims: Dict[str, List[Dict]]):
#     """
#     Rebuilds the summary by replacing original claims with their transformed versions,
#     applying multiple transformations to each claim where available.
#     """
#     # Extract all claims from claim_decomposition
#     all_claims = []
#     for claim_item in claim_decomposition.claims_decomposition:
#         claim_text = extract_claim_text(claim_item)
#         if claim_text:
#             all_claims.append(claim_text)
#
#     # Create a copy of all claims to modify
#     modified_claims = all_claims.copy()
#
#     # Track transformations performed
#     transformations_performed = []
#     total_transformations = 0
#
#     # For each transformed claim, find and replace in modified_claims
#     for original_claim, transformations in transformed_claims.items():
#         if not transformations:
#             continue
#
#         # Find the best matching claim in all_claims
#         best_match = None
#         best_similarity = 0
#
#         for i, claim in enumerate(all_claims):
#             similarity = SequenceMatcher(None, original_claim, claim).ratio()
#             if similarity > 0.9 and similarity > best_similarity:
#                 best_similarity = similarity
#                 best_match = i
#
#         # If we found a good match, apply multiple transformations
#         if best_match is not None:
#             # Start with the original claim
#             current_claim = all_claims[best_match]
#             applied_transformations = []
#
#             # Group transformations by type to avoid conflicts
#             # This prevents applying two NUMBER_SWAP transformations that might target the same number
#             transformations_by_type = {}
#             for t in transformations:
#                 t_type = str(t["error_type"])  # Convert to string to handle enum objects
#                 if t_type not in transformations_by_type:
#                     transformations_by_type[t_type] = []
#                 transformations_by_type[t_type].append(t)
#
#             # Apply one transformation of each type, preferring LLM results
#             for t_type, type_transformations in transformations_by_type.items():
#                 # if the LLM gave us a transformed_claim, use it
#                 llm_opts = [x for x in type_transformations if x.get("llm_generated")]
#                 t = llm_opts[0] if llm_opts else type_transformations[0]
#             # # Apply one transformation of each type
#             # for t_type, type_transformations in transformations_by_type.items():
#             #     # Choose just one transformation per type (first one)
#             #     t = type_transformations[0]
#
#                 # For entity replacement or number swap, we need to check if the original entity
#                 # still exists after previous transformations
#                 if "original_entity" in t and "replacement_entity" in t:
#                     # Replace the entity if it still exists in the current version
#                     if t["original_entity"] in current_claim:
#                         current_claim = current_claim.replace(t["original_entity"], t["replacement_entity"])
#                         applied_transformations.append({
#                             "type": t["error_type"],
#                             "original": t["original_entity"],
#                             "replacement": t["replacement_entity"]
#                         })
#                 # For LLM-generated transformations, use the complete transformed claim
#                 # but only if no other transformation of the same type was applied
#                 elif "llm_generated" in t and t["llm_generated"]:
#                     # Replace the entire claim with the LLM-generated version
#                     current_claim = t["transformed_claim"]
#                     applied_transformations.append({
#                         "type": t["error_type"],
#                         "llm_generated": True,
#                         "original_claim": t["original_claim"],
#                         "transformed_claim": t['transformed_claim']
#                     })
#
#                 # Track each successful transformation
#                 total_transformations += 1
#
#             # Update the claim with all applied transformations
#             modified_claims[best_match] = current_claim
#
#             # Record transformation info
#             transformation_info = {
#                 "original_claim": original_claim,
#                 "transformed_claim": current_claim,
#                 "applied_transformations": applied_transformations
#             }
#             transformations_performed.append(transformation_info)
#
#     # Rebuild the summary by joining the modified claims
#     corrupted_summary = " ".join(modified_claims)
#
#     # Calculate the transformation score
#     total_claims = len(all_claims)
#     transformation_score = total_transformations / total_claims if total_claims > 0 else 0
#
#     # Create the result object
#     result = {
#         "corrupted_summary": corrupted_summary,
#         "transformations_performed": total_transformations,
#         "transformation_list": transformations_performed,
#         "transformation_score": transformation_score
#     }
#
#     return result


def apply_diffs(claim, diffs):
    # Apply from last to first to keep indices correct
    for diff in sorted(diffs, key=lambda d: d['orig_start'], reverse=True):
        claim = (
            claim[:diff['orig_start']] +
            diff['trans_text'] +
            claim[diff['orig_end']:]
        )
    return claim

def control_claims_after_transformation(
    claims_after_transformation: Dict[str, List[Dict[str, Any]]],
    amount_of_transformed_claims: int = None,
    amount_of_correctness_errors: int = None,
    percentage_of_correctness_errors_transformation: float = None
) -> List[Dict[str, List[Dict[str, Any]]]]:

    # 1. Prepare list of claims and their errors
    items = list(claims_after_transformation.items())
    num_claims = len(items)

    # 2. Determine k = total errors to pick
    total_errors = sum(len(errs) for _, errs in items)
    if percentage_of_correctness_errors_transformation is not None:
        k = int(total_errors * percentage_of_correctness_errors_transformation / 100)
    else:
        k = amount_of_correctness_errors
    k = max(1, min(k, total_errors))

    # 3. Recursive generator: allocate rem errors among claims[i:]
    def gen(i, rem):
        if i == num_claims:
            if rem == 0:
                yield []
            return
        claim, errs = items[i]
        n = len(errs)
        # c = how many errors to pick from this claim
        # ensure that the remaining claims can supply rem-c errors
        min_c = max(0, rem - sum(len(e) for _, e in items[i+1:]))
        max_c = min(n, rem)
        for c in range(min_c, max_c + 1):
            for combo in itertools.combinations(errs, c):
                for rest in gen(i+1, rem - c):
                    yield [(claim, list(combo))] + rest

    # 4. Build results
    results = []
    for allocation in gen(0, k):
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for claim, errs in allocation:
            if errs:
                grouped[claim] = errs
        grouped_filtered = apply_transformation_restrictions(grouped)
        results.append(grouped_filtered)
        # results.append(grouped)

    return results

def apply_transformation_restrictions(grouped):
    """Apply restriction rules to a grouped allocation of transformations."""
    # Rule: sentence_negation only allowed if it is the only transformation for that claim
    filtered = {}
    for claim, errors in grouped.items():
        # get types of errors for this claim
        types = {e['error_type'] for e in errors}
        # If 'sentence_negation' is in types, and there are also other types, remove it
        if 'sentence_negation' in types and len(types) > 1:
            filtered_errors = [e for e in errors if e['error_type'] != 'sentence_negation']
            if filtered_errors:
                filtered[claim] = filtered_errors
        else:
            filtered[claim] = errors
    return filtered
def normalize_heb(text: str) -> str:
    """
    Lowercase, strip punctuation and extra whitespace.
    """
    # Keep Hebrew letters, Latin letters, digits and spaces
    cleaned = re.sub(r'[^\wא-ת0-9 ]+', '', text)
    return cleaned.strip().lower()

def token_containment(a: str, b: str) -> bool:
    """
    Return True if the token set of a is a subset of b or vice versa.
    Catches “עמית” ⊆ “עמית אברהם”.
    """
    ta = set(normalize_heb(a).split())
    tb = set(normalize_heb(b).split())
    return bool(ta and (ta <= tb or tb <= ta))

def is_same_entity(a: str, b: str) -> bool:
    """
    Heuristic to detect if two mentions refer to the same entity:
      1. Token containment (subset/superset)
      2. High-overlap string similarity (Levenshtein ratio > 0.8)
      3. Head-noun match + fuzzy on modifiers
    """
    try:
        na, nb = normalize_heb(a), normalize_heb(b)

        # If either normalized string is empty, they can't match
        if not na or not nb:
            return False

        # 1) Token containment
        if token_containment(na, nb):
            return True

        # 2) High-overlap string similarity
        if ratio(na, nb) > 0.8:
            return True

        # 3) Head-noun + fuzzy modifiers
        parts_a = na.split()
        parts_b = nb.split()
        if not parts_a or not parts_b:
            return False

        ha, *ra = parts_a
        hb, *rb = parts_b

        if ha == hb:
            rest_a = " ".join(ra)
            rest_b = " ".join(rb)
            if rest_a and rest_b:
                if token_containment(rest_a, rest_b) or ratio(rest_a, rest_b) > 0.7:
                    return True

        return False

    except Exception:
        # On any unexpected error, assume they are not the same
        return False


def get_entities_by_type(text: str) -> Dict[CorrectnessErrorTypes, List[str]]:
    """
    Extract named entities from `text` and group them by our CorrectnessErrorTypes,
    deduplicated as lists.
    """
    ner = extract_named_entities(text)
    grouped: Dict[CorrectnessErrorTypes, Set[str]] = {}
    for ent in ner:
        et = ent["entity_group"].upper()
        if et == "PER":
            key = CorrectnessErrorTypes.ENTITY_PERSON_SWAP
        elif et in ("LOC", "GPE"):
            key = CorrectnessErrorTypes.ENTITY_LOCATION_SWAP
        elif et == "ORG":
            key = CorrectnessErrorTypes.ENTITY_ORGANIZATION_SWAP
        elif et == "TTL":
            key = CorrectnessErrorTypes.ENTITY_TITLE_SWAP
        elif et == "EVE":
            key = CorrectnessErrorTypes.ENTITY_EVENT_SWAP
        else:
            continue

        grouped.setdefault(key, set()).add(ent["word"])

    # to lists
    return {k: list(v) for k, v in grouped.items()}

def save_dataset_json(records: List[dict], path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def save_dataset_csv(records: List[dict], path: str):
    rows = []

    for rec in records:
        # Add the original entailment pair
        rows.append({
            'premise': rec['source_article'],
            'hypothesis': rec['gold_summary'],
            'label': 'entailment',
            'len_group': rec['len_group'],
            'doc_id': rec['doc_id'],
            'source': rec['source'],
            'error_types': ''  # Empty for entailment pairs
            '',
        })

        # Add all the contradiction pairs from corrupted summaries
        for corrupt in rec.get('correctness_corrupted_summaries', []):
            # Extract error types from the transformation_list
            ets = []
            for t in corrupt.get('transformation_list', []):
                # Handle the Enum type in the data structure
                applied_trans = t.get('applied_transformations', [])
                for at in applied_trans:
                    ty = at.get('type')
                    if isinstance(ty, CorrectnessErrorTypes):
                        ets.append(ty.value)
                    elif isinstance(ty, str):
                        ets.append(ty)

            rows.append({
                'premise': rec['source_article'],
                'hypothesis': corrupt['corrupted_summary'],  # Just the text string
                'label': 'contradiction',
                'len_group': rec['len_group'],
                'doc_id': rec['doc_id'],
                'source': rec['source'],
                'error_types': ','.join(set(ets))  # Use set to remove duplicates
            })

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def token_overlap(a: str, b: str) -> float:
    """
    Compute overlap ratio = |tokens(a) ∩ tokens(b)| / min(|tokens(a)|, |tokens(b)|).
    A value of 1.0 means one is a subset of the other; 0 means no tokens in common.
    """
    a_tok = set(a.split())
    b_tok = set(b.split())
    if not a_tok or not b_tok:
        return 0.0
    return len(a_tok & b_tok) / min(len(a_tok), len(b_tok))


def jaccard(tokens1, tokens2):
    s1, s2 = set(tokens1), set(tokens2)
    if not s1 and not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)

def ngram(tokens, n):
    return list(zip(*(tokens[i:] for i in range(n))))

def filter_diverse_summaries(
    golden: str,
    cands: List[Dict],
    max_same: float = 0.8
) -> List[Dict]:
    """
    Keep only those candidates whose combined unigram/bigram overlap
    with the golden summary is <= max_same, and likewise <= max_same
    against any already-kept candidate.
    """
    def overlap(a: str, b: str) -> float:
        t1 = a.split()
        t2 = b.split()
        uni = jaccard(t1, t2)
        bi  = jaccard(ngram(t1, 2), ngram(t2, 2))
        return max(uni, bi)


    kept = []
    for c in cands:
        txt = c['corrupted_summary']
        max_same = max_same + 0.0013 * (len(c.get('distinct_errors_type_names', [])) ** 0.75)
        # 1) filter vs. golden
        if overlap(txt, golden) > max_same:
            continue

        # 2) filter vs. already-kept
        too_similar = False
        for k in kept:
            if overlap(txt, k['corrupted_summary']) > max_same:
                too_similar = True
                break
        if too_similar:
            continue

        kept.append(c)

    return kept



def find_diff_and_index(original: str, transformed: str):
    matcher = SequenceMatcher(None, original, transformed)
    # Returns a list of "opcodes" describing how to turn original into transformed
    # Each opcode: (tag, i1, i2, j1, j2)
    # 'replace', 'delete', 'insert', or 'equal'
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != 'equal':
            # Only process the *first* difference found
            return {
                'start': i1,
                'end': i2,
                'original_span': original[i1:i2],
                'replacement_span': transformed[j1:j2]
            }
    # No difference found
    return None

def compute_all_diffs(original, transformed):
    matcher = SequenceMatcher(None, original, transformed)
    diffs = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != 'equal':
            diffs.append({
                'tag': tag,  # 'replace', 'insert', 'delete'
                'orig_start': i1,
                'orig_end': i2,
                'trans_start': j1,
                'trans_end': j2,
                'orig_text': original[i1:i2],
                'trans_text': transformed[j1:j2]
            })
    return diffs

# def llm_query_to_openai(system_message, user_message, output_object: Type[BaseModel]):
#
#     print(f"prompt: {system_message}\n {user_message}")
#     try:
#         response = client.chat.completions.create(
#             model="gpt-4o",
#             #model="o3-mini",
#             messages=[
#                 {"role": "system", "content": system_message},
#                 {"role": "user", "content": user_message}
#             ],
#             temperature=0.0,
#             max_tokens=2000,
#             #max_completion_tokens=2000,
#             response_format={"type": "json_object"}
#         )
#         # Extract and parse the JSON response
#         result_json = response.choices[0].message.content
#         print("\nRaw response:")
#         print(result_json.encode('utf-8').decode('utf-8'))
#
#         result = output_object.model_validate_json(result_json)
#         return result
#
#     except Exception as e:
#         print(f"Exception during openai LLM query or parsing: {e}")
#         return None
