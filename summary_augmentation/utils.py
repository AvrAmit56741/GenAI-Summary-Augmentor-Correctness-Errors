# -*- coding: utf-8 -*-
from transformers import pipeline
from tokenizers.decoders import WordPiece
from typing import List, Dict, Type
from enum import Enum
from pydantic import BaseModel, PrivateAttr, Field
from openai import OpenAI
# Define OpenAI API credentials
import google.generativeai as genai
import re
import random
from typing import List, Dict, Tuple, Optional, Any
from pydantic import BaseModel, Field, PrivateAttr
import re
from typing import Optional
from summary_augmentation.correctness_errors_llm_helper_functions import summary_claim_decomposition
from summary_augmentation.correctness_errors_prompts import TransformationDetail, SummaryDecompositionClaims, NegativeAugmentationSummary

from difflib import SequenceMatcher
from typing import Dict, List, Any



class CorrectnessErrorTypes(str, Enum):
    NOUN_GENDER_SWAP = "noun_gender_swap"
    VERB_GENDER_SWAP = "verb_gender_swap"
    VERB_TENSE_SWAP = "verb_tense_swap"
    DIRECTIONAL_PREPOSITION_SWAP = "directional_preposition_swap"

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

GEMINI_API_KEY = "***REMOVED-API-KEY***"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')


def extract_named_entities(text: str):
    """
    Extracts named entities from the provided Hebrew text using the 'dicta-il/dictabert-ner' model.

    Args:
        text (str): The Hebrew text from which to extract named entities.

    Returns:
        list: A list of dictionaries with each entity's type, text, start and end positions.
    """
    # Initialize the NER pipeline with the specified model and aggregation strategy
    ner_pipeline = pipeline('ner', model='dicta-il/dictabert-ner', aggregation_strategy='simple')

    # Set the decoder for the tokenizer to handle wordpieces correctly
    ner_pipeline.tokenizer.backend_tokenizer.decoder = WordPiece()

    # Perform NER on the input text
    ner_results = ner_pipeline(text)

    # Convert results to a list of entity dictionaries with positions
    entities_list = []
    for entity in ner_results:
        entities_list.append({
            "entity_group": entity['entity_group'],
            "word": entity['word'],
            "start": entity['start'],
            "end": entity['end'],
            "score": entity.get('score', 0)
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
#         dict: A dictionary where keys are entity types and values are lists of entities of that type.
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
#     # Initialize a dictionary to store entities categorized by their types
#     entities_dict = {}
#
#     # Iterate over the NER results to populate the dictionary
#     for entity in ner_results:
#         entity_type = entity['entity_group']
#         entity_text = entity['word']
#         if entity_type not in entities_dict:
#             entities_dict[entity_type] = []
#         entities_dict[entity_type].append(entity_text)
#
#     return entities_dict

# def static_number_swap(claims):
# def filter_non_transformed_claims(claims):
def llm_query_to_gemini(system_message, user_message, output_object: Type[BaseModel]):
    print(f"prompt: {system_message}\n {user_message}")
    try:
        response = model.generate_content(
            f"""
            {system_message}
            {user_message}
            """
        )

        result_json = response.text.strip()
        # Remove triple backticks and any markdown-style JSON markers
        clean_json = re.sub(r"^```(?:json)?|```$", "", result_json.strip(), flags=re.MULTILINE).strip()
        print("\nRaw response:")
        print(result_json)
        result = output_object.model_validate_json(clean_json)
        return result
    except Exception as e:
        print(f"Exception during LLM query or parsing: {e}")
        return None



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



def recompose_transformed_summary(claim_decomposition, transformed_claims: Dict[str, List[Dict]]):
    """
    Rebuilds the summary by replacing original claims with their transformed versions.

    Args:
        claim_decomposition: Object containing decomposed claims
        transformed_claims: Dictionary mapping original claims to their transformed versions

    Returns:
        Dictionary with corrupted summary and transformation statistics
    """
    # Extract all claims from claim_decomposition
    all_claims = []
    for claim_item in claim_decomposition.claims_decomposition:
        claim_text = extract_claim_text(claim_item)
        if claim_text:
            all_claims.append(claim_text)

    # Create a copy of all claims to modify
    modified_claims = all_claims.copy()

    # Track transformations performed
    transformations_performed = []
    total_transformations = 0

    # For each transformed claim, find and replace in modified_claims
    for original_claim, transformations in transformed_claims.items():
        # Find the best matching claim in all_claims
        best_match = None
        best_similarity = 0

        for i, claim in enumerate(all_claims):
            similarity = SequenceMatcher(None, original_claim, claim).ratio()
            if similarity > 0.9 and similarity > best_similarity:
                best_similarity = similarity
                best_match = i

        # If we found a good match, replace it with a transformed version
        if best_match is not None and transformations:
            # Use the first transformation (could be extended to choose randomly or by specific criteria)
            transformation = transformations[0]

            # Replace the claim
            modified_claims[best_match] = transformation["transformed_claim"]

            # Track this transformation
            transformation_info = {
                "original_claim": original_claim,
                "transformed_claim": transformation["transformed_claim"],
                "error_type": transformation["error_type"]
            }

            # Add additional fields if they exist
            if "original_entity" in transformation:
                transformation_info["original_entity"] = transformation["original_entity"]
            if "replacement_entity" in transformation:
                transformation_info["replacement_entity"] = transformation["replacement_entity"]

            transformations_performed.append(transformation_info)
            total_transformations += 1

    # Rebuild the summary by joining the modified claims
    corrupted_summary = " ".join(modified_claims)

    # Calculate the transformation score
    total_claims = len(all_claims)
    transformation_score = total_transformations / total_claims if total_claims > 0 else 0

    # Create the result object
    result = {
        "corrupted_summary": corrupted_summary,
        "transformations_performed": total_transformations,
        "transformation_list": transformations_performed,
        "transformation_score": transformation_score
    }

    return result
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
