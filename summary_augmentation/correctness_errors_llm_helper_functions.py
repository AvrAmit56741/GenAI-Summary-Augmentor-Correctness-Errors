# Define the main data structure
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers.retry import RetryWithErrorOutputParser
from typing import List, Tuple
from typing import List, Dict
import logging
from typing import List
import pandas as pd
import json
from pydantic import BaseModel, PrivateAttr, Field
from typing import List
from enum import Enum
from openai import OpenAI
# Define OpenAI API credentials
import google.generativeai as genai
from typing import List, Dict, Type
from langchain_openai import ChatOpenAI
from difflib import SequenceMatcher
from anthropic import Anthropic

from summary_augmentation.correctness_errors_prompts import TransformationDetail, SummaryDecompositionClaims, \
    NegativeAugmentationSummary
from summary_augmentation.correctness_errors_prompts import verify_grammar_transform_summary_prompt, \
    suitable_claim_decomposition_prompt, apply_transformation_on_claims_prompt, \
    apply_halluzaition_transformation_on_claims_prompt, claims_decomposition_prompt, \
    suitable_claim_decomposition_permissive_prompt,faithfulness_summary_back_translation_and_paraphrasing

# Define the main data structure
import google.generativeai as genai
GEMINI_API_KEY = "***REMOVED-API-KEY***"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

# Define OpenAI API credentials
# EYAL
OPENAI_API_KEY = "***REMOVED-API-KEY***"
CLAUDE_API_KEY = "***REMOVED-API-KEY***"
client_claude = Anthropic(api_key=CLAUDE_API_KEY)
# AMIT
# OPENAI_API_KEY = "***REMOVED-API-KEY***"
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.0,
    max_tokens=2000,
    api_key=OPENAI_API_KEY,
    response_format="json_object"
)
client = OpenAI(api_key=OPENAI_API_KEY)
def llm_query_to_claude(system_message: str, user_message: str, output_object: Type[BaseModel], parser=None):
    """
    Query Claude AI with system and user messages, expecting a structured JSON response.

    Args:
        system_message: The system prompt/instructions
        user_message: The user's message/query
        output_object: Pydantic BaseModel class for output validation
        parser: Optional retry parser for handling parsing errors

    Returns:
        Parsed object of type output_object or None if failed
    """
    print(f"prompt: {system_message}\n {user_message}")

    try:
        # Create the message with explicit JSON request
        enhanced_user_message = f"{user_message}\n\nPlease respond with valid JSON only."

        response = client_claude.messages.create(
            #model="claude-3-7-sonnet-20250219",
            model="claude-3-5-sonnet-20241022",  # or "claude-3-opus-20240229" for more powerful model
            max_tokens=2000,
            temperature=0.0,
            system=system_message,
            messages=[
                {"role": "user", "content": enhanced_user_message}
            ]
        )

        # Extract the raw response
        result_json = response.content[0].text
        print("\nRaw response:")
        print(result_json.encode('utf-8').decode('utf-8'))

        # Use the retry parser if provided, otherwise fallback to direct validation
        if parser:
            # The retry parser will handle any parsing errors and retry if needed
            try:
                result = parser.parse(result_json)
                return result
            except Exception as parsing_error:
                # If the retry parser fails after all retries, log it
                print(f"Retry parser failed: {parsing_error}")
                # Fallback to direct parsing as a last resort
                return output_object.model_validate_json(result_json)
        else:
            # Direct parsing without retry
            return output_object.model_validate_json(result_json)

    except Exception as e:
        print(f"Exception during Claude LLM query or parsing: {e}")
        return None

def llm_query_to_gemini(system_message, user_message, output_object: Type[BaseModel], parse=None):
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


def llm_query_to_openai(system_message, user_message, output_object: Type[BaseModel], parser=None, openai_model="gpt-4o"):
    print(f"prompt: {system_message}\n {user_message}")
    try:
        response = client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.0,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )
        # Extract the raw response
        result_json = response.choices[0].message.content
        print("\nRaw response:")
        print(result_json.encode('utf-8').decode('utf-8'))

        # Use the retry parser if provided, otherwise fallback to direct validation
        if parser:
            # The retry parser will handle any parsing errors and retry if needed
            try:
                result = parser.parse(result_json)
                return result
            except Exception as parsing_error:
                # If the retry parser fails after all retries, log it
                print(f"Retry parser failed: {parsing_error}")
                # Fallback to direct parsing as a last resort
                return output_object.model_validate_json(result_json)
        else:
            # Direct parsing without retry
            return output_object.model_validate_json(result_json)

    except Exception as e:
        print(f"Exception during openai LLM query or parsing: {e}")
        return None


def choose_suitable_claim(claim_decomposition: SummaryDecompositionClaims, apply_transformation_types,
                          transformation_examples, model="openai") -> str:
    class SuitableClaimDecomposition(BaseModel):
        claim_correctness_error_justification: str = Field(
            description="justification which explain wht this claim can perform the specific correctness error transformation type")
        original_claim: str = Field(description="the original correctness error transformation suitable claim")
        correctness_error_type: str = Field(description="the CorrectnessErrorTypes which is the claim can to transform")
        applicable_transformation_text_within_claim: str = Field(
            description="the text in the claim which match the mention error transformation type")
        suggested_transformed_claim: str = Field(
            description="Example of transformed with the correctness error type found on text within original_claim")

    class TransformationSuitableClaims(BaseModel):
        reasoning: str = Field(description="reasoning process describe step by step justify your decision")
        suitable_claims: List[SuitableClaimDecomposition] = Field(description="List of suitable claim decomposition")
        # suitable_claims_decomposition: List[Tuple[str, str]] = Field(
        #     description="List of tuples as follows: The original claim which suitable to perform transformation and justification described in details why this claim suitable, the justification should suggest contains example of apply transformation. Tuple of original claim and justification")

    suitable_claims_decomposition_base_parser = JsonOutputParser(pydantic_object=TransformationSuitableClaims)
    retry_parser = RetryWithErrorOutputParser.from_llm(llm=llm, parser=suitable_claims_decomposition_base_parser,
                                                       max_retries=7)

    system, user = suitable_claim_decomposition_permissive_prompt(claim_decomposition, apply_transformation_types,
                                                                  json.dumps(transformation_examples,
                                                                             ensure_ascii=False),
                                                                  suitable_claims_decomposition_base_parser)

    if model=="openai":
        result = llm_query_to_openai(system_message=system, user_message=user, output_object=TransformationSuitableClaims,parser=retry_parser)
    elif model=="claude":
        result = llm_query_to_claude(system_message=system, user_message=user, output_object=TransformationSuitableClaims,parser=retry_parser)
    else:
        result = llm_query_to_openai(system_message=system, user_message=user, output_object=TransformationSuitableClaims,parser=retry_parser)
    return result


def summary_negative_augmentation(apply_transformation_types,
                                  transformation_examples: dict,
                                  suitable_claims_decomposition_list,
                                  # claims_decomposition_list: List[str] ,
                                  unique_transformation_instructions,
                                  model='openai'):
    # Initialize the OpenAI client directly
    class CorruptedTransformedClaims(BaseModel):
        original_and_corrupted_transformed_claims: List[Tuple[str, int]] = Field(
            description="List of tuples: (corrupted_claim, index_of_original_in_suitable_list)")
        # corrupted_transformed_claims_decomposition: List[Tuple[str, str]] = Field(
        #     description="List of tuples as follows: The correctness error corrupted claim after transformation and justification described in details why how tne transformation applied. Tuple of corrupted transformed claim and justification")
        # original_and_corrupted_transformed_claims: List[Tuple[str, str]] = Field(
        #     description="List of tuples as follows: The correctness error corrupted claim after transformation and the original claim id. Tuple of corrupted transformed claim and original claim id")

    parser = JsonOutputParser(pydantic_object=CorruptedTransformedClaims)

    try:

        system, user = apply_transformation_on_claims_prompt(claims_decomposition_list="",
                                                             suitable_claims_decomposition_list=suitable_claims_decomposition_list,
                                                             unique_transformation_instructions=unique_transformation_instructions,
                                                             apply_transformation_types=apply_transformation_types,
                                                             transformation_examples=transformation_examples,
                                                             output_instructions_parser=parser)
        # Send request to the model
        if model == "openai":
            result = llm_query_to_openai(system_message=system, user_message=user, output_object=CorruptedTransformedClaims)
        elif model == "claude":
            result = llm_query_to_claude(system_message=system, user_message=user, output_object=CorruptedTransformedClaims)
        else:
            result = llm_query_to_openai(system_message=system, user_message=user, output_object=CorruptedTransformedClaims)


        # Process results to ensure original claims come from suitable_claims_decomposition_list
        # Re‐map the “original” index back to the true claim text
        cleaned = []
        for corrupted, idx in result.original_and_corrupted_transformed_claims:
            true_orig = suitable_claims_decomposition_list[idx]['original_claim']
            cleaned.append((corrupted, true_orig))
            result.original_and_corrupted_transformed_claims = cleaned
        return result

    except Exception as e:
        print(f"Error: {e}")
        # Return an empty model instance if there's an erro


def back_translation_and_paraphrasing(faithfulness_summary):
    # Initialize the OpenAI client directly
    class SummaryBackTranslationParaphrasing(BaseModel):
        summary_paraphrasing: str = Field(description="HEBREW summary generative rich paraphrasing")
        summary_back_translation: str = Field(description="HEBREW summary back translation hebrew to english and then again to hebrew")
    parser = JsonOutputParser(pydantic_object=SummaryBackTranslationParaphrasing)

    try:
        system, user = faithfulness_summary_back_translation_and_paraphrasing(faithfulness_summary=faithfulness_summary, output_instructions_parser=parser)
        result = llm_query_to_openai(system_message=system, user_message=user, output_object=SummaryBackTranslationParaphrasing)
        return result

    except Exception as e:
        print(f"Error: {e}")



def summary_halluzaition_negative_augmentation(apply_transformation_types,
                                               transformation_examples: dict,
                                               suitable_claims_decomposition_list,
                                               unique_transformation_instructions):
    """
    Use LLM to apply hallucination transformations to claims.

    Args:
        apply_transformation_types: List of hallucination types to apply
        transformation_examples: Examples dictionary (can be empty for hallucinations)
        suitable_claims_decomposition_list: Claims to transform
        unique_transformation_instructions: Specific instructions for this transformation type

    Returns:
        CorruptedTransformedClaims object with hallucinated claims
    """

    # Initialize the OpenAI client directly
    class CorruptedTransformedClaims(BaseModel):
        corrupted_transformed_claims_decomposition: List[Tuple[str, str]] = Field(
            description="List of tuples as follows: The hallucinated claim after transformation and justification described in details why how the hallucination was applied. Tuple of hallucinated claim and justification")
        original_and_corrupted_transformed_claims: List[Tuple[str, str]] = Field(
            description="List of tuples as follows: The hallucinated claim after transformation and the original claim id. Tuple of hallucinated claim and original claim")

    parser = JsonOutputParser(pydantic_object=CorruptedTransformedClaims)

    try:
        system, user = apply_halluzaition_transformation_on_claims_prompt(
            claims_decomposition_list=[],  # Not needed for hallucinations
            suitable_claims_decomposition_list=suitable_claims_decomposition_list,
            unique_transformation_instructions=unique_transformation_instructions,
            apply_transformation_types=apply_transformation_types,
            transformation_examples=transformation_examples,
            output_instructions_parser=parser
        )

        # Send request to the model
        result = llm_query_to_openai(system_message=system, user_message=user, output_object=CorruptedTransformedClaims)
        return result

    except Exception as e:
        print(f"Error in hallucination transformation: {e}")
        # Return an empty model instance if there's an error
        return None


def summary_claim_decomposition(text: str) -> SummaryDecompositionClaims:
    # TODO: change to summary_decomposition and the level will be configable (sentence, atomic claim, coverage claim, section)
    # Initialize the OpenAI client directly

    claim_decomposition_base_parser = JsonOutputParser(pydantic_object=SummaryDecompositionClaims)
    retry_parser = RetryWithErrorOutputParser.from_llm(llm=llm, parser=claim_decomposition_base_parser, max_retries=7)

    try:
        system, user = claims_decomposition_prompt(text, retry_parser)
        # Send request to the model
        result = llm_query_to_openai(system_message=system, user_message=user, output_object=SummaryDecompositionClaims,
                                     parser=retry_parser)

        if result:
            result._original_text = text  # Assign the original text separately
        return result

    except Exception as e:
        print(f"Error in summary_claim_decomposition: {e}")
        return None


def verify_transformed_summary_grammar(transformed_summary_before_validation: str):
    """
    Verifies the grammatical correctness of transformed claims in Hebrew.

    Args:
        transformed_summary_before_validation: The summary *before* grammar validation

    Returns:
        Tuple[(before, after), success_flag]
    """

    class GrammarVerificationResult(BaseModel):
        reasoning_of_grammar_validation_process: str = Field(
            description="step by step reasoning which explain the grammar modifications changes")
        summary_after_grammar_validation_modifications: str = Field(
            description="The generative summary after grammar validation modifications")

    base_parser = JsonOutputParser(pydantic_object=GrammarVerificationResult)
    retry_parser = RetryWithErrorOutputParser.from_llm(llm=llm, parser=base_parser, max_retries=7)

    try:
        system, user = verify_grammar_transform_summary_prompt(
            transformed_summary_before_validation, retry_parser
        )
        result = llm_query_to_openai(
            system_message=system,
            user_message=user,
            output_object=GrammarVerificationResult,
            parser=retry_parser
        )
        # extract the “after” summary
        grammar_verified = result.summary_after_grammar_validation_modifications

        # compute similarity
        sim = SequenceMatcher(None, transformed_summary_before_validation, grammar_verified).ratio()

        if sim < 0.97:
            print(f"Grammar verification made too many changes (sim={sim:.2f}); keeping original.")
            return (transformed_summary_before_validation, transformed_summary_before_validation), False

        # return (transformed_summary_before_validation, grammar_verified), True
        return {
                   "transformed_summary_before_validation": transformed_summary_before_validation,
                   "grammar_verified": grammar_verified,
                   "verification_reasoning": result.reasoning_of_grammar_validation_process
               }, True

    except Exception as e:
        print(f"Error in grammar verification: {e}")
        return (transformed_summary_before_validation, transformed_summary_before_validation), False


def verify_transformed_claims_grammar(transformed_claims):
    """
    Verifies the grammatical correctness of transformed claims in Hebrew.

    Args:
        transformed_claims: Dictionary of transformed claims from apply_factuality_transformation_on_claims

    Returns:
        Dictionary with validation results and suggested corrections
    """

    # Define the output structure with Pydantic
    class GrammarVerificationResult(BaseModel):
        claim_validations: List[Dict[str, Any]] = Field(
            description="List of dictionaries with grammar validation results for each claim")

    parser = JsonOutputParser(pydantic_object=GrammarVerificationResult)

    try:
        # Create the prompt
        system, user = verify_grammar_transformation_prompt(
            transformed_claims=transformed_claims,
            output_instructions_parser=parser
        )

        # Send request to the model
        result = llm_query_to_openai(system_message=system, user_message=user, output_object=GrammarVerificationResult)

        # Process the results to mark claims as grammatically valid or invalid
        verified_claims = {}
        for original_claim, transformations in transformed_claims.items():
            verified_claims[original_claim] = []

            for transformation in transformations:
                # Find the validation result for this transformation
                validation = next(
                    (v for v in result.claim_validations
                     if v.get("transformed_claim") == transformation["transformed_claim"]),
                    None
                )

                # Copy the transformation and add the validation result
                verified_transformation = transformation.copy()
                if validation:
                    verified_transformation["grammatically_valid"] = validation.get("is_valid", False)
                    if not validation.get("is_valid", False) and "suggested_correction" in validation:
                        verified_transformation["grammar_correction"] = validation["suggested_correction"]

                verified_claims[original_claim].append(verified_transformation)

        return verified_claims

    except Exception as e:
        print(f"Error in grammar verification: {e}")
        # Return the original claims if there's an error
        return transformed_claims
