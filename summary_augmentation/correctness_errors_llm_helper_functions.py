# Define the main data structure
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers.retry import RetryWithErrorOutputParser
from langchain.prompts import ChatPromptTemplate
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
    suitable_claim_decomposition_permissive_prompt,faithfulness_summary_back_translation_and_paraphrasing, correctness_errors_finder_prompt
import datetime
import os
from summary_augmentation.logger_config import setup_logger
from summary_augmentation.logger_config import LoggerFactory
logger = LoggerFactory.get_logger(__name__)
# Create llm_api_logs directory if it doesn't exist
os.makedirs("llm_api_logs", exist_ok=True)

api_logger = setup_logger(
    name="llm_api_calls",
    console_output=False,  # Don't spam console
    file_level="DEBUG",
    log_file=f"llm_api_logs/llm_api_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
)

# Define the main data structure
import google.generativeai as genai

# Alternative Gemini keys are kept locally in .env

from summary_augmentation.api_keys import GEMINI_API_KEY  # loaded from .env
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-pro')
#gemini-2.0-flash

# Define OpenAI API credentials
# EYAL
from summary_augmentation.api_keys import OPENAI_API_KEY  # loaded from .env
from summary_augmentation.api_keys import CLAUDE_API_KEY  # loaded from .env
client_claude = Anthropic(api_key=CLAUDE_API_KEY)
# AMIT
# OPENAI_API_KEY = "<redacted - see .env>"
llm = ChatOpenAI(
    #model="gpt-4o",
    model="gpt-5-mini",
    temperature=0.0,
    max_tokens=2000,
    api_key=OPENAI_API_KEY,
    response_format="json_object"
)
client = OpenAI(api_key=OPENAI_API_KEY)


# def _strip_json_fences(s: str) -> str:
#     """
#     Remove Markdown code fences (```json ... ``` or ```) from a Claude response.
#     """
#     # Strip leading ```json or ```
#     s = re.sub(r'^\s*```(?:json)?\s*', '', s, flags=re.IGNORECASE)
#     # Strip trailing ```
#     s = re.sub(r'\s*```\s*$', '', s)
#     return s.strip()
#
# def llm_query_to_claude(
#     system_message: str,
#     user_message: str,
#     output_object: Type[BaseModel],
#     parser=None
# ):
#     """
#     Query Claude AI with system and user messages, expecting a structured JSON response.
#
#     Args:
#         system_message: The system prompt/instructions.
#         user_message: The user's message/query.
#         output_object: Pydantic BaseModel class for output validation.
#         parser: Optional retry parser for handling parsing errors.
#
#     Returns:
#         Parsed object of type output_object or None if failed.
#     """
#     # Combine user message with explicit JSON-only instruction
#     enhanced_user_message = f"{user_message}\n\nPlease respond with valid JSON only."
#
#     try:
#         # Send request to Claude
#         response = client_claude.messages.create(
#             #model="claude-3-7-sonnet-20250219",
#             model="claude-sonnet-4-20250514",
#             max_tokens=2500,
#             temperature=0.1,
#             system=system_message,
#             messages=[{"role": "user", "content": enhanced_user_message}]
#         )
#
#         # Raw text from Claude
#         raw = response.content[0].text
#         print("Raw response from Claude:\n", raw)
#
#         # Clean out any Markdown fences
#         clean_json = _strip_json_fences(raw)
#         print("Cleaned JSON to parse:\n", clean_json)
#
#         # Attempt parsing with retry parser if given
#         if parser:
#             try:
#                 return parser.parse(clean_json)
#             except Exception as retry_err:
#                 print(f"Retry parser failed: {retry_err}")
#                 # Fallback to direct Pydantic validation
#                 return output_object.model_validate_json(clean_json)
#         else:
#             # Direct Pydantic validation
#             return output_object.model_validate_json(clean_json)
#
#     except Exception as e:
#         print(f"Exception during Claude LLM query or parsing: {e}")
#         return None

import re
from typing import Type
from pydantic import BaseModel
from langchain.prompts import ChatPromptTemplate  # ✅ added

def _strip_json_fences(s: str) -> str:
    """
    Remove Markdown code fences (```json ... ``` or ```) from a Claude response.
    """
    # Strip leading ```json or ```
    s = re.sub(r'^\s*```(?:json)?\s*', '', s, flags=re.IGNORECASE | re.DOTALL)  # ✅ DOTALL
    # Strip trailing ```
    s = re.sub(r'\s*```\s*$', '', s, flags=re.DOTALL)  # ✅ DOTALL
    return s.strip()

def llm_query_to_claude(
    system_message: str,
    user_message: str,
    output_object: Type[BaseModel],
    parser=None
):
    """
    Query Claude AI with system and user messages, expecting a structured JSON response.
    """
    logger.info("Starting Claude API call")
    logger.debug(f"System message length: {len(system_message)} chars")
    logger.debug(f"User message length: {len(user_message)} chars")
    # ✅ include a simple end marker so we can stop exactly at the JSON
    enhanced_user_message = (
        f"{user_message}\n\n"
        "Please respond with valid JSON only. End output with </END_JSON>."
    )

    try:
        # Send request to Claude
        response = client_claude.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=3000,
            temperature=0.7,
            system=system_message,
            messages=[{"role": "user", "content": enhanced_user_message}],
            stop_sequences=["</END_JSON>"]  # ✅ added
        )

        # Raw text from Claude
        raw = response.content[0].text
        print("Raw response from Claude:\n", raw)

        # Clean out any Markdown fences
        clean_json = _strip_json_fences(raw)
        print("Cleaned JSON to parse:\n", clean_json)

        # Build a PromptValue mirroring the exact prompt we sent  ✅ added
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", "{system}"),
            ("user", "{user}")
        ])
        prompt_value = chat_prompt.format_prompt(
            system=system_message,
            user=enhanced_user_message
        )

        # Attempt parsing with retry parser if given
        if parser:
            try:
                # ✅ use parse_with_prompt instead of parse(...)
                return parser.parse_with_prompt(clean_json, prompt_value)
            except Exception as retry_err:
                print(f"Retry parser failed: {retry_err}")
                # Fallback to direct Pydantic validation
                return output_object.model_validate_json(clean_json)
        else:
            # Direct parsing without retry
            return output_object.model_validate_json(clean_json)

    except Exception as e:
        print(f"Exception during Claude LLM query or parsing: {e}")
        return None

def llm_query_to_gemini(system_message, user_message, output_object: Type[BaseModel], parser=None):
    logger.info("Starting Gemini API call")
    logger.debug(f"System message length: {len(system_message)} chars")
    logger.debug(f"User message length: {len(user_message)} chars")
    request_data = f"prompt: {system_message}\n {user_message}"
    api_logger.info(f"REQUEST: {json.dumps(request_data, ensure_ascii=False)}")

    #print(f"prompt: {system_message}\n {user_message}")
    try:
        response = model.generate_content(f"{system_message}\n{user_message}")

        # ✅ Robust extraction when response.text is unavailable
        try:
            result_json = response.text.strip()
        except Exception:
            result_json = ""
        if not result_json:
            # Fallback: concatenate any text parts from the first candidate
            candidates = getattr(response, "candidates", []) or []
            if candidates:
                first = candidates[0]
                # optional: log finish_reason for debugging
                fr = getattr(first, "finish_reason", None)
                if fr is not None:
                    print(f"Gemini finish_reason: {fr}")
                content = getattr(first, "content", None)
                parts = getattr(content, "parts", []) if content else []
                texts = []
                for p in parts or []:
                    t = getattr(p, "text", None)
                    if t:
                        texts.append(t)
                result_json = "".join(texts).strip()

        if not result_json:
            raise ValueError("Empty response from Gemini (no text Parts).")

        clean_json = _strip_json_fences(result_json)
        print("Cleaned JSON to parse:\n", clean_json)
        request_data_res = f"RESPONSE: {clean_json}"
        api_logger.info(f"RESPONSE: {json.dumps(request_data_res, ensure_ascii=False)}")

        if parser:
            # mirror the exact prompt we sent (system + user) for retry parsers
            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", "{system}"),
                ("user", "{user}"),
            ])
            prompt_value = chat_prompt.format_prompt(system=system_message, user=user_message)
            try:
                # required for RetryWithErrorOutputParser
                return parser.parse_with_prompt(clean_json, prompt_value)
            except Exception as retry_err:
                print(f"Retry parser failed: {retry_err}")
                return output_object.model_validate_json(clean_json)
        else:
            return output_object.model_validate_json(clean_json)

    except Exception as e:
        print(f"Exception during gemini LLM query or parsing: {e}")
        return None


def llm_query_to_openai(system_message, user_message, output_object: Type[BaseModel], parser=None, openai_model="gpt-4.1-2025-04-14"):
    # gpt-4o-mini-2024-07-18
    # gpt-4.1-2025-04-14"
    logger.info(f"Starting GPT API call with model: {openai_model}")
    logger.debug(f"System message length: {len(system_message)} chars")
    logger.debug(f"User message length: {len(user_message)} chars")
    print(f"prompt: {system_message}\n {user_message}")
    try:
        response = client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            # temperature=0.0,
            # max_tokens=2000,
            temperature=0.1,
            max_tokens=3000,
            response_format={"type": "json_object"}
        )
        # Extract the raw response
        result_json = response.choices[0].message.content
        print("\nRaw response:")
        print(result_json.encode('utf-8').decode('utf-8'))

        if parser:
            # 🔑 Build a PromptValue that mirrors the exact prompt sent to the LLM
            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", "{system}"),
                ("user", "{user}"),
            ])
            prompt_value = chat_prompt.format_prompt(system=system_message, user=user_message)

            try:
                # Use parse_with_prompt so RetryWithErrorOutputParser can retry with the original prompt
                return parser.parse_with_prompt(result_json, prompt_value)
            except Exception as parsing_error:
                print(f"Retry parser failed: {parsing_error}")
                # last-resort: directly validate JSON into your Pydantic model
                return output_object.model_validate_json(result_json)
        else:
            return output_object.model_validate_json(result_json)

    except Exception as e:
        print(f"Exception during openai LLM query or parsing: {e}")
        return None
        # Use the retry parser if provided, otherwise fallback to direct validation
    #     if parser:
    #         # The retry parser will handle any parsing errors and retry if needed
    #         try:
    #             result = parser.parse(result_json)
    #             return result
    #         except Exception as parsing_error:
    #             # If the retry parser fails after all retries, log it
    #             print(f"Retry parser failed: {parsing_error}")
    #             # Fallback to direct parsing as a last resort
    #             return output_object.model_validate_json(result_json)
    #     else:
    #         # Direct parsing without retry
    #         return output_object.model_validate_json(result_json)
    #
    # except Exception as e:
    #     print(f"Exception during openai LLM query or parsing: {e}")
    #     return None


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

    logger.debug(f"Transformation types: {[t.value if hasattr(t, 'value') else str(t) for t in apply_transformation_types]}")
    suitable_claims_decomposition_base_parser = JsonOutputParser(pydantic_object=TransformationSuitableClaims)
    retry_parser = RetryWithErrorOutputParser.from_llm(llm=llm, parser=suitable_claims_decomposition_base_parser,
                                                       max_retries=7)

    system, user = suitable_claim_decomposition_permissive_prompt(claim_decomposition, apply_transformation_types,
                                                                  json.dumps(transformation_examples,
                                                                             ensure_ascii=False),
                                                                  suitable_claims_decomposition_base_parser)
    #logger.info(f"Start suitable_transformation_claims - find  suitable transformation claims")
    print(f"Perform llm query on choose_suitable_claim with model {model}")
    if model=="openai":
        result = llm_query_to_openai(system_message=system, user_message=user, output_object=TransformationSuitableClaims,parser=retry_parser)
    elif model=="claude":
        result = llm_query_to_claude(system_message=system, user_message=user, output_object=TransformationSuitableClaims,parser=retry_parser)
    elif model=="gemini":
        result = llm_query_to_gemini(system_message=system, user_message=user, output_object=TransformationSuitableClaims,parser=retry_parser)
    else:
        result = llm_query_to_openai(system_message=system, user_message=user, output_object=TransformationSuitableClaims,parser=retry_parser)
    if isinstance(result, dict):
        result = TransformationSuitableClaims.model_validate(result)

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
        elif model == "gemini":
            result = llm_query_to_gemini(system_message=system, user_message=user, output_object=CorruptedTransformedClaims)
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

def correctness_error_finder(source_article, summary):
    class CorrectnessError(BaseModel):
        correctness_error_name: str = Field(description="HEBREW unique MRL error name")
        correctness_error_reasoning: str = Field(description="Reasoning step by step explanation of the unique Hebrew morphology of this error, the HEBREW factuality inconsistency between the source article and summary")
        mrl_unique_hebrew_features: str = Field(description="List of the key Hebrew unique MRL features based on Hebrew MRL rules and characteristics: בניינים, פעלים, מגדר, סמיכות ועוד")
        why_its_factual_error: str =Field(description="Explain why it factual error, how this error break the fact from the article")
        evidence_from_summary: str = Field(description="evidence of the Hebrew error from the summary")
        evidence_from_source_article: str = Field(description="evidence of the correct fact from the source article")
        severity: str = Field(description="Severity level (1-5) where 5 indicates critical factual corruption that fundamentally changes core facts (identity confusion, role reversal, ownership errors), 4 indicates significant factual distortion, 3 moderate inconsistency, 2 minor deviation, and 1 negligible impact - prioritize Hebrew MRL-specific errors with high factual impact.")

    class TotalCorrectnessErrors(BaseModel):
        summary_article_heb_correctness_errors: List[CorrectnessError] = Field(description="list of all Hebrew CorrectnessError found in the summary")


    correctness_error_finder_base_parser = JsonOutputParser(pydantic_object=TotalCorrectnessErrors)
    retry_parser = RetryWithErrorOutputParser.from_llm(llm=llm, parser=correctness_error_finder_base_parser,max_retries=7)
    #parser = RetryWithErrorOutputParser(pydantic_object=TotalCorrectnessErrors)

    try:
        system, user = correctness_errors_finder_prompt(source_article=source_article, claims_decomposition_list=summary,  output_instructions_parser=retry_parser)
        #result = llm_query_to_gemini(system_message=system, user_message=user, output_object=TotalCorrectnessErrors)
        result = llm_query_to_claude(system_message=system, user_message=user, output_object=TotalCorrectnessErrors)
        #result = llm_query_to_openai(system_message=system, user_message=user, output_object=TotalCorrectnessErrors, parser=None, openai_model="gpt-5-mini")
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

    logger.info("Starting summary claim decomposition")
    logger.debug(f"Input text length: {len(text)} characters")
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
