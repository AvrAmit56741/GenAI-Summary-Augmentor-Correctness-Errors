# TODO: Add more hebrew specific error
# TODO: Complete config file
# TODO: in the function apply_transformation, filter failed transformation where the original similar to transformed - V
# TODO: Fix bug of decompose - V
# TODO: Add verification of all negative summary
# TODO: Optimized prompts and examples
# TODO: Remove overlap claims_after_transformation
# TODO: NER result consits only complete words without wired symbols (#...) - V
# TODO: provide only the relevant example of the error_type to LLM prompt - V
# TODO: Example to LLM
# TODO: when perform sentence negation no other transformation occurs - V

from summary_augmentation.utils import CorrectnessErrorTypes, ClaimDecompositionMethod

MAXIMUM_ALLOWED_NEGATIVE_SUMMARIES_SIMILARITY_THRESHOLD = 0.9  # Control
PERCENTAGE_TRANSFORMATIONS_PERFORM_ON_NEGATIVE_SUMMARY = 93  # Control how much corrupted the negative summary will be (how much error transformations perform on in)
CLAIM_DECOMPOSITION_METHOD = ClaimDecompositionMethod.SENTENCES

# Main configuration - just set True/False for each transformation type
TRANSFORMATION_CONFIG = {
    "hebrew_llm_transformations": {
        CorrectnessErrorTypes.NOUN_GENDER_SWAP: False,
        CorrectnessErrorTypes.VERB_GENDER_SWAP: False,
        CorrectnessErrorTypes.VERB_TENSE_SWAP: True,
        CorrectnessErrorTypes.DIRECTIONAL_PREPOSITION_SWAP: True,
        CorrectnessErrorTypes.PRONOUN_REFERENCE_ERRORS: True,
        CorrectnessErrorTypes.SPECIFICITY_SHIFT_ERRORS: True,
        CorrectnessErrorTypes.HOMOGRAPHIC_GENDER_ERRORS: True,
        CorrectnessErrorTypes.IMPERSONAL_TO_PERSONAL_VERB_ERRORS: True,
    },
    "general_llm_transformations": {
        CorrectnessErrorTypes.SENTENCE_NEGATION: True,
        CorrectnessErrorTypes.MEASURE_UNIT_SWAP: True,
        CorrectnessErrorTypes.ENTITY_EVENT_SWAP: True,
        CorrectnessErrorTypes.ENTITY_DATE_SWAP: True,
    },
    "ner_based_transformation": {
        CorrectnessErrorTypes.ENTITY_PERSON_SWAP: True,
        CorrectnessErrorTypes.ENTITY_LOCATION_SWAP: True,
        CorrectnessErrorTypes.ENTITY_TITLE_SWAP: True,
        CorrectnessErrorTypes.ENTITY_ORGANIZATION_SWAP: True,
    }
}


def get_enabled_transformations_as_lists():
    """
    Get enabled transformations as lists (for all_suitable_transformation_possible function).

    Returns:
        tuple: (hebrew_llm_transformations, general_llm_transformations, ner_based_transformation)
    """
    hebrew_llm_transformations = [
        error_type for error_type, enabled in TRANSFORMATION_CONFIG["hebrew_llm_transformations"].items()
        if enabled
    ]

    general_llm_transformations = [
        error_type for error_type, enabled in TRANSFORMATION_CONFIG["general_llm_transformations"].items()
        if enabled
    ]

    ner_based_transformation = [
        error_type for error_type, enabled in TRANSFORMATION_CONFIG["ner_based_transformation"].items()
        if enabled
    ]

    return hebrew_llm_transformations, general_llm_transformations, ner_based_transformation


def get_enabled_transformations_as_dicts():
    """
    Get enabled transformations as dictionaries with empty lists (for apply_factuality_transformation_on_claims function).

    Returns:
        tuple: (hebrew_llm_transformations, general_llm_transformations)
    """
    hebrew_llm_transformations = {
        error_type: [] for error_type, enabled in TRANSFORMATION_CONFIG["hebrew_llm_transformations"].items()
        if enabled
    }

    general_llm_transformations = {
        error_type: [] for error_type, enabled in TRANSFORMATION_CONFIG["general_llm_transformations"].items()
        if enabled
    }

    return hebrew_llm_transformations, general_llm_transformations


# Legacy support - individual getters if needed
def get_hebrew_llm_transformations_list():
    return [error_type for error_type, enabled in TRANSFORMATION_CONFIG["hebrew_llm_transformations"].items() if
            enabled]


def get_general_llm_transformations_list():
    return [error_type for error_type, enabled in TRANSFORMATION_CONFIG["general_llm_transformations"].items() if
            enabled]


def get_ner_based_transformation_list():
    return [error_type for error_type, enabled in TRANSFORMATION_CONFIG["ner_based_transformation"].items() if enabled]


def get_hebrew_llm_transformations_dict():
    return {error_type: [] for error_type, enabled in TRANSFORMATION_CONFIG["hebrew_llm_transformations"].items() if
            enabled}


def get_general_llm_transformations_dict():
    return {error_type: [] for error_type, enabled in TRANSFORMATION_CONFIG["general_llm_transformations"].items() if
            enabled}