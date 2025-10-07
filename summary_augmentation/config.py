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
# TODO: NER replacement is not too similar - V
# TODO: Add rule: Homographic gender errors without NER (name swap)

# 12.09
# TODO: Add score filtering to NER -
# TODO: Add another Hebrew NER model -
# TODO: UI evidence extraction - V
# TODO: Add json dump transformation dump for optimzation - V
# TODO: Generate summaries using close source ollama
# TODO: Optimze trasnformation
# TODO: ask cursor to create zeroshot prompts
# TODO: add syntax\grammer validation
# Add an option to download the error dump


from summary_augmentation.utils import CorrectnessErrorTypes, ClaimDecompositionMethod
from dataclasses import dataclass
MAXIMUM_ALLOWED_NEGATIVE_SUMMARIES_SIMILARITY_THRESHOLD = 0.85  # Control
PERCENTAGE_TRANSFORMATIONS_PERFORM_ON_NEGATIVE_SUMMARY = 90  # Control how much corrupted the negative summary will be (how much error transformations perform on in)
CLAIM_DECOMPOSITION_METHOD = ClaimDecompositionMethod.SENTENCES

# Main configuration - just set True/False for each transformation type
TRANSFORMATION_CONFIG = {
    "hebrew_llm_transformations": {
        # NEW
        CorrectnessErrorTypes.HEBREW_ROOT_PATTEN_CONFUSION: True,
        CorrectnessErrorTypes.IDIOM_COLLOCATION_CORRUPTION: True,
        CorrectnessErrorTypes.MORPHOLOGICAL_CONNECTIVE_CONFUSION: True,

        # OLD
        CorrectnessErrorTypes.NOUN_GENDER_SWAP: False,
        CorrectnessErrorTypes.VERB_GENDER_SWAP: False,
        CorrectnessErrorTypes.VERB_TENSE_SWAP: True,
        CorrectnessErrorTypes.DIRECTIONAL_PREPOSITION_SWAP: False,
        CorrectnessErrorTypes.PRONOUN_REFERENCE_ERRORS: True,
        CorrectnessErrorTypes.SPECIFICITY_SHIFT_ERRORS: True,
        CorrectnessErrorTypes.CONSTRUCT_STATE_CONFUSION: True,
        CorrectnessErrorTypes.HOMOGRAPHIC_GENDER_ERRORS: False,
        CorrectnessErrorTypes.IMPERSONAL_TO_PERSONAL_VERB_ERRORS: False,
        CorrectnessErrorTypes.INCREMENTAL_TEMPORAL_STATE_MUTATIONS: False,
        CorrectnessErrorTypes.EVIDENTIALITY_SOURCE_ATTRIBUTION_COLLAPSE: False,
        CorrectnessErrorTypes.VERB_TEMPLATE_AGENCY_REVERSAL: False,
        CorrectnessErrorTypes.DOUBLE_VERB_DEPENDENCY_ERRORS: False,
        CorrectnessErrorTypes.SUBJECT_VERB_PERSON_MISMATCH: False

        # Probably Removed
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


@dataclass
class WorkflowConfig:
    """Configuration for pipeline workflow"""
    # Model selection for different stages
    # choose_suitable_claim_model: str = "gemini"  # Model for choosing suitable claims
    # apply_factuality_transformation_model_hebrew: str = "gemini"  # Model for applying transformations
    # apply_factuality_transformation_model_general: str = "openai"  # Model for applying transformations
    # verification_model: str = "openai"  # Model for grammar verification

    # Model selection for different stages
    choose_suitable_claim_model: str = "openai"  # Model for choosing suitable claims
    apply_factuality_transformation_model_hebrew: str = "openai"  # Model for applying transformations
    apply_factuality_transformation_model_general: str = "openai"  # Model for applying transformations
    verification_model: str = "openai"  # Model for grammar verification


    # Processing limits
    max_candidate_summaries: int = 50
    max_final_summaries: int = 10

    # NER Configuration
    ner_min_score: float = 0.6
    ner_model: str = "dicta-il/dictabert-ner"

    # Logging
    log_llm_calls: bool = True
    log_directory: str = "llm_api_logs"


# Default workflow configuration
WORKFLOW_CONFIG = WorkflowConfig()
def get_workflow_config() -> WorkflowConfig:
    """Get current workflow configuration"""
    return WORKFLOW_CONFIG

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