# CLAUDE_KEY="***REMOVED-API-KEY***"
# curl https://api.anthropic.com/v1/messages \
#     --header "x-api-key: ***REMOVED-API-KEY***" \
#     --header "anthropic-version: 2023-06-01" \
#     --header "content-type: application/json" \
#     --data \
# '{
#     "model": "claude-3-5-sonnet-20241022",
#     "max_tokens": 1024,
#     "messages": [
#         {"role": "user", "content": "Hello, world"}
#     ]

from typing import List, Dict

from pydantic import BaseModel, PrivateAttr, Field
from typing import List

class SummaryDecompositionClaims(BaseModel):
    # claims_decomposition: List[str] = Field(description="List of the claims in the summary after claim decomposition")
    claims_decomposition: List[Dict[int,str]] = Field(description="List of the claims in the summary after claim decomposition and number represent the id or index for each claim")
    total_amount_of_claims: int = Field(description="How many claims decompose from the text")
    _original_text: str = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._original_text = data.get('original_text', '')
    def __str__(self) -> str:
        claims_text = "\n".join([f"- {claim}" for claim in self.claims_decomposition])
        return f"Summary Decomposition ({self.total_amount_of_claims} claims):\n{claims_text}"


# Define nested model for transformation details
class TransformationDetail(BaseModel):
    original_claim: str = Field(description="The original claim from the summary")
    transformation_types_performed: List[str] = Field(description="List of transformation types applied")
    reasoning_process: str = Field(
        description="step by step reasoning in hebrew of how to perform all the error transformations type in the claim")
    negative_claim: str = Field(description="The transformed claim with errors")
    summary_of_transformation_perform: str = Field(description="Description of how the claim was transformed")

    def __str__(self) -> str:
        transformations = ", ".join(self.transformation_types_performed)
        return f"Original: {self.original_claim}\nTransformations: {transformations}\nResult: {self.negative_claim}"


# Define the main data structure
class NegativeAugmentationSummary(BaseModel):
    reasoning_process_of_augmentation: str = Field(
        description="step by step reasoning in Hebrew how to perform **all** the transformation types mention in each separate claim, when perform transformation, the data MUST changed")
    correctness_error_transformation_type: List[str] = Field(
        description="Types of correctness error transformations performed on each claim. NEED to consists all the error transformations types requested")
    correctness_error_transformation_claims: List[str] = Field(description="Claims that were transformed")
    generative_negative_augmentation_summary: str = Field(description="Complete summary with errors introduced")
    total_amount_of_transformed_claims: int = Field(
        description="How many claims transformed and consists correctness error transformation in the new summary")
    detailed_error_transformation: List[TransformationDetail] = Field(
        description="Detailed breakdown of each transformation performed in Hebrew")
    _original_text: str = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._original_text = data.get('original_text', '')

    def __str__(self) -> str:
        error_types = ", ".join(self.correctness_error_transformation_type)

        result = [
            f"Negative Augmentation Summary ({self.total_amount_of_transformed_claims} transformed claims)",
            f"Error types: {error_types}",
            f"\nGenerated negative summary:",
            f"{self.generative_negative_augmentation_summary}",
            f"\nTransformed claims:"
        ]

        for i, claim in enumerate(self.correctness_error_transformation_claims, 1):
            result.append(f"{i}. {claim}")

        return "\n".join(result)


def suitable_claim_decomposition_permissive_prompt(claims_decomposition_list, apply_transformation_types, transformation_examples, output_instructions_parser):
    system_message = f"""
    You are a Hebrew computational linguistics professional Expert.
    Your goal is to chose suitable claims for transformation among article summary atomic claims in **HEBREW**.
    Examine EVERY claim for ALL requested transformation types.
    Do not add words or insert new details into a claim, keep the original text unchanged 
    """

    user_message = f"""
    **Identify Suitable Claims**
        - CRITICAL: Return EVERY claim that can have **AT LEAST ONE** of these transformation types applied: {apply_transformation_types}
        - Use the following examples to understand the transformation types rules and description: **Correctness Corruption Error Transformation Examples**: {transformation_examples}
        - IMPORTANT!: Obey the **Transformation Rules** of each Error Transformation example based on the relevant type!
        - If a claim cannot naturally support **AT LEAST ONE** of required transformations, discard it and select a different claim. You MUST ensure every selected
        - CRITICAL!: Return **ALL** suitable the suitable claims exits in **Claims Decomposition List** that meet the criteria
    **Summary Atomic Claims List - choose claims from that list**: {claims_decomposition_list.claims_decomposition}
    ** 
    the response should be in JSON in the EXACT format:
    {output_instructions_parser.get_format_instructions()}
    """

    return system_message, user_message

def suitable_claim_decomposition_prompt(claims_decomposition_list, apply_transformation_types, transformation_examples, output_instructions_parser):
    system_message = f"""
    You are a Hebrew computational linguistics professional Expert.
    Your goal is to chose suitable claims for transformation among article summary atomic claims in **HEBREW**.
    Do not add words or insert new details into a claim, keep the original text unchanged 
    """

    user_message = f"""
    **Identify Suitable Claims**
        - CRITICAL: Only select claims that can have **ALL** of the following transformation types applied simultaneously: {apply_transformation_types}
        - Use the following examples to understand the transformation types rules and description: **Correctness Corruption Error Transformation Examples**: {transformation_examples}
        - IMPORTANT!: Obey the **Transformation Rules** of each Error Transformation example based on the relevant type!
        - If a claim cannot naturally support **ALL** required transformations, discard it and select a different claim. You MUST ensure every selected
    **Summary Atomic Claims List - choose claims from that list**: {claims_decomposition_list.claims_decomposition}
    ** 
    the response should be in JSON in the EXACT format:
    {output_instructions_parser.get_format_instructions()}
    """

    return system_message, user_message



def claims_decomposition_prompt(text: str, output_instructions_parser):
    # Create the system and user messages
    system_message = """
    You are a Hebrew computational linguistics professional Researcher.
    Your goal is to perform claim decomposition to the following hebrew text into meaningful statements.
    Prioritize broader coverage over granularity - claims can span multiple sentences when they represent connected ideas.
    Preserve contextual relationships between ideas rather than breaking them into minimal atomic facts.
    Do not add words or insert new details into a claim, keep the original text unchanged 
    """

    user_message = f"""
    **Summary**: {text}
    the response should be in JSON in the EXACT format:
    {output_instructions_parser.get_format_instructions()}
    """
    return system_message, user_message



def apply_transformation_on_claims_prompt(claims_decomposition_list, suitable_claims_decomposition_list, unique_transformation_instructions, apply_transformation_types, transformation_examples, output_instructions_parser):

    # Create the system and user messages
    system_message = f"""
    You are a Hebrew computational linguistics professional Researcher.
    Your goal is to perform correctness errors corruption transformation to hebrew atomic claims taken from a hebrew article. Follow these steps carefully:

    **Apply **ALL** Transformations**
        - For claim within **Suitable Claims Decomposition List**, you MUST apply ALL of the error transformation types simultaneously mention **Error Transformation Types**
        - Each transformation must actually change something in the claim.
        - For each error transformation type, change only *ONE* word
        - Before perform the error transformation type, **VERIFY** that the claims **Suitable Claims Decomposition List** can apply all error transformation type mentioned.
        - **When perform transformation on an chosen text based on the error transformation type, the original claim cannot remain unchanged! perform transformation means to change the original data!**

    **IMPORTANT**: Check the section of **Apply Error Transformation Rules Examples** to understand better how perform the hebrew error transformations

    {unique_transformation_instructions}
    """

    user_message = f"""
    **Suitable Claims Decomposition List (indexed)**:
     {[(i, c["original_claim"]) for i, c in enumerate(suitable_claims_decomposition_list)]}
    **Error Transformation Types**: {apply_transformation_types}
    **Apply Error Transformation Rules Examples**: {transformation_examples}
    - IMPORTANT: You MUST apply the transformation on EACH and EVERY claim in the Suitable Claims Decomposition List. Do not skip any claim. Even if only one transformation type is given (like sentence_negation), you should apply it to all claims listed.
    - IMPORTANT!: **OBEY** the **Transformation Rules** of each Error Transformation example based on the relevant type!
    - IMPORTANT2!: **OBEY** the unique transformation instructions for each transformation error type if specify!
    {unique_transformation_instructions}
    the response should be in JSON in the EXACT format:
    {output_instructions_parser.get_format_instructions()}
    """

    return system_message, user_message
# def apply_transformation_on_claims_prompt(claims_decomposition_list, suitable_claims_decomposition_list, unique_transformation_instructions, apply_transformation_types, transformation_examples, output_instructions_parser):
#
#     # Create the system and user messages
#     system_message = f"""
#     You are a Hebrew computational linguistics professional Researcher.
#     Your goal is to perform correctness errors corruption transformation to hebrew atomic claims taken from a hebrew article. Follow these steps carefully:
#
#     **Apply **ALL** Transformations Simultaneously** - תבצע שינוי בטענה רק אם ניתן להפעיל על הטענה **את כל סוגי** השינויים שהוגדרו
#         - For claim within **Suitable Claims Decomposition List**, you MUST apply ALL of the error transformation types simultaneously mention **Error Transformation Types**
#         - Each transformation must actually change something in the claim.
#         - For each error transformation type, change only *ONE* word
#         - Before perform the error transformation type, **VERIFY** that the claims **Suitable Claims Decomposition List** can apply all error transformation type mentioned.
#         - If not - choose another claim from **All Original Claims**
#         - **When perform transformation on an chosen text based on the error transformation type, the original claim cannot remain unchanged! perform transformation means to change the original data!**
#
#     **IMPORTANT**: Check the section of **Apply Error Transformation Rules Examples** to understand better how perform the hebrew error transformations
#
#     {unique_transformation_instructions}
#     """
#
#     user_message = f"""
#     **Suitable Claims Decomposition List**: {suitable_claims_decomposition_list}
#     **Error Transformation Types**: {apply_transformation_types}
#     **All Original Claims** {claims_decomposition_list}
#     **Apply Error Transformation Rules Examples**: {transformation_examples}
#     - IMPORTANT!: **OBEY** the **Transformation Rules** of each Error Transformation example based on the relevant type!
#     - IMPORTANT2!: **OBEY** the unique transformation instructions for each transformation error type if specify!
#     {unique_transformation_instructions}
#     the response should be in JSON in the EXACT format:
#     {output_instructions_parser.get_format_instructions()}
#     """
#
#     return system_message, user_message


def apply_halluzaition_transformation_on_claims_prompt(claims_decomposition_list,
                                                       suitable_claims_decomposition_list,
                                                       unique_transformation_instructions,
                                                       apply_transformation_types,
                                                       transformation_examples,
                                                       output_instructions_parser):
    """Creates prompts for hallucination transformations"""

    # Create the system message
    system_message = f"""
    You are a Hebrew computational linguistics professional Researcher.
    Your goal is to create plausible but factually incorrect claims by transforming hebrew atomic claims. Follow these steps carefully:

    **Apply Hallucination Transformations**
    - For claims in the **Suitable Claims List**, you must apply the hallucination transformation type specified
    - Your transformations should be plausible and contextually appropriate but factually inaccurate
    - Hallucinated information should seem realistic and fit the context of the claim
    - Make minimal changes to the claim - only modify the parts that need hallucination
    - For NER replacements, replace the entity with another entity of the same type that makes sense in context
    - For number addition, add a specific, realistic-sounding number where it would naturally fit in the claim

    {unique_transformation_instructions}
    """

    user_message = f"""
    **Suitable Claims List**: {suitable_claims_decomposition_list}
    **Hallucination Type**: {apply_transformation_types}

    {unique_transformation_instructions}

    The response should be in JSON in the EXACT format:
    {output_instructions_parser.get_format_instructions()}
    """

    return system_message, user_message


def verify_grammar_transform_summary_prompt(corrupted_transformed_summary, output_instructions_parser):
    system_message = """
    You are a Hebrew-grammar expert whose *only* job is to fix **gender/number** agreement errors.

    Rules:
    1. Only adjust morphological agreement (gender & number) between nouns, adjectives, and verbs.
    2. Do *not* change verb tense, aspect, voice, or word order.
    3. Do *not* touch any facts, numbers, names, or semantic content.
    4. Make *only* the minimal change necessary, and no more.
    5. If no such errors exist, return the text exactly as it was.
    6. Reply in JSON exactly as specified—no extra keys, no explanations.
    """

    user_message = f"""
    Verify and correct gender/number agreement errors in the following Hebrew text only:

    \"{corrupted_transformed_summary}\"

    Output JSON schema:
    {output_instructions_parser.get_format_instructions()}
    """

    return system_message, user_message

def perform_multiple_transformations_on_claim(output_instructions_parser):
    system_message = """
    You are a Hebrew error-injection specialist. You will be given a JSON object with two fields:
    - `original_claim`: a Hebrew sentence to corrupt.
    - `applied_transformations`: a list of transformation instructions, each with a `type` and any required parameters.
    
    Your task is to apply *all* of those transformations to `original_claim`—no more, no fewer—producing a single, grammatically valid but error-injected sentence.
    
    Respond with a JSON object containing exactly these keys (no extras):
    - `original_claim`
    - `transformed_claim`
    - `applied_transformations`
    """

    user_message = f"""
    Apply every transformation listed in `applied_transformations` to `original_claim`, preserving Hebrew grammar.  
    Output JSON schema:
    {output_instructions_parser.get_format_instructions()}
    """

    return system_message, user_message


def faithfulness_summary_back_translation_and_paraphrasing(faithfulness_summary, output_instructions_parser):
    system_message = """
    You are an expert bilingual writer and editor.
    
    Your task is to process Hebrew text (article summary) in TWO distinct ways:
    Task Back-Translation - return **HEBREW** summary after Back-Translation
    - Translate the Hebrew text into precise and natural English.
    - Translate the English version back into natural, idiomatic Hebrew.
    - Ensure NO factual data is added, omitted, or altered.
    - Final Hebrew must be factually identical to the original text.
    
    Task Paraphrasing (RICH and NATURAL) - return **HEBREW** summary Paraphrasing
    - Paraphrase the original Hebrew directly into DIFFERENT and VARIED Hebrew phrasing:
    - Significantly change sentence structures and vary expressions.
    - Use synonyms, alternative phrasings, and reordering of ideas where natural.
    - Make the text flow naturally and elegantly, as if edited by a skilled native writer.
    - Avoid simple word swaps only — aim for rich, fluid, and stylistically diverse phrasing.
    - KEEP all factual details strictly unchanged (names, dates, historical facts, claims, etc.).
    - Ensure the final Hebrew version is FACTUALLY IDENTICAL to the original, but EXPRESSED DIFFERENTLY and NATURALLY. 
    
    Both Paraphrasing and Back-Translation are in HEBREW language
    """

    user_message = f"""
    Hebrew summary to perform the back translation and the paraphrasing: {faithfulness_summary}
    Output JSON schema:
    {output_instructions_parser.get_format_instructions()}
    """
    return system_message, user_message
# def verify_grammar_transform_summary_prompt(corrupted_transformed_summary, output_instructions_parser):
#
#     """
#     Sends the corrupted summary to an LLM for grammar verification while preserving intentional errors.
#
#     Args:
#         corrupted_transformed_summary: The summary with intentional errors to check
#
#     Returns:
#         Grammar-verified summary that preserves the intentional errors, or the original if too different
#     """
#
#     # Define the system prompt
#     system_message = """
#     אתה מומחה דקדוק עברי. המשימה שלך היא לתקן בעיות דקדוקיות בלבד בטקסט, מבלי לשנות את התוכן או לתקן שגיאות עובדתיות או מהותיות.
#
#     עקוב אחר ההנחיות הבאות בקפידה:
#     1. תקן רק שגיאות דקדוקיות (כמו התאמת מגדר, מספר, זמני פעלים).
#     2. אל תתקן טעויות תוכן, עובדות לא נכונות, מספרים שגויים, או שמות שגויים.
#     3. אל תוסיף או תמחק מידע.
#     4. שמור על כל השגיאות העובדתיות והמהותיות - הן הוכנסו בכוונה ואסור לשנות אותן.
#     5. בצע שינויים מינימליים בלבד - רק מה שנדרש לתקינות דקדוקית ותחבירית.
#     6. אל תוסיף הערות או הסברים לגבי השינויים שביצעת.
#
#     למשל, אם הטקסט אומר "142% משטח שבאו לא מכוסים מים", אל תתקן את המספר הלא הגיוני 142% או את השם "שבאו", אלא רק דאג שהמילים מתאימות זו לזו מבחינה דקדוקית.
#     דוגמא:
#
#          :שגיאת מילת יחס
#     טקסט שגוי: "החוקרים הגיעו על מסקנה מעניינת."
#     תיקון דקדוקי: "החוקרים הגיעו למסקנה מעניינת."
#     הסבר: הביטוי הנכון הוא "להגיע למסקנה" (עם מילת היחס "ל") ולא "להגיע על מסקנה".
#
#         שגיאת סדר מילים
#     טקסט שגוי: "המחקר החדש בוצע על ידי המדענים המובילים בעולם שלושה."
#     תיקון דקדוקי: "המחקר החדש בוצע על ידי שלושה מהמדענים המובילים בעולם."
#     הסבר: המספר "שלושה" צריך להופיע לפני המושא ("מדענים") ולא אחריו.
#
#     שגיאות מסוג מגדר לשם עצם או פועל ושגיאות מסוג זמן - אסור לך לשנות
#     """
#
#     # Define the user prompt
#     user_message = f"""
#     בדוק את הטקסט הבא מבחינה דקדוקית בלבד. תקן רק שגיאות דקדוק כמו התאמת מגדר, מספר וזמני פעלים, אבל אל תשנה שמות, מספרים, עובדות, או תוכן.
#     הטקסט לבדיקה:
#     "{corrupted_transformed_summary}"
#
#     The response should be in JSON in the EXACT format:
#     {output_instructions_parser.get_format_instructions()}
#     """
#     return system_message, user_message



def verify_grammar_transformation_claim_prompt(transformed_claims, output_instructions_parser):
    """
    Creates prompts to verify that transformed claims are grammatically correct in Hebrew.

    Args:
        transformed_claims: Dictionary of transformed claims from apply_factuality_transformation_on_claims
        output_instructions_parser: Parser for formatting the expected output

    Returns:
        Tuple of (system_message, user_message) for the LLM
    """
    # Extract just the transformed claim texts for verification
    claims_to_verify = []
    for original_claim, transformations in transformed_claims.items():
        for transformation in transformations:
            claims_to_verify.append({
                "transformed_claim": transformation["transformed_claim"],
                "original_claim": original_claim,
                "error_type": transformation["error_type"]
            })

    # Create the system message
    system_message = """
    אתה מומחה דקדוק וניקוד עברי.
    המטרה שלך היא לבדוק טענות בעברית אחרי שעברו טרנספורמציה לוודא שהן תקינות דקדוקית בלבד.

    חשוב מאוד:
    - בדוק רק את התקינות הדקדוקית של הטענות, ולא את האמת העובדתית שלהן
    - אל תבדוק אם הטענה הגיונית או הקשרית, אלא רק אם היא תקינה מבחינת דקדוק ותחביר
    - בדוק התאמה במין ובמספר בין שמות עצם, פעלים ותארים
    - בדוק שימוש נכון במילות יחס
    - בדוק תקינות של מבנים תחביריים ומשפטים
    - אם יש בעיה דקדוקית, ציין בדיוק מהי הבעיה והצע תיקון
    - אם הטענה תקינה דקדוקית, ציין זאת מפורשות
    """

    # Create the user message
    user_message = f"""
    הטענות שיש לבדוק מבחינה דקדוקית:

    {claims_to_verify}

    עבור כל טענה, בדוק אם היא תקינה דקדוקית בלבד (ללא קשר לאמיתות התוכן).

    התשובה צריכה להיות בפורמט JSON המדויק הבא:
    {output_instructions_parser.get_format_instructions()}
    """

    return system_message, user_message
# def summary_negative_augmentation(apply_transformation_types: List[CorrectnessErrorTypes],
#                                   transformation_examples: dict,
#                                   amount_claims_to_apply_transformation: int,
#                                   original_summary: str,
#                                   summary_claims: List[str]) -> NegativeAugmentationSummary:
#     # Initialize the OpenAI client directly
#     parser = JsonOutputParser(pydantic_object=NegativeAugmentationSummary)
#
#
#     # Create the system and user messages
#     system_message = f"""
#     You are a Hebrew computational linguistics professional Researcher.
#     Your goal is to generate summaries with deliberate correctness errors for data augmentation. Follow these steps carefully:
#
#     1. **Select Claims to Transform**
#         - Randomly choose exactly {amount_claims_to_apply_transformation} claims from the decomposed set
#         - DO NOT select the same claim more than once!
#         {summary_claims}
#
#     2. **Identify Suitable Claims**
#         - CRITICAL: Only select claims that can have ALL of the following transformation types applied simultaneously: {apply_transformation_types}
#         - If a claim cannot naturally support all required transformations, discard it and select a different claim. You MUST ensure every selected claim undergoes ALL requested transformations. If fewer than {amount_claims_to_apply_transformation} claims are valid, select fewer claims rather than applying incomplete transformations.
#
#     3. **Select Claims to Transform**
#         - Choose exactly {amount_claims_to_apply_transformation} claims that can support ALL the required transformation types.
#         - If you cannot find enough claims that support ALL transformation types, select fewer claims rather than selecting unsuitable ones.
#
#     4. **Apply **ALL** Transformations Simultaneously** - תבצע שינוי בטענה רק אם ניתן להפעיל על הטענה **את כל סוגי** השינויים שהוגדרו
#         - For each selected claim, you MUST apply ALL of these transformation types simultaneously: {apply_transformation_types}
#         - Each transformation must actually change something in the claim.
#         - **When perform transformation on an chosen text based on the error type, the original text cannot remain unchanged! perform transformation means to change the original data!**
#
#     5. **Maintain Claim Integrity**
#         - Do not add words or insert new details into a claim. Make only the necessary edits to introduce the specific error dictated by the transformation.
#         - The claim after transformation should remain logically fluent and coherence
#
#     6. **Reconstruct the Summary**
#        - Replace the original claims you transformed with their modified versions.
#        - Leave any non-chosen claims unchanged.
#     """