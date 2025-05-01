from typing import List, Dict
from openai import OpenAI
from pydantic import BaseModel, Field
import json
from langchain_core.output_parsers import JsonOutputParser
import summary_augmentation.correctness_errors_examples as correctness_errors_examples
general_error_examples = correctness_errors_examples.general_error_examples
hebrew_error_examples = correctness_errors_examples.hebrew_error_examples
import pandas as pd
from pydantic import BaseModel, PrivateAttr, Field
from typing import List
from enum import Enum

# Define OpenAI API credentials
OPENAI_API_KEY = "***REMOVED-API-KEY***"
client = OpenAI(api_key=OPENAI_API_KEY)


class CorrectnessErrorTypes(Enum, str):
    GENDER_SWAP = "gender_swap"
    MEASURE_UNIT_SWAP = "measure_unit_swap"
    NUMBER_SWAP = "number_swap"

# Define the main data structure
class SummaryDecompositionClaims(BaseModel):
    claim_decomposition: List[str] = Field(description="List of the claims in the summary after claim decomposition")
    total_amount_of_claims: int = Field(description="How many claims decompose from the text")
    _original_text: str = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._original_text = data.get('original_text', '')

# Define nested model for transformation details
class TransformationDetail(BaseModel):
    original_claim: str = Field(description="The original claim from the summary")
    negative_claim: str = Field(description="The transformed claim with errors")
    transformation_types_performed: List[str] = Field(description="List of transformation types applied")
    summary_of_transformation_perform: str = Field(description="Description of how the claim was transformed")


# Define the main data structure
class NegativeAugmentationSummary(BaseModel):
    claim_decomposition: List[str] = Field(description="List of the claims in the summary after claim decomposition")
    reasoning_process_of_choose_claim: str = Field(description="step by step reasoning how choose the claim which can perform **all** the transformation types mention in each separate claim, when perform transformation, the data MUST changed")
    reasoning_process_of_augmentation: str = Field(description="step by step reasoning how to perform **all** the transformation types mention in each separate claim, when perform transformation, the data MUST changed")
    correctness_error_transformation_type: List[str] = Field(description="Types of correctness error transformations performed on each claim")
    correctness_error_transformation_claims: List[str] = Field(description="Claims that were transformed")
    generative_negative_augmentation_summary: str = Field(description="Complete summary with errors introduced")
    total_amount_of_transformed_claims: int = Field(description="How many claims transformed and consists correctness error transformation in the new summary")
    detailed_error_transformation: List[TransformationDetail] = Field(
        description="Detailed breakdown of each transformation performed"
    )


def summary_claim_decomposition(text: str) -> SummaryDecompositionClaims:
    # Initialize the OpenAI client directly

    claim_decomposition_parser = JsonOutputParser(pydantic_object=SummaryDecompositionClaims)

    # Create the system and user messages
    system_message = """
    You are a Hebrew computational linguistics professional Researcher.
    Your goal is to perform claim decomposition to the following hebrew text sentences into individual facts.
    Do not add words or insert new details into a claim, keep the original text unchanged 
    """

    user_message = f"""
    **Summary**: {text}
    the response should be in JSON in the EXACT format:
    {claim_decomposition_parser.get_format_instructions()}
    """
    try:
        # Send request to the model
        print(claim_decomposition_parser.get_format_instructions())
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.0,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )

        # Extract and parse the JSON response
        result_json = response.choices[0].message.content
        print("\nRaw response:")
        print(result_json)

        # Parse into Pydantic model
        result = SummaryDecompositionClaims.parse_raw(result_json)
        return result

    except Exception as e:
        print(f"Error: {e}")
        # Return an empty model instance if there's an error
        return SummaryDecompositionClaims(
            claim_decomposition=[],
            total_amount_of_claims=0,  # Add this missing required field
        )



def summary_negative_augmentation(apply_transformation_types: List[CorrectnessErrorTypes],transformation_examples: dict, amount_claims_to_apply_transformation: int, summary: str) -> NegativeAugmentationSummary:
    # Initialize the OpenAI client directly
    client = OpenAI(api_key=OPENAI_API_KEY)
    parser = JsonOutputParser(pydantic_object=NegativeAugmentationSummary)
    transformation_types_list = [t.strip() for t in apply_transformation_types.split(",")]

    # Create the system and user messages
    system_message = """
    You are a Hebrew computational linguistics professional Researcher.
    Your goal is to generate summaries with deliberate correctness errors for data augmentation. Follow these steps carefully:
    **Apply **ALL** Transformations Simultaneously**
    1. **Claim Decomposition**  
        - decompose the following summary sentences into individual facts.
    
    2. **Select Claims to Transform**  
        - Randomly choose exactly {amount_claims_to_apply_transformation} claims from the decomposed set.
        
    3. **Identify Suitable Claims**
        - CRITICAL: Only select claims that can have ALL of the following transformation types applied simultaneously: {apply_transformation_types}
        - If a claim cannot support ALL of the required transformation types, DO NOT select it.
     
    4. **Select Claims to Transform**  
        - Choose exactly {amount_claims_to_apply_transformation} claims that can support ALL the required transformation types.
        - If you cannot find enough claims that support ALL transformation types, select fewer claims rather than selecting unsuitable ones.
               
    5. **Apply **ALL** Transformations Simultaneously**
        - For each selected claim, you MUST apply ALL of these transformation types simultaneously: {apply_transformation_types}
        - Each transformation must actually change something in the claim.
        - **When perform transformation on an chosen text based on the error type, the original text cannot remain unchanged! perform transformation means to change the original data!**

    6. **Maintain Claim Integrity**  
        - Do not add words or insert new details into a claim. Make only the necessary edits to introduce the specific error dictated by the transformation.

    7. **Reconstruct the Summary**  
       - Replace the original claims you transformed with their modified versions.  
       - Leave any non-chosen claims unchanged.
    """

    user_message = f"""
    **Claims Transformation amount**: {amount_claims_to_apply_transformation}
    **Transformation types to apply to EACH claim**: {transformation_types_list}
        **Apply **ALL** Transformations Simultaneously**
        For example, with "gender_swap, measure_unit":
        1. You must change BOTH gender terms AND measurement units in the SAME claim
        2. Do NOT select claims that don't have BOTH gender terms AND measurement units

    **Examples**: {json.dumps(transformation_examples, ensure_ascii=False)}
    **Summary**: {summary}
    the response should be in JSON in the EXACT format:
    {parser.get_format_instructions()}
    """
    try:
        # Send request to the model
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.0,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )

        # Extract and parse the JSON response
        result_json = response.choices[0].message.content
        print("\nRaw response:")
        print(result_json)

        # Parse into Pydantic model
        result = NegativeAugmentationSummary.model_validate_json(result_json)
        return result

    except Exception as e:
        print(f"Error: {e}")
        # Return an empty model instance if there's an error
        return NegativeAugmentationSummary(
            claim_decomposition=[],
            reasoning_process_of_choose_claim="",  # Add this missing required field
            reasoning_process_of_augmentation="",
            correctness_error_transformation_type=[],
            correctness_error_transformation_claims=[],
            generative_negative_augmentation_summary="",
            total_amount_of_transformed_claims=0,  # Add this missing required field
            detailed_error_transformation=[]
        )


# Add this as the main function at the end of your file
def main(input_csv_path, output_csv_path, num_rows_to_transform, transformation_types="gender_swap, measure_unit"):
    """
    Process NLI dataset by applying negative augmentation to hypothesis statements.

    Args:
        input_csv_path (str): Path to input CSV with columns: index, hypothesis, premise, label
        output_csv_path (str): Path where the augmented CSV will be saved
        num_rows_to_transform (int): Number of rows to transform
        transformation_types (str): Comma-separated transformation types to apply
    """
    # Load the CSV data
    df = pd.read_csv(input_csv_path)

    # Take the first n rows to transform
    if num_rows_to_transform < len(df):
        rows_to_transform = df.sample(n=num_rows_to_transform)
    else:
        rows_to_transform = df  # Use all rows if requested more than available

    # Prepare transformation examples
    selected_transformations = transformation_types.split(",")
    general_filtered_examples = {key: general_error_examples[key] for key in selected_transformations if
                                 key in general_error_examples}
    hebrew_filtered_examples = {key: hebrew_error_examples[key] for key in selected_transformations if
                                key in hebrew_error_examples}
    filtered_examples = {**general_filtered_examples, **hebrew_filtered_examples}

    # Create a list to store new rows
    new_rows = []


    # Process each selected row
    for _, row in rows_to_transform.iterrows():
        summary = row['hypothesis']
        premise = row['premise']

        claim_decomposition = summary_claim_decomposition(text=summary)
        print(claim_decomposition)
        # Apply negative augmentation
        result = summary_negative_augmentation(
            apply_transformation_types=
            [CorrectnessErrorTypes.GENDER_SWAP,
             CorrectnessErrorTypes.MEASURE_UNIT_SWAP,
             CorrectnessErrorTypes.NUMBER_SWAP],
            transformation_examples=filtered_examples,
            amount_claims_to_apply_transformation=3,
            summary=summary
        )

        # Create new row with the transformed data
        new_row = {
            'index': len(df) + len(new_rows) + 1,  # Generate new index
            'premise': premise,  # Keep premise unchanged
            'hypothesis': result.generative_negative_augmentation_summary,  # Use the generated negative summary
            'label': 'contradiction'  # Set label to contradiction
        }

        new_rows.append(new_row)
        print(f"Processed row {len(new_rows)}/{num_rows_to_transform}")

    # Create DataFrame with new rows
    new_df = pd.DataFrame(new_rows)

    # Append new rows to original DataFrame and save
    result_df = pd.concat([df, new_df], ignore_index=True)
    result_df.to_csv(output_csv_path, index=False)

    print(f"Successfully processed {len(new_rows)} rows and saved to {output_csv_path}")
    return result_df

#Example of how to call the main function:
if __name__ == "__main__":
    print("hello")
    main(
        input_csv_path=r"/sise/home/avramit/mfact-summ/mfact/datasets/heb/mfact_dataset_he/output_mafat_less_500_test.csv",
        output_csv_path=r"/sise/home/avramit/mfact-summ/mfact/datasets/heb/mfact_dataset_he/output_mafat_less_500_test_negative.csv",
        num_rows_to_transform=1,
        transformation_types="number_swap, entity_person_swap"
    )
# from typing import List, Dict
# from openai import OpenAI
# from pydantic import BaseModel, Field
# import json
# from langchain_core.output_parsers import JsonOutputParser
# import pandas as pd
# from pydantic import BaseModel, PrivateAttr, Field
# from typing import List
# import random
# from enum import Enum
#
# # Define OpenAI API credentials
# OPENAI_API_KEY = "***REMOVED-API-KEY***"
# client = OpenAI(api_key=OPENAI_API_KEY)
#
#
# class CorrectnessErrorTypes(str, Enum):
#     GENDER_SWAP = "gender_swap"
#     MEASURE_UNIT_SWAP = "measure_unit_swap"
#     NUMBER_SWAP = "number_swap"
#     ENTITY_LOCATION_SWAP = "entity_location_swap"
#     ENTITY_PERSON_SWAP = "entity_person_swap"
#
# # Define the main data structure
# class SummaryDecompositionClaims(BaseModel):
#     claims_decomposition: List[str] = Field(description="List of the claims in the summary after claim decomposition")
#     total_amount_of_claims: int = Field(description="How many claims decompose from the text")
#     _original_text: str = PrivateAttr()
#
#     def __init__(self, **data):
#         super().__init__(**data)
#         self._original_text = data.get('original_text', '')
#
# # Define nested model for transformation details
# class TransformationDetail(BaseModel):
#     original_claim: str = Field(description="The original claim from the summary")
#     transformation_types_performed: List[str] = Field(description="List of transformation types applied")
#     reasoning_process: str = Field(description="step by step reasoning in hebrew of how to perform all the error transformations type in the claim")
#     negative_claim: str = Field(description="The transformed claim with errors")
#     summary_of_transformation_perform: str = Field(description="Description of how the claim was transformed")
#
#
# # Define the main data structure
# class NegativeAugmentationSummary(BaseModel):
#     reasoning_process_of_augmentation: str = Field(description="step by step reasoning in Hebrew how to perform **all** the transformation types mention in each separate claim, when perform transformation, the data MUST changed")
#     correctness_error_transformation_type: List[str] = Field(description="Types of correctness error transformations performed on each claim. NEED to consists all the error transformations types requested")
#     correctness_error_transformation_claims: List[str] = Field(description="Claims that were transformed")
#     generative_negative_augmentation_summary: str = Field(description="Complete summary with errors introduced")
#     total_amount_of_transformed_claims: int = Field(description="How many claims transformed and consists correctness error transformation in the new summary")
#     detailed_error_transformation: List[TransformationDetail] = Field(description="Detailed breakdown of each transformation performed in Hebrew")
#     _original_text: str = PrivateAttr()
#
#     def __init__(self, **data):
#         super().__init__(**data)
#         self._original_text = data.get('original_text', '')
#
# def choose_suitable_claim(claim_decomposition: SummaryDecompositionClaims) -> str:
#
#
# def summary_claim_decomposition(text: str) -> SummaryDecompositionClaims:
#     # Initialize the OpenAI client directly
#
#     claim_decomposition_parser = JsonOutputParser(pydantic_object=SummaryDecompositionClaims)
#
#     # Create the system and user messages
#     system_message = """
#     You are a Hebrew computational linguistics professional Researcher.
#     Your goal is to perform claim decomposition to the following hebrew text sentences into individual facts.
#     Do not add words or insert new details into a claim, keep the original text unchanged
#     """
#
#     user_message = f"""
#     **Summary**: {text}
#     the response should be in JSON in the EXACT format:
#     {claim_decomposition_parser.get_format_instructions()}
#     """
#     try:
#         # Send request to the model
#         print(claim_decomposition_parser.get_format_instructions())
#         response = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "system", "content": system_message},
#                 {"role": "user", "content": user_message}
#             ],
#             temperature=0.0,
#             max_tokens=2000,
#             response_format={"type": "json_object"}
#         )
#
#         # Extract and parse the JSON response
#         result_json = response.choices[0].message.content
#         print("\nRaw response:")
#         print(result_json)
#
#         result = SummaryDecompositionClaims.model_validate_json(result_json)
#         result._original_text = text  # Assign the original text separately
#         return result
#
#     except Exception as e:
#         print(f"Error: {e}")
#         # Return an empty model instance if there's an error
#         return SummaryDecompositionClaims(
#             claim_decomposition=[],
#             total_amount_of_claims=0,  # Add this missing required field
#             original_text=text
#         )
#
#
#
# def summary_negative_augmentation(apply_transformation_types: List[CorrectnessErrorTypes],
#                                   transformation_examples: dict,
#                                   amount_claims_to_apply_transformation: int,
#                                   original_summary: str,
#                                   summary_claims: List[str]) -> NegativeAugmentationSummary:
#     # Initialize the OpenAI client directly
#     client = OpenAI(api_key=OPENAI_API_KEY)
#     parser = JsonOutputParser(pydantic_object=NegativeAugmentationSummary)
#     transformation_types_list = [t.strip() for t in apply_transformation_types]
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
#
#     user_message = f"""
#     **Original Summary**: {original_summary}
#     **Claims to perform the negative augmentation on**: {summary_claims}
#     **Claims Transformation amount**: {amount_claims_to_apply_transformation}
#     **Transformation types to apply to EACH claim**: {transformation_types_list}
#         **Apply **ALL** Transformations Simultaneously**
#         For example, with "gender_swap, measure_unit":
#         1. You must change BOTH gender terms AND measurement units in the SAME claim
#         2. Do NOT select claims that don't have BOTH gender terms AND measurement units
#
#     **Examples**: {json.dumps(transformation_examples, ensure_ascii=False)}
#     the response should be in JSON in the EXACT format:
#     {parser.get_format_instructions()}
#
#     """
#     try:
#         # Send request to the model
#         print("System Message:\n", system_message)
#         print("\nUser Message:\n", user_message)
#         response = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "system", "content": system_message},
#                 {"role": "user", "content": user_message}
#             ],
#             temperature=0.0,
#             max_tokens=2000,
#             response_format={"type": "json_object"}
#         )
#
#         # Extract and parse the JSON response
#         result_json = response.choices[0].message.content
#         print("\nRaw response:")
#         print(result_json)
#
#         # Parse into Pydantic model
#         result = NegativeAugmentationSummary.model_validate_json(result_json)
#         result._original_text = original_summary # Assign the original text separately
#         return result
#
#     except Exception as e:
#         print(f"Error: {e}")
#         # Return an empty model instance if there's an error
#         return NegativeAugmentationSummary(
#             claim_decomposition=[],
#             reasoning_process_of_choose_claim="",  # Add this missing required field
#             reasoning_process_of_augmentation="",
#             correctness_error_transformation_type=[],
#             correctness_error_transformation_claims=[],
#             generative_negative_augmentation_summary="",
#             total_amount_of_transformed_claims=0,  # Add this missing required field
#             detailed_error_transformation=[],
#             original_text=original_summary
#         )
#
#
# # Add this as the main function at the end of your file
# def main():
#     """
#     Process NLI dataset by applying negative augmentation to hypothesis statements.
#
#     Args:
#         input_csv_path (str): Path to input CSV with columns: index, hypothesis, premise, label
#         output_csv_path (str): Path where the augmented CSV will be saved
#         num_rows_to_transform (int): Number of rows to transform
#         transformation_types (str): Comma-separated transformation types to apply
#     """
#
#     input_csv_path=r"/content/output_mafat_less_500_test_new.csv"
#     output_csv_path=r"/content/output_mafat_less_500_test_negative.csv"
#     num_rows_to_transform=20
#     transformation_types=[CorrectnessErrorTypes.GENDER_SWAP, CorrectnessErrorTypes.ENTITY_PERSON_SWAP, CorrectnessErrorTypes.NUMBER_SWAP]
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
#
#     # Create a list to store new rows
#     new_rows = []
#
#     # Process each selected row
#     for _, row in rows_to_transform.iterrows():
#         summary = row['hypothesis']
#         premise = row['premise']
#
#         claim_decomposition = summary_claim_decomposition(text=summary)
#
#         # Apply negative augmentation
#         result = summary_negative_augmentation(
#             apply_transformation_types=[CorrectnessErrorTypes.GENDER_SWAP, CorrectnessErrorTypes.ENTITY_PERSON_SWAP, CorrectnessErrorTypes.NUMBER_SWAP],
#             transformation_examples=filtered_examples,
#             amount_claims_to_apply_transformation=5,
#             original_summary=summary,
#             summary_claims=claim_decomposition.claims_decomposition
#         )
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
#
# #Example of how to call the main function:
# if __name__ == "__main__":
#     print("hello")
#     main()


# # Example Hebrew summary
# hebrew_summary = """
# דוח מבקר המדינה לשנת 2023 ששמו רום אמיר מצביע על מספר ליקויים במערכת החינוך. הדוח מציין כי 30% מהתלמידים לא עומדים בסטנדרטים הבינלאומיים במתמטיקה. בנוסף, המבקר מצא כי תקציב בסך 5 מיליון שקלים שיועד לתגבור לימודי מדעים לא הגיע ליעדו. הדוח גם ציין שמספר המורות המוסמכות ללמד אנגלית ירד ב-15% בשנתיים האחרונות.
# """
#
# long_hebrew_summary = """
# בדוח מבקר המדינה לשנת 2025, הדוקטורית רחל כהן ציינה כי בבתי החולים בירושלים חסר כוח אדם משמעותי במחלקות החירום. על פי הדוח, מספר הרופאים הכשירים ירד ב-20% בשנתיים האחרונות, והתקציב שיועד לגיוס נוספים, המוערך בכ-10 מיליון שקלים, לא נוצל במלואו. במקביל, המחקר שהציגה ד"ר כהן מצביע על כך שמרחק הנסיעה הממוצע של מטופלים מאזורים פריפריאליים עומד על כ-50 קילומטרים עד למרכז הרפואי הקרוב, עובדה שעלולה לפגוע בסיכויי ההחלמה שלהם.
#
# הדוח גם מתייחס להתפטרותו של פרופ׳ אלעד לוי מאוניברסיטת תל אביב, שטען כי נאלץ לפרוש בשל בעיות מערכתיות בניהול כוח האדם. לטענתו, למרות שהחוזה האחרון שלו כלל מימון בסך 2 מיליון דולר לטובת מחקר בתחום הנדסה רפואית, לא הוקצו די משאבים להעסקת סטודנטים וחוקרים נוספים. עוד מצוין בדוח כי התוכניות לבניית אגף חדש בתל השומר בגובה של 50 מטרים נדחו למועד בלתי ידוע, בשל עיכובים בהעברות תקציב בין משרד הבריאות לאוצר.
#
# בנוסף, במסגרת הדוח נותחו נתונים על תוצאות מבחני ההסמכה לאחיות בשנה האחרונה, שמראים על ירידה של 15% בהצלחה הכללית בבחינות. ד"ר כהן רואה בכך עדות לחוסר המוכנות של הסטודנטים, אך מדגישה כי יש גם שיפור קל בהכשרות הקליניות, בעיקר בזכות שיתוף פעולה עם אוניברסיטת בן-גוריון בנגב. לדבריה, המלצות הדוח מתבססות על ניתוחים סטטיסטיים שבוצעו במהלך 2024, לצד ראיונות עם מנהלי מחלקות ומומחים מתחום בריאות הציבור.
# """
#
# # Execute the function and handle errors explicitly
# try:
#     print("Starting negative augmentation with GPT-4o-mini...")
#     apply_transformation_types = "gender_swap, measure_unit, number_swap"
#     selected_transformations = apply_transformation_types.split(",")  # Extract transformation types
#     general_filtered_examples = {key: general_error_examples[key] for key in selected_transformations if key in general_error_examples}
#     hebrew_filtered_examples = {key: hebrew_error_examples[key] for key in selected_transformations if key in hebrew_error_examples}
#     filtered_examples = {**general_filtered_examples, **hebrew_filtered_examples}
#     print(filtered_examples)
#     result = summary_negative_augmentation(
#         apply_transformation_types=apply_transformation_types,
#         transformation_examples=filtered_examples,
#         amount_claims_to_apply_transformation=2,
#         summary=long_hebrew_summary
#     )
#
#     # Print results
#     print("\n=== CLAIM DECOMPOSITION ===")
#     for i, claim in enumerate(result.claim_decomposition, 1):
#         print(f"{i}. {claim}")
#
#     print("\n=== TRANSFORMATION TYPES USED ===")
#     for t_type in result.correctness_error_transformation_type:
#         print(f"- {t_type}")
#
#     print("\n=== FINAL NEGATIVE SUMMARY ===")
#     print(result.generative_negative_augmentation_summary)
#
#     print("\n=== DETAILED TRANSFORMATIONS ===")
#     for i, detail in enumerate(result.detailed_error_transformation, 1):
#         print(f"Transformation {i}:")
#         print(f"  Original: {detail.original_claim}")
#         print(f"  Negative: {detail.negative_claim}")
#         print(f"  Types: {', '.join(detail.transformation_types_performed)}")
#         print(f"  Summary: {detail.summary_of_transformation_perform}")
#
# except Exception as e:
#     print(f"Major error during execution: {e}")