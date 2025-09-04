from typing import Type
from pydantic import BaseModel, Field
from pathlib import Path
import json
import re
import torch
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM
import datetime

OPENAI_API_KEY = "***REMOVED-API-KEY***"  # Replace with your actual API key

def llm_query_gpt(system_message, user_message, output_object: Type[BaseModel], openai_model="gpt-3.5-turbo-1106"):
    client = OpenAI(api_key=OPENAI_API_KEY)
    print(f"prompt: {system_message}\n{user_message}")

    try:
        response = client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": system_message.strip()},
                {"role": "user", "content": user_message.strip()}
            ],
            temperature=0.2,
            max_tokens=2000
        )
        result_json = response.choices[0].message.content.strip()
        print("\nRaw response:")
        print(result_json)

        return output_object.model_validate_json(result_json)

    except Exception as e:
        print(f"Exception during OpenAI LLM query or parsing: {e}")
        return None



def summary_gpt3_5(article):
    class ArticleSummary(BaseModel):
        article_summary: str = Field(description="HEBREW summary generative rich paraphrasing")

    try:
        system, user = summary_prompt_basic(article)
        result = llm_query_gpt(system_message=system, user_message=user, output_object=ArticleSummary)
        return result
    except Exception as e:
        print(f"Error: {e}")
        return None

#
# def summary_prompt_basic(source_article):
#     system_message = """
# אתה מומחה בלסיכום טקסטים עיתונאיים בעברית.
#
# עליך להפיק סיכום מדויק, עשיר בפרטים ונטול פרשנות של הטקסט הבא לפי ההוראות הבאות:
#
# 1. הסיכום חייב להיכתב בשפה העברית בלבד.
# 2. אורך הסיכום חייב להיות לפחות 300 מילים.
# 3. לחלופין, הסיכום יכול להכיל **13 עד 17 משפטים מלאים**.
# 4. הסיכום צריך לכלול את כל **העובדות, הדמויות, המקומות, האירועים, ההקשרים, והמסקנות המרכזיות** מהטקסט.
# 5. אל תחסיר פרטים חשובים שמופיעים באמצע או בסוף הטקסט.
# 6. אין לסטות מהעובדות, אין להוסיף מידע שלא הופיע במקור.
# 7. הסגנון צריך להיות **עיתונאי-אינפורמטיבי**, רהוט ולוגי, לא תמציתי מדי.
#
# 🛑 שים לב: סיכומים קצרים מדי, כוללניים, או שאינם עומדים בדרישות אלה — **אינם יתקבלו**.
#
# 📦 הפלט חייב להיות בפורמט JSON תקני עם שדה יחיד בשם `article_summary`.
#     """
#
#     user_message = f"""
# <תחילת הטקסט לסיכום>
# {source_article}
# <סוף הטקסט לסיכום>
#
# 🔽 הפלט שלך צריך להיראות כך:
# {{
#   "article_summary": "כאן יופיע סיכום מפורט בעברית באורך מלא בהתאם להוראות."
# }}
#     """
#
#     return system_message.strip(), user_message.strip()


# def summary_prompt_basic(source_article):
#     system_message = """
#     You are Hebrew text summarization expert!
#     Your task is to generate HEBREW summary of the article which consists all the main facts from the article
#     Your hebrew summary should be detailed and informative
#      ### Instructions ###
#      1. The text summary MUST consists only HEBREW!
#      2. The text summary MUST be detailed
#      3. The text summary MUST consists all the main facts from the source article
#      4. The text summary MUST consists of AT LEAST 13 sentences!
#      5. First identify all the main facts in the source article, then generate the long fully detailed summary
#
#     🛑 שים לב: סיכומים קצרים מדי, כוללניים, או שאינם עומדים בדרישות אלה — **אינם יתקבלו**.
#
#     📦 הפלט חייב להיות בפורמט JSON תקני עם שדה יחיד בשם `article_summary`.
#         """
#
#     user_message = f"""
# <תחילת הטקסט לסיכום>
# {source_article}
# <סוף הטקסט לסיכום>
#
# 🔽 הפלט שלך צריך להיראות כך:
# {{
#   "article_summary": "כאן יופיע סיכום מפורט בעברית באורך מלא בהתאם להוראות."
# }}
#     """
#
#     return system_message.strip(), user_message.strip()

def summary_prompt_basic(source_article):
    """
    This prompt combines fact extraction and narrative summarization into one process.
    It forces the model to first identify all facts and then weave them into a
    long, detailed summary.
    """
    system_message = """
    You are an expert at creating detailed, fact-rich summaries in Hebrew. 
    Your task is to transform the source article into a single, comprehensive narrative summary that incorporates all key information.

    ### Your Two-Step Process to Follow ###

    **Step 1: Internal Fact Collection (Analysis Only)**
    First, silently read the entire article and identify ALL the key facts. This includes names, dates, statistics, locations, key events, and main conclusions. 
    Create a complete mental checklist of these facts. 
    **Do not write this list in the output.** This step is for your internal analysis only.

    **Step 2: Narrative Summary Writing (The Final Output)**
    Now, using the complete checklist of facts you just created, write a single, flowing, and detailed summary. 
    Your mission is to skillfully weave ALL the facts from your list into a coherent Hebrew narrative.
     The final text should be informative, easy to read, and feel like a complete, self-contained article.

    ### Mandatory Rules for the Final Output (Step 2) ###

    1.  **Length:** The final summary MUST be **at least 15 sentences long**. This is a strict minimum.
    2.  **Factual Completeness:** The summary MUST incorporate all of the important facts you identified in Step 1. Do not omit key details.
    3.  **Style:** The output MUST be a flowing narrative, NOT a list of disconnected points.
    4.  **Format:** The entire output must be a single, valid JSON object with the key `"article_summary"`.
    """

    user_message = f"""
    Source article for processing:
    <article>
    {source_article}
    </article>

    ---
    Now, execute the two-step process and provide the final narrative summary in the required JSON format.
    """

    return system_message.strip(), user_message.strip()




def main(input_path: str, output_path: str, start_line: int = 0, end_line: int = None):
    input_path = Path(input_path)
    output_path = Path(output_path)

    with input_path.open("r", encoding="utf-8") as infile:
        all_lines = infile.readlines()

    if end_line is None:
        end_line = len(all_lines)

    with output_path.open("w", encoding="utf-8") as outfile:
        for i, line in enumerate(all_lines):
            if i < start_line or i >= end_line:
                continue

            try:
                data = json.loads(line)
                article = data.get("text_raw", "")

                summary = summary_gpt3_5(article)
                if not summary:
                    print(f"Skipping line {i} due to failed summary.")
                    continue

                new_summary_entry = {
                    "generative_summary": summary.article_summary,
                    "model": "gpt-3.5-turbo-1106",
                    "prompt": "summary_prompt_basic"
                }

                if "metadata" not in data or not isinstance(data["metadata"], dict):
                    data["metadata"] = {}

                if "generative_summaries" not in data["metadata"]:
                    data["metadata"]["generative_summaries"] = []

                data["metadata"]["generative_summaries"].append(new_summary_entry)
                outfile.write(json.dumps(data, ensure_ascii=False) + "\n")

            except Exception as e:
                print(f"Error processing line {i}: {e}")
                continue


# === Run the script ===
if __name__ == "__main__":

    start = 25
    end = 75

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    input_file = r"C:\Users\amita\Desktop\Thesis\Correctness\data\summarization-7-heb.jsonl\summarization-7-heb_2.jsonl"
    output_file = rf"C:\Users\amita\Desktop\Thesis\summaries_gpt3.5_{timestamp}_start_row_{start}_end_row_{end}.jsonl"
    main(input_file, output_file, start_line=start, end_line=end)
