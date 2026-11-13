from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from pathlib import Path
import json
from typing import List
# Define the main data structure
#from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, PrivateAttr, Field
from openai import OpenAI
# Define OpenAI API credentials
from typing import List, Dict, Type
import json
from typing import Dict
from pathlib import Path
import re
from anthropic import Anthropic
import json
from typing import Dict
from pathlib import Path
import google.generativeai as genai
from summary_augmentation.api_keys import OPENAI_API_KEY  # loaded from .env
from summary_augmentation.api_keys import GEMINI_API_KEY  # loaded from .env
from summary_augmentation.api_keys import CLAUDE_API_KEY  # loaded from .env

genai.configure(api_key=GEMINI_API_KEY)
client_claude = Anthropic(api_key=CLAUDE_API_KEY)
model = genai.GenerativeModel('gemini-2.5-pro')

def llm_query_dicts(article):
    device = "cuda" # the device to load the model onto
    client = OpenAI(api_key=OPENAI_API_KEY)

    model = AutoModelForCausalLM.from_pretrained("dicta-il/dictalm2.0-instruct", torch_dtype=torch.bfloat16, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained("dicta-il/dictalm2.0-instruct")

    messages = [
        {"role": "user", "content": "איזה רוטב אהוב עליך?"},
        {"role": "assistant", "content": "טוב, אני די מחבב כמה טיפות מיץ לימון סחוט טרי. זה מוסיף בדיוק את הכמות הנכונה של טעם חמצמץ לכל מה שאני מבשל במטבח!"},
        {"role": "user", "content": "האם יש לך מתכונים למיונז?"}
    ]

    encoded = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)

    generated_ids = model.generate(encoded, max_new_tokens=50, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)
    print(decoded[0])


# def llm_query_gpt(system_message, user_message, output_object: Type[BaseModel], parser=None, openai_model="gpt-3.5-turbo-instruct-0914"):
#     # gpt-4o-mini-2024-07-18
#     print(f"prompt: {system_message}\n {user_message}")
#     try:
#         response = client.chat.completions.create(
#             model=openai_model,
#             messages=[
#                 {"role": "system", "content": system_message},
#                 {"role": "user", "content": user_message}
#             ],
#             temperature=0.1,
#             max_tokens=1000,
#             response_format={"type": "json_object"}
#         )
#         # Extract the raw response
#         result_json = response.choices[0].message.content
#         print("\nRaw response:")
#         print(result_json.encode('utf-8').decode('utf-8'))
#
#         # Use the retry parser if provided, otherwise fallback to direct validation
#         if parser:
#             # The retry parser will handle any parsing errors and retry if needed
#             try:
#                 result = parser.parse(result_json)
#                 return result
#             except Exception as parsing_error:
#                 # If the retry parser fails after all retries, log it
#                 print(f"Retry parser failed: {parsing_error}")
#                 # Fallback to direct parsing as a last resort
#                 return output_object.model_validate_json(result_json)
#         else:
#             # Direct parsing without retry
#             return output_object.model_validate_json(result_json)
#
#     except Exception as e:
#         print(f"Exception during openai LLM query or parsing: {e}")
#         return None


def llm_query_gpt(system_message, user_message, output_object: Type[BaseModel], parser=None, openai_model="gpt-3.5-turbo-instruct-0914"):

    client = OpenAI(api_key=OPENAI_API_KEY)
    print(f"prompt: {system_message}\n {user_message}")
    try:
        prompt = f"{system_message.strip()}\n{user_message.strip()}"
        response = client.completions.create(
            model=openai_model,
            prompt=prompt,
            temperature=0.1,
            max_tokens=1000
        )
        # Extract the raw response
        result_json = response.choices[0].text.strip()
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
def summary_dicta(article: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        "dicta-il/dictalm2.0-instruct",
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained("dicta-il/dictalm2.0-instruct")

    prompt = f"""
    אתה מומחה לסיכום טקסטים. סכם את הטקסט הבא בעברית באורך של 125 עד 175 מילים, תוך התמקדות ברעיונות המרכזיים בלבד.

    <תחילת הטקסט לסיכום>
    {article}
    <סוף הטקסט לסיכום>

    סיכום:
    """

    encoded = tokenizer(prompt, return_tensors="pt").to(device)

    generated_ids = model.generate(
        **encoded,
        max_new_tokens=2000,
        do_sample=True,
        temperature=0.1,
        top_p=0.9
    )

    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # === Extract everything after the last occurrence of <סוף הטקסט לסיכום>
    split_marker = "<סוף הטקסט לסיכום>"
    if split_marker in decoded:
        summary_part = decoded.split(split_marker)[-1].strip()
    else:
        summary_part = decoded.strip()

    # === Remove [INST] and [/INST] or any leftover formatting tokens
    summary_cleaned = re.sub(r'\[/?INST\]', '', summary_part).strip()

    return summary_cleaned


def summary_gpt3_5(article):
    # Initialize the OpenAI client directly
    class ArticleSummary(BaseModel):
        article_summary: str = Field(description="HEBREW summary generative rich paraphrasing")
    parser = JsonOutputParser(pydantic_object=ArticleSummary)

    try:
        system, user = summary_prompt_basic(source_article=article, output_instructions_parser=parser)
        result = llm_query_gpt(system_message=system, user_message=user, output_object=ArticleSummary)
        return result

    except Exception as e:
        print(f"Error: {e}")


def summary_prompt_basic(source_article, output_instructions_parser):
    system_message = f"""
    אתה מומחה לסיכום טקסטים.
    סכם את הטקסט הבא ב[76–150] מילים
    """

    user_message = f"""
    <תחילת הטקסט לסיכום>
    {source_article}
    <סוף הסטקס לסיכום>
    
    הוראות לפלט שחייב להיות בפורמט הבא
    {output_instructions_parser.get_format_instructions()}
    """

    return system_message, user_message


# === Main logic ===
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

                # Generate summary
                summary_text = summary_dicta(article)
                if not summary_text:
                    print(f"Skipping line {i} due to failed summary.")
                    continue

                new_summary_entry = {
                    "generative_summary": summary_text,
                    "model": "summary_dicta",
                    "prompt": "summary_prompt_basic"
                }

                if "metadata" not in data or not isinstance(data["metadata"], dict):
                    data["metadata"] = {}

                if "generative_summaries" not in data["metadata"]:
                    data["metadata"]["generative_summaries"] = []

                data["metadata"]["generative_summaries"].append(new_summary_entry)

                # Write to output
                outfile.write(json.dumps(data, ensure_ascii=False) + "\n")

            except Exception as e:
                print(f"Error processing line {i}: {e}")
                continue


# === Run the script ===
if __name__ == "__main__":
    input_file = r"C:\Users\amita\Desktop\Thesis\Correctness\data\summarization-7-heb.jsonl\summarization-7-heb_2.jsonl"
    output_file = r"C:\Users\amita\Desktop\Thesis\new.jsonl"

    # Choose line range here (inclusive start, exclusive end)
    start = 0
    end = 50  # e.g., process only the first 10 lines

    main(input_file, output_file, start_line=start, end_line=end)
