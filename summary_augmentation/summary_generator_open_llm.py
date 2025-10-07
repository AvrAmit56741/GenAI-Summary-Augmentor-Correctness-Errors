#
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from pathlib import Path
# from pydantic import BaseModel, Field
# import torch
# import json
#
# # === Load DictaLM 2.0 Instruct model ===
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = AutoModelForCausalLM.from_pretrained(
#     "dicta-il/dictalm2.0-instruct",
#     torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
#     device_map=device
# )
# tokenizer = AutoTokenizer.from_pretrained("dicta-il/dictalm2.0-instruct")
#
# def summary_dicta(article: str):
#     messages = [
#         {
#             "role": "user",
#             "content": "סכם את המאמר הבא בעברית באורך של 125 עד 175 מילים, תוך התמקדות ברעיונות המרכזיים בלבד.\n\n"
#                        f"<תחילת הטקסט לסיכום>\n{article}\n<סוף הטקסט לסיכום>"
#         }
#     ]
#     encoded = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
#     torch.cuda.empty_cache()
#     generated_ids = model.generate(
#         encoded,
#         max_new_tokens=512,
#         do_sample=True,
#         temperature=0.1,
#         top_p=0.9
#     )
#     decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
#     return decoded.strip()
#
#
# def summary_model(article: str):
#     class ArticleSummary(BaseModel):
#         article_summary: str = Field(description="HEBREW summary generative rich paraphrasing")
#
#     try:
#         result_text = summary_dicta(article)
#         result = ArticleSummary(article_summary=result_text)
#         return result
#     except Exception as e:
#         print(f"Error during summarization: {e}")
#         return None
#
#
# def main(input_path: str, output_path: str, start_line: int = 0, end_line: int = None):
#     input_path = Path(input_path)
#     output_path = Path(output_path)
#
#     with input_path.open("r", encoding="utf-8") as infile:
#         all_lines = infile.readlines()
#
#     if end_line is None:
#         end_line = len(all_lines)
#
#     with output_path.open("w", encoding="utf-8") as outfile:
#         for i, line in enumerate(all_lines):
#             if i < start_line or i >= end_line:
#                 continue
#
#             try:
#                 data = json.loads(line)
#                 article = data.get("text_raw", "")
#
#                 summary = summary_model(article)
#                 if not summary:
#                     print(f"Skipping line {i} due to failed summary.")
#                     continue
#
#                 new_summary_entry = {
#                     "generative_summary": summary.article_summary,
#                     "model": "dictalm2.0-instruct",
#                     "prompt": "chat_template"
#                 }
#
#                 if "metadata" not in data or not isinstance(data["metadata"], dict):
#                     data["metadata"] = {}
#
#                 if "generative_summaries" not in data["metadata"]:
#                     data["metadata"]["generative_summaries"] = []
#
#                 data["metadata"]["generative_summaries"].append(new_summary_entry)
#                 outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
#
#             except Exception as e:
#                 print(f"Error processing line {i}: {e}")
#                 continue
#
#
# # === Run the script ===
# if __name__ == "__main__":
#     input_file = r"/home/avramit/dicta/summarization-7-heb_2.jsonl"
#     output_file = r"/home/avramit/dicta/dicta_output.jsonl"
#     start = 0
#     end = 10
#     main(input_file, output_file, start_line=start, end_line=end)

from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from pydantic import BaseModel, Field
import torch
import json
import io

# === Load DictaLM 2.0 Instruct model ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    "dicta-il/dictalm2.0-instruct",
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    device_map=device
)
tokenizer = AutoTokenizer.from_pretrained("dicta-il/dictalm2.0-instruct")

# === Set the pad token to resolve warnings ===
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id


def summary_dicta(article: str):
    messages = [
        {
            "role": "user",
            "content": "סכם את המאמר הבא בעברית באורך של150  עד 200 מילים, תוך התמקדות ברעיונות המרכזיים בלבד.\n\n"
                       f"<תחילת הטקסט לסיכום>\n{article}\n<סוף הטקסט לסיכום>"
        }
    ]

    model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
    input_length = model_inputs.shape[1]
    torch.cuda.empty_cache()

    generated_ids = model.generate(
        model_inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.2,
        top_p=0.9
    )

    new_tokens = generated_ids[0, input_length:]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)

    if "[/INST]" in decoded:
        decoded = decoded.split("[/INST]", 1)[-1]

    return decoded.strip()


def summary_model(article: str):
    class ArticleSummary(BaseModel):
        article_summary: str = Field(description="HEBREW summary generative rich paraphrasing")

    try:
        result_text = summary_dicta(article)
        result = ArticleSummary(article_summary=result_text)
        return result
    except Exception as e:
        print(f"Error during summarization: {e}")
        return None


def main(input_path: str, output_path: str, start_line: int = 0, end_line: int = None):
    input_path = Path(input_path)
    output_path = Path(output_path)

    # === FIX: Correctly check for existing lines before opening in append mode ===
    try:
        # Open the file in read mode ('r') just to count existing lines.
        with output_path.open("r", encoding="utf-8") as f:
            processed_lines = sum(1 for _ in f)

        print(f"✅ Output file found. Resuming from line {processed_lines + 1}.")
        start_line = max(start_line, processed_lines)
    except FileNotFoundError:
        print("📝 Output file not found. Starting from scratch.")
        pass  # If the file doesn't exist, we'll create it in the next step.

    with input_path.open("r", encoding="utf-8") as infile:
        all_lines = infile.readlines()

    if end_line is None:
        end_line = len(all_lines)

    # Now, open in append mode ('a') to add new content without overwriting.
    with output_path.open("a", encoding="utf-8") as outfile:
        for i, line in enumerate(all_lines):
            # The range check now correctly handles resuming.
            if i < start_line or i >= end_line:
                continue

            print(f"Processing line {i + 1}/{end_line}...")

            try:
                data = json.loads(line)
                article = data.get("text_raw", "")

                if not article:
                    print(f"Skipping line {i + 1} due to empty article.")
                    continue

                summary = summary_model(article)
                if not summary:
                    print(f"Skipping line {i + 1} due to failed summary.")
                    continue

                new_summary_entry = {
                    "generative_summary": summary.article_summary,
                    "model": "dictalm2.0-instruct",
                    "prompt": "chat_template"
                }

                if "metadata" not in data or not isinstance(data["metadata"], dict):
                    data["metadata"] = {}

                if "generative_summaries" not in data["metadata"]:
                    data["metadata"]["generative_summaries"] = []

                data["metadata"]["generative_summaries"].append(new_summary_entry)
                outfile.write(json.dumps(data, ensure_ascii=False) + "\n")

            except Exception as e:
                print(f"Error processing line {i + 1}: {e}")
                continue


# === Run the script ===
if __name__ == "__main__":
    print("version 1")
    input_file = r"/home/avramit/dicta/summarization-7-heb_2.jsonl"
    output_file = r"/home/avramit/dicta/summaries_dicta_30_200.jsonl"
    start = 30
    end = 200
    main(input_file, output_file, start_line=start, end_line=end)