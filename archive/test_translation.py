import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
from semantic_chunkers_test import ConsecutiveChunker
# Define model and tokenizer
model_name = "dicta-il/dictalm2.0-instruct"
print("Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
print("Model loaded successfully.")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Tokenizer loaded successfully.")


def preprocess_text(text):
    """
    Preprocesses the input text by handling bullets, special characters,
    measurements, and formatting to enhance translation quality.
    """
    # 1. Replace bullet points with newline and dash
    text = re.sub(r'•\s*', '- ', text)

    # 2. Normalize measurements (e.g., "50 גרם חמאה קרה" -> "50 grams of cold butter")
    # This step can be customized based on specific needs
    # For simplicity, we'll leave measurements as-is

    # 3. Remove problematic symbols (e.g., excessive quotes, backslashes)
    text = text.replace('\\"', '"').replace('\\n', '\n').replace('\\', '')

    # 4. Replace multiple spaces and newlines with single space or standardized delimiter
    text = re.sub(r'\s+', ' ', text)

    # 5. Handle quotes and parentheses
    text = re.sub(r'\"', '"', text)
    text = re.sub(r'\(', '(', text)
    text = re.sub(r'\)', ')', text)

    # 6. Optionally, separate ingredients and instructions with clear delimiters
    # For example, replace "לתחתית העוגה" with "\nIngredients:\n"
    # This requires knowledge of the text structure

    # 7. Trim leading and trailing whitespace
    text = text.strip()

    return text


def translate_text(text):
    print(f"Translating text: {text}")

    preprocessed_text = preprocess_text(text)

    print(f"Translating preprocessed text: {preprocessed_text}")  # Log a snippet

    prompt = f"<s>[INST] You are a translator from Hebrew to English. Translate the following text in Hebrew to English: '{preprocessed_text}' [/INST]"

    # Ensure the pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=32768  # Match the model's max_position_embeddings
    )
    input_ids = encoded.input_ids.cuda()
    attention_mask = encoded.attention_mask.cuda()

    print(f"Generated input IDs: {input_ids}")
    print(f"Attention mask: {attention_mask}")

    try:
        # Generate translation with a limit on new tokens
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=3000,  # Adjust based on expected output size
            early_stopping=True
        )
        print(f"Generated output IDs: {outputs}")


        # Decode and clean up the output
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Raw translated text: {translated_text}")

        # Remove the prompt text from the beginning, if present
        if "[INST]" in translated_text and "[/INST]" in translated_text:
            translated_text = translated_text.split("[/INST]")[1].strip()

        # Log the cleaned translated text
        print(f"Cleaned translated text: {translated_text}")

        # Post-process to handle any remaining Hebrew text
        if re.search(r'[א-ת]', translated_text):
            print("Warning: Translated text still contains Hebrew characters.")
            # Optionally, you can replace or flag the Hebrew text
            translated_text = re.sub(r'[א-ת]', '', translated_text)
            print(f"Post-processed translated text: {translated_text}")

        return translated_text
    except Exception as e:
        print(f"Error during translation: {e}")
        return "Translation Error"


def chunk_and_translate(text):
    # Split text into sentences
    sentences = re.split(r'(?<=[.!?]) +', text)
    translated_chunks = []

    # Process in chunks of 10 sentences
    for i in range(0, len(sentences), 7):
        chunk = ' '.join(sentences[i:i+7])
        print(f"Translating chunk: {chunk}")
        translated_chunk = translate_text(chunk)
        print(f"Translated chunk: {translated_chunk}")
        translated_chunks.append(translated_chunk)

    # Concatenate all translated chunks
    return ' '.join(translated_chunks)

def chunk_and_translate_with_overlap(text, chunk_size=1000, overlap_size=200):
    # Tokenize the text
    tokens = tokenizer.tokenize(text)
    chunks = []

    # Create overlapping chunks
    start_idx = 0
    while start_idx < len(tokens):
        end_idx = min(start_idx + chunk_size, len(tokens))
        chunk = tokens[start_idx:end_idx]
        chunks.append(chunk)
        start_idx += chunk_size - overlap_size

    translated_chunks = []
    previous_context = ""  # To store the previous context for removing overlap
    for chunk in chunks:
        # Convert chunk back to text for translation
        chunk_text = tokenizer.convert_tokens_to_string(chunk)
        translated_chunk = translate_text(chunk_text)

        # Remove the overlapping portion from the translated chunk
        if previous_context and previous_context in translated_chunk:
            translated_chunk = translated_chunk.replace(previous_context, "", 1).strip()

        # Update the previous context
        previous_context = translate_text(
            tokenizer.convert_tokens_to_string(chunk[-overlap_size:])
        )

        translated_chunks.append(translated_chunk)

    # Concatenate all translated chunks
    return ' '.join(translated_chunks)


# Function to process the JSONL file
def process_jsonl(input_path, output_path, use_overlap=False):
    print(f"Processing input file: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        line_count = 0
        processed_count = 0
        for line in infile:
            line_count += 1
            if processed_count >= 20:
                print("Processed 200 lines, stopping further processing.")
                break

            print(f"Processing line {line_count}: {line.strip()}")
            data = json.loads(line)

            # Translate the required fields if they exist
            if "text_raw" in data:
                if use_overlap:
                    data["text_raw_translated"] = chunk_and_translate_with_overlap(data["text_raw"])
                else:
                    data["text_raw_translated"] = chunk_and_translate(data["text_raw"])

            if "summary" in data:
                print("Found 'summary' field.")
                data["summary_translated"] = translate_text(data["summary"])

            if "metadata" in data and "ai_summary" in data["metadata"]:
                print("Found 'metadata.ai_summary' field.")
                data["metadata"]["ai_summary_translated"] = translate_text(data["metadata"]["ai_summary"])

            # Write the updated JSON object to the output file
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
            print(f"Processed line {line_count} successfully.")
            processed_count += 1

    print(f"Processing completed. Output saved to {output_path}")

# Paths to input and output files
input_jsonl_path = "/sise/home/avramit/pycharm_project_502/summarization-7-heb_up.jsonl"  # Replace with your input path
output_jsonl_path = "/sise/home/avramit/pycharm_project_502/output_new_2.jsonl"  # Replace with your output path

use_overlap = True
# Process the input file
process_jsonl(input_jsonl_path, output_jsonl_path)
print("File processing completed!")

# import json
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch
# import re
# import argparse
# import logging
# from tqdm import tqdm  # For progress bars
#
# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s [%(levelname)s] %(message)s',
#     handlers=[
#         logging.StreamHandler()
#     ]
# )
#
#
# def load_model_and_tokenizer(model_name):
#     logging.info("Loading model and tokenizer...")
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     logging.info(f"Using device: {device}")
#
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         device_map="auto" if torch.cuda.is_available() else None,
#         torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
#     )
#     model.to(device)
#     logging.info("Model loaded successfully.")
#
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#     logging.info("Tokenizer loaded successfully.")
#
#     return model, tokenizer, device
#
#
# def translate_text(text, model, tokenizer, device, max_new_tokens=3000):
#     logging.debug(f"Translating text: {text[:50]}...")  # Log a snippet
#     prompt = f"<s>[INST] You are a translator from Hebrew to English. Translate the following text in Hebrew to English: '{text}' [/INST]"
#
#     encoded = tokenizer(
#         prompt,
#         return_tensors="pt",
#         padding=True,
#         truncation=True,
#         max_length=tokenizer.model_max_length
#     ).to(device)
#
#     try:
#         outputs = model.generate(
#             input_ids=encoded.input_ids,
#             attention_mask=encoded.attention_mask,
#             do_sample=False,
#             max_new_tokens=max_new_tokens,
#             early_stopping=True
#         )
#         translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#
#         # Remove prompt markers if present
#         if "[INST]" in translated_text and "[/INST]" in translated_text:
#             translated_text = translated_text.split("[/INST]")[1].strip()
#
#         # Check for Hebrew characters
#         if re.search(r'[א-ת]', translated_text):
#             logging.warning("Translated text contains Hebrew characters. Consider manual review.")
#             # Optionally, handle as needed
#             # translated_text = re.sub(r'[א-ת]', '', translated_text)
#
#         return translated_text
#     except Exception as e:
#         logging.error(f"Error during translation: {e}")
#         return "Translation Error"
#
#
# def process_jsonl(input_path, output_path, model, tokenizer, device, use_overlap=False, max_lines=200):
#     logging.info(f"Processing input file: {input_path}")
#     try:
#         with open(input_path, 'r', encoding='utf-8') as infile, \
#                 open(output_path, 'w', encoding='utf-8') as outfile:
#
#             for line_num, line in enumerate(tqdm(infile, desc="Processing lines"), 1):
#                 if line_num > max_lines:
#                     logging.info(f"Reached processing limit of {max_lines} lines.")
#                     break
#                 try:
#                     data = json.loads(line)
#
#                     if "text_raw" in data:
#                         translated = chunk_and_translate(
#                             data["text_raw"], model, tokenizer, device, use_overlap
#                         )
#                         data["text_raw_translated"] = translated
#
#                     if "summary" in data:
#                         data["summary_translated"] = translate_text(
#                             data["summary"], model, tokenizer, device
#                         )
#
#                     if "metadata" in data and "ai_summary" in data["metadata"]:
#                         data["metadata"]["ai_summary_translated"] = translate_text(
#                             data["metadata"]["ai_summary"], model, tokenizer, device
#                         )
#
#                     outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
#                 except json.JSONDecodeError:
#                     logging.error(f"Invalid JSON at line {line_num}. Skipping.")
#                 except Exception as e:
#                     logging.error(f"Error processing line {line_num}: {e}")
#
#         logging.info(f"Processing completed. Output saved to {output_path}")
#     except FileNotFoundError as e:
#         logging.error(f"File not found: {e}")
#     except Exception as e:
#         logging.error(f"Unexpected error: {e}")
#
#
# def chunk_and_translate(text, model, tokenizer, device, use_overlap, chunk_size=1000, overlap_size=200):
#     if use_overlap:
#         return chunk_and_translate_with_overlap(text, model, tokenizer, device, chunk_size, overlap_size)
#     else:
#         return chunk_and_translate_simple(text, model, tokenizer, device)
#
#
# def chunk_and_translate_simple(text, model, tokenizer, device, sentences_per_chunk=7):
#     sentences = re.split(r'(?<=[.!?]) +', text)
#     translated_chunks = []
#
#     for i in range(0, len(sentences), sentences_per_chunk):
#         chunk = ' '.join(sentences[i:i + sentences_per_chunk])
#         translated_chunk = translate_text(chunk, model, tokenizer, device)
#         translated_chunks.append(translated_chunk)
#
#     return ' '.join(translated_chunks)
#
#
# def chunk_and_translate_with_overlap(text, model, tokenizer, device, chunk_size=1000, overlap_size=200):
#     tokens = tokenizer.tokenize(text)
#     chunks = []
#     start_idx = 0
#     while start_idx < len(tokens):
#         end_idx = min(start_idx + chunk_size, len(tokens))
#         chunk = tokens[start_idx:end_idx]
#         chunks.append(chunk)
#         start_idx += chunk_size - overlap_size
#
#     translated_chunks = []
#     previous_context = ""
#     for chunk in chunks:
#         chunk_text = tokenizer.convert_tokens_to_string(chunk)
#         translated_chunk = translate_text(chunk_text, model, tokenizer, device)
#
#         if previous_context and previous_context in translated_chunk:
#             translated_chunk = translated_chunk.replace(previous_context, "", 1).strip()
#
#         # Update previous_context with the last `overlap_size` tokens of the current chunk
#         overlap_tokens = chunk[-overlap_size:]
#         overlap_text = tokenizer.convert_tokens_to_string(overlap_tokens)
#         previous_context = translate_text(overlap_text, model, tokenizer, device)
#
#         translated_chunks.append(translated_chunk)
#
#     return ' '.join(translated_chunks)
#
#
# def main():
#     parser = argparse.ArgumentParser(description="Hebrew to English Translator")
#     parser.add_argument('--model_name', type=str, default="dicta-il/dictalm2.0-instruct", help='Pre-trained model name')
#     parser.add_argument('--input', type=str, required=True, help='Path to input JSONL file')
#     parser.add_argument('--output', type=str, required=True, help='Path to output JSONL file')
#     parser.add_argument('--use_overlap', action='store_true', help='Enable overlapping chunks for translation')
#     parser.add_argument('--max_lines', type=int, default=200, help='Maximum number of lines to process')
#
#     args = parser.parse_args()
#
#     model, tokenizer, device = load_model_and_tokenizer(args.model_name)
#     process_jsonl(
#         input_path=args.input,
#         output_path=args.output,
#         model=model,
#         tokenizer=tokenizer,
#         device=device,
#         use_overlap=args.use_overlap,
#         max_lines=args.max_lines
#     )
#
#
# if __name__ == "__main__":
#     main()
