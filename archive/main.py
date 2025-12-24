# # -*- coding: utf-8 -*-
# from transformers import pipeline
#
# # Load the text-generation model
# model = pipeline(
#     'text-generation',
#     'dicta-il/dictalm2.0',
#     torch_dtype="bfloat16",
#     device_map='auto'
# )
#
# # Hebrew sentence to translate
# hebrew_sentence = "המאמר מציג שלושה מתכונים לעוגות גבינה חגיגיות לכבוד חג השבועות: טירמיסו, עוגת גבינה קרה ועוגת פרזייה. המתכון הראשון הוא טירמיסו איטלקי בניחוח קפה, המורכב מבצק פריך, קרם קפה ושוקולד חלב וקרם גבינה. המתכון השני הוא לעוגת גבינה קרה עם שילוב של טעמי שוקולד, המורכבת מתחתית אפויה, רוטב שוקולד חלב וקציפת גבינה. המתכון השלישי הוא לעוגת פרזייה - טורט מסקרפונה ותות שדה, המורכבת משכבות של ביסקוויט שקדים, קציפת מסקרפונה ותות שדה, ומעוטרת במרנג. המאמר מלווה בצילומים הממחישים את שלבי ההכנה של כל עוגה"
#
# # Few-shot examples for better performance
# prompt = f"""
# תרגם את המשפט הבא מאנגלית לעברית:
#
# English: "Hello world"
# Hebrew: "שלום עולם"
#
# תרגם את המשפט הבא מעברית לאנגלית:
#
# Hebrew: "{hebrew_sentence}"
# English:
# """
#
# # Generate the translation
# output = model(prompt.strip(), do_sample=False, max_new_tokens=100)
#
# # Extract the generated text
# generated_text = output[0]["generated_text"].replace(prompt.strip(), "").strip()
#
# # Print the translation
# print("Translation:")
# print(generated_text)

import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Define model and tokenizer
model_name = "dicta-il/dictalm2.0-instruct"
print("Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
print("Model loaded successfully.")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Tokenizer loaded successfully.")

def translate_text(text):
    print(f"Translating text: {text}")
    prompt = f"<s>[INST] You are translator from hebrew to english. translate the following text in hebrew to english: '{text}' [/INST]"

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

        print(f"Cleaned translated text: {translated_text}")
        return translated_text
    except Exception as e:
        print(f"Error during translation: {e}")
        return "Translation Error"



# Function to process the JSONL file
def process_jsonl(input_path, output_path):
    print(f"Processing input file: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        line_count = 0
        for line in infile:
            line_count += 1
            print(f"Processing line {line_count}: {line.strip()}")
            data = json.loads(line)

            # Translate the required fields if they exist
            if "text_raw" in data:
                print("Found 'text_raw' field.")
                data["text_raw_translated"] = translate_text(data["text_raw"])

            if "summary" in data:
                print("Found 'summary' field.")
                data["summary_translated"] = translate_text(data["summary"])

            if "metadata" in data and "ai_summary" in data["metadata"]:
                print("Found 'metadata.ai_summary' field.")
                data["metadata"]["ai_summary_translated"] = translate_text(data["metadata"]["ai_summary"])

            # Write the updated JSON object to the output file
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
            print(f"Processed line {line_count} successfully.")
    print(f"Processing completed. Output saved to {output_path}")

# Paths to input and output files
input_jsonl_path = "/sise/home/avramit/pycharm_project_502/sum_heb.jsonl"  # Replace with your input path
output_jsonl_path = "/sise/home/avramit/pycharm_project_502/output.json"  # Replace with your output path

# Process the input file
process_jsonl(input_jsonl_path, output_jsonl_path)
print("File processing completed!")


#
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch
#
# model_name = "dicta-il/dictalm2.0-instruct"
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
#
# prompt = (
#     "<s>[INST] אתה מתורגמן. "
#     "תרגם את הטקסט הבא לעברית: 'Hello world' [/INST]"
# )
#
# input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
# outputs = model.generate(input_ids=input_ids, do_sample=False)
#
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))


# worked one!
# import ctranslate2
# import transformers
# import os
# import json
#
# # Set environment variables
# os.environ['LD_LIBRARY_PATH'] = '/home/avramit/.conda/envs/nmt_test/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
# print(os.environ['LD_LIBRARY_PATH'], flush=True)
#
# # Set target and source languages
# tgt_lang = "eng_Latn"
# src_lang = "heb_Hebr"  # Hebrew (source)
#
# # Initialize the translator and tokenizer
# translator = ctranslate2.Translator("/home/avramit/nllb_ct2_model_13", device="cuda", compute_type="auto")
# tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B", src_lang=src_lang)
# print("Using device:", translator.device, flush=True)
#
# input_file_path = "/home/avramit/translate_dataset/summarization-7-heb.jsonl"
# output_file_path = "/home/avramit/translate_dataset/summarization-7-heb_translate_english.jsonl"
#
# # Initialize a list to store translated records
# translated_records = []
#
# # Function to translate text
# def translate_text(text):
#     sentences = [s.strip() for s in text.split('.') if s.strip()]
#     translated_sentences = []
#
#     for i in range(0, len(sentences), 3):
#         chunk_sentences = sentences[i:i + 3]
#         chunk_text = '. '.join(chunk_sentences)
#         if not chunk_text.endswith('.'):
#             chunk_text += '.'
#
#         # Tokenize the chunk
#         source_tokens = tokenizer.encode(chunk_text, truncation=True, max_length=512)
#
#         # Convert tokens to source format for ctranslate2
#         source_tokens = tokenizer.convert_ids_to_tokens(source_tokens)
#         target_prefix = [tgt_lang]
#
#         # Translate the chunk
#         results = translator.translate_batch([source_tokens], target_prefix=[target_prefix])
#         target_tokens = results[0].hypotheses[0][1:]
#         translated_text = tokenizer.decode(tokenizer.convert_tokens_to_ids(target_tokens), skip_special_tokens=True)
#
#         translated_sentences.append(translated_text)
#
#     return ' '.join(translated_sentences)
#
# # Read input JSONL file
# with open(input_file_path, "r", encoding="utf-8") as file:
#     input_data = [json.loads(line) for line in file]
#
# # Iterate over the input data, process only 100 samples in chunks of 10
# batch_size = 10
# for idx, record in enumerate(input_data[:150]):
#     record["text_raw"] = translate_text(record["text_raw"])
#
#     # Check if 'ai_summary' exists before translating
#     if "ai_summary" in record.get("metadata", {}):
#         record["metadata"]["ai_summary"] = translate_text(record["metadata"]["ai_summary"])
#
#     record["summary"] = translate_text(record["summary"])
#     translated_records.append(record)
#
#     print(f"Processed record {idx + 1}/150", flush=True)
#
#     # Write to file after every 10 records
#     if (idx + 1) % batch_size == 0 or (idx + 1) == len(input_data[:150]):
#         with open(output_file_path, "a", encoding="utf-8") as file:
#             for translated_record in translated_records:
#                 file.write(json.dumps(translated_record, ensure_ascii=False) + "\n")
#         translated_records = []  # Clear the buffer
#
# print(f"Translation completed. Output saved to {output_file_path}", flush=True)

# import ctranslate2
# import transformers
# import os
#
# os.environ['LD_LIBRARY_PATH'] = '/home/avramit/.conda/envs/nmt_test/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
# print(os.environ['LD_LIBRARY_PATH'])
#
#
# # Set target and source languages
# tgt_lang = "eng_Latn"
# src_lang = "heb_Hebr"  # Hebrew (source)
#
# # Initialize the translator and tokenizer
# translator = ctranslate2.Translator("/home/avramit/nllb_ct2_model_13", device="cuda", compute_type="auto")
# tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B", src_lang=src_lang)
# print("Using device:", translator.device)
#
# input_file_path = "/tmp/pycharm_project_214/input_text.txt"
#     # Load input text from a UTF-8 encoded file
# with open(input_file_path, "r", encoding="utf-8") as file:
#     input_text = file.read()
#     # Define the input text in Hebrew
#     # Extract the first element of the tuple if input_text is a tuple
# if isinstance(input_text, tuple):
#     input_text = input_text[0]
#
#
# # Split the input text into sentences
# sentences = [s.strip() for s in input_text.split('.') if s.strip()]
#
# # Initialize a list to store the translated chunks
# translated_article = []
#
# # Process the text in chunks of 3 sentences
# for i in range(0, len(sentences), 4):
#     chunk_sentences = sentences[i:i+4]
#     chunk_text = '. '.join(chunk_sentences)
#     if not chunk_text.endswith('.'):
#         chunk_text += '.'
#
#     # Tokenize the chunk
#     source_tokens = tokenizer.encode(chunk_text, truncation=True, max_length=512)
#     print("Number of tokens in this chunk:", len(source_tokens))
#
#     # Convert tokens to source format for ctranslate2
#     source_tokens = tokenizer.convert_ids_to_tokens(source_tokens)
#     target_prefix = [tgt_lang]
#
#     # Translate the chunk
#     results = translator.translate_batch([source_tokens], target_prefix=[target_prefix])
#     target_tokens = results[0].hypotheses[0][1:]
#     translated_text = tokenizer.decode(tokenizer.convert_tokens_to_ids(target_tokens), skip_special_tokens=True)
#
#     # Append the translated text to the article
#     translated_article.append(translated_text)
#
#     # Print the original and translated chunk
#     print("Original Hebrew Chunk:")
#     print(chunk_text)
#     print("\nTranslated English Chunk:")
#     print(translated_text)
#     print("-----------")
#
# # Combine all translated chunks into a single article
# final_translated_article = ' '.join(translated_article)
#
# # Print the aggregated translated article
# print("\nAggregated Translated Article:")
# print(final_translated_article)


# -*- coding: utf-8 -*-
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
#
# def main():
#     # Load model and tokenizer
#     #tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
#
#
#     tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang="heb_Hebr")
#     model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
#     model.to("cuda")
#     input_file_path = "/tmp/pycharm_project_214/input_text.txt"
#     # Load input text from a UTF-8 encoded file
#     with open(input_file_path, "r", encoding="utf-8") as file:
#         input_text = file.read()
#     # Define the input text in Hebrew
#     # Extract the first element of the tuple if input_text is a tuple
#     if isinstance(input_text, tuple):
#         input_text = input_text[0]
#
#     # Split the input text into sentences (naive split by '.')
#     sentences = [s.strip() for s in input_text.split('.') if s.strip()]
#
#     translated_article = []
#
#     # Process in chunks of 3 sentences
#     for i in range(0, len(sentences), 3):
#         chunk_sentences = sentences[i:i + 3]
#         chunk_text = '. '.join(chunk_sentences)
#         if not chunk_text.endswith('.'):
#             chunk_text += '.'
#
#         # Tokenize the chunk and print the token count
#         inputs = tokenizer(
#             chunk_text,
#             return_tensors="pt",
#             truncation=True,
#             max_length=512
#         )
#         print("Number of tokens in this chunk:", inputs['input_ids'].shape[1])
#
#         inputs = {k: v.to("cuda") for k, v in inputs.items()}
#
#         # Translate the chunk
#         translated_tokens = model.generate(
#             **inputs,
#             forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
#             max_length=300,
#             num_beams=4,
#             repetition_penalty=2.0
#         )
#     # Decode the output
#         output_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
#
#         # Append the translated chunk to the article
#         translated_article.append(output_text)
#
#         # Print the chunk translation
#         print("Original Hebrew Chunk:")
#         print(chunk_text)
#         print("\nTranslated English Text:")
#         print(output_text)
#         print("-----------")
#
#     # Combine all translated chunks into a single article
#     final_translated_article = ' '.join(translated_article)
#
#     # Print the aggregated translated article
#     print("\nAggregated Translated Article:")
#     print(final_translated_article)
#
#
# if __name__ == "__main__":
#     main()
# #

# from transformers import AutoTokenizer
#
# def main():
#     # Define the source and target languages
#     src_lang = "heb_Hebr"
#     tgt_lang = "eng_Latn"
#
#     # Load the CTranslate2 model and tokenizer
#     translator = ctranslate2.Translator("nllb_ct2_model", device="cuda")
#     tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang=src_lang)
#
#     # Define the input text in Hebrew
#     input_text = "שלום, איך אתה מרגיש היום? אני מקווה שהכול בסדר."
#
#     # Prepend the source language token to the input text
#     tokens = tokenizer.tokenize(input_text)
#     tokens = [src_lang] + tokens
#
#     # Perform translation
#     results = translator.translate_batch([tokens], target_prefix=[[tgt_lang]])
#
#     # Decode the output tokens
#     translated_tokens = results[0].hypotheses[0]
#     output_text = tokenizer.convert_tokens_to_string(translated_tokens)
#
#     # Print results
#     print("Original Hebrew Text:")
#     print(input_text)
#     print("\nTranslated English Text:")
#     print(output_text)
#
# if __name__ == "__main__":
#     main()