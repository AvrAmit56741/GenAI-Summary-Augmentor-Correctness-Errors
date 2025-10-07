from typing import Type, Optional
from pydantic import BaseModel, Field
from pathlib import Path
import json
import re
import os
import datetime
import requests
import time
from summary_augmentation.logger_config import LoggerFactory

# Configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
# User-configurable model served by Ollama (examples: "llama3.1:latest", "gemma3:12b", "gpt-oss:20b")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:14b")
# Generation options (adjust as needed)
OLLAMA_OPTIONS = {
    "temperature": 0.0,
    "top_p": 0.9,
    "num_ctx": 8192,
    # "num_predict": 2048,  # uncomment to hard-limit tokens if desired
}

logger = LoggerFactory.get_logger(__name__, console_level="INFO", file_level="DEBUG")


class ArticleSummary(BaseModel):
    article_summary: str = Field(description="HEBREW summary generative rich paraphrasing")


def summary_prompt_basic(source_article: str):
    """
    Hebrew long-form narrative summary. Enforces JSON with article_summary and length/style constraints.
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


def ollama_chat(host: str, model: str, system_message: str, user_message: str, options: dict) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
        "options": options or {},
    }
    logger.info(f"Ollama call | model={model} | host={host}")
    logger.debug(f"Request sizes | system_len={len(system_message)} | user_len={len(user_message)} | options={options}")
    t0 = time.time()
    r = requests.post(f"{host}/api/chat", json=payload, timeout=120)
    dt = time.time() - t0
    try:
        r.raise_for_status()
    except Exception as e:
        logger.error(f"Ollama HTTP error after {dt:.2f}s | status={getattr(r, 'status_code', 'NA')} | err={e}")
        raise
    data = r.json()
    # Ollama returns a single message.content when stream=False
    if isinstance(data, dict) and "message" in data and isinstance(data["message"], dict):
        content = data["message"].get("content", "")
        logger.info(f"Ollama response OK | latency={dt:.2f}s | content_len={len(content)}")
        return content
    logger.warning("Unexpected Ollama response format; returning empty string")
    return ""


def _extract_json(text: str) -> Optional[str]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return m.group(0) if m else None


def llm_query_ollama(system_message: str, user_message: str, output_object: Type[BaseModel], model: str) -> Optional[BaseModel]:
    raw = ollama_chat(OLLAMA_HOST, model, system_message, user_message, OLLAMA_OPTIONS)
    # Try direct JSON
    try:
        res = output_object.model_validate_json(raw)
        logger.info(f"Parse path=direct_json | raw_len={len(raw)}")
        return res
    except Exception:
        j = _extract_json(raw)
        if j:
            try:
                res = output_object.model_validate_json(j)
                logger.info(f"Parse path=extracted_json | raw_len={len(raw)} | json_len={len(j)}")
                return res
            except Exception as e2:
                logger.debug(f"Extracted JSON failed: {e2}")
        # Final fallback: wrap raw text
        try:
            res = output_object.model_validate({"article_summary": raw})
            logger.info(f"Parse path=wrapped_text | raw_len={len(raw)}")
            return res
        except Exception as e3:
            logger.error(f"All parse paths failed | raw_len={len(raw)} | err={e3}")
            return None


def summary_ollama(article: str) -> Optional[ArticleSummary]:
    system, user = summary_prompt_basic(article)
    return llm_query_ollama(system, user, ArticleSummary, OLLAMA_MODEL)


def main(input_path: str, output_path: str, start_line: int = 0, end_line: int = None):
    input_path = Path(input_path)
    output_path = Path(output_path)

    logger.info(f"Starting Ollama summary generation")
    logger.info(f"Config | host={OLLAMA_HOST} | model={OLLAMA_MODEL} | options={OLLAMA_OPTIONS}")
    logger.info(f"I/O | input={input_path} | output={output_path} | start={start_line} | end={end_line}")

    with input_path.open("r", encoding="utf-8") as infile:
        all_lines = infile.readlines()

    if end_line is None:
        end_line = len(all_lines)

    logger.info(f"Loaded {len(all_lines)} lines; processing range [{start_line}, {end_line})")

    written = 0
    with output_path.open("w", encoding="utf-8") as outfile:
        for i, line in enumerate(all_lines):
            if i < start_line or i >= end_line:
                continue

            try:
                data = json.loads(line)
                article = data.get("text_raw", "")
                if not article:
                    logger.info(f"Line {i}: missing 'text_raw'; skipping")
                    continue

                logger.info(f"Line {i}: generating summary (article_len={len(article)})")
                t0 = time.time()
                summary = summary_ollama(article)
                latency = time.time() - t0
                if not summary:
                    logger.error(f"Line {i}: generation failed (latency={latency:.2f}s)")
                    continue

                new_summary_entry = {
                    "generative_summary": summary.article_summary,
                    "model": OLLAMA_MODEL,
                    "prompt": "summary_prompt_basic"
                }

                if "metadata" not in data or not isinstance(data["metadata"], dict):
                    data["metadata"] = {}

                if "generative_summaries" not in data["metadata"]:
                    data["metadata"]["generative_summaries"] = []

                data["metadata"]["generative_summaries"].append(new_summary_entry)
                outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
                written += 1
                logger.info(f"Line {i}: written summary (len={len(summary.article_summary)} | latency={latency:.2f}s)")

            except Exception as e:
                logger.error(f"Line {i}: error {e}")
                continue

    logger.info(f"Completed. Wrote {written} records to {output_path}")


if __name__ == "__main__":
    start = 150
    end = 250
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    input_file = r"C:\\Users\\amita\\Desktop\\Thesis\\Correctness\\data\\summarization-7-heb.jsonl\\summarization-7-heb_2.jsonl"
    model_tag = re.sub(r"[^A-Za-z0-9_.-]+", "-", OLLAMA_MODEL)
    output_file = rf"C:\\Users\\amita\\Desktop\\Thesis\\summaries_ollama_{model_tag}_{timestamp}_start_row_{start}_end_row_{end}.jsonl"
    main(input_file, output_file, start_line=start, end_line=end) 