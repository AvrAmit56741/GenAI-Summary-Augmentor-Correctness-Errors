"""Validation batch: 20 articles, length-corrected prompt (adds back the real production prompt's
150-200 word target, which the "thin" prompt had dropped, causing wild/unfair compression-ratio
variance across models -- see docs/error_finder_instructions_for_hebrew_llm.md for the full finding).

    python exp_length_corrected_val20_bedrock.py zai.glm-4.7-flash
    python exp_length_corrected_val20_bedrock.py mistral.ministral-3-14b-instruct
    python exp_length_corrected_val20_bedrock.py qwen.qwen3-32b-v1:0
    python exp_length_corrected_val20_bedrock.py us.anthropic.claude-haiku-4-5-20251001-v1:0

Output: current_model_summaries_<tag>_lengthcorrected_val20.csv
"""
import csv
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_current_model_summaries_targeted import pick_articles  # noqa: E402

CREDS = os.environ.get("BEDROCK_CREDS_FILE", os.path.expanduser(
    "~/PycharmProjects/urlf-classification/experiments/bedrock_classifier/bedrock_aws_creds.local.env"))
REGION = "us-east-1"
HERE = os.path.dirname(os.path.abspath(__file__))

# Long-form prompt: the REAL style that produced the calibration-anchor confirmed errors
# (business-6667074, culture-4533684, food-917568), taken verbatim from
# summary_augmentation/summary_generator_close_llm.py / summary_generator_ollama.py's
# summary_prompt_basic() -- "at least 15 sentences, no upper cap" (~44% compression in practice).
LONGFORM_SYSTEM = """אתה מומחה ביצירת סיכומים מפורטים ועשירים בפרטים בעברית.
משימתך היא להפוך את המאמר המקורי לסיכום נרטיבי אחד, מקיף, הכולל את כל המידע המרכזי.

### התהליך שלך בשני שלבים ###

**שלב 1: איסוף עובדות פנימי (ניתוח בלבד)**
קרא את המאמר במלואו וזהה את כל העובדות המרכזיות: שמות, תאריכים, נתונים סטטיסטיים, מקומות, אירועים מרכזיים ומסקנות עיקריות.
בנה רשימה מנטלית מלאה של עובדות אלו.
**אל תכתוב את הרשימה הזו בפלט.** שלב זה הוא לניתוח הפנימי שלך בלבד.

**שלב 2: כתיבת הסיכום הנרטיבי (הפלט הסופי)**
כעת, תוך שימוש ברשימת העובדות המלאה שבנית, כתוב סיכום אחד, רציף ומפורט.
משימתך היא לשלב במיומנות את כל העובדות מהרשימה לכדי נרטיב עברי קוהרנטי.
הטקסט הסופי צריך להיות אינפורמטיבי, קריא בקלות, ולהרגיש כמו מאמר שלם ועצמאי.

### כללים מחייבים לפלט הסופי (שלב 2) ###

1. **אורך:** הסיכום הסופי חייב להיות **בין 8 ל-12 משפטים בדיוק, ובסך הכל כ-150 עד 200 מילים**. זהו טווח מחייב ומדויק -- לא קצר יותר ולא ארוך יותר.
2. **שלמות עובדתית:** בתוך המגבלה הזו, הסיכום חייב לכלול את העובדות החשובות ביותר שזיהית בשלב 1 -- בחר את המרכזיות ביותר, אל תנסה לכלול את הכול.
3. **סגנון:** הפלט חייב להיות נרטיב רציף, לא רשימת נקודות מפורדות.
"""
CORRECTED_PROMPT = LONGFORM_SYSTEM + "\n\nהמאמר המקורי:\n<article>\n{article}\n</article>\n\nכתוב כעת את הסיכום הנרטיבי המלא, בין 8 ל-12 משפטים בלבד."


def load_creds():
    for line in open(CREDS):
        m = re.match(r"\s*(?:export\s+)?([A-Za-z_]+)=(.*)", line)
        if m:
            os.environ[m.group(1)] = m.group(2).strip().strip("\"'")


def main():
    model = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    offset = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    tag = re.sub(r"[^A-Za-z0-9_-]", "_", model)
    suffix = f"_restricted_val20_offset{offset}" if offset > 0 else "_restricted_val20"
    out = os.path.join(HERE, f"current_model_summaries_{tag}{suffix}.csv")

    load_creds()
    try:
        import truststore; truststore.inject_into_ssl()
    except Exception:
        pass
    import boto3
    brt = boto3.Session(region_name=REGION).client("bedrock-runtime")

    arts = pick_articles(n, offset=offset)
    print(f"model={model}  selected {len(arts)} known error-prone articles "
          f"(max_sev: {arts.max_sev.value_counts().to_dict()})", flush=True)

    fieldnames = ["doc_id", "source", "orig_max_severity", "orig_n_errors", "premise",
                  "generative_summary", "model", "prompt_style", "gen_seconds"]
    csv_file = open(out, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    for i, r in enumerate(arts.itertuples()):
        prompt = CORRECTED_PROMPT.format(article=r.premise)
        t0 = time.time()
        for attempt in range(4):
            try:
                resp = brt.converse(modelId=model, messages=[{"role": "user", "content": [{"text": prompt}]}],
                                     inferenceConfig={"maxTokens": 1000, "temperature": 0.2})
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        dt = time.time() - t0
        summary = "".join(c.get("text", "") for c in resp["output"]["message"]["content"]).strip()
        row = {"doc_id": r.doc_id, "source": r.source, "orig_max_severity": r.max_sev,
               "orig_n_errors": r.n_err, "premise": r.premise, "generative_summary": summary,
               "model": model, "prompt_style": "restricted_8to12sentences", "gen_seconds": round(dt, 1)}
        writer.writerow(row)
        csv_file.flush()
        print(f"  [{i + 1}/{len(arts)}] {r.doc_id} (orig_sev={r.max_sev}) ({dt:.1f}s, {len(summary)} chars, "
              f"{len(summary.split())} words)", flush=True)
    csv_file.close()

    import pandas as pd
    df = pd.read_csv(out)
    comp = (df.generative_summary.str.len() / df.premise.str.len()).mean()
    print(f"\nsaved {len(df)} summaries -> {out}")
    print(f"mean compression ratio: {comp*100:.1f}% (target: ~16.7%, matching the real MAFAT baseline)")


if __name__ == "__main__":
    main()
