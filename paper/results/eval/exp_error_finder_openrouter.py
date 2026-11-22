"""Error-finder judging pass on the Qwen3.8 Flash / DeepSeek V4 Pro 0813 generated summaries,
via OpenRouter -- same taxonomy, same prompt, same gate as the project's own production pipeline.

Uses the EXACT system+user prompt text from summary_augmentation/correctness_errors_prompts.py's
correctness_errors_finder_prompt() (verbatim, copied below rather than imported, since the real
function builds its format_instructions from a LangChain parser object we don't want as a
dependency here -- the JSON schema instructions below are a manual equivalent of that same
TotalCorrectnessErrors/CorrectnessError Pydantic schema from
summary_augmentation/correctness_errors_llm_helper_functions.py's correctness_error_finder()).

Judge model: Claude Sonnet 5.5 via OpenRouter (the production pipeline's own correctness_error_finder()
calls llm_query_to_claude -- this mirrors that choice without depending on the repo's own API keys,
which an earlier session found dead; OPENROUTER_API_KEY is confirmed working this session).

    python exp_error_finder_openrouter.py current_model_summaries_qwen_qwen3_8-flash_targeted.csv
    python exp_error_finder_openrouter.py current_model_summaries_deepseek_deepseek-v4-pro-0813_targeted.csv

Output: one row per CONFIRMED finding (empty-list summaries produce no rows), same column layout as
thesisdatasets/error_finder/*.csv, saved to <input_stem>_errors_found.csv.
"""
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
JUDGE_MODEL = "anthropic/claude-sonnet-5.5"

SYSTEM_MESSAGE = """
אתה מומחה מחקר בבלשנות חישובית עברית ועיבוד שפה טבעית. המשימה שלך: לגלות ולקטלג שגיאות עובדתיות ייחודיות הנובעות ספציפית מהמורכבות המורפולוגיה של השפה העברית.

היה חוקר מקורי - חפש שגיאות שלא נתקלת בהן קודם!

## מה הופך שגיאה ל"מיוחדת לעברית":
השגיאה חייבת לנבוע מתכונות מורפולוגיות שקיימות בעברית אבל לא בשפות כמו אנגלית:

🔴 הנחיות מהותיות למניעת שגיאות! 🔴
חשוב מאוד: התעלם מכל סוגי השגיאות הבאים באופן מוחלט:
- שגיאות כתיב (מלבד אם הן משנות מילה למילה בעלת משמעות אחרת).
- שגיאות דקדוק או ניסוח שאינן משנות את העובדות.
- שגיאות סטנדרטיות שאינן ייחודיות לעברית (כגון תאריכים, מספרים, שמות פרטיים שהועתקו לא נכון).
- פערים בקישוריות בין משפטים שאינם יוצרים אי-התאמה עובדתית.
התמקד אך ורק בשגיאות עובדתיות שנוצרו בגלל התכונות המורפולוגיות הייחודיות לעברית.


בצע מבחן ניגוד באנגלית: האם אותה שגיאה הייתה יכולה להתקיים ללא מורפולוגיה עברית (למשל באנגלית)? אם כן – דחה את השגיאה
שער סינון: אם אינך מציין לפחות אחת מהתכונות: בניין, גוף/מגדר/מספר, סמיכות, ה״א הידיעה, כינוי קניין מוצמד, קליטיק, החזר רשימה ריקה.

### תכונות מורפולוגיות ייחודיות לעברית:


#### מורכבות מורפולוגית אקסטרמית:
- **מילים רב-מורפמיות**: מילה אחת = 7+ מורפמות (דוגמה: "וכשמביתנו" = ו+כש+מ+ה+בית+של+אנחנו)
- **עמימות מורפולוגית קיצונית**: מילה אחת עם פירושים שונים לחלוטין ("הקפה" = ה+קפה / הקפה-מסלול / הק+פה+של+היא)
- **מורפולוגיה לא-רציפה**: שורש תלת-עיצורי + מישקל עם שילובים מורכבים

#### תהליכי מיזוג ופירוק מורפולוגי:
- **מיזוג מורפמות**: צירוף מורפמות מרובות למילה אחת
- **פירוק שגוי**: פירושים מרובים לאותה רצף אותיות
- **צירופים מורפולוגיים חדשים**: שילובי מורפמות שלא נראו אבל תקפים

#### שגיאות התאמה מורפולוגית:
- אי-התאמה בין מגדר של מילים שונות באותו משפט
- אי-התאמה בין גוף וזמן בפועל לבין כינויים

#### שגיאות העברת הקשרים תרבותיים:
- שגיאות מגדר הנובעות מהקשרים של שפות אחרות (שם זר שמקבל מגדר שגוי)
- הנחות תרבותיות שמשפיעות על מורפולוגיה עברית

#### מערכות דקדוקיות מתקדמות:
- **מערכת בניינים**: 7 בניינים שמשנים פעיל↔פסיבי↔רפלקסיבי↔קאוזטיבי
- **מערכת מגדר מורכבת**: מגדר שמשפיע על כל חלקי המשפט + מגדר טבעי vs. דקדוקי
- **מצב סמיכות (smixut)**: מבנה "X של Y" עם חוקי יידוע מורכבים
- **כינויי קניין מוצמדים**: סיומות שמציינות בעלות רב-ממדית

#### מערכות זמן ויחס ייחודיות:
- **זמנים עבריים מיוחדים**: עבר/הווה/עתיד/ציווי עם השפעות על צורת הפועל
- **מערכת גופים תלת-ממדית**: גוף (1/2/3) × מגדר (זכר/נקבה) × מספר (יחיד/רבים)
- **ה"א הידיעה**: המשפיעה על רמת הספציפיות והיחסים התחביריים

#### הקשרים תרבותיים-לשוניים:
- **רמות לשון מעורבות**: עברית מקראית/משנית/מודרנית במשפט אחד
- **דו-משמעויות תרבותיות**: מילים עם משמעות דתית/חילונית שונה
- **השפעות ארמיות ואחרות**: מבנים שנכנסו מהארמית והשפיעו על המורפולוגיה

#### תכונות ייחודיות נוספות:
- **צירופים מורפולוגיים חדשים**: שילובי מורפמות שלא נראו אבל תקפים
- **פילוח vs. מיזוג מורפולוגי**: אתגר בפירוק למורפמות ובבניה מחדש
- **מילות יחס מוצמדות מורכבות**: (של+מ+ה, כש+מ+ה)
- **דרגות השוואה ייחודיות**: מבנים עבריים לביטוי עוצמה ושוואה

#### שגיאות מגדר דקדוקי עם השפעה עובדתית:
- בלבול בין גוף מוסדי (עמותה) למקום פיזי שמשנה איפה מתרחשת פעולה
- החלפת מגדר שמשנה זיהוי אנשים או ישויות בפועל

### שגיאות העברת הקשרים תרבותיים-מגדריים (קריטי!):
- **התנגשות מגדר שמות זרים**: שם שנתפס כנשי/גברי בתרבות אחת אבל האדם הוא מהמגדר ההפוך במציאות
- **דוגמה קלאסית**: "ננסי" - שם נשי באנגלית, אבל במקור מתייחסים אליו כזכר → בסיכום משנים לנקבה
- **העברת סטריוטיפים תרבותיים**: תפקידים או שמות שמקושרים למגדר מסוים בתרבות זרה
- **שגיאת "תיקון" מגדר**: המודל "מתקן" את המגדר לפי הציפיות התרבותיות במקום לפי העובדות

## דפוסי שגיאות חובה לבדיקה:

### דפוס A: החלפת בניין עם היפוך סוכנות
- מקור: "המחבל נעצר" (נפעל) → סיכום: "המחבל עצר" (פעל)
- שגיאה: סובל הופך לפועל

### דפוס B: כינוי קניין עם שינוי בעלות
- מקור: "ביתו של רמי" → סיכום: "ביתה של רמי"
- שגיאה: מגדר כינוי משנה זהות בעלים

### דפוס C: פירוק מורפולוגי שגוי
- מקור: "וכשמביתנו" (ו+כש+מ+בית+נו) → סיכום פירוק אחר
- שגיאה: פירוק שגוי משנה משמעות

Hebrew causative/voice flips (same root, different binyan)
Detect: root same; binyan change (הפעיל/פיעל/נפעל/התפעל) → agent/patient switch or causativity change.
Example: Original: "המשטרה עצרה את החשוד" → Error: "המשטרה נעצרה על ידי החשוד".
Factual: who did what to whom changes.
Possession suffixes and antecedent drift
Detect: כינויי קניין (ו/ה/ם/ן) on nouns/preps (עליו/אליה/ספרה/דבריהם) and nearest valid antecedent.
Example: "השרה הציגה את מסקנותיה" → "השרה הציגה את מסקנותיו".
Factual: ownership/source attribution flips.
Homographic profession/role with gender agreement
Detect: nouns like "מורה/מרצה/שופט/רופא/מנהל" + verb/adjective gender chain.
Example: "המרצה הציגה מחקרה" → "המרצה הציג מחקרו".
Factual: person identity/gender facts change.
Directional prepositions and clitic stacking
Detect: מ-/ל-/ב-/על/אל swaps; source→target confusion; clitic assimilation with ה ("בה", "לַ").
Example: "נשלח מהשר לראש העיר" → "נשלח לשר מראש העיר".
Factual: sender/receiver reversed.


עוד דוגמאות
החלפת כינויי קניין חבורים (סיומות ו/ה/ם/ן) כך שיתייחסו לאדם או דבר אחר במשפט תוך יצירת שגיאה עובדתית ברורה.
שגיאה שנובעת מהחלפה של אחד משני רכיבי צירוף סמיכות (שמות עצם צמודים), באופן שמשנה את המשמעות העובדתית של הצירוף, ומוביל להבנה שונה של הישות או המידע המרכזי בטענה.
שינוי שגוי של זמן פועל (עבר ↔ הווה ↔ עתיד), הגורם לשיבוש בזמני הפעולה במשפט ויוצר טענה שונה מבחינה כרונולוגית או סתירה עם תיאורי זמן קיימים
שינוי רמת הספציפיות של מידע בטקסט – בין אם באמצעות ה״א הידיעה או על ידי החלפת ישות מזוהה בביטוי כללי – כך שהטענה הופכת לעובדתית שונה.
ניצול מילים שכתיבתן זהה לזכר ונקבה כדי ליצור טעות בזיהוי מגדר הדמות, מה שמוביל לטעות עובדתית על זהות האדם
"שגיאה עובדתית הנובעת מהחלפת גוף הפועל בין גוף ראשון לגוף שלישי, באופן שמטשטש או משנה את זהות מי שמבצע את הפעולה.
החלפת בניין הפועל באופן המהפך מי עשה מה למי - הופך עושה לסובל, קונה למוכר, תובע לנתבע
שיבוש במשפטים עם פועל משועבד שגורם להיפוך בסוכנות או בכיוון הפעולה – יוצר בלבול בזהות הפועל לעומת מושא הפעולה.
שיבוש בזהות הגורם המבצע והגורם המקבל בפעולה כיוונית, שנוצר על ידי החלפת מילת יחס או שינוי סדר הגורמים במשפט
### שגיאות החלפת תפקידים מורפולוגיים:
איך שינוי בבניין/גוף/מגדר משנה מי עושה מה למי

 שגיאות פירוק/מיזוג מורפולוגי:
איך פירוק שגוי של מילה רב-מורפמית יוצר עובדה שגויה

 שגיאות עמימות מורפולוגית:
איך בחירה שגויה בפירוש מילה רב-משמעית משנה עובדות

 שגיאות הקשר תרבותי-לשוני:
איך רמות לשון או הקשרים תרבותיים יוצרים עיוותים עובדתיים

 שגיאות זמניות ויחסיות מורכבות:
איך המערכות הזמניות והיחסיות הייחודיות לעברית יוצרות בלבול

### דפוסי שגיאות נוספים לחיפוש:
- **שגיאות בבניינים**: החלפת בניין שמשנה פעיל/פסיבי/רפלקסיבי
- **שגיאות בזמנים מורכבים**: טעויות בצורות זמן ייחודיות לעברית
- **שגיאות בקליטיקים**: טעויות במילות יחס מוצמדות מורכבות
- **שגיאות בכינויי קניין**: החלפת כינויים שמשנה בעלות

## כללי זיהוי קריטיים:
- התמקד **אך ורק** בשגיאות שמיוחדות לעברית
- התעלם משגיאות כלליות: מספרים שגויים, תאריכים שגויים, שמות פרטיים שגויים
- הכרחי: הסבר למה השגיאה לא יכולה לקרות בשפות כמו אנגלית
- חפש דוגמאות שבהן המבנה המורפולוגי הוא הגורם הישיר והיחיד לשגיאה

היה חוקר סקרן וגלה דוגמאות מקוריות שטרם תוארו!

חשוב!!!
חשוב מאוד! השגיאות העובדתיות צריכות להיות שגיאות עובדתיות מובהקות, ברורות וחד משמעיות שמשנות לגמרי את העובדה ויוצרת אי עקביות\\ממש שקר בין המאמר לסיכום
לא מעניינות שגיאות דקדוק, סגנון או קישוריות שלא משפיעות על התוכן העובדתי!
חשוב!!!

לפני שתדווח על שגיאה - שאל את עצמך:
1. האם זה משנה משהו מהותי בעובדות?
2. האם קורא יבין משהו שונה לגמרי מהמציאות?
3. האם זה יכול להוביל להחלטה שגויה בהתבסס על מידע שגוי?

אם התשובה לא, אל תדווח על השגיאה.
דוגמה לשגיאה טובה: "שר החוץ אמר" במקום "השרה אמרה" - משנה זהות
דוגמה לשגיאה רעה: "הכתבה דנה" במקום "המאמר דן" - לא משנה תוכן עובדתי
"""

FORMAT_INSTRUCTIONS = """החזר אובייקט JSON יחיד ותקין בדיוק במבנה הזה (ואין להוסיף טקסט לפני או אחרי ה-JSON):
{
  "summary_article_heb_correctness_errors": [
    {
      "correctness_error_name": "<שם מקורי וקצר בעברית לסוג השגיאה>",
      "correctness_error_reasoning": "<הסבר שלב-אחר-שלב איך המורפולוגיה העברית הייחודית יצרה את אי-ההתאמה העובדתית>",
      "mrl_unique_hebrew_features": "<רשימת התכונות המורפולוגיות הייחודיות המעורבות: בניינים, גוף/מגדר/מספר, סמיכות, ה\\"א הידיעה, כינוי קניין מוצמד, קליטיק>",
      "why_its_factual_error": "<הסבר למה זו שגיאה עובדתית, איך היא שוברת עובדה מהמאמר>",
      "evidence_from_summary": "<ציטוט מדויק מהסיכום>",
      "evidence_from_source_article": "<ציטוט מדויק מהמאמר המקורי התומך בעובדה הנכונה>",
      "severity": "<מספר שלם 1 עד 5>"
    }
  ]
}
אם אין שגיאות, החזר: {"summary_article_heb_correctness_errors": []}

חשוב לתקינות ה-JSON: בתוך ערכי המחרוזות (strings) אסור להשתמש בתו גרש כפול ("\\"") כלל -- גם לא
לציטוט מילים מהטקסט ולא למונחים כמו ה"א הידיעה. השתמש במקום זאת בגרש בודד (') או בגרשיים עבריים
(״), אחרת ה-JSON ייפגם ולא ניתן יהיה לפרסר אותו."""


def user_message(source_article, summary):
    return f"""
מאמר מקורי:
{source_article}

סיכום לבדיקה:
{summary}

השוה בין המאמר המקורי לסיכום וזהה שגיאות דיוק עובדתי ייחודיות הנובעות מהמורכבות המורפולוגית של העברית.

חשוב במיוחד:
- חפש שגיאות שטרם תוארו או נתקלת בהן
- היה יצירתי בזיהוי דפוסים חדשים
- התמקד במקרים שבהן העברית מאפשרת עיוותים שלא קיימים בשפות אחרות
- שים לב למילים רב-מורפמיות ולעמימויות מורפולוגיות קיצוניות

לכל שגיאה שתמצא:
- תן לה שם מקורי ומדויק בעברית
- הסבר בפירוט איך המורפולוגיה העברית יצרה את השגיאה העובדתית
- פרט את התכונות הלשוניות הייחודיות שמעורבות (בניינים, גופים, מגדר, סמיכות, קליטיקים וכו')
- הבא ראיות מדויקות מהטקסטים
- **הכרחי**: הסבר למה זו שגיאה שלא יכולה לקרות בשפות כמו אנגלית

אם אין שגיאות מיוחדות לעברית - החזר רשימה ריקה.

התגובה צריכה להיות ב-JSON בפורמט המדויק הזה:
{FORMAT_INSTRUCTIONS}
"""


FIELD_NAMES = ["correctness_error_name", "correctness_error_reasoning", "mrl_unique_hebrew_features",
               "why_its_factual_error", "evidence_from_summary", "evidence_from_source_article", "severity"]


def _field_based_repair(text):
    """Fallback for when the judge emits a stray unescaped double-quote inside a string value
    (observed, despite the prompt now forbidding it) and strict json.loads chokes on it. Instead
    of balancing quotes generically, split on each known field-name key -- which the model can't
    accidentally reproduce inside a value -- so one malformed quote only corrupts that one field,
    not the whole parse."""
    # One chunk per finding: split on the field that always starts a new object.
    chunks = re.split(r'"correctness_error_name"\s*:\s*"', text)[1:]
    findings = []
    for chunk in chunks:
        obj = {}
        # name: everything up to the next field key (the opening quote was already consumed by the split)
        m = re.search(r'^(.*?)"\s*,\s*"correctness_error_reasoning"\s*:\s*"', chunk, re.S)
        if not m:
            continue
        obj["correctness_error_name"] = m.group(1)
        rest = chunk[m.end():]
        for i, field in enumerate(FIELD_NAMES[1:], start=1):
            next_field = FIELD_NAMES[i + 1] if i + 1 < len(FIELD_NAMES) else None
            if field == "severity":
                m = re.search(r'^(.*?)"\s*[,}]', rest, re.S)
            elif next_field:
                m = re.search(rf'^(.*?)"\s*,\s*"{next_field}"\s*:\s*"', rest, re.S)
            else:
                continue
            if not m:
                break
            obj[field] = m.group(1)
            rest = rest[m.end():]
        if len(obj) == len(FIELD_NAMES):
            findings.append(obj)
    return {"summary_article_heb_correctness_errors": findings}


def extract_json(text):
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    # Last resort: field-based repair (handles stray unescaped quotes inside values).
    repaired = _field_based_repair(text)
    if repaired["summary_article_heb_correctness_errors"]:
        return repaired
    raise ValueError(f"could not parse JSON from: {text[:300]!r}")


def judge_one(key, source_article, summary):
    import requests
    body = {
        "model": JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": user_message(source_article, summary)},
        ],
        # Claude Sonnet 5.5 is also a reasoning model on OpenRouter, but unlike qwen/qwen3.8-flash
        # it MANDATES reasoning on this endpoint (confirmed: reasoning.enabled=False -> 400
        # "Reasoning is mandatory for this endpoint and cannot be disabled"), and reasoning requires
        # temperature=1.0 (Anthropic's extended-thinking constraint). So instead of disabling it,
        # cap it at reasoning.max_tokens and give a generous overall ceiling so the actual JSON
        # (confirmed to need ~2-5.5k tokens for a multi-finding response) always has room left.
        "temperature": 1.0,
        "max_tokens": 16000,
        "reasoning": {"max_tokens": 4000},
    }
    for attempt in range(4):
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=body, timeout=180,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"] or ""
            data = extract_json(content)
            return data.get("summary_article_heb_correctness_errors", [])
        except Exception as e:
            if attempt == 3:
                raise
            print(f"    retry {attempt} after error: {e}", flush=True)
            time.sleep(2 ** attempt)


def main():
    in_path = sys.argv[1]
    if not os.path.isabs(in_path):
        in_path = os.path.join(HERE, in_path)

    try:
        import truststore; truststore.inject_into_ssl()
    except Exception:
        pass
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(REPO, ".env"), override=True)
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY not found in environment / .env")

    import pandas as pd
    df = pd.read_csv(in_path)
    print(f"judging {len(df)} summaries from {os.path.basename(in_path)} with {JUDGE_MODEL} ...", flush=True)

    out_rows = []
    n_with_errors = 0
    for i, r in enumerate(df.itertuples()):
        t0 = time.time()
        findings = judge_one(key, r.premise, r.generative_summary)
        dt = time.time() - t0
        if findings:
            n_with_errors += 1
        for f in findings:
            out_rows.append({
                "doc_id": r.doc_id, "source": r.source, "model": r.model,
                "premise": r.premise, "generative_summary": r.generative_summary,
                "correctness_error_name": f.get("correctness_error_name"),
                "correctness_error_reasoning": f.get("correctness_error_reasoning"),
                "mrl_unique_hebrew_features": f.get("mrl_unique_hebrew_features"),
                "why_its_factual_error": f.get("why_its_factual_error"),
                "evidence_from_summary": f.get("evidence_from_summary"),
                "evidence_from_source_article": f.get("evidence_from_source_article"),
                "severity": f.get("severity"),
            })
        print(f"  [{i + 1}/{len(df)}] {r.doc_id} -> {len(findings)} finding(s) ({dt:.1f}s)", flush=True)

    out_path = in_path.replace(".csv", "_errors_found.csv")
    pd.DataFrame(out_rows).to_csv(out_path, index=False)
    print(f"\nsaved {len(out_rows)} findings from {n_with_errors}/{len(df)} summaries -> {out_path}")


if __name__ == "__main__":
    main()
