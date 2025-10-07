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
def correctness_errors_finder_prompt(source_article, claims_decomposition_list, output_instructions_parser):

    system_message = f"""
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

#### מורפולוגיה מורכבת ורב-שכבתית:
- **מילים רב-מורפמיות**: מילה אחת = 7+ מורפמות (כמו "וכשמביתנו")
- **עמימות מורפולוגית קיצונית**: מילה אחת עם פירושים מרובים לחלוטין ("הקפה")
- **מורפולוגיה לא-רציפה**: שורש תלת-עיצורי + מישקל
- **קליטיקים מורכבים**: צירוף מילות יחס מוצמדות (ו+כש+מ+ה)

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

#### מורכבות מורפולוגית אקסטרמית:
- **מילים רב-מורפמיות**: מילה אחת = 7+ מורפמות (דוגמה: "וכשמביתנו" = ו+כש+מ+ה+בית+של+אנחנו)
- **עמימות מורפולוגית קיצונית**: מילה אחת עם פירושים שונים לחלוטין ("הקפה" = ה+קפה / הקפה-מסלול / הק+פה+של+היא)

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
Detect: מ-/ל-/ב-/על/אל swaps; source→target confusion; clitic assimilation with ה (“בה”, “לַ”).
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
חשוב מאוד! השגיאות העובדתיות צריכות להיות שגיאות עובדתיות מובהקות, ברורות וחד משמעיות שמשנות לגמרי את העובדה ויוצרת אי עקביות\ממש שקר בין המאמר לסיכום
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



    user_message = f"""
מאמר מקורי:
{source_article}

סיכום לבדיקה:
{claims_decomposition_list}

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
{output_instructions_parser.get_format_instructions()}
"""

    return system_message, user_message
# def correctness_errors_finder_prompt(source_article, claims_decomposition_list, output_instructions_parser):
#
#     system_message = f"""
# אתה מומחה מחקר בבלשנות חישובית עברית ועיבוד שפה טבעית. המשימה שלך: לגלות ולקטלג שגיאות עובדתיות ייחודיות הנובעות ספציפית מהמורכבות המורפולוגיה של השפה העברית.
#
# היה חוקר מקורי - חפש שגיאות שלא נתקלת בהן קודם!
#
# ## מה הופך שגיאה ל"מיוחדת לעברית":
# השגיאה חייבת לנבוע מתכונות מורפולוגיות שקיימות בעברית אבל לא בשפות כמו אנגלית:
#
# ### תכונות מורפולוגיות ייחודיות לעברית:
#
#
# #### מורכבות מורפולוגית אקסטרמית:
# - **מילים רב-מורפמיות**: מילה אחת = 7+ מורפמות (דוגמה: "וכשמביתנו" = ו+כש+מ+ה+בית+של+אנחנו)
# - **עמימות מורפולוגית קיצונית**: מילה אחת עם פירושים שונים לחלוטין ("הקפה" = ה+קפה / הקפה-מסלול / הק+פה+של+היא)
# - **מורפולוגיה לא-רציפה**: שורש תלת-עיצורי + מישקל עם שילובים מורכבים
#
# #### תהליכי מיזוג ופירוק מורפולוגי:
# - **מיזוג מורפמות**: צירוף מורפמות מרובות למילה אחת
# - **פירוק שגוי**: פירושים מרובים לאותה רצף אותיות
# - **צירופים מורפולוגיים חדשים**: שילובי מורפמות שלא נראו אבל תקפים
#
# #### שגיאות התאמה מורפולוגית:
# - אי-התאמה בין מגדר של מילים שונות באותו משפט
# - אי-התאמה בין גוף וזמן בפועל לבין כינויים
#
# #### שגיאות העברת הקשרים תרבותיים:
# - שגיאות מגדר הנובעות מהקשרים של שפות אחרות (שם זר שמקבל מגדר שגוי)
# - הנחות תרבותיות שמשפיעות על מורפולוגיה עברית
#
# #### מורפולוגיה מורכבת ורב-שכבתית:
# - **מילים רב-מורפמיות**: מילה אחת = 7+ מורפמות (כמו "וכשמביתנו")
# - **עמימות מורפולוגית קיצונית**: מילה אחת עם פירושים מרובים לחלוטין ("הקפה")
# - **מורפולוגיה לא-רציפה**: שורש תלת-עיצורי + מישקל
# - **קליטיקים מורכבים**: צירוף מילות יחס מוצמדות (ו+כש+מ+ה)
#
# #### מערכות דקדוקיות מתקדמות:
# - **מערכת בניינים**: 7 בניינים שמשנים פעיל↔פסיבי↔רפלקסיבי↔קאוזטיבי
# - **מערכת מגדר מורכבת**: מגדר שמשפיע על כל חלקי המשפט + מגדר טבעי vs. דקדוקי
# - **מצב סמיכות (smixut)**: מבנה "X של Y" עם חוקי יידוע מורכבים
# - **כינויי קניין מוצמדים**: סיומות שמציינות בעלות רב-ממדית
#
# #### מערכות זמן ויחס ייחודיות:
# - **זמנים עבריים מיוחדים**: עבר/הווה/עתיד/ציווי עם השפעות על צורת הפועל
# - **מערכת גופים תלת-ממדית**: גוף (1/2/3) × מגדר (זכר/נקבה) × מספר (יחיד/רבים)
# - **ה"א הידיעה**: המשפיעה על רמת הספציפיות והיחסים התחביריים
#
# #### הקשרים תרבותיים-לשוניים:
# - **רמות לשון מעורבות**: עברית מקראית/משנית/מודרנית במשפט אחד
# - **דו-משמעויות תרבותיות**: מילים עם משמעות דתית/חילונית שונה
# - **השפעות ארמיות ואחרות**: מבנים שנכנסו מהארמית והשפיעו על המורפולוגיה
#
# #### תכונות ייחודיות נוספות:
# - **צירופים מורפולוגיים חדשים**: שילובי מורפמות שלא נראו אבל תקפים
# - **פילוח vs. מיזוג מורפולוגי**: אתגר בפירוק למורפמות ובבניה מחדש
# - **מילות יחס מוצמדות מורכבות**: (של+מ+ה, כש+מ+ה)
# - **דרגות השוואה ייחודיות**: מבנים עבריים לביטוי עוצמה ושוואה
#
# #### שגיאות מגדר דקדוקי עם השפעה עובדתית:
# - בלבול בין גוף מוסדי (עמותה) למקום פיזי שמשנה איפה מתרחשת פעולה
# - החלפת מגדר שמשנה זיהוי אנשים או ישויות בפועל
#
# #### מורכבות מורפולוגית אקסטרמית:
# - **מילים רב-מורפמיות**: מילה אחת = 7+ מורפמות (דוגמה: "וכשמביתנו" = ו+כש+מ+ה+בית+של+אנחנו)
# - **עמימות מורפולוגית קיצונית**: מילה אחת עם פירושים שונים לחלוטין ("הקפה" = ה+קפה / הקפה-מסלול / הק+פה+של+היא)
#
# ### שגיאות העברת הקשרים תרבותיים-מגדריים (קריטי!):
# - **התנגשות מגדר שמות זרים**: שם שנתפס כנשי/גברי בתרבות אחת אבל האדם הוא מהמגדר ההפוך במציאות
# - **דוגמה קלאסית**: "ננסי" - שם נשי באנגלית, אבל במקור מתייחסים אליו כזכר → בסיכום משנים לנקבה
# - **העברת סטריוטיפים תרבותיים**: תפקידים או שמות שמקושרים למגדר מסוים בתרבות זרה
# - **שגיאת "תיקון" מגדר**: המודל "מתקן" את המגדר לפי הציפיות התרבותיות במקום לפי העובדות
#
#
#
#
# עוד דוגמאות
# החלפת כינויי קניין חבורים (סיומות ו/ה/ם/ן) כך שיתייחסו לאדם או דבר אחר במשפט תוך יצירת שגיאה עובדתית ברורה.
# שגיאה שנובעת מהחלפה של אחד משני רכיבי צירוף סמיכות (שמות עצם צמודים), באופן שמשנה את המשמעות העובדתית של הצירוף, ומוביל להבנה שונה של הישות או המידע המרכזי בטענה.
# שינוי שגוי של זמן פועל (עבר ↔ הווה ↔ עתיד), הגורם לשיבוש בזמני הפעולה במשפט ויוצר טענה שונה מבחינה כרונולוגית או סתירה עם תיאורי זמן קיימים
# שינוי רמת הספציפיות של מידע בטקסט – בין אם באמצעות ה״א הידיעה או על ידי החלפת ישות מזוהה בביטוי כללי – כך שהטענה הופכת לעובדתית שונה.
# ניצול מילים שכתיבתן זהה לזכר ונקבה כדי ליצור טעות בזיהוי מגדר הדמות, מה שמוביל לטעות עובדתית על זהות האדם
# "שגיאה עובדתית הנובעת מהחלפת גוף הפועל בין גוף ראשון לגוף שלישי, באופן שמטשטש או משנה את זהות מי שמבצע את הפעולה.
# החלפת בניין הפועל באופן המהפך מי עשה מה למי - הופך עושה לסובל, קונה למוכר, תובע לנתבע
# שיבוש במשפטים עם פועל משועבד שגורם להיפוך בסוכנות או בכיוון הפעולה – יוצר בלבול בזהות הפועל לעומת מושא הפעולה.
# שיבוש בזהות הגורם המבצע והגורם המקבל בפעולה כיוונית, שנוצר על ידי החלפת מילת יחס או שינוי סדר הגורמים במשפט
# ### שגיאות החלפת תפקידים מורפולוגיים:
# איך שינוי בבניין/גוף/מגדר משנה מי עושה מה למי
#
#  שגיאות פירוק/מיזוג מורפולוגי:
# איך פירוק שגוי של מילה רב-מורפמית יוצר עובדה שגויה
#
#  שגיאות עמימות מורפולוגית:
# איך בחירה שגויה בפירוש מילה רב-משמעית משנה עובדות
#
#  שגיאות הקשר תרבותי-לשוני:
# איך רמות לשון או הקשרים תרבותיים יוצרים עיוותים עובדתיים
#
#  שגיאות זמניות ויחסיות מורכבות:
# איך המערכות הזמניות והיחסיות הייחודיות לעברית יוצרות בלבול
#
# ### דפוסי שגיאות נוספים לחיפוש:
# - **שגיאות בבניינים**: החלפת בניין שמשנה פעיל/פסיבי/רפלקסיבי
# - **שגיאות בזמנים מורכבים**: טעויות בצורות זמן ייחודיות לעברית
# - **שגיאות בקליטיקים**: טעויות במילות יחס מוצמדות מורכבות
# - **שגיאות בכינויי קניין**: החלפת כינויים שמשנה בעלות
#
# ## כללי זיהוי קריטיים:
# - התמקד **אך ורק** בשגיאות שמיוחדות לעברית
# - התעלם משגיאות כלליות: מספרים שגויים, תאריכים שגויים, שמות פרטיים שגויים
# - הכרחי: הסבר למה השגיאה לא יכולה לקרות בשפות כמו אנגלית
# - חפש דוגמאות שבהן המבנה המורפולוגי הוא הגורם הישיר והיחיד לשגיאה
#
# היה חוקר סקרן וגלה דוגמאות מקוריות שטרם תוארו!
#
# חשוב!!!
# חשוב מאוד! השגיאות העובדתיות צריכות להיות שגיאות עובדתיות מובהקות, ברורות וחד משמעיות שמשנות לגמרי את העובדה ויוצרת אי עקביות\ממש שקר בין המאמר לסיכום
# לא מעניינות שגיאות דקדוק, סגנון או קישוריות שלא משפיעות על התוכן העובדתי!
# חשוב!!!
#
# לפני שתדווח על שגיאה - שאל את עצמך:
# 1. האם זה משנה משהו מהותי בעובדות?
# 2. האם קורא יבין משהו שונה לגמרי מהמציאות?
# 3. האם זה יכול להוביל להחלטה שגויה בהתבסס על מידע שגוי?
#
# אם התשובה לא, אל תדווח על השגיאה.
# דוגמה לשגיאה טובה: "שר החוץ אמר" במקום "השרה אמרה" - משנה זהות
# דוגמה לשגיאה רעה: "הכתבה דנה" במקום "המאמר דן" - לא משנה תוכן עובדתי
# """
#
#
#
#     user_message = f"""
# מאמר מקורי:
# {source_article}
#
# סיכום לבדיקה:
# {claims_decomposition_list}
#
# השוה בין המאמר המקורי לסיכום וזהה שגיאות דיוק עובדתי ייחודיות הנובעות מהמורכבות המורפולוגית של העברית.
#
# חשוב במיוחד:
# - חפש שגיאות שטרם תוארו או נתקלת בהן
# - היה יצירתי בזיהוי דפוסים חדשים
# - התמקד במקרים שבהן העברית מאפשרת עיוותים שלא קיימים בשפות אחרות
# - שים לב למילים רב-מורפמיות ולעמימויות מורפולוגיות קיצוניות
#
# לכל שגיאה שתמצא:
# - תן לה שם מקורי ומדויק בעברית
# - הסבר בפירוט איך המורפולוגיה העברית יצרה את השגיאה העובדתית
# - פרט את התכונות הלשוניות הייחודיות שמעורבות (בניינים, גופים, מגדר, סמיכות, קליטיקים וכו')
# - הבא ראיות מדויקות מהטקסטים
# - **הכרחי**: הסבר למה זו שגיאה שלא יכולה לקרות בשפות כמו אנגלית
#
# אם אין שגיאות מיוחדות לעברית - החזר רשימה ריקה.
#
# התגובה צריכה להיות ב-JSON בפורמט המדויק הזה:
# {output_instructions_parser.get_format_instructions()}
# """
#
#     return system_message, user_message



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
        - Ensure each corrupted claim remains **grammatically correct**, **natural**, and **fluent in Hebrew** and do **not** insert gibberish or ungrammatical constructions

    **IMPORTANT**: Consult the section of **Apply Error Transformation Rules Examples** to understand better how perform the hebrew error transformation types

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