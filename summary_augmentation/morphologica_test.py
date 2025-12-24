import csv
import os
from collections import defaultdict
from difflib import SequenceMatcher


def create_lexicon_mapping(rows):
    """יוצר מיפוי של מילים לפי שורש ולקסיקון"""
    lexicon_map = defaultdict(list)

    for i, row in enumerate(rows):
        word = row.get('\ufeffword', row.get('word', ''))
        lex = row.get('lex', '')
        unvocalized_lex = row.get('unvocalized_lex', '')

        # נקה קליטיקים (מילות יחס מוצמדות)
        clean_word = word.split('|')[-1] if '|' in word else word

        lexicon_map[unvocalized_lex].append({
            'index': i,
            'word': clean_word,
            'full_word': word,
            'morph': row
        })

    return lexicon_map


def find_semantic_matches(source_map, summary_map, similarity_threshold=0.87):
    """מוצא התאמות סמנטיות בין מילים"""
    matches = []

    # התאמה מדויקת לפי שורש
    for lex in source_map:
        if lex in summary_map and lex.strip():  # רק שורשים לא ריקים
            for source_entry in source_map[lex]:
                for summary_entry in summary_map[lex]:
                    matches.append((source_entry, summary_entry, 'exact_lex'))

    return matches


def calculate_suspicion_level(source_morph, summary_morph):
    """מחשב רמת חשדנות בלי לקבוע שיש שגיאה"""

    reasons = []
    score = 0

    # בדיקה שזה אותו סוג מילה
    if source_morph.get('POS') != summary_morph.get('POS'):
        return {'reasons': [], 'score': 0}  # סוגי מילים שונים - לא רלוונטי

    pos = source_morph.get('POS')

    # חשדנות גבוהה: שינוי מגדר בישויות מוגדרות
    if pos in ['NOUN', 'VERB']:
        source_gender = source_morph.get('Gender')
        summary_gender = summary_morph.get('Gender')

        if (source_gender in ['MASCULINE', 'FEMININE'] and
                summary_gender in ['MASCULINE', 'FEMININE'] and
                source_gender != summary_gender):

            # חשדנות גבוהה יותר לשמות עצם מוגדרים
            if pos == 'NOUN' and source_morph.get('Prefix') == 'DEFINITEARTICLE':
                reasons.append("definite_noun_gender_change")
                score += 4
            else:
                reasons.append("gender_change")
                score += 2

    # חשדנות בינונית: שינוי בניין בפעלים
    if pos == 'VERB':
        source_binyan = source_morph.get('Binyan', '')
        summary_binyan = summary_morph.get('Binyan', '')

        if source_binyan and summary_binyan and source_binyan != summary_binyan:
            source_voice = get_voice_category(source_binyan)
            summary_voice = get_voice_category(summary_binyan)

            if source_voice != summary_voice and source_voice != 'unknown' and summary_voice != 'unknown':
                reasons.append(f"verb_voice_change_{source_voice}_to_{summary_voice}")
                score += 3

    # חשדנות גבוהה: שינוי כינוי קניין
    if both_have_possessive_suffix(source_morph, summary_morph):
        if source_morph.get('Suf_Person') != summary_morph.get('Suf_Person'):
            reasons.append("possessive_person_change")
            score += 4

        if (source_morph.get('Suf_Gender') != summary_morph.get('Suf_Gender') and
                source_morph.get('Suf_Gender') in ['MASCULINE', 'FEMININE'] and
                summary_morph.get('Suf_Gender') in ['MASCULINE', 'FEMININE']):
            reasons.append("possessive_gender_change")
            score += 4

    # חשדנות נמוכה: שינוי גוף בפועל
    if (pos == 'VERB' and
            source_morph.get('Person') and summary_morph.get('Person') and
            source_morph.get('Person') != summary_morph.get('Person')):
        reasons.append("verb_person_change")
        score += 1

    return {'reasons': reasons, 'score': score}


def get_voice_category(binyan):
    """מסווג בניין לקטגוריית קול"""

    active_binyans = ['BINYAN_PAAL', 'BINYAN_PIEL', 'BINYAN_HIFIL']
    passive_binyans = ['BINYAN_NIFAL', 'BINYAN_PUAL', 'BINYAN_HUFAL']
    reflexive_binyans = ['BINYAN_HITPAEL']

    if binyan in active_binyans:
        return 'active'
    elif binyan in passive_binyans:
        return 'passive'
    elif binyan in reflexive_binyans:
        return 'reflexive'
    else:
        return 'unknown'


def both_have_possessive_suffix(morph1, morph2):
    """בודק אם לשתי המילים יש כינוי קניין"""
    return (morph1.get('Suffix') and morph2.get('Suffix') and
            morph1.get('Suf_Person') and morph2.get('Suf_Person'))


def get_word_context(morph_data, word_index, context_size=3):
    """מחזיר הקשר של מילה (מילים לפני ואחרי)"""

    start = max(0, word_index - context_size)
    end = min(len(morph_data), word_index + context_size + 1)

    context_words = []
    for i in range(start, end):
        word = morph_data[i].get('\ufeffword', morph_data[i].get('word', ''))
        # נקה קליטיקים
        clean_word = word.split('|')[-1] if '|' in word else word

        if i == word_index:
            clean_word = f"**{clean_word}**"  # הדגשת המילה הרלוונטית
        context_words.append(clean_word)

    return ' '.join(context_words)


def identify_morphological_hotspots(source_rows, summary_rows):
    """זיהוי נקודות חשודות מורפולוגית"""

    # יצירת מיפויים לקסיקליים
    source_map = create_lexicon_mapping(source_rows)
    summary_map = create_lexicon_mapping(summary_rows)

    # מציאת התאמות סמנטיות
    matches = find_semantic_matches(source_map, summary_map)

    hotspots = []

    for source_entry, summary_entry, match_type in matches:
        suspicion = calculate_suspicion_level(
            source_entry['morph'],
            summary_entry['morph']
        )

        if suspicion['score'] > 0:
            hotspots.append({
                'source_word': source_entry['word'],
                'summary_word': summary_entry['word'],
                'source_full': source_entry['full_word'],
                'summary_full': summary_entry['full_word'],
                'source_context': get_word_context(source_rows, source_entry['index']),
                'summary_context': get_word_context(summary_rows, summary_entry['index']),
                'suspicion_reasons': suspicion['reasons'],
                'suspicion_score': suspicion['score'],
                'match_type': match_type,
                'source_morph': source_entry['morph'],
                'summary_morph': summary_entry['morph']
            })

    return sorted(hotspots, key=lambda x: x['suspicion_score'], reverse=True)


def build_morphological_hints(hotspots):
    """בונה רמזים מורפולוגיים עבור המודל"""

    if not hotspots:
        return "לא נמצאו נקודות חשודות מורפולוגית."

    hints = []

    for i, hotspot in enumerate(hotspots[:5]):  # רק 5 החשודים ביותר
        reasons_hebrew = translate_reasons_to_hebrew(hotspot['suspicion_reasons'])

        hint = f"""
נקודה חשודה {i + 1} (ציון חשדנות: {hotspot['suspicion_score']}/5):
- במקור: "{hotspot['source_word']}" 
  הקשר: "{hotspot['source_context']}"
- בסיכום: "{hotspot['summary_word']}"
  הקשר: "{hotspot['summary_context']}"
- סיבות לחשדנות: {', '.join(reasons_hebrew)}

בדוק: האם השינוי הזה יוצר שגיאה עובדתית אמיתית?
"""
        hints.append(hint)

    return '\n'.join(hints)


def translate_reasons_to_hebrew(reasons):
    """תרגום סיבות חשדנות לעברית"""

    translations = {
        'definite_noun_gender_change': 'שינוי מגדר בשם עצם מוגדר',
        'gender_change': 'שינוי מגדר',
        'verb_voice_change_active_to_passive': 'שינוי מפעיל לסביל',
        'verb_voice_change_passive_to_active': 'שינוי מסביל לפעיל',
        'possessive_person_change': 'שינוי גוף בכינוי קניין',
        'possessive_gender_change': 'שינוי מגדר בכינוי קניין',
        'verb_person_change': 'שינוי גוף בפועל'
    }

    return [translations.get(reason, reason) for reason in reasons]


def enhanced_factual_error_detection_with_morphology(source_article, summary, hotspots):
    """בניית פרומפט מועשר עם מידע מורפולוגי"""

    morphological_hints = build_morphological_hints(hotspots)

    system_message = f"""
אתה מומחה מחקר בבלשנות חישובית עברית ועיבוד שפה טבעית.

המערכת זיהתה נקודות חשודות מורפולוגית שעלולות להסתיר שגיאות עובדתיות מיוחדות לעברית.

**נקודות חשודות מורפולוגית שנמצאו:**
{morphological_hints}

**המשימה שלך:** לבדוק האם הנקודות החשודות האלה מסתירות שגיאות עובדתיות אמיתיות.

**דגש על שגיאות עובדתיות בלבד:**
✅ שינוי זהות אדם (גבר→אישה או להפך)
✅ שינוי מי עושה מה למי (נותן→מקבל, קונה→מוכר)
✅ שינוי בעלות (שלו→שלה→שלהם)
✅ שינוי כיוון פעולה או סוכנות

❌ **לא מעניין:**
- שינויי ניסוח שלא משנים עובדות
- הבדלי זמן שלא משמעותיים עובדתית
- שגיאות דקדוק טכניות
- שינויים שלא משפיעים על הבנת התוכן

**חשוב:** התמקד רק בנקודות החשודות שצוינו למעלה ובדוק האם הן יוצרות שגיאות עובדתיות אמיתיות.
"""

    user_message = f"""
מאמר מקורי:
{source_article}

סיכום לבדיקה:
{summary}

בדוק בקפידה את הנקודות החשודות מורפולוגית שזוהו.
לכל נקודה חשודה, קבע האם היא יוצרת שגיאה עובדתית אמיתית או שזה רק שינוי לשוני שלא משפיע על העובדות.

דווח רק על שגיאות עובדתיות אמיתיות ובטוחות - כאלה שקורא יבין מהן משהו שונה לגמרי מהמציאות.
"""

    return system_message, user_message


def analyze_morphological_hotspots():
    """פונקציה ראשית מעודכנת"""
    source_path = r"C:\Users\amita\Downloads\MorphologyResults_sum.tsv"
    summary_path = r"C:\Users\amita\Downloads\MorphologyResults_sou.tsv"

    if not os.path.exists(source_path) or not os.path.exists(summary_path):
        print("One or both input files not found.")
        return

    source_rows = read_tsv_rows(source_path)
    summary_rows = read_tsv_rows(summary_path)

    print(f"Source: {len(source_rows)} words, Summary: {len(summary_rows)} words")

    # זיהוי נקודות חשודות
    hotspots = identify_morphological_hotspots(source_rows, summary_rows)

    print(f"\nFound {len(hotspots)} morphological hotspots")

    if hotspots:
        print("\nTop suspicious morphological changes:")
        for i, hotspot in enumerate(hotspots[:5]):
            print(f"{i + 1}. '{hotspot['source_word']}' → '{hotspot['summary_word']}' "
                  f"(score: {hotspot['suspicion_score']}, reasons: {hotspot['suspicion_reasons']})")

        # הכנת מידע לפרומפט
        hints = build_morphological_hints(hotspots)
        print(f"\n--- Morphological hints for LLM ---")
        print(hints)

        return hotspots
    else:
        print("No morphological hotspots detected.")
        return []


def read_tsv_rows(path):
    with open(path, encoding='utf-8-sig') as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        return list(reader)


if __name__ == "__main__":
    hotspots = analyze_morphological_hotspots()

    # לדוגמה - איך להשתמש עם מודל שפה
    if hotspots:
        print("\n--- Ready for LLM integration ---")
        print("You can now use these hotspots with enhanced_factual_error_detection_with_morphology()")