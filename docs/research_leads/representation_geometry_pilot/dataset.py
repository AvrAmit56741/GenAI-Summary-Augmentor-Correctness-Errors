# Gold pilot dataset for the representation-geometry experiment.
# Each example = (faithful_text, erroneous_text) minimal-ish pair + taxonomy error_type +
# model that produced the error + a short provenance note (doc_id / source doc in the project).
# Sources: paper/figures_src/fig6_model_error_catalog.tex (fully-verified full-sentence examples),
# paper/results/eval/new50_confirmed_findings.md (3 confirmed, high-confidence, directly-quoted),
# paper/results/eval/findings_master_list_72.md (additional confirmed rows with directly-quoted
# fragments; faithful counterpart reconstructed via minimal gender-flip where the source article's
# exact full faithful sentence wasn't itself quoted in the markdown -- noted per-row below),
# thesisdatasets/error_finder/*.json (450 raw LLM-judge rows; `correctness_error_name` is free-text
# Hebrew and every row's text is unique, i.e. NOT pre-mapped to this project's canonical taxonomy --
# each row below was manually read and hand-mapped to the nearest taxonomy error_type after keyword
# triage; `evidence_from_source_article`/`evidence_from_summary` are used directly as faithful/
# erroneous, no reconstruction needed since those fields are already an aligned evidence pair).

EXAMPLES = [
    # --- from fig6 (10 full-sentence, independently verified pairs) ---
    dict(id="dicta2.0_uzi", model="DictaLM-2.0-Instruct", error_type="noun_gender_swap",
         faithful="...בן 6 ומאולף. בחג הזה הוא רוצה לזכות באימוץ מרגש.",
         erroneous="עוזי, כלבה יפהפייה, מחפשת בית שיעניק לה אהבה."),
    dict(id="gpt35_shelling", model="GPT-3.5-Turbo", error_type="hebrew_root_pattern_confusion",
         faithful="ישראל התחילה בהרעשה רגע אחרי הפסקת האש, כשהעבירה מסר מאיים כפול לחמאס:",
         erroneous="ישראל מתרגשת להביע את נצחונה, אך חמאס ממשיך להיות מאיים."),
    dict(id="gemma3_minister", model="Gemma3-12B", error_type="noun_gender_swap",
         faithful="...שר התקשורת איוב קרא הפציר ש\"צריך לחשוב פעמים למה זה נחוץ\", אבל תסמכו על השרה רגב להנפיק את המשפט...",
         erroneous="...כאשר שרת התקשורת אף הצהיר על הצורך בשליטה בתאגיד."),
    dict(id="gptoss_wing", model="gpt-oss-20B", error_type="noun_gender_swap",
         faithful="לימים הגישה בעלת הזכויות באגף הדרומי בקשה למתן היתר בנייה, לצורך הריסת האגף שבבעלותה...",
         erroneous="בעל האגף הדרומי ביקש היתר בנייה להריסת האגף שלו ולבנות בניין חדש תחתיו..."),
    dict(id="qwen14b_singers", model="Qwen3-14B", error_type="verb_template_agency_reversal",
         faithful="פרגנתי לכל הזמרים שהיו שם, השמעתי להם את השירים שלהם.",
         erroneous="...שפילמן פרגן עם הזמרים, שמע את השירים שלהם..."),
    dict(id="dicta3.0_gvarin", model="DictaLM-3.0-1.7B-Instruct", error_type="morphological_connective_confusion",
         faithful="בעקבות פסק דין ג'בארין זוכו ארבעה יהודים מהימין שהואשמו בהסתה ועניינם היה תלוי ועומד לפני בית המשפט העליון...",
         erroneous="בעקבות פסק דין ג'בארין, זוכו ארבעה יהודים מהימין והואשמו בהסתה, ועניינם היה תלוי ועומד לפני בית המשפט העליון."),
    dict(id="glm_shoham_fig6", model="GLM-4.7-Flash", error_type="homographic_gender_errors",
         faithful="אות בבאר שהם סמיט, איורים: ולי מינצי (כנרת)",
         erroneous="...כגון \"אות בבאר\" של הם סמיט, שמספר את סיפורו של רבי עקיבא..."),
    dict(id="ministral_yehudit", model="Ministral-3-14B", error_type="construct_state_confusion",
         faithful="\"מסכנה אמא\", אומרת לי יהודית דקל, הצעירה מבין שני ילדיהם...",
         erroneous="...כשבנה, יהודית, משמרת אותם עד היום."),
    dict(id="qwen32b_goldstein_fig6", model="Qwen3-32B", error_type="verb_gender_swap",
         faithful="ישראל גולדשטיין הפיל לבדו, כשהוא חשוף להפגזות, ארבעה מטוסי קרב סוריים",
         erroneous="גולדשטיין, מפקד פלגה בנ\"מ, הפעילה לבדה את העמדה תחת אש קלה והפיל ארבעה מטוסי קרב סוריים..."),
    dict(id="nova_shoshana", model="Amazon Nova Lite", error_type="verb_gender_swap",
         faithful="שושנה דמארי (31 במרץ 1923 -- 14 בפברואר 2006) הייתה זמרת ושחקנית ישראלית - ילידת תימן...",
         erroneous="שושנה דמארי, שנולדה בתימן ועלה לישראל בגיל צעיר, הייתה זמרת ושחקנית מפורסמת..."),

    # --- from new50_confirmed_findings.md (3 confirmed, directly quoted) ---
    dict(id="glm_moamedet_new50", model="GLM-4.7-Flash", error_type="homographic_gender_errors",
         faithful="...בשל כך נפסלה מועמדתו של בן העדה הדרוזית.",
         erroneous="...נפסלה מועמדת דרוזית עקב טעות בייעוד המכרז לערבים במקום לדרוזים."),
    dict(id="qwen32b_sane_new50", model="Qwen3-32B", error_type="verb_gender_swap",
         faithful="אני מתכבד להביא לאישור הכנסת... אני מוכן לגשת מייד להצבעה.",
         erroneous="סגן שר הביטחון, א' סנה, הציג את ההצעה, והציעה להחיל את התיקון..."),
    dict(id="qwen32b_ginzburg_new50", model="Qwen3-32B", error_type="verb_gender_swap",
         faithful="איתן גינזבורג דיבר על חשיבות העברת סמכויות לרשויות מקומיות.",
         erroneous="איתן גינזבורג דיברה על חשיבות העברת סמכויות לרשויות מקומיות."),

    # --- from findings_master_list_72.md (confirmed rows, quoted fragment; faithful counterpart
    #     reconstructed via minimal gender-flip of the quoted error since the full source sentence
    #     wasn't itself quoted verbatim in the markdown -- flagged reconstructed=True) ---
    dict(id="qwen32b_popov_master14", model="Qwen3-32B", error_type="verb_gender_swap",
         faithful="וובה פופוב... נולד במקור באי קרים.",
         erroneous="וובה פופוב... נולדה במקור באי קרים.", reconstructed=True),
    dict(id="glm_ash_master44", model="Qwen3-32B", error_type="verb_gender_swap",
         faithful="גם פרופ' אש דרש לחגוג את חג פורים בהדרגה.",
         erroneous="גם פרופ' אש דרשה לחגוג את חג פורים בהדרגה.", reconstructed=True),
    dict(id="glm_author_master27", model="GLM-4.7-Flash", error_type="noun_gender_swap",
         faithful="המחברת החליטה.",
         erroneous="המחבר החליטה.", reconstructed=True),
    dict(id="ministral_host_master32", model="Ministral-3-14B", error_type="noun_gender_swap",
         faithful="המארחת, המתארת... המארחת יכולה להרשים את אורחיה.",
         erroneous="המארחת, המתארת... המארח יכול להרשים את אורחיו.", reconstructed=True),

    # --- from thesisdatasets/error_finder/*.json (10 new pairs, hand-mapped to the taxonomy;
    #     faithful/erroneous taken directly and verbatim from evidence_from_source_article /
    #     evidence_from_summary, no reconstruction) ---
    dict(id="dicta2.0_watson_finder", model="DictaLM-2.0-Instruct", error_type="verb_gender_swap",
         faithful="מאז שהוּדָה ווטסון בפשע הנתעב שביצע, קסינג'ר מחקה את כל החשבונות החברתיים שלה...",
         erroneous="למרות שהיא הודתה בחקירותיה שהיא לא אהבה את ווטס, היא נמנעה מלהזכיר את שמות אשתו ובנותיו..."),
    dict(id="qwen14b_juan_finder", model="Qwen3-14B", error_type="verb_gender_swap",
         faithful="חואן - כלב מקסים, עדין, טוב לב -יליד מאי 2019. חואן היפה כלב מתמסר ואוהב...",
         erroneous="חואן, כלב מקסים שנולד במאי 2019, מתאפיין בדימוי נבון, רגיש ומעורר שמחה, ומחפשת בית שיבטח לו."),
    dict(id="qwen14b_chupi_finder", model="Qwen3-14B", error_type="verb_gender_swap",
         faithful="צ'ופי - גור בן כ-4 חודשים שנמצא משוטט לבדו כשהיה גור קטן והגיע להסגר עירוני.",
         erroneous="צ'ופי, גור בן ארבע החודשים, נמצאה משוטטת לבדה, ומחפשת משפחה שיתן לה אהבה."),
    dict(id="qwen14b_hason_finder", model="Qwen3-14B", error_type="verb_gender_swap",
         faithful="חסון קטן ואהוב בן 6, כלבלב שכל כך מכור לאהבה",
         erroneous="חסון, כלבלב בן 6 שמחפשת אהבה"),
    dict(id="gptoss_chef_finder", model="gpt-oss-20B", error_type="noun_gender_swap",
         faithful="הראשונה להתארח היא מסעדת לונא הנצרתית. השפית לונא זרייק תגיע לערב חגיגי...",
         erroneous="הערב הראשון, ב-9.8, מתארח שף לונא זרייק ממסעדת לונא הנצרתית, ומביא את המטבח הנצרתי שלה לתל אביב..."),
    dict(id="gptoss_husband_finder", model="gpt-oss-20B", error_type="noun_gender_swap",
         faithful="...שם גם הכירה את בעלה גיארג מתיאש לוי ושם גם ניתנו לשניהם שמות עבריים.",
         erroneous="היא פגשה את גיארג מתיאש לוי, שהפך לבעלתה, והשתתפה בתנועת הנוער הציוני..."),
    dict(id="qwen14b_shoham_finder", model="Qwen3-14B", error_type="homographic_gender_errors",
         faithful="\"אות בבאר\" מאת שהם סמיט",
         erroneous="\"אות בבאר\" של שה סמיט"),
    dict(id="qwen14b_pribar_finder", model="Qwen3-14B", error_type="construct_state_confusion",
         faithful="את רשימת המועמדויות מוביל הסרט 'אסיה' (סרטה של רותי פרי בר)",
         erroneous="הסרט 'אסיה', בבימויו של רותי פרי-בר"),
    dict(id="gpt35_gibush_finder", model="GPT-3.5-Turbo", error_type="hebrew_root_pattern_confusion",
         faithful="ראש הממשלה... קבע דיון מחר עם השרים הנ\"ל על מנת לגבש תוכנית ממשלתית אחידה.",
         erroneous="ראש הממשלה... קבע דיון מחר על גיוס תוכנית ממשלתית אחידה."),
    dict(id="gpt35_schechter_finder", model="GPT-3.5-Turbo", error_type="verb_template_agency_reversal",
         faithful="בתקרית המדוברת, שכטר חשף את ישבנו בפני קהל של 20,000 בני נוער בהופעה.",
         erroneous="עפר שכטר נחשף בפרובוקציה כבר בתחילת קריירתו ונחקר במשטרה."),
]

if __name__ == "__main__":
    from collections import Counter
    c = Counter(e["error_type"] for e in EXAMPLES)
    print(f"{len(EXAMPLES)} examples, {len(c)} error-type families")
    for k, v in c.most_common():
        print(f"  {v:2d}  {k}")
