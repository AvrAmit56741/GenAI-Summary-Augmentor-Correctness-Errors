"""Build a SECOND held-out pool of 50 never-before-used articles, continuing the same
"hardness score" selection methodology used for new50_candidates.csv (see docs/experiments.md
Part D). The first pool exhausted all 21 Knesset + all 4 Bagatz articles, so this second pool
draws only from the remaining Wikipedia/Weizmann articles not already in new50_candidates.csv.

NOTE: the original scoring script that built new50_candidates.csv was not saved as a file (a
recording-policy gap from an earlier session). This is a best-effort reconstruction of the same
keyword-density heuristic (KINSHIP x1, ROLE_GENDER x1.5, PASSIVE-binyan x1.2) -- adequate for a
selection pre-filter, not a measurement, so exact keyword-list parity with the original run is not
required.

    python exp_build_next50_candidates.py

Output: next50_candidates.csv
"""
import os
import re

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
BENCH = os.path.join(REPO, "thesisdatasets/train_datasets/Correctness/data/benchmark/combined.csv")
PREV_CANDIDATES = os.path.join(HERE, "new50_candidates.csv")
OUT = os.path.join(HERE, "next50_candidates.csv")

KINSHIP = ["בן", "בת", "אח", "אחות", "הורה", "הורים", "אב", "אם", "סב", "סבא", "סבתא",
           "דוד", "דודה", "נכד", "נכדה", "בעל", "אישה", "אשתו", "בעלה", "ילד", "ילדה",
           "ילדיה", "ילדיו", "משפחה", "אלמן", "אלמנה", "גרוש", "גרושה"]
ROLE_GENDER = ["מנהל", "מנהלת", "מורה", "רופא", "רופאה", "נשיא", "נשיאה", "ראש", "ראשת",
               "שר", "שרה", "חבר", "חברה", "יושב-ראש", "יושבת-ראש", "מפקד", "מפקדת",
               "מחבר", "מחברת", "סופר", "סופרת", "שחקן", "שחקנית", "זמר", "זמרת",
               "פרופסור", "פרופסורית", "חייל", "חיילת", "שגריר", "שגרירה", "מתמחה",
               "חוקר", "חוקרת", "מהנדס", "מהנדסת", "עיתונאי", "עיתונאית"]
PASSIVE = ["נבנה", "נבנתה", "נכתב", "נכתבה", "נמצא", "נמצאה", "נעשה", "נעשתה", "נבחר",
           "נבחרה", "נולד", "נולדה", "נפתח", "נפתחה", "נסגר", "נסגרה", "הוחלט", "הופעל",
           "הופעלה", "הוקם", "הוקמה", "התקבל", "התקבלה"]


def score(text):
    s = 0.0
    for w in KINSHIP:
        s += len(re.findall(re.escape(w), text)) * 1.0
    for w in ROLE_GENDER:
        s += len(re.findall(re.escape(w), text)) * 1.5
    for w in PASSIVE:
        s += len(re.findall(re.escape(w), text)) * 1.2
    return s


def main():
    bench = pd.read_csv(BENCH)
    bench["premise"] = bench["premise"].astype(str)
    bench_uniq = bench.drop_duplicates(subset=["doc_id"])

    prev = pd.read_csv(PREV_CANDIDATES)
    used_ids = set(prev["doc_id"])

    pool = bench_uniq[bench_uniq["source"].isin(["Wikipedia", "Weizmann"])]
    pool = pool[~pool["doc_id"].isin(used_ids)]
    print(f"remaining never-used pool: {pool['source'].value_counts().to_dict()}")

    pool = pool.copy()
    pool["score"] = pool["premise"].apply(score)
    pool["plen"] = pool["premise"].str.len()

    top = pool.sort_values("score", ascending=False).head(50)
    top[["source", "doc_id", "score", "plen"]].to_csv(OUT, index=False)
    print(f"saved {len(top)} candidates -> {OUT}")
    print(top["source"].value_counts())


if __name__ == "__main__":
    main()
