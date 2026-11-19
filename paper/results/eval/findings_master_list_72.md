# Master list of 72 confirmed findings (new-model only, excludes old-model calibration anchors)

Compiled during the co-author-objection follow-up (see `docs/experiments.md` §19). Excludes the 3 old-model
calibration-anchor findings (business-6667074, culture-4533684, food-917568) used to calibrate the judge in §18 —
those are historical/old-model and not part of this "do current models still err" count.

For each row: doc_id | model | prompt-style | file (relative to paper/results/eval/, except DictaLM which is
elsewhere) | severity as originally judged.

**Verification status (updated 2026-11-17 — full pass complete):** every finding with a known doc_id (55 of
72) has now gone through a full BLIND adversarial re-verification: each premise/summary pair was read fresh,
independent of the original claim, and judged against the mandatory gate (בניין/גוף-מגדר-מספר/סמיכות/ה"א
הידיעה/כינוי קניין מוצמד/קליטיק) + the English-contrast filter + the "must positively assert something false,
not just omit or slip" rule. **Result: of the 55 checked, only 16 reproduced as genuine findings (29%); 39
did not reproduce on a blind re-read and are REJECTED.** The remaining 17 (rows with "doc_id not recorded at
compile time") could not be checked at all and remain of unknown status — neither confirmed nor rejected.

This means the honest, re-verified count of genuine new-model Hebrew-morphology errors found by this
project's error-finder process this session is **16 (from the 55 checkable) + possibly some fraction of the
17 unrecoverable ones**, not the originally-reported 72. See the inline ✅/❌ verdicts below, and
`docs/experiments.md` §19 Part C for the full writeup and discussion of what this means for the paper.

**Caveat on this re-verification itself:** this was one blind read per pair, not an exhaustive line-by-line
parse of every long-form summary (several ran 5,000–9,000 characters). A ❌ below means "I did not locate a
qualifying error on this pass," not "a rigorous proof that none exists" — some false negatives are likely,
especially in the longest long-form summaries. Two NEW findings not in the original claims were also
discovered during this pass (marked 🆕 below) — evidence that blind re-reading surfaces different things than
the original targeted search, in both directions.

## BATCH A — Thin prompt (19 findings)
1. 218050 | DictaLM 3.0 | thin | current_model_summaries_dictalm3.csv or _round2.csv | 2 — ✅ CONFIRMED (verified in §18 itself: ש-/ו- clitic swap, occurs in the fully-generated part of the summary)
2. food-14044992 | GLM-4.7-Flash | thin | current_model_summaries_zai_glm-4_7-flash_targeted*.csv (any offset) | 3 — ✅ CONFIRMED (verified in the earlier GLM-thin slice re-check)
3. animals-6829845 | GLM-4.7-Flash | thin | current_model_summaries_zai_glm-4_7-flash_targeted*.csv | 2 — ✅ CONFIRMED (verified in the earlier GLM-thin slice re-check)
4. culture-10230151 | GLM-4.7-Flash | thin | current_model_summaries_zai_glm-4_7-flash_targeted*.csv | 2 — ❌ REJECTED, did not reproduce
5. culture-10230151 | Ministral-3-14B | thin | current_model_summaries_mistral_ministral-3-14b-instruct_targeted*.csv | 2 — ❌ REJECTED (2026-11-17 blind re-check): long detailed summary (Druze names/army slang/legal jargon), no gender/binyan/smixut break located
6. general-14268637 | GLM-4.7-Flash | thin | current_model_summaries_zai_glm-4_7-flash_targeted*.csv | 3-4 — ✅ CONFIRMED (reinstated after a user-caught error in where I first checked)
7. general-14268637 | Ministral-3-14B | thin | current_model_summaries_mistral_ministral-3-14b-instruct_targeted*.csv | 3-4 — ❌ REJECTED (2026-11-17): long Holocaust-survivor narrative, all gender agreement consistent throughout (ברכה/יהודית fem, אריה masc), no flip located
8. culture-7822389 | Ministral-3-14B | thin | current_model_summaries_mistral_ministral-3-14b-instruct_targeted*.csv | 1 — ❌ REJECTED (2026-11-17): Meir Ariel tribute piece, all names/genders consistent
9. culture-10173038 | Ministral-3-14B | thin | current_model_summaries_mistral_ministral-3-14b-instruct_targeted*.csv | 1 — ❌ REJECTED (2026-11-17): "Bnot HaTom" piece, all three singers consistently fem, no error located
10. general-13160920 | GPT-5.6-Luna | thin | current_model_summaries_openai_gpt-5_6-luna_targeted*.csv | 2 — ❌ REJECTED (2026-11-17): poetry-anthology summary, no gender/binyan break located
11. food-333642 | GLM-4.7-Flash | thin | current_model_summaries_zai_glm-4_7-flash_targeted_offset100.csv | 4 — ✅ CONFIRMED (verified in the earlier GLM-thin slice re-check)
12. culture-13795813 | GLM-4.7-Flash | thin | current_model_summaries_zai_glm-4_7-flash_targeted*.csv | 3 — ❌ REJECTED, did not reproduce
13. culture-14100072 | GLM-4.7-Flash | thin | current_model_summaries_zai_glm-4_7-flash_targeted*.csv | 4 — ✅ CONFIRMED (verified in the earlier GLM-thin slice re-check)
14. celebrities-8932596 | Qwen3-32B | thin | current_model_summaries_qwen_qwen3-32b-v1_0_targeted.csv | 2 (finding A: 1st-person->3rd fem) — ✅ CONFIRMED (2026-11-17): "וובה פופוב... **נולדה** במקור באי קרים" — fem verb for a consistently-male subject (יליד, הוא elsewhere); real gender-identity assertion, unreproducible in English
15. celebrities-8932596 | Qwen3-32B | thin | current_model_summaries_qwen_qwen3-32b-v1_0_targeted.csv | 2 (finding B: reflexive clitic) — ✅ CONFIRMED, moderate confidence (2026-11-17): "התחלת **חתיכת עצמה**" — fem reflexive suffix for male Vova; moderate confidence because "עצמה" is also a legitimate common noun ("intensity") on a different parse
16. magazine-13224346 | Qwen3-32B | thin | current_model_summaries_qwen_qwen3-32b-v1_0_targeted.csv | 2 (finding A: possessive clitic) — ❌ REJECTED (2026-11-17): only located "ביןו" (garbled "בינו"), reads as a spelling/decoding glitch rather than a factual-assertion error
17. magazine-13224346 | Qwen3-32B | thin | current_model_summaries_qwen_qwen3-32b-v1_0_targeted.csv | 3 (finding B: root-pattern collision) — ❌ REJECTED (2026-11-17): could not locate a second distinct qualifying error in this summary
18. culture-10266649 | GLM-4.7-Flash | thin | current_model_summaries_zai_glm-4_7-flash_targeted*.csv | 2-3 — ❌ REJECTED, did not reproduce
19. culture-10266649 | Ministral-3-14B | thin | current_model_summaries_mistral_ministral-3-14b-instruct_targeted*.csv | 2-3 — ❌ REJECTED (2026-11-17): עילגות-dictionary article, no human-identity-bearing gender claims to flip

## BATCH B — Long-form prompt (40 findings), file pattern: current_model_summaries_<tag>_longform_val20[_offsetN].csv
GLM-4.7-Flash (tag: zai_glm-4_7-flash), 9 total:
20. (val20, no offset) — 1 finding, doc_id not recorded at compile time, severity ~1-2 — ⬜ UNVERIFIABLE (doc_id unknown)
21. culture-8566615 | offset50 | 3 — ❌ REJECTED (2026-11-17): candidate found ("לירן... שעברה טראומות", fem, vs. premise's "לירן הוא גבר") but the sentence is inside a list of Big Brother *contestants*, where a second, differently-spelled "לירון ויצמן" (plausibly female) is introduced separately from the male columnist "לירן" — referent too ambiguous to confirm as an error
22. general-13160920 | offset50 | 3 — ❌ REJECTED (2026-11-17): poetry-anthology summary, no gender/binyan break located
23. business-6667074 | offset50 | 2 (GLM's own long-form summary partially reproduces the historical gender-flip pattern) — ✅ CONFIRMED, moderate confidence (2026-11-17): "בעלי הזכויות" (plural/neutral) throughout, then singular fem "**בעלת** הזכויות... **רשאית**" at the very end — matches the known calibration-anchor defect pattern on this exact article; moderate confidence only because the true gender of the real legal party wasn't independently re-verified from source
24. culture-7822389 | offset50 | 2 — ❌ REJECTED (2026-11-17): Meir Ariel tribute, long-form version, all consistent
25. judaism-7489161 | offset50 | 2 — ❌ REJECTED (2026-11-17): US Jewish vote article, no gendered individual claims at risk
26. culture-10173038 | offset50 | 2 (low confidence, flagged as such originally) — ❌ REJECTED (2026-11-17)
27. business-13302729 | offset50 | 1 (low confidence, flagged as such originally) — ✅ CONFIRMED (2026-11-17): "**המחבר** החליטה" — masc common noun ("the [male] author") + fem verb, for a narrator established female via "אני מבשלת"; weak-but-real, matches original low-confidence/severity-1 flag
28. culture-14100072 | offset100 | 4 — ✅ CONFIRMED (2026-11-17): "אות בבאר' של **הם סמיט**" — drops the relative clitic ש from the real author's name שהם סמיט (feminine author "שהם"), misparsing it as the pronoun "הם" (they/masc-plural); a validated, replicated GLM-specific trap on this exact article (also reproduces in the restricted-prompt version, #61 below)

Ministral-3-14B (tag: mistral_ministral-3-14b-instruct), 5 total:
29. (offset20) — 1 finding, doc_id not recorded at compile time — ⬜ UNVERIFIABLE (doc_id unknown)
30. culture-10230151 | offset50 | 2 — ❌ REJECTED (2026-11-17): Druze-names/army-slang/legal-jargon piece, long-form version, no error located
31. general-14268637 | offset50 | 2 (low confidence, flagged as such originally) — ❌ REJECTED (2026-11-17): long Holocaust narrative, long-form version, all gender-consistent
32. business-13302729 | offset50 | 1 ("המארח" self-contradiction) — ✅ CONFIRMED, flagged weak on the English-contrast test (2026-11-17): "**המארחת**, המתארת..." (fem, early) vs. "**המארח** יכול להרשים את אורחיו" (masc, at the end) — real internal self-contradiction, but English speakers also inconsistently drop "hostess" for "host," so this is a weaker English-contrast case than most confirmed findings
33. culture-10266649 | offset100 | 2 — ❌ REJECTED (2026-11-17): could not locate the claimed error in the long-form version

Qwen3-32B (tag: qwen_qwen3-32b-v1_0), 11 total:
34-35. (val20) — 2 findings, doc_ids not recorded at compile time — ⬜ UNVERIFIABLE (doc_ids unknown)
36-38. (offset20) — 3 findings, doc_ids not recorded at compile time — ⬜ UNVERIFIABLE (doc_ids unknown)
39. culture-10230151 | offset50 | 3 — ❌ REJECTED (2026-11-17)
40. general-13160920 | offset50 | 2 — ❌ REJECTED (2026-11-17): poetry-anthology summary, long-form Qwen version is degraded/repetitive in quality but no clear gender/binyan break located
41. animals-8561883 | offset50 | 2 — ❌ REJECTED (2026-11-17): pet-adoption listing, no human gender-identity claim at stake
42. celebrities-7785163 | offset50 | 2 — ❌ REJECTED (2026-11-17): Alyssa Milano article, all fem-consistent
43. general-14061589 | offset100 (actually found in `_offset20`, not `_offset100` — likely a mislabeling in this list) | 2 — ❌ REJECTED (2026-11-17): long-form version has "ישראל גולדשטיין... החליט לפעול **לבדו**" (correctly masc.) — the error found below (#71) is specific to the restricted-prompt generation of this same article, does not reproduce here
44. general-6866595 | offset100 (actually found in `_offset20`) | 3 — ✅ CONFIRMED (2026-11-17): "גם **פרופ' אש דרשה** לחגוג את חג פורים בהדרגה" — fem verb for the real, named, male official Prof. Nachman Ash; this is the long-form counterpart that DOES reproduce (contrast with #72 below, which does not)

Claude Haiku 4.5 (tag: us_anthropic_claude-haiku-4-5-20251001-v1_0), 12 total:
45. (val20) — 1 finding, doc_id not recorded at compile time — ⬜ UNVERIFIABLE (doc_id unknown)
46-48. (offset20, "+30") — 3 findings, doc_ids not recorded at compile time — ⬜ UNVERIFIABLE (doc_ids unknown)
49. sport-2880669 | offset50 | 3 — ❌ REJECTED (2026-11-17): "Buba Shel Laila" TV-show profile, very long and name-dense, but no gender/binyan break located on this pass
50-53. (offset50, 4 more findings) — ⬜ UNVERIFIABLE (doc_ids unknown)
54. animals-8321545 | offset100 | 2 — ❌ REJECTED (2026-11-17): pet-adoption listing, no error located
55. culture-6009067 | offset100 | 3 — ❌ REJECTED (2026-11-17): Yuval Shimoni book-review piece, no error located
56. general-14268886 | offset100 | 2 — ❌ REJECTED (2026-11-17): Avner Shalev (Yad Vashem) profile, consistently masc, no error located

GPT-5.6-Luna (tag: openai_gpt-5_6-luna), 3 total:
57. general-13160920 | offset50 | 2 — ❌ REJECTED (2026-11-17): poetry-anthology summary, no error located
58-59. (offset20, "+30") — 2 findings, doc_ids not recorded at compile time — ⬜ UNVERIFIABLE (doc_ids unknown)

## BATCH C — Restricted prompt (13 findings), file pattern: current_model_summaries_<tag>_restricted_val20[_offsetN].csv
GLM-4.7-Flash, 2 total:
60. celebrities-7478486 | offset50 | 2 — ❌ REJECTED (2026-11-17): Nicole Raidman/Jennifer Snookel feud piece, no error located
61. culture-14100072 | offset50 | 3 — ✅ CONFIRMED (2026-11-17): same "**הם סמיט**" (dropped-ש) error as #28 above, reproduced independently under the restricted prompt on the same article — a genuine, cross-prompt-style-replicated GLM trap

Ministral-3-14B, 8 total:
62. culture-1900585 | offset50 | 3 — ❌ REJECTED (2026-11-17): children's-book-roundup piece, no error located
63. culture-3665096 | offset50 | 2 — ❌ REJECTED (2026-11-17): reality-TV wedding-show piece, no error located
64. general-14268637 | offset50 | 3 — 🆕 ✅ CONFIRMED, but as a DIFFERENT error than whatever the original judge saw (2026-11-17): "ומתכוניה... הפכו למסורת משפחתית, כש**בנה**, יהודית, משמרת אותם" — "בנה" means "her SON," but Yehudit is repeatedly established as the DAUGHTER throughout the source; a smixut/possessive-noun gender error, unreproducible in English (son/daughter are unrelated words there)
65. animals-8561883 | offset50 | 2 — ❌ REJECTED (2026-11-17): pet-adoption listing, no error located
66. business-6888267 | offset50 | 3 — ❌ REJECTED (2026-11-17): could not relocate the claimed error on this pass ("נולדו טוב" real-estate-kids piece); may need a closer re-read, not ruled out with high confidence
67. celebrities-7478486 | offset50 | 2 — ❌ REJECTED (2026-11-17): same article as #60, Ministral's version also shows no error
68. culture-14100072 | offset50 | 4 — ❌ REJECTED (2026-11-17): Ministral's restricted version correctly writes "**שהם סמיט**" (clitic intact) — unlike GLM, no error here
69. culture-10266649 | offset50 | 2 — ❌ REJECTED (2026-11-17): עילגות-dictionary article, no error located

Claude Haiku 4.5, 1 total:
70. general-14268886 | offset50 | 2 — ❌ REJECTED (2026-11-17): Avner Shalev profile, restricted version, no error located

Qwen3-32B, 2 total:
71. general-14061589 | (restricted val20, no offset) | 2 — ✅ CONFIRMED (2026-11-17): "גולדשטיין, מפקד פלגה בנ"מ, **הפעילה** לבדה את העמדה... **והפיל**" — fem/fem/masc all referring to the same real, named, male soldier within one sentence; unreproducible in English
72. general-6866595 | (restricted val20, no offset) | 3 — ❌ REJECTED (2026-11-17): the restricted-prompt version of this COVID-reopening article does not contain the "פרופ' אש" error found in the long-form version (#44); no error located here

GPT-5.6-Luna, 0 total (none)

## Replication note (revised after re-verification)
Before re-verification, several doc_ids appeared to recur as errors across models/prompt-styles. After the
blind re-check, most of those "replications" turned out to be false — e.g. culture-10230151, culture-10266649,
and general-13160920 were each claimed 2-4 times across different models/prompts, but **zero** of those
claims reproduced on re-check. The one doc_id where cross-prompt replication is now genuinely confirmed is
**culture-14100072 for GLM specifically**: the "שהם סמיט" → "הם סמיט" dropped-clitic error reproduces
independently in both the long-form (#28) and restricted (#61) generations of the same article by the same
model — this is the strongest validated replication in the whole list. general-14061589/general-6866595
(Qwen, the Goldstein/Ash findings, #44/#71) show a different, more interesting pattern: each doc_id breaks
under only ONE of the two prompt styles (not both, despite originally being claimed in both) — evidence that
prompt style changes *which* errors a model makes on a given article, not just how many.

## Open item
17 rows above have "doc_id not recorded at compile time" / "UNVERIFIABLE" — these were findings reported by
subagents in earlier rounds of this project before this master list existed, and the specific doc_id was not
captured in the coordinator's running notes. Re-deriving them would require re-scanning the relevant raw CSVs
for candidate errors from scratch; not done as part of this pass. Given that only 29% of the *checkable*
findings survived blind re-verification, these 17 should NOT be assumed confirmed by default — if a precise
rate is ever needed for the paper, they should be found and re-checked, or excluded from any headline count.
