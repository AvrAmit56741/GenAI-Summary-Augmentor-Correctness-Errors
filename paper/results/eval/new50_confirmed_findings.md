# Confirmed findings — new 50-article pool (never-before-tested sources)

3 confirmed Hebrew-morphology errors out of 250 summaries (5 models × 50 articles), found by 10 parallel
error-finder subagents (25 rows each) applying the mandatory-gate + English-contrast methodology from
`docs/error_finder_instructions_for_hebrew_llm.md`, with one coordinator correction (see below).
Selection methodology and generation results are in `docs/experiments.md` §19.

Raw subagent transcripts (read-only, linked by the harness, not committed to the repo):
`/private/tmp/claude-502/-Users-amit-avraham-PycharmProjects-urlf-classification/d7995097-e27a-437a-b264-7895646c5a47/tasks/{a0a898ac2f7dde603,acb588b7993678d10}.output`
(these will not survive a reboot/cleanup — this file is the permanent record).

## Coordinator correction (Qwen rows 1-25 batch)
The reporting subagent for Qwen rows 1-25 (`a0a898ac2f7dde603`) initially confirmed 4/25, including
`23037950` (Bagatz) and `16005360_c01` (Bagatz). Both were downgraded to NOT-confirmed on review: a sibling
Ministral subagent (`a652346c8276c0177`) independently examined the same `23037950` article under
Ministral's own summary and explicitly rejected an identical construction there ("doesn't clear the 'reader
would believe something genuinely different' bar; not a sustained identity-reversal... a single isolated
token amid otherwise-consistent masculine usage"), and both near-misses match the methodology doc's own
explicit rejected bad-example pattern (masculine verb attraction next to a nearby feminine noun, with no
real reader confusion about who the referent is). Net: **Qwen 2/50 confirmed**, not the originally reported 4/50.

---

## 1. GLM-4.7-Flash — `16_ptm_533324` (Knesset) — severity 2

**Mechanism (gate):** עמימות מורפולוגית קיצונית / כינוי קניין מוצמד (extreme morphological ambiguity /
possessive clitic) — one consonantal string, two unrelated readings, via Hebrew's defective (mater lectionis)
spelling.

**Premise (exact quote):** "...בשל כך **נפסלה מועמדתו** של בן העדה הדרוזית."
`מועמדתו` = the abstract noun `מועמדות` ("candidacy") + 3rd-person-masculine possessive clitic `-ו`, i.e.
"**his candidacy** [a male Druze applicant's] was voided" — the medial ו of `מועמדוֹת` is elided, a common
defective-spelling pattern.

**Summary (exact quote, from the original generation run):** "...**נפסלה מועמדת דרוזית** עקב טעות בייעוד
המכרז לערבים במקום לדרוזים."
The model parsed the identical letter string as the free-standing noun `מועמדת` ("[a] female candidate" —
מועמד + feminine suffix ת, a legitimate independent word), stripping out the possessive clitic entirely, and
agreed it with the feminine adjective `דרוזית`.

**Why it's a factual error:** the source describes one male person's voided job application (the referent
being `בן העדה הדרוזית`, "a [male] member of the Druze community"). The summary instead asserts a distinct,
named-gender fact: that a **female** Druze candidate was disqualified — a fabricated person of the wrong
gender, not just a stylistic slip.

**English-contrast test:** unreproducible in English. "His candidacy" and "a female candidate" share no
surface word-form; the collision is purely a product of Hebrew's defective orthography plus the fact that
the feminine abstract-noun suffix (-ות) and the feminine-candidate suffix (-ת) differ by exactly one
routinely-omitted vowel letter. Passes the gate.

**Caveat (originally noted by the judging agent):** depends on reading `מועמדתו` as elided `מועמדותו`; if the
Knesset stenographer intended the literal `מועמדת`+`ו` ("his [sponsored] female candidate"), the ambiguity
itself is still a confirmed Hebrew-specific mechanism, but the "fact that changed" would be murkier. Reported
at severity 2 with moderate (not high) confidence given this residual source-text ambiguity.

**Post-hoc note (2026-11-17 data-quality fix):** this row's `generative_summary` was truncated (cut off
mid-sentence near the end) in the original generation run. It was regenerated with a higher `maxTokens` to
fix the truncation (see `docs/experiments.md` §19, data-quality fixes). The regenerated summary (temperature
0.2, same prompt, same article) **no longer contains this error** — it omits the candidacy detail entirely.
The error-bearing original text is preserved above and in `pre_fix_backups/glm_new50_pre_fix_backup.csv`.
This finding stands as a real, observed error from the original run; its disappearance on regeneration is a
single-sample-variance data point, not a retraction — see §19 for discussion.

---

## 2. Qwen3-32B — `15_ptm_532917` (Knesset) — severity 2

**Mechanism (gate):** גוף/מגדר agreement break across a zero-anaphora coordinated verb chain.

**Premise:** Deputy Defense Minister א' סנה speaks of himself only with masculine verbs ("אני מתכבד להביא
לאישור הכנסת...", "אני מוכן לגשת מייד להצבעה").

**Summary (exact quote):** "סגן שר הביטחון, א' סנה, **הציג** את ההצעה, **והציעה** להחיל את התיקון גם על בני
זוג שהנישואים שלהם פקעו לפני תחילת החוק."
Same subject (א' סנה), same coordinated clause — first verb masculine (`הציג`), second flips feminine
(`והציעה`).

**English-contrast test:** Hebrew marks gender directly on the verb with no separate pronoun; "he
presented... and proposed" has no equivalent slip point in English. Passes the gate.

---

## 3. Qwen3-32B — `23_ptm_586576` (Knesset) — severity 2

**Mechanism (gate):** גוף/מגדר agreement break on a named, real, male public figure.

**Premise:** MK איתן גינזבורג speaking in a one-minute-speeches session (real male MK).

**Summary (exact quote):** "**איתן גינזבורג דיברה** על חשיבות העברת סמכויות לרשויות מקומיות."
`דיברה` is feminine, applied to a named male MK.

**English-contrast test:** "Ginzburg spoke about..." carries zero gender marking in English — unreproducible
there. Passes the gate.

---

## Pattern note
All 3 confirmed findings are Knesset (parliamentary-transcript) articles. Zero findings came from Bagatz,
Wikipedia, or Weizmann, despite those three sources making up 29/50 (58%) of the pool. See
`docs/experiments.md` §19 for the register-dependence discussion this motivates.
