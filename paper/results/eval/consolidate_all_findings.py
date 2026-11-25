"""
One-off consolidation script: merges every confirmed Hebrew-morphology correctness-error
finding from every model/round ever tested in this project (old historical production-pipeline
pool + the first new-run wave + the OpenRouter current-model-check wave) into a single master
JSON, with a consistent schema per finding.

Run once from paper/results/eval/. Not part of any pipeline; safe to delete after use.
"""
import pandas as pd
import glob
import json
import re
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

findings = []
_id = 0


def add(run_tier, model, doc_id, source, error_type, summary, source_claim, false_claim,
        severity, mrl_feature=None, verification_method=None, confidence=None, notes=None):
    global _id
    _id += 1
    findings.append({
        "finding_id": _id,
        "run_tier": run_tier,
        "model": model,
        "article_id": None if pd.isna(doc_id) else str(doc_id),
        "source": None if (source is None or (isinstance(source, float) and pd.isna(source))) else str(source),
        "error_type": None if (error_type is None or (isinstance(error_type, float) and pd.isna(error_type))) else str(error_type),
        "summary": None if (summary is None or (isinstance(summary, float) and pd.isna(summary))) else str(summary),
        "source_claim": None if (source_claim is None or (isinstance(source_claim, float) and pd.isna(source_claim))) else str(source_claim),
        "false_claim": None if (false_claim is None or (isinstance(false_claim, float) and pd.isna(false_claim))) else str(false_claim),
        "severity": None if severity is None or (isinstance(severity, float) and pd.isna(severity)) else int(severity) if float(severity) == int(float(severity)) else float(severity),
        "mrl_feature": None if (mrl_feature is None or (isinstance(mrl_feature, float) and pd.isna(mrl_feature))) else str(mrl_feature),
        "verification_method": verification_method,
        "confidence": confidence,
        "notes": notes,
    })


# ---------------------------------------------------------------------------
# TIER 1: historical production-pipeline pool (old models, Gemini/Claude-judged)
# ---------------------------------------------------------------------------
MODEL_FROM_FILENAME = [
    (r"gpt3\.5", "GPT-3.5-Turbo"),
    (r"dicta2\.0", "DictaLM 2.0"),
    (r"gemma3-12b", "Gemma-3-12B"),
    (r"gpt-oss-20b", "gpt-oss-20B"),
    (r"qwen3_14b", "Qwen3-14B"),
]

hist_files = sorted(glob.glob(os.path.join(REPO, "thesisdatasets/error_finder/*.csv")))
for f in hist_files:
    base = os.path.basename(f)
    model = next((name for pat, name in MODEL_FROM_FILENAME if re.search(pat, base)), "unknown-old-model")
    judge_note = "gemini_finder"
    if "claude_temp0.1" in base:
        judge_note = "claude_temp0.1_finder"
    elif "claude_temp1.0" in base:
        judge_note = "claude_temp1.0_finder"
    df = pd.read_csv(f)
    for _, r in df.iterrows():
        add(
            run_tier="historical_pool",
            model=model,
            doc_id=r.get("doc_id"),
            source=r.get("source"),
            error_type=r.get("correctness_error_name"),
            summary=r.get("correctness_error_reasoning"),
            source_claim=r.get("evidence_from_source_article"),
            false_claim=r.get("evidence_from_summary"),
            severity=r.get("severity"),
            mrl_feature=r.get("mrl_unique_hebrew_features"),
            verification_method=f"automated_production_pipeline_{judge_note}",
            notes=f"source_file={base}",
        )

# ---------------------------------------------------------------------------
# TIER 2: DictaLM 3.0 1.7B-Instruct random-sample finding (not in the historical pool files)
# ---------------------------------------------------------------------------
recal_path = os.path.join(HERE, "current_vs_old_model_errorfinder_recalibrated.csv")
recal = pd.read_csv(recal_path)
recal_found = recal[recal.severity.notna()]
for _, r in recal_found.iterrows():
    if str(r.doc_id) == "218050":  # the two oldmodel_control rows here duplicate the historical-pool entries
        add(
            run_tier="dictalm3_random_sample",
            model="DictaLM 3.0 1.7B-Instruct",
            doc_id=r.doc_id,
            source=r.source,
            error_type=r.correctness_error_name,
            summary=r.correctness_error_reasoning,
            source_claim=r.evidence_from_source_article,
            false_claim=r.evidence_from_summary,
            severity=r.severity,
            mrl_feature=r.mrl_unique_hebrew_features,
            verification_method="claude_manual_judge",
            notes="random (not targeted) sample of 69 MAFAT articles",
        )

# ---------------------------------------------------------------------------
# TIER 3: first new-run wave (GLM-4.7-Flash, Ministral-3-14B, GPT-5.6-Luna, Qwen3-32B,
# Claude Haiku 4.5) -- detail CSVs with full quoted evidence
# ---------------------------------------------------------------------------
MODEL_CANON = {
    "zai.glm-4.7-flash": "GLM-4.7-Flash",
    "GLM-4.7-Flash": "GLM-4.7-Flash",
    "mistral.ministral-3-14b-instruct": "Ministral-3-14B-Instruct",
    "Ministral-3-14B-Instruct": "Ministral-3-14B-Instruct",
    "qwen.qwen3-32b-v1:0": "Qwen3-32B-v1.0",
    "openai.gpt-5.6-luna": "GPT-5.6-Luna",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0": "Claude Haiku 4.5",
}

seen_tier3 = set()  # dedup key: (doc_id, canonical model, evidence_from_summary[:80])

detail_files = [
    "complex_hebrew_new_findings_2026-11-16.csv",
    "glm_hebrew_additional_findings_2026-11-16.csv",
    "longform_complex_hebrew_findings_2026-11-16.csv",
]
for fname in detail_files:
    df = pd.read_csv(os.path.join(HERE, fname))
    for _, r in df.iterrows():
        model = MODEL_CANON.get(r.get("model"), r.get("model"))
        key = (str(r.get("doc_id")), model, str(r.get("evidence_from_summary"))[:80])
        if key in seen_tier3:
            continue
        seen_tier3.add(key)
        add(
            run_tier="new_run_pool",
            model=model,
            doc_id=r.get("doc_id"),
            source=r.get("source") if "source" in df.columns else None,
            error_type=r.get("correctness_error_name"),
            summary=r.get("correctness_error_reasoning"),
            source_claim=r.get("evidence_from_source_article"),
            false_claim=r.get("evidence_from_summary"),
            severity=r.get("severity"),
            mrl_feature=r.get("mrl_unique_hebrew_features"),
            verification_method="claude_manual_judge",
            confidence=r.get("confidence") if "confidence" in df.columns else None,
            notes=f"source_file={fname}",
        )

# A handful of confirmed §7c findings never made it into the detail CSVs above (only a short
# mechanism gloss exists for these, no exact quoted evidence captured in a CSV) -- added from
# docs/error_finder_instructions_for_hebrew_llm.md section 7c directly.
MANUAL_7C_ONLY = [
    dict(model="GLM-4.7-Flash", doc_id="animals-6829845", severity=2,
         error_type="established-masculine referent flipped feminine",
         summary="An established-masculine dog (קירק) is flipped to feminine grammatical agreement throughout the summary."),
    dict(model="Ministral-3-14B-Instruct", doc_id="culture-7822389", severity=1,
         error_type="masculine verb for a female referent",
         summary="Masculine verb form מספר used for the female singer Si Heyman instead of feminine מספרת."),
    dict(model="Ministral-3-14B-Instruct", doc_id="culture-10173038", severity=1,
         error_type="masculine verb for a female referent",
         summary="Masculine התגאה used instead of feminine התגאתה for a female band member."),
    dict(model="GPT-5.6-Luna", doc_id="general-13160920", severity=2,
         error_type="1st-person-to-3rd-person poetry gloss gender/person flip",
         summary="Poetry gloss converts 1st person שהייתי לדודי to 3rd-person feminine שהיה לדודה, changing whose dedication the poem is."),
    dict(model="GLM-4.7-Flash", doc_id="food-333642", severity=4,
         error_type="masculine-plural clitic collapsed to feminine-singular",
         summary="Masculine-plural clitic חבריו (\"his friends\") collapses into feminine-singular חברה (also = \"company\"), swapping identity and number between a burned soldier and the restaurant itself."),
    dict(model="GLM-4.7-Flash", doc_id="culture-13795813", severity=3,
         error_type="homograph misreading (עצום: closed vs. huge)",
         summary="עצום homograph (\"closed [eyes]\" vs. \"huge/immense\") misread, inverting attentive observers into oblivious ones."),
    dict(model="Qwen3-32B-v1.0", doc_id="celebrities-8932596", severity=2,
         error_type="1st-person gender-neutral to 3rd-person feminine misattribution",
         summary="1st-person gender-neutral נולדתי rendered as 3rd-person feminine נולדה for a male dancer."),
    dict(model="Qwen3-32B-v1.0", doc_id="celebrities-8932596", severity=2,
         error_type="reflexive clitic gender flip",
         summary="Reflexive clitic עצמי rendered as feminine עצמה on the same male dancer, in the same summary as the נולדתי/נולדה flip."),
    dict(model="Qwen3-32B-v1.0", doc_id="magazine-13224346", severity=3,
         error_type="possessive clitic gender flip (his→her position)",
         summary="Possessive clitic תפקידי (his position) rendered as תפקידה (her position)."),
    dict(model="Qwen3-32B-v1.0", doc_id="magazine-13224346", severity=2,
         error_type="root-pattern collision (תקנו misread)",
         summary="Root-pattern collision: תקנו (ק.נ.ה, \"buy/believe\") misread as ת.ק.ן (\"fix\"), turning a flat denial into \"refuses to fix.\""),
    dict(model="GLM-4.7-Flash", doc_id="culture-10266649", severity=2,
         error_type="gendered-numeral agreement left uncorrected",
         summary="A linguistics column about Hebrew errors leaves a gendered-numeral agreement error (שלוש שקל) unfixed in the model's \"corrected\" example."),
    dict(model="Ministral-3-14B-Instruct", doc_id="culture-10266649", severity=3,
         error_type="compound-preposition misparse (ל+ב+חוץ)",
         summary="ל+ב+חוץ compound-preposition stack misparsed as ל+מחוץ (\"to crush\"), an unrelated root, in the same linguistics column about Hebrew errors."),
]
for m in MANUAL_7C_ONLY:
    key = (str(m["doc_id"]), m["model"], m["error_type"][:80])
    if key in seen_tier3:
        continue
    seen_tier3.add(key)
    add(
        run_tier="new_run_pool",
        model=m["model"],
        doc_id=m["doc_id"],
        source=None,
        error_type=m["error_type"],
        summary=m["summary"],
        source_claim=None,
        false_claim=None,
        severity=m["severity"],
        mrl_feature=None,
        verification_method="claude_manual_judge",
        notes="exact quoted evidence not preserved in a CSV; see docs/error_finder_instructions_for_hebrew_llm.md section 7c and docs/experiments.md",
    )

# ---------------------------------------------------------------------------
# TIER 4: OpenRouter current-model-check wave (Qwen3.8-Flash, DeepSeek V4 Pro 0813, Kimi K2.6)
# ---------------------------------------------------------------------------
openrouter_files = [
    "manual_judge_all_findings.csv",
    "manual_judge_qwen_offset50_findings.csv",
    "manual_judge_deepseek_offset50_findings.csv",
    "manual_judge_deepseek_offset100_findings.csv",
    "manual_judge_newpool_findings.csv",
    "manual_judge_kimi_newpool_findings.csv",
    "manual_judge_kimi_newpool_round2_findings.csv",
]
OPENROUTER_MODEL_CANON = {
    "qwen/qwen3.8-flash": "Qwen3.8-Flash",
    "deepseek/deepseek-v4-pro-0813": "DeepSeek V4 Pro 0813",
    "moonshotai/kimi-k2.6": "Kimi K2.6",
}
for fname in openrouter_files:
    path = os.path.join(HERE, fname)
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        model = OPENROUTER_MODEL_CANON.get(r.get("model"), r.get("model"))
        add(
            run_tier="openrouter_current_run",
            model=model,
            doc_id=r.get("doc_id"),
            source=r.get("source"),
            error_type=r.get("correctness_error_name"),
            summary=r.get("correctness_error_reasoning"),
            source_claim=r.get("evidence_from_source_article"),
            false_claim=r.get("evidence_from_summary"),
            severity=r.get("severity"),
            mrl_feature=r.get("mrl_unique_hebrew_features"),
            verification_method="claude_manual_judge",
            confidence=r.get("confidence") if "confidence" in df.columns else None,
            notes=f"source_file={fname}",
        )

# borderline case (documented, NOT counted as a confirmed finding -- kept out of `findings`,
# written to a separate top-level key instead)
borderline_path = os.path.join(HERE, "manual_judge_kimi_newpool_round2_borderline.csv")
borderline = []
if os.path.exists(borderline_path):
    bdf = pd.read_csv(borderline_path)
    for _, r in bdf.iterrows():
        borderline.append({
            "model": OPENROUTER_MODEL_CANON.get(r.get("model"), r.get("model")),
            "article_id": str(r.get("doc_id")),
            "source": r.get("source"),
            "error_type": r.get("correctness_error_name"),
            "summary": r.get("correctness_error_reasoning"),
            "source_claim": r.get("evidence_from_source_article"),
            "false_claim": r.get("evidence_from_summary"),
            "severity": r.get("severity"),
            "confidence": r.get("confidence"),
            "reason_not_counted": "gender-fact erasure is a side effect of generic name hallucination, not of a traced morphological misparse",
        })

# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------
by_model = {}
by_tier = {}
for f in findings:
    by_model[f["model"]] = by_model.get(f["model"], 0) + 1
    by_tier[f["run_tier"]] = by_tier.get(f["run_tier"], 0) + 1

output = {
    "meta": {
        "description": "Master catalog of every confirmed Hebrew-morphology correctness-error "
                        "finding across every model and evaluation round in the MorphEvalHeb project, "
                        "from the original historical production-pipeline pool through the "
                        "OpenRouter current-model-check (Qwen3.8-Flash / DeepSeek V4 Pro 0813 / Kimi K2.6).",
        "total_confirmed_findings": len(findings),
        "counts_by_model": by_model,
        "counts_by_run_tier": by_tier,
        "run_tier_definitions": {
            "historical_pool": "Original production pipeline (Gemini/Claude-judged) on old models: GPT-3.5-Turbo, DictaLM 2.0, Gemma-3-12B, gpt-oss-20B, Qwen3-14B.",
            "dictalm3_random_sample": "DictaLM 3.0 1.7B-Instruct, random (not targeted) sample of 69 MAFAT articles.",
            "new_run_pool": "First new-run wave on the 198-article targeted hard pool: GLM-4.7-Flash, Ministral-3-14B-Instruct, GPT-5.6-Luna, Qwen3-32B-v1.0, Claude Haiku 4.5.",
            "openrouter_current_run": "Current-model-check wave (2026): Qwen3.8-Flash, DeepSeek V4 Pro 0813, Kimi K2.6 -- on the 198-article targeted pool plus a disjoint 293-article never-before-summarized pool.",
        },
        "important_caveat": "The 450 'historical_pool' rows come from the project's original "
                             "automated (Gemini/Claude-judged) production pipeline, run BEFORE the "
                             "strict MRL-only taxonomy/mandatory-gate/English-contrast-test methodology "
                             "was established and calibrated later in the project (see "
                             "docs/error_finder_instructions_for_hebrew_llm.md section 6, 'judge "
                             "miscalibration' lesson). They were not re-vetted against that strict gate "
                             "in this consolidation, so this tier likely includes some errors that would "
                             "NOT pass the English-contrast test (i.e. errors an English summarizer could "
                             "have made too) alongside genuine Hebrew-morphology-specific ones. All other "
                             "tiers (dictalm3_random_sample, new_run_pool, openrouter_current_run) WERE "
                             "manually judged by Claude directly against the strict gate + English-contrast "
                             "test and are apples-to-apples comparable with each other.",
        "field_definitions": {
            "error_type": "Short name/category of the Hebrew-morphology mechanism responsible for the error.",
            "summary": "One-paragraph description of how the Hebrew morphology produced the factual error.",
            "source_claim": "Exact quoted text from the source article establishing the true fact.",
            "false_claim": "Exact quoted text from the generated summary asserting the changed/false fact.",
            "verification_method": "Whether this was found by the original automated production pipeline (Gemini/Claude judge) or manually judged by Claude directly in this research session.",
        },
    },
    "findings": findings,
    "documented_borderline_not_counted": borderline,
}

out_path = os.path.join(HERE, "all_models_all_findings_master.json")
with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(output, fh, ensure_ascii=False, indent=2)

print(f"Wrote {len(findings)} confirmed findings + {len(borderline)} borderline to {out_path}")
print()
print("By run_tier:")
for k, v in by_tier.items():
    print(f"  {k}: {v}")
print()
print("By model:")
for k, v in sorted(by_model.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v}")
