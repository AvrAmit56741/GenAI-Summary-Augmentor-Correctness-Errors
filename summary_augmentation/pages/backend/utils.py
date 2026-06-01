import logging
from collections import Counter
from datetime import datetime
from typing import List

import pandas as pd

from backend.sampling_stats import SamplingStats

logger = logging.getLogger(__name__)


def parse_error_types_list(
    raw_error_types_str: str,
) -> List[str]:
    if pd.isna(raw_error_types_str) or str(raw_error_types_str).strip() == "":
        return []
    return [
        error_type.strip()
        for error_type in str(raw_error_types_str).split(",")
        if error_type.strip()
    ]


def enrich_df(
    benchmark_df: pd.DataFrame,
) -> pd.DataFrame:
    enriched = benchmark_df.copy()
    enriched["_source_key"] = (
        enriched["source"].fillna("external") if "source" in enriched.columns else "unknown"
    )
    enriched["_len_bin"] = (
        enriched["len_group"].apply(lambda x: x[0] if isinstance(x, str) and x else "?")
        if "len_group" in enriched.columns
        else "?"
    )
    enriched["_method_key"] = (
        enriched["method"].fillna("standard") if "method" in enriched.columns else "standard"
    )
    enriched["_error_list"] = (
        enriched["error_types"].apply(parse_error_types_list)
        if "error_types" in enriched.columns
        else [[] for _ in range(len(enriched))]
    )
    return enriched


def error_type_coverage(
    enriched_df: pd.DataFrame,
) -> "Counter[str]":
    all_error_types = [
        error_type
        for error_type_list in enriched_df["_error_list"]
        for error_type in error_type_list
    ]
    return Counter(all_error_types)


def estimated_minimum_n(
    enriched_df: pd.DataFrame,
) -> int:
    """Estimate the minimum N needed to cover all error types, source×label pairs, and len bins."""
    contradiction_df = enriched_df[enriched_df["label"] == "contradiction"]
    n_error_types = len({
        error_type
        for error_type_list in contradiction_df["_error_list"]
        for error_type in error_type_list
    })
    n_source_label_pairs = enriched_df["_source_key"].nunique() * enriched_df["label"].nunique()
    n_len_bins = enriched_df["_len_bin"].nunique()
    return n_error_types + n_source_label_pairs + n_len_bins


def generate_subset_markdown_report(
    benchmark_df_enriched: pd.DataFrame,
    subset_df_enriched: pd.DataFrame,
    sampling_stats: SamplingStats,
    target_size: int,
    random_seed: int,
) -> str:
    original_error_coverage = error_type_coverage(benchmark_df_enriched)
    subset_error_coverage_counts = error_type_coverage(subset_df_enriched)
    covered_types_count = sum(
        1 for error_type_name in original_error_coverage
        if subset_error_coverage_counts.get(error_type_name, 0) > 0
    )

    lines: List[str] = []

    lines += [
        "# Benchmark Subset Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Target size:** {target_size:,}  ",
        f"**Random seed:** {random_seed}  ",
        "",
        "## Sampling Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total rows in subset | {len(subset_df_enriched):,} |",
        f"| Phase 1 — mandatory coverage rows | {sampling_stats.mandatory_coverage_count} |",
        f"| Phase 2 — proportional fill rows | {sampling_stats.proportional_fill_count} |",
        f"| Truncated to target size | {'Yes ⚠️' if sampling_stats.truncated else 'No'} |",
        f"| Error types covered | {covered_types_count} / {len(original_error_coverage)} |",
        "",
    ]

    lines += _markdown_distribution_table(
        section_title="Label Distribution",
        original_series=benchmark_df_enriched["label"].value_counts(normalize=True),
        subset_series=subset_df_enriched["label"].value_counts(normalize=True),
        column_header="Label",
        sort_key=lambda label_value: label_value,
    )

    lines += _markdown_distribution_table(
        section_title="Source Distribution",
        original_series=benchmark_df_enriched["_source_key"].value_counts(normalize=True),
        subset_series=subset_df_enriched["_source_key"].value_counts(normalize=True),
        column_header="Source",
        sort_key=lambda source_key_value: source_key_value,
    )

    lines += _markdown_distribution_table(
        section_title="Length Group Distribution",
        original_series=benchmark_df_enriched["_len_bin"].value_counts(normalize=True),
        subset_series=subset_df_enriched["_len_bin"].value_counts(normalize=True),
        column_header="Len Group",
        sort_key=lambda len_bin_value: len_bin_value,
    )

    lines += [
        "## Error Type Coverage",
        "",
        "| Error Type | Original Count | Subset Count | Covered |",
        "|------------|---------------|-------------|---------|",
    ]
    for error_type_name in sorted(original_error_coverage, key=lambda n: -original_error_coverage[n]):
        original_count = original_error_coverage[error_type_name]
        subset_count = subset_error_coverage_counts.get(error_type_name, 0)
        covered_marker = "✅" if subset_count > 0 else "❌"
        lines.append(f"| {error_type_name} | {original_count} | {subset_count} | {covered_marker} |")
    lines.append("")

    return "\n".join(lines)


def _markdown_distribution_table(
    section_title: str,
    original_series: "pd.Series",
    subset_series: "pd.Series",
    column_header: str,
    sort_key,
) -> List[str]:
    rows = [
        f"## {section_title}",
        "",
        f"| {column_header} | Original % | Subset % |",
        f"|{'-' * (len(column_header) + 2)}|-----------|---------|",
    ]
    for category_value in sorted(original_series.index, key=sort_key):
        original_pct = original_series.get(category_value, 0) * 100
        subset_pct = subset_series.get(category_value, 0) * 100
        rows.append(f"| {category_value} | {original_pct:.1f}% | {subset_pct:.1f}% |")
    rows.append("")
    return rows
