import logging
import random
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from backend.sampling_stats import SamplingStats
from backend.utils import enrich_df

logger = logging.getLogger(__name__)

_INTERNAL_COL_PREFIX = "_"


class BenchmarkSampler:
    """
    Two-phase stratified sampler for benchmark CSV datasets.

    Phase 1 (greedy coverage): selects the minimum rows needed to cover every
    error type, every source×label pair, and every len_group bucket.

    Phase 2 (proportional fill): distributes the remaining quota across
    source/len_bin strata, weighting contradiction rows by error-type rarity
    so rare error types remain represented.
    """

    def __init__(
        self,
        benchmark_df: pd.DataFrame,
        target_size: int,
        seed: int = 42,
    ) -> None:
        self._target_size = target_size
        self._seed = seed
        self._df = enrich_df(benchmark_df)
        self._contradiction_df = self._df[self._df["label"] == "contradiction"]
        self._entailment_df = self._df[self._df["label"] == "entailment"]
        self._all_error_types: Set[str] = {
            error_type
            for error_type_list in self._contradiction_df["_error_list"]
            for error_type in error_type_list
        }

    def sample(self) -> Tuple[pd.DataFrame, SamplingStats]:
        np.random.seed(self._seed)
        random.seed(self._seed)

        mandatory_indices = self._run_phase1_coverage()
        logger.info("Phase 1 complete: %d mandatory rows selected", len(mandatory_indices))

        if len(mandatory_indices) >= self._target_size:
            logger.warning(
                "Mandatory coverage (%d) >= target size (%d), truncating",
                len(mandatory_indices),
                self._target_size,
            )
            return self._build_truncated_result(mandatory_indices)

        fill_indices = self._run_phase2_fill(mandatory_indices)
        logger.info("Phase 2 complete: %d fill rows added", len(fill_indices))

        chosen_indices = list(dict.fromkeys(list(mandatory_indices) + fill_indices))
        random.shuffle(chosen_indices)

        return self._build_df(chosen_indices), SamplingStats(
            mandatory_coverage_count=len(mandatory_indices),
            proportional_fill_count=len(chosen_indices) - len(mandatory_indices),
            total_error_types_covered=len(self._all_error_types),
            truncated=False,
        )

    # ── Phase 1 ───────────────────────────────────────────────────────────────

    def _run_phase1_coverage(self) -> Set[int]:
        mandatory_indices: Set[int] = set()
        mandatory_indices |= self._cover_all_error_types(mandatory_indices)
        mandatory_indices |= self._cover_source_label_pairs(mandatory_indices)
        mandatory_indices |= self._cover_len_bin_buckets(mandatory_indices)
        return mandatory_indices

    def _cover_all_error_types(
        self,
        excluded_indices: Set[int],
    ) -> Set[int]:
        uncovered_types = set(self._all_error_types)
        picked_indices: Set[int] = set()
        while uncovered_types:
            best_row_idx, best_coverage_count = self._find_best_covering_row(
                uncovered_types=uncovered_types,
                excluded_indices=excluded_indices | picked_indices,
            )
            if best_row_idx is None:
                break
            picked_indices.add(best_row_idx)
            uncovered_types -= set(self._contradiction_df.loc[best_row_idx, "_error_list"])
        logger.debug(
            "Error type coverage: %d rows picked for %d types",
            len(picked_indices),
            len(self._all_error_types),
        )
        return picked_indices

    def _find_best_covering_row(
        self,
        uncovered_types: Set[str],
        excluded_indices: Set[int],
    ) -> Tuple[Optional[int], int]:
        best_row_idx: Optional[int] = None
        best_coverage_count = 0
        for row_idx, row in self._contradiction_df.iterrows():
            if row_idx in excluded_indices:
                continue
            coverage_count = len(set(row["_error_list"]) & uncovered_types)
            if coverage_count > best_coverage_count:
                best_coverage_count = coverage_count
                best_row_idx = row_idx
        return best_row_idx, best_coverage_count

    def _cover_source_label_pairs(
        self,
        excluded_indices: Set[int],
    ) -> Set[int]:
        picked_indices: Set[int] = set()
        for source_key_value in self._df["_source_key"].unique():
            for label_value in self._df["label"].unique():
                source_label_pool = self._df[
                    (self._df["_source_key"] == source_key_value)
                    & (self._df["label"] == label_value)
                ]
                available_rows = source_label_pool[
                    ~source_label_pool.index.isin(excluded_indices | picked_indices)
                ]
                if not available_rows.empty:
                    picked_indices.add(available_rows.index[0])
        logger.debug("Source×label coverage: %d rows picked", len(picked_indices))
        return picked_indices

    def _cover_len_bin_buckets(
        self,
        excluded_indices: Set[int],
    ) -> Set[int]:
        picked_indices: Set[int] = set()
        for len_bin_value in self._df["_len_bin"].unique():
            len_bin_pool = self._df[self._df["_len_bin"] == len_bin_value]
            available_rows = len_bin_pool[
                ~len_bin_pool.index.isin(excluded_indices | picked_indices)
            ]
            if not available_rows.empty:
                picked_indices.add(available_rows.index[0])
        logger.debug("Len-bin coverage: %d rows picked", len(picked_indices))
        return picked_indices

    # ── Phase 2 ───────────────────────────────────────────────────────────────

    def _run_phase2_fill(
        self,
        mandatory_indices: Set[int],
    ) -> List[int]:
        contradiction_quota, entailment_quota = self._compute_fill_quotas(mandatory_indices)
        rarity_weights_by_error_type = self._compute_rarity_weights()

        contradiction_pool = self._contradiction_df[
            ~self._contradiction_df.index.isin(mandatory_indices)
        ].copy()
        contradiction_pool["_w"] = contradiction_pool["_error_list"].apply(
            lambda error_type_list: self._row_rarity_score(
                error_type_list=error_type_list,
                rarity_weights_by_error_type=rarity_weights_by_error_type,
            )
        )

        entailment_pool = self._entailment_df[
            ~self._entailment_df.index.isin(mandatory_indices)
        ].copy()

        contradiction_fill = self._proportional_sample(
            sample_pool=contradiction_pool,
            target_sample_count=contradiction_quota,
            strata_columns=["_source_key", "_len_bin"],
            weights_column="_w",
        )
        entailment_fill = self._proportional_sample(
            sample_pool=entailment_pool,
            target_sample_count=entailment_quota,
            strata_columns=["_source_key", "_method_key"],
            weights_column=None,
        )
        return contradiction_fill + entailment_fill

    def _compute_fill_quotas(
        self,
        mandatory_indices: Set[int],
    ) -> Tuple[int, int]:
        contradiction_ratio = len(self._contradiction_df) / len(self._df)
        mandatory_contradiction_count = sum(
            1 for row_idx in mandatory_indices
            if self._df.loc[row_idx, "label"] == "contradiction"
        )
        mandatory_entailment_count = len(mandatory_indices) - mandatory_contradiction_count
        target_contradiction_count = round(self._target_size * contradiction_ratio)
        target_entailment_count = self._target_size - target_contradiction_count
        contradiction_quota = max(0, target_contradiction_count - mandatory_contradiction_count)
        entailment_quota = max(0, target_entailment_count - mandatory_entailment_count)
        return contradiction_quota, entailment_quota

    def _compute_rarity_weights(self) -> Dict[str, float]:
        all_error_type_occurrences = [
            error_type
            for error_type_list in self._contradiction_df["_error_list"]
            for error_type in error_type_list
        ]
        occurrence_counts_by_type = Counter(all_error_type_occurrences)
        max_occurrence_count = max(occurrence_counts_by_type.values(), default=1)
        return {
            error_type: max_occurrence_count / occurrence_count
            for error_type, occurrence_count in occurrence_counts_by_type.items()
        }

    def _row_rarity_score(
        self,
        error_type_list: List[str],
        rarity_weights_by_error_type: Dict[str, float],
    ) -> float:
        if not error_type_list:
            return 1.0
        return max(
            (rarity_weights_by_error_type.get(error_type, 1.0) for error_type in error_type_list),
            default=1.0,
        )

    def _proportional_sample(
        self,
        sample_pool: pd.DataFrame,
        target_sample_count: int,
        strata_columns: List[str],
        weights_column: Optional[str],
    ) -> List[int]:
        if target_sample_count <= 0 or sample_pool.empty:
            return []
        capped_count = min(target_sample_count, len(sample_pool))
        stratum_groups = [
            (stratum_key, stratum_df)
            for stratum_key, stratum_df in sample_pool.groupby(strata_columns, group_keys=False)
        ]
        total_pool_size = len(sample_pool)
        sampled_indices: List[int] = []
        allocated_count = 0
        for group_index, (_, stratum_df) in enumerate(stratum_groups):
            is_last_group = group_index == len(stratum_groups) - 1
            if is_last_group:
                stratum_quota = capped_count - allocated_count
            else:
                stratum_quota = max(1, round(capped_count * len(stratum_df) / total_pool_size))
            stratum_quota = min(stratum_quota, len(stratum_df))
            if stratum_quota <= 0:
                continue
            chosen_rows = self._sample_from_stratum(stratum_df, stratum_quota, weights_column)
            sampled_indices.extend(chosen_rows.index.tolist())
            allocated_count += stratum_quota
        return sampled_indices

    def _sample_from_stratum(
        self,
        stratum_df: pd.DataFrame,
        stratum_quota: int,
        weights_column: Optional[str],
    ) -> pd.DataFrame:
        if weights_column and weights_column in stratum_df.columns:
            normalized_weights = stratum_df[weights_column] / stratum_df[weights_column].sum()
            return stratum_df.sample(
                n=stratum_quota,
                weights=normalized_weights,
                random_state=self._seed,
                replace=False,
            )
        return stratum_df.sample(n=stratum_quota, random_state=self._seed)

    # ── Result builders ───────────────────────────────────────────────────────

    def _build_truncated_result(
        self,
        mandatory_indices: Set[int],
    ) -> Tuple[pd.DataFrame, SamplingStats]:
        chosen_indices = list(mandatory_indices)[:self._target_size]
        return self._build_df(chosen_indices), SamplingStats(
            mandatory_coverage_count=len(mandatory_indices),
            proportional_fill_count=0,
            total_error_types_covered=len(self._all_error_types),
            truncated=True,
        )

    def _build_df(
        self,
        chosen_indices: List[int],
    ) -> pd.DataFrame:
        internal_cols = [col for col in self._df.columns if col.startswith(_INTERNAL_COL_PREFIX)]
        return self._df.loc[chosen_indices].drop(columns=internal_cols, errors="ignore")
