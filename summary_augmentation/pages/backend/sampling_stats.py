from dataclasses import dataclass


@dataclass
class SamplingStats:
    mandatory_coverage_count: int
    proportional_fill_count: int
    total_error_types_covered: int
    truncated: bool
