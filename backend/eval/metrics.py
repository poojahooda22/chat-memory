"""Aggregate per-question results into the numbers we report.

One Record per (config, question). summarize() rolls a config's records into the three headline
measures — quality (J score, overall and per category), latency (p50/p95 of search and total),
and cost (mean tokens per question) — mirroring the paper's Table 2.
"""

import math
from dataclasses import dataclass, field


@dataclass
class Record:
    config: str
    sample_id: str
    category: str
    question: str
    gold: str
    predicted: str
    correct: bool
    search_ms: float
    total_ms: float
    prompt_tokens: int
    completion_tokens: int


@dataclass
class Summary:
    config: str
    n: int
    j_overall: float
    j_by_category: dict[str, float] = field(default_factory=dict)
    search_p50: float = 0.0
    search_p95: float = 0.0
    total_p50: float = 0.0
    total_p95: float = 0.0
    mean_prompt_tokens: float = 0.0
    mean_completion_tokens: float = 0.0
    mean_total_tokens: float = 0.0


def _percentile(values: list[float], p: float) -> float:
    """Linear-interpolated percentile (p in 0..100). Empty -> 0."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (p / 100)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - rank) + ordered[high] * (rank - low)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(config: str, records: list[Record]) -> Summary:
    """Roll one config's per-question records into its headline numbers."""
    if not records:
        return Summary(config=config, n=0, j_overall=0.0)

    by_category: dict[str, list[bool]] = {}
    for r in records:
        by_category.setdefault(r.category, []).append(r.correct)

    return Summary(
        config=config,
        n=len(records),
        j_overall=_mean([1.0 if r.correct else 0.0 for r in records]),
        j_by_category={
            cat: _mean([1.0 if c else 0.0 for c in flags]) for cat, flags in by_category.items()
        },
        search_p50=_percentile([r.search_ms for r in records], 50),
        search_p95=_percentile([r.search_ms for r in records], 95),
        total_p50=_percentile([r.total_ms for r in records], 50),
        total_p95=_percentile([r.total_ms for r in records], 95),
        mean_prompt_tokens=_mean([r.prompt_tokens for r in records]),
        mean_completion_tokens=_mean([r.completion_tokens for r in records]),
        mean_total_tokens=_mean([r.prompt_tokens + r.completion_tokens for r in records]),
    )