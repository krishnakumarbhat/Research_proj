from __future__ import annotations

from typing import Sequence


def levenshtein_distance(source: Sequence[str], target: Sequence[str]) -> int:
    if source == target:
        return 0
    if not source:
        return len(target)
    if not target:
        return len(source)

    previous = list(range(len(target) + 1))
    for row_index, source_item in enumerate(source, start=1):
        current = [row_index]
        for column_index, target_item in enumerate(target, start=1):
            substitution_cost = 0 if source_item == target_item else 1
            current.append(
                min(
                    previous[column_index] + 1,
                    current[column_index - 1] + 1,
                    previous[column_index - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def cer(reference: str, hypothesis: str) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return levenshtein_distance(list(reference), list(hypothesis)) / len(reference)


def wer(reference: str, hypothesis: str) -> float:
    reference_tokens = reference.split()
    hypothesis_tokens = hypothesis.split()
    if not reference_tokens:
        return 0.0 if not hypothesis_tokens else 1.0
    return levenshtein_distance(reference_tokens, hypothesis_tokens) / len(reference_tokens)


def hit_at_k(relevant_ids: set[str], ranked_ids: Sequence[str], k: int) -> float:
    top_ids = ranked_ids[:k]
    return 1.0 if any(doc_id in relevant_ids for doc_id in top_ids) else 0.0


def reciprocal_rank(relevant_ids: set[str], ranked_ids: Sequence[str]) -> float:
    for index, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / index
    return 0.0


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
