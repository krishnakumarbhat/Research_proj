from __future__ import annotations

from pathlib import Path

from research_suite.common.io_utils import ensure_directory, write_csv, write_json
from research_suite.common.metrics import hit_at_k, mean, reciprocal_rank
from research_suite.common.vectorizers import available_feature_models
from research_suite.paper2_dynamic_rag.dataset import RagQuery, build_dataset
from research_suite.paper2_dynamic_rag.retrievers import (
    BaseRetriever,
    FrequencyWeightedRetriever,
    GraphRetriever,
    SemanticCacheRetriever,
)


def coverage_at_k(relevant_ids: tuple[str, ...], ranked_ids: list[str], k: int) -> float:
    relevant = set(relevant_ids)
    retrieved = set(ranked_ids[:k])
    return len(relevant & retrieved) / len(relevant)


def build_retriever(algorithm: str, feature_name: str, params: dict[str, object]) -> BaseRetriever:
    if algorithm == "vanilla":
        return BaseRetriever(feature_name=feature_name)
    if algorithm == "semantic_cache":
        return SemanticCacheRetriever(feature_name=feature_name, threshold=float(params["threshold"]))
    if algorithm == "frequency_weighted":
        return FrequencyWeightedRetriever(
            feature_name=feature_name,
            alpha=float(params["alpha"]),
            beta=float(params["beta"]),
            hot_k=int(params["hot_k"]),
            similarity_floor=float(params["similarity_floor"]),
        )
    if algorithm == "frequency_decay":
        return FrequencyWeightedRetriever(
            feature_name=feature_name,
            alpha=float(params["alpha"]),
            beta=float(params["beta"]),
            hot_k=int(params["hot_k"]),
            similarity_floor=float(params["similarity_floor"]),
            decay_lambda=float(params["decay_lambda"]),
        )
    if algorithm == "graph_assisted":
        return GraphRetriever(
            feature_name=feature_name,
            alpha=float(params["alpha"]),
            beta=float(params["beta"]),
            hot_k=int(params["hot_k"]),
            similarity_floor=float(params["similarity_floor"]),
            gamma=float(params["gamma"]),
            decay_lambda=float(params.get("decay_lambda", 0.0)),
        )
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def evaluate_stream(
    documents: list,
    queries: list[RagQuery],
    algorithm: str,
    feature_name: str,
    params: dict[str, object],
    phase: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    retriever = build_retriever(algorithm=algorithm, feature_name=feature_name, params=params)
    retriever.fit(documents)
    query_rows: list[dict[str, object]] = []
    top1_scores: list[float] = []
    top3_scores: list[float] = []
    mrr_scores: list[float] = []
    coverage_scores: list[float] = []
    latencies: list[float] = []
    candidate_counts: list[float] = []
    cache_hits: list[float] = []

    for step, query in enumerate(queries, start=1):
        result = retriever.retrieve(query.text, step=step)
        relevant = set(query.relevant_ids)
        top1 = hit_at_k(relevant, result.ranked_ids, 1)
        top3 = hit_at_k(relevant, result.ranked_ids, 3)
        mrr = reciprocal_rank(relevant, result.ranked_ids)
        coverage = coverage_at_k(query.relevant_ids, result.ranked_ids, 3)
        query_rows.append(
            {
                "paper": "paper2",
                "phase": phase,
                "algorithm": algorithm,
                "feature_model": feature_name,
                "params": params,
                "query_id": query.query_id,
                "family_id": query.family_id,
                "query_type": query.query_type,
                "top1": top1,
                "top3": top3,
                "mrr": round(mrr, 6),
                "coverage_at_3": round(coverage, 6),
                "candidates_scored": result.candidates_scored,
                "latency_ms": round(result.latency_ms, 4),
                "cache_hit": result.cache_hit,
            }
        )
        top1_scores.append(top1)
        top3_scores.append(top3)
        mrr_scores.append(mrr)
        coverage_scores.append(coverage)
        latencies.append(result.latency_ms)
        candidate_counts.append(float(result.candidates_scored))
        cache_hits.append(float(result.cache_hit))
        retriever.update_state(
            relevant_ids=query.relevant_ids,
            step=step,
            query_vector=result.query_vector,
            ranked_ids=result.ranked_ids,
        )

    summary = {
        "paper": "paper2",
        "phase": phase,
        "algorithm": algorithm,
        "feature_model": feature_name,
        "params": params,
        "top1": round(mean(top1_scores), 6),
        "top3": round(mean(top3_scores), 6),
        "mrr": round(mean(mrr_scores), 6),
        "coverage_at_3": round(mean(coverage_scores), 6),
        "avg_candidates": round(mean(candidate_counts), 4),
        "avg_latency_ms": round(mean(latencies), 4),
        "cache_hit_rate": round(mean(cache_hits), 6),
    }
    return summary, query_rows


def objective(summary: dict[str, object]) -> float:
    return (
        (2.0 * float(summary["top1"]))
        + float(summary["coverage_at_3"])
        + float(summary["mrr"])
        + (0.20 * float(summary["cache_hit_rate"]))
        - (0.002 * float(summary["avg_latency_ms"]))
        - (0.0008 * float(summary["avg_candidates"]))
    )


def parameter_grid(algorithm: str, quick: bool) -> list[dict[str, object]]:
    if algorithm == "vanilla":
        return [{}]
    if algorithm == "semantic_cache":
        return [{"threshold": threshold} for threshold in ([0.80, 0.88] if quick else [0.78, 0.84, 0.90])]
    if algorithm == "frequency_weighted":
        betas = [0.15, 0.3] if quick else [0.15, 0.3, 0.45]
        hot_values = [8, 16] if quick else [8, 16, 24]
        return [
            {
                "alpha": alpha,
                "beta": beta,
                "hot_k": hot_k,
                "similarity_floor": floor,
            }
            for alpha in [0.75, 0.85]
            for beta in betas
            for hot_k in hot_values
            for floor in [0.08, 0.14]
        ]
    if algorithm == "frequency_decay":
        return [
            {
                **params,
                "decay_lambda": decay,
            }
            for params in parameter_grid("frequency_weighted", quick=quick)
            for decay in ([0.01] if quick else [0.005, 0.015])
        ]
    if algorithm == "graph_assisted":
        return [
            {
                **params,
                "gamma": gamma,
            }
            for params in parameter_grid("frequency_decay", quick=quick)
            for gamma in ([0.18, 0.3] if quick else [0.12, 0.25, 0.4])
        ]
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def run(output_root: str | Path, quick: bool = False, seed: int = 42) -> dict[str, object]:
    output_dir = ensure_directory(Path(output_root) / "paper2_dynamic_rag")
    documents, queries = build_dataset(quick=quick, seed=seed)
    split_index = max(1, int(len(queries) * 0.35))
    validation_queries = queries[:split_index]
    test_queries = queries[split_index:]

    algorithms = ["vanilla", "semantic_cache", "frequency_weighted", "frequency_decay", "graph_assisted"]
    validation_rows: list[dict[str, object]] = []
    best_choices: dict[tuple[str, str], dict[str, object]] = {}

    for algorithm in algorithms:
        for feature_name in available_feature_models():
            best_score = float("-inf")
            best_summary = None
            for params in parameter_grid(algorithm=algorithm, quick=quick):
                summary, _ = evaluate_stream(
                    documents=documents,
                    queries=validation_queries,
                    algorithm=algorithm,
                    feature_name=feature_name,
                    params=params,
                    phase="validation",
                )
                summary["objective"] = round(objective(summary), 6)
                validation_rows.append(summary)
                if summary["objective"] > best_score:
                    best_score = summary["objective"]
                    best_summary = summary
            best_choices[(algorithm, feature_name)] = best_summary

    test_rows: list[dict[str, object]] = []
    per_query_rows: list[dict[str, object]] = []
    for (algorithm, feature_name), best_summary in best_choices.items():
        summary, query_rows = evaluate_stream(
            documents=documents,
            queries=test_queries,
            algorithm=algorithm,
            feature_name=feature_name,
            params=dict(best_summary["params"]),
            phase="test",
        )
        summary["objective"] = round(objective(summary), 6)
        test_rows.append(summary)
        per_query_rows.extend(query_rows)

    validation_rows.sort(key=lambda row: row["objective"], reverse=True)
    test_rows.sort(key=lambda row: row["objective"], reverse=True)
    write_csv(output_dir / "validation_search.csv", validation_rows)
    write_csv(output_dir / "test_results.csv", test_rows)
    write_csv(output_dir / "test_query_level.csv", per_query_rows)
    write_json(
        output_dir / "metadata.json",
        {
            "paper": "paper2",
            "quick": quick,
            "seed": seed,
            "document_count": len(documents),
            "query_count": len(queries),
            "best_test_config": test_rows[0],
        },
    )
    return {
        "paper": "paper2",
        "validation_rows": validation_rows,
        "test_rows": test_rows,
        "best_config": test_rows[0],
        "document_count": len(documents),
        "query_count": len(queries),
    }