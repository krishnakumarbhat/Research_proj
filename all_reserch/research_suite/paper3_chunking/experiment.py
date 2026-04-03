from __future__ import annotations

from pathlib import Path

import numpy as np

from research_suite.common.io_utils import ensure_directory, write_csv, write_json
from research_suite.common.metrics import hit_at_k, mean, reciprocal_rank
from research_suite.common.text import token_count
from research_suite.common.vectorizers import available_feature_models, build_feature_model
from research_suite.paper3_chunking.chunkers import ChunkRecord, build_chunks
from research_suite.paper3_chunking.dataset import ChunkingQuery, build_dataset


def relevant_chunk_ids(chunks: list[ChunkRecord], query: ChunkingQuery) -> set[str]:
    relevant = set(query.support_sentences)
    return {
        chunk.chunk_id
        for chunk in chunks
        if chunk.doc_id == query.doc_id and relevant.issubset(set(chunk.sent_indices))
    }


def redundancy_ratio(chunks: list[ChunkRecord], original_documents: list) -> float:
    chunk_tokens = sum(token_count(chunk.text) for chunk in chunks)
    source_tokens = sum(token_count(document.text) for document in original_documents)
    return chunk_tokens / source_tokens if source_tokens else 0.0


def evaluate_configuration(
    documents: list,
    queries: list[ChunkingQuery],
    algorithm: str,
    feature_name: str,
    params: dict[str, object],
    phase: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    chunks = build_chunks(documents=documents, algorithm=algorithm, params=params)
    feature_model = build_feature_model(feature_name)
    embedding_texts = [chunk.embedding_text for chunk in chunks]
    feature_model.fit(embedding_texts)
    chunk_matrix = feature_model.transform(embedding_texts)
    chunk_ids = [chunk.chunk_id for chunk in chunks]

    top1_scores: list[float] = []
    top3_scores: list[float] = []
    mrr_scores: list[float] = []
    retrievable_flags: list[float] = []
    query_rows: list[dict[str, object]] = []

    for query in queries:
        relevant = relevant_chunk_ids(chunks, query)
        retrievable = 1.0 if relevant else 0.0
        query_vector = feature_model.transform([query.text])
        similarities = feature_model.similarity(query_vector, chunk_matrix)
        ranked_indices = np.argsort(similarities)[::-1]
        ranked_ids = [chunk_ids[index] for index in ranked_indices[:3]]
        top1 = hit_at_k(relevant, ranked_ids, 1) if relevant else 0.0
        top3 = hit_at_k(relevant, ranked_ids, 3) if relevant else 0.0
        mrr = reciprocal_rank(relevant, ranked_ids) if relevant else 0.0
        top1_scores.append(top1)
        top3_scores.append(top3)
        mrr_scores.append(mrr)
        retrievable_flags.append(retrievable)
        query_rows.append(
            {
                "paper": "paper3",
                "phase": phase,
                "algorithm": algorithm,
                "feature_model": feature_name,
                "params": params,
                "query_id": query.query_id,
                "query_type": query.query_type,
                "retrievable": retrievable,
                "top1": top1,
                "top3": top3,
                "mrr": round(mrr, 6),
            }
        )

    summary = {
        "paper": "paper3",
        "phase": phase,
        "algorithm": algorithm,
        "feature_model": feature_name,
        "params": params,
        "top1": round(mean(top1_scores), 6),
        "top3": round(mean(top3_scores), 6),
        "mrr": round(mean(mrr_scores), 6),
        "retrievable_rate": round(mean(retrievable_flags), 6),
        "redundancy_ratio": round(redundancy_ratio(chunks, documents), 6),
        "chunk_count": len(chunks),
        "avg_chunk_tokens": round(mean([token_count(chunk.text) for chunk in chunks]), 4),
    }
    return summary, query_rows


def objective(summary: dict[str, object]) -> float:
    return (
        (2.0 * float(summary["top1"]))
        + float(summary["retrievable_rate"])
        + float(summary["mrr"])
        - (0.45 * max(0.0, float(summary["redundancy_ratio"]) - 1.0))
        - (0.001 * float(summary["chunk_count"]))
    )


def parameter_grid(algorithm: str, quick: bool) -> list[dict[str, object]]:
    if algorithm == "fixed_no_overlap":
        return [{"token_limit": limit} for limit in ([28, 40] if quick else [28, 40, 52])]
    if algorithm == "fixed_overlap":
        return [
            {"token_limit": limit, "overlap": overlap}
            for limit, overlap in ([(28, 8), (40, 10)] if quick else [(28, 8), (40, 12), (52, 16)])
        ]
    if algorithm == "sentence_boundary":
        return [{"max_tokens": limit} for limit in ([28, 40] if quick else [28, 40, 52])]
    if algorithm == "semantic_boundary":
        thresholds = [0.12, 0.20] if quick else [0.10, 0.18, 0.26]
        limits = [28, 40] if quick else [28, 40, 52]
        return [{"max_tokens": limit, "threshold": threshold} for limit in limits for threshold in thresholds]
    if algorithm == "late_chunking_proxy":
        return [{"max_tokens": limit} for limit in ([28, 40] if quick else [28, 40, 52])]
    if algorithm == "graph_rag":
        return [{"hops": hop} for hop in ([1, 2] if quick else [1, 2, 3])]
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def run(output_root: str | Path, quick: bool = False) -> dict[str, object]:
    output_dir = ensure_directory(Path(output_root) / "paper3_chunking")
    documents, queries = build_dataset(quick=quick)
    split_index = max(1, int(len(queries) * 0.35))
    validation_queries = queries[:split_index]
    test_queries = queries[split_index:]

    algorithms = [
        "fixed_no_overlap",
        "fixed_overlap",
        "sentence_boundary",
        "semantic_boundary",
        "late_chunking_proxy",
        "graph_rag",
    ]
    validation_rows: list[dict[str, object]] = []
    best_choices: dict[tuple[str, str], dict[str, object]] = {}

    for algorithm in algorithms:
        for feature_name in available_feature_models():
            best_score = float("-inf")
            best_summary = None
            for params in parameter_grid(algorithm=algorithm, quick=quick):
                summary, _ = evaluate_configuration(
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
    query_rows: list[dict[str, object]] = []
    for (algorithm, feature_name), best_summary in best_choices.items():
        summary, rows = evaluate_configuration(
            documents=documents,
            queries=test_queries,
            algorithm=algorithm,
            feature_name=feature_name,
            params=dict(best_summary["params"]),
            phase="test",
        )
        summary["objective"] = round(objective(summary), 6)
        test_rows.append(summary)
        query_rows.extend(rows)

    validation_rows.sort(key=lambda row: row["objective"], reverse=True)
    test_rows.sort(key=lambda row: row["objective"], reverse=True)
    write_csv(output_dir / "validation_search.csv", validation_rows)
    write_csv(output_dir / "test_results.csv", test_rows)
    write_csv(output_dir / "test_query_level.csv", query_rows)
    write_json(
        output_dir / "metadata.json",
        {
            "paper": "paper3",
            "quick": quick,
            "document_count": len(documents),
            "query_count": len(queries),
            "best_test_config": test_rows[0],
        },
    )
    return {
        "paper": "paper3",
        "validation_rows": validation_rows,
        "test_rows": test_rows,
        "best_config": test_rows[0],
        "document_count": len(documents),
        "query_count": len(queries),
    }