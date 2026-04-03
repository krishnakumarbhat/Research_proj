from research_suite.paper2_dynamic_rag.dataset import build_dataset
from research_suite.paper2_dynamic_rag.experiment import evaluate_stream
from research_suite.paper2_dynamic_rag.retrievers import (
    BaseRetriever,
    FrequencyWeightedRetriever,
    GraphRetriever,
    SemanticCacheRetriever,
)


def test_paper2_vanilla_stream_runs() -> None:
    documents, queries = build_dataset(quick=True, seed=7)
    summary, _ = evaluate_stream(
        documents=documents,
        queries=queries[:24],
        algorithm="vanilla",
        feature_name="word_tfidf",
        params={},
        phase="test",
    )
    assert 0.0 <= float(summary["top1"]) <= 1.0
    assert float(summary["avg_candidates"]) == len(documents)


def test_paper2_semantic_cache_hits() -> None:
    documents, queries = build_dataset(quick=True, seed=7)
    summary, _ = evaluate_stream(
        documents=documents,
        queries=queries[:48],
        algorithm="semantic_cache",
        feature_name="word_tfidf",
        params={"threshold": 0.80},
        phase="test",
    )
    assert 0.0 <= float(summary["top1"]) <= 1.0
    assert float(summary["cache_hit_rate"]) >= 0.0


def test_paper2_frequency_weighted_runs() -> None:
    documents, queries = build_dataset(quick=True, seed=7)
    summary, _ = evaluate_stream(
        documents=documents,
        queries=queries[:24],
        algorithm="frequency_weighted",
        feature_name="char_tfidf",
        params={"alpha": 0.85, "beta": 0.15, "hot_k": 8, "similarity_floor": 0.08},
        phase="test",
    )
    assert 0.0 <= float(summary["top1"]) <= 1.0
    assert float(summary["avg_candidates"]) <= len(documents)


def test_paper2_graph_assisted_runs() -> None:
    documents, queries = build_dataset(quick=True, seed=7)
    summary, _ = evaluate_stream(
        documents=documents,
        queries=queries[:24],
        algorithm="graph_assisted",
        feature_name="lsa",
        params={
            "alpha": 0.85, "beta": 0.15, "hot_k": 8,
            "similarity_floor": 0.08, "decay_lambda": 0.01, "gamma": 0.3,
        },
        phase="test",
    )
    assert 0.0 <= float(summary["top1"]) <= 1.0


def test_paper2_dataset_sizes() -> None:
    docs_quick, queries_quick = build_dataset(quick=True, seed=42)
    docs_full, queries_full = build_dataset(quick=False, seed=42)
    assert len(docs_quick) == 30  # 18 projects + 12 labs
    assert len(docs_full) == 48   # 36 projects + 12 labs
    assert len(queries_quick) == 180
    assert len(queries_full) == 540
