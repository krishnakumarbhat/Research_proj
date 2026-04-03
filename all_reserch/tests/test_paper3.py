from research_suite.paper3_chunking.chunkers import build_chunks, fixed_chunks, graph_context_chunks
from research_suite.paper3_chunking.dataset import build_dataset
from research_suite.paper3_chunking.experiment import evaluate_configuration, redundancy_ratio


def test_paper3_sentence_chunking_runs() -> None:
    documents, queries = build_dataset(quick=True)
    summary, _ = evaluate_configuration(
        documents=documents,
        queries=queries[:12],
        algorithm="sentence_boundary",
        feature_name="word_tfidf",
        params={"max_tokens": 28},
        phase="test",
    )
    assert 0.0 <= float(summary["retrievable_rate"]) <= 1.0
    assert int(summary["chunk_count"]) > 0


def test_paper3_fixed_no_overlap_chunks() -> None:
    documents, _ = build_dataset(quick=True)
    chunks = build_chunks(documents, algorithm="fixed_no_overlap", params={"token_limit": 40})
    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.algorithm == "fixed_no_overlap"
        assert len(chunk.sent_indices) >= 1


def test_paper3_fixed_overlap_has_redundancy() -> None:
    documents, _ = build_dataset(quick=True)
    chunks_no_overlap = build_chunks(documents, algorithm="fixed_no_overlap", params={"token_limit": 28})
    chunks_overlap = build_chunks(documents, algorithm="fixed_overlap", params={"token_limit": 28, "overlap": 8})
    assert len(chunks_overlap) >= len(chunks_no_overlap)
    assert redundancy_ratio(chunks_overlap, documents) >= redundancy_ratio(chunks_no_overlap, documents)


def test_paper3_late_chunking_has_context() -> None:
    documents, _ = build_dataset(quick=True)
    chunks = build_chunks(documents, algorithm="late_chunking_proxy", params={"max_tokens": 40})
    for chunk in chunks:
        assert chunk.embedding_text != chunk.text
        assert len(chunk.embedding_text) > len(chunk.text)


def test_paper3_graph_rag_connects_entities() -> None:
    documents, _ = build_dataset(quick=True)
    chunks = build_chunks(documents, algorithm="graph_rag", params={"hops": 1})
    assert len(chunks) > 0
    multi_sentence = [c for c in chunks if len(c.sent_indices) > 1]
    assert len(multi_sentence) > 0


def test_paper3_all_algorithms_run() -> None:
    documents, queries = build_dataset(quick=True)
    for algo, params in [
        ("fixed_no_overlap", {"token_limit": 28}),
        ("fixed_overlap", {"token_limit": 28, "overlap": 8}),
        ("sentence_boundary", {"max_tokens": 28}),
        ("semantic_boundary", {"max_tokens": 28, "threshold": 0.15}),
        ("late_chunking_proxy", {"max_tokens": 28}),
        ("graph_rag", {"hops": 1}),
    ]:
        summary, _ = evaluate_configuration(
            documents=documents,
            queries=queries[:6],
            algorithm=algo,
            feature_name="word_tfidf",
            params=params,
            phase="test",
        )
        assert int(summary["chunk_count"]) > 0, f"{algo} produced no chunks"


def test_paper3_dataset_sizes() -> None:
    docs_q, queries_q = build_dataset(quick=True)
    docs_f, queries_f = build_dataset(quick=False)
    assert len(docs_q) == 6
    assert len(docs_f) == 12
    assert len(queries_q) == 42
    assert len(queries_f) == 84