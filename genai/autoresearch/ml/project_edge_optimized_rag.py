from __future__ import annotations

from .common import (
    ProjectArtifact,
    bm25_rank_documents,
    load_hf_dataset_records,
    make_result,
    pick_best_result,
    retrieval_metrics,
    timed_call,
    tfidf_rank_documents,
    hashing_rank_documents,
)


PROJECT_ID = "edge_optimized_rag"
TITLE = "Edge-Optimized RAG with Custom Vector Implementations"
REQUESTED_DATASET = "squad_v2"


def _synthetic_examples(total: int) -> list[dict[str, str]]:
    base = [
        {
            "question": "Which planet is known as the Red Planet?",
            "context": "Mars is often called the Red Planet because iron oxide on its surface gives it a reddish appearance.",
            "answer": "Mars",
        },
        {
            "question": "What gas do plants absorb during photosynthesis?",
            "context": "During photosynthesis, plants absorb carbon dioxide and release oxygen into the atmosphere.",
            "answer": "carbon dioxide",
        },
        {
            "question": "Who wrote the play Hamlet?",
            "context": "Hamlet is a tragedy written by William Shakespeare sometime between 1599 and 1601.",
            "answer": "William Shakespeare",
        },
        {
            "question": "What is the largest mammal on Earth?",
            "context": "The blue whale is the largest animal known to have ever existed, reaching over 30 meters in length.",
            "answer": "blue whale",
        },
    ]
    return [base[index % len(base)].copy() for index in range(total)]


def _load_examples(quick: bool) -> tuple[list[dict[str, str]], str]:
    raw = load_hf_dataset_records("squad_v2", split="validation[:80]" if quick else "validation[:160]")
    parsed = []
    if raw:
        for record in raw:
            answers = record.get("answers") or {}
            texts = answers.get("text") or []
            if not texts:
                continue
            parsed.append(
                {
                    "question": str(record.get("question", "")).strip(),
                    "context": str(record.get("context", "")).strip(),
                    "answer": str(texts[0]).strip(),
                }
            )
        if len(parsed) >= 24:
            return parsed, "squad_v2"
    return _synthetic_examples(32 if quick else 64), "synthetic_qa_proxy"


def run(*, quick: bool = True) -> ProjectArtifact:
    mode = "quick" if quick else "full"
    examples, used_dataset = _load_examples(quick)
    documents = [example["context"] for example in examples]
    queries = [example["question"] for example in examples]
    gold_indices = list(range(len(examples)))

    experiments = [
        (
            "tfidf_bigrams",
            "tfidf",
            "word_bigrams",
            lambda: tfidf_rank_documents(queries, documents, top_k=3, analyzer="word", ngram_range=(1, 2), max_features=5000),
        ),
        (
            "tfidf_latent_svd",
            "tfidf",
            "latent_semantic_projection",
            lambda: tfidf_rank_documents(queries, documents, top_k=3, analyzer="word", ngram_range=(1, 2), max_features=5000, use_svd=True),
        ),
        (
            "bm25_token_search",
            "bm25",
            "bm25_sparse_index",
            lambda: bm25_rank_documents(queries, documents, top_k=3),
        ),
        (
            "hashing_vector_index",
            "hashing",
            "fixed_width_sparse_vectors",
            lambda: hashing_rank_documents(queries, documents, top_k=3),
        ),
    ]

    results = []
    for variant, algorithm, feature_set, ranker in experiments:
        rankings, runtime = timed_call(ranker)
        metrics = retrieval_metrics(rankings, gold_indices, top_k=3)
        results.append(
            make_result(
                project_id=PROJECT_ID,
                project_name=TITLE,
                requested_dataset=REQUESTED_DATASET,
                used_dataset=used_dataset,
                mode=mode,
                variant=variant,
                algorithm=algorithm,
                feature_set=feature_set,
                metric_name="recall_at_1",
                metric_value=metrics["recall_at_1"],
                metric_direction="higher_is_better",
                secondary_metric_name="mrr",
                secondary_metric_value=metrics["mrr"],
                train_samples=len(documents),
                eval_samples=len(queries),
                runtime_sec=runtime,
                notes="Ranks the gold answer context against the rest of the corpus.",
            )
        )

    best = pick_best_result(results)
    findings = [
        f"The strongest retrieval stack was {best.variant} with recall@1 {best.metric_value:.3f}." if best else "No successful run was recorded.",
        "Sparse lexical methods are attractive on edge devices because they avoid ANN servers and still recover relevant contexts on small corpora.",
        "Hashing vectors trade a bit of retrieval quality for predictable memory use and fixed-width indexing.",
    ]
    return ProjectArtifact(
        project_id=PROJECT_ID,
        title=TITLE,
        objective="Compare lightweight retrieval algorithms for question answering without external vector databases.",
        requested_dataset=REQUESTED_DATASET,
        used_dataset=used_dataset,
        mode=mode,
        metric_name="recall_at_1",
        metric_direction="higher_is_better",
        results=results,
        findings=findings,
        notes="The benchmark uses SQuAD contexts when available and otherwise falls back to a compact synthetic QA corpus. The task is retrieval-only rather than end-to-end generation.",
    )


if __name__ == "__main__":
    artifact = run()
    best = pick_best_result(artifact.results)
    print(f"{artifact.title}: {best.variant if best else 'no result'}")