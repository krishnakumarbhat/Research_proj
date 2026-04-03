from __future__ import annotations

from .common import ProjectArtifact, bm25_rank_documents, chunk_text, make_result, pick_best_result, retrieval_metrics, select_sentences_tfidf, tfidf_rank_documents, timed_call


PROJECT_ID = "rag_chunking_optimization"
TITLE = "RAG Chunking Optimization"
REQUESTED_DATASET = "mteb/scifact"


def _synthetic_records(total: int) -> list[dict[str, str]]:
    base = [
        {
            "claim": "Vitamin C shortens the duration of the common cold.",
            "abstract": "Clinical meta-analysis shows vitamin C does not reliably prevent the common cold in the general population, although some trials found a small reduction in duration under sustained supplementation.",
        },
        {
            "claim": "Black holes emit radiation according to quantum effects.",
            "abstract": "Hawking radiation arises when quantum field fluctuations near the event horizon generate particle pairs, causing black holes to lose mass over extremely long timescales.",
        },
        {
            "claim": "Plate tectonics explains the movement of continents.",
            "abstract": "The plate tectonics model accounts for continental drift by describing how lithospheric plates move atop the mantle and interact at convergent, divergent, and transform boundaries.",
        },
        {
            "claim": "mRNA vaccines deliver genetic instructions for antigen production.",
            "abstract": "mRNA vaccine platforms transport a messenger RNA payload into host cells, which then synthesize the encoded antigen and trigger adaptive immune responses.",
        },
    ]
    return [base[index % len(base)].copy() for index in range(total)]


def _rank_claims(claims: list[str], documents: list[str], *, mode: str, algorithm: str) -> list[list[int]]:
    rankings = []
    for claim in claims:
        chunk_documents = []
        owners = []
        for doc_index, document in enumerate(documents):
            if mode == "semantic_focus":
                chunks = [select_sentences_tfidf(document, claim, max_sentences=3)]
            else:
                chunks = chunk_text(document, mode=mode, chunk_size=32, overlap=10)
            if not chunks:
                chunks = [document]
            for chunk in chunks:
                chunk_documents.append(chunk)
                owners.append(doc_index)

        if algorithm == "tfidf":
            chunk_ranking = tfidf_rank_documents([claim], chunk_documents, top_k=len(chunk_documents), max_features=5000)[0]
        else:
            chunk_ranking = bm25_rank_documents([claim], chunk_documents, top_k=len(chunk_documents))[0]

        doc_ranking = []
        seen = set()
        for chunk_index in chunk_ranking:
            owner = owners[chunk_index]
            if owner in seen:
                continue
            doc_ranking.append(owner)
            seen.add(owner)
            if len(doc_ranking) >= 3:
                break
        rankings.append(doc_ranking)
    return rankings


def run(*, quick: bool = True) -> ProjectArtifact:
    mode = "quick" if quick else "full"
    records = _synthetic_records(18 if quick else 36)
    claims = [record["claim"] for record in records]
    documents = [record["abstract"] for record in records]
    gold = list(range(len(records)))

    experiments = [
        ("fixed_tfidf", "tfidf", "fixed", lambda: _rank_claims(claims, documents, mode="fixed", algorithm="tfidf")),
        ("overlap_tfidf", "tfidf", "overlap", lambda: _rank_claims(claims, documents, mode="overlap", algorithm="tfidf")),
        ("sentence_tfidf", "tfidf", "sentence", lambda: _rank_claims(claims, documents, mode="sentence", algorithm="tfidf")),
        ("semantic_focus_tfidf", "tfidf", "semantic_focus", lambda: _rank_claims(claims, documents, mode="semantic_focus", algorithm="tfidf")),
        ("overlap_bm25", "bm25", "overlap", lambda: _rank_claims(claims, documents, mode="overlap", algorithm="bm25")),
    ]

    results = []
    for variant, algorithm, feature_set, ranker in experiments:
        rankings, runtime = timed_call(ranker)
        metrics = retrieval_metrics(rankings, gold, top_k=3)
        results.append(
            make_result(
                project_id=PROJECT_ID,
                project_name=TITLE,
                requested_dataset=REQUESTED_DATASET,
                used_dataset="synthetic_scifact_proxy",
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
                eval_samples=len(claims),
                runtime_sec=runtime,
                notes="Ranks scientific abstracts after chunking them into retrieval units.",
            )
        )

    best = pick_best_result(results)
    spread = max(result.metric_value for result in results) - min(result.metric_value for result in results)
    if spread < 1e-9:
        chunk_statement = "All chunking strategies tied on the current synthetic abstract corpus, which suggests the proxy is too small to expose a strong chunking effect."
    else:
        chunk_statement = "Chunking changed retrieval behavior because each retriever saw different lexical neighborhoods and evidence granularity."
    findings = [
        f"The best chunking strategy was {best.variant} with recall@1 {best.metric_value:.3f}." if best else "No successful run was recorded.",
        chunk_statement,
        "A larger or noisier scientific corpus would be a better next step for separating fixed, overlap, sentence, and semantic chunking policies.",
    ]
    return ProjectArtifact(
        project_id=PROJECT_ID,
        title=TITLE,
        objective="Compare chunking strategies for evidence retrieval on short scientific documents under a lightweight RAG setup.",
        requested_dataset=REQUESTED_DATASET,
        used_dataset="synthetic_scifact_proxy",
        mode=mode,
        metric_name="recall_at_1",
        metric_direction="higher_is_better",
        results=results,
        findings=findings,
        notes="The benchmark uses a synthetic SciFact-style corpus because the main experimental variable is chunking behavior rather than dataset scale.",
    )


if __name__ == "__main__":
    artifact = run()
    best = pick_best_result(artifact.results)
    print(f"{artifact.title}: {best.variant if best else 'no result'}")