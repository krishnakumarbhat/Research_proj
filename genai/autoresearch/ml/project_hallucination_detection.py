from __future__ import annotations

import random

from sklearn.model_selection import train_test_split

from .common import ProjectArtifact, concat_fields, evaluate_text_classifier, make_result, pick_best_result, set_global_seed, timed_call


PROJECT_ID = "hallucination_detection"
TITLE = "Hallucination Detection via Cross-Encoder Proxy"
REQUESTED_DATASET = "McGill-NLP/FaithDial or vectara/huggescript"


def _synthetic_records(total: int) -> list[dict[str, str]]:
    facts = [
        {
            "source": "Source: France's capital city is Paris, and the Seine River passes through Paris.",
            "grounded": [
                "Response: Paris is the capital of France and the Seine runs through the city.",
                "Response: France's capital is Paris.",
                "Response: The French capital is Paris, where the Seine also flows.",
            ],
            "hallucinated": [
                "Response: Paris is the capital of France and is home to the Louvre Museum.",
                "Response: France's capital is Paris, a city on the Seine with millions of residents.",
                "Response: The French capital is Paris and it hosts one of Europe's busiest rail hubs.",
            ],
        },
        {
            "source": "Source: Italy's capital city is Rome, and Rome is located in the Lazio region.",
            "grounded": [
                "Response: Rome is Italy's capital and it lies in Lazio.",
                "Response: Italy's capital city is Rome.",
                "Response: The Italian capital is Rome, in the Lazio region.",
            ],
            "hallucinated": [
                "Response: Rome is Italy's capital and is known for the Colosseum.",
                "Response: Italy's capital city is Rome, a major cultural center in southern Europe.",
                "Response: The Italian capital is Rome and it attracts millions of visitors each year.",
            ],
        },
        {
            "source": "Source: Mercury is the closest planet to the Sun and has no substantial atmosphere.",
            "grounded": [
                "Response: Mercury is nearest to the Sun and lacks a substantial atmosphere.",
                "Response: The closest planet to the Sun is Mercury.",
                "Response: Mercury orbits closest to the Sun.",
            ],
            "hallucinated": [
                "Response: Mercury is nearest to the Sun and completes an orbit in about 88 days.",
                "Response: The closest planet to the Sun is Mercury, which has a large iron core.",
                "Response: Mercury orbits closest to the Sun and shows extreme temperature swings.",
            ],
        },
        {
            "source": "Source: Venus is the second planet from the Sun and rotates more slowly than Earth.",
            "grounded": [
                "Response: Venus is second from the Sun and spins more slowly than Earth.",
                "Response: The second planet from the Sun is Venus.",
                "Response: Venus comes after Mercury in the solar system.",
            ],
            "hallucinated": [
                "Response: Venus is the second planet from the Sun and is often called Earth's sister planet.",
                "Response: The second planet from the Sun is Venus, which has a dense carbon-dioxide atmosphere.",
                "Response: Venus comes after Mercury and appears as one of the brightest objects in the night sky.",
            ],
        },
        {
            "source": "Source: The Pacific Ocean is the largest ocean on Earth and spans between Asia, Australia, and the Americas.",
            "grounded": [
                "Response: The Pacific is Earth's largest ocean and lies between Asia, Australia, and the Americas.",
                "Response: The largest ocean on Earth is the Pacific Ocean.",
                "Response: The Pacific is the biggest ocean basin on the planet.",
            ],
            "hallucinated": [
                "Response: The Pacific is Earth's largest ocean and includes the Mariana Trench.",
                "Response: The largest ocean on Earth is the Pacific Ocean, which covers more area than all land combined.",
                "Response: The Pacific is the biggest ocean basin and touches many volcanic island chains.",
            ],
        },
        {
            "source": "Source: The Atlantic Ocean separates the Americas from Europe and Africa.",
            "grounded": [
                "Response: The Atlantic lies between the Americas and Europe and Africa.",
                "Response: The Atlantic Ocean separates the Americas from Europe and Africa.",
                "Response: Europe and Africa are across the Atlantic from the Americas.",
            ],
            "hallucinated": [
                "Response: The Atlantic Ocean separates the Americas from Europe and Africa and includes the Sargasso Sea.",
                "Response: The Atlantic lies between the Americas and Europe and Africa and connects to the Mediterranean through the Strait of Gibraltar.",
                "Response: Europe and Africa are across the Atlantic from the Americas, and the basin supports major shipping routes.",
            ],
        },
    ]
    rng = random.Random(42)
    rows = []
    for index in range(total):
        fact = facts[index % len(facts)]
        label = "grounded" if rng.random() < 0.55 else "hallucinated"
        candidates = fact["grounded"] if label == "grounded" else fact["hallucinated"]
        response = candidates[index % len(candidates)]
        rows.append({"response": response, "pair": concat_fields(fact["source"], response), "label": label})
    return rows


def run(*, quick: bool = True) -> ProjectArtifact:
    set_global_seed(42)
    mode = "quick" if quick else "full"
    records = _synthetic_records(72 if quick else 144)
    labels = [record["label"] for record in records]
    response_texts = [record["response"] for record in records]
    pair_texts = [record["pair"] for record in records]

    response_train, response_test, pair_train, pair_test, y_train, y_test = train_test_split(
        response_texts,
        pair_texts,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )

    experiments = [
        ("response_only_logreg", "logreg", "response_only", response_train, response_test),
        ("pair_logreg", "logreg", "source_plus_response", pair_train, pair_test),
        ("pair_svm", "linear_svm", "source_plus_response", pair_train, pair_test),
        ("pair_nb", "nb", "source_plus_response", pair_train, pair_test),
    ]

    results = []
    for variant, algorithm, feature_set, train_texts, eval_texts in experiments:
        metrics, runtime = timed_call(
            evaluate_text_classifier,
            train_texts,
            y_train,
            eval_texts,
            y_test,
            algorithm=algorithm,
            analyzer="word",
            ngram_range=(1, 2),
            max_features=5000,
        )
        results.append(
            make_result(
                project_id=PROJECT_ID,
                project_name=TITLE,
                requested_dataset=REQUESTED_DATASET,
                used_dataset="synthetic_faithdial_proxy",
                mode=mode,
                variant=variant,
                algorithm=algorithm,
                feature_set=feature_set,
                metric_name="macro_f1",
                metric_value=metrics["macro_f1"],
                metric_direction="higher_is_better",
                secondary_metric_name="accuracy",
                secondary_metric_value=metrics["accuracy"],
                train_samples=len(y_train),
                eval_samples=len(y_test),
                runtime_sec=runtime,
                notes="Grounding-classification proxy over source/response pairs.",
            )
        )

    best = pick_best_result(results)
    response_only = next(result for result in results if result.feature_set == "response_only")
    pair_best = max((result for result in results if result.feature_set == "source_plus_response"), key=lambda item: item.metric_value)
    if pair_best.metric_value > response_only.metric_value:
        grounding_statement = f"Adding the source context improved macro F1 from {response_only.metric_value:.3f} to {pair_best.metric_value:.3f}."
    elif pair_best.metric_value < response_only.metric_value:
        grounding_statement = f"The response-only baseline outperformed the paired models on this synthetic corpus ({response_only.metric_value:.3f} versus {pair_best.metric_value:.3f})."
    else:
        grounding_statement = f"Response-only and source-plus-response models tied at macro F1 {response_only.metric_value:.3f} on this easy synthetic grounding task."
    findings = [
        f"The strongest hallucination detector was {best.variant} with macro F1 {best.metric_value:.3f}." if best else "No successful run was recorded.",
        grounding_statement,
        "This is a lightweight cross-encoder proxy implemented with sparse lexical models rather than a transformer pair scorer.",
    ]
    return ProjectArtifact(
        project_id=PROJECT_ID,
        title=TITLE,
        objective="Detect grounded versus hallucinated responses using lightweight pairwise text models.",
        requested_dataset=REQUESTED_DATASET,
        used_dataset="synthetic_faithdial_proxy",
        mode=mode,
        metric_name="macro_f1",
        metric_direction="higher_is_better",
        results=results,
        findings=findings,
        notes="A synthetic grounded-dialogue corpus is used here so the task remains fully runnable without a large cross-encoder model or a gated dialogue dataset.",
    )


if __name__ == "__main__":
    artifact = run()
    best = pick_best_result(artifact.results)
    print(f"{artifact.title}: {best.variant if best else 'no result'}")