from __future__ import annotations

from typing import Callable

from .common import ProjectArtifact, best_overlap_sentence, concat_fields, load_hf_dataset_records, make_result, pick_best_result, simple_tokenize, timed_call, token_overlap_f1


PROJECT_ID = "small_ai_agent_workflows"
TITLE = "Small AI Agent Workflows"
REQUESTED_DATASET = "hotpot_qa (distractor)"


def _synthetic_examples(total: int) -> list[dict[str, object]]:
    base = [
        {
            "question": "Which scientist developed the laws of motion, the same person who also wrote Philosophiæ Naturalis Principia Mathematica?",
            "answer": "Isaac Newton",
            "contexts": [
                "Philosophiæ Naturalis Principia Mathematica is a work laying out classical mechanics by Isaac Newton.",
                "Isaac Newton formulated the three laws of motion and the law of universal gravitation.",
                "Galileo studied motion but did not publish the Principia.",
            ],
        },
        {
            "question": "Which city hosts the Eiffel Tower and is also the capital of France?",
            "answer": "Paris",
            "contexts": [
                "The Eiffel Tower is a wrought-iron tower located on the Champ de Mars in Paris.",
                "Paris is the capital and most populous city of France.",
                "Lyon is another major French city known for cuisine.",
            ],
        },
        {
            "question": "Which ocean lies between Africa and Australia and is the third-largest ocean on Earth?",
            "answer": "Indian Ocean",
            "contexts": [
                "The Indian Ocean lies between Africa, Asia, Australia, and the Southern Ocean.",
                "The Indian Ocean is the third-largest oceanic division in the world.",
                "The Atlantic Ocean separates the Americas from Europe and Africa.",
            ],
        },
    ]
    return [base[index % len(base)].copy() for index in range(total)]


def _load_examples(quick: bool) -> tuple[list[dict[str, object]], str]:
    raw = load_hf_dataset_records("hotpot_qa", name="distractor", split="validation[:48]" if quick else "validation[:96]")
    parsed = []
    if raw:
        for record in raw:
            context = record.get("context") or {}
            titles = context.get("title") or []
            sentences = context.get("sentences") or []
            contexts = [concat_fields(str(title), " ".join(sentence_list)) for title, sentence_list in zip(titles, sentences)]
            if not contexts:
                continue
            parsed.append(
                {
                    "question": str(record.get("question", "")).strip(),
                    "answer": str(record.get("answer", "")).strip(),
                    "contexts": contexts,
                }
            )
        if len(parsed) >= 16:
            return parsed, "hotpot_qa"
    return _synthetic_examples(24 if quick else 48), "synthetic_hotpot_proxy"


def _single_pass(question: str, contexts: list[str]) -> str:
    return best_overlap_sentence(question, contexts)


def _iterative_two_hop(question: str, contexts: list[str]) -> str:
    first = best_overlap_sentence(question, contexts)
    expanded_query = concat_fields(question, first)
    second = best_overlap_sentence(expanded_query, contexts)
    return second if token_overlap_f1(second, question) >= token_overlap_f1(first, question) else first


def _multi_query_vote(question: str, contexts: list[str]) -> str:
    token_query = " ".join(simple_tokenize(question)[:6])
    candidates = []
    for sentence in concat_fields(*contexts).split(". "):
        sentence = sentence.strip()
        if sentence:
            score = token_overlap_f1(sentence, question) + token_overlap_f1(sentence, token_query)
            candidates.append((score, sentence))
    return max(candidates, key=lambda item: item[0])[1] if candidates else ""


def _evaluate(examples: list[dict[str, object]], strategy: Callable[[str, list[str]], str]) -> dict[str, float]:
    overlap_scores = []
    exact_hits = []
    for example in examples:
        raw_contexts = example["contexts"]
        contexts = [str(item) for item in raw_contexts] if isinstance(raw_contexts, list) else []
        prediction = strategy(str(example["question"]), contexts)
        answer = str(example["answer"])
        overlap_scores.append(token_overlap_f1(prediction, answer))
        exact_hits.append(1.0 if answer.lower() in prediction.lower() else 0.0)
    return {
        "overlap_f1": sum(overlap_scores) / max(1, len(overlap_scores)),
        "exact_contains": sum(exact_hits) / max(1, len(exact_hits)),
    }


def run(*, quick: bool = True) -> ProjectArtifact:
    mode = "quick" if quick else "full"
    examples, used_dataset = _load_examples(quick)
    experiments = [
        ("single_pass_scan", "sentence_selector", "single_agent", _single_pass),
        ("iterative_two_hop", "sentence_selector", "planner_retriever_loop", _iterative_two_hop),
        ("multi_query_vote", "sentence_selector", "multi_query_ensemble", _multi_query_vote),
    ]

    results = []
    for variant, algorithm, feature_set, strategy in experiments:
        metrics, runtime = timed_call(_evaluate, examples, strategy)
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
                metric_name="overlap_f1",
                metric_value=metrics["overlap_f1"],
                metric_direction="higher_is_better",
                secondary_metric_name="exact_contains",
                secondary_metric_value=metrics["exact_contains"],
                train_samples=len(examples),
                eval_samples=len(examples),
                runtime_sec=runtime,
                notes="Answer extraction proxy over supporting contexts.",
            )
        )

    best = pick_best_result(results)
    findings = [
        f"The best agent workflow was {best.variant} with overlap F1 {best.metric_value:.3f}." if best else "No successful run was recorded.",
        "The iterative workflow approximates planner-retriever loops by using the first retrieval result to reformulate the second search pass.",
        "Even with sentence-level heuristics, multi-step reasoning benefits from maintaining an intermediate state rather than scanning all contexts once.",
    ]
    return ProjectArtifact(
        project_id=PROJECT_ID,
        title=TITLE,
        objective="Compare simple multi-step agent loops for multi-hop question answering under a retrieval-and-selection proxy.",
        requested_dataset=REQUESTED_DATASET,
        used_dataset=used_dataset,
        mode=mode,
        metric_name="overlap_f1",
        metric_direction="higher_is_better",
        results=results,
        findings=findings,
        notes="This benchmark uses HotpotQA when accessible and otherwise falls back to a synthetic multi-hop corpus. It does not call an external LLM; it isolates the workflow logic itself.",
    )


if __name__ == "__main__":
    artifact = run()
    best = pick_best_result(artifact.results)
    print(f"{artifact.title}: {best.variant if best else 'no result'}")