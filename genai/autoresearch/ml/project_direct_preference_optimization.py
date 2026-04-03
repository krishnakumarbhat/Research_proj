from __future__ import annotations

from sklearn.base import clone

from .common import ProjectArtifact, build_text_pipeline, flatten_text, load_hf_dataset_records, make_result, pick_best_result, set_global_seed, simple_tokenize, timed_call, token_overlap_f1


PROJECT_ID = "direct_preference_optimization"
TITLE = "Direct Preference Optimization on Small Language Models (Proxy)"
REQUESTED_DATASET = "argilla/dpo-mix-7k"


def _synthetic_pair_splits(quick: bool) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    train_pairs = [
        {
            "prompt": "Explain why regular exercise improves cardiovascular health.",
            "chosen": "Regular exercise conditions the heart muscle, improves circulation, and can lower resting blood pressure over time.",
            "rejected": "Exercise improves cardiovascular health because cardiovascular health improves when exercise is healthy and regular.",
        },
        {
            "prompt": "Summarize the water cycle in two sentences.",
            "chosen": "Water evaporates from surfaces, condenses into clouds, and returns as precipitation. Runoff and renewed evaporation keep the cycle going.",
            "rejected": "The water cycle is the cycle of water, and water goes through the water cycle with clouds and water.",
        },
        {
            "prompt": "Give one safety tip for laboratory work.",
            "chosen": "Wear eye protection and read the procedure before handling chemicals or equipment.",
            "rejected": "Laboratory safety is important and people should be safe in laboratories.",
        },
        {
            "prompt": "Describe one benefit of unit tests in software projects.",
            "chosen": "Unit tests catch regressions early by checking small pieces of behavior before changes spread through the system.",
            "rejected": "Unit tests are about software projects and they help because testing is useful for software.",
        },
        {
            "prompt": "State one reason plants need sunlight.",
            "chosen": "Plants use sunlight to power photosynthesis, which helps them make the sugars they need for growth.",
            "rejected": "Plants need sunlight because sunlight is needed by plants in nature.",
        },
        {
            "prompt": "Explain one purpose of a budget in personal finance.",
            "chosen": "A budget helps track income and spending so a person can plan expenses and avoid overspending.",
            "rejected": "A budget is for finance and money because budgeting is a financial budget activity.",
        },
    ]
    eval_pairs = [
        {
            "prompt": "Give one reason steady exercise helps the heart and circulation.",
            "chosen": "Steady exercise makes the heart pump blood more efficiently and can improve circulation.",
            "rejected": "Exercise helps the heart because heart exercise is part of heart health and healthy exercise.",
        },
        {
            "prompt": "Explain the water cycle in two short sentences.",
            "chosen": "Water evaporates, condenses into clouds, and falls as precipitation. The process repeats through runoff and renewed evaporation.",
            "rejected": "The water cycle is a process with water and clouds where the water cycle keeps cycling water.",
        },
        {
            "prompt": "Name one laboratory safety practice.",
            "chosen": "Put on eye protection before working with chemicals or lab equipment.",
            "rejected": "A laboratory practice is to practice safety because the lab can be unsafe.",
        },
        {
            "prompt": "Give one value of unit testing in software development.",
            "chosen": "Unit tests reveal regressions quickly by checking narrow behaviors before broader integration breaks.",
            "rejected": "Unit testing is valuable in software because software values tests and testing value.",
        },
        {
            "prompt": "Provide one reason sunlight matters for plants.",
            "chosen": "Sunlight provides the energy plants use in photosynthesis to make food for growth.",
            "rejected": "Sunlight matters for plants because plants need sunlight and sunlight matters to plants.",
        },
        {
            "prompt": "State one use of a household budget.",
            "chosen": "A household budget shows where money is going so spending can be planned and controlled.",
            "rejected": "A budget is useful for a household because household budgeting uses a budget for money.",
        },
    ]
    train_multiplier = 2 if quick else 4
    eval_multiplier = 2 if quick else 3
    return train_pairs * train_multiplier, eval_pairs * eval_multiplier


def _load_pairs(quick: bool) -> tuple[list[dict[str, str]], str]:
    raw = load_hf_dataset_records("argilla/dpo-mix-7k", split="train[:100]" if quick else "train[:220]")
    parsed = []
    if raw:
        for record in raw:
            prompt = flatten_text(record.get("prompt") or record.get("instruction") or record.get("input"))
            chosen = flatten_text(record.get("chosen") or record.get("chosen_response") or record.get("response_1"))
            rejected = flatten_text(record.get("rejected") or record.get("rejected_response") or record.get("response_2"))
            if prompt and chosen and rejected:
                parsed.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
        if len(parsed) >= 40:
            return parsed, "argilla/dpo-mix-7k"
    return [], "synthetic_dpo_proxy"


def _fit_pairwise_model(train_pairs: list[dict[str, str]], algorithm: str):
    texts = []
    labels = []
    for pair in train_pairs:
        texts.append(f"Prompt: {pair['prompt']}\nResponse: {pair['chosen']}")
        labels.append(1)
        texts.append(f"Prompt: {pair['prompt']}\nResponse: {pair['rejected']}")
        labels.append(0)
    model = build_text_pipeline(algorithm, analyzer="word", ngram_range=(1, 2), max_features=6000)
    model.fit(texts, labels)
    return model


def _score_model(model, text: str) -> float:
    estimator = model.named_steps["estimator"]
    if hasattr(estimator, "decision_function"):
        return float(model.decision_function([text])[0])
    if hasattr(estimator, "predict_proba"):
        return float(model.predict_proba([text])[0][1])
    return float(model.predict([text])[0])


def _evaluate_model(model, eval_pairs: list[dict[str, str]]) -> dict[str, float]:
    hits = 0
    margins = []
    for pair in eval_pairs:
        chosen_text = f"Prompt: {pair['prompt']}\nResponse: {pair['chosen']}"
        rejected_text = f"Prompt: {pair['prompt']}\nResponse: {pair['rejected']}"
        chosen_score = _score_model(model, chosen_text)
        rejected_score = _score_model(model, rejected_text)
        hits += 1 if chosen_score > rejected_score else 0
        margins.append(chosen_score - rejected_score)
    return {
        "pairwise_accuracy": hits / max(1, len(eval_pairs)),
        "mean_margin": sum(margins) / max(1, len(margins)),
    }


def _heuristic_reward(pair: dict[str, str]) -> tuple[float, float]:
    prompt = pair["prompt"]
    chosen_score = token_overlap_f1(pair["chosen"], prompt) + 0.02 * len(simple_tokenize(pair["chosen"]))
    rejected_score = token_overlap_f1(pair["rejected"], prompt) + 0.02 * len(simple_tokenize(pair["rejected"]))
    return chosen_score, rejected_score


def _evaluate_heuristic(eval_pairs: list[dict[str, str]]) -> dict[str, float]:
    hits = 0
    margins = []
    for pair in eval_pairs:
        chosen_score, rejected_score = _heuristic_reward(pair)
        hits += 1 if chosen_score > rejected_score else 0
        margins.append(chosen_score - rejected_score)
    return {
        "pairwise_accuracy": hits / max(1, len(eval_pairs)),
        "mean_margin": sum(margins) / max(1, len(margins)),
    }


def run(*, quick: bool = True) -> ProjectArtifact:
    set_global_seed(42)
    mode = "quick" if quick else "full"
    pairs, used_dataset = _load_pairs(quick)
    if used_dataset == "synthetic_dpo_proxy":
        train_pairs, eval_pairs = _synthetic_pair_splits(quick)
    else:
        split = 30 if quick else 60
        train_pairs = pairs[:split]
        eval_pairs = pairs[split : split + (12 if quick else 24)]

    results = []
    heuristic_metrics, heuristic_runtime = timed_call(_evaluate_heuristic, eval_pairs)
    results.append(
        make_result(
            project_id=PROJECT_ID,
            project_name=TITLE,
            requested_dataset=REQUESTED_DATASET,
            used_dataset=used_dataset,
            mode=mode,
            variant="heuristic_reward",
            algorithm="overlap_reward",
            feature_set="prompt_response_lexical_overlap",
            metric_name="pairwise_accuracy",
            metric_value=heuristic_metrics["pairwise_accuracy"],
            metric_direction="higher_is_better",
            secondary_metric_name="mean_margin",
            secondary_metric_value=heuristic_metrics["mean_margin"],
            train_samples=len(train_pairs),
            eval_samples=len(eval_pairs),
            runtime_sec=heuristic_runtime,
            notes="No-training lexical reward baseline.",
        )
    )

    for algorithm in ["logreg", "linear_svm", "nb"]:
        model, fit_runtime = timed_call(_fit_pairwise_model, train_pairs, algorithm)
        metrics = _evaluate_model(model, eval_pairs)
        results.append(
            make_result(
                project_id=PROJECT_ID,
                project_name=TITLE,
                requested_dataset=REQUESTED_DATASET,
                used_dataset=used_dataset,
                mode=mode,
                variant=f"pairwise_{algorithm}",
                algorithm=algorithm,
                feature_set="prompt_plus_response_pair",
                metric_name="pairwise_accuracy",
                metric_value=metrics["pairwise_accuracy"],
                metric_direction="higher_is_better",
                secondary_metric_name="mean_margin",
                secondary_metric_value=metrics["mean_margin"],
                train_samples=len(train_pairs),
                eval_samples=len(eval_pairs),
                runtime_sec=fit_runtime,
                notes="Binary classifier trained on chosen-versus-rejected pairs.",
            )
        )

    best = pick_best_result(results)
    heuristic = next(result for result in results if result.variant == "heuristic_reward")
    learned_best = max((result for result in results if result.variant != "heuristic_reward"), key=lambda item: item.metric_value)
    if learned_best.metric_value > heuristic.metric_value:
        preference_statement = f"The learned preference model improved pairwise accuracy from {heuristic.metric_value:.3f} to {learned_best.metric_value:.3f}."
    elif learned_best.metric_value < heuristic.metric_value:
        preference_statement = f"The heuristic reward outperformed the learned pairwise models on this proxy ({heuristic.metric_value:.3f} versus {learned_best.metric_value:.3f})."
    else:
        preference_statement = f"The heuristic and learned preference models tied at pairwise accuracy {heuristic.metric_value:.3f} on this compact proxy corpus."
    findings = [
        f"The best preference learner was {best.variant} with pairwise accuracy {best.metric_value:.3f}." if best else "No successful run was recorded.",
        preference_statement,
        "The learned pairwise scorers are a lightweight stand-in for a reward model trained before a true DPO update.",
    ]
    return ProjectArtifact(
        project_id=PROJECT_ID,
        title=TITLE,
        objective="Learn preference scores from chosen/rejected response pairs using a CPU-friendly proxy for the reward-model stage of DPO.",
        requested_dataset=REQUESTED_DATASET,
        used_dataset=used_dataset,
        mode=mode,
        metric_name="pairwise_accuracy",
        metric_direction="higher_is_better",
        results=results,
        findings=findings,
        notes="This module does not fine-tune a generator; it benchmarks the smaller and cheaper preference-modeling stage that often precedes DPO or related preference-optimization methods.",
    )


if __name__ == "__main__":
    artifact = run()
    best = pick_best_result(artifact.results)
    print(f"{artifact.title}: {best.variant if best else 'no result'}")