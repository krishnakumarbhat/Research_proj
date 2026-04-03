from __future__ import annotations

from .common import ProjectArtifact, evaluate_text_classifier, make_result, pick_best_result, set_global_seed, timed_call


PROJECT_ID = "synthetic_data_generation_niche_classifiers"
TITLE = "Synthetic Data Generation for Niche Classifiers"
REQUESTED_DATASET = "Symptom2Disease"


def _seed_records() -> list[dict[str, str]]:
    return [
        {"text": "fever dry cough fatigue body aches", "label": "flu"},
        {"text": "chills fever aching muscles fatigue", "label": "flu"},
        {"text": "runny nose sneezing itchy eyes mild headache", "label": "allergy"},
        {"text": "watery eyes sneezing nasal irritation congestion", "label": "allergy"},
        {"text": "chest pressure shortness of breath fatigue dizziness", "label": "heart_issue"},
        {"text": "tight chest breathlessness sweating fatigue", "label": "heart_issue"},
        {"text": "upper abdominal burning nausea bloating poor appetite", "label": "gastritis"},
        {"text": "stomach discomfort after meals nausea fullness", "label": "gastritis"},
        {"text": "frequent urination excessive thirst fatigue blurred vision", "label": "diabetes"},
        {"text": "constant thirst fatigue weight loss blurry vision", "label": "diabetes"},
        {"text": "fever rash red spots cough irritated eyes", "label": "measles"},
        {"text": "skin eruption fever cough light sensitivity tiredness", "label": "measles"},
    ]


def _heldout_records() -> list[dict[str, str]]:
    return [
        {"text": "high temperature persistent cough aching muscles irritated eyes and fatigue", "label": "flu"},
        {"text": "body aches chills dry cough tiredness and mild headache", "label": "flu"},
        {"text": "fever cough tiredness irritated eyes and headache", "label": "flu"},
        {"text": "stuffy nose itchy watery eyes repeated sneezing", "label": "allergy"},
        {"text": "nasal irritation watery eyes congestion sneezing and light cough", "label": "allergy"},
        {"text": "watery eyes congestion sneezing tiredness and headache", "label": "allergy"},
        {"text": "tight chest low energy difficulty breathing and dizziness", "label": "heart_issue"},
        {"text": "pressure in chest breathlessness unusual fatigue and sweating", "label": "heart_issue"},
        {"text": "shortness of breath sweating fatigue dizziness and weakness", "label": "heart_issue"},
        {"text": "burning stomach discomfort with nausea poor appetite and fatigue", "label": "gastritis"},
        {"text": "after meals there is bloating upper stomach pain nausea and tiredness", "label": "gastritis"},
        {"text": "poor appetite bloating nausea tiredness and stomach fullness", "label": "gastritis"},
        {"text": "often urinating with intense thirst blurry sight fatigue and queasiness", "label": "diabetes"},
        {"text": "strong thirst weight loss tiredness blurred vision and nausea", "label": "diabetes"},
        {"text": "intense thirst fatigue weight loss queasiness and low energy", "label": "diabetes"},
        {"text": "skin eruption with fever red spots cough irritated eyes and fatigue", "label": "measles"},
        {"text": "rash fever light sensitivity cough facial spots and tiredness", "label": "measles"},
        {"text": "fever cough tiredness irritated eyes and light sensitivity", "label": "measles"},
    ]


def _synonym_augment(text: str) -> str:
    replacements = {
        "fever": "high temperature",
        "cough": "persistent cough",
        "runny nose": "nasal discharge",
        "upper abdominal burning": "burning stomach discomfort",
        "frequent urination": "often urinating",
        "rash": "skin eruption",
        "fatigue": "low energy",
    }
    augmented = text
    for source, target in replacements.items():
        augmented = augmented.replace(source, target)
    return augmented


def _template_augment(text: str) -> str:
    return f"Patient presents with {text} and reports symptoms have fluctuated over the last three days."


def _mixed_augment(text: str) -> str:
    return f"Patient presents with {_synonym_augment(text)} plus intermittent headache and fatigue over several days."


def _build_corpus(multiplier: int) -> list[dict[str, str]]:
    base = _seed_records()
    records = []
    for repeat in range(multiplier):
        for record in base:
            tokens = record["text"].split()
            shift = repeat % len(tokens)
            rotated = tokens[shift:] + tokens[:shift]
            records.append({"text": " ".join(rotated), "label": record["label"]})
    return records


def run(*, quick: bool = True) -> ProjectArtifact:
    set_global_seed(42)
    mode = "quick" if quick else "full"
    records = _build_corpus(1 if quick else 3)
    heldout = _heldout_records() * (2 if not quick else 1)
    train_texts = [record["text"] for record in records]
    y_train = [record["label"] for record in records]
    test_texts = [record["text"] for record in heldout]
    y_test = [record["label"] for record in heldout]

    augmentations = {
        "seed_only": train_texts,
        "seed_plus_synonyms": train_texts + [_synonym_augment(text) for text in train_texts],
        "seed_plus_templates": train_texts + [_template_augment(text) for text in train_texts],
        "seed_plus_mixed": train_texts + [_synonym_augment(text) for text in train_texts] + [_mixed_augment(text) for text in train_texts],
    }
    augmented_labels = {
        "seed_only": y_train,
        "seed_plus_synonyms": y_train + y_train,
        "seed_plus_templates": y_train + y_train,
        "seed_plus_mixed": y_train + y_train + y_train,
    }

    experiments = [
        ("seed_only_logreg", "logreg", "seed_only"),
        ("seed_plus_synonyms_logreg", "logreg", "seed_plus_synonyms"),
        ("seed_plus_templates_logreg", "logreg", "seed_plus_templates"),
        ("seed_plus_mixed_svm", "linear_svm", "seed_plus_mixed"),
    ]

    results = []
    for variant, algorithm, feature_set in experiments:
        metrics, runtime = timed_call(
            evaluate_text_classifier,
            augmentations[feature_set],
            augmented_labels[feature_set],
            test_texts,
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
                used_dataset="synthetic_symptom2disease_proxy",
                mode=mode,
                variant=variant,
                algorithm=algorithm,
                feature_set=feature_set,
                metric_name="macro_f1",
                metric_value=metrics["macro_f1"],
                metric_direction="higher_is_better",
                secondary_metric_name="accuracy",
                secondary_metric_value=metrics["accuracy"],
                train_samples=len(augmentations[feature_set]),
                eval_samples=len(y_test),
                runtime_sec=runtime,
                notes="Evaluated on held-out paraphrases rather than duplicates of the seed list.",
            )
        )

    best = pick_best_result(results)
    seed_only = next(result for result in results if result.feature_set == "seed_only")
    augmented_best = max((result for result in results if result.feature_set != "seed_only"), key=lambda item: item.metric_value)
    if augmented_best.metric_value > seed_only.metric_value:
        augmentation_statement = f"Synthetic augmentation improved macro F1 from {seed_only.metric_value:.3f} to {augmented_best.metric_value:.3f}."
    elif augmented_best.metric_value < seed_only.metric_value:
        augmentation_statement = f"The seed-only baseline outperformed the augmented variants on this proxy ({seed_only.metric_value:.3f} versus {augmented_best.metric_value:.3f})."
    else:
        augmentation_statement = f"Seed-only and augmented variants tied at macro F1 {seed_only.metric_value:.3f} on the current synthetic dataset."
    findings = [
        f"The best augmentation strategy was {best.variant} with macro F1 {best.metric_value:.3f}." if best else "No successful run was recorded.",
        augmentation_statement,
        "The mixed augmentation regime tests whether stylistic diversity and synonym substitution complement each other better than either alone.",
    ]
    return ProjectArtifact(
        project_id=PROJECT_ID,
        title=TITLE,
        objective="Measure how simple synthetic symptom paraphrases change disease-classification quality under tiny-data conditions.",
        requested_dataset=REQUESTED_DATASET,
        used_dataset="synthetic_symptom2disease_proxy",
        mode=mode,
        metric_name="macro_f1",
        metric_direction="higher_is_better",
        results=results,
        findings=findings,
        notes="This benchmark uses a seed-list-style synthetic corpus because the primary research question is whether cheap synthetic expansion helps a narrow classifier when real data is scarce.",
    )


if __name__ == "__main__":
    artifact = run()
    best = pick_best_result(artifact.results)
    print(f"{artifact.title}: {best.variant if best else 'no result'}")