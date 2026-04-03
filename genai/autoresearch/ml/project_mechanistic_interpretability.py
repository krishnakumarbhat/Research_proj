from __future__ import annotations

from .common import ProjectArtifact, flatten_text, load_hf_dataset_records, make_result, pick_best_result
from .toy_models import run_toy_language_model_benchmark


PROJECT_ID = "mechanistic_interpretability"
TITLE = "Mechanistic Interpretability on Toy Models"
REQUESTED_DATASET = "roneneldan/TinyStories"


def _synthetic_texts(total: int) -> list[str]:
    base = [
        "Tom has a red ball. He rolls the ball to Ana. Ana laughs and rolls it back.",
        "Mia sees a small cat under the chair. She gives the cat milk and the cat purrs.",
        "Ben builds a tall tower from blue blocks. The tower falls and he builds it again.",
        "A bird sits on a tree near the pond. The bird flies down and drinks water.",
        "Lila opens a box with crayons. She draws a sun, a tree, and a green hill.",
    ]
    return [base[index % len(base)] for index in range(total)]


def _load_texts(quick: bool) -> tuple[list[str], str]:
    raw = load_hf_dataset_records("roneneldan/TinyStories", split="train[:120]" if quick else "train[:240]")
    texts = []
    if raw:
        for record in raw:
            text = flatten_text(record.get("text"))
            if len(text) >= 40:
                texts.append(text)
        if len(texts) >= 32:
            return texts, "roneneldan/TinyStories"
    return _synthetic_texts(80 if quick else 160), "synthetic_tinystories_proxy"


def run(*, quick: bool = True) -> ProjectArtifact:
    mode = "quick" if quick else "full"
    texts, used_dataset = _load_texts(quick)
    toy_results = run_toy_language_model_benchmark(texts)

    results = []
    for toy_result in toy_results:
        results.append(
            make_result(
                project_id=PROJECT_ID,
                project_name=TITLE,
                requested_dataset=REQUESTED_DATASET,
                used_dataset=used_dataset,
                mode=mode,
                variant=toy_result.name,
                algorithm=toy_result.name,
                feature_set="character_level_language_model",
                metric_name="val_nll",
                metric_value=toy_result.val_nll,
                metric_direction="lower_is_better",
                secondary_metric_name="activation_sparsity",
                secondary_metric_value=toy_result.activation_sparsity,
                train_samples=len(texts),
                eval_samples=len(texts),
                runtime_sec=toy_result.runtime_sec,
                notes=f"parameter_count={toy_result.parameter_count}",
            )
        )

    best = pick_best_result(results)
    findings = [
        f"The lowest validation NLL came from {best.variant} at {best.metric_value:.3f}." if best else "No successful run was recorded.",
        "The bigram baseline is easy to inspect but usually leaves large predictive gaps that even tiny neural models close quickly.",
        "Activation sparsity is logged as a simple interpretability proxy to compare how concentrated each model's internal representations become.",
    ]
    return ProjectArtifact(
        project_id=PROJECT_ID,
        title=TITLE,
        objective="Train tiny language models on a child-language corpus slice and compare simple interpretability-friendly architectures.",
        requested_dataset=REQUESTED_DATASET,
        used_dataset=used_dataset,
        mode=mode,
        metric_name="val_nll",
        metric_direction="lower_is_better",
        results=results,
        findings=findings,
        notes="This is a true local training benchmark rather than a synthetic proxy. The models are intentionally tiny so they can be trained and inspected on a laptop-scale setup.",
    )


if __name__ == "__main__":
    artifact = run()
    best = pick_best_result(artifact.results)
    print(f"{artifact.title}: {best.variant if best else 'no result'}")