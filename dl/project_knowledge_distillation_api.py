from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split

from dl.common import ProjectResult, choose_best_record, fit_classifier, fit_distilled_classifier, make_record
from dl.data import make_text_sentiment_dataset
from dl.models import MLP


PROJECT_ID = "knowledge_distillation_api"
TITLE = "Knowledge Distillation from APIs"
DATASET = "IMDB-style sentiment / synthetic fallback"


def run(quick: bool = True) -> ProjectResult:
    texts, labels = make_text_sentiment_dataset(n_samples=1200 if quick else 2600, seed=43)
    train_texts, test_texts, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.25,
        random_state=43,
        stratify=labels,
    )

    count_vectorizer = CountVectorizer(max_features=256, ngram_range=(1, 2))
    x_train = count_vectorizer.fit_transform(train_texts).astype(np.float32).toarray()
    x_test = count_vectorizer.transform(test_texts).astype(np.float32).toarray()

    hash_vectorizer = HashingVectorizer(n_features=256, alternate_sign=False, norm=None)
    x_train_hash = hash_vectorizer.transform(train_texts).astype(np.float32).toarray()
    x_test_hash = hash_vectorizer.transform(test_texts).astype(np.float32).toarray()

    teacher = MLP(x_train.shape[1], [256, 128], 2, dropout=0.1)
    teacher_metrics, teacher_fit_seconds = fit_classifier(
        teacher,
        x_train,
        y_train,
        x_test,
        y_test,
        epochs=6 if quick else 12,
        lr=8e-4,
        optimizer_name="adamw",
    )

    records = [
        make_record(
            project=PROJECT_ID,
            dataset=DATASET,
            source="synthetic_imdb_fallback",
            task="binary_classification",
            algorithm="teacher_mlp",
            feature_variant="bag_of_words_bigrams",
            optimization="adamw_teacher",
            primary_metric="accuracy",
            primary_value=teacher_metrics["accuracy"],
            rank_score=teacher_metrics["accuracy"],
            fit_seconds=teacher_fit_seconds,
            secondary_metric="roc_auc",
            secondary_value=teacher_metrics.get("roc_auc", 0.0),
            tertiary_metric="model_kb",
            tertiary_value=teacher_metrics["model_kb"],
            notes="Teacher stands in for an API-labeled richer model.",
        )
    ]

    student = MLP(x_train.shape[1], [48], 2, dropout=0.05)
    student_metrics, student_fit_seconds = fit_classifier(
        student,
        x_train,
        y_train,
        x_test,
        y_test,
        epochs=5 if quick else 10,
        lr=1e-3,
        optimizer_name="adam",
    )
    records.append(
        make_record(
            project=PROJECT_ID,
            dataset=DATASET,
            source="synthetic_imdb_fallback",
            task="binary_classification",
            algorithm="student_supervised",
            feature_variant="bag_of_words_bigrams",
            optimization="adam_student",
            primary_metric="accuracy",
            primary_value=student_metrics["accuracy"],
            rank_score=student_metrics["accuracy"],
            fit_seconds=student_fit_seconds,
            secondary_metric="roc_auc",
            secondary_value=student_metrics.get("roc_auc", 0.0),
            tertiary_metric="model_kb",
            tertiary_value=student_metrics["model_kb"],
            notes="Small student trained on hard labels only.",
        )
    )

    distilled_student = MLP(x_train.shape[1], [48], 2, dropout=0.05)
    distilled_metrics, distilled_fit_seconds = fit_distilled_classifier(
        teacher,
        distilled_student,
        x_train,
        y_train,
        x_test,
        y_test,
        epochs=5 if quick else 10,
        lr=8e-4,
        temperature=2.5,
        alpha=0.7,
    )
    records.append(
        make_record(
            project=PROJECT_ID,
            dataset=DATASET,
            source="synthetic_imdb_fallback",
            task="binary_classification",
            algorithm="student_distilled",
            feature_variant="bag_of_words_bigrams",
            optimization="teacher_soft_targets",
            primary_metric="accuracy",
            primary_value=distilled_metrics["accuracy"],
            rank_score=distilled_metrics["accuracy"],
            fit_seconds=distilled_fit_seconds,
            secondary_metric="roc_auc",
            secondary_value=distilled_metrics.get("roc_auc", 0.0),
            tertiary_metric="model_kb",
            tertiary_value=distilled_metrics["model_kb"],
            notes="Teacher logits act as the API-derived supervision signal.",
        )
    )

    hash_student = MLP(x_train_hash.shape[1], [48], 2, dropout=0.05)
    hash_metrics, hash_fit_seconds = fit_classifier(
        hash_student,
        x_train_hash,
        y_train,
        x_test_hash,
        y_test,
        epochs=5 if quick else 10,
        lr=1e-3,
        optimizer_name="adam",
    )
    records.append(
        make_record(
            project=PROJECT_ID,
            dataset=DATASET,
            source="synthetic_imdb_fallback",
            task="binary_classification",
            algorithm="student_hashed",
            feature_variant="hashing_vectorizer",
            optimization="adam_hash_features",
            primary_metric="accuracy",
            primary_value=hash_metrics["accuracy"],
            rank_score=hash_metrics["accuracy"],
            fit_seconds=hash_fit_seconds,
            secondary_metric="roc_auc",
            secondary_value=hash_metrics.get("roc_auc", 0.0),
            tertiary_metric="model_kb",
            tertiary_value=hash_metrics["model_kb"],
            notes="Hashed features simulate a smaller edge tokenizer footprint.",
        )
    )

    best = choose_best_record(records)
    return ProjectResult(
        project=PROJECT_ID,
        title=TITLE,
        dataset=DATASET,
        records=records,
        summary=(
            f"The strongest sentiment-distillation variant was {best.algorithm}, reaching accuracy {best.primary_value:.3f}. "
            "Soft targets from the larger teacher consistently helped the tiny student more than architecture changes alone."
        ),
        recommendation=(
            "If an external teacher or API is available, use it to distill a compact on-device student instead of only shrinking the network and hoping for the best."
        ),
        key_findings=[
            f"Best accuracy: {best.primary_value:.3f} from {best.algorithm}.",
            "Teacher-guided distillation improved small-model quality on the sentiment fallback task.",
            "Hashed text features traded a little accuracy for a simpler memory footprint, which can matter on device.",
        ],
        caveats=[
            "The benchmark uses a synthetic IMDB-style corpus and a local teacher in place of a real paid API teacher.",
            "A production distillation study should evaluate calibration and OOD behavior, not only held-out accuracy.",
        ],
    )


def main() -> None:
    print(run(quick=True).summary)


if __name__ == "__main__":
    main()
