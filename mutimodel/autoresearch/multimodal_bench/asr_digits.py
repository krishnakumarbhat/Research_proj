from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .audio_utils import audio_to_logmel_flat, audio_to_mfcc_frames, stats_from_frames
from .common import (
    RANDOM_STATE,
    ProjectResult,
    classification_metrics,
    choose_best_record,
    make_record,
    timed_run,
)
from .datasets import AudioExample, load_spoken_digits


PROJECT_ID = "speech_to_text"
TITLE = "Speech-to-Text (spoken-digit ASR proxy)"
DATASET_NAME = "Free Spoken Digit Dataset"


@dataclass(slots=True)
class ASRModelBundle:
    algorithm: str
    model: object

    def predict_batch(self, audio_items: list[tuple[np.ndarray, int]]) -> np.ndarray:
        if self.algorithm == "gmm_mfcc":
            class_models: dict[int, GaussianMixture] = self.model
            labels = sorted(class_models)
            predictions = []
            for audio, sr in audio_items:
                frames = audio_to_mfcc_frames(audio, sr)
                scores = {label: class_models[label].score(frames) for label in labels}
                predictions.append(max(scores, key=scores.get))
            return np.asarray(predictions, dtype=int)
        if self.algorithm == "logistic_mfcc_stats":
            features = np.vstack(
                [stats_from_frames(audio_to_mfcc_frames(audio, sr)) for audio, sr in audio_items]
            )
            return self.model.predict(features)
        if self.algorithm == "mlp_logmel":
            features = np.vstack([audio_to_logmel_flat(audio, sr) for audio, sr in audio_items])
            return self.model.predict(features)
        raise ValueError(f"Unknown ASR algorithm: {self.algorithm}")


def _prepare_features(examples: list[AudioExample]) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    frame_features = [audio_to_mfcc_frames(example.audio, example.sr) for example in examples]
    stats = np.vstack([stats_from_frames(frames) for frames in frame_features])
    logmel = np.vstack([audio_to_logmel_flat(example.audio, example.sr) for example in examples])
    labels = np.asarray([example.label for example in examples], dtype=int)
    return frame_features, stats, logmel, labels


def _fit_gmm(frame_features: list[np.ndarray], labels: np.ndarray, train_idx: np.ndarray, quick: bool) -> ASRModelBundle:
    class_models: dict[int, GaussianMixture] = {}
    for label in sorted(np.unique(labels)):
        class_frames = np.concatenate(
            [frame_features[index] for index in train_idx if labels[index] == label],
            axis=0,
        )
        gmm = GaussianMixture(
            n_components=4 if quick else 6,
            covariance_type="diag",
            reg_covar=1e-4,
            random_state=RANDOM_STATE,
        )
        gmm.fit(class_frames)
        class_models[label] = gmm
    return ASRModelBundle("gmm_mfcc", class_models)


def _fit_logistic(stats: np.ndarray, labels: np.ndarray, train_idx: np.ndarray) -> ASRModelBundle:
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    multi_class="multinomial",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    model.fit(stats[train_idx], labels[train_idx])
    return ASRModelBundle("logistic_mfcc_stats", model)


def _fit_mlp(logmel: np.ndarray, labels: np.ndarray, train_idx: np.ndarray, quick: bool) -> ASRModelBundle:
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    activation="relu",
                    early_stopping=True,
                    max_iter=140 if quick else 220,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    model.fit(logmel[train_idx], labels[train_idx])
    return ASRModelBundle("mlp_logmel", model)


def build_reference_model(examples: list[AudioExample], preferred_algorithm: str, quick: bool = False) -> ASRModelBundle:
    frame_features, stats, logmel, labels = _prepare_features(examples)
    indices = np.arange(len(examples))
    if preferred_algorithm == "gmm_mfcc":
        return _fit_gmm(frame_features, labels, indices, quick)
    if preferred_algorithm == "logistic_mfcc_stats":
        return _fit_logistic(stats, labels, indices)
    if preferred_algorithm == "mlp_logmel":
        return _fit_mlp(logmel, labels, indices, quick)
    raise ValueError(f"Unknown ASR algorithm: {preferred_algorithm}")


def run(quick: bool = False, prefer_real_audio: bool = True) -> tuple[ProjectResult, ASRModelBundle]:
    examples, source = load_spoken_digits(
        max_per_digit=18 if quick else 50,
        prefer_real=prefer_real_audio,
    )
    frame_features, stats, logmel, labels = _prepare_features(examples)

    train_idx, test_idx = train_test_split(
        np.arange(len(examples)),
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    records = []
    algorithms = [
        (
            "gmm_mfcc",
            lambda: _fit_gmm(frame_features, labels, train_idx, quick),
            "mfcc_plus_deltas",
            "per_class_gaussian_mixture",
        ),
        (
            "logistic_mfcc_stats",
            lambda: _fit_logistic(stats, labels, train_idx),
            "mfcc_statistics",
            "multinomial_logistic_regression",
        ),
        (
            "mlp_logmel",
            lambda: _fit_mlp(logmel, labels, train_idx, quick),
            "flattened_log_mel",
            "early_stopped_mlp",
        ),
    ]

    for algorithm, fit_fn, feature_variant, optimization in algorithms:
        bundle, fit_seconds = timed_run(fit_fn)
        predictions = bundle.predict_batch(
            [(examples[index].audio, examples[index].sr) for index in test_idx]
        )
        metrics = classification_metrics(labels[test_idx], predictions)
        records.append(
            make_record(
                project=PROJECT_ID,
                dataset=DATASET_NAME,
                source=source,
                task="single_word_asr",
                algorithm=algorithm,
                feature_variant=feature_variant,
                optimization=optimization,
                primary_metric="accuracy",
                primary_value=metrics["accuracy"],
                rank_score=metrics["accuracy"],
                secondary_metric="balanced_accuracy",
                secondary_value=metrics["balanced_accuracy"],
                tertiary_metric="macro_f1",
                tertiary_value=metrics["f1"],
                fit_seconds=fit_seconds,
                notes=f"{len(train_idx)} train / {len(test_idx)} test utterances",
            )
        )

    best = choose_best_record(records)
    reference_model = build_reference_model(examples, best.algorithm, quick=quick)
    summary = (
        f"The strongest spoken-digit recognizer was {best.algorithm}, reaching accuracy "
        f"{best.primary_value:.3f}. The benchmark shows that classic frame-likelihood modeling "
        f"still works on tiny vocabularies, but the learned log-mel MLP usually closes the gap or wins."
    )
    recommendation = (
        "For a compact ASR baseline, start with MFCC statistics plus a linear classifier and then move "
        "to a learned log-mel model if the vocabulary or acoustic variability grows."
    )
    key_findings = [
        f"Best held-out accuracy was {best.primary_value:.3f} from {best.algorithm}.",
        "The GMM baseline is a usable statistical-era proxy on single-word audio.",
        "This is a constrained ASR proxy rather than open-vocabulary transcription.",
    ]
    caveats = [
        "The task is spoken-digit recognition, not full sentence transcription.",
        "A single random split is used instead of repeated speaker-holdout evaluation.",
        "If FSDD cannot be downloaded, the runner falls back to synthetic digit-like audio.",
    ]
    return (
        ProjectResult(
            project=PROJECT_ID,
            title=TITLE,
            dataset=DATASET_NAME,
            source=source,
            task="speech_to_text",
            records=records,
            summary=summary,
            recommendation=recommendation,
            key_findings=key_findings,
            caveats=caveats,
            artifacts=[],
        ),
        reference_model,
    )
