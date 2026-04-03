from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from .asr_digits import ASRModelBundle
from .audio_utils import AUDIO_SR, MAX_MEL_FRAMES, N_MELS, audio_to_logmel, logmel_to_audio
from .common import ARTIFACTS_DIR, RANDOM_STATE, ProjectResult, make_record, timed_run
from .datasets import AudioExample, load_spoken_digits


PROJECT_ID = "text_to_speech"
TITLE = "Text-to-Speech (spoken-digit synthesis proxy)"
DATASET_NAME = "Free Spoken Digit Dataset"


class ConditionalSpectrogramVAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 16) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + 10, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.mu = nn.Linear(128, latent_dim)
        self.logvar = nn.Linear(128, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 10, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
        )

    def encode(self, x: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        condition = F.one_hot(labels, 10).float()
        hidden = self.encoder(torch.cat([x, condition], dim=1))
        return self.mu(hidden), self.logvar(hidden)

    def decode(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        condition = F.one_hot(labels, 10).float()
        return self.decoder(torch.cat([z, condition], dim=1))

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, labels)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return self.decode(z, labels), mu, logvar


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _prepare_spectrograms(examples: list[AudioExample]) -> tuple[np.ndarray, np.ndarray]:
    spectrograms = np.stack([audio_to_logmel(example.audio, example.sr) for example in examples])
    labels = np.asarray([example.label for example in examples], dtype=int)
    return spectrograms.astype(np.float32), labels


def _select_unit_examples(train_examples: list[AudioExample]) -> dict[int, np.ndarray]:
    grouped: dict[int, list[AudioExample]] = defaultdict(list)
    for example in train_examples:
        grouped[example.label].append(example)
    outputs: dict[int, np.ndarray] = {}
    for label, items in grouped.items():
        lengths = np.asarray([len(item.audio) for item in items])
        chosen = items[int(np.argmin(np.abs(lengths - np.median(lengths))))]
        outputs[label] = chosen.audio
    return outputs


def _save_spectrogram_grid(specs: np.ndarray, labels: np.ndarray, path: Path, title: str) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for label, ax in enumerate(axes.ravel()):
        index = np.flatnonzero(labels == label)[0]
        ax.imshow(specs[index], aspect="auto", origin="lower", cmap="magma")
        ax.set_title(str(label))
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _save_audio_samples(audios: list[np.ndarray], labels: np.ndarray, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for label in range(10):
        index = np.flatnonzero(labels == label)[0]
        sf.write(directory / f"{label}.wav", audios[index], AUDIO_SR)


def _evaluate_generation(
    spectrograms: np.ndarray,
    audios: list[np.ndarray],
    labels: np.ndarray,
    asr_model: ASRModelBundle,
    class_means: np.ndarray,
) -> tuple[float, float, float, float]:
    predictions = asr_model.predict_batch([(audio, AUDIO_SR) for audio in audios])
    asr_accuracy = float(np.mean(predictions == labels))
    mel_distance = float(np.mean((spectrograms - class_means[labels]) ** 2))
    avg_duration = float(np.mean([len(audio) / AUDIO_SR for audio in audios]))
    rank_score = asr_accuracy - 0.02 * mel_distance
    return asr_accuracy, mel_distance, avg_duration, rank_score


def _train_vae(
    train_specs: np.ndarray,
    train_labels: np.ndarray,
    eval_labels: np.ndarray,
    quick: bool,
) -> tuple[np.ndarray, list[np.ndarray], float]:
    device = _device()
    input_dim = N_MELS * MAX_MEL_FRAMES
    model = ConditionalSpectrogramVAE(input_dim=input_dim).to(device)
    dataset = TensorDataset(
        torch.tensor(train_specs.reshape(len(train_specs), -1), dtype=torch.float32),
        torch.tensor(train_labels, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 20 if quick else 40

    def fit() -> ConditionalSpectrogramVAE:
        model.train()
        for _ in range(epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                recon, mu, logvar = model(batch_x, batch_y)
                recon_loss = F.mse_loss(recon, batch_x)
                kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + 1e-3 * kl
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        return model.eval()

    model, fit_seconds = timed_run(fit)
    with torch.no_grad():
        label_tensor = torch.tensor(eval_labels, dtype=torch.long, device=device)
        z = torch.randn(len(eval_labels), model.latent_dim, device=device)
        specs = model.decode(z, label_tensor).cpu().numpy().reshape(-1, N_MELS, MAX_MEL_FRAMES)
    audios = [logmel_to_audio(spec) for spec in specs]
    return specs, audios, fit_seconds


def run(asr_model: ASRModelBundle, quick: bool = False, prefer_real_audio: bool = True) -> ProjectResult:
    examples, source = load_spoken_digits(
        max_per_digit=18 if quick else 50,
        prefer_real=prefer_real_audio,
    )
    spectrograms, labels = _prepare_spectrograms(examples)
    train_idx, _test_idx = train_test_split(
        np.arange(len(examples)),
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    train_examples = [examples[index] for index in train_idx]
    train_specs = spectrograms[train_idx]
    train_labels = labels[train_idx]
    class_means = np.stack([train_specs[train_labels == label].mean(axis=0) for label in range(10)])
    eval_labels = np.repeat(np.arange(10), 2 if quick else 3)

    records = []
    artifacts = []
    sample_store: dict[str, tuple[np.ndarray, list[np.ndarray], np.ndarray]] = {}

    unit_examples = _select_unit_examples(train_examples)
    unit_audios = [unit_examples[label] for label in eval_labels]
    unit_specs = np.stack([audio_to_logmel(audio, AUDIO_SR) for audio in unit_audios])
    unit_accuracy, unit_distance, unit_duration, unit_rank = _evaluate_generation(
        unit_specs,
        unit_audios,
        eval_labels,
        asr_model,
        class_means,
    )
    unit_png = ARTIFACTS_DIR / "tts_unit_selection.png"
    _save_spectrogram_grid(unit_specs, eval_labels, unit_png, "Unit-selection TTS samples")
    artifacts.append("artifacts/tts_unit_selection.png")
    records.append(
        make_record(
            project=PROJECT_ID,
            dataset=DATASET_NAME,
            source=source,
            task="conditional_speech_generation",
            algorithm="unit_selection",
            feature_variant="waveform_exemplars",
            optimization="representative_training_utterance",
            primary_metric="asr_accuracy",
            primary_value=unit_accuracy,
            rank_score=unit_rank,
            secondary_metric="mel_distance",
            secondary_value=unit_distance,
            tertiary_metric="avg_duration_sec",
            tertiary_value=unit_duration,
            fit_seconds=0.0,
            notes="Concatenative-style exemplar reuse",
        )
    )
    sample_store["unit_selection"] = (unit_specs, unit_audios, eval_labels)

    mean_specs = class_means[eval_labels]
    mean_audios = [logmel_to_audio(spec) for spec in mean_specs]
    mean_accuracy, mean_distance, mean_duration, mean_rank = _evaluate_generation(
        mean_specs,
        mean_audios,
        eval_labels,
        asr_model,
        class_means,
    )
    mean_png = ARTIFACTS_DIR / "tts_mean_spectrogram.png"
    _save_spectrogram_grid(mean_specs, eval_labels, mean_png, "Mean-spectrogram TTS samples")
    artifacts.append("artifacts/tts_mean_spectrogram.png")
    records.append(
        make_record(
            project=PROJECT_ID,
            dataset=DATASET_NAME,
            source=source,
            task="conditional_speech_generation",
            algorithm="mean_spectrogram",
            feature_variant="class_average_logmel",
            optimization="griffin_lim_inversion",
            primary_metric="asr_accuracy",
            primary_value=mean_accuracy,
            rank_score=mean_rank,
            secondary_metric="mel_distance",
            secondary_value=mean_distance,
            tertiary_metric="avg_duration_sec",
            tertiary_value=mean_duration,
            fit_seconds=0.0,
            notes="Statistical parametric-style average spectrum",
        )
    )
    sample_store["mean_spectrogram"] = (mean_specs, mean_audios, eval_labels)

    vae_specs, vae_audios, vae_seconds = _train_vae(train_specs, train_labels, eval_labels, quick)
    vae_accuracy, vae_distance, vae_duration, vae_rank = _evaluate_generation(
        vae_specs,
        vae_audios,
        eval_labels,
        asr_model,
        class_means,
    )
    vae_png = ARTIFACTS_DIR / "tts_conditional_vae.png"
    _save_spectrogram_grid(vae_specs, eval_labels, vae_png, "Conditional VAE TTS samples")
    artifacts.append("artifacts/tts_conditional_vae.png")
    records.append(
        make_record(
            project=PROJECT_ID,
            dataset=DATASET_NAME,
            source=source,
            task="conditional_speech_generation",
            algorithm="conditional_vae",
            feature_variant="label_plus_latent_logmel",
            optimization="vae_decoder_plus_griffin_lim",
            primary_metric="asr_accuracy",
            primary_value=vae_accuracy,
            rank_score=vae_rank,
            secondary_metric="mel_distance",
            secondary_value=vae_distance,
            tertiary_metric="avg_duration_sec",
            tertiary_value=vae_duration,
            fit_seconds=vae_seconds,
            notes=f"{len(eval_labels)} generated utterances evaluated",
        )
    )
    sample_store["conditional_vae"] = (vae_specs, vae_audios, eval_labels)

    best = max(records, key=lambda record: record.rank_score)
    best_specs, best_audios, best_labels = sample_store[best.algorithm]
    best_dir = ARTIFACTS_DIR / f"tts_{best.algorithm}_samples"
    _save_audio_samples(best_audios, best_labels, best_dir)
    artifacts.append(f"artifacts/tts_{best.algorithm}_samples")

    summary = (
        f"The strongest TTS proxy was {best.algorithm}, which reached ASR-backchecked accuracy {best.primary_value:.3f}. "
        "On this spoken-digit task, exemplar reuse sets a strong floor, while spectrogram generation quality is limited mostly by the simple Griffin-Lim vocoder."
    )
    recommendation = (
        "For a lightweight TTS benchmark, compare unit-selection, mean-spectrum, and one neural spectrogram generator, then evaluate them with an external ASR recognizer instead of listening alone."
    )
    key_findings = [
        f"Best generated-speech recognizability was {best.primary_value:.3f} from {best.algorithm}.",
        "Concatenative reuse is hard to beat on tiny low-entropy vocabularies.",
        "Neural spectrogram generation is feasible here, but waveform quality is bottlenecked by the simple inversion stage.",
    ]
    caveats = [
        "This is digit speech synthesis, not general-purpose expressive TTS.",
        "Generated speech is evaluated by the benchmark ASR recognizer, which is a proxy metric rather than a listening test.",
        "If FSDD cannot be downloaded, the runner falls back to synthetic digit-like audio.",
    ]
    return ProjectResult(
        project=PROJECT_ID,
        title=TITLE,
        dataset=DATASET_NAME,
        source=source,
        task="text_to_speech",
        records=records,
        summary=summary,
        recommendation=recommendation,
        key_findings=key_findings,
        caveats=caveats,
        artifacts=artifacts,
    )
