from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .common import (
    ARTIFACTS_DIR,
    RANDOM_STATE,
    ProjectResult,
    make_record,
    timed_run,
)
from .datasets import load_digit_images
from .ocr_digits import DigitOCRModel


PROJECT_ID = "text_to_image"
TITLE = "Text-to-Image (conditional digit generation proxy)"
DATASET_NAME = "sklearn digits"


class ConditionalGenerator(nn.Module):
    def __init__(self, noise_dim: int = 32, hidden_dim: int = 128) -> None:
        super().__init__()
        self.noise_dim = noise_dim
        self.net = nn.Sequential(
            nn.Linear(noise_dim + 10, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        condition = F.one_hot(labels, 10).float()
        return self.net(torch.cat([noise, condition], dim=1))


class ConditionalDiscriminator(nn.Module):
    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64 + 10, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        condition = F.one_hot(labels, 10).float()
        return self.net(torch.cat([images, condition], dim=1))


class ConditionalVectorMLP(nn.Module):
    def __init__(self, hidden_dim: int = 192) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64 + 10 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(1)
        condition = F.one_hot(labels, 10).float()
        return self.net(torch.cat([x, condition, t], dim=1))


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _save_sample_grid(samples: np.ndarray, labels: np.ndarray, path: Path, title: str) -> None:
    rows = max(1, min(4, int(np.max(np.bincount(labels))) if labels.size else 1))
    fig, axes = plt.subplots(rows, 10, figsize=(12, 1.4 * rows))
    axes = np.atleast_2d(axes)
    for row in range(rows):
        for label in range(10):
            ax = axes[row, label]
            candidates = np.flatnonzero(labels == label)
            index = candidates[min(row, len(candidates) - 1)]
            ax.imshow(((samples[index].reshape(8, 8) + 1.0) / 2.0).clip(0.0, 1.0), cmap="gray")
            ax.set_title(str(label), fontsize=9)
            ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _evaluate_generated(
    samples: np.ndarray,
    labels: np.ndarray,
    ocr_model: DigitOCRModel,
    class_means: np.ndarray,
) -> tuple[float, float, float, float]:
    images = ((samples.reshape(-1, 8, 8) + 1.0) / 2.0).clip(0.0, 1.0)
    predictions = ocr_model.predict_images(images)
    prompt_accuracy = float(np.mean(predictions == labels))
    diversity = float(np.mean([images[labels == label].var() for label in range(10)]))
    centroid_mse = float(np.mean((samples - class_means[labels]) ** 2))
    rank_score = prompt_accuracy + 0.2 * diversity - 0.05 * centroid_mse
    return prompt_accuracy, diversity, centroid_mse, rank_score


def _train_gan(
    images: np.ndarray,
    labels: np.ndarray,
    eval_labels: np.ndarray,
    quick: bool,
) -> tuple[np.ndarray, float]:
    device = _device()
    generator = ConditionalGenerator().to(device)
    discriminator = ConditionalDiscriminator().to(device)
    dataset = TensorDataset(
        torch.tensor(images, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    g_optim = torch.optim.Adam(generator.parameters(), lr=2e-3, betas=(0.5, 0.999))
    d_optim = torch.optim.Adam(discriminator.parameters(), lr=2e-3, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()
    epochs = 18 if quick else 36

    def fit() -> ConditionalGenerator:
        generator.train()
        discriminator.train()
        for _ in range(epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                batch_size = batch_x.size(0)

                noise = torch.randn(batch_size, generator.noise_dim, device=device)
                fake = generator(noise, batch_y)
                real_targets = torch.full((batch_size, 1), 0.9, device=device)
                fake_targets = torch.zeros((batch_size, 1), device=device)

                d_optim.zero_grad(set_to_none=True)
                d_loss = criterion(discriminator(batch_x, batch_y), real_targets)
                d_loss = d_loss + criterion(discriminator(fake.detach(), batch_y), fake_targets)
                d_loss.backward()
                d_optim.step()

                g_optim.zero_grad(set_to_none=True)
                g_loss = criterion(discriminator(fake, batch_y), real_targets)
                g_loss.backward()
                g_optim.step()
        return generator.eval()

    generator, fit_seconds = timed_run(fit)
    with torch.no_grad():
        label_tensor = torch.tensor(eval_labels, dtype=torch.long, device=device)
        noise = torch.randn(len(eval_labels), generator.noise_dim, device=device)
        samples = generator(noise, label_tensor).cpu().numpy()
    return samples, fit_seconds


def _train_diffusion(
    images: np.ndarray,
    labels: np.ndarray,
    eval_labels: np.ndarray,
    quick: bool,
) -> tuple[np.ndarray, float]:
    device = _device()
    model = ConditionalVectorMLP().to(device)
    dataset = TensorDataset(
        torch.tensor(images, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    steps = 20 if quick else 28
    betas = torch.linspace(1e-4, 5e-2, steps, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    epochs = 22 if quick else 34

    def fit() -> ConditionalVectorMLP:
        model.train()
        for _ in range(epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                t_index = torch.randint(0, steps, (batch_x.size(0),), device=device)
                noise = torch.randn_like(batch_x)
                alpha_bar = alpha_bars[t_index].unsqueeze(1)
                x_t = alpha_bar.sqrt() * batch_x + (1.0 - alpha_bar).sqrt() * noise
                prediction = model(x_t, t_index.float() / (steps - 1), batch_y)
                loss = F.mse_loss(prediction, noise)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        return model.eval()

    model, fit_seconds = timed_run(fit)
    label_tensor = torch.tensor(eval_labels, dtype=torch.long, device=device)
    with torch.no_grad():
        sample = torch.randn(len(eval_labels), 64, device=device)
        for step in reversed(range(steps)):
            t = torch.full((len(eval_labels),), step / max(steps - 1, 1), device=device)
            predicted_noise = model(sample, t, label_tensor)
            alpha = alphas[step]
            alpha_bar = alpha_bars[step]
            beta = betas[step]
            noise = torch.randn_like(sample) if step > 0 else torch.zeros_like(sample)
            sample = (sample - beta * predicted_noise / torch.sqrt(1.0 - alpha_bar)) / torch.sqrt(alpha)
            sample = sample + torch.sqrt(beta) * noise
        samples = sample.clamp(-1.0, 1.0).cpu().numpy()
    return samples, fit_seconds


def _train_flow_matching(
    images: np.ndarray,
    labels: np.ndarray,
    eval_labels: np.ndarray,
    quick: bool,
) -> tuple[np.ndarray, float]:
    device = _device()
    model = ConditionalVectorMLP().to(device)
    dataset = TensorDataset(
        torch.tensor(images, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 18 if quick else 30
    integration_steps = 10 if quick else 16

    def fit() -> ConditionalVectorMLP:
        model.train()
        for _ in range(epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                x0 = torch.randn_like(batch_x)
                t = torch.rand(batch_x.size(0), 1, device=device)
                x_t = (1.0 - t) * x0 + t * batch_x
                target_velocity = batch_x - x0
                prediction = model(x_t, t, batch_y)
                loss = F.mse_loss(prediction, target_velocity)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        return model.eval()

    model, fit_seconds = timed_run(fit)
    label_tensor = torch.tensor(eval_labels, dtype=torch.long, device=device)
    with torch.no_grad():
        sample = torch.randn(len(eval_labels), 64, device=device)
        dt = 1.0 / integration_steps
        for step in range(integration_steps):
            t = torch.full((len(eval_labels), 1), step / integration_steps, device=device)
            velocity = model(sample, t, label_tensor)
            sample = sample + dt * velocity
        samples = sample.clamp(-1.0, 1.0).cpu().numpy()
    return samples, fit_seconds


def run(ocr_model: DigitOCRModel, quick: bool = False) -> ProjectResult:
    images, labels, source = load_digit_images()
    images_flat = (images.reshape(len(images), -1) * 2.0 - 1.0).astype(np.float32)
    class_means = np.vstack([images_flat[labels == label].mean(axis=0) for label in range(10)])
    eval_labels = np.repeat(np.arange(10), 8 if quick else 12)

    records = []
    artifacts = []

    mean_samples = class_means[eval_labels]
    mean_path = ARTIFACTS_DIR / "text_to_image_mean.png"
    _save_sample_grid(mean_samples, eval_labels, mean_path, "Class-mean conditional images")
    mean_accuracy, mean_diversity, mean_mse, mean_rank = _evaluate_generated(
        mean_samples,
        eval_labels,
        ocr_model,
        class_means,
    )
    records.append(
        make_record(
            project=PROJECT_ID,
            dataset=DATASET_NAME,
            source=source,
            task="conditional_image_generation",
            algorithm="class_mean",
            feature_variant="label_to_mean_image",
            optimization="statistical_prototype",
            primary_metric="prompt_accuracy",
            primary_value=mean_accuracy,
            rank_score=mean_rank,
            secondary_metric="diversity",
            secondary_value=mean_diversity,
            tertiary_metric="centroid_mse",
            tertiary_value=mean_mse,
            fit_seconds=0.0,
            notes="Deterministic prototype image per label",
        )
    )
    artifacts.append("artifacts/text_to_image_mean.png")

    for algorithm, trainer, feature_variant, optimization, artifact_name in [
        (
            "conditional_gan",
            _train_gan,
            "noise_plus_label",
            "adversarial_training",
            "text_to_image_gan.png",
        ),
        (
            "diffusion_mlp",
            _train_diffusion,
            "noise_schedule_conditioning",
            "ddpm_style_denoising",
            "text_to_image_diffusion.png",
        ),
        (
            "flow_matching_mlp",
            _train_flow_matching,
            "noise_path_conditioning",
            "ode_flow_matching",
            "text_to_image_flow.png",
        ),
    ]:
        samples, fit_seconds = trainer(images_flat, labels, eval_labels, quick)
        artifact_path = ARTIFACTS_DIR / artifact_name
        _save_sample_grid(samples, eval_labels, artifact_path, algorithm)
        prompt_accuracy, diversity, centroid_mse, rank_score = _evaluate_generated(
            samples,
            eval_labels,
            ocr_model,
            class_means,
        )
        records.append(
            make_record(
                project=PROJECT_ID,
                dataset=DATASET_NAME,
                source=source,
                task="conditional_image_generation",
                algorithm=algorithm,
                feature_variant=feature_variant,
                optimization=optimization,
                primary_metric="prompt_accuracy",
                primary_value=prompt_accuracy,
                rank_score=rank_score,
                secondary_metric="diversity",
                secondary_value=diversity,
                tertiary_metric="centroid_mse",
                tertiary_value=centroid_mse,
                fit_seconds=fit_seconds,
                notes=f"{len(eval_labels)} conditional samples evaluated",
            )
        )
        artifacts.append(f"artifacts/{artifact_name}")

    best = max(records, key=lambda record: record.rank_score)
    summary = (
        f"The best text-to-image proxy was {best.algorithm} with prompt accuracy {best.primary_value:.3f}. "
        "On this tiny digit task, diffusion and flow-style models are practical enough to benchmark directly, "
        "while the mean-image baseline exposes the diversity collapse of purely statistical synthesis."
    )
    recommendation = (
        "For controllable small-scale generation, prefer diffusion or flow-style generators over prototype-only baselines. "
        "Use prompt accuracy together with diversity, because deterministic means can look correct but fail to generate variety."
    )
    key_findings = [
        f"Best prompt-conditioned image score came from {best.algorithm} at {best.primary_value:.3f} prompt accuracy.",
        "The class-mean baseline is stable but usually collapses diversity almost completely.",
        "Diffusion and flow matching are both easy to benchmark on low-dimensional image spaces.",
    ]
    caveats = [
        "This is a conditional digit generator, not a high-resolution natural-image benchmark.",
        "Prompt correctness is scored by a separate OCR classifier rather than human preference.",
        "The GAN baseline is intentionally small and may underperform stronger tuned adversarial models.",
    ]
    return ProjectResult(
        project=PROJECT_ID,
        title=TITLE,
        dataset=DATASET_NAME,
        source=source,
        task="text_to_image",
        records=records,
        summary=summary,
        recommendation=recommendation,
        key_findings=key_findings,
        caveats=caveats,
        artifacts=artifacts,
    )
