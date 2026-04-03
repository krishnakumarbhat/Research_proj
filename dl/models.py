from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Flatten()]
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(current_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class _SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(inputs)
        return inputs.sign()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (inputs,) = ctx.saved_tensors
        mask = (inputs.abs() <= 1.0).to(grad_output.dtype)
        return grad_output * mask


class SignActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _SignSTE.apply(x)


class SignMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            SignActivation(),
            nn.Linear(hidden_dim, hidden_dim),
            SignActivation(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class Conv1DClassifier(nn.Module):
    def __init__(self, in_channels: int, seq_len: int, num_classes: int, hidden_channels: int = 32) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(hidden_channels * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x).squeeze(-1)
        return self.classifier(features)


class DepthwiseSeparableConv1DClassifier(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, hidden_channels: int = 24) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels),
            nn.Conv1d(in_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1, groups=hidden_channels),
            nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(hidden_channels * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x).squeeze(-1)
        return self.classifier(features)


class ResidualConv1DClassifier(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, hidden_channels: int = 32) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.block = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
        )
        self.pool = nn.Sequential(nn.ReLU(), nn.MaxPool1d(2), nn.AdaptiveAvgPool1d(1))
        self.head = nn.Linear(hidden_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stem = self.stem(x)
        residual = stem + self.block(stem)
        pooled = self.pool(residual).squeeze(-1)
        return self.head(pooled)


class FusionConv1DClassifier(nn.Module):
    def __init__(self, accel_channels: int, gyro_channels: int, num_classes: int) -> None:
        super().__init__()
        self.accel_branch = nn.Sequential(
            nn.Conv1d(accel_channels, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.gyro_branch = nn.Sequential(
            nn.Conv1d(gyro_channels, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        accel = self.accel_branch(x[:, :3]).squeeze(-1)
        gyro = self.gyro_branch(x[:, 3:]).squeeze(-1)
        return self.head(torch.cat([accel, gyro], dim=1))


class GRUModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence = x.transpose(1, 2) if x.ndim == 3 else x
        _, hidden = self.gru(sequence)
        return self.head(hidden[-1])


class LiquidModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.state_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence = x.transpose(1, 2) if x.ndim == 3 else x
        batch_size = sequence.shape[0]
        hidden = torch.zeros(batch_size, self.state_proj.out_features, device=sequence.device)
        for time_index in range(sequence.shape[1]):
            current = sequence[:, time_index]
            delta = torch.sigmoid(self.gate(torch.cat([current, hidden], dim=1)))
            proposal = torch.tanh(self.input_proj(current) + self.state_proj(hidden))
            hidden = hidden + delta * (proposal - hidden)
        return self.head(hidden)


class TinyCNN2D(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, width: int = 16) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(width * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x).flatten(start_dim=1)
        return self.head(features)


class DepthwiseCNN2D(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, width: int = 12) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, width, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width, kernel_size=3, padding=1, groups=width),
            nn.Conv2d(width, width * 2, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(width * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x).flatten(start_dim=1)
        return self.head(features)


class VectorAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 16) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flattened = x.flatten(start_dim=1)
        latent = self.encoder(flattened)
        reconstruction = self.decoder(latent)
        return reconstruction.reshape_as(flattened).reshape_as(x)


class ConvAutoencoder1D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(channels, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(8, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded[:, :, : x.shape[-1]]


class FourierFeatures(nn.Module):
    def __init__(self, input_dim: int, mapping_size: int = 16) -> None:
        super().__init__()
        self.register_buffer("weights", torch.randn(input_dim, mapping_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projection = 2 * torch.pi * x @ self.weights
        return torch.cat([torch.sin(projection), torch.cos(projection)], dim=1)
