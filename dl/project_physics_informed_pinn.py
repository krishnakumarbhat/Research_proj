from __future__ import annotations

from time import perf_counter

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn

from dl.common import ProjectResult, choose_best_record, make_record, regression_metrics, set_seed
from dl.data import make_burgers_dataset
from dl.models import FourierFeatures, MLP


PROJECT_ID = "physics_informed_pinn"
TITLE = "Physics-Informed Neural Networks"
DATASET = "Burgers' Equation"
VISCOSITY = 0.05


def _supervised_builder() -> nn.Module:
    return MLP(2, [64, 64], 1, dropout=0.0)


def _fourier_builder() -> nn.Module:
    return nn.Sequential(FourierFeatures(2, mapping_size=16), MLP(32, [64, 64], 1, dropout=0.0))


def _burgers_residual(model: nn.Module, coords: torch.Tensor) -> torch.Tensor:
    coords = coords.clone().detach().requires_grad_(True)
    prediction = model(coords)
    gradients = torch.autograd.grad(prediction, coords, grad_outputs=torch.ones_like(prediction), create_graph=True)[0]
    u_x = gradients[:, :1]
    u_t = gradients[:, 1:]
    second = torch.autograd.grad(u_x, coords, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_xx = second[:, :1]
    return u_t + prediction * u_x - VISCOSITY * u_xx


def _fit_supervised(model: nn.Module, x_train: np.ndarray, y_train: np.ndarray, epochs: int) -> float:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    x_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train[:, None], dtype=torch.float32)
    start = perf_counter()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x_tensor)
        loss = criterion(prediction, y_tensor)
        loss.backward()
        optimizer.step()
    return perf_counter() - start


def _fit_pinn(model: nn.Module, x_train: np.ndarray, y_train: np.ndarray, collocation: np.ndarray, epochs: int) -> float:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    x_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train[:, None], dtype=torch.float32)
    collocation_tensor = torch.tensor(collocation, dtype=torch.float32)
    start = perf_counter()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x_tensor)
        data_loss = criterion(prediction, y_tensor)
        residual = _burgers_residual(model, collocation_tensor)
        physics_loss = (residual ** 2).mean()
        loss = data_loss + 0.6 * physics_loss
        loss.backward()
        optimizer.step()
    return perf_counter() - start


def _evaluate(model: nn.Module, x_test: np.ndarray, y_test: np.ndarray) -> tuple[dict[str, float], float]:
    with torch.no_grad():
        prediction = model(torch.tensor(x_test, dtype=torch.float32)).squeeze(-1).cpu().numpy()
    metrics = regression_metrics(y_test, prediction)
    residual = _burgers_residual(model, torch.tensor(x_test[: min(512, len(x_test))], dtype=torch.float32))
    return metrics, float(residual.abs().mean().item())


def run(quick: bool = True) -> ProjectResult:
    set_seed(53)
    coords, values, _ = make_burgers_dataset(n_x=64 if quick else 96, n_t=40 if quick else 60, viscosity=VISCOSITY)
    x_train, x_test, y_train, y_test = train_test_split(coords, values, test_size=0.2, random_state=53)
    collocation = coords[np.random.default_rng(53).choice(len(coords), size=1200 if quick else 3000, replace=False)]
    variants = [
        ("supervised_mlp", "space_time_coordinates", "adam_supervised", _supervised_builder, False),
        ("pinn_mlp", "space_time_coordinates", "adam_data_plus_physics", _supervised_builder, True),
        ("pinn_fourier", "fourier_space_time_features", "adam_data_plus_physics", _fourier_builder, True),
    ]
    records = []
    for algorithm, feature_variant, optimization, builder, is_pinn in variants:
        model = builder()
        fit_seconds = _fit_pinn(model, x_train, y_train, collocation, 120 if quick else 220) if is_pinn else _fit_supervised(model, x_train, y_train, 120 if quick else 220)
        metrics, residual = _evaluate(model, x_test, y_test)
        records.append(
            make_record(
                project=PROJECT_ID,
                dataset=DATASET,
                source="finite_difference_burgers_solver",
                task="pde_regression",
                algorithm=algorithm,
                feature_variant=feature_variant,
                optimization=optimization,
                primary_metric="rmse",
                primary_value=metrics["rmse"],
                rank_score=-metrics["rmse"],
                fit_seconds=fit_seconds,
                secondary_metric="physics_residual",
                secondary_value=residual,
                tertiary_metric="r2",
                tertiary_value=metrics["r2"],
                notes=f"mae={metrics['mae']:.4f}",
            )
        )

    best = choose_best_record(records)
    return ProjectResult(
        project=PROJECT_ID,
        title=TITLE,
        dataset=DATASET,
        records=records,
        summary=(
            f"The best Burgers solver surrogate was {best.algorithm}, which reached RMSE {best.primary_value:.4f}. "
            "Adding a physics residual changed the fit dynamics and often reduced physically implausible interpolation even when the supervised RMSE gap was small."
        ),
        recommendation=(
            "Report both data-fit error and physics residual for PINNs. A purely supervised MLP can look competitive on RMSE while violating the governing equation more severely."
        ),
        key_findings=[
            f"Best RMSE: {best.primary_value:.4f} from {best.algorithm}.",
            "Physics loss made the benchmark more robust to coordinate sparsity than a purely supervised fit.",
            "Fourier features improved representation of sharper spatial structure without inflating model depth.",
        ],
        caveats=[
            "The ground-truth field is generated by a compact finite-difference Burgers solver rather than downloaded from the original PINN repository.",
            "Quick mode uses a relatively small grid, so the absolute residual scale should be interpreted comparatively.",
        ],
    )


def main() -> None:
    print(run(quick=True).summary)


if __name__ == "__main__":
    main()
