from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, load_diabetes
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from ml.common import DATA_CACHE, ProjectResult, make_record, timed_run


PROJECT_ID = "evolutionary_feature_engineering"
TITLE = "Automated Feature Engineering via Evolutionary Algorithms"
DATASET_NAME = "House Prices / Ames-style Regression"


def _load_dataset() -> tuple[pd.DataFrame, str, str]:
    local_candidates = [
        DATA_CACHE / "house_prices" / "train.csv",
        Path(__file__).resolve().parent / "train.csv",
    ]
    local_path = next((path for path in local_candidates if path.exists()), None)
    if local_path is not None:
        frame = pd.read_csv(local_path)
        target = "SalePrice"
        source = "local_kaggle_copy"
    else:
        try:
            dataset = fetch_openml(name="house_prices", version=1, as_frame=True)
            frame = dataset.frame.copy()
            target = dataset.target.name
            source = "openml"
        except Exception:
            dataset = load_diabetes(as_frame=True)
            frame = dataset.frame.copy()
            target = dataset.target.name
            source = "sklearn_diabetes_fallback"
    return frame, target, source


def _safe_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / np.where(np.abs(b) < 1e-6, 1.0, b)


def _apply_formula(frame: pd.DataFrame, formula: tuple[str, str, str]) -> pd.Series:
    operator, left, right = formula
    if operator == "add":
        return frame[left] + frame[right]
    if operator == "sub":
        return frame[left] - frame[right]
    if operator == "mul":
        return frame[left] * frame[right]
    if operator == "div":
        return _safe_ratio(frame[left], frame[right])
    raise ValueError(operator)


def _score_formulas(train_x, train_y, valid_x, valid_y, formulas):
    engineered_train = train_x.copy()
    engineered_valid = valid_x.copy()
    for index, formula in enumerate(formulas):
        feature_name = f"engineered_{index}"
        engineered_train[feature_name] = _apply_formula(train_x, formula)
        engineered_valid[feature_name] = _apply_formula(valid_x, formula)
    model = Ridge(alpha=1.0)
    model.fit(engineered_train, train_y)
    predictions = model.predict(engineered_valid)
    return float(np.sqrt(mean_squared_error(valid_y, predictions)))


def _evolve_formulas(train_x, train_y, valid_x, valid_y, quick: bool) -> list[tuple[str, str, str]]:
    rng = np.random.default_rng(42)
    numeric_columns = train_x.columns.tolist()
    population_size = 10 if quick else 16
    generations = 6 if quick else 10
    feature_count = 4 if quick else 6
    operators = ["add", "sub", "mul", "div"]

    def random_formula():
        left, right = rng.choice(numeric_columns, size=2, replace=False)
        return str(rng.choice(operators)), str(left), str(right)

    population = [[random_formula() for _ in range(feature_count)] for _ in range(population_size)]
    best_formulas = population[0]
    best_score = float("inf")
    for _ in range(generations):
        scored = []
        for formulas in population:
            score = _score_formulas(train_x, train_y, valid_x, valid_y, formulas)
            scored.append((score, formulas))
            if score < best_score:
                best_score = score
                best_formulas = formulas
        scored.sort(key=lambda item: item[0])
        parents = [formulas for _, formulas in scored[: max(2, population_size // 3)]]
        population = parents.copy()
        while len(population) < population_size:
            parent_a = parents[int(rng.integers(0, len(parents)))]
            parent_b = parents[int(rng.integers(0, len(parents)))]
            cut = rng.integers(1, feature_count)
            child = list(parent_a[:cut]) + list(parent_b[cut:])
            mutate_index = rng.integers(0, feature_count)
            child[mutate_index] = random_formula()
            population.append(child)
    return best_formulas


def run(quick: bool = True) -> ProjectResult:
    frame, target, source = _load_dataset()
    numeric_frame = frame.select_dtypes(include=[np.number]).dropna().reset_index(drop=True)
    target_series = numeric_frame[target]
    feature_frame = numeric_frame.drop(columns=[target])
    feature_frame = feature_frame.loc[:, feature_frame.nunique() > 1]

    train_x, test_x, train_y, test_y = train_test_split(feature_frame, target_series, test_size=0.2, random_state=42)
    fit_x, valid_x, fit_y, valid_y = train_test_split(train_x, train_y, test_size=0.25, random_state=42)
    formulas = _evolve_formulas(fit_x, fit_y, valid_x, valid_y, quick)

    engineered_train = train_x.copy()
    engineered_test = test_x.copy()
    for index, formula in enumerate(formulas):
        feature_name = f"engineered_{index}"
        engineered_train[feature_name] = _apply_formula(train_x, formula)
        engineered_test[feature_name] = _apply_formula(test_x, formula)

    records = []
    baselines = {
        "ridge_raw": (Ridge(alpha=1.0), train_x, test_x, "raw_numeric"),
        "random_forest_raw": (RandomForestRegressor(n_estimators=220 if not quick else 140, random_state=42, n_jobs=-1), train_x, test_x, "raw_numeric"),
        "ridge_evolved": (Ridge(alpha=1.0), engineered_train, engineered_test, "evolved_interactions"),
        "hist_gradient_evolved": (
            HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, max_iter=240 if not quick else 140, random_state=42),
            engineered_train,
            engineered_test,
            "evolved_interactions",
        ),
    }
    for algorithm, (model, used_train, used_test, variant) in baselines.items():
        _, fit_seconds = timed_run(lambda current=model, x=used_train: current.fit(x, train_y))
        predictions = model.predict(used_test)
        rmse = float(np.sqrt(mean_squared_error(test_y, predictions)))
        records.append(
            make_record(
                project=PROJECT_ID,
                dataset=DATASET_NAME,
                source=source,
                task="tabular_regression",
                algorithm=algorithm,
                feature_variant=variant,
                optimization="evolutionary_formula_search" if "evolved" in algorithm else "baseline_features",
                primary_metric="rmse",
                primary_value=rmse,
                rank_score=-rmse,
                secondary_metric="feature_count",
                secondary_value=float(used_train.shape[1]),
                tertiary_metric="engineered_features",
                tertiary_value=float(len(formulas) if "evolved" in algorithm else 0),
                fit_seconds=fit_seconds,
                notes="; ".join(f"{op}({a},{b})" for op, a, b in formulas[:3]) if "evolved" in algorithm else "No engineered formulas",
            )
        )

    best = max(records, key=lambda record: record.rank_score)
    return ProjectResult(
        project=PROJECT_ID,
        title=TITLE,
        dataset=DATASET_NAME,
        records=records,
        summary=(
            f"Evolutionary interaction search produced {len(formulas)} engineered features. The best downstream model was {best.algorithm} with RMSE {best.primary_value:.3f}."
        ),
        recommendation=(
            "Use a small evolutionary search when the feature space is rich but hand-engineering is slow. Even a short CPU-only run can surface interactions that linear baselines miss."
        ),
        key_findings=[
            f"Best model: {best.algorithm} with RMSE {best.primary_value:.3f}.",
            f"Top evolved formulas included {'; '.join(f'{op}({a},{b})' for op, a, b in formulas[:3])}.",
            "Feature search is most useful when paired with a simpler downstream learner that benefits from crafted interactions.",
        ],
        caveats=[
            "If the Kaggle house-prices dataset is unavailable, the runner uses OpenML House Prices and then a diabetes-regression fallback.",
            "The search explores only arithmetic pairwise formulas, not arbitrary symbolic programs.",
            "A single validation split is used as the fitness function for speed.",
        ],
    )


if __name__ == "__main__":
    result = run(quick=True)
    print(result.summary)
