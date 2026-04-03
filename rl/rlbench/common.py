from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable, Protocol

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parent.parent
DATA_CACHE = ROOT / "data_cache"
RESULTS_DIR = ROOT / "results"
RANDOM_STATE = 42


@dataclass(slots=True)
class ExperimentRecord:
    project: str
    dataset: str
    source: str
    task: str
    algorithm: str
    feature_variant: str
    optimization: str
    primary_metric: str
    primary_value: float
    rank_score: float
    secondary_metric: str = ""
    secondary_value: float | None = None
    tertiary_metric: str = ""
    tertiary_value: float | None = None
    fit_seconds: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class ProjectResult:
    project: str
    title: str
    dataset: str
    records: list[ExperimentRecord]
    summary: str
    recommendation: str
    key_findings: list[str]
    caveats: list[str]


class DiscreteEnv(Protocol):
    n_states: int
    n_actions: int

    def reset(self, rng: np.random.Generator) -> int:
        ...

    def step(self, action: int, rng: np.random.Generator) -> tuple[int, float, bool, dict[str, float]]:
        ...


def ensure_directories() -> None:
    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def timed_run(fn: Callable[[], object]) -> tuple[object, float]:
    start = perf_counter()
    result = fn()
    return result, perf_counter() - start


def download_to_cache(url: str, relative_path: str, timeout: int = 60) -> Path | None:
    ensure_directories()
    destination = DATA_CACHE / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException:
        return None
    destination.write_bytes(response.content)
    return destination


def existing_path(candidates: Iterable[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def make_record(
    *,
    project: str,
    dataset: str,
    source: str,
    task: str,
    algorithm: str,
    feature_variant: str,
    optimization: str,
    primary_metric: str,
    primary_value: float,
    rank_score: float,
    fit_seconds: float,
    notes: str = "",
    secondary_metric: str = "",
    secondary_value: float | None = None,
    tertiary_metric: str = "",
    tertiary_value: float | None = None,
) -> ExperimentRecord:
    return ExperimentRecord(
        project=project,
        dataset=dataset,
        source=source,
        task=task,
        algorithm=algorithm,
        feature_variant=feature_variant,
        optimization=optimization,
        primary_metric=primary_metric,
        primary_value=float(primary_value),
        rank_score=float(rank_score),
        secondary_metric=secondary_metric,
        secondary_value=None if secondary_value is None else float(secondary_value),
        tertiary_metric=tertiary_metric,
        tertiary_value=None if tertiary_value is None else float(tertiary_value),
        fit_seconds=float(fit_seconds),
        notes=notes,
    )


def records_to_frame(records: Iterable[ExperimentRecord]) -> pd.DataFrame:
    return pd.DataFrame(record.to_dict() for record in records)


def choose_best_record(records: Iterable[ExperimentRecord]) -> ExperimentRecord:
    return max(records, key=lambda record: record.rank_score)


def epsilon_greedy(values: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(0, len(values)))
    best = np.flatnonzero(values == values.max())
    return int(rng.choice(best))


def tabular_q_learning(
    env: DiscreteEnv,
    *,
    episodes: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    epsilon_decay: float = 0.995,
    min_epsilon: float = 0.02,
    max_steps: int = 200,
    seed: int = RANDOM_STATE,
    reward_shaping: Callable[[int, int, float, int, dict[str, float]], float] | None = None,
    action_filter: Callable[[int, int], int] | None = None,
    warm_start: np.ndarray | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q_values = np.zeros((env.n_states, env.n_actions), dtype=float) if warm_start is None else warm_start.copy()
    current_epsilon = epsilon
    for _ in range(episodes):
        state = env.reset(rng)
        for _ in range(max_steps):
            action = epsilon_greedy(q_values[state], current_epsilon, rng)
            if action_filter is not None:
                action = int(action_filter(state, action))
            next_state, reward, done, info = env.step(action, rng)
            shaped_reward = reward
            if reward_shaping is not None:
                shaped_reward = reward_shaping(state, action, reward, next_state, info)
            target = shaped_reward
            if not done:
                target += gamma * float(np.max(q_values[next_state]))
            q_values[state, action] += alpha * (target - q_values[state, action])
            state = next_state
            if done:
                break
        current_epsilon = max(min_epsilon, current_epsilon * epsilon_decay)
    return q_values


def double_q_learning(
    env: DiscreteEnv,
    *,
    episodes: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    epsilon_decay: float = 0.995,
    min_epsilon: float = 0.02,
    max_steps: int = 200,
    seed: int = RANDOM_STATE,
    reward_shaping: Callable[[int, int, float, int, dict[str, float]], float] | None = None,
    action_filter: Callable[[int, int], int] | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q_a = np.zeros((env.n_states, env.n_actions), dtype=float)
    q_b = np.zeros((env.n_states, env.n_actions), dtype=float)
    current_epsilon = epsilon
    for _ in range(episodes):
        state = env.reset(rng)
        for _ in range(max_steps):
            combined = q_a[state] + q_b[state]
            action = epsilon_greedy(combined, current_epsilon, rng)
            if action_filter is not None:
                action = int(action_filter(state, action))
            next_state, reward, done, info = env.step(action, rng)
            shaped_reward = reward if reward_shaping is None else reward_shaping(state, action, reward, next_state, info)
            if rng.random() < 0.5:
                next_action = int(np.argmax(q_a[next_state]))
                target = shaped_reward if done else shaped_reward + gamma * q_b[next_state, next_action]
                q_a[state, action] += alpha * (target - q_a[state, action])
            else:
                next_action = int(np.argmax(q_b[next_state]))
                target = shaped_reward if done else shaped_reward + gamma * q_a[next_state, next_action]
                q_b[state, action] += alpha * (target - q_b[state, action])
            state = next_state
            if done:
                break
        current_epsilon = max(min_epsilon, current_epsilon * epsilon_decay)
    return q_a + q_b


def evaluate_q_policy(
    env: DiscreteEnv,
    q_values: np.ndarray,
    *,
    episodes: int,
    max_steps: int = 200,
    seed: int = RANDOM_STATE,
    action_filter: Callable[[int, int], int] | None = None,
) -> dict[str, float]:
    def policy(state: int) -> int:
        action = int(np.argmax(q_values[state]))
        if action_filter is not None:
            action = int(action_filter(state, action))
        return action

    return evaluate_policy(env, policy, episodes=episodes, max_steps=max_steps, seed=seed)


def evaluate_policy(
    env: DiscreteEnv,
    policy: Callable[[int], int],
    *,
    episodes: int,
    max_steps: int = 200,
    seed: int = RANDOM_STATE,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    returns = []
    successes = []
    violations = []
    steps = []
    for _ in range(episodes):
        state = env.reset(rng)
        total_reward = 0.0
        violation_count = 0.0
        success_flag = 0.0
        step_count = 0
        for step_index in range(1, max_steps + 1):
            action = int(policy(state))
            next_state, reward, done, info = env.step(action, rng)
            total_reward += reward
            violation_count += float(info.get("violation", 0.0))
            success_flag = max(success_flag, float(info.get("success", 0.0)))
            state = next_state
            step_count = step_index
            if done:
                break
        returns.append(total_reward)
        successes.append(success_flag)
        violations.append(violation_count)
        steps.append(step_count)
    return {
        "average_return": float(np.mean(returns)),
        "success_rate": float(np.mean(successes)),
        "violation_rate": float(np.mean(np.asarray(violations) > 0)),
        "mean_violations": float(np.mean(violations)),
        "mean_steps": float(np.mean(steps)),
    }


def transitions_to_buckets(transitions: Iterable[tuple[int, int, float, int, bool]]) -> dict[tuple[int, int], list[tuple[float, int, bool]]]:
    buckets: dict[tuple[int, int], list[tuple[float, int, bool]]] = defaultdict(list)
    for state, action, reward, next_state, done in transitions:
        buckets[(state, action)].append((float(reward), int(next_state), bool(done)))
    return buckets


def offline_fitted_q_iteration(
    transitions: Iterable[tuple[int, int, float, int, bool]],
    *,
    n_states: int,
    n_actions: int,
    gamma: float,
    iterations: int,
    conservative_penalty: float = 0.0,
) -> np.ndarray:
    grouped = transitions_to_buckets(transitions)
    q_values = np.zeros((n_states, n_actions), dtype=float)
    for _ in range(iterations):
        next_q = q_values.copy()
        for state in range(n_states):
            for action in range(n_actions):
                outcomes = grouped.get((state, action), [])
                if not outcomes:
                    next_q[state, action] = q_values[state, action] - conservative_penalty
                    continue
                targets = []
                for reward, next_state, done in outcomes:
                    target = reward
                    if not done:
                        target += gamma * float(np.max(q_values[next_state]))
                    targets.append(target)
                penalty = conservative_penalty / np.sqrt(len(outcomes)) if conservative_penalty else 0.0
                next_q[state, action] = float(np.mean(targets) - penalty)
        q_values = next_q
    return q_values


def offline_behavior_cloning_policy(
    transitions: Iterable[tuple[int, int, float, int, bool]],
    *,
    n_states: int,
    n_actions: int,
) -> np.ndarray:
    counts = np.zeros((n_states, n_actions), dtype=float)
    for state, action, _, _, _ in transitions:
        counts[state, action] += 1.0
    return np.argmax(counts, axis=1)


def offline_reward_model_policy(
    transitions: Iterable[tuple[int, int, float, int, bool]],
    *,
    n_states: int,
    n_actions: int,
) -> np.ndarray:
    sums = np.zeros((n_states, n_actions), dtype=float)
    counts = np.zeros((n_states, n_actions), dtype=float)
    for state, action, reward, _, _ in transitions:
        sums[state, action] += reward
        counts[state, action] += 1.0
    means = np.divide(sums, np.maximum(counts, 1.0))
    unseen_mask = counts == 0
    means[unseen_mask] = means.min() - 1.0
    return np.argmax(means, axis=1)


def action_map_policy(
    action_map: np.ndarray,
    *,
    action_filter: Callable[[int, int], int] | None = None,
) -> Callable[[int], int]:
    def policy(state: int) -> int:
        action = int(action_map[state])
        if action_filter is not None:
            action = int(action_filter(state, action))
        return action

    return policy


def collect_transitions(
    env: DiscreteEnv,
    policy: Callable[[int], int],
    *,
    episodes: int,
    max_steps: int = 200,
    epsilon: float = 0.0,
    seed: int = RANDOM_STATE,
    action_filter: Callable[[int, int], int] | None = None,
) -> list[tuple[int, int, float, int, bool]]:
    rng = np.random.default_rng(seed)
    transitions: list[tuple[int, int, float, int, bool]] = []
    for _ in range(episodes):
        state = env.reset(rng)
        for _ in range(max_steps):
            action = int(policy(state))
            if epsilon and rng.random() < epsilon:
                action = int(rng.integers(0, env.n_actions))
            if action_filter is not None:
                action = int(action_filter(state, action))
            next_state, reward, done, _ = env.step(action, rng)
            transitions.append((state, action, reward, next_state, done))
            state = next_state
            if done:
                break
    return transitions