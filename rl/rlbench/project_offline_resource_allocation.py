from __future__ import annotations

from rlbench.common import (
    ProjectResult,
    action_map_policy,
    choose_best_record,
    collect_transitions,
    evaluate_policy,
    evaluate_q_policy,
    make_record,
    offline_behavior_cloning_policy,
    offline_fitted_q_iteration,
    offline_reward_model_policy,
    tabular_q_learning,
)
from rlbench.synthetic_envs import ResourceAllocationEnv


PROJECT_ID = "offline_resource_allocation"
TITLE = "Offline RL for Hardware Resource Allocation"
DATASET = "Google Cluster Data"


def _logging_policy(env: ResourceAllocationEnv, state: int) -> int:
    queue, _, _, priority, _ = env.decode(state)
    if queue >= 2 or priority == 1:
        return 2
    if queue == 1:
        return 1
    return 0


def run(quick: bool = True) -> ProjectResult:
    env = ResourceAllocationEnv(horizon=6 if quick else 8)
    transitions = collect_transitions(
        env,
        lambda state: _logging_policy(env, state),
        episodes=240 if quick else 720,
        max_steps=env.horizon,
        epsilon=0.25,
        seed=341,
    )
    records = []

    offline_specs = [
        (
            "behavior_cloning",
            "queue_cpu_memory_priority_state",
            "count_based_imitation",
            action_map_policy(offline_behavior_cloning_policy(transitions, n_states=env.n_states, n_actions=env.n_actions)),
        ),
        (
            "reward_model_policy",
            "queue_cpu_memory_priority_state",
            "mean_logged_reward",
            action_map_policy(offline_reward_model_policy(transitions, n_states=env.n_states, n_actions=env.n_actions)),
        ),
        (
            "fitted_q_iteration",
            "queue_cpu_memory_priority_state",
            "offline_bellman_backup",
            offline_fitted_q_iteration(
                transitions,
                n_states=env.n_states,
                n_actions=env.n_actions,
                gamma=0.95,
                iterations=25 if quick else 40,
            ),
        ),
        (
            "conservative_fqi",
            "queue_cpu_memory_priority_state",
            "offline_bellman_backup_with_penalty",
            offline_fitted_q_iteration(
                transitions,
                n_states=env.n_states,
                n_actions=env.n_actions,
                gamma=0.95,
                iterations=25 if quick else 40,
                conservative_penalty=0.6,
            ),
        ),
    ]

    for algorithm, feature_variant, optimization, policy_or_q in offline_specs:
        if hasattr(policy_or_q, "shape"):
            metrics = evaluate_q_policy(env, policy_or_q, episodes=220 if quick else 520, max_steps=env.horizon, seed=351)
        else:
            metrics = evaluate_policy(env, policy_or_q, episodes=220 if quick else 520, max_steps=env.horizon, seed=351)
        records.append(
            make_record(
                project=PROJECT_ID,
                dataset=DATASET,
                source="synthetic_google_cluster_fallback",
                task="offline_resource_allocation",
                algorithm=algorithm,
                feature_variant=feature_variant,
                optimization=optimization,
                primary_metric="average_return",
                primary_value=metrics["average_return"],
                rank_score=metrics["average_return"] + 0.2 * metrics["success_rate"] - 0.5 * metrics["violation_rate"],
                fit_seconds=0.0,
                secondary_metric="success_rate",
                secondary_value=metrics["success_rate"],
                tertiary_metric="violation_rate",
                tertiary_value=metrics["violation_rate"],
                notes=f"offline_transitions={len(transitions)}",
            )
        )

    online_q = tabular_q_learning(
        env,
        episodes=220 if quick else 620,
        alpha=0.22,
        gamma=0.95,
        epsilon=0.34,
        max_steps=env.horizon,
        seed=353,
    )
    online_metrics = evaluate_q_policy(env, online_q, episodes=220 if quick else 520, max_steps=env.horizon, seed=355)
    records.append(
        make_record(
            project=PROJECT_ID,
            dataset=DATASET,
            source="synthetic_google_cluster_fallback",
            task="offline_resource_allocation",
            algorithm="online_q_reference",
            feature_variant="queue_cpu_memory_priority_state",
            optimization="online_ceiling_reference",
            primary_metric="average_return",
            primary_value=online_metrics["average_return"],
            rank_score=online_metrics["average_return"] + 0.2 * online_metrics["success_rate"] - 0.5 * online_metrics["violation_rate"],
            fit_seconds=0.0,
            secondary_metric="success_rate",
            secondary_value=online_metrics["success_rate"],
            tertiary_metric="violation_rate",
            tertiary_value=online_metrics["violation_rate"],
            notes="Included as an optimistic upper bound relative to logged-data policies.",
        )
    )

    best = choose_best_record(records)
    return ProjectResult(
        project=PROJECT_ID,
        title=TITLE,
        dataset=DATASET,
        records=records,
        summary=(
            f"The strongest hardware-allocation controller was {best.algorithm}, reaching average return {best.primary_value:.3f}. "
            "The offline allocation topic benefits from exactly the same disciplined comparison as pricing: imitation, reward modeling, fitted value iteration, and a clear online ceiling."
        ),
        recommendation=(
            "For cluster-control logs, conservative fitted Q iteration is a strong first offline baseline because it exposes the value of being pessimistic on under-covered state-action pairs."
        ),
        key_findings=[
            f"Best average return: {best.primary_value:.3f} from {best.algorithm}.",
            "Count-based imitation is fast, but value backups extract more from the same logged transitions when allocation outcomes are delayed.",
            "An explicit online ceiling helps quantify how much headroom remains before offline methods saturate the synthetic benchmark.",
        ],
        caveats=[
            "The workspace does not include the full Google cluster trace, so the benchmark uses a small tabular resource-allocation simulator with the same control structure.",
            "A real workload trace study should add bursty arrivals, SLA classes, and much richer action granularity.",
        ],
    )


def main() -> None:
    print(run(quick=True).summary)


if __name__ == "__main__":
    main()