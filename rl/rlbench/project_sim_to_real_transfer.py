from __future__ import annotations

from rlbench.common import ProjectResult, choose_best_record, evaluate_policy, evaluate_q_policy, make_record, tabular_q_learning
from rlbench.synthetic_envs import SimTransferEnv


PROJECT_ID = "sim_to_real_transfer"
TITLE = "Sim-to-Real Transfer for Minimal Binaries"
DATASET = "MicroRLEnv"


def _robust_rule(env: SimTransferEnv, state: int) -> int:
    offset, velocity, _, _ = env.decode(state)
    signed_offset = offset - 2
    signed_velocity = velocity - 2
    if signed_offset > 0 or signed_velocity > 1:
        return 0
    if signed_offset < 0 or signed_velocity < -1:
        return 2
    return 1


def _train_sequence(envs: list[SimTransferEnv], quick: bool, warm_start=None, seed_offset: int = 0):
    q_values = warm_start
    for index, env in enumerate(envs):
        q_values = tabular_q_learning(
            env,
            episodes=150 if quick else 420,
            alpha=0.24,
            gamma=0.95,
            epsilon=0.34,
            max_steps=env.horizon,
            seed=301 + seed_offset + index,
            warm_start=q_values,
        )
    return q_values


def run(quick: bool = True) -> ProjectResult:
    sim_env = SimTransferEnv(horizon=6 if quick else 8, reality_gap=0.0)
    target_env = SimTransferEnv(horizon=6 if quick else 8, reality_gap=0.18)
    randomized_envs = [
        SimTransferEnv(horizon=6 if quick else 8, reality_gap=-0.12),
        SimTransferEnv(horizon=6 if quick else 8, reality_gap=0.0),
        SimTransferEnv(horizon=6 if quick else 8, reality_gap=0.12),
    ]

    sim_only_q = _train_sequence([sim_env], quick)
    domain_rand_q = _train_sequence(randomized_envs, quick, seed_offset=10)
    finetuned_q = _train_sequence([target_env], quick, warm_start=sim_only_q, seed_offset=20)

    specs = [
        ("sim_only_q", "offset_velocity_terrain_state", "train_in_nominal_sim_only", sim_only_q),
        ("domain_randomized_q", "offset_velocity_terrain_state", "train_across_gap_samples", domain_rand_q),
        ("sim_plus_target_finetune", "offset_velocity_terrain_state", "warm_start_and_short_real_adaptation", finetuned_q),
    ]

    records = []
    for algorithm, feature_variant, optimization, q_values in specs:
        metrics = evaluate_q_policy(target_env, q_values, episodes=220 if quick else 520, max_steps=target_env.horizon, seed=321)
        records.append(
            make_record(
                project=PROJECT_ID,
                dataset=DATASET,
                source="synthetic_pybullet_transfer_fallback",
                task="sim_to_real_transfer",
                algorithm=algorithm,
                feature_variant=feature_variant,
                optimization=optimization,
                primary_metric="average_return",
                primary_value=metrics["average_return"],
                rank_score=metrics["average_return"] + 0.3 * metrics["success_rate"],
                fit_seconds=0.0,
                secondary_metric="success_rate",
                secondary_value=metrics["success_rate"],
                tertiary_metric="mean_steps",
                tertiary_value=metrics["mean_steps"],
                notes=f"q_table_bytes={q_values.nbytes}",
            )
        )

    rule_metrics = evaluate_policy(target_env, lambda state: _robust_rule(target_env, state), episodes=220 if quick else 520, max_steps=target_env.horizon, seed=323)
    records.append(
        make_record(
            project=PROJECT_ID,
            dataset=DATASET,
            source="synthetic_pybullet_transfer_fallback",
            task="sim_to_real_transfer",
            algorithm="robust_rule_controller",
            feature_variant="offset_velocity_terrain_state",
            optimization="handcrafted_gap_robust_rule",
            primary_metric="average_return",
            primary_value=rule_metrics["average_return"],
            rank_score=rule_metrics["average_return"] + 0.3 * rule_metrics["success_rate"],
            fit_seconds=0.0,
            secondary_metric="success_rate",
            secondary_value=rule_metrics["success_rate"],
            tertiary_metric="mean_steps",
            tertiary_value=rule_metrics["mean_steps"],
            notes="Minimal-controller baseline for deployment discussions.",
        )
    )

    best = choose_best_record(records)
    return ProjectResult(
        project=PROJECT_ID,
        title=TITLE,
        dataset=DATASET,
        records=records,
        summary=(
            f"The strongest sim-to-real strategy was {best.algorithm}, reaching average return {best.primary_value:.3f}. "
            "The result separates three common transfer tactics cleanly: train only in sim, randomize the sim, or warm-start then fine-tune on the target dynamics."
        ),
        recommendation=(
            "If deployment space is tight, log the controller footprint and compare domain randomization against short real-world fine-tuning. That trade-off is often the practical decision point."
        ),
        key_findings=[
            f"Best average return on the target dynamics: {best.primary_value:.3f} from {best.algorithm}.",
            "Domain randomization provides a useful zero-shot transfer baseline, but short target adaptation is often the cleaner way to close the final gap.",
            "The tiny rule controller remains worth reporting because minimal binaries are part of the actual deployment requirement here.",
        ],
        caveats=[
            "The workspace uses a tiny tabular transfer problem rather than a full PyBullet joint-control training loop.",
            "A stronger study would export and benchmark actual policy binaries on-device instead of only logging Q-table size.",
        ],
    )


def main() -> None:
    print(run(quick=True).summary)


if __name__ == "__main__":
    main()