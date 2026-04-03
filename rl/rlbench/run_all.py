from __future__ import annotations

import argparse
from pathlib import Path

from rlbench.common import ProjectResult, ROOT, ensure_directories
from rlbench.project_curriculum_jssp import run as run_curriculum_jssp
from rlbench.project_dynamic_pricing import run as run_dynamic_pricing
from rlbench.project_execution_trading import run as run_execution_trading
from rlbench.project_hierarchical_logistics import run as run_hierarchical_logistics
from rlbench.project_hvac_optimization import run as run_hvac_optimization
from rlbench.project_mapf_optimization import run as run_mapf_optimization
from rlbench.project_meta_market_adaptation import run as run_meta_market_adaptation
from rlbench.project_multi_agent_traffic import run as run_multi_agent_traffic
from rlbench.project_neuro_symbolic_planning import run as run_neuro_symbolic_planning
from rlbench.project_offline_resource_allocation import run as run_offline_resource_allocation
from rlbench.project_reward_shaping_energy import run as run_reward_shaping_energy
from rlbench.project_safe_chemical import run as run_safe_chemical
from rlbench.project_safe_deterministic_envelopes import run as run_safe_deterministic_envelopes
from rlbench.project_sim_to_real_transfer import run as run_sim_to_real_transfer
from rlbench.project_supply_chain_routing import run as run_supply_chain_routing
from rlbench.reporting import write_csv_outputs, write_markdown_report


PROJECT_RUNNERS = {
    "dynamic_pricing": run_dynamic_pricing,
    "supply_chain_routing": run_supply_chain_routing,
    "multi_agent_traffic": run_multi_agent_traffic,
    "reward_shaping_energy": run_reward_shaping_energy,
    "execution_trading": run_execution_trading,
    "safe_chemical": run_safe_chemical,
    "curriculum_jssp": run_curriculum_jssp,
    "hierarchical_logistics": run_hierarchical_logistics,
    "meta_market_adaptation": run_meta_market_adaptation,
    "hvac_optimization": run_hvac_optimization,
    "safe_deterministic_envelopes": run_safe_deterministic_envelopes,
    "sim_to_real_transfer": run_sim_to_real_transfer,
    "offline_resource_allocation": run_offline_resource_allocation,
    "neuro_symbolic_planning": run_neuro_symbolic_planning,
    "mapf_optimization": run_mapf_optimization,
}


def execute(
    *,
    quick: bool = True,
    selected_projects: list[str] | None = None,
    continue_on_error: bool = False,
    write_outputs: bool = True,
) -> tuple[list[ProjectResult], list[tuple[str, str]], dict[str, Path]]:
    ensure_directories()
    selected = selected_projects or list(PROJECT_RUNNERS.keys())
    results: list[ProjectResult] = []
    failures: list[tuple[str, str]] = []
    for project in selected:
        if project not in PROJECT_RUNNERS:
            message = f"Unknown project id: {project}"
            if continue_on_error:
                failures.append((project, message))
                continue
            raise KeyError(message)
        try:
            results.append(PROJECT_RUNNERS[project](quick=quick))
        except Exception as exc:
            failures.append((project, str(exc)))
            if not continue_on_error:
                raise

    artifacts: dict[str, Path] = {}
    if write_outputs and results:
        records_path, summary_path = write_csv_outputs(results, ROOT / "results")
        report_path = write_markdown_report(results, ROOT / "research_report.md")
        artifacts = {
            "records_csv": records_path,
            "summary_csv": summary_path,
            "report_md": report_path,
        }
    return results, failures, artifacts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CPU-first RL research benchmark suite.")
    parser.add_argument("--projects", default="all", help="Comma-separated project ids to run, or 'all'.")
    parser.add_argument("--full", action="store_true", help="Run larger training budgets instead of quick mode.")
    parser.add_argument("--continue-on-error", action="store_true", help="Keep running remaining projects if one fails.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    selected = None if args.projects == "all" else [item.strip() for item in args.projects.split(",") if item.strip()]
    results, failures, artifacts = execute(
        quick=not args.full,
        selected_projects=selected,
        continue_on_error=args.continue_on_error,
        write_outputs=True,
    )
    print(f"Completed {len(results)} project runs.")
    if failures:
        print("Failures:")
        for project, message in failures:
            print(f"- {project}: {message}")
    if artifacts:
        print("Artifacts:")
        for name, path in artifacts.items():
            print(f"- {name}: {path}")


if __name__ == "__main__":
    main()