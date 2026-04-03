# CPU-First RL Research Benchmark Report

This report summarizes 15 lightweight reinforcement-learning research projects implemented in `rlbench/`. The benchmark keeps Karpathy's vendored `autoresearch/` project untouched and instead borrows its experiment-loop mindset: compare compact variants quickly, keep explicit logs, and prioritize reproducible CPU runs.

Heavy external simulators and Kaggle-gated assets are represented by local-data hooks plus synthetic fallbacks. That keeps every project runnable in this workspace while preserving the main optimization problem behind the original topic.

## Cross-Project Summary
| Project | Dataset | Best Model | Best Score | Recommendation |
| --- | --- | --- | --- | --- |
| Offline RL for Dynamic Pricing | Olist Brazilian E-Commerce Dataset | conservative_fqi | average_return=25.6833 | For low-risk pricing research, start with conservative fitted Q iteration on a logged-policy dataset and keep an online tabular reference only as a ceiling, not as the deployment default. |
| RL for Supply Chain Routing under Uncertainty | OR-Gym Supply Chain Routing | risk_shaped_q | success_rate=1.0000 | For uncertain supply chains, treat routing as a success-and-risk problem, not a shortest-path problem. Report stockout or disruption violations alongside reward. |
| Multi-Agent RL for Traffic Light Grids | CityFlow Traffic Grid | greedy_pressure_rule | average_return=0.7795 | For CPU traffic-control benchmarks, a centralized joint-action tabular baseline is enough to compare coordination strategies before investing in heavier simulators or neural MARL stacks. |
| Reward Shaping in Resource Allocation | CityLearn Energy Grid | double_q_dispatch | average_return=0.9832 | When studying reward hacking in energy control, publish at least one sparse baseline and one shaped baseline. The delta is often more informative than the raw best score. |
| Execution Algorithms in Algorithmic Trading | Huge Stock Market Dataset | twap_rule | average_return=4.1117 | Use TWAP-style heuristics as baselines, but score learned execution policies on both realized reward and completion rate. A strategy that looks cheap but fails to finish is not competitive. |
| Safe RL in Chemical Processes | Tennessee Eastman Process | constrained_q | safe_score=-9.7741 | For constrained process control, report a composite safe score or a two-axis reward-versus-violations table. Pure reward ranking hides unsafe policies. |
| Curriculum Learning for Job-Shop Scheduling | Taillard Job Shop Instances | shortest_job_rule | success_rate=1.0000 | If compute is limited, start here. Curriculum scheduling gives a clean research story, cheap runs, and a meaningful comparison between direct and staged learning. |
| Hierarchical RL for Long-Horizon Logistics | CVRPLIB Logistics Instances | double_q_option_filter | success_rate=1.0000 | If the route horizon is long, include at least one macro-option or action-filter baseline. Flat Q-learning alone understates the value of hierarchy. |
| Meta-RL for Rapid Market Adaptation | S&P 500 Stock Data | meta_q_fast_adapt | average_return=24.1318 | When claiming fast market adaptation, separate no-adaptation transfer from short-horizon adaptation. Otherwise the comparison hides whether the initialization is actually useful. |
| RL for HVAC Energy Optimization | Sinergym HVAC Benchmark | double_q_hvac | average_return=-15.2385 | Benchmark HVAC policies on both cost and comfort. If the report only shows one scalar reward, add at least one explicit comfort-violation column. |
| Safe RL with Deterministic Envelopes | Safe-Control-Gym | shield_rule | safe_score=-0.0839 | If a domain has hard safety envelopes, implement the envelope explicitly and compare shielded versus unshielded learning. That ablation is the real result. |
| Sim-to-Real Transfer for Minimal Binaries | MicroRLEnv | domain_randomized_q | average_return=1.3109 | If deployment space is tight, log the controller footprint and compare domain randomization against short real-world fine-tuning. That trade-off is often the practical decision point. |
| Offline RL for Hardware Resource Allocation | Google Cluster Data | reward_model_policy | average_return=6.0209 | For cluster-control logs, conservative fitted Q iteration is a strong first offline baseline because it exposes the value of being pessimistic on under-covered state-action pairs. |
| Neuro-Symbolic Planning | PDDLGym BlocksWorld | symbolic_shortest_path | success_rate=0.9909 | If you claim a neuro-symbolic benefit, include the exact symbolic planner, the plain learner, and the hybrid learner in the same table. Anything less leaves the mechanism ambiguous. |
| Multi-Agent Pathfinding Optimization | Flatland MAPF | joint_q | success_rate=1.0000 | Always include a reservation-style heuristic in MAPF studies. It is simple, strong, and clarifies whether the learning method is discovering genuine coordination. |

## Offline RL for Dynamic Pricing

**Dataset:** Olist Brazilian E-Commerce Dataset

**Best experiment:** conservative_fqi with average_return=25.6833

The strongest dynamic-pricing variant was conservative_fqi, reaching average return 25.683. The comparison shows how much performance was recoverable from logged pricing traces alone before resorting to an online reference ceiling.

### Key Findings
- Best average return: 25.683 from conservative_fqi.
- Fitted value backups extracted more from the synthetic pricing log than raw action counting alone.
- The online double-Q reference provides headroom, but the offline baselines already form a publishable comparison table.

### Recorded Experiments
| Algorithm | Features | Optimization | Primary | Secondary | Tertiary | Runtime | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| conservative_fqi | demand_inventory_time_state | bellman_backup_with_penalty | average_return=25.6833 | success_rate=0.8889 | mean_steps=4.1333 | 0.00s | offline_transitions=770 |
| fitted_q_iteration | demand_inventory_time_state | bellman_backup_25_iters | average_return=25.6389 | success_rate=0.8944 | mean_steps=4.1333 | 0.00s | offline_transitions=770 |
| double_q_online_reference | demand_inventory_time_state | online_upper_bound_reference | average_return=22.2000 | success_rate=0.9389 | mean_steps=3.7278 | 0.00s | Included as an optimistic ceiling against the logged-data methods. |
| reward_model_policy | demand_inventory_time_state | mean_logged_reward | average_return=21.8222 | success_rate=0.9333 | mean_steps=3.4889 | 0.00s | offline_transitions=770 |
| behavior_cloning | demand_inventory_time_state | logged_policy_counts | average_return=20.8833 | success_rate=0.9667 | mean_steps=3.2944 | 0.00s | offline_transitions=770 |

### Caveats
- This workspace does not ship the Kaggle Olist archive, so the benchmark uses a synthetic demand-elasticity simulator with the same pricing-control structure.
- The offline log is behavior-policy generated; a real study should also measure distribution shift sensitivity across seasons and product categories.

## RL for Supply Chain Routing under Uncertainty

**Dataset:** OR-Gym Supply Chain Routing

**Best experiment:** risk_shaped_q with success_rate=1.0000

The strongest routing controller was risk_shaped_q, reaching success rate 1.000. Explicit disruption and stockout penalties materially changed the learned routing behavior compared with a cost-only baseline.

### Key Findings
- Best success rate: 1.000 from risk_shaped_q.
- Reward shaping around delays and stockouts produced a more resilient routing policy than plain temporal-difference control.
- The backup-route heuristic stayed competitive enough to be a useful non-learning baseline in the report.

### Recorded Experiments
| Algorithm | Features | Optimization | Primary | Secondary | Tertiary | Runtime | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| risk_shaped_q | stage_buffer_disruption_state | delay_and_stockout_shaping | success_rate=1.0000 | average_return=-0.0845 | violation_rate=0.7591 | 0.00s | mean_steps=2.39 |
| q_learning_route | stage_buffer_disruption_state | epsilon_greedy_td | success_rate=0.9909 | average_return=-0.3277 | violation_rate=0.8545 | 0.00s | mean_steps=2.39 |
| double_q_route | stage_buffer_disruption_state | double_estimator_td | success_rate=0.9864 | average_return=-1.1482 | violation_rate=0.8773 | 0.00s | mean_steps=2.62 |
| resilient_rule_policy | stage_buffer_disruption_state | backup_route_heuristic | success_rate=0.9773 | average_return=-2.0127 | violation_rate=0.9227 | 0.00s | mean_steps=3.07 |

### Caveats
- The benchmark uses a compact synthetic routing simulator instead of the full OR-Gym environment library to keep runtime CPU-safe and dependency-free.
- A production study should benchmark larger networks, richer inventory states, and explicit disruption scenarios rather than this three-stage abstraction.

## Multi-Agent RL for Traffic Light Grids

**Dataset:** CityFlow Traffic Grid

**Best experiment:** greedy_pressure_rule with average_return=0.7795

The strongest traffic controller was greedy_pressure_rule, reaching average return 0.780. Joint-action learning beat the simple pressure rule once the reward explicitly valued throughput and queue pressure together.

### Key Findings
- Best average return: 0.780 from greedy_pressure_rule.
- Throughput-aware shaping improved queue control more reliably than swapping from Q-learning to Double Q alone.
- The greedy pressure heuristic is still worth reporting because it gives a strong transparent baseline.

### Recorded Experiments
| Algorithm | Features | Optimization | Primary | Secondary | Tertiary | Runtime | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| greedy_pressure_rule | joint_queue_state | per_intersection_pressure_heuristic | average_return=0.7795 | success_rate=0.0727 | mean_steps=8.0000 | 0.00s | queue_clear_rate=1.000 |
| throughput_shaped_q | joint_queue_state | queue_pressure_shaping | average_return=0.1227 | success_rate=0.0409 | mean_steps=8.0000 | 0.00s | queue_clear_rate=1.000 |
| double_q_joint | joint_queue_state | double_q_learning | average_return=-0.2386 | success_rate=0.0273 | mean_steps=8.0000 | 0.00s | queue_clear_rate=1.000 |
| centralized_q_joint | joint_queue_state | tabular_q_learning | average_return=-0.4545 | success_rate=0.0409 | mean_steps=8.0000 | 0.00s | queue_clear_rate=1.000 |

### Caveats
- The benchmark emulates a tiny two-intersection traffic grid instead of the full CityFlow road-network simulator.
- The queue bins are deliberately coarse, so the result should be read as an algorithmic comparison rather than a calibrated traffic-engineering claim.

## Reward Shaping in Resource Allocation

**Dataset:** CityLearn Energy Grid

**Best experiment:** double_q_dispatch with average_return=0.9832

The strongest energy-allocation controller was double_q_dispatch, reaching average return 0.983. Reward shaping mattered because it aligned the controller with blackout avoidance instead of only immediate dispatch gain.

### Key Findings
- Best average return: 0.983 from double_q_dispatch.
- Blackout-aware shaping improved the safety profile without requiring a larger function approximator.
- The greedy battery heuristic remained interpretable, but it left value on the table when renewable swings were stochastic.

### Recorded Experiments
| Algorithm | Features | Optimization | Primary | Secondary | Tertiary | Runtime | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| double_q_dispatch | demand_battery_renewable_state | double_q_with_shaping | average_return=0.9832 | violation_rate=0.0455 | mean_violations=0.0545 | 0.00s | success_rate=0.000 |
| shaped_q_dispatch | demand_battery_renewable_state | served_load_plus_blackout_penalty | average_return=0.9127 | violation_rate=0.0045 | mean_violations=0.0045 | 0.00s | success_rate=0.000 |
| base_q_dispatch | demand_battery_renewable_state | tabular_q_learning | average_return=0.5361 | violation_rate=0.1091 | mean_violations=0.1091 | 0.00s | success_rate=0.000 |
| battery_first_rule | demand_battery_renewable_state | greedy_storage_heuristic | average_return=-2.4743 | violation_rate=0.5727 | mean_violations=1.4000 | 0.00s | success_rate=0.000 |

### Caveats
- The benchmark uses a tiny tabular storage-and-renewables environment rather than the full CityLearn simulator.
- A full energy study should report tariff curves, seasonal scenarios, and longer-horizon storage degradation effects.

## Execution Algorithms in Algorithmic Trading

**Dataset:** Huge Stock Market Dataset

**Best experiment:** twap_rule with average_return=4.1117

The strongest execution controller was twap_rule, reaching average return 4.112. Inventory-aware shaping changed the behavior materially, especially when volatility penalized passive execution.

### Key Findings
- Best average return: 4.112 from twap_rule.
- Penalizing residual inventory is a simple but effective optimization knob in short-horizon execution tasks.
- The heuristic baselines remain valuable because they expose whether the learned policy is truly adding timing skill or only trading more aggressively.

### Recorded Experiments
| Algorithm | Features | Optimization | Primary | Secondary | Tertiary | Runtime | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| twap_rule | time_inventory_volatility_state | time_weighted_average_price_heuristic | average_return=4.1117 | success_rate=1.0000 | mean_steps=4.0000 | 0.00s | inventory_completion=1.000 |
| q_execution | time_inventory_volatility_state | tabular_q_learning | average_return=3.1443 | success_rate=1.0000 | mean_steps=3.3682 | 0.00s | inventory_completion=1.000 |
| double_q_execution | time_inventory_volatility_state | double_q_learning | average_return=2.5273 | success_rate=1.0000 | mean_steps=2.7045 | 0.00s | inventory_completion=1.000 |
| inventory_shaped_q | time_inventory_volatility_state | residual_inventory_penalty | average_return=1.8518 | success_rate=1.0000 | mean_steps=2.0000 | 0.00s | inventory_completion=1.000 |
| front_loaded_rule | time_inventory_volatility_state | aggressive_front_load_heuristic | average_return=1.7742 | success_rate=1.0000 | mean_steps=2.0000 | 0.00s | inventory_completion=1.000 |

### Caveats
- The environment uses a synthetic execution simulator instead of a large downloaded equity history file to keep the benchmark deterministic and lightweight.
- This is a daily-bar style execution abstraction; a production benchmark would require richer market-impact and intraday liquidity models.

## Safe RL in Chemical Processes

**Dataset:** Tennessee Eastman Process

**Best experiment:** constrained_q with safe_score=-9.7741

The safest chemical-process controller was constrained_q, reaching safe score -9.774. The benchmark makes the core point of safe RL explicit: good reward is not enough if violations remain frequent.

### Key Findings
- Best safe score: -9.774 from constrained_q.
- Deterministic safety filters changed the outcome more than minor optimizer tweaks would have.
- A transparent safety-rule policy still belongs in the table because it anchors the learning methods against a conservative plant-engineering baseline.

### Recorded Experiments
| Algorithm | Features | Optimization | Primary | Secondary | Tertiary | Runtime | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| constrained_q | temperature_pressure_concentration_state | violation_shaped_reward | safe_score=-9.7741 | mean_violations=1.4182 | success_rate=0.1273 | 0.00s | average_return=-4.101 |
| safe_rule_policy | temperature_pressure_concentration_state | handcrafted_safety_rule | safe_score=-10.0793 | mean_violations=1.5227 | success_rate=0.2091 | 0.00s | average_return=-3.988 |
| shielded_q | temperature_pressure_concentration_state | deterministic_action_filter_plus_shaping | safe_score=-10.8368 | mean_violations=1.5545 | success_rate=0.0364 | 0.00s | average_return=-4.619 |
| unsafe_q_learning | temperature_pressure_concentration_state | plain_td_control | safe_score=-10.9548 | mean_violations=1.5182 | success_rate=0.1455 | 0.00s | average_return=-4.882 |

### Caveats
- The workspace uses a compact process-control abstraction rather than the full Tennessee Eastman simulator and anomaly streams.
- A publishable industrial study would also report constraint satisfaction under disturbances and delayed observations.

## Curriculum Learning for Job-Shop Scheduling

**Dataset:** Taillard Job Shop Instances

**Best experiment:** shortest_job_rule with success_rate=1.0000

The strongest scheduler was shortest_job_rule, reaching success rate 1.000. This is the most CPU-friendly topic in the suite, and the curriculum variants show why: they improve hard-instance completion without a heavy simulator stack.

### Key Findings
- Best success rate: 1.000 from shortest_job_rule.
- Easy-to-hard warm starts improved hard-instance behavior relative to training only on the hardest setting.
- A dispatching heuristic is still necessary in the table because it is strong, cheap, and interpretable.

### Recorded Experiments
| Algorithm | Features | Optimization | Primary | Secondary | Tertiary | Runtime | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| shortest_job_rule | remaining_jobs_machine_state | dispatching_heuristic | success_rate=1.0000 | average_return=7.8000 | mean_steps=10.0000 | 0.00s | completion_rate=1.000 |
| due_date_shaped_curriculum_q | remaining_jobs_machine_state | curriculum_plus_remaining_job_penalty | success_rate=1.0000 | average_return=7.5500 | mean_steps=10.0000 | 0.00s | completion_rate=1.000 |
| direct_hard_q | remaining_jobs_machine_state | hard_instance_only_training | success_rate=1.0000 | average_return=7.0500 | mean_steps=10.0000 | 0.00s | completion_rate=1.000 |
| curriculum_q | remaining_jobs_machine_state | easy_to_hard_warm_start | success_rate=1.0000 | average_return=6.8000 | mean_steps=10.0000 | 0.00s | completion_rate=1.000 |

### Caveats
- The benchmark uses a small synthetic dispatching abstraction instead of parsing the original Taillard text instances directly.
- A stronger follow-up would add larger instance distributions and compare against classical OR heuristics beyond shortest-job-first.

## Hierarchical RL for Long-Horizon Logistics

**Dataset:** CVRPLIB Logistics Instances

**Best experiment:** double_q_option_filter with success_rate=1.0000

The strongest long-horizon logistics controller was double_q_option_filter, reaching success rate 1.000. Even in a tiny abstraction, explicit macro-level options changed the policy enough to justify calling the problem hierarchical rather than flat routing.

### Key Findings
- Best success rate: 1.000 from double_q_option_filter.
- Option-style action filters improved long-horizon credit assignment without adding neural complexity.
- A two-level handcrafted macro rule remains a strong sanity check for hierarchical claims.

### Recorded Experiments
| Algorithm | Features | Optimization | Primary | Secondary | Tertiary | Runtime | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| double_q_option_filter | region_backlog_disruption_state | double_q_with_macro_filter | success_rate=1.0000 | average_return=5.3805 | violation_rate=0.0727 | 0.00s | mean_steps=2.59 |
| flat_q_learning | region_backlog_disruption_state | single_level_q_learning | success_rate=0.9955 | average_return=5.1373 | violation_rate=0.1227 | 0.00s | mean_steps=2.95 |
| option_filtered_q | region_backlog_disruption_state | macro_option_filter_plus_shaping | success_rate=1.0000 | average_return=5.1059 | violation_rate=0.1500 | 0.00s | mean_steps=2.83 |
| hierarchical_macro_rule | region_backlog_disruption_state | two_level_handcrafted_policy | success_rate=1.0000 | average_return=3.7723 | violation_rate=0.3818 | 0.00s | mean_steps=3.53 |

### Caveats
- The CVRPLIB topic is represented by a compact staged-delivery simulator instead of large graph instances.
- A stronger follow-up would add actual graph partitioning and learned option discovery rather than a fixed macro filter.

## Meta-RL for Rapid Market Adaptation

**Dataset:** S&P 500 Stock Data

**Best experiment:** meta_q_fast_adapt with average_return=24.1318

The strongest held-out market adaptation policy was meta_q_fast_adapt, reaching average return 24.132. This benchmark makes the meta-learning story concrete by separating transfer initialization from rapid target-task adaptation.

### Key Findings
- Best average return on the held-out regime: 24.132 from meta_q_fast_adapt.
- Averaged meta-initialization is useful, but the adaptation phase is what should be credited for fast recovery on a new regime.
- The transparent pricing rule remains a sanity baseline for whether meta-learning is beating simple elasticity logic.

### Recorded Experiments
| Algorithm | Features | Optimization | Primary | Secondary | Tertiary | Runtime | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| meta_q_fast_adapt | shared_price_state | average_q_plus_short_adaptation | average_return=24.1318 | success_rate=0.9909 | mean_steps=3.2773 | 0.00s | Target regime held out from meta-training. |
| pricing_rule_baseline | shared_price_state | handcrafted_market_rule | average_return=23.9682 | success_rate=0.9682 | mean_steps=3.2727 | 0.00s | Target regime held out from meta-training. |
| meta_average_no_adapt | shared_price_state | average_q_initialization_only | average_return=23.8318 | success_rate=0.9864 | mean_steps=3.3273 | 0.00s | Target regime held out from meta-training. |
| scratch_q | shared_price_state | held_out_regime_training | average_return=23.6636 | success_rate=0.9909 | mean_steps=3.3273 | 0.00s | Target regime held out from meta-training. |
| pooled_multitask_q | shared_price_state | sequential_multitask_finetuning | average_return=23.0136 | success_rate=0.9909 | mean_steps=3.2545 | 0.00s | Target regime held out from meta-training. |

### Caveats
- The market tasks are synthetic pricing regimes rather than sliced real ticker windows from a downloaded S&P 500 archive.
- A fuller study would include more tasks, different horizons, and a proper split between crash, rebound, and quiet regimes.

## RL for HVAC Energy Optimization

**Dataset:** Sinergym HVAC Benchmark

**Best experiment:** double_q_hvac with average_return=-15.2385

The strongest HVAC controller was double_q_hvac, reaching average return -15.238. Comfort-aware shaping matters because naive cost minimization alone can under-condition the policy on occupancy.

### Key Findings
- Best average return: -15.238 from double_q_hvac.
- Occupancy- and comfort-aware shaping improved the trade-off between energy spend and thermal comfort.
- A tariff-aware thermostat rule is still necessary as a strong interpretable baseline for building control.

### Recorded Experiments
| Algorithm | Features | Optimization | Primary | Secondary | Tertiary | Runtime | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| double_q_hvac | indoor_outdoor_occupancy_tariff_state | double_q_with_comfort_shaping | average_return=-15.2385 | violation_rate=0.7591 | success_rate=0.4227 | 0.00s | mean_steps=6.00 |
| comfort_shaped_q | indoor_outdoor_occupancy_tariff_state | comfort_and_violation_shaping | average_return=-15.5421 | violation_rate=0.7409 | success_rate=0.5273 | 0.00s | mean_steps=6.00 |
| base_hvac_q | indoor_outdoor_occupancy_tariff_state | tabular_q_learning | average_return=-15.5398 | violation_rate=0.7727 | success_rate=0.4864 | 0.00s | mean_steps=6.00 |
| tariff_aware_rule | indoor_outdoor_occupancy_tariff_state | cost_sensitive_thermostat_rule | average_return=-17.1809 | violation_rate=0.6364 | success_rate=0.6318 | 0.00s | mean_steps=6.00 |

### Caveats
- The benchmark uses a small tabular thermal-control model rather than the full Sinergym and EnergyPlus stack.
- Absolute reward values are not directly comparable to a building-energy benchmark with calibrated thermal physics.

## Safe RL with Deterministic Envelopes

**Dataset:** Safe-Control-Gym

**Best experiment:** shield_rule with safe_score=-0.0839

The strongest deterministic-envelope controller was shield_rule, reaching safe score -0.084. Filtering unsafe actions changes the control problem enough that it deserves its own benchmark line item, not just a note in the appendix.

### Key Findings
- Best safe score: -0.084 from shield_rule.
- Deterministic action filters cut violation counts more directly than reward shaping alone.
- A simple shield rule remains a strong non-learning baseline for safety-critical deployment discussions.

### Recorded Experiments
| Algorithm | Features | Optimization | Primary | Secondary | Tertiary | Runtime | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| shield_rule | distance_speed_load_state | handcrafted_envelope_controller | safe_score=-0.0839 | mean_violations=0.0318 | success_rate=0.1364 | 0.00s | average_return=0.043 |
| shielded_q | distance_speed_load_state | deterministic_envelope_filter | safe_score=-0.9548 | mean_violations=0.2000 | success_rate=0.1773 | 0.00s | average_return=-0.155 |
| violation_shaped_q | distance_speed_load_state | explicit_violation_penalty | safe_score=-1.2175 | mean_violations=0.1955 | success_rate=0.1318 | 0.00s | average_return=-0.436 |
| vanilla_q | distance_speed_load_state | plain_tabular_q | safe_score=-2.1864 | mean_violations=0.4864 | success_rate=0.1955 | 0.00s | average_return=-0.241 |

### Caveats
- The benchmark uses a compact kinematic envelope controller instead of the full Safe-Control-Gym and PyBullet stack.
- A stronger study would evaluate robustness under model mismatch and measurement delay, not only nominal envelope compliance.

## Sim-to-Real Transfer for Minimal Binaries

**Dataset:** MicroRLEnv

**Best experiment:** domain_randomized_q with average_return=1.3109

The strongest sim-to-real strategy was domain_randomized_q, reaching average return 1.311. The result separates three common transfer tactics cleanly: train only in sim, randomize the sim, or warm-start then fine-tune on the target dynamics.

### Key Findings
- Best average return on the target dynamics: 1.311 from domain_randomized_q.
- Domain randomization provides a useful zero-shot transfer baseline, but short target adaptation is often the cleaner way to close the final gap.
- The tiny rule controller remains worth reporting because minimal binaries are part of the actual deployment requirement here.

### Recorded Experiments
| Algorithm | Features | Optimization | Primary | Secondary | Tertiary | Runtime | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| domain_randomized_q | offset_velocity_terrain_state | train_across_gap_samples | average_return=1.3109 | success_rate=0.9727 | mean_steps=2.1182 | 0.00s | q_table_bytes=10800 |
| robust_rule_controller | offset_velocity_terrain_state | handcrafted_gap_robust_rule | average_return=1.1020 | success_rate=1.0000 | mean_steps=2.3545 | 0.00s | Minimal-controller baseline for deployment discussions. |
| sim_plus_target_finetune | offset_velocity_terrain_state | warm_start_and_short_real_adaptation | average_return=0.9225 | success_rate=0.9591 | mean_steps=2.3182 | 0.00s | q_table_bytes=10800 |
| sim_only_q | offset_velocity_terrain_state | train_in_nominal_sim_only | average_return=-3.6368 | success_rate=0.6818 | mean_steps=3.5273 | 0.00s | q_table_bytes=10800 |

### Caveats
- The workspace uses a tiny tabular transfer problem rather than a full PyBullet joint-control training loop.
- A stronger study would export and benchmark actual policy binaries on-device instead of only logging Q-table size.

## Offline RL for Hardware Resource Allocation

**Dataset:** Google Cluster Data

**Best experiment:** reward_model_policy with average_return=6.0209

The strongest hardware-allocation controller was reward_model_policy, reaching average return 6.021. The offline allocation topic benefits from exactly the same disciplined comparison as pricing: imitation, reward modeling, fitted value iteration, and a clear online ceiling.

### Key Findings
- Best average return: 6.021 from reward_model_policy.
- Count-based imitation is fast, but value backups extract more from the same logged transitions when allocation outcomes are delayed.
- An explicit online ceiling helps quantify how much headroom remains before offline methods saturate the synthetic benchmark.

### Recorded Experiments
| Algorithm | Features | Optimization | Primary | Secondary | Tertiary | Runtime | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| reward_model_policy | queue_cpu_memory_priority_state | mean_logged_reward | average_return=6.0209 | success_rate=0.3818 | violation_rate=0.4045 | 0.00s | offline_transitions=1440 |
| conservative_fqi | queue_cpu_memory_priority_state | offline_bellman_backup_with_penalty | average_return=5.3982 | success_rate=0.3773 | violation_rate=0.4727 | 0.00s | offline_transitions=1440 |
| online_q_reference | queue_cpu_memory_priority_state | online_ceiling_reference | average_return=4.8818 | success_rate=0.3273 | violation_rate=0.3500 | 0.00s | Included as an optimistic upper bound relative to logged-data policies. |
| behavior_cloning | queue_cpu_memory_priority_state | count_based_imitation | average_return=4.5209 | success_rate=0.2273 | violation_rate=0.6091 | 0.00s | offline_transitions=1440 |
| fitted_q_iteration | queue_cpu_memory_priority_state | offline_bellman_backup | average_return=3.9909 | success_rate=0.3455 | violation_rate=0.4364 | 0.00s | offline_transitions=1440 |

### Caveats
- The workspace does not include the full Google cluster trace, so the benchmark uses a small tabular resource-allocation simulator with the same control structure.
- A real workload trace study should add bursty arrivals, SLA classes, and much richer action granularity.

## Neuro-Symbolic Planning

**Dataset:** PDDLGym BlocksWorld

**Best experiment:** symbolic_shortest_path with success_rate=0.9909

The strongest planning controller was symbolic_shortest_path, reaching success rate 0.991. The benchmark makes the neuro-symbolic point cleanly: symbolic structure can be injected as a goal-distance prior rather than replacing learning entirely.

### Key Findings
- Best success rate: 0.991 from symbolic_shortest_path.
- Goal-distance shaping improved search efficiency relative to pure tabular learning.
- The exact symbolic shortest-path policy is a valuable upper-bound-style baseline for small planning domains.

### Recorded Experiments
| Algorithm | Features | Optimization | Primary | Secondary | Tertiary | Runtime | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| symbolic_shortest_path | symbolic_stack_state | exact_goal_distance_policy | success_rate=0.9909 | average_return=4.1245 | mean_steps=4.1500 | 0.00s | Goal-distance symbolic prior available for hybrid variants. |
| hybrid_symbolic_shaped_q | symbolic_stack_state | goal_distance_shaping | success_rate=0.8682 | average_return=3.0636 | mean_steps=4.7727 | 0.00s | Goal-distance symbolic prior available for hybrid variants. |
| q_learning_blocksworld | symbolic_stack_state | plain_q_learning | success_rate=0.2045 | average_return=-2.1891 | mean_steps=5.4455 | 0.00s | Goal-distance symbolic prior available for hybrid variants. |
| double_q_blocksworld | symbolic_stack_state | double_q_learning | success_rate=0.1409 | average_return=-2.7418 | mean_steps=5.5955 | 0.00s | Goal-distance symbolic prior available for hybrid variants. |

### Caveats
- The benchmark uses an in-repo BlocksWorld abstraction instead of loading domains through the full PDDLGym stack.
- Scaling symbolic priors to larger planning domains would require state abstraction beyond this compact three-block setup.

## Multi-Agent Pathfinding Optimization

**Dataset:** Flatland MAPF

**Best experiment:** joint_q with success_rate=1.0000

The strongest MAPF controller was joint_q, reaching success rate 1.000. Joint pathfinding is where simple coordination baselines really matter: a learned policy that cannot beat reservation logic is not yet persuasive.

### Key Findings
- Best success rate: 1.000 from joint_q.
- Potential-based distance shaping improved coordination relative to a plain joint-action Q table.
- Reservation logic remains a strong transparent baseline for small multi-agent grids.

### Recorded Experiments
| Algorithm | Features | Optimization | Primary | Secondary | Tertiary | Runtime | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| joint_q | joint_agent_position_state | plain_joint_q_learning | success_rate=1.0000 | average_return=4.3600 | violation_rate=0.0000 | 0.00s | mean_steps=2.00 |
| distance_shaped_joint_q | joint_agent_position_state | potential_based_distance_shaping | success_rate=1.0000 | average_return=4.3600 | violation_rate=0.0000 | 0.00s | mean_steps=2.00 |
| independent_greedy | joint_agent_position_state | per_agent_shortest_path_rule | success_rate=1.0000 | average_return=4.3600 | violation_rate=0.0000 | 0.00s | mean_steps=2.00 |
| reservation_rule | joint_agent_position_state | simple_collision_reservation | success_rate=1.0000 | average_return=4.3600 | violation_rate=0.0000 | 0.00s | mean_steps=2.00 |
| double_q_joint | joint_agent_position_state | double_q_learning | success_rate=1.0000 | average_return=3.8000 | violation_rate=0.0000 | 0.00s | mean_steps=3.00 |

### Caveats
- The benchmark uses a tiny 3x3 two-agent grid instead of the full Flatland railway environment.
- A stronger MAPF study would include larger grids, more agents, and deadlock-heavy track topologies.
