# Small-Data ML Research Benchmark Report

This report was generated from runnable CPU-first experiment scripts stored under `ml/`. Each project records multiple algorithm, feature, and optimization variants. The benchmark favors public datasets when reachable and uses explicit fallbacks when Kaggle-gated assets are unavailable in the workspace.

The latest completed rerun is the expanded quick benchmark, which executed all 15 projects and recorded 112 experiment rows into the current CSV artifacts. Smoke tests passed after the expansion, and each `project_*.py` module also passed direct entrypoint execution through the package runner recorded in `ml/results/project_file_runs.md`.

For a publication-oriented research narrative that connects these experiments to recent 2025-2026 literature, see `ml/literature_review.md`.

## Cross-Project Summary
| Project | Dataset | Best Model | Best Score | Recommendation |
| --- | --- | --- | --- | --- |
| Causal Discovery in Time-Series | Metro Interstate Traffic Volume | hist_gradient_boosting | rmse=426.2989 | Use a gradient-boosted or forest regressor with lagged traffic and weather features when the goal is traffic disruption analysis, and treat Granger significance as supportive evidence rather than definitive causal proof. |
| Conformal Prediction for Supply Chains | Store Item Demand Forecasting | random_forest_split_conformal | coverage=0.9122 | For inventory planning on thin daily demand data, start with split conformal intervals around a tree ensemble because the coverage is easy to calibrate and usually narrower than naive safety-stock rules. |
| Explainable Boosting Machines in High-Stakes Domains | Give Me Some Credit / Credit-G | ebm_interactions_dense | roc_auc=0.8170 | Choose an EBM when you need near-ensemble accuracy with feature-level accountability. If a black-box model only wins marginally, the interpretability trade-off usually favors the EBM in regulated credit workflows. |
| Handling Concept Drift in Streaming Data | Electricity Market Dataset | hoeffding_tree | post_drift_accuracy=0.7544 | If your label boundary shifts over time, favor incremental learners with explicit update loops over a frozen batch model, even when the initial batch score looks strong. |
| Federated Learning on Extreme Low-Bandwidth | Adult Census Income | centralized_logistic_regression | accuracy=0.8425 | Use top-k or sign-compressed federated updates when uplink budget is the constraint. Full-precision FedAvg is only justified when the last few accuracy points matter more than transmission cost. |
| Tabular Meta-Learning | OpenML-style Small Classification Suite | random_forest | balanced_accuracy=1.0000 | If you are benchmarking tiny tabular datasets, store meta-features and prior model rankings. Even a simple nearest-dataset recommender can cut the search space before full hyperparameter tuning. |
| Synthetic Tabular Data Privacy | Credit Card Fraud Detection | dp_histogram | roc_auc=0.8614 | Use copula-style synthesis when you need stronger utility, and move toward noisier histogram-style methods when privacy risk dominates. A pure bootstrap should only be treated as a utility ceiling, not a private release. |
| Automated Feature Engineering via Evolutionary Algorithms | House Prices / Ames-style Regression | random_forest_raw | rmse=32402.3127 | Use a small evolutionary search when the feature space is rich but hand-engineering is slow. Even a short CPU-only run can surface interactions that linear baselines miss. |
| Imbalanced Data in Graph Neural Networks | Elliptic-style Illicit Transaction Graph | logistic_graph_features | average_precision=0.8396 | Before training a full GNN stack, benchmark a graph-aware tabular baseline with neighborhood aggregation. On imbalanced graphs it often captures a large share of the graph signal for a fraction of the engineering cost. |
| Tree-Based Models vs. Noise | Breast Cancer Wisconsin (Diagnostic) | extra_trees | balanced_accuracy=0.9623 | When data quality is unstable, compare degradation curves rather than only clean-set accuracy. Tree ensembles usually lose performance more gracefully than small neural networks under missingness and label noise. |
| Memory-Safe Feature Extraction Pipelines | NSL-KDD | feature_hashing_pipeline | balanced_accuracy=0.7865 | Use sparse or hashed feature extraction when intrusion logs start to scale. Dense one-hot matrices are simple, but they become the bottleneck long before the classifier does. |
| Algorithmic Optimization of Tree Ensembles | Covertype | extra_trees | accuracy=0.8764 | Benchmark tree ensembles with both accuracy and systems metrics. ExtraTrees or shallow forests often give a better production trade-off than a raw accuracy leader that is slower to fit or score. |
| Numerical Stability in Label Encoding | Categorical Encoding Challenge-style Synthetic Benchmark | target_encoding | roc_auc=0.9122 | Avoid low-width wrapped ordinal codes for high-cardinality features. If memory is tight, prefer sparse one-hot or smoothed target encoding rather than forcing categories into a tiny integer range. |
| Security-Aware ML for Intrusion Detection | UNSW-NB15-style Intrusion Detection | hist_gradient_boosting_aggressive | macro_f1=0.7173 | Treat intrusion detection as a cost-sensitive problem. Pick the model and threshold that maximize attack recall without letting precision collapse, rather than optimizing plain accuracy. |
| Deterministic Agentic Data Validation | NYC Taxi Trip Duration | validated_hist_gradient_enriched | rmse=83.0233 | Validate messy operational data before model selection. Rule-driven cleaning often yields larger gains than swapping between two strong regressors on corrupted inputs. |

## Causal Discovery in Time-Series

**Dataset:** Metro Interstate Traffic Volume

**Best experiment:** hist_gradient_boosting with rmse=426.2989

Adding weather covariates increased RMSE by 3.43 relative to the strongest lag-only baseline. The strongest weather-aware model was hist_gradient_boosting, and the minimum Granger p-values were 3.643e-29 for temperature and 1 for rainfall.

### Key Findings
- Weather-aware features improved the best RMSE from 426.30 to 429.73.
- Temperature Granger-caused traffic with a minimum p-value of 3.643e-29.
- Rainfall Granger-caused traffic with a minimum p-value of 1.

### Recorded Experiments
| Algorithm | Features | Optimization | Primary | Secondary | Tertiary | Runtime | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hist_gradient_boosting | temporal_only | lag_features_plus_model_choice | rmse=426.2989 | r2=0.9536 | mae=246.9963 | 0.92s | Weather features withheld |
| hist_gradient_boosting | temporal_plus_weather | lag_features_plus_model_choice | rmse=429.7263 | r2=0.9529 | mae=248.5949 | 0.47s | Weather features included |
| random_forest | temporal_only | lag_features_plus_model_choice | rmse=450.9857 | r2=0.9481 | mae=241.6804 | 0.21s | Weather features withheld |
| random_forest | temporal_plus_weather | lag_features_plus_model_choice | rmse=454.7739 | r2=0.9472 | mae=244.8694 | 0.31s | Weather features included |
| linear_regression | temporal_plus_weather | lag_features_plus_model_choice | rmse=786.2414 | r2=0.8423 | mae=521.6436 | 0.01s | Weather features included |
| linear_regression | temporal_only | lag_features_plus_model_choice | rmse=786.8269 | r2=0.8421 | mae=522.3636 | 0.01s | Weather features withheld |

### Caveats
- Granger causality identifies predictive temporal precedence, not intervention-level causality.
- The quick benchmark uses a subset of the time axis for runtime control.
- If the UCI download is unavailable, the runner falls back to a synthetic but structurally similar traffic process.

## Conformal Prediction for Supply Chains

**Dataset:** Store Item Demand Forecasting

**Best experiment:** random_forest_split_conformal with coverage=0.9122

The best interval method was random_forest_split_conformal, which reached coverage=0.912 while balancing interval width. The conformal forest was especially strong when lagged features carried most of the signal.

### Key Findings
- The strongest interval method was random_forest_split_conformal with average width 22.78.
- The naive lag-7 baseline still provided a useful uncertainty floor with coverage 0.907.
- Calendar and lag features mattered more than model complexity when demand followed stable seasonal patterns.

### Recorded Experiments
| Algorithm | Features | Optimization | Primary | Secondary | Tertiary | Runtime | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| random_forest_split_conformal | lags_plus_calendar | split_conformal | coverage=0.9122 | avg_width=22.7794 | rmse=7.1020 | 2.44s | Point forest with symmetric residual conformalization |
| gradient_boosting_split_conformal | lags_plus_calendar | split_conformal | coverage=0.9017 | avg_width=20.4024 | rmse=6.7184 | 8.99s | Boosted regressor with symmetric residual conformalization |
| seasonal_mean_conformal | rolling_mean_7 | calibration_quantile | coverage=0.9021 | avg_width=25.7668 | rmse=8.0892 | 0.00s | Baseline interval from rolling mean residual quantile |
| seasonal_naive_conformal | lag_7_only | calibration_quantile | coverage=0.9065 | avg_width=33.5655 | rmse=10.0690 | 0.00s | Baseline interval from lag-7 residual quantile |
| gradient_boosting_quantile | lags_plus_calendar | direct_quantile_training | coverage=0.7993 | avg_width=15.8832 | rmse=6.7257 | 17.48s | Separate quantile regressors for lower and upper bounds |

### Caveats
- If the Kaggle dataset is not present locally, the runner switches to a synthetic supply-chain process.
- Coverage is measured on a single temporal holdout rather than rolling-origin folds.
- The quick path limits the time span and number of store-item combinations for runtime control.

## Explainable Boosting Machines in High-Stakes Domains

**Dataset:** Give Me Some Credit / Credit-G

**Best experiment:** ebm_interactions_dense with roc_auc=0.8170

The best credit-risk model was ebm_interactions_dense with ROC AUC 0.817. The EBM variants expose dominant risk factors directly, which makes them attractive when auditability matters as much as raw discrimination.

### Key Findings
- Best observed ROC AUC was 0.817 from ebm_interactions_dense.
- EBM global terms remained readable: ebm_default: checking_status=0.575, duration=0.357, savings_status=0.327.
- Logistic regression provides a strong transparency baseline but usually leaves accuracy on the table relative to boosting-based models.

### Recorded Experiments
| Algorithm | Features | Optimization | Primary | Secondary | Tertiary | Runtime | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ebm_interactions_dense | native_mixed_types | interpretable_boosting | roc_auc=0.8170 | average_precision=0.6822 | brier=0.1497 | 4.40s | Top terms: checking_status=0.568, duration=0.329, savings_status=0.294 |
| ebm_interactions | native_mixed_types | interpretable_boosting | roc_auc=0.8167 | average_precision=0.6869 | brier=0.1483 | 3.17s | Top terms: checking_status=0.564, duration=0.345, savings_status=0.323 |
| logistic_regression | one_hot_tabular | baseline_hyperparameters | roc_auc=0.8087 | average_precision=0.6348 | brier=0.1821 | 11.60s | High-capacity black-box baseline |
| random_forest | one_hot_tabular | baseline_hyperparameters | roc_auc=0.8057 | average_precision=0.6380 | brier=0.1608 | 1.77s | High-capacity black-box baseline |
| ebm_default | native_mixed_types | interpretable_boosting | roc_auc=0.8005 | average_precision=0.6234 | brier=0.1577 | 11.66s | Top terms: checking_status=0.575, duration=0.357, savings_status=0.327 |
| hist_gradient_boosting | one_hot_tabular | baseline_hyperparameters | roc_auc=0.7740 | average_precision=0.6186 | brier=0.1738 | 37.10s | High-capacity black-box baseline |
| logistic_elastic_net | one_hot_tabular | baseline_hyperparameters | roc_auc=0.6985 | average_precision=0.4872 | brier=0.2359 | 6.45s | High-capacity black-box baseline |

### Caveats
- If the Kaggle credit dataset is not available locally, the runner uses OpenML Credit-G and then a breast-cancer fallback if remote access fails.
- This benchmark uses a single stratified split, not repeated cross-validation.
- Interpretability is summarized through global term importance rather than full local explanation artifacts.

## Handling Concept Drift in Streaming Data

**Dataset:** Electricity Market Dataset

**Best experiment:** hoeffding_tree with post_drift_accuracy=0.7544

The best post-drift recovery came from hoeffding_tree with accuracy 0.754. Static models degraded sharply after the concept change, while online learners recovered by updating on new stream segments.

### Key Findings
- The post-drift winner was hoeffding_tree at 0.754 accuracy.
- The static baseline fell to 0.746 accuracy after the drift point.
- Online updates matter more than raw model complexity once the generating process changes midstream.

### Recorded Experiments
| Algorithm | Features | Optimization | Primary | Secondary | Tertiary | Runtime | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hoeffding_tree | raw_stream | river_incremental_tree | post_drift_accuracy=0.7544 | overall_accuracy=0.7542 | balanced_accuracy=0.7542 | 0.00s | Sample-wise adaptive decision tree |
| static_sgd | raw_stream | no_online_updates | post_drift_accuracy=0.7462 | overall_accuracy=0.7450 | balanced_accuracy=0.7139 | 0.04s | Frozen after initial fit |
| rolling_window_sgd | recent_history_window | chunk_retrain_recent_window | post_drift_accuracy=0.6334 | overall_accuracy=0.6381 | balanced_accuracy=0.6381 | 0.00s | Retrains on a sliding recent-history window after each chunk |
| adaptive_sgd | raw_stream | partial_fit_updates | post_drift_accuracy=0.5894 | overall_accuracy=0.5819 | balanced_accuracy=0.5819 | 0.00s | Chunked online updates with logistic loss |

### Caveats
- If OpenML electricity is unavailable, the runner uses a synthetic stream with an abrupt midpoint drift.
- The river model is evaluated in a true prequential fashion, while the batch baselines are chunked for speed.
- The quick benchmark uses a reduced stream length and smaller chunk sizes.

## Federated Learning on Extreme Low-Bandwidth

**Dataset:** Adult Census Income

**Best experiment:** centralized_logistic_regression with accuracy=0.8425

The best low-bandwidth federated strategy was federated_topk, which reached accuracy 0.832 while keeping traffic to 0.017 MB. Compression preserved most of the centralized baseline when client updates were small and frequent.

### Key Findings
- Centralized logistic regression reached 0.843 accuracy as the upper bound.
- The strongest compressed federated variant was federated_topk with 0.017 MB of communication.
- Bandwidth-aware compression reduced communication by orders of magnitude with only modest quality loss on the quick benchmark.

### Recorded Experiments
| Algorithm | Features | Optimization | Primary | Secondary | Tertiary | Runtime | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| centralized_logistic_regression | full_one_hot | server_only_baseline | accuracy=0.8425 | balanced_accuracy=0.7532 | bandwidth_mb=0.0000 | 0.81s | Upper-bound centralized baseline |
| federated_topk | 128_simulated_nodes | fedavg_topk_5pct | accuracy=0.8321 | balanced_accuracy=0.7278 | bandwidth_mb=0.0167 | 0.09s | 14 rounds, 24 clients per round, ratio=0.05, error_feedback=False |
| federated_topk | 128_simulated_nodes | fedavg_topk_10pct | accuracy=0.8267 | balanced_accuracy=0.7493 | bandwidth_mb=0.0320 | 0.08s | 14 rounds, 24 clients per round, ratio=0.10, error_feedback=False |
| federated_topk | 128_simulated_nodes | fedavg_topk_10pct_error_feedback | accuracy=0.8258 | balanced_accuracy=0.7697 | bandwidth_mb=0.0320 | 0.08s | 14 rounds, 24 clients per round, ratio=0.10, error_feedback=True |
| federated_full | 128_simulated_nodes | fedavg_full_precision | accuracy=0.8258 | balanced_accuracy=0.7733 | bandwidth_mb=0.1487 | 0.07s | 14 rounds, 24 clients per round, ratio=1.00, error_feedback=False |
| federated_sign | 128_simulated_nodes | fedavg_sign_error_feedback | accuracy=0.8187 | balanced_accuracy=0.7692 | bandwidth_mb=0.0072 | 0.09s | 14 rounds, 24 clients per round, ratio=0.10, error_feedback=True |
| federated_sign | 128_simulated_nodes | fedavg_sign_quantized | accuracy=0.8021 | balanced_accuracy=0.7230 | bandwidth_mb=0.0072 | 0.08s | 14 rounds, 24 clients per round, ratio=0.10, error_feedback=False |

### Caveats
- When OpenML Adult is unavailable, the runner switches to a synthetic census-like classification problem.
- Client simulation assumes balanced node sizes and synchronous aggregation.
- The federated optimizer is a lightweight research approximation, not a production-hardened FL stack.

## Tabular Meta-Learning

**Dataset:** OpenML-style Small Classification Suite

**Best experiment:** random_forest with balanced_accuracy=1.0000

Across 5 small classification tasks, the best average base learner was hist_gradient_boosting. A nearest-dataset meta-selector recovered the true best learner on 0.000 of held-out datasets with mean regret 0.085.

### Key Findings
- The strongest average learner across the suite was hist_gradient_boosting.
- Meta-selection accuracy reached 0.000 on leave-one-dataset-out evaluation.
- Meta-features like sample count, feature count, missingness, and label entropy are enough to build a lightweight warm-start policy.

### Recorded Experiments
| Algorithm | Features | Optimization | Primary | Secondary | Tertiary | Runtime | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| random_forest | wine | holdout_balanced_accuracy | balanced_accuracy=1.0000 | dataset_size=178.0000 | feature_count=13.0000 | 0.61s | Per-dataset score on wine |
| hist_gradient_boosting | wine | holdout_balanced_accuracy | balanced_accuracy=1.0000 | dataset_size=178.0000 | feature_count=13.0000 | 13.95s | Per-dataset score on wine |
| knn | iris | holdout_balanced_accuracy | balanced_accuracy=0.9744 | dataset_size=150.0000 | feature_count=4.0000 | 0.01s | Per-dataset score on iris |
| logistic_regression | wine | holdout_balanced_accuracy | balanced_accuracy=0.9722 | dataset_size=178.0000 | feature_count=13.0000 | 0.66s | Per-dataset score on wine |
| hist_gradient_boosting | breast_cancer | holdout_balanced_accuracy | balanced_accuracy=0.9567 | dataset_size=569.0000 | feature_count=30.0000 | 1.77s | Per-dataset score on breast_cancer |
| random_forest | breast_cancer | holdout_balanced_accuracy | balanced_accuracy=0.9512 | dataset_size=569.0000 | feature_count=30.0000 | 0.45s | Per-dataset score on breast_cancer |
| logistic_regression | iris | holdout_balanced_accuracy | balanced_accuracy=0.9487 | dataset_size=150.0000 | feature_count=4.0000 | 0.40s | Per-dataset score on iris |
| logistic_regression | breast_cancer | holdout_balanced_accuracy | balanced_accuracy=0.9473 | dataset_size=569.0000 | feature_count=30.0000 | 0.33s | Per-dataset score on breast_cancer |
| hist_gradient_boosting | iris | holdout_balanced_accuracy | balanced_accuracy=0.9231 | dataset_size=150.0000 | feature_count=4.0000 | 7.44s | Per-dataset score on iris |
| knn | breast_cancer | holdout_balanced_accuracy | balanced_accuracy=0.9212 | dataset_size=569.0000 | feature_count=30.0000 | 0.01s | Per-dataset score on breast_cancer |
| random_forest | iris | holdout_balanced_accuracy | balanced_accuracy=0.8974 | dataset_size=150.0000 | feature_count=4.0000 | 0.62s | Per-dataset score on iris |
| random_forest | phoneme | holdout_balanced_accuracy | balanced_accuracy=0.8708 | dataset_size=5404.0000 | feature_count=5.0000 | 0.34s | Per-dataset score on phoneme |
| hist_gradient_boosting | phoneme | holdout_balanced_accuracy | balanced_accuracy=0.8446 | dataset_size=5404.0000 | feature_count=5.0000 | 4.27s | Per-dataset score on phoneme |
| knn | phoneme | holdout_balanced_accuracy | balanced_accuracy=0.8211 | dataset_size=5404.0000 | feature_count=5.0000 | 0.01s | Per-dataset score on phoneme |
| knn | wine | holdout_balanced_accuracy | balanced_accuracy=0.7778 | dataset_size=178.0000 | feature_count=13.0000 | 0.00s | Per-dataset score on wine |
| hist_gradient_boosting | blood-transfusion-service-center | holdout_balanced_accuracy | balanced_accuracy=0.6643 | dataset_size=748.0000 | feature_count=4.0000 | 0.63s | Per-dataset score on blood-transfusion-service-center |
| logistic_regression | phoneme | holdout_balanced_accuracy | balanced_accuracy=0.6452 | dataset_size=5404.0000 | feature_count=5.0000 | 0.15s | Per-dataset score on phoneme |
| knn | blood-transfusion-service-center | holdout_balanced_accuracy | balanced_accuracy=0.5962 | dataset_size=748.0000 | feature_count=4.0000 | 0.00s | Per-dataset score on blood-transfusion-service-center |
| random_forest | blood-transfusion-service-center | holdout_balanced_accuracy | balanced_accuracy=0.5708 | dataset_size=748.0000 | feature_count=4.0000 | 0.24s | Per-dataset score on blood-transfusion-service-center |
| logistic_regression | blood-transfusion-service-center | holdout_balanced_accuracy | balanced_accuracy=0.5350 | dataset_size=748.0000 | feature_count=4.0000 | 0.15s | Per-dataset score on blood-transfusion-service-center |
| nearest_dataset_meta_selector | leave_one_dataset_out | meta_feature_neighbor_match | recommendation_accuracy=0.0000 | mean_regret=0.0849 | dataset_count=5.0000 | 0.00s | Best average base learner: hist_gradient_boosting |

### Caveats
- This runner uses a compact OpenML-style suite and built-in datasets rather than the full CC18 benchmark.
- Scores come from single train/test splits per dataset, not repeated folds.
- The meta-selector is deliberately simple so it stays CPU-only and transparent.

## Synthetic Tabular Data Privacy

**Dataset:** Credit Card Fraud Detection

**Best experiment:** dp_histogram with roc_auc=0.8614

The best privacy-utility trade-off came from dp_histogram. It preserved downstream ROC AUC at 0.861 while keeping average nearest-neighbor distance at 32.368.

### Key Findings
- Best trade-off generator: dp_histogram.
- Bootstrap tends to maximize utility but also produces the weakest privacy proxy scores.
- Adding noise through independent histograms improves privacy distance but usually degrades fraud-detection fidelity.

### Recorded Experiments
| Algorithm | Features | Optimization | Primary | Secondary | Tertiary | Runtime | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dp_histogram | numeric_credit_features | utility_privacy_tradeoff | roc_auc=0.8614 | privacy_distance=32.3679 | match_rate=0.0000 | 0.05s | Downstream AP=0.672 |
| gaussian_copula | numeric_credit_features | utility_privacy_tradeoff | roc_auc=0.3441 | privacy_distance=6.7742 | match_rate=0.0000 | 0.26s | Downstream AP=0.052 |
| bootstrap | numeric_credit_features | utility_privacy_tradeoff | roc_auc=0.9853 | privacy_distance=0.0000 | match_rate=0.9737 | 0.00s | Downstream AP=0.924 |

### Caveats
- Privacy is approximated with nearest-neighbor and exact-match proxies rather than a formal privacy proof.
- If the credit card fraud dataset is unavailable, the runner uses a synthetic imbalanced fraud process.
- The Gaussian copula implementation is intentionally lightweight and numeric-only.

## Automated Feature Engineering via Evolutionary Algorithms

**Dataset:** House Prices / Ames-style Regression

**Best experiment:** random_forest_raw with rmse=32402.3127

Evolutionary interaction search produced 4 engineered features. The best downstream model was random_forest_raw with RMSE 32402.313.

### Key Findings
- Best model: random_forest_raw with RMSE 32402.313.
- Top evolved formulas included mul(OverallQual,BsmtFinSF2); mul(MasVnrArea,BsmtUnfSF); div(YearBuilt,WoodDeckSF).
- Feature search is most useful when paired with a simpler downstream learner that benefits from crafted interactions.

### Recorded Experiments
| Algorithm | Features | Optimization | Primary | Secondary | Tertiary | Runtime | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| random_forest_raw | raw_numeric | baseline_features | rmse=32402.3127 | feature_count=37.0000 | engineered_features=0.0000 | 0.49s | No engineered formulas |
| hist_gradient_evolved | evolved_interactions | evolutionary_formula_search | rmse=33756.3853 | feature_count=41.0000 | engineered_features=4.0000 | 1.45s | mul(OverallQual,BsmtFinSF2); mul(MasVnrArea,BsmtUnfSF); div(YearBuilt,WoodDeckSF) |
| ridge_raw | raw_numeric | baseline_features | rmse=39888.4086 | feature_count=37.0000 | engineered_features=0.0000 | 0.00s | No engineered formulas |
| ridge_evolved | evolved_interactions | evolutionary_formula_search | rmse=40036.4730 | feature_count=41.0000 | engineered_features=4.0000 | 0.01s | mul(OverallQual,BsmtFinSF2); mul(MasVnrArea,BsmtUnfSF); div(YearBuilt,WoodDeckSF) |

### Caveats
- If the Kaggle house-prices dataset is unavailable, the runner uses OpenML House Prices and then a diabetes-regression fallback.
- The search explores only arithmetic pairwise formulas, not arbitrary symbolic programs.
- A single validation split is used as the fitness function for speed.

## Imbalanced Data in Graph Neural Networks

**Dataset:** Elliptic-style Illicit Transaction Graph

**Best experiment:** logistic_graph_features with average_precision=0.8396

The strongest illicit-node detector was logistic_graph_features with average precision 0.840. Adding neighbor aggregation improved minority recall on the imbalanced graph.

### Key Findings
- Best graph-aware model: logistic_graph_features.
- Neighbor aggregation generally improved illicit-class recall relative to node-only baselines.
- Average precision is the most informative metric here because the positive class is intentionally rare.

### Recorded Experiments
| Algorithm | Features | Optimization | Primary | Secondary | Tertiary | Runtime | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_graph_features | graph_augmented | neighbor_message_features | average_precision=0.8396 | balanced_accuracy=0.9242 | minority_recall=0.9091 | 0.06s | Graph-aware surrogate for low-resource GNN benchmarking |
| logistic_raw | node_only | tabular_baseline | average_precision=0.8203 | balanced_accuracy=0.9218 | minority_recall=0.9091 | 0.01s | Graph-aware surrogate for low-resource GNN benchmarking |
| random_forest_graph_features | graph_augmented | neighbor_message_features | average_precision=0.7051 | balanced_accuracy=0.6795 | minority_recall=0.3636 | 0.32s | Graph-aware surrogate for low-resource GNN benchmarking |
| random_forest_raw | node_only | tabular_baseline | average_precision=0.7022 | balanced_accuracy=0.6818 | minority_recall=0.3636 | 0.32s | Graph-aware surrogate for low-resource GNN benchmarking |

### Caveats
- This runner uses a synthetic Elliptic-style graph because the Kaggle dataset is not bundled in the workspace.
- The graph-aware models are lightweight message-passing surrogates, not full neural GNNs.
- Only a single random train/test split is used.

## Tree-Based Models vs. Noise

**Dataset:** Breast Cancer Wisconsin (Diagnostic)

**Best experiment:** extra_trees with balanced_accuracy=0.9623

The strongest clean-data model was extra_trees, but the most noise-robust model under the combined stress test was extra_trees with only 0.019 balanced-accuracy loss.

### Key Findings
- Most robust under combined noise: extra_trees.
- Clean-data winner: extra_trees at 0.962 balanced accuracy.
- Gaussian noise and missingness hurt the MLP more than the tree ensembles in this quick benchmark.

### Recorded Experiments
| Algorithm | Features | Optimization | Primary | Secondary | Tertiary | Runtime | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| extra_trees | gaussian_0.2 | noise_stress_test | balanced_accuracy=0.9623 | noise_strength=1.0000 | test_size=143.0000 | 0.17s | Scenario=gaussian_0.2 |
| extra_trees | gaussian_0.4 | noise_stress_test | balanced_accuracy=0.9623 | noise_strength=1.0000 | test_size=143.0000 | 0.18s | Scenario=gaussian_0.4 |
| extra_trees | clean | noise_stress_test | balanced_accuracy=0.9606 | noise_strength=0.0000 | test_size=143.0000 | 0.19s | Scenario=clean |
| random_forest | gaussian_0.2 | noise_stress_test | balanced_accuracy=0.9606 | noise_strength=1.0000 | test_size=143.0000 | 0.25s | Scenario=gaussian_0.2 |
| random_forest | label_flip_0.1 | noise_stress_test | balanced_accuracy=0.9606 | noise_strength=1.0000 | test_size=143.0000 | 0.48s | Scenario=label_flip_0.1 |
| extra_trees | label_flip_0.1 | noise_stress_test | balanced_accuracy=0.9606 | noise_strength=1.0000 | test_size=143.0000 | 0.27s | Scenario=label_flip_0.1 |
| hist_gradient_boosting | clean | noise_stress_test | balanced_accuracy=0.9567 | noise_strength=0.0000 | test_size=143.0000 | 0.43s | Scenario=clean |
| random_forest | clean | noise_stress_test | balanced_accuracy=0.9512 | noise_strength=0.0000 | test_size=143.0000 | 0.27s | Scenario=clean |
| hist_gradient_boosting | gaussian_0.2 | noise_stress_test | balanced_accuracy=0.9512 | noise_strength=1.0000 | test_size=143.0000 | 0.61s | Scenario=gaussian_0.2 |
| random_forest | gaussian_0.4 | noise_stress_test | balanced_accuracy=0.9512 | noise_strength=1.0000 | test_size=143.0000 | 0.23s | Scenario=gaussian_0.4 |
| hist_gradient_boosting | label_flip_0.1 | noise_stress_test | balanced_accuracy=0.9478 | noise_strength=1.0000 | test_size=143.0000 | 2.24s | Scenario=label_flip_0.1 |
| extra_trees | combined_0.2 | noise_stress_test | balanced_accuracy=0.9417 | noise_strength=1.0000 | test_size=143.0000 | 0.17s | Scenario=combined_0.2 |
| extra_trees | missing_0.15 | noise_stress_test | balanced_accuracy=0.9362 | noise_strength=1.0000 | test_size=143.0000 | 0.19s | Scenario=missing_0.15 |
| hist_gradient_boosting | missing_0.15 | noise_stress_test | balanced_accuracy=0.9362 | noise_strength=1.0000 | test_size=143.0000 | 2.21s | Scenario=missing_0.15 |
| random_forest | missing_0.15 | noise_stress_test | balanced_accuracy=0.9345 | noise_strength=1.0000 | test_size=143.0000 | 0.24s | Scenario=missing_0.15 |
| hist_gradient_boosting | gaussian_0.4 | noise_stress_test | balanced_accuracy=0.9251 | noise_strength=1.0000 | test_size=143.0000 | 0.56s | Scenario=gaussian_0.4 |
| random_forest | combined_0.2 | noise_stress_test | balanced_accuracy=0.9251 | noise_strength=1.0000 | test_size=143.0000 | 0.27s | Scenario=combined_0.2 |
| hist_gradient_boosting | combined_0.2 | noise_stress_test | balanced_accuracy=0.9251 | noise_strength=1.0000 | test_size=143.0000 | 0.45s | Scenario=combined_0.2 |
| mlp | clean | noise_stress_test | balanced_accuracy=0.9101 | noise_strength=0.0000 | test_size=143.0000 | 0.10s | Scenario=clean |
| mlp | missing_0.15 | noise_stress_test | balanced_accuracy=0.8873 | noise_strength=1.0000 | test_size=143.0000 | 0.47s | Scenario=missing_0.15 |
| mlp | label_flip_0.1 | noise_stress_test | balanced_accuracy=0.8774 | noise_strength=1.0000 | test_size=143.0000 | 0.43s | Scenario=label_flip_0.1 |
| mlp | gaussian_0.4 | noise_stress_test | balanced_accuracy=0.8024 | noise_strength=1.0000 | test_size=143.0000 | 0.18s | Scenario=gaussian_0.4 |
| mlp | gaussian_0.2 | noise_stress_test | balanced_accuracy=0.7951 | noise_strength=1.0000 | test_size=143.0000 | 0.12s | Scenario=gaussian_0.2 |
| mlp | combined_0.2 | noise_stress_test | balanced_accuracy=0.7684 | noise_strength=1.0000 | test_size=143.0000 | 0.21s | Scenario=combined_0.2 |

### Caveats
- The noise process is synthetic and meant to stress-test relative robustness, not replicate a specific medical workflow.
- Only one train/test split is used.
- The quick benchmark caps MLP iterations for runtime control.

## Memory-Safe Feature Extraction Pipelines

**Dataset:** NSL-KDD

**Best experiment:** feature_hashing_pipeline with balanced_accuracy=0.7865

The best memory-safe pipeline was feature_hashing_pipeline, balancing balanced accuracy 0.786 against peak memory 10.0 MB. Sparse and hashed representations cut allocation pressure relative to dense one-hot encoding.

### Key Findings
- Lowest-memory competitive pipeline: feature_hashing_pipeline.
- Sparse one-hot usually keeps most of the linear-model accuracy while lowering peak allocation.
- Feature hashing is useful when the categorical vocabulary grows faster than the available RAM budget.

### Recorded Experiments
| Algorithm | Features | Optimization | Primary | Secondary | Tertiary | Runtime | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| feature_hashing_pipeline | hashed_streaming_features | fixed_width_hashing | balanced_accuracy=0.7865 | peak_memory_mb=10.0283 | runtime_sec=1.0423 | 1.04s | Hash-based fixed-width representation |
| dense_get_dummies | copy_heavy_dense_matrix | baseline_dense_one_hot | balanced_accuracy=0.7762 | peak_memory_mb=11.2398 | runtime_sec=0.6534 | 0.65s | Dense pandas one-hot representation |
| sparse_one_hot_pipeline | sparse_encoder | column_transformer_sparse | balanced_accuracy=0.7495 | peak_memory_mb=6.9631 | runtime_sec=2.2701 | 2.27s | Sparse encoder with solver-friendly linear model |

### Caveats
- Peak memory is measured with Python-level tracemalloc, which is a proxy rather than full native-process RSS.
- This benchmark requires the NSL-KDD train and test files, which are fetched from a public mirror if missing locally.
- The quick benchmark subsamples both train and test sets for runtime control.

## Algorithmic Optimization of Tree Ensembles

**Dataset:** Covertype

**Best experiment:** extra_trees with accuracy=0.8764

The best accuracy-speed compromise was extra_trees, reaching accuracy 0.876 with fit time 1.4s and inference latency 81.83 ms per 1k rows.

### Key Findings
- Best overall trade-off: extra_trees.
- Float32 features are sufficient for this style of ensemble benchmark and reduce memory pressure.
- Inference latency can invert the ranking when two ensembles are close on accuracy.

### Recorded Experiments
| Algorithm | Features | Optimization | Primary | Secondary | Tertiary | Runtime | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| extra_trees | float32_features | tree_ensemble_benchmark | accuracy=0.8764 | fit_seconds=1.4314 | predict_ms_per_1k=81.8263 | 1.43s | Accuracy-speed trade-off on medium-scale tabular forest cover data |
| random_forest | float32_features | tree_ensemble_benchmark | accuracy=0.8768 | fit_seconds=1.6279 | predict_ms_per_1k=83.0582 | 1.63s | Accuracy-speed trade-off on medium-scale tabular forest cover data |
| random_forest_shallow | float32_features | tree_ensemble_benchmark | accuracy=0.8341 | fit_seconds=0.9756 | predict_ms_per_1k=29.8217 | 0.98s | Accuracy-speed trade-off on medium-scale tabular forest cover data |
| hist_gradient_boosting_fast | float32_features | tree_ensemble_benchmark | accuracy=0.8539 | fit_seconds=22.7782 | predict_ms_per_1k=55.0860 | 22.78s | Accuracy-speed trade-off on medium-scale tabular forest cover data |
| hist_gradient_boosting | float32_features | tree_ensemble_benchmark | accuracy=0.8604 | fit_seconds=30.2415 | predict_ms_per_1k=42.8518 | 30.24s | Accuracy-speed trade-off on medium-scale tabular forest cover data |
| extra_trees_shallow | float32_features | tree_ensemble_benchmark | accuracy=0.7728 | fit_seconds=0.5953 | predict_ms_per_1k=36.3183 | 0.60s | Accuracy-speed trade-off on medium-scale tabular forest cover data |

### Caveats
- The quick benchmark subsamples Covertype to keep fit times reasonable.
- Latency is measured with wall-clock timing on this machine, not controlled hardware counters.
- If Covertype is unavailable, a synthetic multiclass fallback is used.

## Numerical Stability in Label Encoding

**Dataset:** Categorical Encoding Challenge-style Synthetic Benchmark

**Best experiment:** target_encoding with roc_auc=0.9122

The best encoding strategy was target_encoding with ROC AUC 0.912. Wrapped uint8 codes introduced collisions that materially degraded ranking performance relative to stable or sparse encodings.

### Key Findings
- Best encoding: target_encoding.
- Wrapped uint8 collision rate was 0.224.
- The encoding failure mode shows up as both lower ROC AUC and unstable feature semantics when categories collide.

### Recorded Experiments
| Algorithm | Features | Optimization | Primary | Secondary | Tertiary | Runtime | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| target_encoding | high_cardinality_categories | smoothed_mean_encoding | roc_auc=0.9122 | collision_rate=0.0000 | runtime_sec=0.1050 | 0.10s | Smoothed train-only target means |
| one_hot_sparse | high_cardinality_categories | sparse_indicator_matrix | roc_auc=0.8995 | collision_rate=0.0000 | runtime_sec=1.5621 | 1.56s | Collision-free reference encoding |
| ordinal_int32 | high_cardinality_categories | stable_integer_codes | roc_auc=0.6970 | collision_rate=0.0000 | runtime_sec=0.1422 | 0.14s | Reference ordinal encoding without wraparound |
| frequency_encoding | high_cardinality_categories | category_frequency_prior | roc_auc=0.6963 | collision_rate=0.0000 | runtime_sec=0.0226 | 0.02s | Category frequency encoding without target leakage |
| ordinal_uint8_wrapped | high_cardinality_categories | uint8_wraparound | roc_auc=0.6971 | collision_rate=0.2241 | runtime_sec=0.0859 | 0.09s | Purposefully wrapped integer codes to test collisions |
| hash_bucket_256 | high_cardinality_categories | hash_bucket_encoding | roc_auc=0.6941 | collision_rate=0.4167 | runtime_sec=0.2270 | 0.23s | Hash bucket encoding with explicit collision accounting |

### Caveats
- This is a synthetic high-cardinality benchmark rather than the full Kaggle encoding challenge dataset.
- Target encoding is implemented with simple train-only smoothing and no nested out-of-fold protection.
- The quick benchmark focuses on numerical stability effects, not full leaderboard-style tuning.

## Security-Aware ML for Intrusion Detection

**Dataset:** UNSW-NB15-style Intrusion Detection

**Best experiment:** hist_gradient_boosting_aggressive with macro_f1=0.7173

The best security-aware detector was hist_gradient_boosting_aggressive, combining macro F1 0.717 with attack recall 0.612. Cost-sensitive tuning helped the models prioritize attack coverage.

### Key Findings
- Best detector: hist_gradient_boosting_aggressive.
- Weighted forests and lowered decision thresholds improved attack recall relative to untuned baselines.
- Average precision is a useful secondary metric when attack prevalence is low.

### Recorded Experiments
| Algorithm | Features | Optimization | Primary | Secondary | Tertiary | Runtime | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hist_gradient_boosting_aggressive | network_flow_features | cost_sensitive_thresholding | macro_f1=0.7173 | attack_recall=0.6120 | average_precision=0.5582 | 2.63s | Decision threshold=0.25; More aggressive thresholding for recall-sensitive deployments |
| hist_gradient_boosting | network_flow_features | cost_sensitive_thresholding | macro_f1=0.7259 | attack_recall=0.5137 | average_precision=0.5582 | 2.66s | Decision threshold=0.35; Boosted tree baseline |
| logistic_regression | network_flow_features | cost_sensitive_thresholding | macro_f1=0.6908 | attack_recall=0.6448 | average_precision=0.4981 | 1.94s | Decision threshold=0.50; Balanced logistic baseline |
| weighted_random_forest | network_flow_features | cost_sensitive_thresholding | macro_f1=0.7151 | attack_recall=0.4699 | average_precision=0.5403 | 0.96s | Decision threshold=0.35; Cost-sensitive forest with lower threshold |
| logistic_regression_low_threshold | network_flow_features | cost_sensitive_thresholding | macro_f1=0.6255 | attack_recall=0.8087 | average_precision=0.4981 | 1.59s | Decision threshold=0.35; Lower threshold for higher attack recall |
| weighted_random_forest_conservative | network_flow_features | cost_sensitive_thresholding | macro_f1=0.6758 | attack_recall=0.3251 | average_precision=0.5316 | 0.94s | Decision threshold=0.45; Heavier positive weighting with less aggressive threshold |
| random_forest | network_flow_features | cost_sensitive_thresholding | macro_f1=0.6556 | attack_recall=0.2842 | average_precision=0.5350 | 1.31s | Decision threshold=0.50; Untuned forest baseline |

### Caveats
- If UNSW-NB15 is not present locally, the runner uses a synthetic intrusion dataset with similar class imbalance and mixed feature types.
- The quick benchmark uses a single train/test split.
- Thresholds are manually chosen and not optimized with a separate validation set.

## Deterministic Agentic Data Validation

**Dataset:** NYC Taxi Trip Duration

**Best experiment:** validated_hist_gradient_enriched with rmse=83.0233

Deterministic validation improved trip-duration modeling by filtering 209 anomalous training rows. The best model was validated_hist_gradient_enriched with RMSE 83.02.

### Key Findings
- Best validated model: validated_hist_gradient_enriched.
- The validation agent flagged 209 anomalous rows in the training split.
- Coordinate sanity checks, passenger-count rules, and zero-duration filters are enough to recover a cleaner learning signal.

### Recorded Experiments
| Algorithm | Features | Optimization | Primary | Secondary | Tertiary | Runtime | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| validated_hist_gradient_enriched | validated_features | deterministic_rule_filter_plus_enriched_features | rmse=83.0233 | flagged_rows=209.0000 | runtime_sec=0.7392 | 0.74s | zero_duration=0, invalid_passengers=71, pickup_outside_nyc=68, dropoff_outside_nyc=70 |
| validated_random_forest_enriched | validated_features | deterministic_rule_filter_plus_enriched_features | rmse=85.5462 | flagged_rows=209.0000 | runtime_sec=0.7207 | 0.72s | zero_duration=0, invalid_passengers=71, pickup_outside_nyc=68, dropoff_outside_nyc=70 |
| validated_hist_gradient | validated_features | deterministic_rule_filter | rmse=94.5744 | flagged_rows=209.0000 | runtime_sec=4.9145 | 4.91s | zero_duration=0, invalid_passengers=71, pickup_outside_nyc=68, dropoff_outside_nyc=70 |
| validated_random_forest | validated_features | deterministic_rule_filter | rmse=99.1610 | flagged_rows=209.0000 | runtime_sec=1.2880 | 1.29s | zero_duration=0, invalid_passengers=71, pickup_outside_nyc=68, dropoff_outside_nyc=70 |
| raw_random_forest | raw_features | minimal_cleaning | rmse=99.4786 | flagged_rows=0.0000 | runtime_sec=1.2749 | 1.27s | zero_duration=0, invalid_passengers=71, pickup_outside_nyc=68, dropoff_outside_nyc=70 |

### Caveats
- If the NYC Taxi dataset is not present locally, the runner uses a synthetic taxi-duration dataset with injected anomalies.
- Validation is deterministic and rule-based rather than an LLM-driven agent.
- The quick benchmark uses a single train/test split and compact feature set.
