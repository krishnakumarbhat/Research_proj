# Publish-Oriented Literature Review Draft for Small-Data, CPU-First ML Research

## Abstract

This review synthesizes the research landscape behind fifteen compact-data machine learning topics that can be studied on commodity CPUs with modest RAM: time-series causal discovery, conformal prediction for forecasting, Explainable Boosting Machines (EBMs), concept drift, low-bandwidth federated learning, tabular meta-learning, privacy-preserving synthetic data, automated feature engineering, imbalanced graph learning, robustness to noise, memory-safe feature pipelines, tree-ensemble optimization, numerical stability in encoding, security-aware intrusion detection, and deterministic data validation. The practical goal is not merely to summarize methods, but to identify which topics remain scientifically meaningful under strict resource constraints and which experimental designs can still produce publishable findings using megabyte-scale datasets. A cross-cutting pattern emerges: recent work increasingly values calibrated uncertainty, controllable efficiency, formal or domain-informed constraints, and benchmark rigor over brute-force model scale. This trend is especially favorable to CPU-first tabular research. Among the topics reviewed here, EBM-based high-stakes tabular modeling, conformal time-series forecasting, drift adaptation, and privacy-utility evaluation for synthetic tabular data appear particularly strong for low-data publishable studies.

## Scope and Review Strategy

This document is a publish-oriented narrative review rather than a formal PRISMA systematic review. It combines canonical background literature with recent arXiv and conference signals retrieved on 2026-03-31 for the topics represented in this repository. The emphasis is on methods that are feasible with less than 1 GB of data, can be meaningfully benchmarked without GPUs, and naturally connect to the runnable experiments in this repository.

The review asks four recurring questions.

1. What is the mature baseline for the topic?
2. What changed in the 2024-2026 literature?
3. What can realistically be reproduced or extended on a CPU-only setup?
4. What would make a paper from this topic credible rather than merely demonstrative?

## Cross-Cutting Themes in Recent Research

Four major trends connect nearly all of the topics in this benchmark.

First, data-centric evaluation has become more important than incremental model novelty. In several subfields, the most credible recent work does not merely introduce a new model but also clarifies benchmark protocol, distribution shift, privacy attack surfaces, or evaluation failure modes. IGL-Bench for imbalanced graph learning and recent time-series conformal benchmarking illustrate this clearly.

Second, the literature is moving from point prediction toward decision-aware uncertainty. Conformal forecasting, regime-aware uncertainty wrappers, and robust cybersecurity evaluation all reflect the same pressure: decision systems require calibrated bands, not just accurate means.

Third, interpretability is no longer treated as a purely post-hoc add-on. Recent EBM work increasingly injects domain constraints, monotonic structure, or even formal verification into the model-development loop. This matters for publishable small-data research because structural rigor often matters more than parameter count.

Fourth, systems efficiency is being reframed as a research contribution, not an implementation detail. Communication compression in federated learning, sparse or hashed tabular pipelines, and vectorized histogram construction for tree ensembles now appear as first-class research problems because deployment constraints are part of the real task.

## 1. Causal Discovery in Time-Series

The foundational line in time-series causal inference remains rooted in Granger causality, vector autoregression, and conditional independence testing. In practice, this family of methods asks whether a lagged signal improves prediction of another series after controlling for the past. For traffic and weather data, this is useful because it gives a falsifiable, temporal notion of precedence even when true intervention data are unavailable.

Modern causal discovery in time series has pushed beyond pairwise Granger tests into multivariate and graph-based frameworks such as PCMCI and related Tigramite tooling, which were designed to reduce false positives under autocorrelation and high-dimensional lag structures. The current literature also increasingly distinguishes between predictive precedence and intervention-level causality, a distinction that is critical for publishable claims. A credible paper in this area should therefore avoid causal overstatement and frame results as observational causal discovery under explicit assumptions.

For a small-data CPU-first project, the publishable angle is not to claim definitive causation from a compact traffic dataset, but to compare how causal conclusions change under different lag structures, exogenous controls, or nonstationarity adjustments. The most credible contribution would combine a strong temporal forecasting benchmark with cautious causal interpretation and ablation on confounding variables.

## 2. Conformal Prediction for Supply Chains and Forecasting

Conformal prediction has become one of the most important uncertainty-quantification frameworks for practitioners because it offers distribution-free coverage guarantees with weak modeling assumptions. The central challenge in time-series settings is that standard conformal guarantees assume exchangeability, which sequential data violate.

Recent work makes this challenge explicit. Shang's 2026 work on conformal prediction for high-dimensional functional time series contrasts split and sequential conformal methods in a forecasting context. Sabashvili's 2026 benchmarking paper surveys algorithmic categories for conformal time-series forecasting and emphasizes that the main design problem is how to respond to temporal dependence without losing calibration. Lu et al. 2025 extend this logic to regime-switching forecasts, showing that adaptive conformal wrappers can preserve useful coverage under nonstationarity.

For publication, the strongest contribution is usually not a brand-new conformal method but a careful comparison of interval quality across operational conditions: nominal coverage, average width, misspecification tolerance, and behavior under drift or promotion shocks. In inventory research, interval efficiency and decision impact matter more than abstract calibration alone.

## 3. Explainable Boosting Machines in High-Stakes Domains

EBMs inherit the generalized additive model intuition but improve flexibility through boosting, bagging, and optional low-order interactions. Their importance in finance and healthcare comes from the fact that they expose global shape functions directly, making them interpretable without sacrificing all nonlinear expressiveness.

The recent literature shows that EBM research is maturing from pure accuracy-versus-interpretability comparison into constraint-aware and verification-aware modeling. Hsiao et al. 2026 propose domain-informed EBMs that explicitly correct non-physical shape functions while preserving most predictive performance. Kumar 2026 studies formal verification of tree-based models and EBMs, demonstrating that unconstrained high-performing models can violate domain specifications and that a verify-fix-verify loop can improve trustworthiness. These works matter because they shift the question from whether EBMs are interpretable to whether the learned explanations are admissible under domain rules.

This is exactly why EBM research is so strong for minimal-data publication. The contribution can live in the explanation geometry, monotonic constraints, calibration, verification, or regulatory usability rather than in scaling. A compact credit or health dataset is enough if the paper clearly shows what the model learned, what constraints were imposed, and what performance trade-off followed.

## 4. Handling Concept Drift in Streaming Data

Concept drift research has historically focused on detectors and adaptive learners such as DDM, EDDM, ADWIN, sliding windows, and online trees. More recent work is reframing drift as a continual-learning problem with explicit forgetting, replay, or architecture growth.

Recent signals are especially interesting. Wozniak et al. 2026 connect machine unlearning to drift mitigation, suggesting that efficient removal of stale data can replace expensive full retraining. Halstead et al. 2026 propose FiCSUM, which emphasizes richer meta-information for concept fingerprints and recurring-concept recognition. Fujiwara et al. 2026 introduce CALIPER, a data-only method for determining when enough post-drift data have accumulated to justify retraining. Giannini et al. 2025 and 2024 extend the streaming-continual-learning line with MAGIC Net and MAcPNN, combining temporal dependence, architectural plasticity, and selective collaboration.

For publication, a small-data stream study becomes stronger when it evaluates not only accuracy after drift but also retraining timing, adaptation cost, memory footprint, and robustness to recurring concepts. That framing turns a toy drift experiment into a systems-level contribution.

## 5. Federated Learning under Extreme Low Bandwidth

Communication-efficient federated learning sits at the intersection of optimization, systems, and privacy. Classical baselines such as FedAvg, FedProx, SCAFFOLD, signSGD, and top-k sparsification established that communication can be compressed aggressively, but convergence and heterogeneity remain the main obstacles.

The newest literature sharpens this direction. FedCEF (Qiu et al. 2026) combines biased compression with error feedback and control variates for non-convex composite optimization. CA-HFP (Hu et al. 2026) studies curvature-aware pruning and model reconstruction for heterogeneous edge devices. Omeke et al. 2026 show that hierarchical communication structure and selective cooperative aggregation can dramatically reduce energy in ultra-low-bandwidth settings. Badi et al. 2026 extend communication efficiency to multimodal latent-space consensus, while semantic/split federated communication work in vehicular settings highlights that compression and protocol structure are now inseparable.

For a CPU-only publishable study, the best contribution is often an empirical frontier: accuracy versus bytes transmitted, robustness under heterogeneous clients, or error-feedback versus simple sparsification. A small tabular dataset can still support a useful paper if the communication budget is the primary independent variable.

## 6. Tabular Meta-Learning

Tabular meta-learning has been transformed by Prior-Data Fitted Networks and TabPFN-style zero-shot prediction. The core idea is that a model can learn an implicit algorithm for small supervised tasks by pretraining on a large distribution of synthetic tasks rather than on one fixed benchmark.

Wu and Bergman's 2025 APT paper extends this line by using adversarially pre-trained synthetic task generators and a more flexible class-handling architecture. The importance of this result is methodological: it argues that the quality and diversity of synthetic task priors may matter as much as downstream architecture choice.

A publishable CPU-first angle here is not to compete with foundation-scale tabular meta-learners, but to study lightweight task descriptors, warm-start policies, and model recommendation strategies on curated small suites. If a simple meta-feature recommender consistently reduces search cost without hurting accuracy, that is valuable and feasible.

## 7. Privacy-Preserving Synthetic Tabular Data

Synthetic tabular data research now sits in tension between utility, privacy, and controllability. Earlier work concentrated on CTGAN, TVAE, copula models, and differential privacy variants. The recent literature is expanding along three axes: stronger generators, better privacy evaluation, and structure-aware generation.

Recent papers show where the field is moving. MIDST 2026 centers membership inference attacks on diffusion-generated synthetic tabular data, emphasizing that synthetic data are not automatically private. ReTabSyn 2026 introduces reinforcement learning to prioritize predictive signal preservation under small-data and imbalanced settings. Tugnoli et al. 2026 inject causal structure into TabPFN-style synthetic generation to improve structural fidelity and treatment-effect preservation. TabDLM 2026 extends synthesis to tables that include free-form text, while Jakobsen et al. 2026 show that retrieval-grounded LLMs can generate synthetic psychiatric data without direct access to real training examples.

For publication, the key is to report privacy-utility trade-offs with explicit attack proxies or formal privacy guarantees rather than only synthetic realism scores. A good paper on a small fraud dataset can still be strong if it shows how utility changes under increasingly strict privacy constraints and whether causal or class-imbalance structure is preserved.

## 8. Automated Feature Engineering via Evolutionary Search

Automated feature engineering has evolved from manual transformation libraries and exhaustive search toward differentiable, evolutionary, and LLM-guided program search. The older but still relevant line includes feature-construction systems such as Featuretools and algorithmic methods like AutoCross. More recent work emphasizes adaptive search spaces and semantic proposal mechanisms.

DIFER provided an early differentiable route for automated feature engineering. More recent work broadens the design space. ELATE 2025 uses a language model inside an evolutionary framework for time-series feature engineering, and LLM-FE 2025 explicitly treats feature engineering as a program-search problem in which LLM proposals are refined by evolutionary feedback. This is important because it suggests that future AutoFE systems may blend symbolic search, data feedback, and language priors instead of choosing only one paradigm.

For a publishable compact project, the strongest question is usually not whether evolutionary search beats all baselines, but when it is worth the extra search cost. A good paper can analyze search efficiency, discovered feature semantics, model simplification, and transferability across related tabular tasks.

## 9. Imbalanced Graph Learning and GNNs

Class imbalance in graph learning is more difficult than imbalance in IID tabular data because topology itself amplifies bias. Minority nodes often suffer from both label scarcity and poor message propagation, which means that naive class reweighting is often insufficient.

The literature has become much more structured recently. BAT reframed the problem as topological bias rather than simple class-ratio bias and showed that graph augmentation can mitigate imbalance without conventional class rebalancing. IGL-Bench, accepted to ICLR 2025, established much-needed evaluation consistency across datasets, algorithms, and imbalance conditions. Uni-GNN 2024 then pushed toward unified structural and semantic connectivity, expanding message propagation beyond immediate neighbors and using balanced pseudo-labeling for minority classes.

A CPU-first publication strategy here is to build strong graph-aware non-neural baselines before escalating to full GNNs. If neighborhood aggregation and topological augmentation explain much of the gain, that is a publishable and practically useful result, especially in fraud or illicit-transaction settings.

## 10. Tree-Based Models versus Noise

Noise-robust tabular learning has not coalesced into a single fashionable subfield, but it remains essential for scientific credibility. Many benchmark gains disappear once label flips, missingness, or feature perturbations are introduced. Tree ensembles are often strong here because split-based partitioning can remain stable under moderate corruption, although not indefinitely.

The recent literature on robust tabular learning increasingly favors stress-testing, calibration, and noise-aware objectives over raw clean-data accuracy. This is less a single-method literature than a convergence of data-centric AI, robust statistics, and reliability evaluation. For publication, that means a well-designed degradation study can be more valuable than a marginal accuracy improvement on clean data, especially if it compares multiple corruption types and reports degradation curves rather than one-off noisy scores.

## 11. Memory-Safe Feature Extraction Pipelines

Memory-efficient tabular pipelines are under-discussed in mainstream ML papers even though they determine whether practical systems can run at all. In cybersecurity and logging domains, the key challenge is often not model capacity but feature explosion from categorical encodings, copies, and intermediate materialization.

The systems literature increasingly treats sparse encodings, hashing, Arrow-compatible representations, and streaming or zero-copy interfaces as part of the learning problem. The research opportunity for a small-data CPU-first project is therefore clear: benchmark the accuracy-memory-runtime trade-off of dense, sparse, and hashed pipelines under realistic operational constraints. This is publishable when framed as a reproducible systems evaluation rather than a coding anecdote.

## 12. Algorithmic Optimization of Tree Ensembles

Tree ensembles remain among the strongest models for medium-scale tabular data, but recent research has shifted from pure predictive comparisons toward algorithmic and hardware efficiency. Histogram construction, feature quantization, SIMD vectorization, GPU acceleration, and specialized oblique-split implementations now determine practical throughput.

The 2026 work on vectorized adaptive histograms for sparse oblique forests is a good example: it treats split finding and histogram construction as systems bottlenecks and shows speedups against existing forest implementations. Research on distributed or federated XGBoost variants similarly underscores that preserving the optimization structure of tree methods under systems constraints is now a legitimate research direction.

A publishable CPU-only study should therefore report training time, inference latency, memory usage, and sometimes calibration or robustness alongside accuracy. That framing better matches the current literature than a leaderboard-style comparison alone.

## 13. Numerical Stability in Categorical Encoding

Encoding research spans one-hot, ordinal, target encoding, count or frequency encoding, hashing, and ordered target statistics popularized by CatBoost. While most applied papers treat encoding as preprocessing, the stability consequences can be severe: collisions, leakage, overfitting, and silent dtype wraparound can all materially change model behavior.

This topic is especially attractive for resource-limited publishable work because the experimental cost is low but the engineering relevance is high. A strong paper would isolate collision-induced degradation, compare leakage-safe target encoders, and quantify when hash-bucket compression becomes unacceptable. This is a place where a carefully designed synthetic benchmark can be scientifically useful because the failure modes are easier to control.

## 14. Security-Aware Intrusion Detection

Intrusion detection research is moving from single-score classification toward trustworthy, adaptive, and explainable security analytics. Recent work is especially concerned with how robust and interpretable a detector remains under drift, adversarial perturbation, or explanation demands from analysts.

Rajhans and Khawarey 2026 study adversarial robustness and explainability drift in cybersecurity classifiers, showing that robustness and explanation stability degrade together. ExpIDS 2025 explicitly targets drift-adaptable explainability in a network intrusion setting. eX-NIDS 2025 uses LLMs to generate richer explanations for malicious flows, while Galwaduge and Samarabandu 2025 propose diffusion-based actionable counterfactual explanations that move beyond attribution toward operational response.

For publication, the best framing is often cost-sensitive and analyst-centered: maximize attack recall under acceptable false-alarm rates, then quantify explanation usefulness or robustness. The field increasingly rewards this broader evaluation perspective.

## 15. Deterministic Agentic Data Validation

Data validation is receiving renewed attention because agentic and LLM-integrated systems are exposing how brittle downstream pipelines become when inputs violate silent assumptions. Traditional deterministic tools such as schema validators, range checks, and rule-based expectation frameworks remain extremely strong because they provide auditable failures and predictable remediation.

Recent work on agentic AI faults reinforces this rather than replacing it. Shah et al. 2026 show that many agentic failures arise from mismatches between probabilistic artifacts and deterministic interface constraints, including data validation and runtime environment handling. This implies a practical design principle for publishable data-quality research: use deterministic validation as the contract layer, and let more flexible AI components operate on top of already validated data.

For a CPU-first study, a strong contribution is to quantify how much predictive performance and operational reliability improve once a deterministic validation layer removes obvious anomalies. That kind of paper is feasible on messy trip, healthcare, or transaction datasets and has clear production value.

## Synthesis: Which Topics Are Strongest for Small-Data Publication?

Four topics stand out for particularly strong publishable potential under strict data and hardware constraints.

1. EBMs in high-stakes tabular modeling because the contribution can focus on explanation quality, calibration, monotonicity, or verification rather than scale.
2. Conformal forecasting because uncertainty quality, interval efficiency, and adaptation under shift are publishable even on small operational datasets.
3. Concept drift because adaptation policy, retraining sufficiency, and streaming efficiency can be studied on compact benchmarks without GPUs.
4. Synthetic tabular privacy because privacy attacks, utility trade-offs, and structure-preservation questions remain open and experimentally tractable.

The repository benchmark supports this prioritization. In the current experiment suite, dense-interaction EBM variants remain competitive against black-box baselines on compact credit data, conformal tree methods deliver strong interval coverage on supply-chain demand, and lightweight adaptive learners remain informative under drift. These are exactly the kinds of results that translate well into resource-aware research papers.

## How to Turn This into a Publishable Paper

A publishable manuscript should narrow to one primary question rather than attempting to cover all fifteen topics. The benchmark in this repository is best used as a screening instrument that identifies a high-value direction and provides comparative context.

For the EBM direction, the paper should be framed around interpretable risk modeling under regulatory constraints, with ablations on interaction count, calibration, and domain constraints.

For conformal forecasting, the paper should be framed around operational uncertainty under temporal dependence, with explicit emphasis on coverage, width, and nonstationarity.

For concept drift, the paper should focus on adaptation policy rather than only learner choice, comparing online updates, rolling retraining, and detection-triggered adaptation.

For synthetic data privacy, the paper should report utility and privacy together, preferably including membership or linkage risk proxies and class-imbalance preservation.

## Selected References

1. Lou, Y., Caruana, R., Gehrke, J., and Hooker, G. Accurate intelligible models with pairwise interactions. 2013.
2. Nori, H., Jenkins, S., Koch, P., and Caruana, R. Interpretml: A unified framework for machine learning interpretability. 2019.
3. Hsiao, C.-H., Kumar, K., and Rathje, E. M. Domain-informed explainable boosting machines for trustworthy lateral spread predictions. 2026.
4. Kumar, K. Formal verification of tree-based machine learning models for lateral spreading. 2026.
5. Angelopoulos, A. N., and Bates, S. Conformal prediction: A gentle introduction. 2023.
6. Shang, H. L. Conformal prediction for high-dimensional functional time series: Applications to subnational mortality. 2026.
7. Sabashvili, A. Conformal Prediction Algorithms for Time Series Forecasting: Methods and Benchmarking. 2026.
8. Lu, E. D., Findling, C., Clausel, M., Leite, A., Gong, W., and Kersaudy, P. Adaptive Regime-Switching Forecasts with Distribution-Free Uncertainty: Deep Switching State-Space Models Meet Conformal Prediction. 2025.
9. Bifet, A., and Gavalda, R. Learning from time-changing data with adaptive windowing. 2007.
10. Wozniak, M., Klonowski, M., Maczynski, M., and Krawczyk, B. Unlearning-based sliding window for continual learning under concept drift. 2026.
11. Halstead, B., Koh, Y. S., Riddle, P., Pechenizkiy, M., Bifet, A., and Pears, R. Fingerprinting Concepts in Data Streams with Supervised and Unsupervised Meta-Information. 2026.
12. Fujiwara, R., Matsubara, Y., and Sakurai, Y. When to Retrain after Drift: A Data-Only Test of Post-Drift Data Size Sufficiency. 2026.
13. McMahan, B. et al. Communication-efficient learning of deep networks from decentralized data. 2017.
14. Qiu, P., Ouyang, C., Xiong, Y., You, K., Liu, W., and Shi, Y. Compressed Proximal Federated Learning for Non-Convex Composite Optimization on Heterogeneous Data. 2026.
15. Hu, G., Teng, Y., Wu, P., and Ma, S. CA-HFP: Curvature-Aware Heterogeneous Federated Pruning with Model Reconstruction. 2026.
16. Omeke, K., Mollel, M., Zhang, L., Abbasi, Q. H., and Imran, M. A. Energy-Efficient Hierarchical Federated Anomaly Detection for the Internet of Underwater Things via Selective Cooperative Aggregation. 2026.
17. Hollmann, N. et al. TabPFN: A transformer that solves small tabular classification problems in a second. 2023.
18. Wu, Y., and Bergman, D. L. Zero-shot Meta-learning for Tabular Prediction Tasks with Adversarially Pre-trained Transformer. 2025.
19. Xu, L., Skoularidou, M., Cuesta-Infante, A., and Veeramachaneni, K. Modeling tabular data using conditional GAN. 2019.
20. Shafieinejad, M. et al. MIDST Challenge at SaTML 2025: Membership Inference over Diffusion-models-based Synthetic Tabular Data. 2026.
21. Lin, X. et al. ReTabSyn: Realistic Tabular Data Synthesis via Reinforcement Learning. 2026.
22. Tugnoli, D. et al. Improving TabPFN's Synthetic Data Generation by Integrating Causal Structure. 2026.
23. Murray, A. et al. ELATE: Evolutionary Language Model for Automated Time-series Engineering. 2025.
24. Abhyankar, N., Shojaee, P., and Reddy, C. K. LLM-FE: Automated Feature Engineering for Tabular Data with LLMs as Evolutionary Optimizers. 2025.
25. Liu, Z. et al. Class-Imbalanced Graph Learning without Class Rebalancing. 2024.
26. Qin, J. et al. IGL-Bench: Establishing the Comprehensive Benchmark for Imbalanced Graph Learning. 2025.
27. Alchihabi, A., Yan, H., and Guo, Y. Overcoming Class Imbalance: Unified GNN Learning with Structural and Semantic Connectivity Representations. 2024.
28. Rajhans, M., and Khawarey, V. Empirical Analysis of Adversarial Robustness and Explainability Drift in Cybersecurity Classifiers. 2026.
29. Kumar, A., Fok, K. W., and Thing, V. L. L. ExpIDS: A Drift-adaptable Network Intrusion Detection System With Improved Explainability. 2025.
30. Houssel, P. R. B., Layeghy, S., Singh, P., and Portmann, M. eX-NIDS: A Framework for Explainable Network Intrusion Detection Leveraging Large Language Models. 2025.
31. Galwaduge, V., and Samarabandu, J. Tabular Diffusion based Actionable Counterfactual Explanations for Network Intrusion Detection. 2025.
32. Lubonja, A. et al. Vectorized Adaptive Histograms for Sparse Oblique Forests. 2026.
33. Shah, M. B., Morovati, M. M., Rahman, M. M., and Khomh, F. Characterizing Faults in Agentic AI: A Taxonomy of Types, Symptoms, and Root Causes. 2026.
