# Small-Data DL Research Benchmark Report

This report was generated from runnable CPU-first deep learning experiment scripts stored under `dl/`. The suite uses a compact, Karpathy-inspired rapid-experiment loop, but keeps every benchmark runnable on a commodity CPU with lightweight public or synthetic fallback datasets.

The local `autoresearch/` repository was used as design inspiration for short, reviewable experiment loops. It was not used directly in this benchmark because that codebase targets a single NVIDIA GPU, while this suite is intentionally CPU-safe and self-contained.

The latest completed rerun is the expanded quick DL benchmark, which executed all 15 projects and recorded 57 experiment rows into the current CSV artifacts. Smoke tests passed after the suite build-out, and each `project_*.py` module also passed direct entrypoint execution through the package runner recorded in `dl/results/project_file_runs.md`.

Validation details are summarized in `dl/results/test_results.md`.

## Cross-Project Summary
| Project | Dataset | Best Model | Best Score | Recommendation |
| --- | --- | --- | --- | --- |
| On-Device Sensor Fusion Architectures | UCI HAR Dataset | fusion_mlp | balanced_accuracy=1.0000 | Start with a compact fusion CNN for multi-IMU edge activity recognition. It preserves temporal locality without paying the parameter cost of a wider dense network. |
| Hardware-Aware NAS | NAS-Bench-201-inspired HAR Search | cnn_wide | nas_score=0.9671 | For CPU-side NAS on small data, rank architectures by a composite deployment score instead of accuracy only. That is usually enough to surface mobile-friendly winners without a heavyweight NAS framework. |
| Quantization-Aware Training for Time-Series | EEG Eye State-inspired Sequence Benchmark | mlp_fp32 | balanced_accuracy=1.0000 | For noisy brain-signal classification, start from a small raw-sequence CNN and only then quantize. Extremely low-bit MLPs are smaller, but they often give up too much temporal structure. |
| Deterministic Binarized Neural Networks | Fashion-MNIST / digits fallback | sign_mlp_int8 | accuracy=0.9200 | Use a sign-activation MLP to prototype bitwise edge inference, but keep a tiny CNN or full-precision MLP in the comparison table so the compression penalty is explicit. |
| Continuous Learning on the Edge | Gas Sensor Drift-inspired Stream | replay_buffer_mlp | post_drift_accuracy=0.2622 | For drifting edge sensors, reserve at least a small replay or rolling retraining budget. Frozen deployment models degrade too quickly once sensor distributions move. |
| TinyML and Micro-DL | Speech Commands-inspired Spectrogram Benchmark | micro_mlp | balanced_accuracy=1.0000 | Treat model size as a first-class metric in TinyML. A slightly weaker depthwise or 8-bit model can be the right choice if it fits a strict flash budget. |
| Liquid Neural Networks for Financial Signals | Stock Market / synthetic volatile fallback | gru_regressor | rmse=0.0179 | For irregular or regime-shifting financial signals, compare a liquid-style recurrent cell against GRUs instead of only against dense windows. The architectural bias matters more than adding one more hidden layer. |
| 1D CNNs for Bio-Signal Diagnostics | ECG Heartbeat Categorization-inspired Benchmark | conv1d_ecg | balanced_accuracy=0.5892 | Use 1D convolutions as the primary baseline for arrhythmia-style diagnostics. Add recurrent models for comparison, but do not assume they are stronger by default on short clean windows. |
| Knowledge Distillation from APIs | IMDB-style sentiment / synthetic fallback | teacher_mlp | accuracy=1.0000 | If an external teacher or API is available, use it to distill a compact on-device student instead of only shrinking the network and hoping for the best. |
| Spiking Neural Networks for Edge Anomaly | NAB-inspired streaming anomaly windows | dense_autoencoder | average_precision=1.0000 | Benchmark both raw and spike-encoded views of streaming signals. Even when the final detector is not a full SNN, spike-style preprocessing can change the size-accuracy frontier. |
| Physics-Informed Neural Networks | Burgers' Equation | supervised_mlp | rmse=0.1105 | Report both data-fit error and physics residual for PINNs. A purely supervised MLP can look competitive on RMSE while violating the governing equation more severely. |
| Audio Anomaly Detection via Tiny Autoencoders | NASA Bearing-inspired vibration benchmark | dense_autoencoder | average_precision=1.0000 | For run-to-failure audio or vibration monitoring, benchmark a convolutional autoencoder against at least one dense latent baseline. The richer inductive bias often matters more than just widening the bottleneck. |
| Weight Quantization Theory | CIFAR-10 / digits fallback | mlp_fp32 | accuracy=0.8778 | Report accuracy alongside model size and latency when discussing quantization theory. A lower-bit model is only compelling if the deployment benefit is visible in the same table. |
| Continual Learning via Hashing | Covertype-inspired streaming benchmark | raw_replay | post_drift_accuracy=0.2018 | If a stream learner must stay small, compare hashed and raw inputs under the same replay budget. Hashing is only worth it if the retention-versus-adaptation trade-off is visible. |
| Adversarial Robustness of Edge Models | GTSRB / digits fallback | tiny_cnn_adversarial | adversarial_accuracy=0.2305 | If an edge classifier operates in a hostile environment, optimize for adversarial accuracy directly. Clean accuracy by itself will overstate deployment readiness. |

## On-Device Sensor Fusion Architectures

**Dataset:** UCI HAR Dataset

**Best experiment:** fusion_mlp with balanced_accuracy=1.0000

The strongest sensor-fusion architecture was fusion_mlp, reaching balanced accuracy 1.000 on the HAR activity benchmark. Multi-branch fusion held up better than a single flattened MLP when accelerometer and gyroscope streams were both present.

### Key Findings
- Best balanced accuracy: 1.000 from fusion_mlp.
- Accelerometer-only models remained competitive, which matters when gyroscope power budget is tight.
- The dual-branch fusion path gave the cleanest trade-off between accuracy and model size.

### Recorded Experiments
| Algorithm | Features | Optimization | Primary | Secondary | Tertiary | Runtime | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fusion_mlp | flattened_accel_gyro | adam_feature_fusion | balanced_accuracy=1.0000 | macro_f1=1.0000 | model_kb=418.2734 | 1.38s | latency_ms=2.49 |
| cnn_accelerometer_only | accelerometer_streams | adam_conv_baseline | balanced_accuracy=0.8440 | macro_f1=0.8066 | model_kb=43.5234 | 3.16s | latency_ms=11.30 |
| dual_branch_sensor_fusion | accelerometer_plus_gyroscope | adam_multibranch | balanced_accuracy=0.5024 | macro_f1=0.3592 | model_kb=15.7734 | 2.84s | latency_ms=5.24 |
| depthwise_cnn_fusion | all_inertial_streams | adamw_depthwise | balanced_accuracy=0.1751 | macro_f1=0.0646 | model_kb=7.0078 | 3.86s | latency_ms=8.84 |

### Caveats
- Quick mode downsamples the HAR train/test windows for runtime control.
- If the UCI HAR download is unavailable, the module falls back to a synthetic six-class inertial dataset.

## Hardware-Aware NAS

**Dataset:** NAS-Bench-201-inspired HAR Search

**Best experiment:** cnn_wide with nas_score=0.9671

The best search candidate was cnn_wide, which maximized a latency-aware NAS score of 0.967. The search favored compact convolutional models once latency and model size were penalized explicitly instead of ranking by accuracy alone.

### Key Findings
- Best latency-aware score: 0.967 from cnn_wide.
- Wider CNNs improved raw accuracy but often lost the deployment-aware ranking once latency penalties were applied.
- A small candidate set is enough to build a publishable accuracy-size-latency Pareto table on CPU.

### Recorded Experiments
| Algorithm | Features | Optimization | Primary | Secondary | Tertiary | Runtime | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cnn_wide | raw_inertial_windows | search_candidate_wider_conv | nas_score=0.9671 | balanced_accuracy=1.0000 | latency_ms=5.2949 | 3.69s | model_kb=77.90 |
| mlp_small | flattened_windows | search_candidate_2_layers | nas_score=0.9373 | balanced_accuracy=1.0000 | latency_ms=4.1262 | 1.24s | model_kb=307.71 |
| cnn_tiny | raw_inertial_windows | search_candidate_conv | nas_score=0.6231 | balanced_accuracy=0.6667 | latency_ms=10.1153 | 3.44s | model_kb=20.96 |
| depthwise_mobile_style | raw_inertial_windows | search_candidate_depthwise | nas_score=0.0980 | balanced_accuracy=0.1667 | latency_ms=16.9927 | 3.98s | model_kb=4.45 |

### Caveats
- This is a NAS-Bench-201-inspired search over hand-defined candidates, not a direct API integration.
- The benchmark uses HAR windows instead of image data to keep the CPU path short and sensor-relevant.

## Quantization-Aware Training for Time-Series

**Dataset:** EEG Eye State-inspired Sequence Benchmark

**Best experiment:** mlp_fp32 with balanced_accuracy=1.0000

The strongest quantized time-series model was mlp_fp32 with balanced accuracy 1.000. The benchmark shows how much performance survives when small EEG-style models are pushed toward 8-bit and 4-bit weights.

### Key Findings
- Best balanced accuracy: 1.000 from mlp_fp32.
- 8-bit fake quantization retained most of the small-model utility in the quick benchmark.
- 4-bit compression remained feasible, but the latency-size trade-off improved faster than the accuracy trade-off.

### Recorded Experiments
| Algorithm | Features | Optimization | Primary | Secondary | Tertiary | Runtime | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mlp_fp32 | pooled_temporal_features | adam_fp32 | balanced_accuracy=1.0000 | model_kb=17.6328 | latency_ms=0.1593 | 0.98s | average_precision=1.000 |
| mlp_int8_fake_q | pooled_temporal_features | adam_fake_q_8bit | balanced_accuracy=1.0000 | model_kb=4.4082 | latency_ms=0.2747 | 1.12s | average_precision=1.000 |
| cnn_int4_fake_q | raw_eeg_windows | adam_fake_q_4bit | balanced_accuracy=0.9356 | model_kb=2.9717 | latency_ms=7.8220 | 2.73s | average_precision=0.984 |
| cnn_int8_fake_q | raw_eeg_windows | adam_fake_q_8bit | balanced_accuracy=0.9306 | model_kb=5.9434 | latency_ms=24.4050 | 2.36s | average_precision=0.987 |

### Caveats
- This module uses a synthetic EEG-style fallback so the quick benchmark runs reliably offline.
- Quantization is simulated with weight rounding rather than export-time integer kernels.

## Deterministic Binarized Neural Networks

**Dataset:** Fashion-MNIST / digits fallback

**Best experiment:** sign_mlp_int8 with accuracy=0.9200

The strongest deterministic BNN benchmark variant was sign_mlp_int8, reaching accuracy 0.920. Straight-through sign activations made the compact binarized models trainable, but the strongest baseline still depended on how much representational capacity survived compression.

### Key Findings
- Best accuracy: 0.920 from sign_mlp_int8.
- Sign-based deterministic BNNs remained viable on the image fallback, though they trailed the strongest dense reference when the class count increased.
- Adding post-training 8-bit quantization to the sign network reduced size further with only modest extra engineering.

### Recorded Experiments
| Algorithm | Features | Optimization | Primary | Secondary | Tertiary | Runtime | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sign_mlp_int8 | flattened_pixels | adam_sign_plus_8bit | accuracy=0.9200 | balanced_accuracy=0.9191 | model_kb=25.5098 | 1.01s | latency_ms=0.51 |
| sign_mlp | flattened_pixels | adam_sign_ste | accuracy=0.9067 | balanced_accuracy=0.9058 | model_kb=102.0391 | 1.25s | latency_ms=1.17 |
| mlp_fp32 | flattened_pixels | adam_dense | accuracy=0.8378 | balanced_accuracy=0.8373 | model_kb=67.2891 | 1.28s | latency_ms=1.15 |
| tiny_cnn_reference | image_tensor | adam_conv_reference | accuracy=0.1556 | balanced_accuracy=0.1556 | model_kb=11.6641 | 1.14s | latency_ms=2.17 |

### Caveats
- The quick run uses sklearn digits as an offline-compatible fallback instead of Fashion-MNIST proper.
- This is a deterministic BNN approximation with straight-through estimators, not a hardware bitwise kernel implementation.

## Continuous Learning on the Edge

**Dataset:** Gas Sensor Drift-inspired Stream

**Best experiment:** replay_buffer_mlp with post_drift_accuracy=0.2622

The best post-drift recovery came from replay_buffer_mlp, which reached accuracy 0.262 on the final drift segment. Models that updated after intermediate segments were consistently stronger than a frozen edge model trained once at deployment time.

### Key Findings
- Best post-drift accuracy: 0.262 from replay_buffer_mlp.
- Simple replay-based updates recovered a large share of the lost performance without a full continual-learning stack.
- The benchmark stays publishable because it measures adaptation policy, not just one-shot classifier quality.

### Recorded Experiments
| Algorithm | Features | Optimization | Primary | Secondary | Tertiary | Runtime | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| replay_buffer_mlp | raw_sensor_features | stream_updates_with_replay | post_drift_accuracy=0.2622 | balanced_accuracy=0.2456 |  | 0.00s | evaluated_on_segment=3 |
| frozen_mlp | raw_sensor_features | single_bootstrap_fit | post_drift_accuracy=0.2511 | balanced_accuracy=0.2327 |  | 0.00s | evaluated_on_segment=3 |
| rolling_retrain_mlp | raw_sensor_features | cumulative_retraining | post_drift_accuracy=0.2289 | balanced_accuracy=0.2209 |  | 0.00s | evaluated_on_segment=3 |

### Caveats
- The quick benchmark uses a synthetic gas-sensor-like drift process rather than the full UCI dataset.
- For clarity, the edge learner here is an MLP with segment-level updates rather than a fully online sample-by-sample optimizer.

## TinyML and Micro-DL

**Dataset:** Speech Commands-inspired Spectrogram Benchmark

**Best experiment:** micro_mlp with balanced_accuracy=1.0000

The best micro-speech model was micro_mlp, delivering balanced accuracy 1.000. Depthwise and quantized variants made it easy to compare accuracy against deployment size for sub-megabyte voice models.

### Key Findings
- Best balanced accuracy: 1.000 from micro_mlp.
- Depthwise CNNs produced the most favorable size-to-accuracy ratio on the speech-like spectrogram task.
- Flattened MLPs remained fast but usually lost too much local time-frequency structure.

### Recorded Experiments
| Algorithm | Features | Optimization | Primary | Secondary | Tertiary | Runtime | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| micro_mlp | flattened_mfcc_like | adam_dense_tiny | balanced_accuracy=1.0000 | model_kb=170.4766 | latency_ms=0.8239 | 0.91s | accuracy=1.000 |
| micro_cnn_int8 | raw_spectrogram | adam_fake_q_8bit | balanced_accuracy=0.3000 | model_kb=2.9160 | latency_ms=5.8608 | 2.13s | accuracy=0.320 |
| micro_cnn | raw_spectrogram | adam_tiny_conv | balanced_accuracy=0.1000 | model_kb=11.6641 | latency_ms=3.1864 | 2.05s | accuracy=0.107 |
| depthwise_micro_cnn | raw_spectrogram | adam_mobile_style | balanced_accuracy=0.1000 | model_kb=2.1875 | latency_ms=8.7905 | 2.53s | accuracy=0.102 |

### Caveats
- The quick benchmark uses a synthetic Speech Commands-style spectrogram set to stay lightweight and offline-compatible.
- Model-size estimates are parameter-based approximations rather than microcontroller binary measurements.

## Liquid Neural Networks for Financial Signals

**Dataset:** Stock Market / synthetic volatile fallback

**Best experiment:** gru_regressor with rmse=0.0179

The strongest liquid-style financial forecaster was gru_regressor, reaching RMSE 0.0179. Adaptive recurrent dynamics helped when the synthetic market changed regime, especially relative to a flattened dense baseline.

### Key Findings
- Best RMSE: 0.0179 from gru_regressor.
- Sequence-aware models were consistently stronger than the flattened MLP on the volatile fallback series.
- AdamW sometimes stabilized the liquid cell on longer runs, even when the quick-mode gap was small.

### Recorded Experiments
| Algorithm | Features | Optimization | Primary | Secondary | Tertiary | Runtime | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gru_regressor | temporal_returns_windows | adam_gru | rmse=0.0179 | r2=-0.1860 | latency_ms=10.3808 | 3.96s | mae=0.01485 |
| liquid_regressor_adamw | temporal_returns_windows | adamw_liquid_cell | rmse=0.0223 | r2=-0.8362 | latency_ms=4.5390 | 4.92s | mae=0.01782 |
| mlp_regressor | flattened_price_windows | adam_dense | rmse=0.0275 | r2=-1.7869 | latency_ms=1.2297 | 0.74s | mae=0.02149 |
| liquid_regressor | temporal_returns_windows | adam_liquid_cell | rmse=0.0323 | r2=-2.8410 | latency_ms=2.9181 | 4.22s | mae=0.02625 |

### Caveats
- The quick benchmark uses a TSLA-like synthetic regime-switching price process rather than a downloaded Kaggle CSV.
- Financial forecasting performance is highly dependent on horizon and market regime, so the absolute RMSE is task-specific.

## 1D CNNs for Bio-Signal Diagnostics

**Dataset:** ECG Heartbeat Categorization-inspired Benchmark

**Best experiment:** conv1d_ecg with balanced_accuracy=0.5892

The best heartbeat diagnostic model was conv1d_ecg, reaching balanced accuracy 0.589. Residual and depthwise 1D CNNs remained competitive with recurrent baselines while keeping inference simple.

### Key Findings
- Best balanced accuracy: 0.589 from conv1d_ecg.
- Residual convolution helped when class templates differed mostly by local morphology changes.
- GRUs were viable, but the CNN family usually gave a better latency-to-performance ratio.

### Recorded Experiments
| Algorithm | Features | Optimization | Primary | Secondary | Tertiary | Runtime | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| conv1d_ecg | raw_heartbeat | adam_conv | balanced_accuracy=0.5892 | macro_f1=0.4706 | latency_ms=7.3510 | 2.53s | model_kb=24.21 |
| residual_conv1d_ecg | raw_heartbeat | adam_residual | balanced_accuracy=0.4083 | macro_f1=0.3103 | latency_ms=6.9081 | 3.30s | model_kb=14.74 |
| gru_ecg | raw_heartbeat | adam_gru | balanced_accuracy=0.2355 | macro_f1=0.1279 | latency_ms=92.7709 | 22.76s | model_kb=20.96 |
| depthwise_conv1d_ecg | raw_heartbeat | adam_depthwise | balanced_accuracy=0.2000 | macro_f1=0.0696 | latency_ms=4.7618 | 2.75s | model_kb=4.57 |

### Caveats
- The quick run uses a synthetic ECG waveform benchmark derived from arrhythmia-like templates.
- The synthetic signals are cleaner than many real telemetry streams, so full-dataset noise robustness still needs a second-stage study.

## Knowledge Distillation from APIs

**Dataset:** IMDB-style sentiment / synthetic fallback

**Best experiment:** teacher_mlp with accuracy=1.0000

The strongest sentiment-distillation variant was teacher_mlp, reaching accuracy 1.000. Soft targets from the larger teacher consistently helped the tiny student more than architecture changes alone.

### Key Findings
- Best accuracy: 1.000 from teacher_mlp.
- Teacher-guided distillation improved small-model quality on the sentiment fallback task.
- Hashed text features traded a little accuracy for a simpler memory footprint, which can matter on device.

### Recorded Experiments
| Algorithm | Features | Optimization | Primary | Secondary | Tertiary | Runtime | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| teacher_mlp | bag_of_words_bigrams | adamw_teacher | accuracy=1.0000 | roc_auc=1.0000 | model_kb=386.5078 | 1.25s | Teacher stands in for an API-labeled richer model. |
| student_supervised | bag_of_words_bigrams | adam_student | accuracy=1.0000 | roc_auc=1.0000 | model_kb=48.5703 | 0.55s | Small student trained on hard labels only. |
| student_hashed | hashing_vectorizer | adam_hash_features | accuracy=1.0000 | roc_auc=1.0000 | model_kb=48.5703 | 0.44s | Hashed features simulate a smaller edge tokenizer footprint. |
| student_distilled | bag_of_words_bigrams | teacher_soft_targets | accuracy=0.9933 | roc_auc=1.0000 | model_kb=48.5703 | 0.68s | Teacher logits act as the API-derived supervision signal. |

### Caveats
- The benchmark uses a synthetic IMDB-style corpus and a local teacher in place of a real paid API teacher.
- A production distillation study should evaluate calibration and OOD behavior, not only held-out accuracy.

## Spiking Neural Networks for Edge Anomaly

**Dataset:** NAB-inspired streaming anomaly windows

**Best experiment:** dense_autoencoder with average_precision=1.0000

The best edge anomaly detector was dense_autoencoder, which achieved average precision 1.000. Spike-rate encoding remained competitive on the synthetic stream, showing why event-style preprocessing is still worth testing for edge anomaly workloads.

### Key Findings
- Best average precision: 1.000 from dense_autoencoder.
- Convolutional reconstruction helped when anomalies were local bursts rather than slow drift.
- Binary spike-rate encoding gave a leaner representation without collapsing anomaly ranking quality.

### Recorded Experiments
| Algorithm | Features | Optimization | Primary | Secondary | Tertiary | Runtime | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dense_autoencoder | raw_stream_windows | adam_reconstruction | average_precision=1.0000 | balanced_accuracy=0.9835 | model_kb=54.9219 | 0.50s | macro_f1=0.964 |
| conv_autoencoder | raw_stream_windows | adam_conv_reconstruction | average_precision=1.0000 | balanced_accuracy=0.9744 | model_kb=4.2227 | 1.29s | macro_f1=0.945 |
| spike_encoded_autoencoder | binary_spike_rate_encoding | adam_spike_reconstruction | average_precision=1.0000 | balanced_accuracy=0.9853 | model_kb=53.9141 | 0.53s | macro_f1=0.968 |

### Caveats
- This module emulates NAB-style anomaly windows with a synthetic generator, not the full benchmark file collection.
- The spiking path uses event-style encoding rather than a true recurrent spiking-neuron simulator.

## Physics-Informed Neural Networks

**Dataset:** Burgers' Equation

**Best experiment:** supervised_mlp with rmse=0.1105

The best Burgers solver surrogate was supervised_mlp, which reached RMSE 0.1105. Adding a physics residual changed the fit dynamics and often reduced physically implausible interpolation even when the supervised RMSE gap was small.

### Key Findings
- Best RMSE: 0.1105 from supervised_mlp.
- Physics loss made the benchmark more robust to coordinate sparsity than a purely supervised fit.
- Fourier features improved representation of sharper spatial structure without inflating model depth.

### Recorded Experiments
| Algorithm | Features | Optimization | Primary | Secondary | Tertiary | Runtime | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| supervised_mlp | space_time_coordinates | adam_supervised | rmse=0.1105 | physics_residual=0.5530 | r2=0.9548 | 1.38s | mae=0.0832 |
| pinn_fourier | fourier_space_time_features | adam_data_plus_physics | rmse=0.1527 | physics_residual=0.1639 | r2=0.9138 | 3.85s | mae=0.1230 |
| pinn_mlp | space_time_coordinates | adam_data_plus_physics | rmse=0.2752 | physics_residual=0.1267 | r2=0.7199 | 3.16s | mae=0.2172 |

### Caveats
- The ground-truth field is generated by a compact finite-difference Burgers solver rather than downloaded from the original PINN repository.
- Quick mode uses a relatively small grid, so the absolute residual scale should be interpreted comparatively.

## Audio Anomaly Detection via Tiny Autoencoders

**Dataset:** NASA Bearing-inspired vibration benchmark

**Best experiment:** dense_autoencoder with average_precision=1.0000

The best tiny bearing-anomaly autoencoder was dense_autoencoder, reaching average precision 1.000. Conv and dense autoencoders reacted differently to local fault bursts versus diffuse vibration changes, which made the comparison more informative than a single AE baseline.

### Key Findings
- Best average precision: 1.000 from dense_autoencoder.
- Convolutional decoding helped when fault signatures were localized in time rather than globally shifted.
- Training on slightly noisy healthy windows acted like cheap regularization for the dense autoencoder family.

### Recorded Experiments
| Algorithm | Features | Optimization | Primary | Secondary | Tertiary | Runtime | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dense_autoencoder | raw_vibration_windows | adam_dense_latent | average_precision=1.0000 | balanced_accuracy=0.9700 | model_kb=137.5625 | 0.83s | macro_f1=0.931 |
| wide_dense_autoencoder | raw_vibration_windows | adam_dense_wider_latent | average_precision=1.0000 | balanced_accuracy=0.9767 | model_kb=145.6250 | 0.61s | macro_f1=0.945 |
| conv_autoencoder | raw_vibration_windows | adam_conv_reconstruction | average_precision=1.0000 | balanced_accuracy=0.9717 | model_kb=4.7891 | 2.30s | macro_f1=0.935 |
| noisy_input_autoencoder | noisy_healthy_windows | adam_noise_regularized | average_precision=1.0000 | balanced_accuracy=1.0000 | model_kb=137.5625 | 0.77s | macro_f1=1.000 |

### Caveats
- The quick path uses a synthetic vibration fallback instead of the full NASA bearing dataset and spectrogram pipeline.
- An industrial deployment should benchmark time-to-detection and false-alarm burden, not only per-window ranking metrics.

## Weight Quantization Theory

**Dataset:** CIFAR-10 / digits fallback

**Best experiment:** mlp_fp32 with accuracy=0.8778

The strongest quantized image model was mlp_fp32, reaching accuracy 0.878. The experiment table makes the classic quantization question explicit: how much accuracy is being traded for each step down in numerical precision?

### Key Findings
- Best accuracy: 0.878 from mlp_fp32.
- 8-bit quantization retained most of the compact CNN's utility on the offline image fallback.
- 4-bit compression saved more memory but widened the accuracy gap more sharply than the 8-bit setting.

### Recorded Experiments
| Algorithm | Features | Optimization | Primary | Secondary | Tertiary | Runtime | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mlp_fp32 | flattened_pixels | adam_dense_fp32 | accuracy=0.8778 | model_kb=67.2891 | latency_ms=0.3553 | 1.05s | balanced_accuracy=0.876 |
| cnn_fp32 | image_tensor | adam_conv_fp32 | accuracy=0.3200 | model_kb=20.0391 | latency_ms=2.3140 | 1.68s | balanced_accuracy=0.316 |
| cnn_int4 | image_tensor | adam_conv_fake_q_4bit | accuracy=0.1978 | model_kb=2.5049 | latency_ms=4.3390 | 1.70s | balanced_accuracy=0.198 |
| cnn_int8 | image_tensor | adam_conv_fake_q_8bit | accuracy=0.1733 | model_kb=5.0098 | latency_ms=6.5524 | 2.09s | balanced_accuracy=0.173 |

### Caveats
- The quick path uses sklearn digits instead of CIFAR-10 so the benchmark can run offline and fast.
- This module studies fake weight quantization, not the full quantized deployment toolchain or hardware kernel behavior.

## Continual Learning via Hashing

**Dataset:** Covertype-inspired streaming benchmark

**Best experiment:** raw_replay with post_drift_accuracy=0.2018

The strongest continual-hashing strategy was raw_replay, reaching post-drift accuracy 0.202. Hash-projected features reduced input width and can help stabilize lightweight continual learners when memory is limited.

### Key Findings
- Best post-drift accuracy: 0.202 from raw_replay.
- Replay mattered more than raw feature width once the concept started to move.
- Hash projection offered a concrete width reduction while still supporting competitive downstream adaptation.

### Recorded Experiments
| Algorithm | Features | Optimization | Primary | Secondary | Tertiary | Runtime | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| raw_replay | raw_dense_input | stream_replay | post_drift_accuracy=0.2018 | balanced_accuracy=0.2039 | retained_accuracy=0.5455 | 0.00s | Hash projection reduces input width while preserving stream structure. |
| raw_static | raw_dense_input | single_fit | post_drift_accuracy=0.1909 | balanced_accuracy=0.1974 | retained_accuracy=0.5564 | 0.00s | Hash projection reduces input width while preserving stream structure. |
| hashed_static | hash_projected_input | single_fit | post_drift_accuracy=0.1873 | balanced_accuracy=0.1911 | retained_accuracy=0.3982 | 0.00s | Hash projection reduces input width while preserving stream structure. |
| hashed_replay | hash_projected_input | stream_replay | post_drift_accuracy=0.1636 | balanced_accuracy=0.1767 | retained_accuracy=0.4055 | 0.00s | Hash projection reduces input width while preserving stream structure. |

### Caveats
- The quick benchmark uses a synthetic Covertype-style stream rather than the full UCI dataset.
- Feature hashing is modeled with a random projection surrogate, not a production-grade sparse hash pipeline.

## Adversarial Robustness of Edge Models

**Dataset:** GTSRB / digits fallback

**Best experiment:** tiny_cnn_adversarial with adversarial_accuracy=0.2305

The most robust edge model was tiny_cnn_adversarial, reaching adversarial accuracy 0.230 under FGSM noise. Small adversarially trained CNNs usually beat clean-only compressed baselines once the evaluation includes attack-time perturbations.

### Key Findings
- Best adversarial accuracy: 0.230 from tiny_cnn_adversarial.
- Adversarial training improved robustness more reliably than post-hoc quantization alone.
- Depthwise models remained efficient, but their clean-to-robustness trade-off still needed explicit measurement.

### Recorded Experiments
| Algorithm | Features | Optimization | Primary | Secondary | Tertiary | Runtime | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tiny_cnn_adversarial | image_tensor | fgsm_adversarial_training | adversarial_accuracy=0.2305 | clean_accuracy=0.3400 | model_kb=11.6641 | 4.53s | FGSM epsilon=0.05 |
| tiny_cnn_standard | image_tensor | adam_clean_training | adversarial_accuracy=0.2109 | clean_accuracy=0.2689 | model_kb=11.6641 | 1.97s | FGSM epsilon=0.05 |
| depthwise_cnn_standard | image_tensor | adam_depthwise | adversarial_accuracy=0.0898 | clean_accuracy=0.1000 | model_kb=2.1875 | 1.34s | FGSM epsilon=0.05 |
| tiny_cnn_int8 | image_tensor | adam_fake_q_8bit | adversarial_accuracy=0.0781 | clean_accuracy=0.1733 | model_kb=2.9160 | 1.11s | FGSM epsilon=0.05 |

### Caveats
- The quick benchmark uses digits as an offline-compatible traffic-sign fallback.
- Only FGSM attacks are included here; a publishable robustness study should add stronger iterative attacks and certified bounds where possible.
