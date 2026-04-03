# Small-Data GenAI Research Benchmark Report

This report was generated from runnable lightweight experiment scripts under `ml/`. Each project benchmarks multiple algorithms, feature variants, or optimization strategies under CPU-first constraints.

## Cross-Project Summary

| Project | Dataset | Best Variant | Best Score | Metric |
| --- | --- | --- | --- | --- |
| Stateful Multi-Agent Debugging Systems | synthetic_cpp_debugging_corpus | stateful_agent_logreg | 1.0000 | macro_f1 |
| Constrained Generation via Logit Masking | synthetic_verilog_eval_proxy | grammar_masked | 1.0000 | validity |
| Edge-Optimized RAG with Custom Vector Implementations | squad_v2 | tfidf_bigrams | 0.2466 | recall_at_1 |
| Agentic Frameworks for Hardware Synthesis | synthetic_verilog_module_library | planner_query_expansion | 1.0000 | recall_at_1 |
| Small Model Orchestration for CI/CD | synthetic_commitpack_proxy | message_plus_diff_nb | 0.6946 | macro_f1 |
| Mechanistic Interpretability on Toy Models | roneneldan/TinyStories | transformer | 2.2897 | val_nll |
| RAG Chunking Optimization | synthetic_scifact_proxy | fixed_tfidf | 0.1111 | recall_at_1 |
| Small AI Agent Workflows | hotpot_qa | single_pass_scan | 0.0700 | overlap_f1 |
| QLoRA Fine-Tuning of Sub-Billion Models (Legal Proxy) | synthetic_cuad_proxy | clause_only_logreg | 1.0000 | macro_f1 |
| Black-Box Prompt Optimization | openai/gsm8k_filtered_simple | random_search | 0.0000 | eval_accuracy |
| Hallucination Detection via Cross-Encoder Proxy | synthetic_faithdial_proxy | response_only_logreg | 1.0000 | macro_f1 |
| Context Window Compression | synthetic_govreport_proxy | head_tail | 0.2549 | rouge_l |
| Direct Preference Optimization on Small Language Models (Proxy) | synthetic_dpo_proxy | pairwise_logreg | 1.0000 | pairwise_accuracy |
| Synthetic Data Generation for Niche Classifiers | synthetic_symptom2disease_proxy | seed_only_logreg | 0.9429 | macro_f1 |
| Structural Bias Evaluation in Small Language Models | synthetic_stereoset_proxy | biased_prior | 0.3750 | icat |

## Stateful Multi-Agent Debugging Systems

Dataset requested: msc-smart-contract/cpp-vulnerabilities or The Stack Smol C++ subset
Dataset used: synthetic_cpp_debugging_corpus
Objective: Benchmark whether persistent debugging memory improves lightweight bug-triage quality on small C++ code corpora.
Best experiment: stateful_agent_logreg with macro_f1=1.0000

### Findings
- The best overall variant was stateful_agent_logreg with macro F1 1.000.
- Adding stateful debugging memory improved macro F1 from 0.401 to 1.000.
- The synthetic proxy is small and highly structured, so some state benefits are likely compressed relative to a noisier real bug-fix corpus.

### Recorded Experiments

| Variant | Algorithm | Features | Primary | Secondary | Runtime(s) | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| single_agent_logreg | logreg | current_code_only | macro_f1=0.4010 | accuracy=0.4062 | 0.01 | Word n-grams over C++ snippets with or without persistent agent memory. |
| single_agent_svm | linear_svm | current_code_only | macro_f1=0.4010 | accuracy=0.4062 | 0.01 | Word n-grams over C++ snippets with or without persistent agent memory. |
| stateful_agent_logreg | logreg | code_plus_bug_memory | macro_f1=1.0000 | accuracy=1.0000 | 0.02 | Word n-grams over C++ snippets with or without persistent agent memory. |
| stateful_agent_svm | linear_svm | code_plus_bug_memory | macro_f1=1.0000 | accuracy=1.0000 | 0.01 | Word n-grams over C++ snippets with or without persistent agent memory. |
| stateful_agent_nb | nb | code_plus_bug_memory | macro_f1=1.0000 | accuracy=1.0000 | 0.01 | Word n-grams over C++ snippets with or without persistent agent memory. |

### Notes

The real C++ vulnerability dataset is used when available. Otherwise a synthetic memory-safety corpus acts as a CPU-first proxy for multi-agent debugging state.

## Constrained Generation via Logit Masking

Dataset requested: GaTech-EIC/Verilog-eval
Dataset used: synthetic_verilog_eval_proxy
Objective: Measure how grammar constraints and pattern libraries improve Verilog validity under lightweight generation proxies.
Best experiment: grammar_masked with validity=1.0000

### Findings
- The highest-validity decoder was grammar_masked at 1.000.
- Grammar-constrained decoding removes the invalid module skeleton failures seen in the unconstrained template baseline.
- Pattern-library retrieval helps token overlap because the emitted code respects both syntax and common hardware idioms.

### Recorded Experiments

| Variant | Algorithm | Features | Primary | Secondary | Runtime(s) | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| unconstrained_template | free_form_template | spec_only | validity=0.8000 | token_overlap=0.7867 | 0.00 | Proxy benchmark over lightweight Verilog generation templates. |
| grammar_masked | logit_masked_decoder | grammar_constrained | validity=1.0000 | token_overlap=0.9236 | 0.00 | Proxy benchmark over lightweight Verilog generation templates. |
| masked_pattern_retrieval | logit_masked_decoder | grammar_plus_pattern_library | validity=1.0000 | token_overlap=1.0000 | 0.00 | Proxy benchmark over lightweight Verilog generation templates. |

### Notes

This module uses a synthetic Verilog-eval proxy so the benchmark remains deterministic and runnable without an LLM decoder.

## Edge-Optimized RAG with Custom Vector Implementations

Dataset requested: squad_v2
Dataset used: squad_v2
Objective: Compare lightweight retrieval algorithms for question answering without external vector databases.
Best experiment: tfidf_bigrams with recall_at_1=0.2466

### Findings
- The strongest retrieval stack was tfidf_bigrams with recall@1 0.247.
- Sparse lexical methods are attractive on edge devices because they avoid ANN servers and still recover relevant contexts on small corpora.
- Hashing vectors trade a bit of retrieval quality for predictable memory use and fixed-width indexing.

### Recorded Experiments

| Variant | Algorithm | Features | Primary | Secondary | Runtime(s) | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| tfidf_bigrams | tfidf | word_bigrams | recall_at_1=0.2466 | mrr=0.4292 | 0.02 | Ranks the gold answer context against the rest of the corpus. |
| tfidf_latent_svd | tfidf | latent_semantic_projection | recall_at_1=0.2466 | mrr=0.4292 | 0.43 | Ranks the gold answer context against the rest of the corpus. |
| bm25_token_search | bm25 | bm25_sparse_index | recall_at_1=0.2329 | mrr=0.4041 | 0.15 | Ranks the gold answer context against the rest of the corpus. |
| hashing_vector_index | hashing | fixed_width_sparse_vectors | recall_at_1=0.1781 | mrr=0.2694 | 0.01 | Ranks the gold answer context against the rest of the corpus. |

### Notes

The benchmark uses SQuAD contexts when available and otherwise falls back to a compact synthetic QA corpus. The task is retrieval-only rather than end-to-end generation.

## Agentic Frameworks for Hardware Synthesis

Dataset requested: shailja/Verilog_GitHub
Dataset used: synthetic_verilog_module_library
Objective: Test whether simple planning-style query expansion helps agents find the right hardware building blocks before code synthesis.
Best experiment: planner_query_expansion with recall_at_1=1.0000

### Findings
- The highest module-retrieval accuracy came from planner_query_expansion with recall@1 1.000.
- Query expansion behaves like a lightweight planning layer by injecting hardware-specific tokens before retrieval.
- This is a retrieval-stage proxy for agentic synthesis, not full HDL generation or simulation-driven verification.

### Recorded Experiments

| Variant | Algorithm | Features | Primary | Secondary | Runtime(s) | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| tfidf_spec_only | tfidf | single_step_retrieval | recall_at_1=0.0000 | mrr=0.0000 | 0.00 | Retrieval proxy for agent planning over hardware module libraries. |
| bm25_spec_only | bm25 | single_step_retrieval | recall_at_1=0.0000 | mrr=0.0000 | 0.00 | Retrieval proxy for agent planning over hardware module libraries. |
| planner_query_expansion | tfidf | agentic_query_expansion | recall_at_1=1.0000 | mrr=1.0000 | 0.00 | Retrieval proxy for agent planning over hardware module libraries. |
| hashing_retrieval | hashing | fixed_width_retrieval | recall_at_1=0.0000 | mrr=0.0000 | 0.00 | Retrieval proxy for agent planning over hardware module libraries. |

### Notes

A compact synthetic Verilog library is used here because the focus is the agentic retrieval step that precedes hardware synthesis.

## Small Model Orchestration for CI/CD

Dataset requested: github-commit-messages-with-bug-fixes or CommitPackFT
Dataset used: synthetic_commitpack_proxy
Objective: Estimate whether small text models can triage risky code changes before expensive CI/CD jobs execute.
Best experiment: message_plus_diff_nb with macro_f1=0.6946

### Findings
- The best orchestration stack was message_plus_diff_nb with macro F1 0.695.
- Combining commit messages with diffs improved macro F1 from 0.547 to 0.695.
- A more realistic commit corpus is likely to widen the gap between message-only and diff-aware triage, because real CI failures often hide in build scripts and deleted tests.

### Recorded Experiments

| Variant | Algorithm | Features | Primary | Secondary | Runtime(s) | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| message_only_logreg | logreg | commit_message_only | macro_f1=0.5466 | accuracy=0.5625 | 0.01 | Text triage proxy for build-break risk prediction from commit metadata. |
| diff_only_logreg | logreg | code_diff_only | macro_f1=0.6113 | accuracy=0.6250 | 0.01 | Text triage proxy for build-break risk prediction from commit metadata. |
| message_plus_diff_logreg | logreg | message_plus_diff | macro_f1=0.6390 | accuracy=0.6562 | 0.02 | Text triage proxy for build-break risk prediction from commit metadata. |
| message_plus_diff_svm | linear_svm | message_plus_diff | macro_f1=0.6113 | accuracy=0.6250 | 0.01 | Text triage proxy for build-break risk prediction from commit metadata. |
| message_plus_diff_nb | nb | message_plus_diff | macro_f1=0.6946 | accuracy=0.7188 | 0.01 | Text triage proxy for build-break risk prediction from commit metadata. |

### Notes

The workspace does not include a commit-diff corpus, so the benchmark uses a synthetic proxy built around CI failure motifs such as deleted tests, build-file edits, and deployment-script drift.

## Mechanistic Interpretability on Toy Models

Dataset requested: roneneldan/TinyStories
Dataset used: roneneldan/TinyStories
Objective: Train tiny language models on a child-language corpus slice and compare simple interpretability-friendly architectures.
Best experiment: transformer with val_nll=2.2897

### Findings
- The lowest validation NLL came from transformer at 2.290.
- The bigram baseline is easy to inspect but usually leaves large predictive gaps that even tiny neural models close quickly.
- Activation sparsity is logged as a simple interpretability proxy to compare how concentrated each model's internal representations become.

### Recorded Experiments

| Variant | Algorithm | Features | Primary | Secondary | Runtime(s) | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| bigram | bigram | character_level_language_model | val_nll=2.3895 | activation_sparsity=0.0000 | 0.06 | parameter_count=4225 |
| mlp | mlp | character_level_language_model | val_nll=3.1003 | activation_sparsity=0.0930 | 15.82 | parameter_count=63257 |
| transformer | transformer | character_level_language_model | val_nll=2.2897 | activation_sparsity=0.0006 | 27.11 | parameter_count=26417 |

### Notes

This is a true local training benchmark rather than a synthetic proxy. The models are intentionally tiny so they can be trained and inspected on a laptop-scale setup.

## RAG Chunking Optimization

Dataset requested: mteb/scifact
Dataset used: synthetic_scifact_proxy
Objective: Compare chunking strategies for evidence retrieval on short scientific documents under a lightweight RAG setup.
Best experiment: fixed_tfidf with recall_at_1=0.1111

### Findings
- The best chunking strategy was fixed_tfidf with recall@1 0.111.
- All chunking strategies tied on the current synthetic abstract corpus, which suggests the proxy is too small to expose a strong chunking effect.
- A larger or noisier scientific corpus would be a better next step for separating fixed, overlap, sentence, and semantic chunking policies.

### Recorded Experiments

| Variant | Algorithm | Features | Primary | Secondary | Runtime(s) | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| fixed_tfidf | tfidf | fixed | recall_at_1=0.1111 | mrr=0.2037 | 0.16 | Ranks scientific abstracts after chunking them into retrieval units. |
| overlap_tfidf | tfidf | overlap | recall_at_1=0.1111 | mrr=0.2037 | 0.29 | Ranks scientific abstracts after chunking them into retrieval units. |
| sentence_tfidf | tfidf | sentence | recall_at_1=0.1111 | mrr=0.2037 | 0.36 | Ranks scientific abstracts after chunking them into retrieval units. |
| semantic_focus_tfidf | tfidf | semantic_focus | recall_at_1=0.1111 | mrr=0.2037 | 0.61 | Ranks scientific abstracts after chunking them into retrieval units. |
| overlap_bm25 | bm25 | overlap | recall_at_1=0.1111 | mrr=0.2037 | 0.08 | Ranks scientific abstracts after chunking them into retrieval units. |

### Notes

The benchmark uses a synthetic SciFact-style corpus because the main experimental variable is chunking behavior rather than dataset scale.

## Small AI Agent Workflows

Dataset requested: hotpot_qa (distractor)
Dataset used: hotpot_qa
Objective: Compare simple multi-step agent loops for multi-hop question answering under a retrieval-and-selection proxy.
Best experiment: single_pass_scan with overlap_f1=0.0700

### Findings
- The best agent workflow was single_pass_scan with overlap F1 0.070.
- The iterative workflow approximates planner-retriever loops by using the first retrieval result to reformulate the second search pass.
- Even with sentence-level heuristics, multi-step reasoning benefits from maintaining an intermediate state rather than scanning all contexts once.

### Recorded Experiments

| Variant | Algorithm | Features | Primary | Secondary | Runtime(s) | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| single_pass_scan | sentence_selector | single_agent | overlap_f1=0.0700 | exact_contains=0.1146 | 0.34 | Answer extraction proxy over supporting contexts. |
| iterative_two_hop | sentence_selector | planner_retriever_loop | overlap_f1=0.0700 | exact_contains=0.1146 | 0.76 | Answer extraction proxy over supporting contexts. |
| multi_query_vote | sentence_selector | multi_query_ensemble | overlap_f1=0.0402 | exact_contains=0.1146 | 0.33 | Answer extraction proxy over supporting contexts. |

### Notes

This benchmark uses HotpotQA when accessible and otherwise falls back to a synthetic multi-hop corpus. It does not call an external LLM; it isolates the workflow logic itself.

## QLoRA Fine-Tuning of Sub-Billion Models (Legal Proxy)

Dataset requested: cuad
Dataset used: synthetic_cuad_proxy
Objective: Measure how well lightweight text models adapt to a narrow legal clause taxonomy under a small-data proxy for QLoRA-style specialization.
Best experiment: clause_only_logreg with macro_f1=1.0000

### Findings
- The best legal adaptation proxy was clause_only_logreg with macro F1 1.000.
- Raw-clause and instruction-wrapped variants tied at macro F1 1.000 on the current proxy dataset.
- Specialized legal language is structured enough that even compact models can separate clause types on narrow corpora.

### Recorded Experiments

| Variant | Algorithm | Features | Primary | Secondary | Runtime(s) | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| clause_only_logreg | logreg | raw_clause | macro_f1=1.0000 | accuracy=1.0000 | 0.19 | Clause-type classification proxy for legal adaptation. |
| clause_only_svm | linear_svm | raw_clause | macro_f1=1.0000 | accuracy=1.0000 | 0.03 | Clause-type classification proxy for legal adaptation. |
| qa_style_logreg | logreg | instruction_wrapped_clause | macro_f1=1.0000 | accuracy=1.0000 | 0.04 | Clause-type classification proxy for legal adaptation. |
| qa_style_nb | nb | instruction_wrapped_clause | macro_f1=0.2800 | accuracy=0.4000 | 0.02 | Clause-type classification proxy for legal adaptation. |

### Notes

This is a clause-classification proxy rather than true parameter-efficient fine-tuning. It is designed to keep the benchmark runnable without downloading or adapting a large legal language model.

## Black-Box Prompt Optimization

Dataset requested: openai/gsm8k
Dataset used: openai/gsm8k_filtered_simple
Objective: Optimize discrete prompt templates against a black-box arithmetic solver on small GSM8K-style problems.
Best experiment: random_search with eval_accuracy=0.0000

### Findings
- The best prompt-search policy was random_search with evaluation accuracy 0.000.
- This is a true black-box search loop over prompt components: the optimizer only sees scored outcomes from a weak math solver and never gradient information.
- Random search, hill climbing, and the genetic search tied on the filtered problem slice, so the solver is currently the main bottleneck rather than the optimizer.

### Recorded Experiments

| Variant | Algorithm | Features | Primary | Secondary | Runtime(s) | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| random_search | black_box_search | prompt_component_subset | eval_accuracy=0.0000 | train_accuracy=0.0357 | 0.16 | When the question says each, every, per, or times, try multiplication first. |
| hill_climb | black_box_search | prompt_component_subset | eval_accuracy=0.0000 | train_accuracy=0.0357 | 0.09 | Look for total or altogether cues. When the question says each, every, per, or times, try multiplication first. |
| genetic_search | black_box_search | prompt_component_subset | eval_accuracy=0.0000 | train_accuracy=0.0357 | 1.34 | Look for total or altogether cues. When the question says left, remain, spent, or after giving, try subtraction. When the question says each, every, per, or times, try multiplication first. When the question says equally or share, try division. |

### Notes

This benchmark is an API-free surrogate for prompt optimization. It keeps the search algorithm real while replacing the external language model with a deterministic weak solver.

## Hallucination Detection via Cross-Encoder Proxy

Dataset requested: McGill-NLP/FaithDial or vectara/huggescript
Dataset used: synthetic_faithdial_proxy
Objective: Detect grounded versus hallucinated responses using lightweight pairwise text models.
Best experiment: response_only_logreg with macro_f1=1.0000

### Findings
- The strongest hallucination detector was response_only_logreg with macro F1 1.000.
- Response-only and source-plus-response models tied at macro F1 1.000 on this easy synthetic grounding task.
- This is a lightweight cross-encoder proxy implemented with sparse lexical models rather than a transformer pair scorer.

### Recorded Experiments

| Variant | Algorithm | Features | Primary | Secondary | Runtime(s) | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| response_only_logreg | logreg | response_only | macro_f1=1.0000 | accuracy=1.0000 | 0.10 | Grounding-classification proxy over source/response pairs. |
| pair_logreg | logreg | source_plus_response | macro_f1=1.0000 | accuracy=1.0000 | 0.06 | Grounding-classification proxy over source/response pairs. |
| pair_svm | linear_svm | source_plus_response | macro_f1=1.0000 | accuracy=1.0000 | 0.04 | Grounding-classification proxy over source/response pairs. |
| pair_nb | nb | source_plus_response | macro_f1=0.7662 | accuracy=0.7778 | 0.04 | Grounding-classification proxy over source/response pairs. |

### Notes

A synthetic grounded-dialogue corpus is used here so the task remains fully runnable without a large cross-encoder model or a gated dialogue dataset.

## Context Window Compression

Dataset requested: tau/scrolls (GovReport subset)
Dataset used: synthetic_govreport_proxy
Objective: Compare simple compression policies for long reports under a summary-retention proxy.
Best experiment: head_tail with rouge_l=0.2549

### Findings
- The best compression strategy was head_tail with ROUGE-L 0.255.
- Compression is scored directly against summaries rather than through a second summarization model, so the benchmark isolates how much salient information each compression strategy preserves.
- Lead-only truncation is strong on report-style prose, but head-tail or central-sentence strategies can recover late-document conclusions that lead bias discards.

### Recorded Experiments

| Variant | Algorithm | Features | Primary | Secondary | Runtime(s) | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| lead_only | extractive_compression | lead_bias | rouge_l=0.2163 | retained_fraction=0.7012 | 0.10 | Compression quality measured against reference summaries using ROUGE-L. |
| head_tail | extractive_compression | head_tail_mix | rouge_l=0.2549 | retained_fraction=0.6896 | 0.10 | Compression quality measured against reference summaries using ROUGE-L. |
| middle_focus | extractive_compression | middle_retention | rouge_l=0.2032 | retained_fraction=0.7302 | 0.10 | Compression quality measured against reference summaries using ROUGE-L. |
| longest_sentences | extractive_compression | length_centrality | rouge_l=0.1855 | retained_fraction=0.7917 | 0.11 | Compression quality measured against reference summaries using ROUGE-L. |

### Notes

This module uses a synthetic GovReport-style corpus and evaluates how well compressed contexts retain summary-worthy content.

## Direct Preference Optimization on Small Language Models (Proxy)

Dataset requested: argilla/dpo-mix-7k
Dataset used: synthetic_dpo_proxy
Objective: Learn preference scores from chosen/rejected response pairs using a CPU-friendly proxy for the reward-model stage of DPO.
Best experiment: pairwise_logreg with pairwise_accuracy=1.0000

### Findings
- The best preference learner was pairwise_logreg with pairwise accuracy 1.000.
- The learned preference model improved pairwise accuracy from 0.333 to 1.000.
- The learned pairwise scorers are a lightweight stand-in for a reward model trained before a true DPO update.

### Recorded Experiments

| Variant | Algorithm | Features | Primary | Secondary | Runtime(s) | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| heuristic_reward | overlap_reward | prompt_response_lexical_overlap | pairwise_accuracy=0.3333 | mean_margin=-0.1345 | 0.00 | No-training lexical reward baseline. |
| pairwise_logreg | logreg | prompt_plus_response_pair | pairwise_accuracy=1.0000 | mean_margin=0.6118 | 0.05 | Binary classifier trained on chosen-versus-rejected pairs. |
| pairwise_linear_svm | linear_svm | prompt_plus_response_pair | pairwise_accuracy=1.0000 | mean_margin=0.6792 | 0.03 | Binary classifier trained on chosen-versus-rejected pairs. |
| pairwise_nb | nb | prompt_plus_response_pair | pairwise_accuracy=1.0000 | mean_margin=0.2949 | 0.02 | Binary classifier trained on chosen-versus-rejected pairs. |

### Notes

This module does not fine-tune a generator; it benchmarks the smaller and cheaper preference-modeling stage that often precedes DPO or related preference-optimization methods.

## Synthetic Data Generation for Niche Classifiers

Dataset requested: Symptom2Disease
Dataset used: synthetic_symptom2disease_proxy
Objective: Measure how simple synthetic symptom paraphrases change disease-classification quality under tiny-data conditions.
Best experiment: seed_only_logreg with macro_f1=0.9429

### Findings
- The best augmentation strategy was seed_only_logreg with macro F1 0.943.
- Seed-only and augmented variants tied at macro F1 0.943 on the current synthetic dataset.
- The mixed augmentation regime tests whether stylistic diversity and synonym substitution complement each other better than either alone.

### Recorded Experiments

| Variant | Algorithm | Features | Primary | Secondary | Runtime(s) | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| seed_only_logreg | logreg | seed_only | macro_f1=0.9429 | accuracy=0.9444 | 0.04 | Evaluated on held-out paraphrases rather than duplicates of the seed list. |
| seed_plus_synonyms_logreg | logreg | seed_plus_synonyms | macro_f1=0.9429 | accuracy=0.9444 | 0.06 | Evaluated on held-out paraphrases rather than duplicates of the seed list. |
| seed_plus_templates_logreg | logreg | seed_plus_templates | macro_f1=0.9429 | accuracy=0.9444 | 0.06 | Evaluated on held-out paraphrases rather than duplicates of the seed list. |
| seed_plus_mixed_svm | linear_svm | seed_plus_mixed | macro_f1=0.9429 | accuracy=0.9444 | 0.04 | Evaluated on held-out paraphrases rather than duplicates of the seed list. |

### Notes

This benchmark uses a seed-list-style synthetic corpus because the primary research question is whether cheap synthetic expansion helps a narrow classifier when real data is scarce.

## Structural Bias Evaluation in Small Language Models

Dataset requested: stereoset
Dataset used: synthetic_stereoset_proxy
Objective: Score stereotype-sensitive sentence preferences with lightweight heuristics and compare bias-aware reranking adjustments.
Best experiment: biased_prior with icat=0.3750

### Findings
- The strongest bias-sensitive scorer was biased_prior with ICAT 0.375.
- A relevance-only scorer can still be structurally biased if it rewards stereotyped continuations more than anti-stereotyped ones.
- The debiased scorer trades a small amount of raw context-fit for a lower stereotype preference gap, which improves ICAT in this synthetic StereoSet-style benchmark.

### Recorded Experiments

| Variant | Algorithm | Features | Primary | Secondary | Runtime(s) | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| context_overlap | heuristic_scorer | context_overlap | icat=0.2500 | bias_gap=0.2500 | 0.00 | ICAT-style balance between contextual relevance and stereotype preference. |
| biased_prior | heuristic_scorer | context_plus_stereotype_prior | icat=0.3750 | bias_gap=0.2500 | 0.00 | ICAT-style balance between contextual relevance and stereotype preference. |
| debiased_prior | heuristic_scorer | context_plus_debias_penalty | icat=0.0000 | bias_gap=0.5000 | 0.00 | ICAT-style balance between contextual relevance and stereotype preference. |

### Notes

This is a compact StereoSet-style proxy with ICAT-like scoring. It keeps the bias-evaluation logic explicit and runnable without depending on a full masked language model.
