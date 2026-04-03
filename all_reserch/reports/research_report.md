# Research Experiment Report

Generated on 2026-04-01 from the local runnable benchmark suite.

## Scope

This workspace contains three executable research prototypes aligned with the target paper directions. Paper 1 is a controlled OCR proxy for visual text rendering, while Papers 2 and 3 are retrieval benchmarks with validation search, held-out testing, and exported artifacts.

Current execution profile: Full benchmark.

## Overall Best Results

| Paper   | Best configuration               | Primary metric | Value  | Comment                                                 |
| ------- | -------------------------------- | -------------- | ------ | ------------------------------------------------------- |
| Paper 1 | layout_robust (layout_guided)    | Mean CER       | 0.0000 | CER improvement over best baseline: 0.9040              |
| Paper 2 | graph_assisted / lsa             | Objective      | 2.2944 | Cuts candidates by 6.07 vs vanilla word_tfidf           |
| Paper 3 | late_chunking_proxy / char_tfidf | Objective      | 1.7670 | Highest Top-1 overall: late_chunking_proxy / char_tfidf |

## Paper 1

The strongest visual text configuration was layout_robust, and every structured rendering method reduced mean CER to 0.0. The fastest zero-CER option remained layout_guided at 4.4959 ms.

## Paper 2

The best deployment-aware configuration was graph_assisted with lsa, while the highest raw Top-1 came from graph_assisted with lsa at 0.4701.

## Paper 3

The best objective score came from late_chunking_proxy with char_tfidf. The highest Top-3 recall came from late_chunking_proxy with char_tfidf at 0.5091.

## Artifacts

Detailed tables, figures, and per-query outputs are available under the artifacts directory and in the detailed markdown report.
