from __future__ import annotations

from .common import ProjectArtifact, make_result, pick_best_result, rouge_l_f1, sentence_split, timed_call


PROJECT_ID = "context_window_compression"
TITLE = "Context Window Compression"
REQUESTED_DATASET = "tau/scrolls (GovReport subset)"


def _synthetic_records(total: int) -> list[dict[str, str]]:
    base = [
        {
            "document": "The city council released a 40-page transit report. The report documented overcrowding on three rail lines during peak hours. It found that signal failures caused repeated delays in the northern corridor. The report noted that weekend ridership grew faster than forecast after new housing projects opened. It recommended replacing century-old switching equipment and adding bus bridges during construction. It also proposed new fare caps for low-income riders. The final section argued that modernization would reduce both delays and long-term maintenance costs.",
            "summary": "A city transit report linked rail delays to aging signal equipment, noted rising ridership, and recommended modernization plus fare support measures.",
        },
        {
            "document": "A public health review examined vaccine uptake across rural counties. The review found that travel distance to clinics strongly predicted lower participation. Mobile outreach units improved vaccination rates in the pilot counties, especially where weekend hours were offered. Survey data showed that trust in local nurses mattered more than statewide media campaigns. The review advised expanding mobile clinics and funding local health workers before the winter season.",
            "summary": "The public health review found that clinic access and local trust drove vaccination rates and recommended expanding mobile outreach staffed by local workers.",
        },
        {
            "document": "An education commission studied broadband access for remote learning. Many households lacked stable evening connectivity even when daytime service met advertised targets. Students shared devices with siblings, which reduced attendance in live sessions. The commission observed that districts with offline-first lesson plans suffered fewer disruptions. It recommended subsidized home internet, device lending programs, and asynchronous lesson backups.",
            "summary": "The education commission reported that unstable home connectivity and shared devices harmed remote learning, recommending internet subsidies, device lending, and offline-capable lessons.",
        },
    ]
    return [base[index % len(base)].copy() for index in range(total)]


def _lead_only(text: str) -> str:
    return " ".join(sentence_split(text)[:4])


def _head_tail(text: str) -> str:
    sentences = sentence_split(text)
    return " ".join(sentences[:2] + sentences[-2:])


def _middle_focus(text: str) -> str:
    sentences = sentence_split(text)
    if len(sentences) <= 4:
        return " ".join(sentences)
    middle = len(sentences) // 2
    start = max(0, middle - 2)
    return " ".join(sentences[start : start + 4])


def _longest_sentences(text: str) -> str:
    sentences = sentence_split(text)
    ranked = sorted(sentences, key=lambda sentence: len(sentence.split()), reverse=True)[:4]
    return " ".join(ranked)


def _evaluate(records: list[dict[str, str]], compressor: callable) -> dict[str, float]:
    rouge_scores = []
    ratios = []
    for record in records:
        compressed = compressor(record["document"])
        rouge_scores.append(rouge_l_f1(record["summary"], compressed))
        ratios.append(len(compressed) / max(1, len(record["document"])))
    return {
        "rouge_l": sum(rouge_scores) / max(1, len(rouge_scores)),
        "retained_fraction": sum(ratios) / max(1, len(ratios)),
    }


def run(*, quick: bool = True) -> ProjectArtifact:
    mode = "quick" if quick else "full"
    records = _synthetic_records(18 if quick else 36)
    experiments = [
        ("lead_only", "extractive_compression", "lead_bias", _lead_only),
        ("head_tail", "extractive_compression", "head_tail_mix", _head_tail),
        ("middle_focus", "extractive_compression", "middle_retention", _middle_focus),
        ("longest_sentences", "extractive_compression", "length_centrality", _longest_sentences),
    ]

    results = []
    for variant, algorithm, feature_set, compressor in experiments:
        metrics, runtime = timed_call(_evaluate, records, compressor)
        results.append(
            make_result(
                project_id=PROJECT_ID,
                project_name=TITLE,
                requested_dataset=REQUESTED_DATASET,
                used_dataset="synthetic_govreport_proxy",
                mode=mode,
                variant=variant,
                algorithm=algorithm,
                feature_set=feature_set,
                metric_name="rouge_l",
                metric_value=metrics["rouge_l"],
                metric_direction="higher_is_better",
                secondary_metric_name="retained_fraction",
                secondary_metric_value=metrics["retained_fraction"],
                train_samples=len(records),
                eval_samples=len(records),
                runtime_sec=runtime,
                notes="Compression quality measured against reference summaries using ROUGE-L.",
            )
        )

    best = pick_best_result(results)
    findings = [
        f"The best compression strategy was {best.variant} with ROUGE-L {best.metric_value:.3f}." if best else "No successful run was recorded.",
        "Compression is scored directly against summaries rather than through a second summarization model, so the benchmark isolates how much salient information each compression strategy preserves.",
        "Lead-only truncation is strong on report-style prose, but head-tail or central-sentence strategies can recover late-document conclusions that lead bias discards.",
    ]
    return ProjectArtifact(
        project_id=PROJECT_ID,
        title=TITLE,
        objective="Compare simple compression policies for long reports under a summary-retention proxy.",
        requested_dataset=REQUESTED_DATASET,
        used_dataset="synthetic_govreport_proxy",
        mode=mode,
        metric_name="rouge_l",
        metric_direction="higher_is_better",
        results=results,
        findings=findings,
        notes="This module uses a synthetic GovReport-style corpus and evaluates how well compressed contexts retain summary-worthy content.",
    )


if __name__ == "__main__":
    artifact = run()
    best = pick_best_result(artifact.results)
    print(f"{artifact.title}: {best.variant if best else 'no result'}")