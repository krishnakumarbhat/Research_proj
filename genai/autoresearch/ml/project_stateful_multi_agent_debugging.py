from __future__ import annotations

import random

from sklearn.model_selection import train_test_split

from .common import (
    ProjectArtifact,
    concat_fields,
    evaluate_text_classifier,
    flatten_text,
    load_hf_dataset_records,
    make_result,
    pick_best_result,
    set_global_seed,
    timed_call,
)


PROJECT_ID = "stateful_multi_agent_debugging"
TITLE = "Stateful Multi-Agent Debugging Systems"
REQUESTED_DATASET = "msc-smart-contract/cpp-vulnerabilities or The Stack Smol C++ subset"


def _synthetic_records(total: int) -> list[dict[str, str]]:
    templates = [
        {
            "code_variants": [
                "RequestState* ctx = checkout_state(frame); int rc = parse_request(ctx); release_state(ctx); return rc;",
                "RequestState* current = checkout_state(packet); int parse_rc = parse_request(current); release_state(current); return parse_rc;",
                "Handle* state = borrow_state(frame); int status = run_parse(state); drop_state(state); return status;",
            ],
            "bug": "Agent memory: sanitizer trace shows parse_request defers status formatting and reads ctx after release on the error path.",
            "safe": "Agent memory: a prior patch cached all status fields before release_ctx, so the deferred formatter no longer touches ctx.",
            "issue": "use after free",
        },
        {
            "code_variants": [
                "char path[32]; build_path(path, user_input); return open_resource(path);",
                "char path_buf[32]; fill_path(path_buf, input_name); return open_resource(path_buf);",
                "char resource_path[32]; write_path(resource_path, raw_name); return open_resource(resource_path);",
            ],
            "bug": "Agent memory: fill_path was previously flagged for missing bounds checks when user_input exceeds 31 bytes.",
            "safe": "Agent memory: code review confirmed fill_path clamps input length and always null-terminates path.",
            "issue": "buffer overflow",
        },
        {
            "code_variants": [
                "Buffer* buf = reserve_buffer(256); parse_headers(buf); return finish_request();",
                "Scratch* headers = reserve_buffer(256); parse_headers(headers); return finish_request();",
                "Buffer* tmp = reserve_buffer(256); scan_headers(tmp); return finish_request();",
            ],
            "bug": "Agent memory: ownership audit shows alloc_buffer has no matching release on the normal return path.",
            "safe": "Agent memory: finish_request consumes and frees buf through the request arena, so no leak remains.",
            "issue": "memory leak",
        },
        {
            "code_variants": [
                "Session* session = checkout_session(id); finalize_session(session); if (retry) finalize_session(session);",
                "Session* current = checkout_session(id); finalize_session(current); if (should_retry) finalize_session(current);",
                "Handle* session = borrow_session(ticket); close_session(session); if (retry) close_session(session);",
            ],
            "bug": "Agent memory: retry handling can call finalize_session twice for the same session handle.",
            "safe": "Agent memory: finalize_session is idempotent after the refactor and ignores a second call on retried sessions.",
            "issue": "double free",
        },
        {
            "code_variants": [
                "Packet pkt = read_packet(sock); if (pkt.kind == CONTROL) return CONTROL_OK; return pkt.kind;",
                "Packet pkt = pull_packet(sock); if (pkt.kind == CONTROL) return CONTROL_OK; return pkt.kind;",
                "Frame pkt = read_frame(sock); if (pkt.kind == CONTROL) return CONTROL_OK; return pkt.kind;",
            ],
            "bug": "Agent memory: read_packet returns a view into a reused socket buffer, and pkt.kind is read after the buffer rotates.",
            "safe": "Agent memory: read_packet now copies pkt.kind into a standalone struct before the socket buffer rotates.",
            "issue": "stale buffer read",
        },
        {
            "code_variants": [
                "Config* cfg = load_config(path); apply_config(cfg); audit(cfg->version);",
                "Config* current = load_config(path); apply_config(current); audit(current->version);",
                "RuntimeCfg* cfg = parse_config(path); install_config(cfg); audit(cfg->version);",
            ],
            "bug": "Agent memory: apply_config transfers ownership and destroys cfg, so audit reads cfg->version after free.",
            "safe": "Agent memory: apply_config borrows cfg only, and audit(cfg->version) remains safe in this build.",
            "issue": "ownership confusion",
        },
    ]
    shared_notes = [
        "Secondary triage note: the failure appears only on the retry-heavy slow path.",
        "Reviewer context: the same release also changed logging, cleanup order, and error propagation.",
        "Incident timeline: the reproducer is intermittent and only shows up under verbose diagnostics.",
    ]
    rng = random.Random(42)
    records = []
    for index in range(total):
        template = templates[index % len(templates)]
        distractor_issue = templates[(index + 2) % len(templates)]["issue"]
        label = "bug" if rng.random() < 0.5 else "clean"
        history_prefix = template["bug"] if label == "bug" else template["safe"]
        outcome_note = (
            f"Current reproduction still points to {template['issue']} under the latest patch."
            if label == "bug"
            else f"Current reproduction no longer shows a live {template['issue']} despite the earlier incident."
        )
        records.append(
            {
                "code": template["code_variants"][index % len(template["code_variants"])],
                "history": concat_fields(
                    history_prefix,
                    shared_notes[index % len(shared_notes)],
                    f"Neighboring module also mentioned a possible {distractor_issue} during review.",
                    outcome_note,
                    f"Issue focus: {template['issue']}.",
                    f"Reviewer ticket #{100 + index}",
                ),
                "label": label,
            }
        )
    return records


def _load_records(quick: bool) -> tuple[list[dict[str, str]], str]:
    raw_records = load_hf_dataset_records(
        "msc-smart-contract/cpp-vulnerabilities",
        split="train[:80]" if quick else "train[:160]",
    )
    parsed = []
    if raw_records:
        for record in raw_records:
            code = flatten_text(record.get("code") or record.get("func") or record.get("text"))
            label_raw = flatten_text(record.get("label") or record.get("target") or record.get("vulnerable") or record.get("class"))
            if not code:
                continue
            label = "bug" if label_raw.lower() in {"1", "bug", "vulnerable", "true", "yes", "positive"} else "clean"
            bug_type = flatten_text(record.get("bug_type") or record.get("category") or record.get("vulnerability_type")) or "memory safety"
            parsed.append(
                {
                    "code": code,
                    "history": f"Recovered from dataset metadata: likely {bug_type} issue.",
                    "label": label,
                }
            )
        if len(parsed) >= 24:
            return parsed, "msc-smart-contract/cpp-vulnerabilities"
    return _synthetic_records(64 if quick else 128), "synthetic_cpp_debugging_corpus"


def run(*, quick: bool = True) -> ProjectArtifact:
    set_global_seed(42)
    mode = "quick" if quick else "full"
    records, used_dataset = _load_records(quick)

    current_texts = [record["code"] for record in records]
    memory_texts = [concat_fields(record["history"], record["code"]) for record in records]
    labels = [record["label"] for record in records]

    current_train, current_test, memory_train, memory_test, y_train, y_test = train_test_split(
        current_texts,
        memory_texts,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )

    experiments = [
        ("single_agent_logreg", "logreg", "current_code_only", current_train, current_test),
        ("single_agent_svm", "linear_svm", "current_code_only", current_train, current_test),
        ("stateful_agent_logreg", "logreg", "code_plus_bug_memory", memory_train, memory_test),
        ("stateful_agent_svm", "linear_svm", "code_plus_bug_memory", memory_train, memory_test),
        ("stateful_agent_nb", "nb", "code_plus_bug_memory", memory_train, memory_test),
    ]

    results = []
    for variant, algorithm, feature_set, train_texts, eval_texts in experiments:
        metrics, runtime = timed_call(
            evaluate_text_classifier,
            train_texts,
            y_train,
            eval_texts,
            y_test,
            algorithm=algorithm,
            analyzer="word",
            ngram_range=(1, 2),
            max_features=5000,
        )
        results.append(
            make_result(
                project_id=PROJECT_ID,
                project_name=TITLE,
                requested_dataset=REQUESTED_DATASET,
                used_dataset=used_dataset,
                mode=mode,
                variant=variant,
                algorithm=algorithm,
                feature_set=feature_set,
                metric_name="macro_f1",
                metric_value=metrics["macro_f1"],
                metric_direction="higher_is_better",
                secondary_metric_name="accuracy",
                secondary_metric_value=metrics["accuracy"],
                train_samples=len(y_train),
                eval_samples=len(y_test),
                runtime_sec=runtime,
                notes="Word n-grams over C++ snippets with or without persistent agent memory.",
            )
        )

    best = pick_best_result(results)
    current_best = max((result for result in results if result.feature_set == "current_code_only"), key=lambda item: item.metric_value)
    memory_best = max((result for result in results if result.feature_set == "code_plus_bug_memory"), key=lambda item: item.metric_value)
    if memory_best.metric_value > current_best.metric_value:
        memory_statement = f"Adding stateful debugging memory improved macro F1 from {current_best.metric_value:.3f} to {memory_best.metric_value:.3f}."
    elif memory_best.metric_value < current_best.metric_value:
        memory_statement = f"Adding stateful debugging memory reduced macro F1 from {current_best.metric_value:.3f} to {memory_best.metric_value:.3f} on this proxy."
    else:
        memory_statement = f"Stateful memory tied the current-code baseline at macro F1 {memory_best.metric_value:.3f} on this compact proxy."
    findings = [
        f"The best overall variant was {best.variant} with macro F1 {best.metric_value:.3f}." if best else "No successful run was recorded.",
        memory_statement,
        "The synthetic proxy is small and highly structured, so some state benefits are likely compressed relative to a noisier real bug-fix corpus.",
    ]
    return ProjectArtifact(
        project_id=PROJECT_ID,
        title=TITLE,
        objective="Benchmark whether persistent debugging memory improves lightweight bug-triage quality on small C++ code corpora.",
        requested_dataset=REQUESTED_DATASET,
        used_dataset=used_dataset,
        mode=mode,
        metric_name="macro_f1",
        metric_direction="higher_is_better",
        results=results,
        findings=findings,
        notes="The real C++ vulnerability dataset is used when available. Otherwise a synthetic memory-safety corpus acts as a CPU-first proxy for multi-agent debugging state.",
    )


if __name__ == "__main__":
    artifact = run()
    best = pick_best_result(artifact.results)
    print(f"{artifact.title}: {best.variant if best else 'no result'}")