from __future__ import annotations

import random

from sklearn.model_selection import train_test_split

from .common import ProjectArtifact, concat_fields, evaluate_text_classifier, make_result, pick_best_result, set_global_seed, timed_call


PROJECT_ID = "small_model_orchestration_ci_cd"
TITLE = "Small Model Orchestration for CI/CD"
REQUESTED_DATASET = "github-commit-messages-with-bug-fixes or CommitPackFT"


def _synthetic_records(total: int) -> list[dict[str, str]]:
    templates = [
        {
            "messages": [
                "Update release workflow for node service",
                "Refresh node-service release workflow",
            ],
            "neutral": [
                "- NODE_OPTIONS=--max-old-space-size=2048\n+ NODE_OPTIONS=--max-old-space-size=2048",
                "- cache: npm\n+ cache: npm",
            ],
            "risky": [
                "- uses: actions/setup-node@v4\n+ uses: actions/setup-node@main",
                "- ./smoke_test.sh\n+ # smoke test skipped",
                "- npm ci\n+ npm install",
            ],
            "safe": [
                "- uses: actions/setup-node@v4\n+ uses: actions/setup-node@v4",
                "- npm ci\n+ npm ci --prefer-offline",
                "+ ./smoke_test.sh",
                "+ allow_failure: false",
            ],
            "caution": [
                "+ note: keep smoke test coverage enabled for release jobs",
                "+ validation remains blocking for production deploys",
            ],
        },
        {
            "messages": [
                "Refactor parser integration for token upgrade",
                "Refresh parser token integration layer",
            ],
            "neutral": [
                "- parse_token(stream)\n+ parse_token(stream)",
                "- token_cache.reserve(256)\n+ token_cache.reserve(256)",
            ],
            "risky": [
                "- tests/parser_golden.json\n+ tests removed",
                "- assert token.kind == IDENT\n+ assert token.type == IDENTIFIER",
                "+ skip parser compatibility shim",
            ],
            "safe": [
                "+ tests/parser_golden.json updated",
                "+ compatibility shim added",
                "+ parser fallback for IDENT retained",
                "+ allow_failure: false",
            ],
            "caution": [
                "+ note: do not drop parser goldens during token migrations",
                "+ compatibility path remains covered by regression tests",
            ],
        },
        {
            "messages": [
                "Refresh deployment environment handling",
                "Restructure environment bootstrap for deployment",
            ],
            "neutral": [
                "- export APP_ENV=prod\n+ export APP_ENV=prod",
                "- export REGION=us-east-1\n+ export REGION=us-east-1",
            ],
            "risky": [
                "- ./validate_env.sh\n+ # validation removed",
                "- export API_URL=${LEGACY_URL}\n+ export API_URL=${NEW_ENDPOINT}",
                "+ no fallback for LEGACY_URL",
            ],
            "safe": [
                "+ ./validate_env.sh",
                "+ fallback for LEGACY_URL retained",
                "+ allow_failure: false",
            ],
            "caution": [
                "+ comment: keep validation blocking for production environments",
                "+ note: rollout should abort if env verification fails",
            ],
        },
        {
            "messages": [
                "Tune native build flags for linux release",
                "Adjust native linux linker and runtime flags",
            ],
            "neutral": [
                "-CFLAGS += -O2\n+CFLAGS += -O2",
                "-include config/build.h\n+include config/build.h",
            ],
            "risky": [
                "-LDFLAGS += -pthread\n+LDFLAGS += -Wl,--as-needed",
                "-include core/runtime.h\n+include runtime/new_api.h",
                "+ compatibility wrapper removed",
            ],
            "safe": [
                "+LDFLAGS += -pthread -Wl,--as-needed",
                "+ include runtime/new_api.h",
                "+ compatibility wrapper kept",
            ],
            "caution": [
                "+ note: runtime wrapper remains required on older builders",
                "+ keep pthread linkage for release binaries",
            ],
        },
        {
            "messages": [
                "Restructure nightly test job",
                "Rework nightly test orchestration",
            ],
            "neutral": [
                "- schedule: nightly\n+ schedule: nightly",
                "- timeout-minutes: 30\n+ timeout-minutes: 30",
            ],
            "risky": [
                "- pytest tests/smoke.py\n+ pytest tests/critical.py",
                "- coverage upload\n+ allow_failure: true",
                "+ fail-fast disabled",
            ],
            "safe": [
                "+ pytest tests/smoke.py tests/critical.py",
                "+ coverage upload",
                "+ fail-fast preserved",
                "+ allow_failure: false",
            ],
            "caution": [
                "+ note: critical tests must not replace smoke coverage",
                "+ coverage remains mandatory for merge eligibility",
            ],
        },
    ]
    rng = random.Random(42)
    records = []
    for index in range(total):
        template = templates[index % len(templates)]
        label = "break" if rng.random() < 0.5 else "safe"
        message = template["messages"][index % len(template["messages"])]
        neutral = rng.choice(template["neutral"])
        safe_line = rng.choice(template["safe"])
        caution_line = rng.choice(template["caution"])
        supporting_safe = rng.choice(templates[(index + 1) % len(templates)]["safe"])
        if label == "break":
            diff_lines = [
                neutral,
                safe_line,
                rng.choice(template["risky"]),
                supporting_safe if index % 3 == 0 else caution_line,
            ]
        else:
            diff_lines = [
                neutral,
                safe_line,
                supporting_safe,
                caution_line,
            ]
        diff = "\n".join(diff_lines)
        records.append(
            {
                "message": message,
                "diff": diff,
                "combo": concat_fields(message, diff),
                "label": label,
            }
        )
    return records


def run(*, quick: bool = True) -> ProjectArtifact:
    set_global_seed(42)
    mode = "quick" if quick else "full"
    records = _synthetic_records(64 if quick else 128)
    labels = [record["label"] for record in records]
    message_texts = [record["message"] for record in records]
    diff_texts = [record["diff"] for record in records]
    combo_texts = [record["combo"] for record in records]

    message_train, message_test, diff_train, diff_test, combo_train, combo_test, y_train, y_test = train_test_split(
        message_texts,
        diff_texts,
        combo_texts,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )

    experiments = [
        ("message_only_logreg", "logreg", "commit_message_only", message_train, message_test),
        ("diff_only_logreg", "logreg", "code_diff_only", diff_train, diff_test),
        ("message_plus_diff_logreg", "logreg", "message_plus_diff", combo_train, combo_test),
        ("message_plus_diff_svm", "linear_svm", "message_plus_diff", combo_train, combo_test),
        ("message_plus_diff_nb", "nb", "message_plus_diff", combo_train, combo_test),
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
                used_dataset="synthetic_commitpack_proxy",
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
                notes="Text triage proxy for build-break risk prediction from commit metadata.",
            )
        )

    best = pick_best_result(results)
    message_only = next(result for result in results if result.variant == "message_only_logreg")
    combo_best = max((result for result in results if result.feature_set == "message_plus_diff"), key=lambda item: item.metric_value)
    if combo_best.metric_value > message_only.metric_value:
        combo_statement = f"Combining commit messages with diffs improved macro F1 from {message_only.metric_value:.3f} to {combo_best.metric_value:.3f}."
    elif combo_best.metric_value < message_only.metric_value:
        combo_statement = f"On this proxy, commit messages alone outperformed the message-plus-diff variants ({message_only.metric_value:.3f} versus {combo_best.metric_value:.3f})."
    else:
        combo_statement = f"Commit messages and message-plus-diff variants tied at macro F1 {combo_best.metric_value:.3f} on this easy proxy dataset."
    findings = [
        f"The best orchestration stack was {best.variant} with macro F1 {best.metric_value:.3f}." if best else "No successful run was recorded.",
        combo_statement,
        "A more realistic commit corpus is likely to widen the gap between message-only and diff-aware triage, because real CI failures often hide in build scripts and deleted tests.",
    ]
    return ProjectArtifact(
        project_id=PROJECT_ID,
        title=TITLE,
        objective="Estimate whether small text models can triage risky code changes before expensive CI/CD jobs execute.",
        requested_dataset=REQUESTED_DATASET,
        used_dataset="synthetic_commitpack_proxy",
        mode=mode,
        metric_name="macro_f1",
        metric_direction="higher_is_better",
        results=results,
        findings=findings,
        notes="The workspace does not include a commit-diff corpus, so the benchmark uses a synthetic proxy built around CI failure motifs such as deleted tests, build-file edits, and deployment-script drift.",
    )


if __name__ == "__main__":
    artifact = run()
    best = pick_best_result(artifact.results)
    print(f"{artifact.title}: {best.variant if best else 'no result'}")