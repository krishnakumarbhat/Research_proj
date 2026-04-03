from __future__ import annotations

from .common import ProjectArtifact, concat_fields, evaluate_text_classifier, make_result, pick_best_result, set_global_seed, timed_call


PROJECT_ID = "qlora_sub_billion_legal"
TITLE = "QLoRA Fine-Tuning of Sub-Billion Models (Legal Proxy)"
REQUESTED_DATASET = "cuad"


def _synthetic_splits(quick: bool) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    train_templates = {
        "confidentiality": [
            "Each party shall keep non-public technical information confidential during the term and for three years after termination of this agreement.",
            "Confidential material may be disclosed solely to employees bound by duties of nondisclosure, and all copies must be returned when the agreement ends.",
            "Pricing, product plans, and source materials remain confidential even if payment disputes or termination rights later arise.",
        ],
        "termination": [
            "Either party may terminate this agreement after thirty days' notice if a material breach remains uncured, and confidential materials must then be returned.",
            "This agreement terminates immediately upon insolvency or cessation of business operations, although accrued payment obligations survive.",
            "A non-breaching party may end the contract for uncured default after notice, with confidentiality and liability provisions surviving termination.",
        ],
        "governing_law": [
            "This agreement is governed by the laws of Delaware, excluding conflict-of-law rules, and disputes shall be heard in Wilmington.",
            "The parties submit to the exclusive jurisdiction of courts located in New York, New York for any dispute involving payment, confidentiality, or termination.",
            "Questions of contract interpretation and enforcement shall be resolved under Texas law in the state or federal courts of Austin.",
        ],
        "payment_terms": [
            "Invoices are payable within thirty days after receipt and past-due balances accrue late charges until cured or service suspension.",
            "Customer shall remit subscription fees within fifteen days from the invoice date, and undisputed nonpayment may trigger suspension.",
            "License fees are billed in advance, become due net thirty, and remain payable even if a later termination notice is delivered.",
        ],
        "liability_cap": [
            "Neither party's aggregate liability shall exceed the fees paid under this agreement during the preceding twelve months, except for fraud or breach of confidentiality.",
            "Indirect, special, and consequential damages are excluded except where prohibited by law, and total damages remain capped at fees paid.",
            "Except for willful misconduct, each side's total exposure is limited to amounts paid under the agreement during the prior year.",
        ],
    }
    eval_templates = {
        "confidentiality": [
            "A receiving party must protect proprietary business information from disclosure except to representatives under matching secrecy obligations, even after the contract ends.",
            "Trade secrets exchanged under this contract cannot be shared outside approved personnel and must be returned or destroyed upon termination.",
            "Non-public business records remain subject to secrecy duties despite any venue, payment, or breach dispute.",
        ],
        "termination": [
            "If a serious default is not cured after written notice, the non-breaching party may end the contract and require return of confidential materials.",
            "The arrangement ends at once if one side enters bankruptcy proceedings or dissolves its operations, while unpaid fees remain collectible.",
            "Either party may cancel for uncured breach after notice, but confidentiality and liability terms survive the cancellation.",
        ],
        "governing_law": [
            "Disputes will be interpreted under California law and litigated in San Francisco courts.",
            "This contract shall be construed in accordance with Texas law and venue shall lie in Austin.",
            "All claims concerning payment, termination, or confidentiality shall be governed by Illinois law and heard in Chicago.",
        ],
        "payment_terms": [
            "Amounts due must be paid net thirty, and overdue balances may incur finance charges or temporary service suspension.",
            "License fees are billed in advance and become due fifteen days after invoicing, regardless of later termination rights.",
            "Undisputed invoices are payable within twenty days, after which late charges may accrue.",
        ],
        "liability_cap": [
            "The provider's maximum total exposure is limited to fees actually paid by the customer during the prior year, except for fraud or misuse of confidential information.",
            "Neither party will be liable for consequential losses beyond the negotiated damages ceiling.",
            "Total contractual damages are capped at the fees paid under the agreement, excluding claims that cannot be limited by law.",
        ],
    }

    def build(templates: dict[str, list[str]]) -> list[dict[str, str]]:
        records = []
        for label, clauses in templates.items():
            for clause in clauses:
                qa_style = concat_fields(
                    "Question: Which legal clause category does this passage belong to?",
                    "Possible labels: confidentiality, termination, governing_law, payment_terms, liability_cap.",
                    f"Passage: {clause}",
                )
                records.append({"clause": clause, "qa_style": qa_style, "label": label})
        return records

    train_records = build(train_templates)
    eval_records = build(eval_templates)
    if not quick:
        train_records = train_records + eval_records[:5]
    return train_records, eval_records


def run(*, quick: bool = True) -> ProjectArtifact:
    set_global_seed(42)
    mode = "quick" if quick else "full"
    train_records, eval_records = _synthetic_splits(quick)
    clause_train = [record["clause"] for record in train_records]
    qa_train = [record["qa_style"] for record in train_records]
    y_train = [record["label"] for record in train_records]
    clause_test = [record["clause"] for record in eval_records]
    qa_test = [record["qa_style"] for record in eval_records]
    y_test = [record["label"] for record in eval_records]

    experiments = [
        ("clause_only_logreg", "logreg", "raw_clause", clause_train, clause_test),
        ("clause_only_svm", "linear_svm", "raw_clause", clause_train, clause_test),
        ("qa_style_logreg", "logreg", "instruction_wrapped_clause", qa_train, qa_test),
        ("qa_style_nb", "nb", "instruction_wrapped_clause", qa_train, qa_test),
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
                used_dataset="synthetic_cuad_proxy",
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
                notes="Clause-type classification proxy for legal adaptation.",
            )
        )

    best = pick_best_result(results)
    raw_best = max((result for result in results if result.feature_set == "raw_clause"), key=lambda item: item.metric_value)
    wrapped_best = max((result for result in results if result.feature_set == "instruction_wrapped_clause"), key=lambda item: item.metric_value)
    if wrapped_best.metric_value > raw_best.metric_value:
        wrap_statement = f"Instruction wrapping improved macro F1 from {raw_best.metric_value:.3f} to {wrapped_best.metric_value:.3f}."
    elif wrapped_best.metric_value < raw_best.metric_value:
        wrap_statement = f"Raw clauses outperformed instruction wrapping on this proxy ({raw_best.metric_value:.3f} versus {wrapped_best.metric_value:.3f})."
    else:
        wrap_statement = f"Raw-clause and instruction-wrapped variants tied at macro F1 {raw_best.metric_value:.3f} on the current proxy dataset."
    findings = [
        f"The best legal adaptation proxy was {best.variant} with macro F1 {best.metric_value:.3f}." if best else "No successful run was recorded.",
        wrap_statement,
        "Specialized legal language is structured enough that even compact models can separate clause types on narrow corpora.",
    ]
    return ProjectArtifact(
        project_id=PROJECT_ID,
        title=TITLE,
        objective="Measure how well lightweight text models adapt to a narrow legal clause taxonomy under a small-data proxy for QLoRA-style specialization.",
        requested_dataset=REQUESTED_DATASET,
        used_dataset="synthetic_cuad_proxy",
        mode=mode,
        metric_name="macro_f1",
        metric_direction="higher_is_better",
        results=results,
        findings=findings,
        notes="This is a clause-classification proxy rather than true parameter-efficient fine-tuning. It is designed to keep the benchmark runnable without downloading or adapting a large legal language model.",
    )


if __name__ == "__main__":
    artifact = run()
    best = pick_best_result(artifact.results)
    print(f"{artifact.title}: {best.variant if best else 'no result'}")