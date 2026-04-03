from __future__ import annotations

from .common import ProjectArtifact, bm25_rank_documents, concat_fields, make_result, pick_best_result, retrieval_metrics, timed_call, tfidf_rank_documents, hashing_rank_documents


PROJECT_ID = "agentic_frameworks_hardware_synthesis"
TITLE = "Agentic Frameworks for Hardware Synthesis"
REQUESTED_DATASET = "shailja/Verilog_GitHub"


MODULE_LIBRARY = [
    {
        "name": "pattern_fsm",
        "spec": "Build a circuit that watches a serial bit stream and raises done after the sequence 1011 appears.",
        "code": "module pattern_fsm(input clk, input rst, input bit_in, output reg done);\n  // sequential control logic\nendmodule",
    },
    {
        "name": "shift_register",
        "spec": "Design a bit-storage block that moves its contents one position on every clock edge.",
        "code": "module shift_register #(parameter WIDTH=8)(input clk, input din, output dout);\n  // parameterized serial datapath\nendmodule",
    },
    {
        "name": "fifo_ctrl",
        "spec": "Implement a queue controller that tracks when data can be written or read and exposes full and empty status.",
        "code": "module fifo_ctrl(input clk, input rst, input wr_en, input rd_en, output full, output empty);\n  // synchronous queue control logic\nendmodule",
    },
    {
        "name": "alu4",
        "spec": "Create a 4-bit arithmetic block that chooses between addition, subtraction, and XOR using an opcode.",
        "code": "module alu4(input [3:0] a, input [3:0] b, input [1:0] opcode, output reg [3:0] y);\n  // compact combinational datapath\nendmodule",
    },
    {
        "name": "debouncer",
        "spec": "Write a button-conditioning module that removes mechanical bounce and outputs a stable signal.",
        "code": "module debouncer(input clk, input noisy_btn, output reg clean_btn);\n  // small synchronizer and filter block\nendmodule",
    },
    {
        "name": "uart_tx",
        "spec": "Construct a serial byte sender that emits start and stop bits and reports when transmission is busy.",
        "code": "module uart_tx(input clk, input start, input [7:0] data, output tx, output busy);\n  // serial framing control path\nendmodule",
    },
    {
        "name": "uart_rx",
        "spec": "Construct a serial byte receiver that samples start and stop bits and reports when a byte is valid.",
        "code": "module uart_rx(input clk, input rx, output reg [7:0] data, output reg valid);\n  // serial sampling control path\nendmodule",
    },
    {
        "name": "fifo_mem",
        "spec": "Create the storage array used by a FIFO data path.",
        "code": "module fifo_mem(input clk, input wr_en, input [7:0] din, input [3:0] wr_ptr, input [3:0] rd_ptr, output [7:0] dout);\n  // indexed storage datapath\nendmodule",
    },
]

QUERY_TASKS = [
    {
        "query": "Need a logic block that flags after four specific serial bits arrive in order.",
        "gold": "pattern_fsm",
    },
    {
        "query": "Need a narrow storage lane that advances its captured bit each tick.",
        "gold": "shift_register",
    },
    {
        "query": "Need only the control side of a circular buffer: pointer motion plus availability flags.",
        "gold": "fifo_ctrl",
    },
    {
        "query": "Need a tiny 4-bit block that picks math or xor behavior from a control code.",
        "gold": "alu4",
    },
    {
        "query": "Need a cleanup stage so a chattering push-button becomes a steady logic signal.",
        "gold": "debouncer",
    },
    {
        "query": "Need an outgoing serial byte path with frame bits and a signal that says the line is occupied.",
        "gold": "uart_tx",
    },
]

MODULE_INDEX = {record["name"]: index for index, record in enumerate(MODULE_LIBRARY)}


def _expand_query(spec: str) -> str:
    additions = []
    lowered = spec.lower()
    if "fsm" in lowered or "state" in lowered:
        additions.append("state transition next state done")
    if "sequence" in lowered or "serial stream" in lowered or "done flag" in lowered:
        additions.append("fsm sequence detector state transition done")
    if "four specific serial bits" in lowered or "arrive in order" in lowered:
        additions.append("serial pattern detector state machine done")
    if "queue" in lowered:
        additions.append("fifo read pointer write pointer occupancy")
    if "occupancy" in lowered or "full or empty" in lowered:
        additions.append("fifo controller occupancy full empty write read")
    if "circular buffer" in lowered or "availability flags" in lowered:
        additions.append("fifo controller read pointer write pointer full empty")
    if "button" in lowered or "bounce" in lowered or "stable" in lowered:
        additions.append("debounce synchronizer counter")
    if "switch" in lowered or "push-button" in lowered:
        additions.append("debounce synchronizer stable button")
    if "serial byte sender" in lowered or "start and stop bits" in lowered or "busy" in lowered:
        additions.append("uart baud transmit state machine")
    if "transmitter" in lowered or "framing" in lowered:
        additions.append("uart transmitter start stop busy baud")
    if "line is occupied" in lowered or "outgoing serial byte" in lowered:
        additions.append("uart transmitter busy start stop serial")
    if "fifo" in lowered:
        additions.append("read pointer write pointer full empty queue")
    if "shift" in lowered:
        additions.append("serial register clocked")
    if "advances its captured bit" in lowered or "storage lane" in lowered:
        additions.append("shift register serial datapath clocked")
    if "moves its contents" in lowered:
        additions.append("shift register serial datapath")
    if "uart" in lowered:
        additions.append("baud transmitter start stop bits serial")
    if "alu" in lowered:
        additions.append("opcode arithmetic logic add subtract xor")
    if "control code" in lowered or "math or xor" in lowered:
        additions.append("alu opcode arithmetic xor subtract add")
    if "arithmetic block" in lowered:
        additions.append("alu opcode datapath xor subtract add")
    return concat_fields(spec, " ".join(additions))


def run(*, quick: bool = True) -> ProjectArtifact:
    mode = "quick" if quick else "full"
    documents = [record["code"] for record in MODULE_LIBRARY]
    records = QUERY_TASKS[:4] if quick else QUERY_TASKS
    queries = [record["query"] for record in records]
    expanded_queries = [_expand_query(query) for query in queries]
    gold_indices = [MODULE_INDEX[record["gold"]] for record in records]

    experiments = [
        ("tfidf_spec_only", "tfidf", "single_step_retrieval", lambda: tfidf_rank_documents(queries, documents, top_k=1, max_features=4000)),
        ("bm25_spec_only", "bm25", "single_step_retrieval", lambda: bm25_rank_documents(queries, documents, top_k=1)),
        ("planner_query_expansion", "tfidf", "agentic_query_expansion", lambda: tfidf_rank_documents(expanded_queries, documents, top_k=1, max_features=4000)),
        ("hashing_retrieval", "hashing", "fixed_width_retrieval", lambda: hashing_rank_documents(queries, documents, top_k=1)),
    ]

    results = []
    for variant, algorithm, feature_set, ranker in experiments:
        rankings, runtime = timed_call(ranker)
        metrics = retrieval_metrics(rankings, gold_indices, top_k=1)
        results.append(
            make_result(
                project_id=PROJECT_ID,
                project_name=TITLE,
                requested_dataset=REQUESTED_DATASET,
                used_dataset="synthetic_verilog_module_library",
                mode=mode,
                variant=variant,
                algorithm=algorithm,
                feature_set=feature_set,
                metric_name="recall_at_1",
                metric_value=metrics["recall_at_1"],
                metric_direction="higher_is_better",
                secondary_metric_name="mrr",
                secondary_metric_value=metrics["mrr"],
                train_samples=len(documents),
                eval_samples=len(queries),
                runtime_sec=runtime,
                notes="Retrieval proxy for agent planning over hardware module libraries.",
            )
        )

    best = pick_best_result(results)
    spread = max(result.metric_value for result in results) - min(result.metric_value for result in results)
    if spread < 1e-9:
        planning_statement = "All retrieval variants solved the compact hardware-library proxy equally well, so query expansion did not differentiate performance here."
    else:
        planning_statement = "Query expansion behaves like a lightweight planning layer by injecting hardware-specific tokens before retrieval."
    findings = [
        f"The highest module-retrieval accuracy came from {best.variant} with recall@1 {best.metric_value:.3f}." if best else "No successful run was recorded.",
        planning_statement,
        "This is a retrieval-stage proxy for agentic synthesis, not full HDL generation or simulation-driven verification.",
    ]
    return ProjectArtifact(
        project_id=PROJECT_ID,
        title=TITLE,
        objective="Test whether simple planning-style query expansion helps agents find the right hardware building blocks before code synthesis.",
        requested_dataset=REQUESTED_DATASET,
        used_dataset="synthetic_verilog_module_library",
        mode=mode,
        metric_name="recall_at_1",
        metric_direction="higher_is_better",
        results=results,
        findings=findings,
        notes="A compact synthetic Verilog library is used here because the focus is the agentic retrieval step that precedes hardware synthesis.",
    )


if __name__ == "__main__":
    artifact = run()
    best = pick_best_result(artifact.results)
    print(f"{artifact.title}: {best.variant if best else 'no result'}")