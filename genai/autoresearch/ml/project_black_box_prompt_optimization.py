from __future__ import annotations

import random
import re

from .common import ProjectArtifact, load_hf_dataset_records, make_result, pick_best_result, set_global_seed, timed_call


PROJECT_ID = "black_box_prompt_optimization"
TITLE = "Black-Box Prompt Optimization"
REQUESTED_DATASET = "openai/gsm8k"
PROMPT_COMPONENTS = [
    ("sum_rule", "Look for total or altogether cues."),
    ("subtract_rule", "When the question says left, remain, spent, or after giving, try subtraction."),
    ("multiply_rule", "When the question says each, every, per, or times, try multiplication first."),
    ("divide_rule", "When the question says equally or share, try division."),
    ("sequence_rule", "If the question has then/after, prefer compound operations."),
]


def _extract_number(text: str) -> float | None:
    match = re.search(r"####\s*([-0-9.,]+)", text)
    if match:
        return float(match.group(1).replace(",", ""))
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    return float(numbers[-1]) if numbers else None


def _is_simple_problem(question: str) -> bool:
    numbers = re.findall(r"\d+", question)
    keywords = ["total", "altogether", "left", "remain", "each", "every", "per", "share", "equally", "times"]
    return 2 <= len(numbers) <= 4 and any(keyword in question.lower() for keyword in keywords)


def _synthetic_examples(total: int) -> list[dict[str, float | str]]:
    base = [
        ("Lena has 4 bags with 3 marbles in each bag. How many marbles does she have in total?", 12.0),
        ("A store had 18 apples and sold 7 of them. How many apples are left?", 11.0),
        ("Noah splits 24 stickers equally among 6 friends. How many stickers does each friend get?", 4.0),
        ("A bus has 12 riders, then 5 more get on. How many riders are on the bus now?", 17.0),
        ("Mia buys 3 packs with 8 pencils each and then gives away 4 pencils. How many pencils remain?", 20.0),
        ("A baker makes 5 trays with 6 muffins on each tray and sells 8 muffins. How many muffins remain?", 22.0),
        ("Sam has 42 cards and shares them equally among 7 players. How many cards does each player get?", 6.0),
        ("A class has 14 students, and 3 more arrive later. How many students are there now?", 17.0),
        ("Ella saves 9 dollars each week for 4 weeks and then spends 6 dollars. How much money does she have left?", 30.0),
    ]
    return [{"question": question, "answer": answer} for question, answer in (base[index % len(base)] for index in range(total))]


def _load_examples(quick: bool) -> tuple[list[dict[str, float | str]], str]:
    raw = load_hf_dataset_records("openai/gsm8k", name="main", split="train[:100]" if quick else "train[:220]")
    if raw is None:
        raw = load_hf_dataset_records("gsm8k", name="main", split="train[:100]" if quick else "train[:220]")

    parsed = []
    if raw:
        for record in raw:
            question = str(record.get("question", "")).strip()
            answer_value = _extract_number(str(record.get("answer", "")))
            if answer_value is None or not _is_simple_problem(question):
                continue
            parsed.append({"question": question, "answer": answer_value})
        if len(parsed) >= 40:
            return parsed, "openai/gsm8k_filtered_simple"
    return _synthetic_examples(48 if quick else 96), "synthetic_gsm8k_proxy"


def _prompt_text(bits: tuple[int, ...]) -> str:
    active = [description for bit, (_, description) in zip(bits, PROMPT_COMPONENTS) if bit]
    return " ".join(active) if active else "Return the best arithmetic answer."


def _candidate_values(numbers: list[float]) -> dict[str, float]:
    if not numbers:
        return {"sum": 0.0}
    candidates = {
        "sum": sum(numbers),
        "subtract_chain": numbers[0] - sum(numbers[1:]),
    }
    if len(numbers) >= 3:
        candidates["add_then_subtract_rest"] = numbers[0] + numbers[1] - sum(numbers[2:])
    if len(numbers) >= 2 and numbers[1] != 0:
        candidates["divide_first_two"] = numbers[0] / numbers[1]
        candidates["divide_then_add_rest"] = numbers[0] / numbers[1] + sum(numbers[2:])
        candidates["divide_then_subtract_rest"] = numbers[0] / numbers[1] - sum(numbers[2:])
    if len(numbers) >= 2:
        product = numbers[0] * numbers[1]
        candidates["multiply_first_two"] = product
        candidates["mul_then_add_rest"] = product + sum(numbers[2:])
        candidates["mul_then_subtract_rest"] = product - sum(numbers[2:])
    return candidates


def _score_candidate(name: str, question: str, bits: tuple[int, ...]) -> float:
    lowered = question.lower()
    enabled = {component_name for bit, (component_name, _) in zip(bits, PROMPT_COMPONENTS) if bit}
    score = 0.0
    has_add = any(token in lowered for token in ["total", "altogether", "in all", "now", "more", "arrive", "get on"])
    has_sub = any(token in lowered for token in ["left", "remain", "gave away", "sold", "spends", "spend", "after giving"])
    has_mul = any(token in lowered for token in ["each", "every", "per", "times"])
    has_div = any(token in lowered for token in ["share", "equally", "split"])
    has_sequence = any(token in lowered for token in ["then", "after", "later"])

    if has_add and not has_mul and not has_div:
        score += 2.0 if name == "sum" and "sum_rule" in enabled else 0.5 if name == "sum" else 0.0
    if has_sub and not has_mul and not has_div:
        score += 2.0 if name in {"subtract_chain", "add_then_subtract_rest"} and "subtract_rule" in enabled else 0.5 if name in {"subtract_chain", "add_then_subtract_rest"} else 0.0
    if has_mul and not has_div and not has_sub:
        score += 2.0 if name in {"multiply_first_two", "mul_then_add_rest"} and "multiply_rule" in enabled else 0.5 if name in {"multiply_first_two", "mul_then_add_rest"} else 0.0
    if has_div:
        score += 2.0 if name in {"divide_first_two", "divide_then_add_rest", "divide_then_subtract_rest"} and "divide_rule" in enabled else 0.5 if name in {"divide_first_two", "divide_then_add_rest", "divide_then_subtract_rest"} else 0.0
    if has_mul and has_sub:
        score += 3.0 if name == "mul_then_subtract_rest" and {"multiply_rule", "subtract_rule", "sequence_rule"}.issubset(enabled) else 0.0
    if has_mul and has_add:
        score += 3.0 if name == "mul_then_add_rest" and {"multiply_rule", "sum_rule", "sequence_rule"}.issubset(enabled) else 0.0
    if has_div and has_add:
        score += 2.5 if name == "divide_then_add_rest" and {"divide_rule", "sum_rule", "sequence_rule"}.issubset(enabled) else 0.0
    if has_div and has_sub:
        score += 2.5 if name == "divide_then_subtract_rest" and {"divide_rule", "subtract_rule", "sequence_rule"}.issubset(enabled) else 0.0
    if has_sequence and "sequence_rule" in enabled and name in {"mul_then_add_rest", "mul_then_subtract_rest", "add_then_subtract_rest", "divide_then_add_rest", "divide_then_subtract_rest"}:
        score += 1.0
    return score


def _solve(question: str, bits: tuple[int, ...]) -> float:
    numbers = [float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", question)]
    candidates = _candidate_values(numbers)
    ranked = max(candidates.items(), key=lambda item: (_score_candidate(item[0], question, bits), -abs(item[1])))
    return round(ranked[1], 3)


def _accuracy(examples: list[dict[str, float | str]], bits: tuple[int, ...]) -> float:
    hits = 0
    for example in examples:
        prediction = _solve(str(example["question"]), bits)
        answer = float(example["answer"])
        if abs(prediction - answer) < 1e-6:
            hits += 1
    return hits / max(1, len(examples))


def _neighbors(bits: tuple[int, ...]) -> list[tuple[int, ...]]:
    neighbors = []
    for index in range(len(bits)):
        flipped = list(bits)
        flipped[index] = 1 - flipped[index]
        neighbors.append(tuple(flipped))
    return neighbors


def _random_prompt() -> tuple[int, ...]:
    bits = [random.randint(0, 1) for _ in PROMPT_COMPONENTS]
    return tuple(bits if any(bits) else [1, 0, 0, 0, 0])


def _random_search(train_examples: list[dict[str, float | str]], budget: int) -> tuple[tuple[int, ...], float]:
    best_bits = _random_prompt()
    best_score = _accuracy(train_examples, best_bits)
    for _ in range(budget - 1):
        candidate = _random_prompt()
        score = _accuracy(train_examples, candidate)
        if score > best_score:
            best_bits, best_score = candidate, score
    return best_bits, best_score


def _hill_climb(train_examples: list[dict[str, float | str]], steps: int) -> tuple[tuple[int, ...], float]:
    current = _random_prompt()
    current_score = _accuracy(train_examples, current)
    for _ in range(steps):
        scored_neighbors = [(neighbor, _accuracy(train_examples, neighbor)) for neighbor in _neighbors(current)]
        best_neighbor, best_score = max(scored_neighbors, key=lambda item: item[1])
        if best_score <= current_score:
            break
        current, current_score = best_neighbor, best_score
    return current, current_score


def _genetic_search(train_examples: list[dict[str, float | str]], generations: int, population_size: int) -> tuple[tuple[int, ...], float]:
    population = [_random_prompt() for _ in range(population_size)]
    for _ in range(generations):
        scored = sorted(((bits, _accuracy(train_examples, bits)) for bits in population), key=lambda item: item[1], reverse=True)
        parents = [item[0] for item in scored[:2]]
        offspring = parents[:]
        while len(offspring) < population_size:
            left, right = random.sample(parents, 2)
            cut = random.randint(1, len(PROMPT_COMPONENTS) - 1)
            child = tuple(list(left[:cut]) + list(right[cut:]))
            mutate_index = random.randrange(len(PROMPT_COMPONENTS))
            child = tuple((1 - bit) if index == mutate_index and random.random() < 0.35 else bit for index, bit in enumerate(child))
            if not any(child):
                child = (1, 0, 0, 0, 0)
            offspring.append(child)
        population = offspring
    best_bits = max(population, key=lambda bits: _accuracy(train_examples, bits))
    return best_bits, _accuracy(train_examples, best_bits)


def run(*, quick: bool = True) -> ProjectArtifact:
    set_global_seed(42)
    random.seed(42)
    mode = "quick" if quick else "full"
    examples, used_dataset = _load_examples(quick)
    split = 28 if quick else 56
    train_examples = examples[:split]
    eval_examples = examples[split : split + (12 if quick else 24)]

    searches = [
        ("random_search", lambda: _random_search(train_examples, 6 if quick else 16)),
        ("hill_climb", lambda: _hill_climb(train_examples, 8 if quick else 18)),
        ("genetic_search", lambda: _genetic_search(train_examples, 5 if quick else 10, 8 if quick else 12)),
    ]

    results = []
    for variant, searcher in searches:
        (best_bits, train_accuracy), runtime = timed_call(searcher)
        eval_accuracy = _accuracy(eval_examples, best_bits)
        results.append(
            make_result(
                project_id=PROJECT_ID,
                project_name=TITLE,
                requested_dataset=REQUESTED_DATASET,
                used_dataset=used_dataset,
                mode=mode,
                variant=variant,
                algorithm="black_box_search",
                feature_set="prompt_component_subset",
                metric_name="eval_accuracy",
                metric_value=eval_accuracy,
                metric_direction="higher_is_better",
                secondary_metric_name="train_accuracy",
                secondary_metric_value=train_accuracy,
                train_samples=len(train_examples),
                eval_samples=len(eval_examples),
                runtime_sec=runtime,
                notes=_prompt_text(best_bits),
            )
        )

    best = pick_best_result(results)
    spread = max(result.metric_value for result in results) - min(result.metric_value for result in results)
    if spread < 1e-9:
        search_statement = "Random search, hill climbing, and the genetic search tied on the filtered problem slice, so the solver is currently the main bottleneck rather than the optimizer." 
    else:
        search_statement = "The search algorithms separated once the prompt components interacted differently with the weak solver on held-out arithmetic problems."
    findings = [
        f"The best prompt-search policy was {best.variant} with evaluation accuracy {best.metric_value:.3f}." if best else "No successful run was recorded.",
        "This is a true black-box search loop over prompt components: the optimizer only sees scored outcomes from a weak math solver and never gradient information.",
        search_statement,
    ]
    return ProjectArtifact(
        project_id=PROJECT_ID,
        title=TITLE,
        objective="Optimize discrete prompt templates against a black-box arithmetic solver on small GSM8K-style problems.",
        requested_dataset=REQUESTED_DATASET,
        used_dataset=used_dataset,
        mode=mode,
        metric_name="eval_accuracy",
        metric_direction="higher_is_better",
        results=results,
        findings=findings,
        notes="This benchmark is an API-free surrogate for prompt optimization. It keeps the search algorithm real while replacing the external language model with a deterministic weak solver.",
    )


if __name__ == "__main__":
    artifact = run()
    best = pick_best_result(artifact.results)
    print(f"{artifact.title}: {best.variant if best else 'no result'}")