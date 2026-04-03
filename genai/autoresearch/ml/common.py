from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import random
import re
from time import perf_counter
from typing import Any, Iterable, Literal, Sequence, cast

import numpy as np
from rouge_score import rouge_scorer
from sklearn.base import clone
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC


WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_']+")
AnalyzerKind = Literal["word", "char", "char_wb"]
ROOT = Path(__file__).resolve().parent
DATA_CACHE_DIR = ROOT / "data_cache"
RESULTS_DIR = ROOT / "results"


@dataclass
class ExperimentResult:
    project_id: str
    project_name: str
    requested_dataset: str
    used_dataset: str
    mode: str
    variant: str
    algorithm: str
    feature_set: str
    metric_name: str
    metric_value: float
    metric_direction: str
    secondary_metric_name: str = ""
    secondary_metric_value: float | None = None
    train_samples: int = 0
    eval_samples: int = 0
    runtime_sec: float = 0.0
    status: str = "ok"
    notes: str = ""

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProjectArtifact:
    project_id: str
    title: str
    objective: str
    requested_dataset: str
    used_dataset: str
    mode: str
    metric_name: str
    metric_direction: str
    results: list[ExperimentResult]
    findings: list[str]
    notes: str = ""


def ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return " ".join(flatten_text(item) for item in value.values() if flatten_text(item))
    if isinstance(value, (list, tuple)):
        return " ".join(flatten_text(item) for item in value if flatten_text(item))
    return str(value).strip()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_directories() -> None:
    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def timed_call(function: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    start = perf_counter()
    result = function(*args, **kwargs)
    return result, perf_counter() - start


def load_hf_dataset_records(
    path: str,
    *,
    name: str | None = None,
    split: str = "train[:128]",
) -> list[dict[str, Any]] | None:
    try:
        from datasets import load_dataset

        dataset = load_dataset(path, name=name, split=split)
    except Exception:
        return None
    return [dict(record) for record in dataset]


def simple_tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in WORD_RE.finditer(text)]


def sentence_split(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [part.strip() for part in parts if part and part.strip()]


def concat_fields(*parts: str) -> str:
    return "\n".join(part.strip() for part in parts if part and part.strip())


def format_metric(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_table(rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> str:
    if not rows:
        header = "| " + " | ".join(columns) + " |"
        divider = "| " + " | ".join(["---"] * len(columns)) + " |"
        return "\n".join([header, divider])

    def escape(cell: Any) -> str:
        return format_metric(cell).replace("|", "\\|")

    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(escape(row.get(column, "")) for column in columns) + " |")
    return "\n".join([header, divider, *body])


def build_text_pipeline(
    algorithm: str,
    *,
    analyzer: AnalyzerKind = "word",
    ngram_range: tuple[int, int] = (1, 2),
    max_features: int = 5000,
) -> Pipeline:
    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        lowercase=True,
        ngram_range=ngram_range,
        max_features=max_features,
    )

    if algorithm == "logreg":
        estimator = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        )
    elif algorithm == "linear_svm":
        estimator = LinearSVC(class_weight="balanced", random_state=42)
    elif algorithm == "nb":
        estimator = MultinomialNB()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    return Pipeline([("vectorizer", vectorizer), ("estimator", estimator)])


def classification_metrics(y_true: Sequence[Any], y_pred: Sequence[Any]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def evaluate_text_classifier(
    train_texts: Sequence[str],
    train_labels: Sequence[Any],
    eval_texts: Sequence[str],
    eval_labels: Sequence[Any],
    *,
    algorithm: str,
    analyzer: AnalyzerKind = "word",
    ngram_range: tuple[int, int] = (1, 2),
    max_features: int = 5000,
) -> dict[str, float]:
    pipeline = build_text_pipeline(
        algorithm,
        analyzer=analyzer,
        ngram_range=ngram_range,
        max_features=max_features,
    )
    pipeline.fit(train_texts, train_labels)
    predictions = pipeline.predict(eval_texts)
    return classification_metrics(eval_labels, list(np.asarray(predictions).reshape(-1)))


def evaluate_numeric_classifier(
    estimator: Any,
    train_features: np.ndarray,
    train_labels: Sequence[Any],
    eval_features: np.ndarray,
    eval_labels: Sequence[Any],
) -> dict[str, float]:
    model = clone(estimator)
    model.fit(train_features, train_labels)
    predictions = model.predict(eval_features)
    return classification_metrics(eval_labels, predictions)


def default_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=250,
        max_depth=8,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
    )


def token_overlap_f1(prediction: str, reference: str) -> float:
    pred_tokens = simple_tokenize(prediction)
    ref_tokens = simple_tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counts: dict[str, int] = {}
    ref_counts: dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    overlap = sum(min(pred_counts.get(token, 0), ref_counts.get(token, 0)) for token in ref_counts)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def verilog_validity_score(text: str) -> float:
    checks = [
        bool(re.search(r"\bmodule\b", text)),
        text.count("(") == text.count(")"),
        bool(re.search(r"module\s+[A-Za-z_][A-Za-z0-9_]*\s*\(", text)),
        ";" in text,
        "endmodule" in text,
    ]
    return sum(1 for check in checks if check) / len(checks)


def chunk_text(
    text: str,
    *,
    mode: str,
    chunk_size: int = 45,
    overlap: int = 12,
) -> list[str]:
    if mode == "sentence":
        return sentence_split(text)

    words = text.split()
    if not words:
        return []

    if mode == "fixed":
        step = max(1, chunk_size)
    elif mode == "overlap":
        step = max(1, chunk_size - overlap)
    else:
        raise ValueError(f"Unsupported chunking mode: {mode}")

    chunks = []
    for start in range(0, len(words), step):
        piece = words[start : start + chunk_size]
        if piece:
            chunks.append(" ".join(piece))
        if start + chunk_size >= len(words):
            break
    return chunks


def tfidf_rank_documents(
    queries: Sequence[str],
    documents: Sequence[str],
    *,
    top_k: int = 3,
    analyzer: AnalyzerKind = "word",
    ngram_range: tuple[int, int] = (1, 2),
    max_features: int = 8000,
    use_svd: bool = False,
) -> list[list[int]]:
    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        lowercase=True,
        ngram_range=ngram_range,
        max_features=max_features,
    )
    doc_matrix = vectorizer.fit_transform(documents)
    query_matrix = vectorizer.transform(queries)

    if use_svd and min(doc_matrix.shape) > 2:
        n_components = max(2, min(64, doc_matrix.shape[0] - 1, doc_matrix.shape[1] - 1))
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        normalizer = Normalizer(copy=False)
        doc_matrix = normalizer.fit_transform(svd.fit_transform(doc_matrix))
        query_matrix = normalizer.transform(svd.transform(query_matrix))

    scores = query_matrix @ getattr(doc_matrix, "transpose")()
    if hasattr(scores, "toarray"):
        scores = cast(Any, scores).toarray()

    rankings = []
    for row in np.asarray(scores):
        rankings.append(list(np.argsort(-row)[:top_k]))
    return rankings


def hashing_rank_documents(
    queries: Sequence[str],
    documents: Sequence[str],
    *,
    top_k: int = 3,
    n_features: int = 2**15,
) -> list[list[int]]:
    vectorizer = HashingVectorizer(
        n_features=n_features,
        alternate_sign=False,
        norm="l2",
        ngram_range=(1, 2),
    )
    doc_matrix = vectorizer.transform(documents)
    query_matrix = vectorizer.transform(queries)
    scores = query_matrix @ getattr(doc_matrix, "transpose")()
    if hasattr(scores, "toarray"):
        scores = cast(Any, scores).toarray()
    rankings = []
    for row in np.asarray(scores):
        rankings.append(list(np.argsort(-row)[:top_k]))
    return rankings


def bm25_rank_documents(
    queries: Sequence[str],
    documents: Sequence[str],
    *,
    top_k: int = 3,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[list[int]]:
    tokenized_docs = [simple_tokenize(document) for document in documents]
    doc_lengths = np.array([len(tokens) for tokens in tokenized_docs], dtype=np.float64)
    avg_doc_length = float(doc_lengths.mean()) if len(doc_lengths) else 1.0

    document_frequency: dict[str, int] = {}
    for tokens in tokenized_docs:
        for token in set(tokens):
            document_frequency[token] = document_frequency.get(token, 0) + 1

    rankings = []
    for query in queries:
        query_tokens = simple_tokenize(query)
        row_scores = []
        for index, doc_tokens in enumerate(tokenized_docs):
            tf_counts: dict[str, int] = {}
            for token in doc_tokens:
                tf_counts[token] = tf_counts.get(token, 0) + 1

            score = 0.0
            for token in query_tokens:
                if token not in tf_counts:
                    continue
                df = document_frequency.get(token, 0)
                idf = math.log(1 + (len(documents) - df + 0.5) / (df + 0.5))
                tf = tf_counts[token]
                denom = tf + k1 * (1 - b + b * doc_lengths[index] / max(avg_doc_length, 1e-8))
                score += idf * (tf * (k1 + 1)) / max(denom, 1e-8)
            row_scores.append(score)
        rankings.append(list(np.argsort(-np.asarray(row_scores))[:top_k]))
    return rankings


def retrieval_metrics(rankings: Sequence[Sequence[int]], gold_indices: Sequence[int], *, top_k: int = 3) -> dict[str, float]:
    hits_at_1 = 0
    hits_at_k = 0
    reciprocal_ranks = []
    for ranking, gold_index in zip(rankings, gold_indices):
        ranking_list = list(ranking)
        if ranking_list and ranking_list[0] == gold_index:
            hits_at_1 += 1
        if gold_index in ranking_list[:top_k]:
            hits_at_k += 1
            reciprocal_ranks.append(1.0 / (ranking_list.index(gold_index) + 1))
        else:
            reciprocal_ranks.append(0.0)
    total = max(1, len(gold_indices))
    return {
        "recall_at_1": hits_at_1 / total,
        f"recall_at_{top_k}": hits_at_k / total,
        "mrr": float(np.mean(reciprocal_ranks)),
    }


def select_sentences_tfidf(text: str, query: str, *, max_sentences: int = 3) -> str:
    sentences = sentence_split(text)
    if len(sentences) <= max_sentences:
        return " ".join(sentences)
    probe = query.strip() or sentences[0]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=4000)
    sentence_matrix = vectorizer.fit_transform(sentences)
    query_vector = vectorizer.transform([probe])
    scores = cast(Any, query_vector @ getattr(sentence_matrix, "transpose")()).toarray()[0]
    best_indices = sorted(np.argsort(-scores)[:max_sentences])
    return " ".join(sentences[index] for index in best_indices)


def rouge_l_f1(reference: str, prediction: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return float(scorer.score(reference, prediction)["rougeL"].fmeasure)


def best_overlap_sentence(question: str, contexts: Iterable[str]) -> str:
    best_sentence = ""
    best_score = -1.0
    for context in contexts:
        for sentence in sentence_split(context):
            score = token_overlap_f1(sentence, question)
            if score > best_score:
                best_score = score
                best_sentence = sentence
    return best_sentence


def pick_best_result(results: Sequence[ExperimentResult]) -> ExperimentResult | None:
    valid = [result for result in results if result.status == "ok"]
    if not valid:
        return None
    direction = valid[0].metric_direction
    if direction == "higher_is_better":
        return max(valid, key=lambda result: result.metric_value)
    return min(valid, key=lambda result: result.metric_value)


def make_result(
    *,
    project_id: str,
    project_name: str,
    requested_dataset: str,
    used_dataset: str,
    mode: str,
    variant: str,
    algorithm: str,
    feature_set: str,
    metric_name: str,
    metric_value: float,
    metric_direction: str,
    secondary_metric_name: str = "",
    secondary_metric_value: float | None = None,
    train_samples: int = 0,
    eval_samples: int = 0,
    runtime_sec: float = 0.0,
    status: str = "ok",
    notes: str = "",
) -> ExperimentResult:
    return ExperimentResult(
        project_id=project_id,
        project_name=project_name,
        requested_dataset=requested_dataset,
        used_dataset=used_dataset,
        mode=mode,
        variant=variant,
        algorithm=algorithm,
        feature_set=feature_set,
        metric_name=metric_name,
        metric_value=metric_value,
        metric_direction=metric_direction,
        secondary_metric_name=secondary_metric_name,
        secondary_metric_value=secondary_metric_value,
        train_samples=train_samples,
        eval_samples=eval_samples,
        runtime_sec=runtime_sec,
        status=status,
        notes=notes,
    )
