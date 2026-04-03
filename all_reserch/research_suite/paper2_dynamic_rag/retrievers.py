from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from time import perf_counter

import numpy as np
from scipy import sparse

from research_suite.common.text import tokenize_words
from research_suite.common.vectorizers import FeatureModel, build_feature_model
from research_suite.paper2_dynamic_rag.dataset import RagDocument


@dataclass
class RetrievalResult:
    ranked_ids: list[str]
    latency_ms: float
    candidates_scored: int
    cache_hit: int
    query_vector: object


class BaseRetriever:
    def __init__(self, feature_name: str, top_k: int = 3) -> None:
        self.feature_name = feature_name
        self.top_k = top_k
        self.feature_model: FeatureModel = build_feature_model(feature_name)
        self.cache_matrix = None
        self.cache_rankings: list[list[str]] = []

    def fit(self, documents: list[RagDocument]) -> None:
        self.documents = documents
        self.doc_ids = [document.doc_id for document in documents]
        self.doc_texts = [document.text for document in documents]
        self.doc_index = {doc_id: index for index, doc_id in enumerate(self.doc_ids)}
        self.feature_model.fit(self.doc_texts)
        self.doc_matrix = self.feature_model.transform(self.doc_texts)
        self.doc_tokens = [set(tokenize_words(text)) for text in self.doc_texts]
        self.inverted_index: dict[str, set[int]] = defaultdict(set)
        for index, tokens in enumerate(self.doc_tokens):
            for token in tokens:
                self.inverted_index[token].add(index)
        self.access_counts = np.zeros(len(documents), dtype=float)
        self.last_access = np.zeros(len(documents), dtype=float)
        self.graph: dict[int, Counter[int]] = defaultdict(Counter)
        self.cache_matrix = None
        self.cache_rankings = []

    def retrieve(self, query_text: str, step: int) -> RetrievalResult:
        started = perf_counter()
        query_vector = self.feature_model.transform([query_text])
        scores = self.feature_model.similarity(query_vector, self.doc_matrix)
        ranked_indices = np.argsort(scores)[::-1]
        ranked_ids = [self.doc_ids[index] for index in ranked_indices[: self.top_k]]
        latency_ms = (perf_counter() - started) * 1000.0
        return RetrievalResult(
            ranked_ids=ranked_ids,
            latency_ms=latency_ms,
            candidates_scored=len(self.doc_ids),
            cache_hit=0,
            query_vector=query_vector,
        )

    def update_state(
        self,
        relevant_ids: tuple[str, ...],
        step: int,
        query_vector: object,
        ranked_ids: list[str],
    ) -> None:
        del query_vector
        del ranked_ids
        relevant_indices = [self.doc_index[doc_id] for doc_id in relevant_ids if doc_id in self.doc_index]
        for index in relevant_indices:
            self.access_counts[index] += 1.0
            self.last_access[index] = float(step)
        for left, right in combinations(relevant_indices, 2):
            self.graph[left][right] += 1
            self.graph[right][left] += 1

    def _append_cache(self, query_vector: object, ranked_ids: list[str]) -> None:
        if self.cache_matrix is None:
            self.cache_matrix = query_vector
        elif sparse.issparse(query_vector):
            self.cache_matrix = sparse.vstack([self.cache_matrix, query_vector])
        else:
            self.cache_matrix = np.vstack([np.asarray(self.cache_matrix), np.asarray(query_vector)])
        self.cache_rankings.append(ranked_ids)

    def _lexical_candidates(self, query_text: str) -> set[int]:
        candidate_indices: set[int] = set()
        for token in tokenize_words(query_text):
            candidate_indices.update(self.inverted_index.get(token, set()))
        return candidate_indices

    def _effective_counts(self, step: int, decay_lambda: float = 0.0) -> np.ndarray:
        if decay_lambda <= 0:
            return self.access_counts.copy()
        delta = np.maximum(0.0, float(step) - self.last_access)
        return self.access_counts * np.exp(-decay_lambda * delta)


class SemanticCacheRetriever(BaseRetriever):
    def __init__(self, feature_name: str, threshold: float, top_k: int = 3) -> None:
        super().__init__(feature_name=feature_name, top_k=top_k)
        self.threshold = threshold

    def retrieve(self, query_text: str, step: int) -> RetrievalResult:
        del step
        started = perf_counter()
        query_vector = self.feature_model.transform([query_text])
        if self.cache_matrix is not None:
            similarities = self.feature_model.similarity(query_vector, self.cache_matrix)
            best_index = int(np.argmax(similarities))
            if float(similarities[best_index]) >= self.threshold:
                latency_ms = (perf_counter() - started) * 1000.0
                return RetrievalResult(
                    ranked_ids=self.cache_rankings[best_index][: self.top_k],
                    latency_ms=latency_ms,
                    candidates_scored=0,
                    cache_hit=1,
                    query_vector=query_vector,
                )
        scores = self.feature_model.similarity(query_vector, self.doc_matrix)
        ranked_indices = np.argsort(scores)[::-1]
        ranked_ids = [self.doc_ids[index] for index in ranked_indices[: self.top_k]]
        latency_ms = (perf_counter() - started) * 1000.0
        return RetrievalResult(
            ranked_ids=ranked_ids,
            latency_ms=latency_ms,
            candidates_scored=len(self.doc_ids),
            cache_hit=0,
            query_vector=query_vector,
        )

    def update_state(
        self,
        relevant_ids: tuple[str, ...],
        step: int,
        query_vector: object,
        ranked_ids: list[str],
    ) -> None:
        super().update_state(relevant_ids=relevant_ids, step=step, query_vector=query_vector, ranked_ids=ranked_ids)
        self._append_cache(query_vector=query_vector, ranked_ids=ranked_ids)


class FrequencyWeightedRetriever(BaseRetriever):
    def __init__(
        self,
        feature_name: str,
        alpha: float,
        beta: float,
        hot_k: int,
        similarity_floor: float,
        decay_lambda: float = 0.0,
        top_k: int = 3,
    ) -> None:
        super().__init__(feature_name=feature_name, top_k=top_k)
        self.alpha = alpha
        self.beta = beta
        self.hot_k = hot_k
        self.similarity_floor = similarity_floor
        self.decay_lambda = decay_lambda

    def _candidate_indices(self, query_text: str, step: int) -> list[int]:
        lexical = self._lexical_candidates(query_text)
        effective_counts = self._effective_counts(step=step, decay_lambda=self.decay_lambda)
        hot_indices = np.argsort(effective_counts)[::-1][: self.hot_k]
        candidate_indices = set(index for index in hot_indices if effective_counts[index] > 0.0)
        candidate_indices.update(lexical)
        if len(candidate_indices) < min(8, len(self.doc_ids) // 4):
            return list(range(len(self.doc_ids)))
        return sorted(candidate_indices)

    def _score_candidates(self, query_vector: object, candidate_indices: list[int], step: int) -> tuple[np.ndarray, np.ndarray]:
        candidate_matrix = self.doc_matrix[candidate_indices]
        similarities = self.feature_model.similarity(query_vector, candidate_matrix)
        effective_counts = self._effective_counts(step=step, decay_lambda=self.decay_lambda)[candidate_indices]
        scores = (self.alpha * similarities) + (self.beta * np.log1p(effective_counts))
        return scores, similarities

    def retrieve(self, query_text: str, step: int) -> RetrievalResult:
        started = perf_counter()
        query_vector = self.feature_model.transform([query_text])
        candidate_indices = self._candidate_indices(query_text=query_text, step=step)
        scores, similarities = self._score_candidates(query_vector, candidate_indices, step=step)
        if float(np.max(similarities)) < self.similarity_floor and len(candidate_indices) < len(self.doc_ids):
            candidate_indices = list(range(len(self.doc_ids)))
            scores, similarities = self._score_candidates(query_vector, candidate_indices, step=step)
        ranked_local = np.argsort(scores)[::-1]
        ranked_ids = [self.doc_ids[candidate_indices[index]] for index in ranked_local[: self.top_k]]
        latency_ms = (perf_counter() - started) * 1000.0
        return RetrievalResult(
            ranked_ids=ranked_ids,
            latency_ms=latency_ms,
            candidates_scored=len(candidate_indices),
            cache_hit=0,
            query_vector=query_vector,
        )


class GraphRetriever(FrequencyWeightedRetriever):
    def __init__(
        self,
        feature_name: str,
        alpha: float,
        beta: float,
        hot_k: int,
        similarity_floor: float,
        gamma: float,
        decay_lambda: float = 0.0,
        top_k: int = 3,
    ) -> None:
        super().__init__(
            feature_name=feature_name,
            alpha=alpha,
            beta=beta,
            hot_k=hot_k,
            similarity_floor=similarity_floor,
            decay_lambda=decay_lambda,
            top_k=top_k,
        )
        self.gamma = gamma

    def retrieve(self, query_text: str, step: int) -> RetrievalResult:
        started = perf_counter()
        query_vector = self.feature_model.transform([query_text])
        candidate_indices = self._candidate_indices(query_text=query_text, step=step)
        base_scores, similarities = self._score_candidates(query_vector, candidate_indices, step=step)
        preliminary_ranked = np.argsort(base_scores)[::-1][:2]
        neighbor_bonus: defaultdict[int, float] = defaultdict(float)
        for local_index in preliminary_ranked:
            anchor = candidate_indices[int(local_index)]
            total_weight = sum(self.graph[anchor].values()) or 1
            for neighbor, edge_weight in self.graph[anchor].items():
                neighbor_bonus[neighbor] = max(neighbor_bonus[neighbor], edge_weight / total_weight)
        if neighbor_bonus:
            expanded = sorted(set(candidate_indices) | set(neighbor_bonus.keys()))
            candidate_indices = expanded
            base_scores, similarities = self._score_candidates(query_vector, candidate_indices, step=step)
            bonuses = np.array([self.gamma * neighbor_bonus.get(index, 0.0) for index in candidate_indices])
            scores = base_scores + bonuses
        else:
            scores = base_scores
        if float(np.max(similarities)) < self.similarity_floor and len(candidate_indices) < len(self.doc_ids):
            candidate_indices = list(range(len(self.doc_ids)))
            base_scores, _ = self._score_candidates(query_vector, candidate_indices, step=step)
            bonuses = np.array([self.gamma * neighbor_bonus.get(index, 0.0) for index in candidate_indices])
            scores = base_scores + bonuses
        ranked_local = np.argsort(scores)[::-1]
        ranked_ids = [self.doc_ids[candidate_indices[index]] for index in ranked_local[: self.top_k]]
        latency_ms = (perf_counter() - started) * 1000.0
        return RetrievalResult(
            ranked_ids=ranked_ids,
            latency_ms=latency_ms,
            candidates_scored=len(candidate_indices),
            cache_hit=0,
            query_vector=query_vector,
        )
