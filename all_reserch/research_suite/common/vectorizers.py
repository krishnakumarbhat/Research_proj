from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import Normalizer


MatrixLike = np.ndarray | sparse.spmatrix


class FeatureModel(Protocol):
    name: str

    def fit(self, texts: list[str]) -> "FeatureModel":
        ...

    def transform(self, texts: list[str]) -> MatrixLike:
        ...

    def similarity(self, query_vector: MatrixLike, doc_matrix: MatrixLike) -> np.ndarray:
        ...


@dataclass
class WordTfidfFeature:
    name: str = "word_tfidf"

    def __post_init__(self) -> None:
        self.vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), norm="l2")

    def fit(self, texts: list[str]) -> "WordTfidfFeature":
        self.vectorizer.fit(texts)
        return self

    def transform(self, texts: list[str]) -> MatrixLike:
        return self.vectorizer.transform(texts)

    def similarity(self, query_vector: MatrixLike, doc_matrix: MatrixLike) -> np.ndarray:
        return cosine_similarity(query_vector, doc_matrix).ravel()


@dataclass
class CharTfidfFeature:
    name: str = "char_tfidf"

    def __post_init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            analyzer="char_wb",
            ngram_range=(3, 5),
            norm="l2",
        )

    def fit(self, texts: list[str]) -> "CharTfidfFeature":
        self.vectorizer.fit(texts)
        return self

    def transform(self, texts: list[str]) -> MatrixLike:
        return self.vectorizer.transform(texts)

    def similarity(self, query_vector: MatrixLike, doc_matrix: MatrixLike) -> np.ndarray:
        return cosine_similarity(query_vector, doc_matrix).ravel()


@dataclass
class LsaFeature:
    components: int = 32
    random_state: int = 42
    name: str = "lsa"

    def __post_init__(self) -> None:
        self.vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), norm="l2")
        self.svd: TruncatedSVD | None = None
        self.normalizer = Normalizer(copy=False)
        self.fallback_dense = False

    def fit(self, texts: list[str]) -> "LsaFeature":
        tfidf = self.vectorizer.fit_transform(texts)
        if tfidf.shape[0] < 3 or tfidf.shape[1] < 3:
            self.fallback_dense = True
            return self

        n_components = max(2, min(self.components, tfidf.shape[0] - 1, tfidf.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=n_components, random_state=self.random_state)
        self.normalizer.fit(self.svd.fit_transform(tfidf))
        return self

    def transform(self, texts: list[str]) -> MatrixLike:
        tfidf = self.vectorizer.transform(texts)
        if self.fallback_dense or self.svd is None:
            dense = tfidf.toarray()
            norms = np.linalg.norm(dense, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            return dense / norms
        dense = self.svd.transform(tfidf)
        return self.normalizer.transform(dense)

    def similarity(self, query_vector: MatrixLike, doc_matrix: MatrixLike) -> np.ndarray:
        return cosine_similarity(query_vector, doc_matrix).ravel()


def build_feature_model(name: str) -> FeatureModel:
    if name == "word_tfidf":
        return WordTfidfFeature()
    if name == "char_tfidf":
        return CharTfidfFeature()
    if name == "lsa":
        return LsaFeature()
    raise ValueError(f"Unsupported feature model: {name}")


def available_feature_models() -> list[str]:
    return ["word_tfidf", "char_tfidf", "lsa"]