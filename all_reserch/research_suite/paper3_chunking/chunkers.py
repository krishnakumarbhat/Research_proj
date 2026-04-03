from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from research_suite.common.text import extract_entities, token_count, tokenize_words
from research_suite.paper3_chunking.dataset import ChunkingDocument


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    doc_id: str
    algorithm: str
    text: str
    embedding_text: str
    sent_indices: tuple[int, ...]


def _flat_tokens(document: ChunkingDocument) -> tuple[list[str], list[int]]:
    tokens: list[str] = []
    sentence_map: list[int] = []
    for sent_index, sentence in enumerate(document.sentences):
        sentence_tokens = sentence.split()
        tokens.extend(sentence_tokens)
        sentence_map.extend([sent_index] * len(sentence_tokens))
    return tokens, sentence_map


def fixed_chunks(document: ChunkingDocument, token_limit: int, overlap: int, algorithm: str) -> list[ChunkRecord]:
    if overlap >= token_limit:
        raise ValueError("overlap must be smaller than token_limit")
    tokens, sentence_map = _flat_tokens(document)
    chunks: list[ChunkRecord] = []
    start = 0
    chunk_index = 0
    while start < len(tokens):
        end = min(start + token_limit, len(tokens))
        chunk_tokens = tokens[start:end]
        sent_indices = tuple(sorted(set(sentence_map[start:end])))
        text = " ".join(chunk_tokens)
        chunks.append(
            ChunkRecord(
                chunk_id=f"{document.doc_id}_{algorithm}_{chunk_index:02d}",
                doc_id=document.doc_id,
                algorithm=algorithm,
                text=text,
                embedding_text=text,
                sent_indices=sent_indices,
            )
        )
        if end == len(tokens):
            break
        start = end - overlap
        chunk_index += 1
    return chunks


def sentence_chunks(
    document: ChunkingDocument,
    max_tokens: int,
    algorithm: str,
    include_context: bool = False,
) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    current_sentences: list[str] = []
    current_indices: list[int] = []
    current_tokens = 0
    chunk_index = 0
    for sent_index, sentence in enumerate(document.sentences):
        sentence_tokens = token_count(sentence)
        if current_sentences and current_tokens + sentence_tokens > max_tokens:
            text = " ".join(current_sentences)
            embedding_text = f"{document.global_context} {text}" if include_context else text
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{document.doc_id}_{algorithm}_{chunk_index:02d}",
                    doc_id=document.doc_id,
                    algorithm=algorithm,
                    text=text,
                    embedding_text=embedding_text,
                    sent_indices=tuple(current_indices),
                )
            )
            current_sentences = []
            current_indices = []
            current_tokens = 0
            chunk_index += 1
        current_sentences.append(sentence)
        current_indices.append(sent_index)
        current_tokens += sentence_tokens
    if current_sentences:
        text = " ".join(current_sentences)
        embedding_text = f"{document.global_context} {text}" if include_context else text
        chunks.append(
            ChunkRecord(
                chunk_id=f"{document.doc_id}_{algorithm}_{chunk_index:02d}",
                doc_id=document.doc_id,
                algorithm=algorithm,
                text=text,
                embedding_text=embedding_text,
                sent_indices=tuple(current_indices),
            )
        )
    return chunks


def semantic_chunks(document: ChunkingDocument, max_tokens: int, threshold: float) -> list[ChunkRecord]:
    sentences = list(document.sentences)
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(sentences)
    similarities = cosine_similarity(matrix[:-1], matrix[1:]).diagonal() if len(sentences) > 1 else np.array([])

    chunks: list[ChunkRecord] = []
    current_sentences: list[str] = []
    current_indices: list[int] = []
    current_tokens = 0
    chunk_index = 0
    for sent_index, sentence in enumerate(sentences):
        sentence_tokens = token_count(sentence)
        should_split = False
        if current_sentences and current_tokens + sentence_tokens > max_tokens:
            should_split = True
        elif current_sentences and sent_index > 0 and float(similarities[sent_index - 1]) < threshold:
            should_split = True
        if should_split:
            text = " ".join(current_sentences)
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{document.doc_id}_semantic_{chunk_index:02d}",
                    doc_id=document.doc_id,
                    algorithm="semantic_boundary",
                    text=text,
                    embedding_text=text,
                    sent_indices=tuple(current_indices),
                )
            )
            current_sentences = []
            current_indices = []
            current_tokens = 0
            chunk_index += 1
        current_sentences.append(sentence)
        current_indices.append(sent_index)
        current_tokens += sentence_tokens
    if current_sentences:
        text = " ".join(current_sentences)
        chunks.append(
            ChunkRecord(
                chunk_id=f"{document.doc_id}_semantic_{chunk_index:02d}",
                doc_id=document.doc_id,
                algorithm="semantic_boundary",
                text=text,
                embedding_text=text,
                sent_indices=tuple(current_indices),
            )
        )
    return chunks


def graph_context_chunks(document: ChunkingDocument, hops: int) -> list[ChunkRecord]:
    adjacency: dict[int, set[int]] = defaultdict(set)
    sentence_entities = [set(extract_entities(sentence)) for sentence in document.sentences]
    for index in range(len(document.sentences) - 1):
        adjacency[index].add(index + 1)
        adjacency[index + 1].add(index)
    for left in range(len(document.sentences)):
        for right in range(left + 1, len(document.sentences)):
            if sentence_entities[left] & sentence_entities[right]:
                adjacency[left].add(right)
                adjacency[right].add(left)
    pronouns = {"she", "he", "that", "it"}
    for index, sentence in enumerate(document.sentences[1:], start=1):
        sentence_tokens = tokenize_words(sentence)
        if sentence_tokens and sentence_tokens[0] in pronouns:
            adjacency[index].add(index - 1)
            adjacency[index - 1].add(index)

    seen_signatures: set[tuple[int, ...]] = set()
    chunks: list[ChunkRecord] = []
    for anchor in range(len(document.sentences)):
        frontier = {anchor}
        visited = {anchor}
        for _ in range(hops):
            next_frontier = set()
            for node in frontier:
                next_frontier.update(adjacency[node])
            next_frontier -= visited
            visited.update(next_frontier)
            frontier = next_frontier
            if not frontier:
                break
        signature = tuple(sorted(visited))
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        text = " ".join(document.sentences[index] for index in signature)
        chunks.append(
            ChunkRecord(
                chunk_id=f"{document.doc_id}_graph_{anchor:02d}",
                doc_id=document.doc_id,
                algorithm="graph_rag",
                text=text,
                embedding_text=text,
                sent_indices=signature,
            )
        )
    return chunks


def build_chunks(documents: list[ChunkingDocument], algorithm: str, params: dict[str, object]) -> list[ChunkRecord]:
    all_chunks: list[ChunkRecord] = []
    for document in documents:
        if algorithm == "fixed_no_overlap":
            all_chunks.extend(
                fixed_chunks(
                    document=document,
                    token_limit=int(params["token_limit"]),
                    overlap=0,
                    algorithm=algorithm,
                )
            )
        elif algorithm == "fixed_overlap":
            all_chunks.extend(
                fixed_chunks(
                    document=document,
                    token_limit=int(params["token_limit"]),
                    overlap=int(params["overlap"]),
                    algorithm=algorithm,
                )
            )
        elif algorithm == "sentence_boundary":
            all_chunks.extend(
                sentence_chunks(
                    document=document,
                    max_tokens=int(params["max_tokens"]),
                    algorithm=algorithm,
                )
            )
        elif algorithm == "semantic_boundary":
            all_chunks.extend(
                semantic_chunks(
                    document=document,
                    max_tokens=int(params["max_tokens"]),
                    threshold=float(params["threshold"]),
                )
            )
        elif algorithm == "late_chunking_proxy":
            all_chunks.extend(
                sentence_chunks(
                    document=document,
                    max_tokens=int(params["max_tokens"]),
                    algorithm=algorithm,
                    include_context=True,
                )
            )
        elif algorithm == "graph_rag":
            all_chunks.extend(graph_context_chunks(document=document, hops=int(params["hops"])))
        else:
            raise ValueError(f"Unsupported chunking algorithm: {algorithm}")
    return all_chunks
