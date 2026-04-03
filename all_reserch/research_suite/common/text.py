from __future__ import annotations

import re


WORD_RE = re.compile(r"[A-Za-z0-9']+")
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")
ENTITY_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")


def tokenize_words(text: str) -> list[str]:
    return [match.group(0).lower() for match in WORD_RE.finditer(text)]


def token_count(text: str) -> int:
    return len(tokenize_words(text))


def split_sentences(text: str) -> list[str]:
    sentences = [sentence.strip() for sentence in SENTENCE_BOUNDARY_RE.split(text.strip()) if sentence.strip()]
    return sentences if sentences else [text.strip()]


def extract_entities(text: str) -> list[str]:
    unique_entities: list[str] = []
    seen: set[str] = set()
    for entity in ENTITY_RE.findall(text):
        normalized = entity.strip()
        if normalized and normalized not in seen:
            unique_entities.append(normalized)
            seen.add(normalized)
    return unique_entities
