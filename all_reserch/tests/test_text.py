from research_suite.common.text import extract_entities, split_sentences, token_count, tokenize_words


def test_tokenize_words_basic() -> None:
    assert tokenize_words("Hello World") == ["hello", "world"]


def test_tokenize_words_with_punctuation() -> None:
    tokens = tokenize_words("Dr. Chen's report, v2.0!")
    assert "chen's" in tokens
    assert "report" in tokens


def test_token_count() -> None:
    assert token_count("one two three") == 3
    assert token_count("") == 0


def test_split_sentences() -> None:
    text = "First sentence. Second sentence! Third sentence?"
    sentences = split_sentences(text)
    assert len(sentences) == 3
    assert sentences[0] == "First sentence."


def test_split_sentences_single() -> None:
    assert split_sentences("No period") == ["No period"]


def test_extract_entities() -> None:
    entities = extract_entities("Elena Brooks works at Nova Lab with Marcus Lee.")
    assert "Elena Brooks" in entities
    assert "Nova Lab" in entities
    assert "Marcus Lee" in entities


def test_extract_entities_no_duplicates() -> None:
    entities = extract_entities("Nova Lab is great. Nova Lab is excellent.")
    assert entities.count("Nova Lab") == 1
