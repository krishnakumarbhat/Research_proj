from research_suite.common.vectorizers import (
    CharTfidfFeature,
    LsaFeature,
    WordTfidfFeature,
    available_feature_models,
    build_feature_model,
)


SAMPLE_TEXTS = [
    "The robot uses a lidar guidance stack for navigation.",
    "Elena Brooks founded Atlas Rover at Nova Lab.",
    "A graphene pack powers the drone during long flights.",
    "The quarterly memo notes climate package updates.",
    "Marcus Lee leads the Nimbus Drone project.",
]


def test_available_feature_models() -> None:
    models = available_feature_models()
    assert "word_tfidf" in models
    assert "char_tfidf" in models
    assert "lsa" in models


def test_word_tfidf_fit_transform() -> None:
    model = WordTfidfFeature()
    model.fit(SAMPLE_TEXTS)
    matrix = model.transform(SAMPLE_TEXTS)
    assert matrix.shape[0] == len(SAMPLE_TEXTS)


def test_char_tfidf_fit_transform() -> None:
    model = CharTfidfFeature()
    model.fit(SAMPLE_TEXTS)
    matrix = model.transform(SAMPLE_TEXTS)
    assert matrix.shape[0] == len(SAMPLE_TEXTS)


def test_lsa_fit_transform() -> None:
    model = LsaFeature()
    model.fit(SAMPLE_TEXTS)
    matrix = model.transform(SAMPLE_TEXTS)
    assert matrix.shape[0] == len(SAMPLE_TEXTS)
    assert matrix.shape[1] <= 32


def test_similarity_shape() -> None:
    model = build_feature_model("word_tfidf")
    model.fit(SAMPLE_TEXTS)
    doc_mat = model.transform(SAMPLE_TEXTS)
    query_vec = model.transform(["robot navigation"])
    sim = model.similarity(query_vec, doc_mat)
    assert sim.shape == (len(SAMPLE_TEXTS),)
    assert all(s >= -0.01 for s in sim)


def test_build_feature_model_invalid() -> None:
    try:
        build_feature_model("nonexistent")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
