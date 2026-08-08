from app.services.ingestion.token_count import EmbeddingTokenCounter


def test_token_count_is_empty_safe_and_deterministic_without_model_loading():
    counter = EmbeddingTokenCounter("voyage-4-lite", "test-revision")

    assert counter.count("") == 0
    assert counter.count("normal text") == 2
    assert counter.count("normal text") == 2
    assert (
        counter.version
        == "voyage_token_estimator:voyage-4-lite@test-revision"
    )


def test_token_count_handles_long_text_deterministically():
    counter = EmbeddingTokenCounter("voyage-4-lite", "test-revision")
    text = "token " * 10_000

    assert counter.count(text) == 10_000
    assert counter.count(text) == 10_000
