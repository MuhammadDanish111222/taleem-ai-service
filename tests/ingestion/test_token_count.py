from app.services.ingestion.token_count import EmbeddingTokenCounter


class FakeTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [ord(character) for character in text if not character.isspace()]


def test_token_count_is_empty_safe_and_deterministic_without_model_loading():
    counter = EmbeddingTokenCounter(
        "BAAI/bge-base-en-v1.5", "test-revision", tokenizer=FakeTokenizer()
    )

    assert counter.count("") == 0
    assert counter.count("normal text") == 10
    assert counter.count("normal text") == 10
    assert (
        counter.version
        == "huggingface_auto_tokenizer:BAAI/bge-base-en-v1.5@test-revision"
    )


def test_token_count_handles_long_text_deterministically():
    counter = EmbeddingTokenCounter(
        "BAAI/bge-base-en-v1.5", "test-revision", tokenizer=FakeTokenizer()
    )
    text = "token " * 10_000

    assert counter.count(text) == 50_000
    assert counter.count(text) == 50_000
