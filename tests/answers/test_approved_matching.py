from app.services.answers.approved_matching import lexical_score, select_lexical_match


def test_lexical_match_accepts_safe_definition_rewording():
    assert lexical_score("Define chemistry.", "What is chemistry?") == 1.0
    assert (
        select_lexical_match(
            "Define chemistry.", [("revision-1", "What is chemistry?", "chapter-1")]
        )
        == "revision-1"
    )


def test_lexical_match_rejects_shared_word_with_different_intent():
    assert (
        select_lexical_match(
            "What are the elements of a chemical equation?",
            [("revision-1", "What is an element?", "chapter-1")],
        )
        is None
    )


def test_lexical_match_rejects_ambiguous_top_candidates():
    assert (
        select_lexical_match(
            "Define chemistry.",
            [
                ("revision-1", "What is chemistry?", "chapter-1"),
                ("revision-2", "Explain chemistry.", "chapter-1"),
            ],
        )
        is None
    )
