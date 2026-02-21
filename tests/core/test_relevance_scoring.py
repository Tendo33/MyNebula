from nebula.core.relevance import (
    RelevanceComponents,
    build_relevance_components,
    calculate_relevance_score,
    collect_relevance_reasons,
)


def test_relevance_weighted_score_prefers_semantic():
    a = RelevanceComponents(
        semantic=0.9,
        tag_overlap=0.2,
        same_star_list=0.0,
        same_language=0.0,
    )
    b = RelevanceComponents(
        semantic=0.4,
        tag_overlap=0.9,
        same_star_list=1.0,
        same_language=1.0,
    )

    assert calculate_relevance_score(a) > calculate_relevance_score(b)


def test_build_relevance_components_normalizes_booleans_and_tags():
    components = build_relevance_components(
        semantic_similarity=0.88,
        anchor_tags=["Agent", "RAG"],
        candidate_tags=["agent"],
        same_star_list=True,
        same_language=False,
    )

    assert components.semantic == 0.88
    assert components.same_star_list == 1.0
    assert components.same_language == 0.0
    assert 0.0 <= components.tag_overlap <= 1.0


def test_collect_reasons_contains_semantic_label():
    reasons = collect_relevance_reasons(
        RelevanceComponents(
            semantic=0.8,
            tag_overlap=0.5,
            same_star_list=1.0,
            same_language=1.0,
        )
    )
    assert any(reason.startswith("semantic:") for reason in reasons)
    assert "same-star-list" in reasons
