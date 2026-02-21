from nebula.core.tag_normalization import (
    merge_and_normalize_tag_sources,
    normalize_tag_list,
    normalize_tag_token,
    weighted_tag_overlap_score,
)


def test_normalize_tag_token_handles_case_and_punctuation():
    assert normalize_tag_token(" Agent_Memory ") == "agent-memory"
    assert normalize_tag_token("RAG Memory") in {"rag-memory", "agent-memory", "rag"}


def test_normalize_tag_list_deduplicates_and_splits_tokens():
    tags = normalize_tag_list(["RAG, Agent", "agent", "AI-AGENTS"])
    assert "agent" in tags
    assert "rag" in tags or "rag-memory" in tags


def test_weighted_tag_overlap_score_in_range():
    score = weighted_tag_overlap_score(["agent", "rag"], ["agent", "llm"])
    assert 0 < score < 1


def test_merge_and_normalize_tag_sources_merges_sources():
    merged = merge_and_normalize_tag_sources(["agent"], ["AI-AGENT"])
    assert merged == ["agent"]
