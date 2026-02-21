from nebula.core.embedding import EmbeddingService


def test_build_repo_text_prefers_ai_summary_and_includes_readme_signal():
    service = EmbeddingService()
    text = service.build_repo_text(
        full_name="o/r",
        description="raw description",
        topics=["agent"],
        readme_content="# Features\n- fast\n# Usage\nrun it",
        language="Python",
        ai_summary="ai summary",
        ai_tags=["agent-memory"],
    )

    assert "ai summary" in text
    assert "raw description" not in text
    assert "Readme:" in text
    assert "Features" in text or "features" in text


def test_extract_readme_signal_caps_length():
    service = EmbeddingService()
    signal = service._extract_readme_signal("x" * 5000, max_chars=300)
    assert len(signal) == 300
