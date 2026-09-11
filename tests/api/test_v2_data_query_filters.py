from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.dialects import postgresql

from nebula.api.v2.data import (
    _array_text_ilike,
    _build_topic_filter_condition,
    _data_repo_order_by,
    _escape_ilike_fragment,
    _parse_month_window,
    _parse_stars_threshold,
    _trimmed_query,
)
from nebula.db import StarredRepo


def test_parse_stars_threshold_accepts_optional_spaces():
    assert _parse_stars_threshold("stars:>42") == 42
    assert _parse_stars_threshold("stars: > 42") == 42
    assert _parse_stars_threshold("  Stars:\t>\t7  ") == 7


def test_trimmed_query_collapses_blank_input():
    assert _trimmed_query(None) == ""
    assert _trimmed_query("   ") == ""
    assert _trimmed_query("  graph  ") == "graph"


def test_parse_month_window_rejects_invalid_calendar_month():
    try:
        _parse_month_window("2026-13")
    except HTTPException as exc:
        assert exc.status_code == 422
        assert "valid calendar month" in str(exc.detail)
    else:
        raise AssertionError("expected invalid month to raise HTTPException")


def test_topic_filter_condition_normalizes_case_for_dashboard_links():
    condition = _build_topic_filter_condition("  AI  ")
    statement = select(StarredRepo.id).where(condition)
    compiled = str(
        statement.compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )

    assert "unnest" in compiled
    assert "lower(trim(" in compiled
    assert "= 'ai'" in compiled


def test_escape_ilike_fragment_neutralizes_wildcards():
    assert _escape_ilike_fragment("100%") == "%100\\%%"
    assert _escape_ilike_fragment("a_b") == "%a\\_b%"
    assert _escape_ilike_fragment("plain") == "%plain%"


def test_array_text_search_uses_concatenated_text_not_unnest():
    condition = _array_text_ilike(StarredRepo.topics, "graph")
    statement = select(StarredRepo.id).where(condition)
    compiled = str(
        statement.compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )

    assert "array_to_string" in compiled
    assert "unnest" not in compiled
    assert "ESCAPE" in compiled.upper()


def test_data_repo_order_by_adds_stable_id_tiebreaker():
    order_by = _data_repo_order_by("starred_at", "desc")
    statement = select(StarredRepo.id).order_by(*order_by)
    compiled = str(
        statement.compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )

    assert "starred_repos.starred_at DESC" in compiled
    assert "starred_repos.id DESC" in compiled
