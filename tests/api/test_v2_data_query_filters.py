from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.dialects import postgresql

from nebula.api.v2.data import (
    _build_topic_filter_condition,
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
