from dataclasses import dataclass

from sqlalchemy.dialects import postgresql

from nebula.api.repos import (
    _build_related_cache_key,
    _build_related_cache_upsert_stmt,
    _deserialize_related_results,
    _rank_related_candidates,
    _serialize_related_results,
)


@dataclass
class DummyRepo:
    id: int
    github_repo_id: int
    full_name: str
    owner: str
    name: str
    description: str | None
    language: str | None
    topics: list[str]
    html_url: str
    homepage_url: str | None
    stargazers_count: int
    forks_count: int
    watchers_count: int
    open_issues_count: int
    ai_summary: str | None
    cluster_id: int | None
    coord_x: float | None
    coord_y: float | None
    coord_z: float | None
    starred_at: object | None
    repo_updated_at: object | None
    is_embedded: bool
    is_summarized: bool
    star_list_id: int | None
    ai_tags: list[str] | None
    embedding: list[float] | None


def _dummy_repo(repo_id: int, embedding: list[float], tags: list[str], stars: int) -> DummyRepo:
    return DummyRepo(
        id=repo_id,
        github_repo_id=1000 + repo_id,
        full_name=f"o/r{repo_id}",
        owner="o",
        name=f"r{repo_id}",
        description="desc",
        language="Python",
        topics=tags,
        html_url=f"https://github.com/o/r{repo_id}",
        homepage_url=None,
        stargazers_count=stars,
        forks_count=0,
        watchers_count=0,
        open_issues_count=0,
        ai_summary=None,
        cluster_id=None,
        coord_x=None,
        coord_y=None,
        coord_z=None,
        starred_at=None,
        repo_updated_at=None,
        is_embedded=True,
        is_summarized=True,
        star_list_id=1,
        ai_tags=tags,
        embedding=embedding,
    )


def test_rank_related_candidates_returns_score_sorted_items():
    anchor = _dummy_repo(1, [1.0, 0.0], ["agent", "rag"], 10)
    c1 = _dummy_repo(2, [0.99, 0.01], ["agent"], 5)
    c2 = _dummy_repo(3, [0.3, 0.7], ["frontend"], 500)

    ranked = _rank_related_candidates(
        anchor_repo=anchor,
        candidates=[c1, c2],
        min_score=0.0,
        min_semantic=0.0,
        limit=5,
    )

    assert len(ranked) == 2
    assert ranked[0].repo.id == 2
    assert ranked[0].score >= ranked[1].score


def test_rank_related_candidates_filters_by_min_score():
    anchor = _dummy_repo(1, [1.0, 0.0], ["agent"], 10)
    c1 = _dummy_repo(2, [0.0, 1.0], ["frontend"], 5)

    ranked = _rank_related_candidates(
        anchor_repo=anchor,
        candidates=[c1],
        min_score=0.9,
        min_semantic=0.0,
        limit=5,
    )

    assert ranked == []


def test_rank_related_candidates_filters_low_semantic_even_with_other_signals():
    anchor = _dummy_repo(1, [1.0, 0.0], ["agent"], 10)
    candidate = _dummy_repo(2, [0.2, 0.98], ["agent"], 5)

    ranked = _rank_related_candidates(
        anchor_repo=anchor,
        candidates=[candidate],
        min_score=0.0,
        min_semantic=0.65,
        limit=5,
    )

    assert ranked == []


def test_related_cache_key_is_stable_under_rounding():
    key_a = _build_related_cache_key(min_score=0.4, min_semantic=0.65, limit=20)
    key_b = _build_related_cache_key(min_score=0.40000001, min_semantic=0.6500001, limit=20)

    assert key_a == key_b


def test_related_results_cache_round_trip_restores_ranked_items():
    anchor = _dummy_repo(1, [1.0, 0.0], ["agent", "rag"], 10)
    c1 = _dummy_repo(2, [0.99, 0.01], ["agent"], 5)
    c2 = _dummy_repo(3, [0.98, 0.02], ["rag"], 8)

    ranked = _rank_related_candidates(
        anchor_repo=anchor,
        candidates=[c1, c2],
        min_score=0.0,
        min_semantic=0.0,
        limit=5,
    )
    serialized = _serialize_related_results(ranked)
    restored = _deserialize_related_results(
        serialized,
        repo_by_id={repo.id: repo for repo in [c1, c2]},
        limit=5,
    )

    assert [item.repo.id for item in restored] == [item.repo.id for item in ranked]
    assert [round(item.score, 6) for item in restored] == [
        round(item.score, 6) for item in ranked
    ]


def test_related_cache_upsert_stmt_uses_conflict_update():
    stmt = _build_related_cache_upsert_stmt(
        user_id=1,
        anchor_repo_id=2,
        cache_key="k",
        items=[{"repo_id": 3, "score": 0.9}],
        anchor_updated_at=None,
        user_last_sync_at=None,
    )

    compiled = str(stmt.compile(dialect=postgresql.dialect()))

    assert "INSERT INTO repo_related_caches" in compiled
    assert "ON CONFLICT (user_id, anchor_repo_id, cache_key) DO UPDATE" in compiled
