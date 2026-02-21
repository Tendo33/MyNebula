from dataclasses import dataclass

from nebula.api.repos import _rank_related_candidates


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
        limit=5,
    )

    assert ranked == []
