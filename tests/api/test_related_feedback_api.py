import pytest
from pydantic import ValidationError
from sqlalchemy.dialects import postgresql

from nebula.schemas.repo import RelatedFeedbackRequest


def test_related_feedback_request_accepts_valid_values():
    req = RelatedFeedbackRequest(
        candidate_repo_id=2,
        feedback="helpful",
        score_snapshot=0.91,
        model_version="v1",
    )
    assert req.feedback == "helpful"


def test_related_feedback_request_rejects_invalid_feedback():
    with pytest.raises(ValidationError):
        RelatedFeedbackRequest(candidate_repo_id=2, feedback="bad")


@pytest.mark.asyncio
async def test_related_feedback_route_requires_admin_and_csrf():
    from nebula.api.v2 import repos as repos_api

    route = next(
        (
            route
            for route in repos_api.router.routes
            if route.path == "/{repo_id}/related-feedback"
        ),
        None,
    )

    assert route is not None
    dependency_calls = {
        getattr(getattr(dependency, "call", None), "__name__", "")
        for dependency in route.dependant.dependencies
    }
    assert "require_admin" in dependency_calls
    assert "require_admin_csrf" in dependency_calls


@pytest.mark.asyncio
async def test_submit_related_feedback_executes_atomic_upsert(monkeypatch):
    from types import SimpleNamespace

    from nebula.api.v2 import repos as repos_api

    async def fake_get_default_user(_db):
        return SimpleNamespace(id=1)

    monkeypatch.setattr(repos_api, "get_default_user", fake_get_default_user)

    class _FakeResult:
        def __init__(self, value):
            self._value = value

        def scalar_one_or_none(self):
            return self._value

    class _FakeDb:
        def __init__(self):
            self.commit_calls = 0
            self.statements: list[object] = []

        async def execute(self, _statement):
            self.statements.append(_statement)
            if len(self.statements) == 1:
                return _FakeResult(10)
            if len(self.statements) == 2:
                return _FakeResult(11)
            return _FakeResult(None)

        async def commit(self):
            self.commit_calls += 1

    db = _FakeDb()

    payload = RelatedFeedbackRequest(
        candidate_repo_id=11,
        feedback="helpful",
        score_snapshot=0.77,
        model_version="v1",
    )
    response = await repos_api.submit_related_feedback(
        repo_id=10,
        request=payload,
        db=db,
    )

    assert response.status == "ok"
    assert db.commit_calls == 1
    assert len(db.statements) == 3

    upsert_statement = db.statements[-1]
    compiled = str(upsert_statement.compile(dialect=postgresql.dialect()))
    assert "ON CONFLICT" in compiled
