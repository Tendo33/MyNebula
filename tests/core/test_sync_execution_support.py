from types import SimpleNamespace

import pytest


class _FakeScalarResult:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class _FakeExecuteResult:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return _FakeScalarResult(self._rows)


class _FakeDb:
    def __init__(self, rows_by_batch):
        self.rows_by_batch = rows_by_batch
        self.calls = 0

    async def execute(self, _statement):
        rows = self.rows_by_batch[self.calls]
        self.calls += 1
        return _FakeExecuteResult(rows)


@pytest.mark.asyncio
async def test_prefetch_existing_repos_batches_and_indexes_by_github_id():
    from nebula.application.services.sync_execution_support import (
        prefetch_existing_repos,
    )

    db = _FakeDb(
        [
            [
                SimpleNamespace(github_repo_id=101, full_name="octo/one"),
                SimpleNamespace(github_repo_id=102, full_name="octo/two"),
            ],
            [
                SimpleNamespace(github_repo_id=103, full_name="octo/three"),
            ],
        ]
    )

    existing = await prefetch_existing_repos(
        db,
        user_id=7,
        github_ids=[101, 102, 103],
        batch_size=2,
    )

    assert db.calls == 2
    assert sorted(existing) == [101, 102, 103]
    assert existing[103].full_name == "octo/three"


def test_collect_readme_targets_marks_new_and_changed_repos():
    from nebula.application.services.sync_execution_support import (
        collect_readme_targets,
    )
    from nebula.utils import compute_content_hash, compute_topics_hash

    repos = [
        SimpleNamespace(
            id=1,
            full_name="octo/new-repo",
            description="brand new",
            topics=["graph"],
        ),
        SimpleNamespace(
            id=2,
            full_name="octo/stable",
            description="stable desc",
            topics=["python"],
        ),
        SimpleNamespace(
            id=3,
            full_name="octo/changed",
            description="changed desc",
            topics=["vector"],
        ),
    ]

    existing_map = {
        2: SimpleNamespace(
            description_hash=compute_content_hash("stable desc"),
            topics_hash=compute_topics_hash(["python"]),
        ),
        3: SimpleNamespace(
            description_hash=compute_content_hash("old desc"),
            topics_hash=compute_topics_hash(["vector"]),
        ),
    }

    readme_targets, repo_hashes = collect_readme_targets(repos, existing_map)

    assert readme_targets == ["octo/new-repo", "octo/changed"]
    assert repo_hashes[1][0] == compute_content_hash("brand new")
    assert repo_hashes[3][1] == compute_topics_hash(["vector"])
