from inspect import getsource
from pathlib import Path


def test_search_and_related_queries_keep_cosine_distance_ordering():
    from nebula.api.v2 import repos as repos_api
    from nebula.application.services import related_repo_service

    search_source = getsource(repos_api.search_repos)
    related_source = getsource(related_repo_service.get_related_repos)

    assert "cosine_distance" in search_source
    assert (
        "order_by(StarredRepo.embedding.cosine_distance(query_embedding))"
        in search_source
    )
    assert "cosine_distance" in related_source
    assert (
        "order_by(StarredRepo.embedding.cosine_distance(anchor_repo.embedding))"
        in related_source
    )


def test_ann_migration_exists_for_starred_repo_embeddings():
    versions_dir = Path(__file__).resolve().parents[2] / "alembic" / "versions"
    matching_files = sorted(
        versions_dir.glob("*add_starred_repo_embedding_ann_index.py")
    )

    assert matching_files, "Expected ANN index migration file to exist"

    migration_source = matching_files[-1].read_text(encoding="utf-8")

    assert "USING ivfflat" in migration_source
    assert "vector_cosine_ops" in migration_source
    assert "ix_starred_repos_embedding_cosine_ann" in migration_source
