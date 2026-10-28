from pathlib import Path

import pytest
from fastapi.testclient import TestClient

FRONTEND_DIST = Path(__file__).parent.parent / "frontend" / "dist"


@pytest.mark.skipif(not FRONTEND_DIST.exists(), reason="frontend not built")
def test_spa_fallback_returns_404_for_missing_files(monkeypatch):
    from nebula import main as main_module

    async def _noop_async():
        return None

    for name in (
        "init_db",
        "reconcile_orphaned_jobs",
        "close_db",
        "close_scheduler_service",
        "close_embedding_service",
        "close_llm_service",
    ):
        monkeypatch.setattr(main_module, name, _noop_async)
    monkeypatch.setattr(main_module, "setup_logging", lambda **_kwargs: None)
    monkeypatch.setattr(
        main_module,
        "get_scheduler_service",
        lambda: type("_S", (), {"start": staticmethod(_noop_async)})(),
    )

    with TestClient(main_module.create_app()) as client:
        assert client.get("/no-such-file.php").status_code == 404
        assert client.get("/sitemap.xml").status_code == 404
        spa = client.get("/graph")
        assert spa.status_code == 200
        assert "text/html" in spa.headers["content-type"]
