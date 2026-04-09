from fastapi.testclient import TestClient


def test_health_reports_degraded_when_scheduler_not_running(monkeypatch):
    from nebula import main as main_module

    async def _noop_async():
        return None

    monkeypatch.setattr(main_module, "setup_logging", lambda **_kwargs: None)
    monkeypatch.setattr(main_module, "init_db", _noop_async)
    monkeypatch.setattr(main_module, "close_db", _noop_async)
    monkeypatch.setattr(main_module, "close_scheduler_service", _noop_async)
    monkeypatch.setattr(main_module, "close_embedding_service", _noop_async)
    monkeypatch.setattr(main_module, "close_llm_service", _noop_async)
    monkeypatch.setattr(
        main_module,
        "get_scheduler_service",
        lambda: type(
            "_Scheduler",
            (),
            {
                "start": staticmethod(_noop_async),
                "is_running": False,
                "last_error": "boot failed",
            },
        )(),
    )

    async def fake_check_db_connection():
        return True

    monkeypatch.setattr("nebula.db.database.check_db_connection", fake_check_db_connection)

    with TestClient(main_module.create_app()) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["scheduler"]["status"] == "stopped"
