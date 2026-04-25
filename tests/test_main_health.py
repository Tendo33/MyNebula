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

    monkeypatch.setattr(
        "nebula.db.database.check_db_connection", fake_check_db_connection
    )

    with TestClient(main_module.create_app()) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["scheduler"]["status"] == "stopped"


def test_create_app_filters_blank_cors_origins(monkeypatch):
    from nebula import main as main_module
    from nebula.core.config import AppSettings

    monkeypatch.setattr(
        main_module,
        "get_app_settings",
        lambda: AppSettings(
            cors_origins="https://app.example.com, ,https://admin.example.com"
        ),
    )

    app = main_module.create_app()
    cors_middleware = next(
        middleware
        for middleware in app.user_middleware
        if middleware.cls.__name__ == "CORSMiddleware"
    )

    assert cors_middleware.kwargs["allow_origins"] == [
        "https://app.example.com",
        "https://admin.example.com",
    ]


def test_setup_logging_redacts_sensitive_extras_without_type_error(tmp_path):
    from nebula.utils import logger_util

    log_path = tmp_path / "app.log"

    try:
        logger_util.setup_logging(
            format_string="{extra}",
            log_file=str(log_path),
            catch=False,
        )

        logger_util.get_logger("test").bind(
            api_key="top-secret",
            nested={"token": "nested-secret", "safe": "visible"},
        ).info("structured log")
    finally:
        logger_util.logger.remove()
        logger_util.logger.configure(patcher=None)

    content = log_path.read_text(encoding="utf-8")
    assert "***REDACTED***" in content
    assert "top-secret" not in content
    assert "nested-secret" not in content
