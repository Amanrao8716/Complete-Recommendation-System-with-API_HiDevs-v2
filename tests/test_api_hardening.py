"""Security, robustness and configuration tests for the REST API."""

import logging

import pytest
from sqlalchemy.exc import OperationalError

from api.app import create_app
from common.config import Settings
from common.errors import ConfigError
from common.logging_config import (
    JsonFormatter,
    RequestIdFilter,
    configure_logging,
    request_id_var,
)
from engine.orchestrator import RecommendationOrchestrator


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


def _boom(*_args, **_kwargs):
    raise OperationalError("SELECT 1", {}, Exception("db down"))


def test_non_ascii_api_key_is_rejected_not_a_server_error(orchestrator):
    client = create_app(orchestrator=orchestrator, api_key="secret")
    client = client.test_client()
    response = client.get("/recommend/1", headers={"X-API-Key": "sécret"})
    assert response.status_code == 401
    assert response.get_json()["error"]["code"] == "unauthorized"


def test_unsafe_request_ids_are_replaced(client):
    for bad in ("bad id with spaces", "x" * 65, "a;b", "tab\tid", "é"):
        try:
            response = client.get("/health", headers={"X-Request-ID": bad})
        except UnicodeEncodeError:  # some test clients refuse the header
            continue
        assert response.headers["X-Request-ID"] != bad
        assert len(response.headers["X-Request-ID"]) == 12


def test_security_headers_present(client):
    response = client.get("/health")
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["Cache-Control"] == "no-store"


def test_oversized_body_gets_413_json(orchestrator):
    cfg = Settings(max_request_bytes=64)
    app = create_app(orchestrator=orchestrator, api_key="", settings=cfg)
    response = app.test_client().post(
        "/feedback", data="x" * 500, content_type="application/json"
    )
    assert response.status_code == 413
    assert "error" in response.get_json()


def test_database_outage_returns_503(orchestrator, monkeypatch):
    app = create_app(orchestrator=orchestrator, api_key="")
    monkeypatch.setattr(orchestrator, "get_recommendations_detailed", _boom)
    response = app.test_client().get("/recommend/1")
    body = response.get_json()
    assert response.status_code == 503
    assert body["error"]["code"] == "database_unavailable"
    assert "db down" not in str(body)  # internals are not leaked


def test_unexpected_error_does_not_leak_details(orchestrator, monkeypatch):
    app = create_app(orchestrator=orchestrator, api_key="")

    def explode(*_a, **_k):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(orchestrator, "get_recommendations_detailed", explode)
    response = app.test_client().get("/recommend/1")
    assert response.status_code == 500
    assert "secret internal detail" not in response.get_data(as_text=True)


def test_limits_come_from_settings(session_factory):
    cfg = Settings(max_limit=10, default_limit=3)
    orch = RecommendationOrchestrator(session_factory, settings=cfg)
    app = create_app(orchestrator=orch, api_key="", settings=cfg)
    client = app.test_client()
    assert client.get("/recommend/1").get_json()["count"] == 3
    assert client.get("/recommend/1?limit=10").status_code == 200
    rejected = client.get("/recommend/1?limit=11")
    assert rejected.status_code == 400
    assert "between 1 and 10" in rejected.get_json()["error"]["message"]


def test_invalid_environment_fails_fast(monkeypatch):
    monkeypatch.setenv("PORT", "not-a-port")
    with pytest.raises(ConfigError):
        create_app(database_url="sqlite://")


def test_access_log_is_structured(client):
    handler = _Capture()
    logger = logging.getLogger("recsys.api")
    logger.addHandler(handler)
    old_level = logger.level
    logger.setLevel(logging.INFO)
    try:
        client.get("/recommend/9999")
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
    record = [r for r in handler.records if hasattr(r, "status")][-1]
    assert record.status == 404 and record.path == "/recommend/9999"
    assert record.method == "GET" and record.duration_ms >= 0


def test_json_formatter_and_request_id_filter():
    token = request_id_var.set("req-123")
    try:
        record = logging.LogRecord(
            "x", logging.INFO, __file__, 1, "hello %s", ("world",), None
        )
        record.user_id = 7
        assert RequestIdFilter().filter(record)
        text = JsonFormatter().format(record)
    finally:
        request_id_var.reset(token)
    import json

    data = json.loads(text)
    assert data["message"] == "hello world" and data["level"] == "INFO"
    assert data["request_id"] == "req-123" and data["user_id"] == 7


def test_configure_logging_is_idempotent():
    root = logging.getLogger()
    before = list(root.handlers)
    try:
        configure_logging("INFO", "json")
        configure_logging("INFO", "text")
        ours = [
            h for h in root.handlers if getattr(h, "_recsys_handler", False)
        ]
        assert len(ours) == 1
    finally:
        root.handlers[:] = before
