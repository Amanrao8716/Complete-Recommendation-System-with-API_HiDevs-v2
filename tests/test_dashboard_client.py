"""Tests for the dashboard's API client (no Streamlit needed)."""

import threading

import pytest
from werkzeug.serving import make_server

from api.app import create_app
from dashboard.api_client import (
    ApiClientError,
    HttpClient,
    InProcessClient,
    build_inprocess_client,
)


@pytest.fixture()
def inprocess(orchestrator):
    app = create_app(orchestrator=orchestrator, api_key="")
    return InProcessClient(app.test_client())


@pytest.fixture()
def live_server(orchestrator):
    app = create_app(orchestrator=orchestrator, api_key="s3cret")
    server = make_server("127.0.0.1", 0, app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    thread.join(timeout=5)


def test_inprocess_recommend_feedback_health_metrics(inprocess):
    assert inprocess.health()["status"] == "ok"
    recs = inprocess.recommend(1, limit=3, strategy="hybrid")
    assert recs["count"] == 3 and recs["strategy"] == "hybrid"
    stored = inprocess.feedback(1, 7, "rate", rating=5)
    assert stored["status"] == "recorded"
    assert inprocess.metrics()["total_requests"] >= 3


def test_inprocess_errors_become_client_errors(inprocess):
    with pytest.raises(ApiClientError) as caught:
        inprocess.recommend(9999)
    assert caught.value.status == 404
    assert caught.value.code == "user_not_found"
    assert caught.value.request_id
    with pytest.raises(ApiClientError) as caught:
        inprocess.recommend(1, limit=0)
    assert caught.value.status == 400
    with pytest.raises(ApiClientError):
        inprocess.feedback(1, 7, "rate")  # rating missing


def test_http_client_with_api_key(live_server):
    client = HttpClient(live_server, api_key="s3cret")
    assert client.health()["status"] == "ok"
    assert client.recommend(1, 2)["count"] == 2


def test_http_client_wrong_key_is_401(live_server):
    with pytest.raises(ApiClientError) as caught:
        HttpClient(live_server, api_key="wrong").recommend(1)
    assert caught.value.status == 401 and caught.value.code == "unauthorized"


def test_http_client_unreachable_server_message():
    client = HttpClient("http://127.0.0.1:9", timeout=1)
    with pytest.raises(ApiClientError) as caught:
        client.health()
    assert caught.value.status == 0
    assert caught.value.code == "connection_error"


def test_error_string_is_readable():
    err = ApiClientError(404, "not_found", "nope", "abc123")
    assert str(err) == "404 not_found: nope (request abc123)"


def test_build_inprocess_client_seeds_empty_database(tmp_path):
    client = build_inprocess_client(f"sqlite:///{tmp_path / 'demo.db'}")
    health = client.health()
    assert health["status"] == "ok" and health["users"] > 0
    assert client.recommend(1, 3)["count"] == 3
