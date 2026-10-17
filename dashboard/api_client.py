"""Small client used by the dashboard to talk to the recommendation API.

Two interchangeable clients share one interface:

* :class:`HttpClient` calls a running server over HTTP.
* :class:`InProcessClient` calls the Flask app directly (no server and no
  network needed), which makes the dashboard a one-command demo.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from typing import Any

from flask import Flask
from flask.testing import FlaskClient

DEFAULT_API_URL = "http://127.0.0.1:5000"
STRATEGIES = (
    "auto",
    "hybrid",
    "skill_based",
    "collaborative",
    "popularity",
    "cold_start",
)
FEEDBACK_TYPES = ("view", "like", "complete", "rate", "dislike")


class ApiClientError(Exception):
    """An API call failed.

    Args:
        status: HTTP status code, or ``0`` if the server was unreachable.
        code: Machine-readable error code from the API.
        message: Human-readable explanation.
        request_id: The API's request ID, when available.
    """

    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.request_id = request_id

    def __str__(self) -> str:
        """Readable one-line description."""
        suffix = f" (request {self.request_id})" if self.request_id else ""
        return f"{self.status} {self.code}: {self.message}{suffix}"


class RecsysClient:
    """Base class: endpoint methods built on a transport ``_request``."""

    def _request(
        self,
        method: str,
        path: str,
        params: Mapping[str, Any] | None = None,
        payload: Any = None,
    ) -> tuple[int, Any]:
        """Send a request and return ``(status_code, parsed_json_or_None)``."""
        raise NotImplementedError

    def _call(
        self,
        method: str,
        path: str,
        params: Mapping[str, Any] | None = None,
        payload: Any = None,
    ) -> dict[str, Any]:
        """Return the JSON object of a successful call or raise."""
        status, body = self._request(method, path, params, payload)
        if status >= 400:
            error = body.get("error", {}) if isinstance(body, dict) else {}
            raise ApiClientError(
                status,
                str(error.get("code", "error")),
                str(error.get("message", f"HTTP {status}")),
                error.get("request_id"),
            )
        if not isinstance(body, dict):
            raise ApiClientError(
                status, "bad_response", "The server did not return JSON"
            )
        return body

    def health(self) -> dict[str, Any]:
        """Return ``/health``; a degraded (503) report is returned too."""
        status, body = self._request("GET", "/health")
        if isinstance(body, dict) and "status" in body:
            return body
        raise ApiClientError(status, "bad_response", "Unexpected health reply")

    def metrics(self) -> dict[str, Any]:
        """Return ``/metrics``."""
        return self._call("GET", "/metrics")

    def recommend(
        self, user_id: int, limit: int = 5, strategy: str = "auto"
    ) -> dict[str, Any]:
        """Return ``/recommend/<user_id>`` for the given options."""
        return self._call(
            "GET",
            f"/recommend/{int(user_id)}",
            {"limit": limit, "strategy": strategy},
        )

    def feedback(
        self,
        user_id: int,
        content_id: int,
        type_: str,
        rating: float | None = None,
    ) -> dict[str, Any]:
        """Record feedback through ``POST /feedback``."""
        payload: dict[str, Any] = {
            "user_id": int(user_id),
            "content_id": int(content_id),
            "type": type_,
        }
        if rating is not None:
            payload["rating"] = rating
        return self._call("POST", "/feedback", payload=payload)


class HttpClient(RecsysClient):
    """Talk to a running API server over HTTP.

    Args:
        base_url: Server address, e.g. ``http://127.0.0.1:5000``.
        api_key: Sent as ``X-API-Key`` when given.
        timeout: Seconds to wait for each request.
    """

    def __init__(
        self, base_url: str, api_key: str = "", timeout: float = 10.0
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        params: Mapping[str, Any] | None = None,
        payload: Any = None,
    ) -> tuple[int, Any]:
        url = self.base_url + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        headers = {"Accept": "application/json"}
        data = None
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        request = urllib.request.Request(
            url, data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as r:
                status, raw = r.status, r.read()
        except urllib.error.HTTPError as exc:
            status, raw = exc.code, exc.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ApiClientError(
                0,
                "connection_error",
                f"Cannot reach {self.base_url}: {exc}",
            ) from exc
        try:
            return status, json.loads(raw)
        except ValueError:
            return status, None


class InProcessClient(RecsysClient):
    """Call a Flask app directly, without starting a server."""

    def __init__(self, client: FlaskClient, api_key: str = "") -> None:
        self._client = client
        self._headers = {"X-API-Key": api_key} if api_key else {}

    def _request(
        self,
        method: str,
        path: str,
        params: Mapping[str, Any] | None = None,
        payload: Any = None,
    ) -> tuple[int, Any]:
        response = self._client.open(
            path,
            method=method,
            query_string=dict(params) if params else None,
            json=payload,
            headers=self._headers,
        )
        return response.status_code, response.get_json(silent=True)


def build_inprocess_client(database_url: str | None = None) -> InProcessClient:
    """Create the app against a local database and wrap it in a client.

    An empty database is filled with the same synthetic dataset as
    ``scripts/seed_data.py`` so the demo always has something to show.
    """
    from api.app import create_app
    from data.database import (
        init_db,
        make_engine,
        make_session_factory,
        session_scope,
    )
    from data.repositories import UserRepository
    from engine.orchestrator import RecommendationOrchestrator
    from scripts.seed_data import generate_dataset, seed_database

    engine = make_engine(database_url)
    init_db(engine)
    factory = make_session_factory(engine)
    with session_scope(factory) as session:
        empty = UserRepository(session).count() == 0
    if empty:
        seed_database(factory, generate_dataset())
    app: Flask = create_app(
        orchestrator=RecommendationOrchestrator(factory), api_key=""
    )
    return InProcessClient(app.test_client())
