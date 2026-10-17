"""Flask REST API serving recommendations.

Run locally::

    python -m api.app            # development server
    gunicorn -c gunicorn.conf.py 'api.app:create_app()'   # production

All behaviour is configured through environment variables; see
:mod:`common.config` for the full list.
"""

from __future__ import annotations

import hmac
import logging
import re
import time
import uuid
from typing import Any

from flask import Flask, Response, g, jsonify, request
from flask.typing import ResponseReturnValue
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from werkzeug.exceptions import HTTPException

from api.errors import ApiError
from api.metrics import MetricsRegistry
from common.config import Settings
from common.errors import NotFoundError, RecsysError, ValidationError
from common.logging_config import configure_logging, request_id_var
from data.database import (
    init_db,
    make_engine,
    make_session_factory,
    session_scope,
)
from engine.orchestrator import (
    MAX_LIMIT,
    VALID_STRATEGIES,
    ContentNotFoundError,
    InvalidFeedbackError,
    InvalidRequestError,
    RecommendationOrchestrator,
    UserNotFoundError,
)

__all__ = [
    "ApiError",
    "MAX_LIMIT",
    "MetricsRegistry",
    "PROTECTED_ENDPOINTS",
    "create_app",
]

logger = logging.getLogger("recsys.api")
PROTECTED_ENDPOINTS = {"recommend", "feedback", "metrics"}
# Only safe characters may be echoed into headers and log lines.
_REQUEST_ID = re.compile(r"[A-Za-z0-9._\-]{1,64}")


def _parse_int(value: str | None, name: str) -> int:
    """Parse a query/path value as an integer or raise a 400 ``ApiError``."""
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ApiError(
            400, "invalid_request", f"{name} must be an integer"
        ) from None


def _require_int(payload: dict[str, Any], name: str) -> int:
    """Return ``payload[name]`` if it is a real integer (not a bool)."""
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ApiError(400, "invalid_request", f"{name} must be an integer")
    return value


def _keys_match(supplied: str, expected: str) -> bool:
    """Constant-time key comparison that also accepts non-ASCII input."""
    return hmac.compare_digest(
        supplied.encode("utf-8"), expected.encode("utf-8")
    )


def create_app(
    database_url: str | None = None,
    api_key: str | None = None,
    orchestrator: RecommendationOrchestrator | None = None,
    cache_ttl: float | None = None,
    settings: Settings | None = None,
) -> Flask:
    """Application factory.

    Args:
        database_url: Overrides ``DATABASE_URL``.
        api_key: Overrides ``API_KEY``; ``""`` explicitly disables auth.
        orchestrator: Use an existing orchestrator (tests, scripts).
        cache_ttl: Overrides ``CACHE_TTL``.
        settings: Full settings object; defaults to ``Settings.from_env()``.

    Raises:
        ConfigError: If the environment holds invalid settings.
    """
    cfg = (settings or Settings.from_env()).with_overrides(
        database_url=database_url, api_key=api_key, cache_ttl=cache_ttl
    )
    if not logging.getLogger().handlers:
        configure_logging(cfg.log_level, cfg.log_format)

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = cfg.max_request_bytes
    if orchestrator is None:
        engine = make_engine(cfg.database_url)
        init_db(engine)
        orchestrator = RecommendationOrchestrator(
            make_session_factory(engine), settings=cfg
        )
    required_key = cfg.api_key

    metrics = MetricsRegistry(cfg.metrics_window)
    app.extensions["orchestrator"] = orchestrator
    app.extensions["metrics"] = metrics
    app.extensions["settings"] = cfg
    logger.info("application created", extra={"settings": cfg.public_dict()})

    # ------------------------------------------------------------------
    # Request lifecycle: tracing, auth, logging, metrics
    # ------------------------------------------------------------------
    @app.before_request
    def start_request() -> None:
        """Assign a request ID, start the timer and check the API key."""
        incoming = request.headers.get("X-Request-ID", "")
        g.request_id = (
            incoming
            if _REQUEST_ID.fullmatch(incoming)
            else uuid.uuid4().hex[:12]
        )
        g.request_token = request_id_var.set(g.request_id)
        g.start = time.perf_counter()
        if required_key and request.endpoint in PROTECTED_ENDPOINTS:
            supplied = request.headers.get("X-API-Key", "")
            if not _keys_match(supplied, required_key):
                raise ApiError(
                    401, "unauthorized", "Missing or invalid API key"
                )

    @app.after_request
    def finish_request(response: Response) -> Response:
        """Add headers, record metrics and write the access log."""
        duration_ms = (
            time.perf_counter() - g.get("start", time.perf_counter())
        ) * 1000
        rule = request.url_rule.rule if request.url_rule else "unmatched"
        response.headers["X-Request-ID"] = g.get("request_id", "-")
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        metrics.record(rule, response.status_code, duration_ms)
        logger.info(
            "request_id=%s %s %s -> %s %.1fms",
            g.get("request_id", "-"),
            request.method,
            request.path,
            response.status_code,
            duration_ms,
            extra={
                "method": request.method,
                "path": request.path,
                "status": response.status_code,
                "duration_ms": round(duration_ms, 2),
            },
        )
        return response

    @app.teardown_request
    def end_request(_exc: BaseException | None) -> None:
        """Clear the request ID from the logging context."""
        token = g.pop("request_token", None)
        if token is not None:
            try:
                request_id_var.reset(token)
            except ValueError:  # token from another context
                request_id_var.set("-")

    def error_response(
        status: int, code: str, message: str
    ) -> tuple[Response, int]:
        """Build the standard JSON error body."""
        body = {
            "error": {
                "code": code,
                "message": message,
                "request_id": g.get("request_id"),
            }
        }
        return jsonify(body), status

    @app.errorhandler(ApiError)
    def handle_api_error(exc: ApiError) -> ResponseReturnValue:
        """Render an :class:`ApiError`."""
        return error_response(exc.status, exc.code, exc.message)

    @app.errorhandler(HTTPException)
    def handle_http_error(exc: HTTPException) -> ResponseReturnValue:
        """Render Werkzeug HTTP errors (404, 405, 413...) as JSON."""
        return error_response(
            exc.code or 500,
            exc.name.lower().replace(" ", "_"),
            exc.description or exc.name,
        )

    @app.errorhandler(OperationalError)
    def handle_database_error(exc: OperationalError) -> ResponseReturnValue:
        """Map database outages to a 503 instead of a 500."""
        logger.error("database unavailable", exc_info=exc)
        return error_response(
            503, "database_unavailable", "Database temporarily unavailable"
        )

    @app.errorhandler(RecsysError)
    def handle_domain_error(exc: RecsysError) -> ResponseReturnValue:
        """Map project exceptions that escaped a route to 4xx/5xx."""
        if isinstance(exc, NotFoundError):
            return error_response(404, "not_found", str(exc))
        if isinstance(exc, ValidationError):
            return error_response(400, "invalid_request", str(exc))
        logger.exception("unhandled domain error")
        return error_response(500, "internal_error", "Internal server error")

    @app.errorhandler(Exception)
    def handle_unexpected(exc: Exception) -> ResponseReturnValue:
        """Last resort: log the traceback, hide details from clients."""
        logger.exception("unhandled error")
        return error_response(500, "internal_error", "Internal server error")

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------
    @app.get("/health")
    def health() -> ResponseReturnValue:
        """Liveness plus a database round-trip."""
        try:
            with session_scope(orchestrator.session_factory) as session:
                session.execute(text("SELECT 1"))
            database, status = "ok", 200
        except Exception:  # noqa: BLE001 - any failure means "degraded"
            logger.exception("health check failed")
            database, status = "error", 503
        stats = orchestrator.stats()
        body = {
            "status": "ok" if status == 200 else "degraded",
            "database": database,
            "users": stats["users"],
            "content_items": stats["content_items"],
            "request_id": g.request_id,
        }
        return jsonify(body), status

    @app.get("/metrics", endpoint="metrics")
    def metrics_endpoint() -> ResponseReturnValue:
        """Request counts, latency percentiles and cache statistics."""
        body = metrics.snapshot()
        body["recommender"] = orchestrator.stats()
        return jsonify(body)

    @app.get("/recommend/<user_id>")
    def recommend(user_id: str) -> ResponseReturnValue:
        """Return ranked recommendations for one user."""
        uid = _parse_int(user_id, "user_id")
        limit = _parse_int(
            request.args.get("limit", str(orchestrator.default_limit)),
            "limit",
        )
        strategy = request.args.get("strategy", "auto")
        if not 1 <= limit <= orchestrator.max_limit:
            raise ApiError(
                400,
                "invalid_request",
                f"limit must be between 1 and {orchestrator.max_limit}",
            )
        if strategy not in VALID_STRATEGIES:
            raise ApiError(
                400,
                "invalid_request",
                f"strategy must be one of {', '.join(VALID_STRATEGIES)}",
            )
        try:
            result = orchestrator.get_recommendations_detailed(
                uid, limit, strategy
            )
        except UserNotFoundError as exc:
            raise ApiError(404, "user_not_found", str(exc)) from exc
        except InvalidRequestError as exc:
            raise ApiError(400, "invalid_request", str(exc)) from exc
        latency = (time.perf_counter() - g.start) * 1000
        return jsonify(
            {
                "request_id": g.request_id,
                "user_id": uid,
                "strategy": result.strategy,
                "count": len(result.items),
                "cached": result.cached,
                "latency_ms": round(latency, 2),
                "recommendations": [i.to_dict() for i in result.items],
            }
        )

    @app.post("/feedback")
    def feedback() -> ResponseReturnValue:
        """Record a view/like/complete/rate/dislike interaction."""
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            raise ApiError(
                400, "invalid_request", "Request body must be a JSON object"
            )
        missing = [
            f for f in ("user_id", "content_id", "type") if f not in payload
        ]
        if missing:
            raise ApiError(
                400, "invalid_request", f"Missing fields: {', '.join(missing)}"
            )
        user_id = _require_int(payload, "user_id")
        content_id = _require_int(payload, "content_id")
        type_ = payload["type"]
        if not isinstance(type_, str):
            raise ApiError(400, "invalid_request", "type must be a string")
        try:
            record = orchestrator.record_feedback(
                user_id, content_id, type_, payload.get("rating")
            )
        except (UserNotFoundError, ContentNotFoundError) as exc:
            raise ApiError(404, "not_found", str(exc)) from exc
        except InvalidFeedbackError as exc:
            raise ApiError(400, "invalid_request", str(exc)) from exc
        return (
            jsonify(
                {
                    "request_id": g.request_id,
                    "status": "recorded",
                    "interaction": record,
                }
            ),
            201,
        )

    return app


def main() -> None:
    """Run the development server using settings from the environment."""
    cfg = Settings.from_env()
    configure_logging(cfg.log_level, cfg.log_format)
    create_app(settings=cfg).run(host=cfg.host, port=cfg.port, threaded=True)


if __name__ == "__main__":
    main()
