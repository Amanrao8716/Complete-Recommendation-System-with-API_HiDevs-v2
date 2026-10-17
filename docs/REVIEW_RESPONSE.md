# Response to the project review

The review asked for four things (priority order). Each is addressed below
with the exact files to look at.

## 1. Code quality
*"Incorporate type hints, docstrings and comments."*

- Every function across `api/`, `common/`, `data/`, `engine/` and
  `scripts/` now have full type annotations and docstrings; `mypy`
  settings live in `pyproject.toml` (`disallow_untyped_defs = true`).
- Dataclasses use precise field types (`frozenset[int]`, `dict[str, float]`),
  scorers are described by a `Protocol`, and the API is split into
  `app.py`, `errors.py` and `metrics.py`.
- Formatting/lint config: `black` at 79 columns + `flake8` (`setup.cfg`),
  run in CI.

## 2. Production readiness
*"Dockerfile, dependency locking, structured logging."*

- **Docker**: multi-stage `Dockerfile`, non-root user, `/data` volume,
  `HEALTHCHECK`, graceful shutdown (`scripts/docker_entrypoint.sh`),
  `docker-compose.yml`, `.dockerignore`, `gunicorn.conf.py`.
- **Dependency locking**: `requirements.in` -> pinned `requirements.txt`
  (runtime only) and `requirements-dev.txt` (tools); matplotlib/pytest/flake8
  no longer ship in the production image.
- **Structured logging**: `common/logging_config.py` emits JSON lines with a
  per-request `request_id` (context variable), method/path/status/latency
  fields, and masks secrets.
- **CI**: `.github/workflows/ci.yml` (lint, types, tests + coverage gate,
  Docker build + container smoke test).
- Extras: SQLite WAL + busy timeout, `pool_pre_ping` for other databases,
  `X-Content-Type-Options`/`Cache-Control: no-store` headers, request-size
  limit.

## 3. Error handling
*"Go beyond a single try/except: input validation and granular exceptions."*

- Exception hierarchy in `common/errors.py` (`RecsysError` ->
  `ValidationError`, `NotFoundError`, `ConfigError`), mapped to HTTP codes in
  one place (`api/app.py`).
- Validation in every layer: similarity functions, evaluator (`k`, input
  types), `WeightedScorer` (negative/NaN/unknown weights), `diversify`,
  `CandidateGenerator` (counts, unknown strategy), orchestrator (id types,
  limit, strategy, rating), API (JSON body, field types, limits).
- Database outages -> `503 database_unavailable`; unexpected errors ->
  `500` without leaking internals; a failed refresh keeps the previous
  in-memory snapshot (tested).
- Bugs found and fixed while doing this:
  - a non-ASCII `X-API-Key` header crashed `hmac.compare_digest` (HTTP 500);
  - client-supplied `X-Request-ID` was echoed unsanitised into logs;
  - NDCG/precision could count repeated items twice.

## 4. Configuration management
*"Externalise hard-coded values such as `max_candidates`."*

- `common/config.py`: validated, immutable `Settings` read from environment
  variables; documented in the README and `.env.example`.
- Formerly hard-coded values now configurable: `N_NEIGHBORS`, `N_POPULAR`,
  `MAX_LIMIT`, `DEFAULT_LIMIT`, `DIVERSITY_PENALTY`, `PROFICIENCY_GAIN`,
  `CACHE_TTL`, `CACHE_MAX_SIZE`, `MAX_REQUEST_BYTES`, `METRICS_WINDOW`,
  `HOST`, `PORT`, `LOG_*`, and the per-strategy scorer weights
  (`STRATEGY_WEIGHTS_FILE`).
- Precedence: explicit argument > `Settings` > built-in default; bad values
  fail at start-up with a message naming the variable.

## Backwards compatibility
Public functions, constructor arguments, internal attributes used by the
original tests and all HTTP routes/status codes/error codes are unchanged.
New behaviour is opt-in through new keyword arguments or environment
variables.
