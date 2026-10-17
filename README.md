# Day 30 Capstone: Recommendation System

A complete learning-content recommendation service: **SQLite database →
recommendation engine (5 strategies, caching, explanations) → Flask REST
API**, with tests, offline evaluation, a load test, Docker and a Postman
collection.

```
User Request → API → RecommendationOrchestrator → Database (SQLite)
                ↓              ↓
              Metrics     Cache + candidate generation + scoring
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/seed_data.py --reset        # 40 users, 60 content items
python -m api.app                          # http://127.0.0.1:5000

curl "http://127.0.0.1:5000/recommend/1?limit=5"
curl -X POST http://127.0.0.1:5000/feedback \
     -H "Content-Type: application/json" \
     -d '{"user_id": 1, "content_id": 7, "type": "complete"}'
```

In GitHub Codespaces, forward port 5000 and use the forwarded URL.
`scripts/demo.sh` runs a scripted walkthrough, handy for the demo video.

### Configuration (environment variables)

Nothing is hard-coded: every tunable is read once at start-up into a
validated, immutable `Settings` object (`common/config.py`). Invalid values
stop the app immediately with a message naming the variable, e.g.
`ConfigError: PORT='abc' is invalid: expected an integer`.
Copy `.env.example` for a complete template.

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `sqlite:///recsys.db` | Database location |
| `API_KEY` | _(empty = auth off)_ | If set, `/recommend`, `/feedback`, `/metrics` require header `X-API-Key` |
| `HOST` / `PORT` | `127.0.0.1` / `5000` | Dev-server bind address and port (Docker sets `HOST=0.0.0.0`) |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `LOG_FORMAT` | `json` | `json` (structured) or `text` (human friendly) |
| `CACHE_TTL` | `300` | Recommendation cache TTL in seconds (`0` disables) |
| `CACHE_MAX_SIZE` | `5000` | Maximum cached result sets |
| `MAX_LIMIT` / `DEFAULT_LIMIT` | `50` / `5` | Largest / default `limit` query parameter |
| `N_NEIGHBORS` | `10` | Similar users used for collaborative scores |
| `N_POPULAR` | `20` | Extra popular items proposed per request |
| `DIVERSITY_PENALTY` | `0.92` | Re-rank penalty per repeated category, `(0, 1]` |
| `PROFICIENCY_GAIN` | `0.15` | Skill gain when a user completes content |
| `MAX_REQUEST_BYTES` | `16384` | Largest accepted request body (else `413`) |
| `METRICS_WINDOW` | `2000` | Latency samples kept per endpoint |
| `STRATEGY_WEIGHTS_FILE` | _(none)_ | JSON file overriding scorer weights per strategy, see `config/strategy_weights.example.json` |

Local development tip: `LOG_FORMAT=text python -m api.app` prints readable
log lines; set `HOST=0.0.0.0` if a forwarded port (Codespaces) is not
reachable.

### Structured logging

Logs are JSON lines on stderr (12-factor style); every line carries the
`request_id` of the request that produced it:

```json
{"timestamp": "2026-10-15T03:58:07.145+00:00", "level": "INFO", "logger": "recsys.api", "message": "request_id=demo-req-1 GET /recommend/1 -> 200 0.6ms", "method": "GET", "path": "/recommend/1", "status": 200, "duration_ms": 0.56, "request_id": "demo-req-1"}
```

Secrets (`API_KEY`, passwords inside `DATABASE_URL`) are masked when
settings are logged. Client-supplied `X-Request-ID` values are accepted only
if they match `[A-Za-z0-9._-]{1,64}`, which prevents log injection.

## API

| Method | Path | Description |
|---|---|---|
| GET | `/recommend/<user_id>?limit=5&strategy=auto` | Recommendations (limit 1-50). Strategies: `auto`, `hybrid`, `skill_based`, `collaborative`, `popularity`, `cold_start` |
| POST | `/feedback` | Body: `{"user_id", "content_id", "type", "rating"?}`; `type` is `view`, `like`, `complete`, `rate` (needs `rating`) or `dislike`; `rating` 1-5 |
| GET | `/health` | Liveness + DB check |
| GET | `/metrics` | Request counts, latency percentiles per endpoint, cache hit rate |

Every response carries an `X-Request-ID` header (a client-supplied one is
echoed back) which also appears in the logs and in error bodies.

Errors are always JSON: `{"error": {"code", "message", "request_id"}}`.

| Status | `code` | When |
|---|---|---|
| 400 | `invalid_request` | Bad path/query/body, unknown strategy, out-of-range `limit`/`rating` |
| 401 | `unauthorized` | Missing or wrong `X-API-Key` (non-ASCII keys are rejected cleanly) |
| 404 | `user_not_found` / `not_found` | Unknown user, content item or route |
| 405 / 413 | `method_not_allowed` / `request_entity_too_large` | Wrong method / body over `MAX_REQUEST_BYTES` |
| 503 | `database_unavailable` | The database cannot be reached (details only in the logs) |
| 500 | `internal_error` | Anything unexpected; traceback is logged, never returned |

Example response:

```json
{
  "request_id": "9f2c1a7d03be", "user_id": 1, "strategy": "hybrid",
  "count": 1, "cached": false, "latency_ms": 1.4,
  "recommendations": [{
    "content_id": 6, "title": "Full-Stack Web Project",
    "category": "Web Development", "difficulty": 4, "score": 0.4982,
    "strategy": "hybrid",
    "explanation": "Recommended because it fits your interest in Web Development and builds skills you care about (JavaScript, HTML and React).",
    "components": {"skill_match": 0.62, "category_affinity": 0.6, "...": 0}
  }]
}
```

## Dashboard (Streamlit)

A small web UI for trying the system without `curl`: ranked cards with
explanations and score breakdowns, a before/after feedback demo, and live
health/metrics.

```bash
pip install -r requirements-ui.txt
streamlit run dashboard/streamlit_app.py      # opens http://localhost:8501
```

By default it runs the engine inside Streamlit (the demo database is created
and seeded automatically). To use a running API instead, start
`python -m api.app`, then pick **Connect to a running API** in the sidebar
(set the URL and, if used, the API key). `dashboard/api_client.py` holds all
API logic and is covered by `tests/test_dashboard_client.py`.

## How the engine works

1. **Candidate generation** (`engine/candidate_gen.py`): items sharing skills
   the user needs, items in categories they like, items liked by similar
   users (cosine similarity on interaction vectors), popular items.
   Already-seen items are removed; popularity backfills if too few remain.
2. **Scoring** (`engine/scorer.py`): five scorers (skill match, category
   affinity, difficulty fit, collaborative, popularity) combined with
   per-strategy weights, then a light category-diversity re-rank.
3. **Strategies**: `hybrid` (default for users with history),
   `skill_based`, `collaborative`, `popularity`, `cold_start`.
4. **Cold start**: users with no history and no skills get `cold_start`:
   interests (if any) + popularity + beginner-friendly difficulty.
5. **Feedback loop**: `POST /feedback` stores the interaction, updates the
   in-memory model immediately, invalidates that user's cache and, for
   `complete`, raises proficiency in the content's skills.
6. **Explanations**: built from the two largest weighted score components.
7. **Cache**: in-memory dict with TTL, per (user, limit, strategy), cleared
   per user on feedback.

## Project layout

```
common/    config.py (validated Settings), errors.py (exception hierarchy),
           logging_config.py (JSON logging + request IDs)
data/      database.py, models.py (6 tables), repositories.py
engine/    orchestrator.py, similarity.py, candidate_gen.py, scorer.py,
           evaluator.py, domain.py
api/       app.py (routes + error handling), errors.py, metrics.py
tests/     test_data.py (sample dataset), test_repositories.py,
           test_engine.py, test_api.py, test_scripts.py,
           test_config.py, test_engine_validation.py,
           test_api_hardening.py, test_orchestrator_config.py,
           test_dashboard_client.py
scripts/   seed_data.py, evaluate.py, load_test.py, demo.sh,
           docker_entrypoint.sh
dashboard/ streamlit_app.py (UI), api_client.py (HTTP / in-process client)
config/    strategy_weights.example.json
```

Code quality: every function is type-annotated and documented
(`mypy` config in `pyproject.toml`), formatted with `black` (79 columns)
and checked with `flake8`. Inputs are validated at every layer and failures
raise subclasses of `common.errors.RecsysError`
(`ValidationError`, `NotFoundError`, `ConfigError`), which also inherit
`ValueError`/`LookupError` so older `except` clauses keep working.

> The `similarity`, `candidate_gen`, `scorer` and `evaluator` modules are the
> "Day 29" components. If you have your own Day 29 versions, keep the public
> interfaces used by `orchestrator.py` (`generate(ctx, strategy, ...)`,
> `WeightedScorer.score(...)`, `evaluate_rankings(...)`).

## Tests, evaluation, load test

```bash
pip install -r requirements-dev.txt
make check                                    # flake8 + black + mypy + tests
pytest --cov --cov-report=term-missing        # tests with coverage
python scripts/evaluate.py                    # writes evaluation_report.md
python scripts/load_test.py --workers 10 --requests 100
python scripts/load_test.py --url http://127.0.0.1:5000   # live server
```

The test dataset (`tests/test_data.py`) has 14 users and 24 content items.
Besides the original suites, `test_config.py`, `test_engine_validation.py`,
`test_api_hardening.py` and `test_orchestrator_config.py` cover settings,
input validation, security behaviour (API-key handling, request-ID
sanitising, body-size limit), 503 handling and configuration-driven
behaviour. `evaluation_report.md` contains precision@5 / recall@5 /
NDCG@5 per strategy, a feedback-learning curve, charts and load-test
results.

## Docker

```bash
docker build -t recsys .
docker run -p 5000:5000 -e API_KEY=secret -v recsys-data:/data recsys
curl -H "X-API-Key: secret" localhost:5000/recommend/1

docker compose up --build        # same thing with a named volume
```

The image is multi-stage (pinned `requirements.txt` only, no dev tools),
runs as a non-root user, keeps the SQLite file on the `/data` volume,
emits JSON logs, has a `HEALTHCHECK` on `/health` and shuts down cleanly on
`SIGTERM`. On first start it seeds an empty database (disable with
`SEED_ON_START=false`). Gunicorn settings (`WEB_CONCURRENCY`,
`GUNICORN_THREADS`, `GUNICORN_TIMEOUT`) are in `gunicorn.conf.py`.
For Render/Heroku-style deployment, use the Dockerfile and set `PORT`.

### Dependencies

`requirements.in` lists direct dependencies; `requirements.txt` pins the
full runtime set (used by Docker); `requirements-dev.txt` adds test, lint
and chart tools. Regenerate the pins with
`pip-compile --generate-hashes requirements.in`.

## Known limitations

- SQLite and an in-process snapshot/cache suit a single instance (hence
  one gunicorn worker with threads); for several instances use Postgres
  and a shared cache such as Redis.
- Recommendations for other users are not invalidated when one user gives
  feedback (their cached lists expire after `CACHE_TTL`).
- The evaluation dataset is synthetic; absolute metric values are only
  meaningful relative to the baselines.
## 🎥 Project Demo

[![Watch the Project Demo](https://img.youtube.com/vi/gCBZ4UgiULQ/maxresdefault.jpg)](https://youtu.be/gCBZ4UgiULQ)

▶️ **[Watch the full demo on YouTube](https://youtu.be/gCBZ4UgiULQ)**
