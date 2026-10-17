# syntax=docker/dockerfile:1

# ---- build stage: install pinned dependencies into a virtualenv ----------
FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
COPY requirements.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install -r requirements.txt

# ---- runtime stage: small image, non-root user ---------------------------
FROM python:3.12-slim
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0 \
    PORT=5000 \
    LOG_FORMAT=json \
    DATABASE_URL=sqlite:////data/recsys.db

RUN useradd --create-home --uid 10001 app \
    && mkdir /data && chown app:app /data
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=app:app api ./api
COPY --chown=app:app common ./common
COPY --chown=app:app data ./data
COPY --chown=app:app engine ./engine
COPY --chown=app:app scripts ./scripts
COPY --chown=app:app gunicorn.conf.py ./

USER app
VOLUME /data
EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request as u; u.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '5000'), timeout=3)" || exit 1

CMD ["sh", "scripts/docker_entrypoint.sh"]
