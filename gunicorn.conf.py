"""Gunicorn settings, all overridable through environment variables.

The recommender keeps an in-memory snapshot and cache per process, so the
default is ONE worker with several threads.  Several workers would each
hold their own copy and not see each other's feedback until restart; scale
out only after moving state to a shared store (Postgres + Redis).
"""

import os

bind = f"{os.environ.get('HOST', '0.0.0.0')}:{os.environ.get('PORT', '5000')}"
workers = int(os.environ.get("WEB_CONCURRENCY", "1"))
threads = int(os.environ.get("GUNICORN_THREADS", "8"))
timeout = int(os.environ.get("GUNICORN_TIMEOUT", "30"))
graceful_timeout = 30
keepalive = 5
worker_tmp_dir = "/dev/shm"  # avoids slow heartbeat files on container disks
loglevel = os.environ.get("LOG_LEVEL", "info").lower()
errorlog = "-"
accesslog = None  # the app writes its own structured access log
