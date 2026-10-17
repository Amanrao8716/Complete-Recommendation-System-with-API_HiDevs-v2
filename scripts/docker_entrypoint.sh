#!/bin/sh
# Container entrypoint: optionally seed an empty database, then serve.
set -eu

if [ "${SEED_ON_START:-true}" = "true" ]; then
    python scripts/seed_data.py --if-empty
fi

# exec so gunicorn receives SIGTERM from `docker stop` and shuts down cleanly
exec gunicorn -c gunicorn.conf.py 'api.app:create_app()'
