#!/usr/bin/env bash
# Runs all SQL migrations in order against the local Postgres instance.
# This script is mounted into the postgres container's docker-entrypoint-initdb.d
# directory and runs automatically on first database creation.

set -euo pipefail

MIGRATIONS_DIR="/migrations"

echo "==> Running howler-agents migrations..."

for migration in "$MIGRATIONS_DIR"/*.sql; do
    if [ -f "$migration" ]; then
        echo "  Applying: $(basename "$migration")"
        psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f "$migration" 2>&1 | sed 's/^/    /'
    fi
done

echo "==> Migrations complete."
