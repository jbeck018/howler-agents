#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
MIGRATIONS_DIR="$ROOT_DIR/migrations"

DATABASE_URL="${DATABASE_URL_SYNC:-postgresql://howler:howler@localhost:5432/howler_agents}"

echo "==> Running migrations against: ${DATABASE_URL%%@*}@***"

for migration in "$MIGRATIONS_DIR"/*.sql; do
    echo "  Applying: $(basename "$migration")"
    psql "$DATABASE_URL" -f "$migration" 2>&1 | sed 's/^/    /'
done

echo "==> Migrations complete."
