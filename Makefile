.PHONY: setup dev test build lint proto docker-up docker-down clean migrate

# --- Setup ---
setup: setup-python setup-node
	@echo "Setup complete. Copy .env.example to .env and fill in your keys."

setup-python:
	uv sync --all-packages

setup-node:
	pnpm install

# --- Development ---
dev:
	@echo "Starting all services..."
	$(MAKE) -j3 dev-service dev-ui dev-docs

dev-service:
	cd packages/howler-agents-service && uv run python -m howler_agents_service.main

dev-ui:
	pnpm --filter howler-agents-ui run dev

dev-docs:
	pnpm --filter howler-agents-docs run dev

# --- Testing ---
test: test-python test-node

test-python:
	uv run pytest packages/howler-agents-core/tests packages/howler-agents-service/tests -v

test-node:
	pnpm -r run test

# --- Build ---
build: build-python build-node

build-python:
	uv build --package howler-agents-core
	uv build --package howler-agents-service

build-node:
	pnpm -r run build

# --- Linting ---
lint: lint-python lint-node

lint-python:
	uv run ruff check packages/
	uv run mypy

lint-node:
	pnpm -r run lint
	pnpm -r run typecheck

# --- Proto Generation ---
proto:
	./scripts/gen-proto.sh

# --- Database ---
migrate:
	./scripts/migrate.sh

# --- Docker ---
docker-up:
	docker compose -f deploy/docker/docker-compose.yml up -d

docker-down:
	docker compose -f deploy/docker/docker-compose.yml down

docker-build:
	docker compose -f deploy/docker/docker-compose.yml build

# --- Clean ---
clean:
	find packages -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find packages -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find packages -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find packages -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find packages -type d -name node_modules -exec rm -rf {} + 2>/dev/null || true
	find packages -type d -name dist -exec rm -rf {} + 2>/dev/null || true
	rm -rf .mypy_cache .ruff_cache .pytest_cache
