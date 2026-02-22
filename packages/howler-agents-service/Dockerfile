# Multi-stage build for Howler Agents Service
FROM python:3.12-slim AS builder

WORKDIR /workspace

# Copy packages only (no workspace root needed)
COPY packages/howler-agents-core packages/howler-agents-core
COPY packages/howler-agents-service packages/howler-agents-service

# Install with pip — avoids uv workspace resolution issues in Docker.
# pip ignores [tool.uv.sources] and installs both local packages directly.
RUN python -m venv .venv && \
    .venv/bin/pip install --no-cache-dir \
      ./packages/howler-agents-core[all] \
      ./packages/howler-agents-service

# Runtime
FROM python:3.12-slim AS runtime

WORKDIR /app
COPY --from=builder /workspace/.venv /app/.venv
COPY packages/howler-agents-service/src /app/src
COPY migrations /app/migrations

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"

EXPOSE 50051 8080

CMD ["python", "-m", "howler_agents_service.main"]
