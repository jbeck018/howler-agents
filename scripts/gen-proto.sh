#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
PROTO_DIR="$ROOT_DIR/proto"

echo "==> Generating proto stubs..."

# Ensure output directories exist
mkdir -p "$ROOT_DIR/packages/howler-agents-service/src/howler_agents_service/generated"
mkdir -p "$ROOT_DIR/packages/howler-agents-ts/src/generated"

# Generate using buf
cd "$PROTO_DIR"

if command -v buf &>/dev/null; then
    buf generate
else
    echo "buf not found. Install: https://buf.build/docs/installation"
    echo "Falling back to grpcio-tools for Python only..."

    # Python stubs via grpcio-tools
    python -m grpc_tools.protoc \
        -I"$PROTO_DIR" \
        --python_out="$ROOT_DIR/packages/howler-agents-service/src/howler_agents_service/generated" \
        --pyi_out="$ROOT_DIR/packages/howler-agents-service/src/howler_agents_service/generated" \
        --grpc_python_out="$ROOT_DIR/packages/howler-agents-service/src/howler_agents_service/generated" \
        "$PROTO_DIR/howler_agents/v1/types.proto" \
        "$PROTO_DIR/howler_agents/v1/events.proto" \
        "$PROTO_DIR/howler_agents/v1/service.proto"

    echo "WARNING: TypeScript stubs not generated (requires buf). Install buf for full generation."
fi

echo "==> Proto generation complete."
