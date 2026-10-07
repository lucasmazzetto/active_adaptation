#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
IMAGE="${ACTIVE_ADAPTATION_DOCKER_IMAGE:-active-adaptation:isaaclab-2.3.2}"

if ! command -v docker >/dev/null 2>&1; then
    echo "Docker Engine is required to build the image." >&2
    exit 127
fi

exec docker build --tag "$IMAGE" --file "$ROOT_DIR/Dockerfile" "$@" "$ROOT_DIR"
