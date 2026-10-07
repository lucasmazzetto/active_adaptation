#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
IMAGE="${ACTIVE_ADAPTATION_DOCKER_IMAGE:-active-adaptation:isaaclab-2.3.2}"
CONTAINER_NAME="active-adaptation-isaac"
CACHE_DIR="${ACTIVE_ADAPTATION_DOCKER_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/active-adaptation-docker}"
REPO_CONTAINER_PATH=/workspace/active_adaptation

usage() {
    cat <<'USAGE'
Usage:
  docker/run.sh train [Hydra overrides...]
  docker/run.sh play <checkpoint-path> [Hydra overrides...]
  docker/run.sh shell

Build the image first with:
  docker/build.sh
USAGE
}

command_name="${1:-}"
if [[ -z "$command_name" ]]; then
    usage >&2
    exit 2
fi
shift

case "$command_name" in
    train)
        app_command=(/opt/venv/bin/python scripts/train_ppo.py \
            task=facet/Go2Force algo=ppo_facet_train task.num_envs=4 \
            headless=false total_frames=300000 wandb.mode=offline "$@")
        ;;
    play)
        if (($# == 0)); then
            echo "play requires a checkpoint path inside the checkout" >&2
            usage >&2
            exit 2
        fi
        checkpoint="$1"
        shift
        if [[ "$checkpoint" != /* ]]; then
            checkpoint="$ROOT_DIR/$checkpoint"
        fi
        checkpoint="$(realpath -e -- "$checkpoint")"
        case "$checkpoint" in
            "$ROOT_DIR"/*) ;;
            *) echo "checkpoint must be inside $ROOT_DIR" >&2; exit 2 ;;
        esac
        checkpoint="${REPO_CONTAINER_PATH}${checkpoint#"$ROOT_DIR"}"
        app_command=(/opt/venv/bin/python scripts/play.py \
            task=facet/Go2Force algo=ppo_facet_train task.num_envs=4 \
            headless=false "checkpoint_path=$checkpoint" "$@")
        ;;
    shell)
        app_command=(/bin/bash -l "$@")
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac

eula_args=()
case "${ACCEPT_EULA:-}" in
    [Yy]|[Yy][Ee][Ss]|1)
        eula_args+=(--env "ACCEPT_EULA=$ACCEPT_EULA" --env OMNI_KIT_ACCEPT_EULA=YES)
        ;;
esac

if ! command -v docker >/dev/null 2>&1; then
    echo "Docker Engine is required. Install Docker and NVIDIA Container Toolkit first." >&2
    exit 127
fi
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "Image '$IMAGE' is missing. Build it with: docker/build.sh" >&2
    exit 1
fi

existing="$(docker ps -aq --filter "name=^/${CONTAINER_NAME}$")"
if [[ -n "$existing" ]]; then
    if docker inspect --format '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null | grep -qx true; then
        echo "Managed Isaac container '$CONTAINER_NAME' is already running. Stop it before starting another." >&2
        exit 1
    fi
    docker rm "$CONTAINER_NAME" >/dev/null
fi

if [[ -z "${DISPLAY:-}" ]]; then
    echo "DISPLAY is unset. Start an XWayland/X11 desktop session before launching the GUI." >&2
    exit 1
fi
if [[ ! -d /tmp/.X11-unix ]]; then
    echo "X11 socket directory /tmp/.X11-unix is missing." >&2
    exit 1
fi

host_xauthority="${XAUTHORITY:-$HOME/.Xauthority}"
if [[ ! -r "$host_xauthority" ]] || ! command -v xauth >/dev/null 2>&1; then
    echo "A readable XAUTHORITY file and xauth utility are required for authenticated XWayland access." >&2
    exit 1
fi

mkdir -p "$CACHE_DIR"
chmod 700 "$CACHE_DIR"
mkdir -p "$CACHE_DIR/aa" "$CACHE_DIR/xdg" "$CACHE_DIR/warp" \
    "$CACHE_DIR/kit" "$CACHE_DIR/kit-config" "$CACHE_DIR/kit-data" "$CACHE_DIR/home"
chmod 700 "$CACHE_DIR" "$CACHE_DIR/home"
cookie_file="$(mktemp "${TMPDIR:-/tmp}/active-adaptation-xauth.XXXXXX")"
chmod 600 "$cookie_file"
cleanup() { rm -f "$cookie_file"; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' HUP TERM
xauth -f "$host_xauthority" extract "$cookie_file" "$DISPLAY"
if [[ ! -s "$cookie_file" ]]; then
    echo "No X11 authorization cookie was found for DISPLAY=$DISPLAY." >&2
    exit 1
fi

docker run --rm --name "$CONTAINER_NAME" \
    --init --interactive --tty --stop-timeout 30 \
    --gpus all \
    --user "$(id -u):$(id -g)" \
    --workdir "$REPO_CONTAINER_PATH" \
    --mount "type=bind,src=$ROOT_DIR,dst=$REPO_CONTAINER_PATH" \
    --mount "type=bind,src=$CACHE_DIR/aa,dst=/var/cache/active-adaptation/aa" \
    --mount "type=bind,src=$CACHE_DIR/xdg,dst=/var/cache/active-adaptation/xdg" \
    --mount "type=bind,src=$CACHE_DIR/warp,dst=/var/cache/active-adaptation/warp" \
    --mount "type=bind,src=$CACHE_DIR/kit,dst=/var/cache/active-adaptation/kit" \
    --mount "type=bind,src=$CACHE_DIR/kit-config,dst=/var/cache/active-adaptation/kit-config" \
    --mount "type=bind,src=$CACHE_DIR/kit-data,dst=/var/cache/active-adaptation/kit-data" \
    --mount "type=bind,src=$CACHE_DIR/home,dst=/var/cache/active-adaptation/home" \
    --mount "type=bind,src=/tmp/.X11-unix,dst=/tmp/.X11-unix" \
    --mount "type=bind,src=$cookie_file,dst=/tmp/.Xauthority,readonly" \
    --env DISPLAY \
    --env WAYLAND_DISPLAY \
    --env XAUTHORITY=/tmp/.Xauthority \
    --env NVIDIA_DRIVER_CAPABILITIES=all \
    --env AA_CACHE_DIR=/var/cache/active-adaptation/aa \
    --env XDG_CACHE_HOME=/var/cache/active-adaptation/xdg \
    --env WARP_CACHE_PATH=/var/cache/active-adaptation/warp \
    --env OMNI_KIT_CACHE_ROOT=/var/cache/active-adaptation/kit \
    --env OMNI_KIT_USER_CACHE=/var/cache/active-adaptation/kit \
    --env OMNI_KIT_USER_CONFIG=/var/cache/active-adaptation/kit-config \
    --env OMNI_KIT_USER_DATA=/var/cache/active-adaptation/kit-data \
    --env HOME=/var/cache/active-adaptation/home \
    "${eula_args[@]}" \
    "$IMAGE" "${app_command[@]}"
