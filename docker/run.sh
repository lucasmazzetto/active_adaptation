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
            "+app.kit_args='--portable-root /var/cache/active-adaptation/kit'" \
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
            "+app.kit_args='--portable-root /var/cache/active-adaptation/kit'" \
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

if ! command -v docker >/dev/null 2>&1; then
    echo "Docker Engine is required. Install Docker and NVIDIA Container Toolkit first." >&2
    exit 127
fi
if ! docker_runtimes="$(docker info --format '{{range $name, $runtime := .Runtimes}}{{println $name}}{{end}}' 2>&1)"; then
    echo "Cannot access the Docker daemon:" >&2
    echo "$docker_runtimes" >&2
    echo "If your Docker group membership is not active, run: newgrp docker" >&2
    exit 1
fi
if ! grep -qx nvidia <<< "$docker_runtimes"; then
    echo "NVIDIA GPU runtime is not configured for Docker." >&2
    echo "Run ./docker/setup_gpu.sh in your terminal; it requires sudo." >&2
    exit 1
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

# Kit's own prompt writes into its root-owned installation. Record consent in
# the user's cache and use Kit's environment variable to avoid that write.
eula_file="$CACHE_DIR/EULA_ACCEPTED"
eula_choice="${ACCEPT_EULA:-}"
if [[ -z "$eula_choice" ]]; then
    if [[ -f "$eula_file" ]] && grep -Eiq '^(y|yes|1)$' "$eula_file"; then
        eula_choice=yes
    else
        if [[ ! -t 0 ]]; then
            echo "Accept Isaac Sim's license by running interactively or setting ACCEPT_EULA=Y." >&2
            exit 1
        fi
        echo "NVIDIA Omniverse license: https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html"
        read -r -p "Do you accept the EULA? (Yes/No): " eula_choice
    fi
fi
case "$eula_choice" in
    [Yy]|[Yy][Ee][Ss]|1)
        (umask 077; printf 'yes\n' > "$eula_file")
        ;;
    *)
        echo "The EULA was not accepted." >&2
        exit 1
        ;;
esac

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
    --runtime=nvidia \
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
    --env ACCEPT_EULA=Y \
    --env OMNI_KIT_ACCEPT_EULA=YES \
    "$IMAGE" "${app_command[@]}"
