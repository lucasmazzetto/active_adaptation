#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    cat <<'USAGE'
Usage: docker/setup_gpu.sh

Install NVIDIA Container Toolkit on an Ubuntu/Debian host, configure Docker,
restart its daemon, and verify GPU access. Requires sudo and a working NVIDIA
driver. Stop running containers before invoking this script.
USAGE
    exit 0
fi
if (($#)); then
    echo "Usage: docker/setup_gpu.sh" >&2
    exit 2
fi
for prerequisite in apt-get docker nvidia-smi systemctl; do
    if ! command -v "$prerequisite" >/dev/null 2>&1; then
        echo "Required host command is missing: $prerequisite" >&2
        exit 1
    fi
done
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

root_command=()
if ((EUID != 0)); then
    root_command=(sudo)
    sudo -v
fi
if [[ -n "$("${root_command[@]}" docker ps -q)" ]]; then
    echo "Stop running Docker containers before configuring and restarting Docker." >&2
    exit 1
fi

if ! command -v nvidia-ctk >/dev/null 2>&1 || ! command -v nvidia-container-runtime >/dev/null 2>&1; then
    "${root_command[@]}" apt-get update
    "${root_command[@]}" apt-get install -y --no-install-recommends ca-certificates curl gnupg
    setup_dir="$(mktemp -d)"
    trap 'rm -rf "$setup_dir"' EXIT
    curl --fail --silent --show-error --location --retry 3 \
        https://nvidia.github.io/libnvidia-container/gpgkey -o "$setup_dir/key.asc"
    gpg --batch --dearmor --output "$setup_dir/key.gpg" "$setup_dir/key.asc"
    curl --fail --silent --show-error --location --retry 3 \
        https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        -o "$setup_dir/repository.list"
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        "$setup_dir/repository.list" > "$setup_dir/signed-repository.list"
    "${root_command[@]}" install -m 0644 "$setup_dir/key.gpg" \
        /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    "${root_command[@]}" install -m 0644 "$setup_dir/signed-repository.list" \
        /etc/apt/sources.list.d/nvidia-container-toolkit.list
    "${root_command[@]}" apt-get update
    "${root_command[@]}" apt-get install -y nvidia-container-toolkit
fi

"${root_command[@]}" nvidia-ctk runtime configure --runtime=docker
"${root_command[@]}" systemctl restart docker
"${root_command[@]}" docker run --rm --runtime=nvidia --gpus all ubuntu:22.04 nvidia-smi
echo "Docker GPU access is ready. After accepting Isaac Sim's license, launch:"
echo "  ACCEPT_EULA=Y ./docker/run.sh train"
