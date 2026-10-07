# FACET with Isaac Sim in Docker

This setup runs the repository's Isaac Lab environment in a container and opens Isaac Sim on the local desktop through XWayland/X11. The live checkout is mounted at `/workspace/active_adaptation`; Hydra outputs, W&B offline runs, and checkpoints are therefore available on the host. Container project discovery and Isaac caches live under `~/.cache/active-adaptation-docker` by default.

The image uses Ubuntu 22.04, Python 3.11.17, Isaac Sim 5.1.0, Isaac Lab 2.3.2.post1, and PyTorch 2.7.0 with CUDA 12.8. It uses the host NVIDIA driver through NVIDIA Container Toolkit.

## Host prerequisites

Install Docker Engine and the Docker Compose plugin. On an Ubuntu/Debian host with a working NVIDIA driver, configure GPU access using the setup script:

```bash
./docker/setup_gpu.sh
```

It asks for your sudo password, installs NVIDIA Container Toolkit from NVIDIA's stable repository, configures Docker, restarts its daemon, and checks GPU access. Stop running Docker containers first; the script refuses to restart Docker while containers are running. This host setup does not require rebuilding the image or logging out.

If NVIDIA Container Toolkit is already installed, the equivalent configuration commands are:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --rm --runtime=nvidia --gpus all ubuntu:22.04 nvidia-smi
```

Follow the current [Docker Engine installation guide](https://docs.docker.com/engine/install/), [Compose plugin installation guide](https://docs.docker.com/compose/install/linux/), and [NVIDIA Container Toolkit installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for distribution-specific steps.

Accept the Isaac Sim license and terms before using the image. The first build downloads Isaac Sim packages from NVIDIA's package index. The launcher asks for acceptance on the first launch, or you can pass `ACCEPT_EULA=Y`. It saves your choice in `EULA_ACCEPTED` under the Docker cache directory and forwards `OMNI_KIT_ACCEPT_EULA=YES` on subsequent launches. This avoids Isaac's built-in prompt writing into its root-owned Python installation. Telemetry consent remains optional; training defaults to W&B offline mode and does not require W&B credentials.

The local desktop must expose X11 or XWayland through `DISPLAY`, `/tmp/.X11-unix`, and a readable `XAUTHORITY` cookie file. For Wayland desktops, the launcher uses the session's XWayland display. It extracts only the current display cookie into a temporary read-only file; it does not disable display authentication.

## Build and launch

From the repository root:

```bash
docker/build.sh
ACCEPT_EULA=Y docker/run.sh train
```

The build script passes your UID and GID into the image and creates a matching named account. This lets OptiX resolve the runtime username for its disk cache while outputs remain owned by your host user. Rebuild the image when using it with a different host UID. Direct `docker build` invocations should pass `--build-arg HOST_UID="$(id -u)" --build-arg HOST_GID="$(id -g)"`.

Training starts `facet/Go2Force` with `ppo_facet_train`, four environments, GUI rendering, 300,000 frames, and offline W&B. Extra Hydra overrides go at the end:

```bash
ACCEPT_EULA=Y docker/run.sh train task.num_envs=8 total_frames=1000000 seed=7
```

Play a checkpoint produced inside the checkout:

```bash
ACCEPT_EULA=Y docker/run.sh play outputs/2026-10-07/12-00-00-Go2Force-ppo_facet/checkpoint_latest.pt
```

Playback loads the saved teacher policy. Student adaptation and finetuning are separate training workflows. To open an interactive container shell, run `ACCEPT_EULA=Y docker/run.sh shell`.

By default, training outputs are in `outputs/`, playback outputs are in `outputs_play/`, and checkpoints are in the Hydra/W&B run directory under `outputs/`. Since the repository is mounted and the container runs with your UID and GID, those files remain host-owned. The image tag and cache location can be changed with `ACTIVE_ADAPTATION_DOCKER_IMAGE` and `ACTIVE_ADAPTATION_DOCKER_CACHE`.

Only one managed Isaac container may run at once. `Ctrl+C` interrupts the current command. To request a graceful stop from another terminal, run `docker stop --time 30 active-adaptation-isaac`. Closing the Isaac window ends the simulator process; the launcher then closes the managed container. `docker/run.sh` removes its temporary Xauthority cookie when the command exits.

The container keeps separate, persistent project-discovery, shader, Warp, Kit, and user configuration caches in the configured Docker cache directory. Remove that directory only when you want a clean container cache.

Training and playback pass Kit's `--portable-root` option pointing to the writable Kit cache mount. Kit logs, downloaded extensions, and user settings therefore stay outside the root-owned Python installation.

## Runtime checks

After building, the image build checks native importer and graphics libraries with the system linker and verifies Python package versions plus PyTorch's CUDA build. Verify username lookup with:

```bash
docker run --rm --user "$(id -u):$(id -g)" \
  --entrypoint /opt/venv/bin/python active-adaptation:isaaclab-2.3.2 \
  -c 'import os, pwd; print(pwd.getpwuid(os.getuid()).pw_name)'
```

With Docker configured, confirm GPU access using:

```bash
docker run --rm --runtime=nvidia --gpus all active-adaptation:isaaclab-2.3.2 \
  /opt/venv/bin/python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'
```

Then launch training and verify the Isaac viewport moves, a checkpoint appears in the mounted output directory, and the output is owned by your host UID. Stop and restart the container, then play that checkpoint to confirm loading and cache persistence. Check `nvidia-smi` after Ctrl+C, closing the GUI, startup failure, or `docker stop` to confirm no managed Isaac process retains GPU memory. The host desktop may continue to use GPU memory.

For a short training check that runs ten PPO updates and the script's final evaluation, use `ACCEPT_EULA=Y docker/run.sh train total_frames=1280`. With four environments and 32 rollout steps, 1,280 frames produce ten updates.
