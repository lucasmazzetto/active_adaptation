# Isaac Sim 5.1.0 + Isaac Lab v2.3.2 + this repository's dependencies.
ARG ISAACSIM_VERSION=5.1.0
FROM nvcr.io/nvidia/isaac-sim:${ISAACSIM_VERSION}

# The base image runs as uid 1234 (isaac-sim) and cannot apt-get or write /workspace.
# Run as root instead; OMNI_KIT_ALLOW_ROOT lets Kit start under root.
USER root

ENV ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y \
    OMNI_KIT_ALLOW_ROOT=1 \
    LANG=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    HOME=/root \
    ISAACSIM_ROOT_PATH=/isaac-sim \
    ISAACLAB_PATH=/workspace/isaaclab \
    AA_PATH=/workspace/active_adaptation \
    PATH=/isaac-sim/kit/python/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git wget libglib2.0-0 ncurses-term jq \
    && rm -rf /var/lib/apt/lists/*

# Isaac Sim only ships `python3`; the docs and scripts in this repo call `python`.
RUN ln -sf /isaac-sim/kit/python/bin/python3 /usr/local/bin/python

COPY docker/isaac_env.sh /opt/isaac_env.sh
COPY docker/make_constraints.py /opt/make_constraints.py

# Isaac Lab pulls in flatdict, an sdist whose setup.py imports pkg_resources. Pip's isolated
# build env does not provide it, which makes the isaaclab core install fail *silently*
# (isaaclab.sh keeps going). Install it up front against the interpreter's own setuptools.
# `toml` is imported by isaaclab/__init__.py and is not bundled with Isaac Sim
# (the upstream Isaac Lab Dockerfile installs it for the same reason).
RUN python -m pip install --no-cache-dir -U wheel toml \
    && python -m pip install --no-cache-dir --no-build-isolation "flatdict==4.0.1"

# Isaac Lab, installed into Isaac Sim's own Python. "--install none" skips RL libraries we do not use.
RUN git clone --depth 1 -b v2.3.2 https://github.com/isaac-sim/IsaacLab.git ${ISAACLAB_PATH} \
    && ln -sf ${ISAACSIM_ROOT_PATH} ${ISAACLAB_PATH}/_isaac_sim \
    && ${ISAACLAB_PATH}/isaaclab.sh --install none \
    && bash -c 'source /opt/isaac_env.sh && python -c "import isaaclab; print(\"isaaclab import OK:\", isaaclab.__file__)"'

# Freeze the versions Isaac Sim ships (torch, numpy, warp) so later pip installs cannot change them.
# torch et al. are only importable once setup_python_env.sh has been sourced, and isaaclab.sh
# writes terminal escape codes to stdout, so source the env directly and keep only requirement lines.
RUN bash -c 'source /opt/isaac_env.sh && python /opt/make_constraints.py' \
      | grep -oE '[A-Za-z0-9_.-]+==[0-9][A-Za-z0-9_.]*' > /opt/constraints.txt \
    && cat /opt/constraints.txt \
    && test -s /opt/constraints.txt

# Dependencies of this repo (mirrors pyproject.toml; the repo itself is mounted at run time).
# tensordict/torchrl must be recent: this codebase reads `action_log_prob`, which is only the
# default log-prob key once `composite_lp_aggregate` defaults to False (tensordict >= ~0.9).
# On 0.7/0.8 the key is `sample_log_prob` and every PPO update raises KeyError. The repo's
# uv.lock pins 0.7.0 but is stale (it references a projects/hdmi that is not in this tree);
# the README's `pip install -U tensordict torchrl` is what the code actually targets.
RUN python -m pip install --no-cache-dir -c /opt/constraints.txt \
        setproctitle hydra-core omegaconf wandb einops termcolor pygame onnxruntime onnxscript \
        jaxtyping "tensordict==0.13.0" "torchrl==0.13.3" imageio av moviepy mujoco linuxfd pdbpp \
        "simple-raycaster @ git+https://github.com/btx0424/simple-raycaster.git"

# mjlab powers the MuJoCo backend (`backend=mujoco`), which is the only way to get a viewport
# on this machine: Isaac Sim 5.1's RTX renderer crashes against driver 595.x, headless or not.
# Installed after the Isaac deps and before the warp cleanup so the bundled warp still wins.
RUN python -m pip install --no-cache-dir -c /opt/constraints.txt "mjlab==1.2.0"

# Isaac Lab depends on an unpinned `warp-lang`, so pip pulls the newest release, which is
# incompatible with the warp that Isaac Sim's own extensions use. Remove the pip copy; the
# bundled extension (put on PYTHONPATH by isaac_env.sh) is the one and only warp.
RUN python -m pip uninstall -y warp-lang \
    && bash -c 'source /opt/isaac_env.sh && python -c "import warp; assert warp.config.version.startswith(\"1.8\"), warp.config.version; assert hasattr(warp.types, \"array\"); print(\"warp\", warp.config.version, warp.__file__)"'

# Install this repository into the image. At run time the bind mount replaces the
# source at the same path, so the editable install keeps pointing at live code while
# the console scripts (aa-list-tasks, ...) and metadata persist in the image.
COPY . ${AA_PATH}
RUN python -m pip install --no-cache-dir --no-deps -e ${AA_PATH} \
    && bash -c 'source /opt/isaac_env.sh && python -c "import active_adaptation; print(\"active_adaptation import OK\")"' \
    && which aa-list-tasks

# Vulkan presentation for the GUI. This laptop drives its displays from the AMD integrated
# GPU while Isaac renders on the NVIDIA card. The container only gets NVIDIA's Vulkan ICD, so
# with no AMD driver present Vulkan can create no presentable surface at all and Isaac Sim's
# RTX renderer crashes on start-up (librtx.scenedb, carbOnPluginStartup). Mesa supplies the
# radeon ICD that owns the X surface; with it, vulkaninfo reports a working xlib/xcb surface.
# Harmless for headless runs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        mesa-vulkan-drivers libvulkan1 libgl1-mesa-dri \
    && rm -rf /var/lib/apt/lists/*

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR ${AA_PATH}
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
