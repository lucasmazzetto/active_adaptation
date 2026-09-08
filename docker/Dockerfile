# Isaac Sim + Isaac Lab + this repository's dependencies.
#
# Two supported pairings, selected with build args (see docker/compose.yaml):
#   ISAACSIM_VERSION=5.1.0  ISAACLAB_TAG=v2.3.2                (default; verified for training; headless on driver 595.x)
#   ISAACSIM_VERSION=6.0.0  ISAACLAB_TAG=v3.0.0-beta2.patch1   (renderer works on 595.x, but Isaac Lab 3.0 uses XYZW quaternions: not training-ready)
ARG ISAACSIM_VERSION=5.1.0
FROM nvcr.io/nvidia/isaac-sim:${ISAACSIM_VERSION}
ARG ISAACSIM_VERSION
ARG ISAACLAB_TAG=v2.3.2

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

# Isaac Lab 2.x only: `toml` is imported by isaaclab/__init__.py and is not bundled with
# Isaac Sim 5.1, and flatdict is an sdist whose setup.py imports pkg_resources, which pip's
# isolated build env lacks, making the 2.x isaaclab install fail silently. Isaac Lab 3.x needs
# neither, and the flatdict trick breaks on the newer pip shipped with Isaac Sim 6.0.
RUN if [ "${ISAACLAB_TAG#v3}" = "${ISAACLAB_TAG}" ]; then \
        python -m pip install --no-cache-dir -U wheel toml \
        && python -m pip install --no-cache-dir --no-build-isolation "flatdict==4.0.1"; \
    else echo "Isaac Lab ${ISAACLAB_TAG}: skipping 2.x-only toml/flatdict workaround"; fi

# Isaac Lab, installed into Isaac Sim's own Python.
#   2.x: `--install none` skips the RL frameworks.
#   3.x: `-i core` installs the core submodules (incl. isaaclab_physx) and no RL extras;
#        its installer also installs torch 2.10 (cu128) itself, since Isaac Sim 6.0 bundles none.
RUN git clone --depth 1 -b ${ISAACLAB_TAG} https://github.com/isaac-sim/IsaacLab.git ${ISAACLAB_PATH} \
    && ln -sf ${ISAACSIM_ROOT_PATH} ${ISAACLAB_PATH}/_isaac_sim \
    && if [ "${ISAACLAB_TAG#v3}" != "${ISAACLAB_TAG}" ]; then ${ISAACLAB_PATH}/isaaclab.sh -i core; else ${ISAACLAB_PATH}/isaaclab.sh --install none; fi \
    && bash -c 'source /opt/isaac_env.sh && python -c "import isaaclab; print(\"isaaclab import OK:\", isaaclab.__file__)"'

# Freeze torch/torchvision/numpy AFTER the Isaac Lab install (on 3.x that is where torch comes
# from) so the repo's dependency install cannot change them. isaac_env.sh must be sourced for
# the bundled packages to be importable. Keep only requirement lines.
RUN bash -c 'source /opt/isaac_env.sh && python /opt/make_constraints.py' \
      | grep -oE '[A-Za-z0-9_.-]+==[0-9][A-Za-z0-9_.]*' > /opt/constraints.txt \
    && cat /opt/constraints.txt \
    && test -s /opt/constraints.txt

# Dependencies of this repo (mirrors pyproject.toml; the repo itself is mounted at run time).
# tensordict/torchrl must be recent: this codebase reads `action_log_prob`, which is only the
# default log-prob key once `composite_lp_aggregate` defaults to False (tensordict >= ~0.9).
RUN python -m pip install --no-cache-dir -c /opt/constraints.txt \
        setproctitle hydra-core omegaconf wandb einops termcolor pygame onnxruntime onnxscript \
        jaxtyping "tensordict==0.13.0" "torchrl==0.13.3" imageio av moviepy mujoco linuxfd pdbpp \
        "simple-raycaster @ git+https://github.com/btx0424/simple-raycaster.git"

# mjlab powers the MuJoCo backend (`backend=mujoco`).
RUN python -m pip install --no-cache-dir -c /opt/constraints.txt "mjlab==1.2.0"

# warp: Isaac Sim ships warp as the Kit extension omni.warp.core, which its own extensions are
# built against, and isaac_env.sh puts that copy first on PYTHONPATH. If pip pulled in a
# *different* warp-lang (Isaac Lab 2.x pins none, so it took the newest), remove it so there is
# exactly one warp. The invariant checked below is: the importable warp IS the bundled extension
# copy (same version, same path). Note that `warp.types.array` disappeared after 1.8, so that older
# heuristic must not be used here.
RUN bash -c 'source /opt/isaac_env.sh; \
      bundled=$(python -c "import warp; print(warp.config.version)"); \
      pipv=$(python -m pip show warp-lang 2>/dev/null | awk "/^Version:/{print \$2}"); \
      echo "bundled warp=$bundled pip warp-lang=${pipv:-none}"; \
      if [ -n "$pipv" ] && [ "$pipv" != "$bundled" ]; then python -m pip uninstall -y warp-lang; fi; \
      python -c "import warp, sys; v=warp.config.version; f=warp.__file__; print(\"warp\", v, f); \
          assert v == \"$bundled\" and \"/extscache/omni.warp.core-\" in f, (v, f)"'

# Vulkan presentation for the GUI on a hybrid-graphics laptop (displays on the AMD iGPU,
# rendering on NVIDIA): without Mesa's radeon ICD no presentable surface can be created.
RUN apt-get update && apt-get install -y --no-install-recommends \
        mesa-vulkan-drivers libvulkan1 libgl1-mesa-dri \
    && rm -rf /var/lib/apt/lists/*

# Install this repository into the image. At run time the bind mount replaces the source at
# the same path, so the editable install keeps pointing at live code while console scripts
# (aa-list-tasks, ...) and metadata persist in the image.
COPY . ${AA_PATH}
RUN python -m pip install --no-cache-dir --no-deps -e ${AA_PATH} \
    && bash -c 'source /opt/isaac_env.sh && python -c "import active_adaptation; print(\"active_adaptation import OK\")"' \
    && which aa-list-tasks

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR ${AA_PATH}
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
