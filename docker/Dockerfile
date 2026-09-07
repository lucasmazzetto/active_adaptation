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
RUN python -m pip install --no-cache-dir -U wheel \
    && python -m pip install --no-cache-dir --no-build-isolation "flatdict==4.0.1"

# Isaac Lab, installed into Isaac Sim's own Python. "--install none" skips RL libraries we do not use.
RUN git clone --depth 1 -b v2.3.2 https://github.com/isaac-sim/IsaacLab.git ${ISAACLAB_PATH} \
    && ln -sf ${ISAACSIM_ROOT_PATH} ${ISAACLAB_PATH}/_isaac_sim \
    && ${ISAACLAB_PATH}/isaaclab.sh --install none \
    && python -c "import isaaclab; print('isaaclab import OK:', isaaclab.__file__)"

# Freeze the versions Isaac Sim ships (torch, numpy, warp) so later pip installs cannot change them.
# torch et al. are only importable once setup_python_env.sh has been sourced, and isaaclab.sh
# writes terminal escape codes to stdout, so source the env directly and keep only requirement lines.
RUN bash -c 'source /opt/isaac_env.sh && python /opt/make_constraints.py' \
      | grep -oE '[A-Za-z0-9_.-]+==[0-9][A-Za-z0-9_.]*' > /opt/constraints.txt \
    && cat /opt/constraints.txt \
    && test -s /opt/constraints.txt

# Dependencies of this repo (mirrors pyproject.toml; the repo itself is mounted at run time).
RUN python -m pip install --no-cache-dir -c /opt/constraints.txt \
        setproctitle hydra-core omegaconf wandb einops termcolor pygame onnxruntime onnxscript \
        jaxtyping "tensordict==0.7.0" "torchrl==0.7.0" imageio av moviepy mujoco linuxfd pdbpp \
        "simple-raycaster @ git+https://github.com/btx0424/simple-raycaster.git"

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR ${AA_PATH}
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
