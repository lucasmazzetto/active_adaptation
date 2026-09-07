# Isaac Sim 5.1.0 + Isaac Lab v2.3.2 + this repository's dependencies.
ARG ISAACSIM_VERSION=5.1.0
FROM nvcr.io/nvidia/isaac-sim:${ISAACSIM_VERSION}

ENV ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y \
    OMNI_KIT_ALLOW_ROOT=1 \
    LANG=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    ISAACSIM_ROOT_PATH=/isaac-sim \
    ISAACLAB_PATH=/workspace/isaaclab \
    AA_PATH=/workspace/active_adaptation

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git wget libglib2.0-0 ncurses-term jq \
    && rm -rf /var/lib/apt/lists/*

# Isaac Lab, installed into Isaac Sim's own Python. "--install none" skips RL libraries we do not use.
RUN git clone --depth 1 -b v2.3.2 https://github.com/isaac-sim/IsaacLab.git ${ISAACLAB_PATH} \
    && ln -sf ${ISAACSIM_ROOT_PATH} ${ISAACLAB_PATH}/_isaac_sim \
    && ${ISAACLAB_PATH}/isaaclab.sh --install none

# Freeze the versions Isaac Sim ships (torch, numpy, warp) so later pip installs cannot change them.
COPY docker/make_constraints.py /opt/make_constraints.py
RUN ${ISAACLAB_PATH}/isaaclab.sh -p /opt/make_constraints.py > /opt/constraints.txt && cat /opt/constraints.txt

# Dependencies of this repo (mirrors pyproject.toml; the repo itself is mounted at run time).
RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install -c /opt/constraints.txt \
        setproctitle hydra-core omegaconf wandb einops termcolor pygame onnxruntime onnxscript \
        jaxtyping "tensordict==0.7.0" "torchrl==0.7.0" imageio av moviepy mujoco linuxfd pdbpp \
        "simple-raycaster @ git+https://github.com/btx0424/simple-raycaster.git"

COPY docker/isaac_env.sh /opt/isaac_env.sh
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR ${AA_PATH}
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
