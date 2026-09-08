# PLAN.md — Train the FACET policy (Go2) in Docker with Isaac Sim

This document is written for an engineer who has not worked in this repository before.
Every topic is split into small tasks. Do them in order. Each task says **what to do**,
**how to check it worked**, and **why** it is needed. Commands are meant to be run from
the repository root (`/mnt/Shared/Projects/active_adaptation`) unless stated otherwise.

> Task 0: when this plan is approved, save this exact document as `PLAN.md` in the
> repository root and commit it on a new branch (`git checkout -b fix/facet-go2-docker`).

---

## 0. Read this first: what we are doing and why

### 0.1 What the code base is

`active_adaptation` is a reinforcement-learning (RL) framework for legged robots. It
runs the robot in a physics simulator (Isaac Sim through Isaac Lab, or MuJoCo), collects
experience from thousands of parallel copies of the robot, and trains a neural-network
policy with PPO (Proximal Policy Optimization).

The framework is used by several papers. The one we care about is **FACET**
(*Force-Adaptive Control via Impedance Reference Tracking for Legged Robots*,
arXiv 2505.06883). Its code lives in `projects/facet/`.

### 0.2 What FACET does, in one picture

Instead of telling the robot "walk at 1 m/s", FACET tells it "behave like a virtual
mass on a spring and damper". The command manager simulates that virtual mass–spring–
damper (the *reference dynamics*) and the policy is rewarded for making the real robot
follow it. When someone pushes the robot, the virtual system yields, so the robot yields
too. That is what "compliance" means here.

```
                 setpoint (where the spring is anchored)
                        |
                        v
   external force ---> [ virtual mass m, spring kp, damper kd ]  ---> reference pos/vel
   (impulses we                                                        for the next
    inject in sim)                                                     8 / 16 / 32 steps
                                                                             |
   robot state --------------------------------------------------------------+---> rewards
        ^                                                                    |
        |                                                                    v
   simulator <---- joint targets <---- policy network <---- observations (command, imu,
                                                              joint states, prev actions)
```

Key files:

| File | What it contains |
|---|---|
| `projects/facet/facet_commands/impedance_manip.py` | `ImpedanceCommandManager`: samples spring parameters, integrates the reference dynamics, injects impulse forces, exposes the `command` and `command_hidden` observations |
| `projects/facet/facet_commands/rewards.py` | `impedance_pos`, `impedance_vel`, `impedance_yaw_pos`, `impedance_yaw_vel` (+ end-effector versions) |
| `projects/facet/facet_commands/impedance.py` | older `Impedance` class from the paper era. **We do not use it.** |
| `active_adaptation/learning/ppo/ppo_facet.py` | the FACET PPO variant: a *teacher* actor that sees privileged information, and a *student* actor fed by a GRU that estimates that information from proprioception only |
| `cfg/train.yaml` | Hydra root config for training (`task=...`, `algo=...`, `backend=...`) |
| `scripts/train_ppo.py` | the training loop |

### 0.3 Why it does not run today

The framework was refactored after FACET was written, and FACET was not updated.
Concretely (all verified by reading the code):

1. The FACET task YAMLs sit in `projects/facet/task/`, a folder the project loader never
   scans. Hydra cannot find `task=Go2Force`.
2. `Go2Force.yaml` uses config keys and class names that no longer exist
   (`action:` instead of `input:`, `alpha` instead of `alpha_range`, observations
   `command_hidden`/`applied_torque`, termination `soft_contact`, extra kwargs on
   `motor_params` and `undesired_contact`, `command:` with no value).
3. `ppo_facet.py` does not match the current `PPOBase` contract
   (`get_rollout_policy` lacks the `critic` argument, `compute_value` is missing,
   `load_state_dict` refers to attributes that do not exist, and the symmetry code
   crashes when `symaug=False`).
4. There is no Isaac Sim on this machine. We will run it in Docker.

### 0.4 How the pieces fit once we are done

```
 host (Ubuntu 24.04, RTX 3070 Ti 8 GB, driver 595, Docker + NVIDIA toolkit)
 └── docker compose (docker/compose.yaml)
     └── container: nvcr.io/nvidia/isaac-sim:6.0.0
         ├── /isaac-sim                 Isaac Sim 6.0.0 + its bundled Python 3.12 (+ torch 2.10 cu128 from Isaac Lab)
         ├── /workspace/isaaclab        Isaac Lab v3.0.0-beta2.patch1 (git clone, `isaaclab.sh -i core`)
         └── /workspace/active_adaptation   <-- this repo, bind-mounted from the host
             ├── .cache/projects.json    (created by entrypoint; enables the `facet` project)
             ├── projects/cfg/task/facet/Go2Force.yaml   (moved + fixed task config)
             └── outputs/...             (checkpoints, offline wandb runs)
```

Config resolution at run time:

```
 python scripts/train_ppo.py task=facet/Go2Force algo=ppo_facet_train
        |
        | import active_adaptation  -> registers a Hydra search-path plugin
        |                           -> reads .cache/projects.json
        |                           -> for every ENABLED project adds <pkg>/../cfg to Hydra
        |                              (projects/facet -> projects/cfg)
        v
 Hydra loads cfg/train.yaml + projects/cfg/task/facet/Go2Force.yaml + algo from ConfigStore
        |
        v
 aa.init()  -> starts Isaac Sim (AppLauncher) -> imports projects.facet
                                               (this registers ImpedanceCommandManager,
                                                impedance_* rewards by class name)
        v
 make_env_policy() -> builds env from YAML (each YAML key = a registered class name)
                   -> builds ppo_facet.PPOPolicy
```

---

## 1. Prerequisites on the host

### Task 1.1 — Confirm the GPU and Docker stack
Run:
```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
docker --version && docker compose version
docker info | grep -i runtimes          # must list "nvidia"
nvidia-ctk --version
df -h /var/lib/docker                    # need >= 40 GB free for the image
```
Expected (already checked on this machine): RTX 3070 Ti Laptop, 8192 MiB, driver 595.x,
Docker 29, Compose v5, runtime list contains `nvidia`, ~68 GB free.

### Task 1.2 — Confirm the base image is reachable
```bash
docker manifest inspect nvcr.io/nvidia/isaac-sim:6.0.0 | head -5
```
Expected: JSON starting with `"schemaVersion": 2`. No NGC login is required.

### Task 1.3 — Know the memory budget
Isaac Sim itself uses about 3 GB of the 8 GB VRAM. The paper trains with 4096 robots;
that will not fit here. We will train with 1024–2048 robots (Topic 9). Do not change
`num_envs` inside the YAML; override it on the command line.

---

## 2. Docker files (do this first so the image builds while you edit code)

Create a new folder `docker/` in the repo root with the five files below.

### Task 2.1 — `docker/Dockerfile`
The file below is the one in the repo (Isaac Sim 6.0 + Isaac Lab 3.0). Read the comments: each
`RUN` exists because of a concrete failure met while building.
```dockerfile
# Isaac Sim 6.0 + Isaac Lab 3.0 + this repository's dependencies.
#
# Isaac Sim 6.0 is the first release whose RTX renderer starts on NVIDIA driver 595.x (5.1's
# crashes in librtx.scenedb on start-up, headless renders included). It pairs with Isaac Lab 3.0,
# which this codebase targets natively (see active_adaptation/envs/backends/isaac/adapter.py).
ARG ISAACSIM_VERSION=6.0.0
FROM nvcr.io/nvidia/isaac-sim:${ISAACSIM_VERSION}
ARG ISAACSIM_VERSION
ARG ISAACLAB_TAG=v3.0.0-beta2.patch1

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

# Isaac Lab, installed into Isaac Sim's own Python. `-i core` installs the core submodules
# (isaaclab, isaaclab_physx, ...) without the RL extras; its installer also installs torch
# (cu128) itself, since Isaac Sim 6.0 bundles none.
RUN git clone --depth 1 -b ${ISAACLAB_TAG} https://github.com/isaac-sim/IsaacLab.git ${ISAACLAB_PATH} \
    && ln -sf ${ISAACSIM_ROOT_PATH} ${ISAACLAB_PATH}/_isaac_sim \
    && ${ISAACLAB_PATH}/isaaclab.sh -i core \
    && bash -c 'source /opt/isaac_env.sh && python -c "import isaaclab, isaaclab_physx; print(\"isaaclab import OK:\", isaaclab.__file__)"'

# Freeze torch/torchvision/numpy AFTER the Isaac Lab install (that is where torch comes from)
# so the repo's dependency install cannot change them. isaac_env.sh must be sourced for the
# bundled packages to be importable. Keep only requirement lines.
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
# *different* warp-lang, remove it so there is exactly one warp. The invariant checked below is:
# the importable warp IS the bundled extension copy (same version, same path).
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
```
Why: the official Isaac Lab Dockerfile does the same steps (clone, symlink `_isaac_sim`,
`isaaclab.sh -i core`). Torch comes from the Isaac Lab installer (Isaac Sim 6.0 bundles none),
and the constraints file freezes it before the repo's own dependencies are installed.

### Task 2.2 — `docker/make_constraints.py`
```python
# Prints pip constraint lines for the packages Isaac Sim bundles, so they are never upgraded.
import importlib
for name, dist in [("torch", "torch"), ("torchvision", "torchvision"), ("numpy", "numpy"), ("warp", "warp-lang")]:
    try:
        version = importlib.import_module(name).__version__.split("+")[0]
        print(f"{dist}=={version}")
    except Exception:
        pass
```

### Task 2.3 — `docker/isaac_env.sh`
Isaac Sim's Python only works when a few environment variables point at the Isaac Sim
tree. Isaac Sim's own `python.sh` sets them; we copy that logic so plain `python`,
`pip`, and the `aa-*` console scripts work inside the container.
```bash
# Source Isaac Sim's Python environment (same variables as /isaac-sim/python.sh).
export ISAAC_PATH=/isaac-sim
export CARB_APP_PATH=/isaac-sim/kit
export EXP_PATH=/isaac-sim/apps
source /isaac-sim/setup_python_env.sh
export PATH=/isaac-sim/kit/python/bin:${PATH}
```
Check while building: run `docker compose run --rm aa cat /isaac-sim/python.sh` once the
image exists and confirm it exports the same three variables and sources the same file.
If the names differ in this Isaac Sim version, copy the exports from that file instead.

### Task 2.4 — `docker/entrypoint.sh`
```bash
#!/usr/bin/env bash
# Runs on every container start.
set -euo pipefail
source /opt/isaac_env.sh
cd /workspace/active_adaptation

# 1. Register this repo (editable) with Isaac Sim's Python. --no-deps: deps were installed at build time.
if ! python -m pip show active_adaptation >/dev/null 2>&1; then
    python -m pip install --no-deps -e /workspace/active_adaptation
fi

# 2. Discover projects and enable only `facet`.
#    The Hydra search-path plugin and the project importer both ignore disabled projects,
#    so without this step neither the task YAML nor ImpedanceCommandManager is found.
aa-discover-projects >/dev/null
python - <<'EOF'
import json, pathlib
p = pathlib.Path("/workspace/active_adaptation/.cache/projects.json")
d = json.loads(p.read_text())
for name, info in d.get("environment", {}).items():
    info["enabled"] = (name == "facet")
p.write_text(json.dumps(d, indent=2))
print("projects.json:", {k: v["enabled"] for k, v in d["environment"].items()})
EOF

exec "$@"
```

### Task 2.5 — `docker/compose.yaml`
```yaml
services:
  aa:                                  # headless training / evaluation
    build:
      context: ..
      dockerfile: docker/Dockerfile
      args:
        ISAACSIM_VERSION: ${ISAACSIM_VERSION:-6.0.0}
        ISAACLAB_TAG: ${ISAACLAB_TAG:-v3.0.0-beta2.patch1}
    # Isaac Sim 6.0 + Isaac Lab 3.0: the only supported pairing. The codebase targets the Isaac
    # Lab 3.0 API natively (XYZW quaternions converted in envs/backends/isaac/adapter.py), and
    # 6.0's renderer is the first that starts on NVIDIA driver 595.x, so the GUI works too.
    image: active-adaptation-isaac:${ISAACSIM_VERSION:-6.0.0}
    runtime: nvidia
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    environment:
      - NVIDIA_DRIVER_CAPABILITIES=all      # needed for Vulkan/RTX inside the container
      - ACCEPT_EULA=Y
      - PRIVACY_CONSENT=Y
      - OMNI_KIT_ALLOW_ROOT=1
      - WANDB_MODE=${WANDB_MODE:-offline}   # offline by default; wandb.init honours this variable
      - WANDB_API_KEY=${WANDB_API_KEY:-}
    env_file:
      - path: .env
        required: false
    shm_size: "8gb"
    network_mode: host
    stdin_open: true
    tty: true
    volumes:
      - ..:/workspace/active_adaptation          # code, outputs/, .cache/
      - isaac-cache-kit:/isaac-sim/kit/cache      # shader/extension caches: make 2nd start fast
      - isaac-cache-ov:/root/.cache/ov
      - isaac-cache-pip:/root/.cache/pip
      - isaac-cache-gl:/root/.cache/nvidia/GLCache
      - isaac-cache-compute:/root/.nv/ComputeCache
      - isaac-logs:/root/.nvidia-omniverse/logs
      - isaac-data:/root/.local/share/ov/data

  aa-gui:                              # play.py with a window (run `xhost +local:root` on the host first)
    extends: aa
    profiles: [gui]
    environment:
      - DISPLAY=${DISPLAY}
      - NVIDIA_DRIVER_CAPABILITIES=all
      - ACCEPT_EULA=Y
      - PRIVACY_CONSENT=Y
      - OMNI_KIT_ALLOW_ROOT=1
    volumes:
      - /tmp/.X11-unix:/tmp/.X11-unix:rw

volumes:
  isaac-cache-kit: {}
  isaac-cache-ov: {}
  isaac-cache-pip: {}
  isaac-cache-gl: {}
  isaac-cache-compute: {}
  isaac-logs: {}
  isaac-data: {}
```
Why: `network_mode: host` and offline wandb keep the container self-contained; named volumes
keep Kit's shader and extension caches, so the second start is fast. The `gui` profile shares
the X11 socket for `play.py ... headless=false`.

### Task 2.6 — `docker/.env.example` and `.gitignore`
`docker/.env.example`:
```
# Copy to docker/.env (gitignored). Leave WANDB_API_KEY empty for offline runs.
WANDB_MODE=offline
WANDB_API_KEY=
```
Append to `.gitignore`:
```
docker/.env
```

### Task 2.7 — Build the image (takes 20–40 minutes, ~15 GB download)
```bash
cd docker
docker compose build
```
Check:
```bash
docker compose run --rm aa python -c "import torch, isaaclab, tensordict, torchrl; print(torch.__version__, torch.cuda.is_available())"
```
Expected: a line like `2.10.0+cu128 True` after the entrypoint prints `projects.json: {...}`.
While this builds, continue with Topic 3.

---

## 3. Move the FACET task configs to where the loader looks

### Why
`active_adaptation/project_loading/discovery.py::_task_dir_for_path` and
`project_loading/plugin.py::collect_environment_cfg_paths` both compute the config folder
of a project as *the folder next to the package* + `cfg`. For `projects/facet` that is
**`projects/cfg/`** (it already exists and holds `Quadruped.yaml`). The FACET YAMLs are
in `projects/facet/task/`, which nothing scans.

### Task 3.1 — Move the files with git
```bash
mkdir -p projects/cfg/task/facet
git mv projects/facet/task/facet/Go2Force.yaml projects/cfg/task/facet/Go2Force.yaml
git mv projects/facet/task/facet/B1Z1.yaml     projects/cfg/task/facet/B1Z1.yaml
git rm  projects/facet/task/facet/Go2ForceHIM.yaml      # legacy Impedance class + stale paths; git history keeps it
rmdir -p projects/facet/task/facet 2>/dev/null || true
```
Add this comment as the first line of `B1Z1.yaml`:
`# NOTE: not yet repaired for the current framework (see PLAN.md, Topic 11).`

### Task 3.2 — Check
```bash
ls projects/cfg/task/facet/         # Go2Force.yaml  B1Z1.yaml
ls projects/facet/                  # no `task` folder any more
```
After Topic 9 starts the container, `aa-list-tasks` must print `facet/Go2Force`.

---

## 4. Rewrite `projects/cfg/task/facet/Go2Force.yaml`

### Why
Each key under `input`, `observation`, `reward`, `termination`, `randomization` is looked
up **by name** in a class registry (`active_adaptation/registry.py::RegistryMixin.make`).
An unknown name raises `ValueError: Class 'X' not found`. Extra constructor arguments
raise `TypeError`. The table lists every mismatch found by comparing the YAML with the
constructors in `active_adaptation/envs/mdp/**`.

| Old | New | Reason |
|---|---|---|
| top-level `action:` | `input: { action: ... }` | `env_base.py` reads `cfg.input` only |
| `_target_: active_adaptation.envs.mdp.action.JointPosition` | `_target_: JointPosition` | registry keys are bare class names |
| `alpha: [0.5, 1.0]` | `alpha_range: [0.5, 1.0]` | `JointPosition.__init__` parameter name |
| `command:` (empty value) inside observation groups | `command: {}` | `parse_component_spec` rejects `None` ("must be a mapping") |
| `# priv:` commented out | real `priv:` group | `ppo_facet` needs a `priv` observation group |
| `command_hidden` | keep, but re-enable the class (Topic 5) | class is commented out today |
| `applied_torque` | remove | class is commented out; not needed |
| `undesired_contact: {..., enabled: true}` (reward) | drop `enabled` | reward ctor is `(env, body_names, weight)` |
| `soft_contact` (termination) | `undesired_contact: {body_names: [Head.*, .*_calf]}` | `soft_contact` does not exist |
| `motor_params.actuator_name`, `motor_params.scale_factor_range` | remove | ctor is `(env, stiffness_range, damping_range, armature_range)` |

### Task 4.1 — Replace the whole file with this content
```yaml
# @package task
# FACET (arXiv 2505.06883) on Unitree Go2: track a virtual mass-spring-damper under external forces.
name: Go2Force

viewer:
  resolution: [1280, 720]
  lookat: [0., 0., 0.]
  eye: [4.0, 4.0, 4.0]

robot:
  name: go2
terrain: plane

num_envs: 4096            # override on the CLI for small GPUs, e.g. task.num_envs=1024
max_episode_length: 1000

sim:
  step_dt: 0.02
  isaac_physics_dt: 0.005
  mujoco_physics_dt: 0.002

input:
  action:
    _target_: JointPosition
    action_scaling: {.*_joint: 0.5}
    max_delay: 1
    alpha_range: [0.5, 1.0]

command:
  _target_: ImpedanceCommandManager
  virtual_mass_range: [2., 4., 6.]   # Go2 base mass is ~6.9 kg
  eef_body_name: null
  arm_body_name: null
  ref_steps: [8, 16, 32]             # reference states the rewards look at
  temporal_smoothing: 32

observation:
  policy:                            # what the deployed policy sees
    command: {}
    gravity_multistep:   {steps: 4, noise_std: 0.05}
    joint_pos_multistep: {steps: 4, noise_std: 0.05}
    joint_vel_multistep: {steps: 4, noise_std: 0.2}
    prev_actions:        {steps: 3}
  priv:                              # privileged: teacher only, student must estimate it
    command_hidden: {}
    root_linvel_b:  {yaw_only: true}
    root_angvel_b:  {}
    contact_indicator: {body_names: [".*_foot"]}
  ext:                               # true external wrench (privileged)
    external_forces:  {body_names: ["base"]}
    external_torques: {body_names: ["base"]}

reward:
  loco:
    impedance_yaw_pos:  {weight: 0.75}
    impedance_yaw_vel:  {weight: 1.0}
    impedance_pos:      {weight: 0.75}
    impedance_vel:      {weight: 1.5}
    angvel_xy_l2:       {weight: 0.01}
    linvel_z_l2:        {weight: 2.0}
    base_height_l1:     {weight: 3.0, target_height: 0.35}
    energy_l1:          {weight: 0.0002}
    joint_acc_l2:       {weight: 2.5e-7}
    joint_torques_l2:   {weight: 2.0e-4}
    survival:           {weight: 1.0}
    action_rate_l2:     {weight: 0.01}
    action_rate2_l2:    {weight: 0.005}
    feet_sliding:       {weight: 0.2, body_names: .*_foot}
    feet_air_time:      {weight: 0.2, body_names: .*_foot, thres: 0.3}
    undesired_contact:  {weight: 2.0, body_names: [Head.*, .*_calf, .*thigh]}
  debug:
    _enabled_: false
    feet_sliding:       {weight: 1.0, body_names: .*_foot}
    feet_contact_count: {weight: 1.0, body_names: .*_foot}

termination:
  crash: {body_names_expr: [Head.*, base, .*_calf], t_thres: 0.1}
  fall_over: {xy_thres: 0.9}
  undesired_contact: {body_names: [Head.*, .*_calf]}

randomization:
  perturb_body_mass:
    (?!(payload|base|Head.*)).*: [0.8, 1.2]
    base: [0.8, 1.2]
  perturb_body_materials:
    body_names: [.*_foot, .*_calf]
    static_friction_range:  [0.5, 4.0]
    dynamic_friction_range: [0.5, 4.0]
    restitution_range:      [0.0, 0.5]
  motor_params:
    stiffness_range: {.*: [0.8, 1.2]}
    damping_range:   {.*: [0.8, 1.2]}
    armature_range:  {.*: [0.0, 0.02]}
  reset_joint_states_scale:
    pos_scales: {.*: [0.8, 1.2]}
  random_joint_offset:
    .*hip_joint:   [-0.05, 0.05]
    .*thigh_joint: [-0.05, 0.05]
    .*calf_joint:  [-0.05, 0.05]
```

### Task 4.2 — Check every class name exists
```bash
for n in JointPosition ImpedanceCommandManager command command_hidden gravity_multistep joint_pos_multistep \
         joint_vel_multistep prev_actions root_linvel_b root_angvel_b contact_indicator external_forces \
         external_torques impedance_yaw_pos impedance_yaw_vel impedance_pos impedance_vel angvel_xy_l2 \
         linvel_z_l2 base_height_l1 energy_l1 joint_acc_l2 joint_torques_l2 survival action_rate_l2 \
         action_rate2_l2 feet_sliding feet_air_time undesired_contact feet_contact_count crash fall_over \
         perturb_body_mass perturb_body_materials motor_params reset_joint_states_scale random_joint_offset; do
  printf "%-26s " $n; grep -rlE "^class $n\b" active_adaptation projects | head -1; echo
done
```
Every line must show a file. `command_hidden` will only pass after Topic 5.

---

## 5. Re-enable the `command_hidden` observation

### Why
The teacher benefits from seeing where the virtual system will be in 8/16/32 steps.
`ImpedanceCommandManager.command_hidden` already computes that tensor, but the
observation class that forwards it is commented out in
`active_adaptation/envs/mdp/observations/common.py` (lines ~79–100). The commented
version also hard-codes a symmetry layout for the *old* `Impedance` class, which is wrong
for the new manager. We restore the class and let the command manager own the symmetry.

Symmetry background: the policy is trained with left/right mirroring
(`ppo_facet.symaug`). Every observation must say how it mirrors: a permutation plus a
sign per element (`active_adaptation/utils/symmetry.py::SymmetryTransform`). For a
vector in the robot frame, mirroring flips `y`. For roll/pitch/yaw, mirroring flips
roll and yaw. For angular velocity, mirroring flips `wx` and `wz`.

### Task 5.1 — Restore the observation class in `observations/common.py`
Replace the commented block with:
```python
class command_hidden(Observation):
    """Privileged view of the command manager (e.g. future reference states)."""
    def __init__(self, env):
        super().__init__(env)
        self.command_manager = self.env.command_manager

    @override
    def compute(self):
        return self.command_manager.command_hidden

    @override
    def symmetry_transform(self):
        return self.command_manager.symmetry_transform_hidden()
```
(`@override` is already imported in this file; the `command` class above it uses it.)

### Task 5.2 — Add `symmetry_transform_hidden` to `ImpedanceCommandManager`
In `projects/facet/facet_commands/impedance_manip.py`, right after the existing
`symmetry_transform` method, add:
```python
    def symmetry_transform_hidden(self):
        """Mirror map for `command_hidden`; must match the concatenation order in that property."""
        n = len(self.ref_steps)
        flip_y = SymmetryTransform(perm=torch.arange(3), signs=torch.tensor([1., -1., 1.]))          # xyz vectors
        flip_roll_yaw = SymmetryTransform(perm=torch.arange(3), signs=torch.tensor([-1., 1., -1.]))  # rpy / angular vel
        parts = [
            flip_y.repeat(n),         # base position error, n steps
            flip_y.repeat(n),         # base linear velocity, n steps
            flip_roll_yaw.repeat(n),  # base rpy error, n steps
            flip_roll_yaw.repeat(n),  # base angular velocity, n steps
        ]
        if self.has_eef:
            parts += [flip_y.repeat(n), flip_y.repeat(n)]   # eef position error, eef velocity
        return SymmetryTransform.cat(parts)
```

### Task 5.3 — Check the sizes match
Open `command_hidden` in the same file and count: 4 blocks × `n` steps × 3 numbers
(= 36 for `ref_steps: [8, 16, 32]`). The transform above has the same length. The env
will assert this indirectly when `ppo_facet` calls `symmetry_transform()` on the group.

---

## 6. Fix the hard-coded marker path

`impedance_manip.py::make_frame_marker` loads a USD file from another developer's home
directory. It only runs when a GUI is attached (`play.py`), but it would crash there.

### Task 6.1 — Use Isaac Lab's built-in frame marker
Replace the body of `make_frame_marker` with:
```python
def make_frame_marker(name: str):
    cfg = FRAME_MARKER_CFG.replace(prim_path=f"/Visuals/Command/{name}")
    cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker = VisualizationMarkers(cfg)
    marker.set_visibility(True)
    return marker
```
`FRAME_MARKER_CFG` is already imported at the top of the file (inside the
`if active_adaptation.get_backend() == "isaac":` block).

---

## 6b. (added during implementation) External wrench API changed in Isaac Lab 2.3

**Not in the original plan; found while implementing.** Isaac Lab up to 2.2 exposed the raw
buffers `Articulation._external_force_b` / `_external_torque_b` plus a `has_external_wrench`
flag. **Isaac Lab 2.3.2 removed all three** and replaced them with `WrenchComposer`, reached
as `asset.permanent_wrench_composer`. Verified against the installed tree and against
`isaaclab/utils/wrench_composer.py` at tag v2.3.2 (`_external_force_b` appears zero times in
`articulation.py` and `rigid_object.py`).

Three places in this repository still used the old buffers and would raise `AttributeError`
on the first physics substep:

1. `ImpedanceCommandManager.pre_step` — applying the FACET impulse forces.
2. `external_forces` / `external_torques` observations (`observations/extero.py`) — the whole
   `ext` group that `ppo_facet` requires.
3. A gait reward in `rewards/locomotion.py` that adds external foot forces to ground reaction force.

The MuJoCo backend in this repo (`envs/backends/mujoco/mujoco.py`) still uses the old buffers,
so the code has to support both.

### Task 6b.1 — Add a compatibility layer
New file `active_adaptation/envs/mdp/utils/wrench.py` with three functions:

* `add_external_wrench(asset, forces, torques, body_ids, positions, is_global)` — uses
  `permanent_wrench_composer.add_forces_and_torques(...)` when present, else `+=` on the old
  buffers and sets `has_external_wrench`. Accumulates, matching the `+=` the old code used.
* `get_external_force_b(asset)` / `get_external_torque_b(asset)` — return
  `composed_force_as_torch` / `composed_torque_as_torch` when present, else the old buffers.
  Both are `(num_envs, num_bodies, 3)` in the body link frame, so call sites are unchanged
  apart from the accessor.

Why the *permanent* composer: `write_data_to_sim` folds permanent into instantaneous and then
resets only the instantaneous one, while the environment clears both at the start of every
substep via `scene.zero_external_wrenches()`. So a permanent wrench written in `pre_step`
survives the step and is still readable by the observation `update()` that runs after the
substep loop. That preserves the original ordering.

### Task 6b.2 — Use it at the three call sites
Replace the direct buffer access with the helpers. In the command manager the world-frame
impulse is still rotated into the body frame explicitly (`quat_rotate_inverse` on
`root_link_quat_w`) and applied to body 0 with `body_ids=[0]`, so behaviour matches the
original code rather than relying on the composer's `is_global` path.

### Task 6b.3 — Check
```bash
grep -rn "_external_force_b\|_external_torque_b\|has_external_wrench" active_adaptation projects --include=*.py
```
The only remaining hits should be: `utils/wrench.py` itself, the MuJoCo backend, the Isaac
adapter's `zero_external_wrenches` (already `hasattr`-guarded), the legacy fallback inside
`randomizations/common.py::_set_external_wrench`, and the unused `impedance.py`.

---

## 6c. (added during implementation) Four more blockers found by actually running it

Each of these stopped a run outright and is fixed on this branch.

### Task 6c.1 — tensordict/torchrl must be recent
The whole codebase reads `tensordict["action_log_prob"]`. That is only the default log-prob
key once `composite_lp_aggregate` defaults to False. Measured in the container:

| tensordict / torchrl | `composite_lp_aggregate()` | log-prob key |
|---|---|---|
| 0.7.0 / 0.7.0 | True | `sample_log_prob` -> every PPO update raises `KeyError` |
| 0.8.3 / 0.8.1 | True | `sample_log_prob` |
| 0.13.0 / 0.13.3 | False | `action_log_prob` (what the code wants) |

The repo's `uv.lock` pins 0.7.0, but that lock is stale: it belongs to `venv/isaaclab`, which
depends on a `projects/hdmi` package that does not exist in this tree. The README's
`pip install -U tensordict torchrl` is what the code targets. The Dockerfile pins the verified
`tensordict==0.13.0` / `torchrl==0.13.3` (reproducible, and new enough), with torch held at
Isaac Sim's 2.7.0 by the constraints file.

### Task 6c.2 — `ScopedTimer` crashed after training
`active_adaptation/utils/profiling.py` keys timers by name as singletons and raised
`ValueError: Timer simulation already has a parent rollout` if the same scope was entered
under a different parent. That happens on every run: `simulation` nests under `rollout`
while collecting, then runs bare during the post-training evaluation. Fix: record the parent
on first use and never re-parent or raise. The link only shapes the printed summary, and
timing accumulates on the singleton either way.

### Task 6c.3 — offline wandb runs have no name
`scripts/train_ppo.py` did `run.name.split("-")` , but wandb only auto-generates a name when
syncing online; offline and disabled runs leave it `None`. Fix: fall back to `run.id`.
This matters because the chosen default for this project is `wandb.mode=offline`.

### Task 6c.4 — `cfg/eval.yaml` could not start
It was the only entry config missing `backend` and `device`, so `scripts/eval.py` died in
`active_adaptation.init()` with `Missing key backend`. It also had a stray YAML tag,
`enable_cameras: ! ${..eval_render}`. Both fixed.

---

## 7. Bring `ppo_facet.py` up to the current `PPOBase` contract

File: `active_adaptation/learning/ppo/ppo_facet.py`. Reference for the contract:
`ppo_base.py` and the working `ppo.py`. The training script (`scripts/train_ppo.py`)
calls `policy.get_rollout_policy("train", critic=...)`, `policy.train_op(data)`, and,
when `store_transitions=False`, `policy.compute_value(td)`.

```
 phase "train"     : vecnorm -> encoder_priv -> actor_teacher      (PPO on teacher; GRU adapt trained every 2nd update)
 phase "adapt"     : vecnorm -> adapt_module  -> actor_student     (only GRU regression; optional)
 phase "finetune"  : vecnorm -> adapt_ema     -> actor_student     (PPO on student + GRU regression)
```

### Task 7.1 — `get_rollout_policy` must accept `critic`
Replace the method with:
```python
    def get_rollout_policy(self, mode: str = "train", critic: bool = False):
        modules = [self.vecnorm]
        if self.cfg.phase == "train":
            modules += [self.encoder_priv, self.actor_teacher]
        elif self.cfg.phase == "adapt":
            modules += [self.adapt_module, self.actor_student]
        elif self.cfg.phase == "finetune":
            modules += [self.adapt_ema, self.actor_student]   # EMA copy for stability
        if critic:
            modules.append(self.critic)
        return Seq(*modules)
```

### Task 7.2 — Add `compute_value`
Add next to it:
```python
    def compute_value(self, tensordict: TensorDict):
        self.vecnorm(tensordict)
        return self.critic(tensordict)
```

### Task 7.3 — Only build the mirrored batch when `symaug` is on
In `_update`, the block that creates `symmetry = tensordict.empty()` … `torch.cat(...)`
runs unconditionally, but `self.obs_transform` etc. only exist when `symaug=True`
(and `ppo_facet_adapt`/`ppo_facet_finetune` set `symaug=False`). Wrap it:
```python
        bsize = tensordict.shape[0]
        if self.cfg.symaug:
            symmetry = tensordict.empty()
            symmetry["_obs_normed"]  = self.obs_transform(tensordict["_obs_normed"])
            symmetry["_priv_normed"] = self.priv_transform(tensordict["_priv_normed"])
            symmetry["_ext_normed"]  = self.ext_transform(tensordict["_ext_normed"])
            symmetry[ACTION_KEY]     = self.act_transform(tensordict[ACTION_KEY])
            for k in ("action_log_prob", "is_init", "step_count", "ret", "adv"):
                symmetry[k] = tensordict[k]
            tensordict = torch.cat([tensordict, symmetry], dim=0)
```
and, further down inside `with torch.no_grad():`, replace the `symmetry_loss` line with:
```python
            if self.cfg.symaug:
                symmetry_loss = F.mse_loss(tensordict["loc"][bsize:], self.act_transform(tensordict["loc"][:bsize]))
            else:
                symmetry_loss = torch.zeros((), device=self.device)
```

### Task 7.4 — Fix `load_state_dict`
`self.actor` and `self.actor_adapt` do not exist on this class. Replace the method with:
```python
    def load_state_dict(self, state_dict, strict=True):
        failed_keys = super().load_state_dict(state_dict, strict)
        if state_dict.get("last_phase", "train") == "train" and self.cfg.phase != "train":
            # Starting adapt/finetune from a teacher checkpoint: initialise the student from the teacher once.
            hard_copy_(self.actor_teacher, self.actor_student)
        return failed_keys
```
`hard_copy_` is already imported through `from .common import *`.

### Task 7.6 — (added during implementation) Rename the command manager's force hook

**Not in the original plan; found while implementing.** `ImpedanceCommandManager` applied its
impulse forces from a method named `step(self, substep)`. The environment never calls a method
with that name: `env_base.py:214` wires `self.command_manager.pre_step`, and `MDPComponent`
defines the hooks `reset / update / pre_step / post_step / startup / debug_draw` only. The
external forces therefore never reached the robot, so the trained policy would never experience
a push. That contradicts the whole method and the Definition of Done in Topic 12.

Fix applied in `projects/facet/facet_commands/impedance_manip.py`: rename `step` to `pre_step`
and leave a comment saying why. The old `Impedance` class in `impedance.py` has the same defect
but is unused (Topic 11).

### Task 7.5 — Check the file still parses
```bash
python3 -c "import ast; ast.parse(open('active_adaptation/learning/ppo/ppo_facet.py').read()); print('ok')"
```

---

## 8. Documentation updates

### Task 8.1 — README
Add a section **"FACET (Go2) in Docker"** to `README.md` containing the commands from
Topic 9 and the note that this GPU needs `task.num_envs=1024..2048`. Replace the stale
debug example (`task=Go2/Go2Force algo=ppo_dic_train`) with
`task=facet/Go2Force algo=ppo_facet_train`.

### Task 8.2 — `projects/facet/readme.md`
Fill the empty "Usage" block with the same commands.

---

## 9. Run it

All commands from the `docker/` folder. `docker compose run --rm aa <cmd>` starts a
fresh container, runs the entrypoint (installs the repo, enables `facet`), then `<cmd>`.

### Task 9.1 — The task must be visible
```bash
docker compose run --rm aa aa-list-tasks | grep facet
```
Expected: `facet/Go2Force` (and `facet/B1Z1`). If not, see Topic 10.

### Task 9.2 — Smoke test (a few minutes, 64 robots, no wandb)
```bash
docker compose run --rm aa python scripts/train_ppo.py \
    task=facet/Go2Force algo=ppo_facet_train \
    task.num_envs=64 total_frames=200_000 wandb.mode=disabled
```
What you must see in the log, in this order:
1. `Importing project: facet from ...`
2. `Reward group: loco` followed by lines `impedance_yaw_pos: 0.75`, `impedance_pos: 0.75`, …
3. `Creating policy <class 'active_adaptation.learning.ppo.ppo_facet.PPOPolicy'> on device cuda:0`
4. `[Info]: Add TensorDictPrimer ...` (the GRU hidden state `adapt_hx`)
5. A progress bar; ~97 iterations (200 000 frames / (64 robots × 32 steps)).
6. Logged numbers `actor/policy_loss`, `adapt/priv_loss`, `adapt/ext_loss` are finite, and
   `train/stats/episode_len` increases over the run (robots survive longer).

The first start compiles shaders and can take 3–5 minutes before the progress bar. The
named cache volumes make later starts much faster.

### Task 9.3 — Real teacher run
```bash
docker compose run --rm aa python scripts/train_ppo.py \
    task=facet/Go2Force algo=ppo_facet_train task.num_envs=1024 wandb.mode=offline
```
**Measured on this machine** (RTX 3070 Ti Laptop, 8 GB), so no guesswork is needed:

| `task.num_envs` | peak VRAM | per iteration | frames/iteration | 150 M frames |
|---|---|---|---|---|
| 2048 | 4.7 GB | 2.4 s | 65 536 | ~1.5 h |
| 4096 | 6.7 GB | 3.1 s | 131 072 | ~1 h |

4096 (the paper's setting) fits with ~1.4 GB to spare and is also the fastest, so use it:

```bash
docker compose run --rm aa python scripts/train_ppo.py task=facet/Go2Force algo=ppo_facet_train \
    task.num_envs=4096 total_frames=150_000_000 wandb.mode=offline
```
If another GPU-using application is open, drop to 2048. On `CUDA out of memory`, add
`task.sim.gpu_max_rigid_patch_count=81920 task.sim.gpu_found_lost_aggregate_pairs_capacity=16777216`.
You can stop early: checkpoints are written every `checkpoint_interval=4` iterations.

Find results:
```bash
find ../outputs -name 'checkpoint_*.pt' | tail -3   # under <run>/wandb/offline-run-*/files/
find ../outputs -name "cfg.yaml" | tail -1
```
Offline wandb runs can be uploaded later with `wandb sync <run dir>` (needs an API key).

### Task 9.4 — Student fine-tune (what gets deployed)
```bash
docker compose run --rm aa python scripts/train_ppo.py \
    task=facet/Go2Force algo=ppo_facet_finetune task.num_envs=1024 wandb.mode=offline \
    checkpoint_path=<path from Task 9.3 to checkpoint_final.pt>
```
Expected: log prints `Successfully loaded [...]` for the checkpoint, then the same kind of
training output with `adapt/*` losses decreasing.

### Task 9.5 — Look at the behaviour
Headless video:
```bash
docker compose run --rm aa python scripts/eval.py \
    task=facet/Go2Force algo=ppo_facet_finetune task.num_envs=16 \
    checkpoint_path=<ckpt> eval_render=true wandb.mode=disabled
```
Interactive window (optional):
```bash
xhost +local:root
docker compose --profile gui run --rm aa-gui python scripts/play.py \
    task=facet/Go2Force algo=ppo_facet_finetune task.num_envs=16 checkpoint_path=<ckpt>
```
In the window you should see, per robot, a red arrow to the spring setpoint, a green
arrow for the reference velocity, and an orange arrow when an impulse is applied
(`ImpedanceCommandManager.debug_draw`). The robot should lean into and follow the
setpoint and give way when pushed.

### Task 9.6 — Export for deployment (optional)
```bash
docker compose run --rm aa python scripts/play.py task=facet/Go2Force algo=ppo_facet_finetune \
    checkpoint_path=<ckpt> export_policy=true headless=true
```

---

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Could not find 'task/facet/Go2Force'` (Hydra) | `.cache/projects.json` missing or `facet` disabled, or YAML not under `projects/cfg/task/facet/` | run any command through the entrypoint (it rewrites `projects.json`); check Topic 3 |
| `Class 'ImpedanceCommandManager' not found in Command.registry` | `projects.facet` was not imported → project disabled | same as above; `cat .cache/projects.json` |
| `Class 'xyz' not found in Observation.registry` | typo in YAML or class really missing | Task 4.2 loop |
| `Component 'command' must be a mapping` | `command:` with no value | write `command: {}` |
| `TypeError: __init__() got an unexpected keyword argument` | YAML passes an argument the class does not take | compare with the constructor (Topic 4 table) |
| `KeyError: 'priv'` in `ppo_facet` | `priv:` group missing or still commented | Task 4.1 |
| `AttributeError: 'PPOPolicy' object has no attribute 'obs_transform'` | Task 7.3 not applied | Topic 7 |
| `get_rollout_policy() got an unexpected keyword argument 'critic'` | Task 7.1 not applied | Topic 7 |
| Isaac Sim exits with EULA / privacy prompt | env vars missing | `ACCEPT_EULA=Y PRIVACY_CONSENT=Y` are in compose; check `docker compose config` |
| `Failed to create any GPU devices` / Vulkan errors | container cannot see the driver | `NVIDIA_DRIVER_CAPABILITIES=all`, `runtime: nvidia`; `docker run --rm --gpus all nvcr.io/nvidia/isaac-sim:6.0.0 nvidia-smi` |
| `CUDA out of memory` | too many robots for 8 GB | lower `task.num_envs`; Task 9.3 |
| Very slow first start | shader compilation | wait; caches are persisted in named volumes |
| `ModuleNotFoundError: active_adaptation` | entrypoint skipped (e.g. `docker exec` into a container) | `source /opt/isaac_env.sh && pip install --no-deps -e /workspace/active_adaptation` |
| Build fails: `apt-get ... Permission denied` | base image runs as uid 1234 | `USER root` in the Dockerfile (already applied) |
| Build fails: `Invalid requirement: '\x1b[3g'` | `isaaclab.sh` writes terminal escape codes into the captured constraints file | generate constraints by sourcing `isaac_env.sh` and filter with `grep -oE` (already applied) |
| `KeyError: 'action_log_prob'` in the PPO update | tensordict too old, so the log-prob key is `sample_log_prob` | tensordict/torchrl must be recent (Topic 6c) |
| `ValueError: Timer simulation already has a parent rollout` after training | `ScopedTimer` singletons refused re-use across scopes | Topic 6c |
| `AttributeError: 'NoneType' object has no attribute 'split'` at start-up | wandb leaves `run.name` unset offline | Topic 6c |
| `ConfigAttributeError: Missing key backend` from `eval.py` | `cfg/eval.yaml` lacked `backend`/`device` | Topic 6c |
| GUI: `cannot open display` | X11 not shared | `xhost +local:root`, `--profile gui`, `echo $DISPLAY` non-empty |
| `... takes 1 positional argument but 2 positional arguments ... were given` from a `write_*_to_sim_index` | Isaac Lab 3.0 writers are keyword-only | call them through `IsaacArticulation` (Topic 6d), never on the raw asset |
| `'WrenchComposer' object has no attribute 'composed_force_as_torch'` | 3.0 renamed the read-back to `out_force_b` / `out_torque_b` | use `utils/wrench.py` helpers (already applied) |
| `DeprecationWarning: root_physx_view` | 3.0 alias of `root_view` | use `asset.root_view` |
| Robot spawns / resets with a wrong orientation, or heading obs look rotated | a quaternion crossed the Isaac boundary un-converted | run `scripts/tools/check_isaac_quat.py` (Topic 6d); it names the stray read site |
| Video is black, or `Failed to upload DomeLight texture` | sky HDR not downloaded, or (before this fix) the Kit app loop was never pumped headless | download the assets (README); `env_base.render` pumps `omni.kit.app` when no GUI |
| `Reading .torch on a wp.quatf-typed ProxyArray` warnings | `WARN_ON_TORCH_QUATF_ACCESS=1` is set | expected only from `adapter.py`; anything else is a bug (see row above) |

---

## 6d. (added during implementation) Why the GUI crashed, and the move to Isaac Sim 6.0 / Isaac Lab 3.0

### What was wrong
Isaac Sim 5.1's RTX renderer crashes on start-up on this machine (driver 595.84), headless or
not, with the **stock** `nvcr.io/nvidia/isaac-sim:5.1.0` image too, so it was not our setup.
Isaac Sim 5.1 predates the 595 driver series. Physics-only training never touches the
renderer, which is why headless training worked on 5.1 while the GUI and video did not.
Vulkan presentation on this hybrid-graphics laptop (displays on the Radeon iGPU, rendering on
the RTX 3070 Ti) additionally needs Mesa's `radeon` ICD inside the container (in the Dockerfile).

### The fix: target Isaac Sim 6.0 + Isaac Lab 3.0 natively
Isaac Sim 6.0's renderer starts on this driver. It pairs with Isaac Lab 3.0
(`v3.0.0-beta2.patch1`), whose API differs from 2.3 in four ways that matter here. The repo
now targets 3.0 only (no 2.x fallbacks); the 5.1 pairing is gone.

| Isaac Lab 3.0 change | Where it is handled |
|---|---|
| PhysX is a pluggable backend: `SimulationCfg(physics=PhysxCfg(...))` from `isaaclab_physx.physics`; `SimulationContext` owns the stage (`sim.stage`), no `attach_stage_to_usd_context` | `envs/backends/isaac/env.py` |
| Every `asset.data.*` / `sensor.data.*` field is a warp-backed `ProxyArray` (`.torch` gives a zero-copy tensor) | `IsaacData` in `envs/backends/isaac/adapter.py` unwraps to tensors |
| **Quaternions are `(x, y, z, w)`**; this repo and its MuJoCo backends are `(w, x, y, z)` | `IsaacData` converts every quaternion-bearing read (`*_quat_w`, `*_pose_w`, `*_state_w`, `default_root_state`) to WXYZ; `IsaacArticulation` converts root pose/state writes back; `AssetCfg` emits XYZW into `InitialStateCfg`; marker orientations go through `to_isaac_quat` |
| Writers split into keyword-only `*_index` / `*_mask`; joint/body index tensors must be `int32` | `IsaacArticulation` implements the repo's backend-agnostic names (`write_root_state_to_sim`, `write_joint_state_to_sim`, `set_joint_position_target`, ...) on the `_index` API and casts indices |
| PhysX view getters return warp arrays; setters take CPU `float32` / `int32` warp arrays; `root_physx_view` is a deprecated alias of `root_view` | `envs/mdp/utils/physx_view.py`; mass / material / CoM randomizers and two rewards use `root_view` |
| Wrench composer: `add/set_forces_and_torques_index`, read-back `out_force_b` / `out_torque_b` | `envs/mdp/utils/wrench.py` |
| `isaacsim.core.utils` moved to `extsDeprecated`; `has_gui` is a property; no `clear_all_callbacks` | `env_base.py` seeds itself; `IsaacSimAdapter.has_gui()`; `close()` |

The adapter is the **only** place that knows about Isaac Lab's conventions. MDP code, the
math helpers and the MuJoCo backends are unchanged. The scene adapter wraps every articulation,
rigid object and sensor once and caches the wrappers, so `scene.articulations["robot"]` always
returns the same object.

### How to prove the quaternion boundary is right
`scripts/tools/check_isaac_quat.py` builds `Go2Force` with 4 robots and checks, against
quantities Isaac Lab computes internally (so independent of any convention):

1. `heading_w` equals the yaw of `euler_from_quat(root_quat_w)` (repo WXYZ math);
2. `projected_gravity_b` equals `quat_rotate_inverse(root_quat_w, (0, 0, -1))`;
3. a root state written through the adapter reads back with the same quaternion;
4. with `WARN_ON_TORCH_QUATF_ACCESS=1` no raw quaternion read happens outside `adapter.py`.

```bash
cd docker && docker compose run --rm aa python scripts/tools/check_isaac_quat.py \
    task=facet/Go2Force algo=ppo_facet_train task.num_envs=4 wandb.mode=disabled
```
If 1 or 2 fail with a 90/180 degree yaw error or gravity on the wrong axis, the roll direction
in `from_isaac_quat` / `to_isaac_quat` is inverted.

---

## 11. Known issues left out of scope (do not fix now, just know they exist)

- `ppo_dic.py` (`algo=ppo_dic_*`) still lacks the `env` constructor argument the factory passes; unused by this plan.
- The randomizations `drag`, `stumble`, `pull`, `spring_grf` define a `step()` hook the env never calls, so they apply no force. `Go2Force.yaml` does not use them.
- `B1Z1.yaml` (loco-manipulator) needs the `body_pos_b` / `body_vel_b` observations re-enabled
  (`observations/body.py`, commented out), an end-effector-aware `symmetry_transform` in
  `ImpedanceCommandManager.symmetry_transform` (currently 12-dim, ignores the 6 eef dims),
  and the same YAML fixes as Topic 4.
- The old `Impedance` command (`impedance.py`) is kept for reference only; its constructor
  defaults do not match the current `Command` base and it should not be selected from YAML.

---

## 11b. What was actually verified on this machine

All of the following were run end to end in the container on the RTX 3070 Ti Laptop GPU
(8 GB, driver 595.84), Isaac Sim 6.0.0 + Isaac Lab v3.0.0-beta2.patch1, laptop on AC power.

| Check | Result |
|---|---|
| `docker compose build` | succeeds; image `active-adaptation-isaac:6.0.0`, 33.5 GB |
| torch / CUDA in container | torch 2.10 (cu128), `cuda_available True`; warp 1.13.0 resolved to Isaac Sim's bundled `omni.warp.core` |
| `aa-list-tasks` | lists `facet/Go2Force` and `facet/B1Z1` |
| `scripts/tools/check_isaac_quat.py` (4 envs, 20 steps) | PASSED: heading, projected gravity and root-state round trip agree with the WXYZ math; 484 raw quaternion reads, all from `adapter.py` |
| Training, 16 envs, 20k frames | 39/39 iterations, exit 0, `checkpoint_final.pt` written, no deprecation warnings |
| Same checkpoint (`checkpoint_100.pt`, trained on 5.1) evaluated with 256 envs, seed 0 | 5.1 image + old code: survival 71.1, crash 51%, fall 49%. 6.0 image + this code: survival 72.2, crash 58%, fall 43%. Agrees within noise |
| GUI: `play.py ... headless=false` via the `gui` profile | Isaac Sim window (`isaaclab.python.kit` experience) opens on driver 595.84, environment steps with the viewport camera controller and debug draw; first start needs ~3.5 min of shader compilation |
| Training, 4096 envs | 4.1 s/iteration over 11 iterations (warm-up included), 6.7 GB VRAM |
| Video: `eval.py ... eval_render=true` (4 envs) | writes `active_adaptation/recording-<time>.mp4`, 1000 frames at 50 fps with real content (mean intensity 36; the first attempt was all black, see below) |

Bugs found by these gates and fixed on the way: the Isaac Lab 3.0 writers are keyword-only
(`root_pose=`, `position=`, ...); the wrench composer read-back is `out_force_b`; warp 1.13's
Python-scope `rand_init` rejects `uint32` (FACET seed kept as an int); Isaac Lab sensors refresh
lazily on `sensor.data` access, so the sensor wrapper must never cache that container (the first
6.0 evaluation had zero contacts and looked far too good); headless RGB capture needs the Kit app
loop pumped explicitly (`omni.kit.app.get_app().update()`, as Isaac Lab 3.0's own recorder does)
because `sim.render()` no longer does it without a Kit visualizer, otherwise the annotator
returns an empty array forever and the video is black; the perspective camera is positioned via
`isaacsim.core.rendering_manager.ViewportManager`, since `sim.set_camera_view` only reaches
visualizers; `headless=false` must request the `kit` visualizer (`active_adaptation/__init__.py`).

If Kit logs `Failed to upload DomeLight texture ... .hdr`, the sky HDR from the asset dataset is
missing (README, "Asset download and placement"); the scene then falls back to an untextured dome
light so renders are still lit.

The 20k-frame smoke run is far too short to produce a competent gait; it only proves the
pipeline. Real training is Task 9.3.

---

## 12. Definition of done (checklist)

- [ ] `PLAN.md` committed on branch `fix/facet-go2-docker`
- [ ] `docker/` contains Dockerfile, compose.yaml, entrypoint.sh, isaac_env.sh, make_constraints.py, .env.example; `docker/.env` gitignored
- [ ] `docker compose build` succeeds; torch reports CUDA available inside the container
- [ ] `aa-list-tasks` prints `facet/Go2Force`
- [ ] Smoke test (Task 9.2) completes with finite losses and increasing episode length
- [ ] Teacher run (Task 9.3) writes checkpoints; student run (Task 9.4) loads them
- [x] A rendered video or GUI session shows the robot tracking the setpoint and yielding to pushes
- [x] `scripts/tools/check_isaac_quat.py` passes on the Isaac Sim 6.0 image
- [x] `play.py ... headless=false` through the `gui` profile opens an Isaac Sim window
- [ ] README and `projects/facet/readme.md` updated
