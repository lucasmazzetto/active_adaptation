"""Verify the Isaac Lab 3.0 quaternion boundary of the Isaac backend adapter.

Isaac Lab 3.0 stores quaternions as (x, y, z, w); this repository uses (w, x, y, z) and converts
in ``active_adaptation/envs/backends/isaac/adapter.py``. A wrong conversion does not crash, it
silently rotates everything wrongly, so this script checks the converted data against two
quantities Isaac Lab computes internally and therefore independently of any convention:

1. ``heading_w`` must equal the yaw of ``euler_from_quat(root_quat_w)`` (repo WXYZ math).
2. ``projected_gravity_b`` must equal ``quat_rotate_inverse(root_quat_w, (0, 0, -1))``.
3. A root state written through the adapter (WXYZ) must read back with the same quaternion.
4. With ``WARN_ON_TORCH_QUATF_ACCESS=1`` (set below), Isaac Lab warns on every raw read of a
   quaternion field; none may originate outside the adapter.

Run inside the Isaac container, e.g.

    docker compose run --rm aa python scripts/tools/check_isaac_quat.py \\
        task=facet/Go2Force algo=ppo_facet_train task.num_envs=4 wandb.mode=disabled

Exit code 0 means every check passed.
"""

import os
import sys
import warnings

os.environ.setdefault("WARN_ON_TORCH_QUATF_ACCESS", "1")

import hydra
import torch
from omegaconf import OmegaConf

import active_adaptation as aa

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(FILE_DIR, "..", "..", "cfg")

ATOL = 2e-3
ROLLOUT_STEPS = 20


@hydra.main(config_path=CONFIG_PATH, config_name="train", version_base=None)
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    cfg.headless = True
    aa.init(cfg, auto_rank=False)

    from active_adaptation.helpers import make_env_policy
    from active_adaptation.utils.math import euler_from_quat, quat_rotate_inverse, wrap_to_pi
    from active_adaptation.envs.backends.isaac import adapter as isaac_adapter

    env, _ = make_env_policy(cfg)
    base = env.base_env
    robot = base.scene.articulations["robot"]
    data = robot.data
    device = base.device
    failures: list[str] = []
    checks = {"unit_norm": 0, "heading": 0, "gravity": 0}

    def record(name: str, err: torch.Tensor, step):
        checks[name] += 1
        worst = float(err.abs().max())
        if worst > ATOL:
            failures.append(f"step {step}: {name} mismatch, max abs error {worst:.4g}")

    gravity_dir = torch.tensor([0.0, 0.0, -1.0], device=device).expand(base.num_envs, 3)

    def check_invariants(step):
        q = data.root_quat_w  # WXYZ after the adapter
        record("unit_norm", q.norm(dim=-1) - 1.0, step)
        yaw = euler_from_quat(q)[..., 2]
        record("heading", wrap_to_pi(yaw - data.heading_w), step)
        record("gravity", quat_rotate_inverse(q, gravity_dir) - data.projected_gravity_b, step)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        td = env.reset()
        check_invariants("reset")
        env.rollout(
            ROLLOUT_STEPS,
            tensordict=td,
            auto_reset=False,
            break_when_any_done=False,
            callback=lambda _env, _td: check_invariants(_env.base_env.episode_length_buf[0].item()),
        )

        # 3. Round trip through the adapter's root-state writer (WXYZ in, WXYZ out).
        yaw_in = torch.linspace(-2.5, 2.5, base.num_envs, device=device)
        q_in = torch.stack(
            [torch.cos(yaw_in / 2), torch.zeros_like(yaw_in), torch.zeros_like(yaw_in), torch.sin(yaw_in / 2)], -1
        )
        root_state = data.default_root_state.clone()
        root_state[:, :3] += base.scene.env_origins
        root_state[:, 2] += 0.5
        root_state[:, 3:7] = q_in
        root_state[:, 7:] = 0.0
        robot.write_root_state_to_sim(root_state)
        base.scene.write_data_to_sim()
        base.sim.step(render=False)          # one physics step (5 ms) barely moves the base
        base.scene.update(base.physics_dt)
        q_out = data.root_quat_w
        # q and -q are the same rotation; compare via the yaw angle and the dot product.
        dot = (q_in * q_out).sum(-1).abs()
        yaw_out = euler_from_quat(q_out)[..., 2]
        round_trip_err = torch.maximum((1.0 - dot), wrap_to_pi(yaw_out - yaw_in).abs())
        worst = float(round_trip_err.max())
        if worst > 2e-2:
            failures.append(f"round trip: written WXYZ quaternion read back differently, max error {worst:.4g}")

    # 4. Raw quaternion reads outside the adapter.
    adapter_file = os.path.abspath(isaac_adapter.__file__)
    stray = sorted({
        f"{w.filename}:{w.lineno}"
        for w in caught
        if "quatf" in str(w.message) and os.path.abspath(w.filename) != adapter_file
    })
    for site in stray:
        failures.append(f"raw quaternion read bypassing the adapter at {site}")

    print("\n=== Isaac Lab 3.0 quaternion boundary check ===")
    print(f"num_envs={base.num_envs}  steps={ROLLOUT_STEPS}  atol={ATOL}")
    print(f"checks run: {checks}")
    print(f"quatf .torch reads seen: {sum('quatf' in str(w.message) for w in caught)} (adapter only: {not stray})")
    if failures:
        print("FAILED:")
        for f in failures:
            print("  -", f)
        env.close()
        sys.exit(1)
    print("PASSED: heading, projected gravity and root-state round trip agree with the WXYZ math.")
    env.close()


if __name__ == "__main__":
    main()
