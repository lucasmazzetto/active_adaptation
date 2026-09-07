# FACET

Force-Adaptive Control via Impedance Reference Tracking for Legged Robots (arXiv 2505.06883).

* Command manager: `facet_commands/impedance_manip.py` (`ImpedanceCommandManager`)
* Rewards: `facet_commands/rewards.py`
* Task configs: `projects/cfg/task/facet/` (`facet/Go2Force` is maintained; `facet/B1Z1` is a follow-up)
* Algorithm: `active_adaptation/learning/ppo/ppo_facet.py` (`ppo_facet_train` -> `ppo_facet_finetune`)

## Usage

Inside the Docker image described in the top-level README (`docker/`):

```bash
python scripts/train_ppo.py task=facet/Go2Force algo=ppo_facet_train task.num_envs=1024 wandb.mode=offline
python scripts/train_ppo.py task=facet/Go2Force algo=ppo_facet_finetune task.num_envs=1024 wandb.mode=offline \
    checkpoint_path=<teacher checkpoint>
```

See `PLAN.md` at the repository root for the full walkthrough.
