"""Backend-compatible access to external wrenches.

Isaac Lab (>= 2.3, including 3.0) applies external wrenches through
:class:`~isaaclab.utils.wrench_composer.WrenchComposer`, reachable as
``asset.permanent_wrench_composer``. The MuJoCo backend in this repository
(``envs/backends/mujoco/mujoco.py``) keeps the older ``_external_force_b`` /
``_external_torque_b`` buffers with a ``has_external_wrench`` flag.

These helpers hide that difference so MDP components do not have to branch on the
simulator version. Wrenches are always expressed in each body's link frame unless
``is_global`` is set.
"""

from __future__ import annotations

import torch


def _composer(asset):
    """Return the Isaac Lab permanent wrench composer, or None on the MuJoCo backend."""
    return getattr(asset, "permanent_wrench_composer", None)


def add_external_wrench(
    asset,
    forces: torch.Tensor | None = None,
    torques: torch.Tensor | None = None,
    body_ids=None,
    positions: torch.Tensor | None = None,
    is_global: bool = False,
) -> None:
    """Accumulate an external wrench on the given bodies.

    Args:
        asset: the articulation to push on.
        forces: ``(num_envs, num_selected_bodies, 3)`` or None.
        torques: ``(num_envs, num_selected_bodies, 3)`` or None.
        body_ids: list/tensor of body indices, or None for all bodies.
        positions: optional application points, same shape as ``forces``.
        is_global: whether ``forces``/``torques`` are given in the world frame.

    The wrench persists until the environment clears it, which happens once per
    physics substep via ``scene.zero_external_wrenches()``. Call this from a
    ``pre_step`` hook so the force is re-applied on every substep.
    """
    composer = _composer(asset)
    if composer is not None:
        composer.add_forces_and_torques_index(
            forces=forces,
            torques=torques,
            positions=positions,
            body_ids=body_ids,
            is_global=is_global,
        )
        return

    # The in-repo MuJoCo backend.
    if is_global:
        raise NotImplementedError("is_global=True requires the Isaac Lab wrench composer.")
    if positions is not None:
        raise NotImplementedError(
            "Application points require the Isaac Lab wrench composer; "
            "convert them to an equivalent torque instead."
        )
    index = slice(None) if body_ids is None else body_ids
    if forces is not None:
        asset._external_force_b[:, index] += forces
    if torques is not None:
        asset._external_torque_b[:, index] += torques
    if hasattr(asset, "has_external_wrench"):
        asset.has_external_wrench = True


def set_external_wrench(
    asset,
    forces: torch.Tensor | None = None,
    torques: torch.Tensor | None = None,
    body_ids=None,
    positions: torch.Tensor | None = None,
    is_global: bool = False,
) -> None:
    """Replace (rather than accumulate) the external wrench on the given bodies.

    Same arguments and version handling as :func:`add_external_wrench`.
    """
    composer = _composer(asset)
    if composer is not None:
        composer.set_forces_and_torques_index(
            forces=forces,
            torques=torques,
            positions=positions,
            body_ids=body_ids,
            is_global=is_global,
        )
        return

    if is_global or positions is not None:
        raise NotImplementedError("is_global/positions require the Isaac Lab wrench composer.")
    index = slice(None) if body_ids is None else body_ids
    if forces is not None:
        asset._external_force_b[:, index] = forces
    if torques is not None:
        asset._external_torque_b[:, index] = torques
    if hasattr(asset, "has_external_wrench"):
        asset.has_external_wrench = True


def get_external_force_b(asset) -> torch.Tensor:
    """Currently applied external force per body, in the link frame. ``(num_envs, num_bodies, 3)``."""
    composer = _composer(asset)
    if composer is not None:
        # Composes the input buffers into the body frame if anything changed since the last read.
        return composer.out_force_b.torch
    return asset._external_force_b


def get_external_torque_b(asset) -> torch.Tensor:
    """Currently applied external torque per body, in the link frame. ``(num_envs, num_bodies, 3)``."""
    composer = _composer(asset)
    if composer is not None:
        return composer.out_torque_b.torch
    return asset._external_torque_b


__all__ = ["add_external_wrench", "set_external_wrench", "get_external_force_b", "get_external_torque_b"]
