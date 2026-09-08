"""Torch <-> warp boundary for PhysX tensor views (``asset.root_view``) on Isaac Lab 3.0.

View getters (``get_masses``, ``get_inertias``, ``get_coms``, ``get_material_properties``,
``get_dof_projected_joint_forces``, ...) return warp arrays; setters expect CPU ``float32`` warp
arrays plus CPU ``int32`` warp index arrays, and PhysX "model" properties (mass, inertia, CoM,
materials) must be written on the CPU (see Isaac Lab's own ``events.py``). These helpers let the
randomizers work in torch on the CPU and convert at the view boundary.
"""

from __future__ import annotations

import torch
import warp as wp


def view_to_torch(x) -> torch.Tensor:
    """Torch view of a PhysX view getter result (zero-copy for warp arrays)."""
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, wp.array):
        return wp.to_torch(x)
    return torch.as_tensor(x)


def torch_to_view(t: torch.Tensor) -> wp.array:
    """CPU float32 warp array for a PhysX view setter."""
    return wp.from_torch(t.detach().contiguous().to("cpu", torch.float32), dtype=wp.float32)


def view_indices(n: int) -> wp.array:
    """All-instance CPU int32 index array for a PhysX view setter."""
    return wp.from_torch(torch.arange(n, dtype=torch.int32), dtype=wp.int32)


__all__ = ["view_to_torch", "torch_to_view", "view_indices"]
