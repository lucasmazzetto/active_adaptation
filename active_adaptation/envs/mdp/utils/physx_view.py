"""Version-tolerant access to PhysX tensor views (``asset.root_physx_view``).

Isaac Lab <= 2.3: view getters return CPU ``torch.Tensor`` and setters accept CPU torch
tensors with ``int64`` index tensors.

Isaac Lab >= 3.0: getters return ``warp`` arrays, setters expect CPU ``float32`` warp arrays
and CPU ``int32`` warp index arrays (see Isaac Lab's own ``events.py`` and
``rigid_object.py``: "when updating 'model' properties with PhysX we need to do it on CPU").
``root_physx_view`` still exists in 3.0 as a deprecated alias of ``root_view``.

These helpers let callers work in torch on the CPU and convert at the boundary.
"""

from __future__ import annotations

import torch


def _is_warp(x) -> bool:
    return not isinstance(x, torch.Tensor) and type(x).__module__.startswith("warp")


def view_to_torch(x) -> torch.Tensor:
    """Return a torch tensor for whatever a PhysX view getter returned."""
    if isinstance(x, torch.Tensor):
        return x
    if _is_warp(x):
        import warp as wp
        return wp.to_torch(x)
    return torch.as_tensor(x)


def torch_to_view(t: torch.Tensor, like) -> object:
    """Convert ``t`` into what the view setter expects, judged by a getter result ``like``."""
    t = t.detach().contiguous().cpu()
    if _is_warp(like):
        import warp as wp
        return wp.from_torch(t.to(torch.float32), dtype=wp.float32)
    return t


def view_indices(n: int, like) -> object:
    """All-instance index array in the form the view setter expects."""
    if _is_warp(like):
        import warp as wp
        return wp.from_torch(torch.arange(n, dtype=torch.int32), dtype=wp.int32)
    return torch.arange(n)


__all__ = ["view_to_torch", "torch_to_view", "view_indices"]
