"""Isaac Lab 3.0 backend adapters.

This module is the single place where this repository talks to Isaac Lab's asset, sensor,
scene and simulation objects. Three things differ between Isaac Lab 3.0 and the conventions
used everywhere else in this codebase (and by the MuJoCo backends), and all three are handled
here so that MDP components never see them:

* **Array type.** Isaac Lab 3.0 exposes every ``asset.data.*`` / ``sensor.data.*`` field as a
  warp-backed :class:`~isaaclab.utils.warp.proxy_array.ProxyArray`. The wrappers below return the
  zero-copy ``.torch`` view instead, so callers always get plain tensors.
* **Quaternion order.** Isaac Lab 3.0 stores quaternions as ``(x, y, z, w)``; this repository's
  math (:mod:`active_adaptation.utils.math`) and the MuJoCo backends use ``(w, x, y, z)``. Reads of
  quaternion-bearing fields are converted to WXYZ, and writes of root poses/states are converted
  back to XYZW before reaching Isaac Lab.
* **Write API.** Isaac Lab 3.0 split the writers into ``*_index`` / ``*_mask`` variants and no
  longer casts joint/body index tensors to ``int32``. :class:`IsaacArticulation` implements the
  backend-agnostic method names used by the MDP layer (the same names the MuJoCo backend
  articulation provides) on top of the ``*_index`` API, normalising indices on the way.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from typing_extensions import override

from active_adaptation.envs.adapters import SimAdapter, SceneAdapter

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.scene import InteractiveScene
    from isaaclab.sensors import SensorBase
    from isaaclab.sim import SimulationContext


# ----------------------------------------------------------------------------------------------
# Quaternion convention helpers (repo: WXYZ, Isaac Lab 3.0: XYZW)
# ----------------------------------------------------------------------------------------------


def from_isaac_quat(q: torch.Tensor) -> torch.Tensor:
    """XYZW (Isaac Lab 3.0) -> WXYZ (this repository). Returns a new tensor."""
    return torch.roll(q, 1, dims=-1)


def to_isaac_quat(q: torch.Tensor) -> torch.Tensor:
    """WXYZ (this repository) -> XYZW (Isaac Lab 3.0). Returns a new tensor."""
    return torch.roll(q, -1, dims=-1)


def _convert_pose_quat(x: torch.Tensor, fn) -> torch.Tensor:
    """Apply ``fn`` to the quaternion slice ``[..., 3:7]`` of a pose (7) or state (13) tensor."""
    out = x.clone()
    out[..., 3:7] = fn(x[..., 3:7])
    return out


def _as_tensor(value: Any) -> Any:
    """Return the torch view of a ProxyArray, or ``value`` unchanged for anything else."""
    # ProxyArray exposes both `.torch` and `.warp`; nothing else we hand out does.
    if isinstance(value, torch.Tensor):
        return value
    if hasattr(value, "torch") and hasattr(value, "warp"):
        return value.torch
    return value


def _as_int32_ids(ids):
    """Isaac Lab 3.0 leaves torch index tensors untouched, and warp needs int32."""
    if isinstance(ids, torch.Tensor) and ids.dtype != torch.int32:
        return ids.to(torch.int32)
    return ids


# ----------------------------------------------------------------------------------------------
# Data wrappers
# ----------------------------------------------------------------------------------------------

# Fields that are a bare quaternion (last dim 4).
_QUAT_FIELDS = frozenset({
    "root_quat_w", "root_link_quat_w", "root_com_quat_w",
    "body_quat_w", "body_link_quat_w", "body_com_quat_w", "body_com_quat_b",
    "quat_w",  # sensors
})
# Fields that are a pose (pos 3 + quat 4) or a state (pose 7 + vel 6): quaternion at [..., 3:7].
_POSE_FIELDS = frozenset({
    "root_pose_w", "root_link_pose_w", "root_com_pose_w",
    "body_pose_w", "body_link_pose_w", "body_com_pose_w", "body_com_pose_b",
    "root_state_w", "root_link_state_w", "root_com_state_w",
    "body_state_w", "body_link_state_w", "body_com_state_w",
    "default_root_pose", "default_root_state",
    "pose_w",  # sensors
})


class IsaacData:
    """Wraps an Isaac Lab data container (``asset.data`` / ``sensor.data``).

    Every attribute read returns a plain tensor (ProxyArrays are unwrapped) and quaternions are
    returned in WXYZ. Attribute writes (this repo attaches a few custom fields to ``data``, e.g.
    ``body_materials``) are forwarded to the wrapped object.
    """

    __slots__ = ("_inner",)

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)

    def __getattr__(self, name: str):
        value = _as_tensor(getattr(self._inner, name))
        if isinstance(value, torch.Tensor):
            if name in _QUAT_FIELDS:
                return from_isaac_quat(value)
            if name in _POSE_FIELDS:
                return _convert_pose_quat(value, from_isaac_quat)
        return value

    def __setattr__(self, name: str, value) -> None:
        setattr(self._inner, name, value)

    def __repr__(self) -> str:
        return f"IsaacData({self._inner!r})"


class IsaacSensor:
    """Wraps an Isaac Lab sensor: ``.data`` is an :class:`IsaacData`, methods return tensors."""

    __slots__ = ("_inner",)

    def __init__(self, inner: "SensorBase"):
        object.__setattr__(self, "_inner", inner)

    @property
    def data(self) -> IsaacData:
        # Isaac Lab sensors are lazy: `scene.update()` only marks them outdated, and the buffers are
        # refreshed when the sensor's `data` property is read. Read it on every access (never cache
        # the container) or the sensor silently keeps returning its initial zeros.
        return IsaacData(self._inner.data)

    @property
    def inner(self):
        return self._inner

    def compute_first_contact(self, dt: float, **kwargs) -> torch.Tensor:
        return _as_tensor(self._inner.compute_first_contact(dt, **kwargs))

    def compute_first_air(self, dt: float, **kwargs) -> torch.Tensor:
        return _as_tensor(self._inner.compute_first_air(dt, **kwargs))

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    def __setattr__(self, name: str, value) -> None:
        setattr(self._inner, name, value)


# Per-joint writers/targets: repo name -> keyword the Isaac Lab 3.0 ``*_index`` method takes for the
# value (those methods are keyword-only). All take ``(value, joint_ids=None, env_ids=None)`` here.
_JOINT_WRITERS = {
    "write_joint_position_to_sim": "position",
    "write_joint_velocity_to_sim": "velocity",
    "write_joint_stiffness_to_sim": "stiffness",
    "write_joint_damping_to_sim": "damping",
    "write_joint_armature_to_sim": "armature",
    "write_joint_friction_coefficient_to_sim": "joint_friction_coeff",
    "write_joint_position_limit_to_sim": "limits",
    "write_joint_velocity_limit_to_sim": "limits",
    "write_joint_effort_limit_to_sim": "limits",
    "set_joint_position_target": "target",
    "set_joint_velocity_target": "target",
    "set_joint_effort_target": "target",
}


def _make_joint_writer(name: str, value_kw: str):
    target = f"{name}_index"

    def writer(self, value, joint_ids=None, env_ids=None, **kwargs):
        getattr(self._inner, target)(
            **{value_kw: value},
            joint_ids=_as_int32_ids(joint_ids),
            env_ids=_as_int32_ids(env_ids),
            **kwargs,
        )

    writer.__name__ = name
    writer.__doc__ = f"Isaac Lab 3.0 ``{target}`` with index normalisation."
    return writer


class IsaacArticulation:
    """Wraps an Isaac Lab ``Articulation`` (or ``RigidObject``) behind the repo's asset API.

    * ``.data`` is an :class:`IsaacData` (tensors, WXYZ quaternions).
    * Root pose/state writers take WXYZ quaternions and forward XYZW to the ``*_index`` API.
    * Joint writers/targets accept int64 index tensors and forward int32.
    * Everything else (``find_bodies``, ``find_joints``, ``num_instances``, ``root_view``,
      ``permanent_wrench_composer``, ``actuators``, ``write_data_to_sim``, ...) passes through.
    """

    __slots__ = ("_inner", "_data")

    def __init__(self, inner: "Articulation | RigidObject"):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_data", IsaacData(inner.data))

    @property
    def data(self) -> IsaacData:
        return self._data

    @property
    def inner(self):
        """The wrapped Isaac Lab asset."""
        return self._inner

    # -- root writers (WXYZ in) ---------------------------------------------------------------

    def write_root_state_to_sim(self, root_state: torch.Tensor, env_ids=None) -> None:
        """Root link pose (WXYZ) + root CoM velocity, as Isaac Lab <= 2.3 defined it."""
        env_ids = _as_int32_ids(env_ids)
        self._inner.write_root_link_pose_to_sim_index(
            root_pose=_convert_pose_quat(root_state[:, :7], to_isaac_quat), env_ids=env_ids
        )
        self._inner.write_root_com_velocity_to_sim_index(root_velocity=root_state[:, 7:], env_ids=env_ids)

    def write_root_link_state_to_sim(self, root_state: torch.Tensor, env_ids=None) -> None:
        env_ids = _as_int32_ids(env_ids)
        self._inner.write_root_link_pose_to_sim_index(
            root_pose=_convert_pose_quat(root_state[:, :7], to_isaac_quat), env_ids=env_ids
        )
        self._inner.write_root_link_velocity_to_sim_index(root_velocity=root_state[:, 7:], env_ids=env_ids)

    def write_root_com_state_to_sim(self, root_state: torch.Tensor, env_ids=None) -> None:
        env_ids = _as_int32_ids(env_ids)
        self._inner.write_root_com_pose_to_sim_index(
            root_pose=_convert_pose_quat(root_state[:, :7], to_isaac_quat), env_ids=env_ids
        )
        self._inner.write_root_com_velocity_to_sim_index(root_velocity=root_state[:, 7:], env_ids=env_ids)

    def write_root_pose_to_sim(self, root_pose: torch.Tensor, env_ids=None) -> None:
        self._inner.write_root_pose_to_sim_index(
            root_pose=_convert_pose_quat(root_pose, to_isaac_quat), env_ids=_as_int32_ids(env_ids)
        )

    def write_root_link_pose_to_sim(self, root_pose: torch.Tensor, env_ids=None) -> None:
        self._inner.write_root_link_pose_to_sim_index(
            root_pose=_convert_pose_quat(root_pose, to_isaac_quat), env_ids=_as_int32_ids(env_ids)
        )

    def write_root_com_pose_to_sim(self, root_pose: torch.Tensor, env_ids=None) -> None:
        self._inner.write_root_com_pose_to_sim_index(
            root_pose=_convert_pose_quat(root_pose, to_isaac_quat), env_ids=_as_int32_ids(env_ids)
        )

    def write_root_velocity_to_sim(self, root_velocity: torch.Tensor, env_ids=None) -> None:
        self._inner.write_root_velocity_to_sim_index(root_velocity=root_velocity, env_ids=_as_int32_ids(env_ids))

    def write_root_link_velocity_to_sim(self, root_velocity: torch.Tensor, env_ids=None) -> None:
        self._inner.write_root_link_velocity_to_sim_index(
            root_velocity=root_velocity, env_ids=_as_int32_ids(env_ids)
        )

    def write_root_com_velocity_to_sim(self, root_velocity: torch.Tensor, env_ids=None) -> None:
        self._inner.write_root_com_velocity_to_sim_index(
            root_velocity=root_velocity, env_ids=_as_int32_ids(env_ids)
        )

    # -- joint writers ----------------------------------------------------------------------

    def write_joint_state_to_sim(self, position, velocity, joint_ids=None, env_ids=None) -> None:
        joint_ids = _as_int32_ids(joint_ids)
        env_ids = _as_int32_ids(env_ids)
        self._inner.write_joint_position_to_sim_index(position=position, joint_ids=joint_ids, env_ids=env_ids)
        self._inner.write_joint_velocity_to_sim_index(velocity=velocity, joint_ids=joint_ids, env_ids=env_ids)

    for _name, _kw in _JOINT_WRITERS.items():
        locals()[_name] = _make_joint_writer(_name, _kw)
    del _name, _kw

    # -- passthrough ------------------------------------------------------------------------

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    def __setattr__(self, name: str, value) -> None:
        setattr(self._inner, name, value)

    def __repr__(self) -> str:
        return f"IsaacArticulation({self._inner!r})"


# ----------------------------------------------------------------------------------------------
# Sim / scene adapters
# ----------------------------------------------------------------------------------------------


class IsaacSimAdapter(SimAdapter):
    def __init__(self, sim: "SimulationContext"):
        self._sim = sim

    def get_physics_dt(self) -> float:
        return self._sim.get_physics_dt()

    def has_gui(self) -> bool:
        return bool(self._sim.has_gui)  # a property on Isaac Lab 3.0

    def step(self, render: bool = False) -> None:
        self._sim.step(render=render)

    def render(self) -> None:
        self._sim.render()

    def set_camera_view(self, eye=None, target=None, **kwargs) -> None:
        if eye is not None and target is not None:
            self._sim.set_camera_view(eye=eye, target=target)

    @property
    def stage(self):
        """The USD stage the simulation was built on."""
        return self._sim.stage

    def __getattr__(self, name):
        return getattr(self._sim, name)


class IsaacSceneAdapter(SceneAdapter):
    def __init__(self, scene: "InteractiveScene"):
        self._scene: "InteractiveScene" = scene
        # Wrap once and cache: the MDP layer looks entities up on every step and relies on
        # getting the same objects back (e.g. attributes attached to `asset.data`).
        self._articulations = {k: IsaacArticulation(v) for k, v in scene.articulations.items()}
        self._rigid_objects = {k: IsaacArticulation(v) for k, v in scene.rigid_objects.items()}
        self._sensors = {k: IsaacSensor(v) for k, v in scene.sensors.items()}

    @override
    def zero_external_wrenches(self) -> None:
        for asset in self._scene.articulations.values():
            asset.instantaneous_wrench_composer.reset()
            asset.permanent_wrench_composer.reset()

    @property
    def ground_mesh(self):
        """Warp ground mesh for the Isaac ground plane or mesh.

        This mirrors the logic previously implemented at the environment
        level, but keeps the backend-specific USD and warp handling inside
        the Isaac scene adapter.
        """
        if hasattr(self, "_ground_mesh"):
            return self._ground_mesh

        # Local imports to avoid making IsaacLab a hard dependency when other
        # backends are used.
        import numpy as np
        import warp as wp
        from isaaclab.utils.warp import convert_to_warp_mesh
        from isaaclab.terrains.trimesh.utils import make_plane
        from pxr import UsdGeom
        import isaaclab.sim as sim_utils

        mesh_prim_path = "/World/ground"
        device = wp.get_device(str(self._scene.device))

        # Check if there is a PhysX plane; otherwise fall back to a mesh prim.
        mesh_prim = sim_utils.get_first_matching_child_prim(
            mesh_prim_path, lambda prim: prim.GetTypeName() == "Plane"
        )
        if mesh_prim is None:
            mesh_prim = sim_utils.get_first_matching_child_prim(
                mesh_prim_path, lambda prim: prim.GetTypeName() == "Mesh"
            )
            if mesh_prim is None or not mesh_prim.IsValid():
                raise RuntimeError(f"Invalid mesh prim path: {mesh_prim_path}")
            mesh_prim = UsdGeom.Mesh(mesh_prim)
            points = np.asarray(mesh_prim.GetPointsAttr().Get())
            indices = np.asarray(mesh_prim.GetFaceVertexIndicesAttr().Get())
            wp_mesh = convert_to_warp_mesh(points, indices, device=device)
        else:
            mesh = make_plane(size=(2e6, 2e6), height=0.0, center_zero=True)
            wp_mesh = convert_to_warp_mesh(mesh.vertices, mesh.faces, device=device)

        self._ground_mesh = wp_mesh
        return self._ground_mesh

    @property
    def articulations(self) -> dict[str, IsaacArticulation]:
        return self._articulations

    @property
    def rigid_objects(self) -> dict[str, IsaacArticulation]:
        return self._rigid_objects

    @property
    def sensors(self) -> dict[str, IsaacSensor]:
        return self._sensors

    def __getattr__(self, name):
        return getattr(self._scene, name)

    def __getitem__(self, key):
        if key in self._articulations:
            return self._articulations[key]
        if key in self._rigid_objects:
            return self._rigid_objects[key]
        if key in self._sensors:
            return self._sensors[key]
        return self._scene[key]

    def create_sphere_marker(
        self,
        prim_path: str,
        color: tuple[float, float, float],
        radius: float = 0.05,
    ):
        """Create an Isaac Lab VisualizationMarkers with a single sphere (for GUI debug).

        Returns a VisualizationMarkers instance. Call .set_visibility(True) and
        .visualize(positions_tensor) to use it.
        """
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        import isaaclab.sim as sim_utils

        marker = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path=prim_path,
                markers={
                    "sphere": sim_utils.SphereCfg(
                        radius=radius,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                    ),
                },
            )
        )
        marker.set_visibility(True)
        return marker

    def create_arrow_marker(
        self,
        prim_path: str,
        color: tuple[float, float, float] = (1.0, 0.0, 0.0),
        scale: tuple[float, float, float] = (1.0, 0.1, 0.1),
    ):
        """Create an Isaac Lab VisualizationMarkers with a single arrow (for GUI debug).

        Returns a VisualizationMarkers instance. Call .set_visibility(True) and
        .visualize(positions_tensor) to use it.
        """
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
        import isaaclab.sim as sim_utils
        marker = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path=prim_path,
                markers={
                    "arrow": sim_utils.UsdFileCfg(
                        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                        scale=scale,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                    )
                },
            )
        )
        marker.set_visibility(True)
        return marker

    @override
    def get_spawn_origins(self, env_ids: torch.Tensor) -> torch.Tensor:
        if self._scene.terrain.terrain_origins is None:
            return self.env_origins[env_ids]

        terrain_origins = self._scene.terrain.terrain_origins.reshape(-1, 3)
        idx = torch.randint(
            0,
            terrain_origins.shape[0],
            (len(env_ids),),
            device=env_ids.device,
        )
        return terrain_origins[idx]


__all__ = [
    "IsaacSimAdapter",
    "IsaacSceneAdapter",
    "IsaacArticulation",
    "IsaacSensor",
    "IsaacData",
    "from_isaac_quat",
    "to_isaac_quat",
]
