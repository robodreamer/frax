"""Distances and velocity Jacobians for selected URDF primitive pairs.

MuJoCo computes signed distances between the original primitives. FRAX supplies
their attachment poses and point Jacobians. This module is for fixed-base local
experiments; it does not model mesh geometry or continuous swept collisions.
"""

import jax
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from frax.ik.constraints import InequalityConstraint
from frax.utils import urdfpy


def attachment(model, link_name):
    """Return the nearest moving parent joint and its transform to the link."""
    by_child = {joint.child: joint for joint in model.joints}
    offset = np.eye(4)
    link = link_name
    while link in by_child:
        joint = by_child[link]
        if joint.joint_type != "fixed":
            return joint.name, offset
        offset = joint.origin @ offset
        link = joint.parent
    if link != model.base_link.name:
        raise ValueError(f"Unknown link: {link_name}")
    return None, offset


class PrimitivePairChecker:
    def __init__(self, model, robot, link_pairs):
        if robot.floating_base is not None:
            raise ValueError("Primitive pair checks currently require a fixed base")
        self.robot = robot
        links = {link.name: link for link in model.links}
        selected = list(dict.fromkeys(name for pair in link_pairs for name in pair))
        spec = mujoco.MjSpec()
        self.geometries = []
        link_geoms = {}
        for link_name in selected:
            if link_name not in links:
                raise ValueError(f"Unknown collision link: {link_name}")
            parent, link_offset = attachment(model, link_name)
            parent_idx = None if parent is None else robot.joint_name_to_index[parent]
            link_geoms[link_name] = []
            for collision in links[link_name].collisions:
                shape = collision.geometry.geometry
                if isinstance(shape, urdfpy.Box):
                    geom_type, size = (
                        mujoco.mjtGeom.mjGEOM_BOX,
                        [*(np.asarray(shape.size) / 2)],
                    )
                elif isinstance(shape, urdfpy.Cylinder):
                    geom_type, size = (
                        mujoco.mjtGeom.mjGEOM_CYLINDER,
                        [shape.radius, shape.length / 2],
                    )
                elif isinstance(shape, urdfpy.Sphere):
                    geom_type, size = mujoco.mjtGeom.mjGEOM_SPHERE, [shape.radius]
                else:
                    raise TypeError(
                        f"Unsupported collision primitive on {link_name}: {type(shape).__name__}"
                    )
                i = len(self.geometries)
                body = spec.worldbody.add_body(name=f"primitive_body_{i}", mocap=True)
                body.add_geom(
                    name=f"primitive_{i}",
                    type=geom_type,
                    size=size,
                    contype=0,
                    conaffinity=0,
                )
                self.geometries.append((parent_idx, link_offset @ collision.origin))
                link_geoms[link_name].append(i)
            if not link_geoms[link_name]:
                raise ValueError(f"No collision primitives on {link_name}")
        self.pairs = [
            (a, b)
            for first, second in link_pairs
            for a in link_geoms[first]
            for b in link_geoms[second]
        ]
        if not self.pairs:
            raise ValueError("No primitive collision pairs")
        parents = sorted(
            {parent for parent, _ in self.geometries if parent is not None}
        )
        self.relevant_joint_indices = (
            np.flatnonzero(np.any(robot.ancestor_mask[parents], axis=0))
            if parents
            else np.array([], dtype=int)
        )
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.geom_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"primitive_{i}")
            for i in range(len(self.geometries))
        ]
        self.mocap_ids = [
            self.model.body_mocapid[
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, f"primitive_body_{i}"
                )
            ]
            for i in range(len(self.geometries))
        ]

        @jax.jit
        def kinematics(q):
            tfs = robot.joint_to_world_transforms(q)
            jv, jw = robot._joint_jacobians(tfs)
            return tfs, jv, jw

        self.kinematics = kinematics

    def evaluate(self, q):
        tfs, jv, jw = (np.asarray(x) for x in self.kinematics(q))
        for mocap, (parent, offset) in zip(
            self.mocap_ids, self.geometries, strict=True
        ):
            pose = offset if parent is None else tfs[parent] @ offset
            self.data.mocap_pos[mocap] = pose[:3, 3]
            self.data.mocap_quat[mocap] = Rotation.from_matrix(pose[:3, :3]).as_quat(
                scalar_first=True
            )
        mujoco.mj_forward(self.model, self.data)
        distances, jacobians = [], []
        for first, second in self.pairs:
            points = np.zeros(6)
            distance = mujoco.mj_geomDistance(
                self.model,
                self.data,
                self.geom_ids[first],
                self.geom_ids[second],
                10.0,
                points,
            )
            delta = points[3:] - points[:3]
            length = np.linalg.norm(delta)
            if length > 1e-12:
                normal = delta / length * (-1.0 if distance < 0 else 1.0)
            else:
                normal = np.array([1.0, 0.0, 0.0])

            def point_jac(geom, point):
                parent = self.geometries[geom][0]
                if parent is None:
                    return np.zeros((3, self.robot.nv))
                lever = point - tfs[parent, :3, 3]
                return jv[parent] + np.cross(jw[parent].T, lever).T

            jac = normal @ (
                point_jac(second, points[3:]) - point_jac(first, points[:3])
            )
            distances.append(distance)
            jacobians.append(jac)
        return np.asarray(distances), np.asarray(jacobians)


@jax.tree_util.register_static
class PrimitivePairConstraint(InequalityConstraint):
    """Velocity damper using geometry distances and Jacobians supplied in ref."""

    def __init__(self, robot, num_rows, gain=10.0, margin=0.0, penalty=1e6):
        self.robot = robot
        self.num_rows = num_rows
        self.gain = gain
        self.margin = margin
        self.penalty = penalty

    def rows(self, q, ref):
        return -ref["pair_jacobian"], self.gain * (ref["pair_distance"] - self.margin)


def guard_step(
    checker,
    previous_q,
    proposed_q,
    previous_distances,
    previous_jacobian,
    tolerance=1e-6,
):
    """Backtrack when sampled configurations penetrate or worsen penetration."""
    # A previously intersecting pair may recover, but should not get worse.
    floor = np.minimum(previous_distances, 0.0) - tolerance
    if not np.isfinite(proposed_q).all():
        return previous_q, previous_distances, previous_jacobian, 0.0
    delta = proposed_q - previous_q
    relevant = checker.relevant_joint_indices
    max_delta = np.max(np.abs(delta[relevant]), initial=0.0)
    for fraction in (1.0, 0.5, 0.25, 0.125, 0.0625):
        samples = max(1, int(np.ceil(fraction * max_delta / 0.02)))
        if samples > 32:
            continue
        for sample in range(1, samples + 1):
            q = previous_q + fraction * sample / samples * delta
            distances, jacobian = checker.evaluate(q)
            if not (
                np.isfinite(distances).all()
                and np.isfinite(jacobian).all()
                and np.all(distances >= floor)
            ):
                break
        else:
            return q, distances, jacobian, fraction
    return previous_q, previous_distances, previous_jacobian, 0.0
