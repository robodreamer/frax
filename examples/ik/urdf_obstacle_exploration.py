"""Explore fixed-base bimanual IK and a spherical obstacle with a local URDF.

Collision primitives are enclosed by spheres for this experiment. This is a
conservative approximation and does not include self-collision constraints.
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"
os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=false --xla_cpu_scheduler_type=CPU_SCHEDULER_TYPE_MEMORY_OPTIMIZED"
)
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import threading
import time
from collections import deque
from pathlib import Path

import jax
import numpy as np
import viser
from scipy.spatial.transform import Rotation
from viser.extras import ViserUrdf

from frax import Robot
from frax.ik import (
    DiffIKSolver,
    FrameTask,
    JointLimits,
    JointTask,
    ObstacleConstraint,
    ResidualEquality,
    SphereObstacles,
)
from frax.utils import urdfpy
from frax.utils.rotation_utils import rmat_to_quat_wxyz
from frax.utils.urdf_parser import merge_fixed_links


def enclosing_spheres(link):
    positions, radii = [], []
    for collision in link.collisions:
        shape = collision.geometry.geometry
        if isinstance(shape, urdfpy.Sphere):
            radius = shape.radius
        elif isinstance(shape, urdfpy.Box):
            radius = np.linalg.norm(shape.size) / 2
        elif isinstance(shape, urdfpy.Cylinder):
            radius = np.hypot(shape.radius, shape.length / 2)
        else:
            raise TypeError(
                f"Unsupported collision primitive on {link.name}: {type(shape).__name__}"
            )
        positions.append(tuple(collision.origin[:3, 3]))
        radii.append(float(radius))
    return tuple(positions), tuple(radii)


def collision_data(path, robot):
    model = merge_fixed_links(urdfpy.URDF.load(path), [])
    joints = {j.name: j for j in model.joints}
    links = {link.name: link for link in model.links}
    spheres = [
        enclosing_spheres(links[joints[name].child]) for name in robot.joint_names
    ]
    root = enclosing_spheres(model.base_link)
    return {
        "positions": tuple(p for p, _ in spheres),
        "radii": tuple(r for _, r in spheres),
        "root_positions": root[0],
        "root_radii": root[1],
        "root_sc_pairs": (),
        "root_sc_tols": (),
        "body_sc_pairs": (),
        "body_sc_tols": (),
    }


def frame_definition(model, frame):
    by_child = {joint.child: joint for joint in model.joints}
    offset = np.eye(4)
    link = frame
    while link in by_child:
        joint = by_child[link]
        if joint.joint_type != "fixed":
            return joint.name, offset
        offset = joint.origin @ offset
        link = joint.parent
    raise ValueError(f"Frame {frame!r} is not attached to a moving joint")


def path_displacement(pattern, sim_time, index):
    phase = 2.0 * np.pi * 0.12 * sim_time
    direction = 1.0 if index == 0 else -1.0
    displacement = np.array([0.0, direction * 0.12 * np.sin(phase), 0.0])
    if pattern == "circle":
        displacement[2] = 0.12 * (1.0 - np.cos(phase))
    elif pattern != "sine":
        raise ValueError(f"Unknown path pattern: {pattern}")
    return displacement


class ActiveJointVisualizer:
    """Update only movable joint frames after ViserUrdf initializes the model.

    ViserUrdf's frame handles are private, so fall back to its public update
    method if a different Viser version does not expose the expected layout.
    Mimic joints and their drivers must be frozen by the caller.
    """

    def __init__(self, visual, robot, active, q_home, viewer_idxs):
        self.visual = visual
        self.viewer_idxs = viewer_idxs
        visual.update_cfg(q_home[viewer_idxs])
        joints = getattr(visual, "_joint_map_values", ())
        frames = getattr(visual, "_joint_frames", ())
        self.fallback = len(joints) != len(frames) or not joints
        if self.fallback:
            return
        selected = [
            (j, f) for j, f in zip(joints, frames, strict=True) if j.name in active
        ]
        if {j.name for j, _ in selected} != active:
            self.fallback = True
            return
        self.revolute = [
            (j, f) for j, f in selected if j.type in ("revolute", "continuous")
        ]
        self.prismatic = [(j, f) for j, f in selected if j.type == "prismatic"]
        if len(self.revolute) + len(self.prismatic) != len(selected):
            self.fallback = True
            return
        self.rot_indices = np.array(
            [robot.joint_name_to_index[j.name] for j, _ in self.revolute], dtype=int
        )
        self.rot_axes = np.array([j.axis for j, _ in self.revolute]).reshape(-1, 3)
        if self.revolute:
            self.rot_axes /= np.linalg.norm(self.rot_axes, axis=1, keepdims=True)
        if self.revolute:
            self.base_rot = Rotation.from_matrix(
                np.array(
                    [
                        (np.eye(4) if j.origin is None else j.origin)[:3, :3]
                        for j, _ in self.revolute
                    ]
                )
            )
        self.prev_q = q_home.copy()
        self.robot = robot

    def update(self, q):
        if self.fallback:
            self.visual.update_cfg(q[self.viewer_idxs])
            return
        if self.revolute and np.any(
            q[self.rot_indices] != self.prev_q[self.rot_indices]
        ):
            rotations = self.base_rot * Rotation.from_rotvec(
                self.rot_axes * q[self.rot_indices, None]
            )
            for (_, frame), quat, idx in zip(
                self.revolute,
                rotations.as_quat(scalar_first=True),
                self.rot_indices,
                strict=True,
            ):
                if q[idx] != self.prev_q[idx]:
                    frame.wxyz = quat
        for joint, frame in self.prismatic:
            idx = self.robot.joint_name_to_index[joint.name]
            if q[idx] != self.prev_q[idx]:
                origin = np.eye(4) if joint.origin is None else joint.origin
                frame.position = origin[:3, 3] + origin[:3, :3] @ joint.axis * q[idx]
        self.prev_q = q.copy()


def main(args):
    original = urdfpy.URDF.load(args.urdf)
    frames = [frame_definition(original, name) for name in args.frames]
    bare = Robot(args.urdf)
    robot = Robot(args.urdf, collision_data=collision_data(args.urdf, bare))
    q_home = np.clip(
        np.zeros(robot.nq),
        robot.actuated_joint_lower_limits,
        robot.actuated_joint_upper_limits,
    )
    active = {
        name
        for name in robot.joint_names
        if any(name.startswith(p) for p in args.active_prefix)
    }
    if not active:
        raise ValueError("No joints matched --active-prefix")
    for name in args.control_joints:
        if name not in active:
            raise ValueError(f"Controlled joint {name!r} is not active")
    for joint in original.joints:
        if joint.mimic is None:
            continue
        master = joint.mimic.joint
        if joint.name in active or master in active:
            raise ValueError("Mimic joints and their drivers must remain frozen")
        multiplier = joint.mimic.multiplier
        offset = joint.mimic.offset
        expected = (1.0 if multiplier is None else multiplier) * q_home[
            robot.joint_name_to_index[master]
        ] + (0.0 if offset is None else offset)
        if not np.isclose(q_home[robot.joint_name_to_index[joint.name]], expected):
            raise ValueError("Frozen initial configuration violates a mimic joint")
    frozen = np.array(
        [i for i, name in enumerate(robot.joint_names) if name not in active], dtype=int
    )
    indices = np.array([robot.joint_name_to_index[name] for name, _ in frames])
    offsets = np.array([offset for _, offset in frames])
    posture_gains = np.ones(robot.nv)
    posture_weights = np.full(robot.nv, 0.1)
    for name in args.control_joints:
        i = robot.joint_name_to_index[name]
        posture_gains[i] = 4.0
        posture_weights[i] = 1.0
    tasks = [
        FrameTask(robot, indices, gain_pos=5.0, gain_rot=5.0, offsets=offsets),
        JointTask(
            robot,
            np.arange(robot.nv),
            "q_posture",
            gain=posture_gains,
            weight=posture_weights,
        ),
    ]
    if args.num_obstacles < 1:
        raise ValueError("At least one obstacle is required")
    obstacle = SphereObstacles(
        np.zeros((args.num_obstacles, 3)),
        np.full(args.num_obstacles, args.radius),
        np.ones(args.num_obstacles, dtype=bool),
    )
    constraints = [
        JointLimits(robot),
        ObstacleConstraint(robot, obstacle, ref_key="obstacle"),
    ]
    pair_checker = None
    if args.self_pair:
        from primitive_pair_collision import (
            PrimitivePairChecker,
            PrimitivePairConstraint,
        )

        pair_checker = PrimitivePairChecker(original, robot, args.self_pair)
        pair_distances, pair_jacobian = pair_checker.evaluate(q_home)
        if (
            not np.isfinite(pair_distances).all()
            or not np.isfinite(pair_jacobian).all()
        ):
            raise ValueError("Non-finite initial primitive pair distances or Jacobians")
        constraints.append(
            PrimitivePairConstraint(robot, len(pair_distances), margin=0.002)
        )
    if len(frozen):
        constraints.append(
            ResidualEquality(
                robot, lambda q, ref: q[frozen] - ref["q_home"][frozen], len(frozen)
            )
        )
    solver = DiffIKSolver(robot, tasks, constraints, damping=1e-3)
    step = jax.jit(solver.step)
    frame_tfs = jax.jit(lambda q: robot.joint_to_world_transforms(q)[indices] @ offsets)
    sphere_pos = jax.jit(robot.link_collision_positions)
    start_tfs = np.asarray(frame_tfs(q_home))
    initial_centers = []
    for i in range(args.num_obstacles):
        if i < 2:
            center = start_tfs[i, :3, 3] + np.array(
                [0.0, 0.25 if i == 0 else -0.25, 0.0]
            )
        else:
            center = start_tfs[:, :3, 3].mean(axis=0) + np.array(
                [0.55, 0.2 * (i - 2), 0.2]
            )
        initial_centers.append(center)

    server = viser.ViserServer(port=args.port)
    server.scene.add_grid("/grid", width=3.0, height=3.0)
    visual = ViserUrdf(server, Path(args.urdf), root_node_name="/robot")
    viewer_names = visual.get_actuated_joint_names()
    viewer_idxs = [robot.joint_name_to_index[n] for n in viewer_names]
    visual_updates = ActiveJointVisualizer(visual, robot, active, q_home, viewer_idxs)
    mode_control = server.gui.add_dropdown(
        "Mode", ("teleop", "path"), initial_value=args.mode
    )
    pattern_control = server.gui.add_dropdown(
        "Path pattern", ("sine", "circle"), initial_value=args.path_pattern
    )
    reset_button = server.gui.add_button("Reset scene")
    reset_requested = threading.Event()

    @reset_button.on_click
    def request_reset(_):
        reset_requested.set()

    teleop_controls, path_controls = [], []
    for i, tf in enumerate(start_tfs):
        position = tf[:3, 3]
        wxyz = np.asarray(rmat_to_quat_wxyz(tf[:3, :3]))
        teleop_controls.append(
            server.scene.add_transform_controls(
                f"/target/teleop/{i}",
                position=position,
                wxyz=wxyz,
                scale=0.15,
                line_width=1.5,
                visible=args.mode == "teleop",
            )
        )
        path_controls.append(
            server.scene.add_frame(
                f"/target/path/{i}",
                position=position,
                wxyz=wxyz,
                axes_length=0.1,
                axes_radius=0.002,
                origin_radius=0.004,
                visible=args.mode == "path",
            )
        )
    controls = teleop_controls if args.mode == "teleop" else path_controls
    obstacle_controls, obstacle_enabled = [], []
    for i, center in enumerate(initial_centers):
        control = server.scene.add_transform_controls(
            f"/obstacle/{i}",
            position=center,
            scale=0.15,
            line_width=1.5,
            disable_rotations=True,
        )
        server.scene.add_icosphere(
            f"/obstacle/{i}/sphere",
            radius=args.radius,
            color=(230, 50, 50),
            opacity=0.5,
        )
        obstacle_controls.append(control)
        obstacle_enabled.append(
            server.gui.add_checkbox(f"Avoid obstacle {i + 1}", True)
        )
    show_spheres = server.gui.add_checkbox("Show approximate collision spheres", False)
    show_desired_traces = server.gui.add_checkbox("Show commanded traces", True)
    show_actual_traces = server.gui.add_checkbox("Show actual traces", True)
    joint_controls = {}
    for name in args.control_joints:
        i = robot.joint_name_to_index[name]
        lower, upper = (
            robot.actuated_joint_lower_limits[i],
            robot.actuated_joint_upper_limits[i],
        )
        if not np.isfinite(lower) or not np.isfinite(upper):
            raise ValueError(
                f"Controlled joint {name!r} must have finite position limits"
            )
        joint_controls[i] = server.gui.add_slider(
            name,
            min=float(lower),
            max=float(upper),
            step=0.01,
            initial_value=float(q_home[i]),
        )
    status = server.gui.add_markdown("Starting solver...")
    radii = np.asarray(robot.flat_collision_radii)
    markers = [
        server.scene.add_icosphere(
            f"/collision/{i}",
            radius=r,
            color=(50, 140, 230),
            opacity=0.25,
            visible=False,
        )
        for i, r in enumerate(radii)
    ]
    state = solver.init(q_home)
    solve_times = deque(maxlen=100)
    geometry_times = deque(maxlen=100)
    work_times = deque(maxlen=100)
    viewer_times = deque(maxlen=100)
    periods = deque(maxlen=100)
    previous_start = None
    last_status = 0.0
    spheres_visible = False
    desired_history = [deque(maxlen=500) for _ in controls]
    actual_history = [deque(maxlen=500) for _ in controls]
    desired_paths = [None] * len(controls)
    actual_paths = [None] * len(controls)
    path_time = 0.0
    path_origins = start_tfs[:, :3, 3].copy()
    current_mode = args.mode
    current_pattern = args.path_pattern
    step_count = 0
    try:
        while True:
            started = time.perf_counter()
            if previous_start is not None:
                periods.append(started - previous_start)
            previous_start = started
            if reset_requested.is_set():
                reset_requested.clear()
                state = solver.init(q_home)
                if pair_checker is not None:
                    pair_distances, pair_jacobian = pair_checker.evaluate(q_home)
                current_mode = mode_control.value
                current_pattern = pattern_control.value
                controls = (
                    teleop_controls if current_mode == "teleop" else path_controls
                )
                path_origins = start_tfs[:, :3, 3].copy()
                path_time = 0.0
                step_count = 0
                with server.atomic():
                    for i, tf in enumerate(start_tfs):
                        for target in (teleop_controls[i], path_controls[i]):
                            target.position = tf[:3, 3]
                            target.wxyz = np.asarray(rmat_to_quat_wxyz(tf[:3, :3]))
                        teleop_controls[i].visible = current_mode == "teleop"
                        path_controls[i].visible = current_mode == "path"
                    for i, control in enumerate(obstacle_controls):
                        control.position = initial_centers[i]
                        obstacle_enabled[i].value = True
                    for index, control in joint_controls.items():
                        control.value = float(q_home[index])
                    show_spheres.value = False
                    for history in desired_history + actual_history:
                        history.clear()
                    for handles in (desired_paths, actual_paths):
                        for i, handle in enumerate(handles):
                            if handle is not None:
                                handle.remove()
                                handles[i] = None
            requested_mode = mode_control.value
            requested_pattern = pattern_control.value
            if requested_mode != current_mode:
                next_controls = (
                    teleop_controls if requested_mode == "teleop" else path_controls
                )
                with server.atomic():
                    for old, new in zip(controls, next_controls, strict=True):
                        new.position = np.asarray(old.position).copy()
                        new.wxyz = np.asarray(old.wxyz).copy()
                        old.visible = False
                        new.visible = True
                controls = next_controls
                current_mode = requested_mode
                path_origins = np.array([control.position for control in controls])
                path_time = 0.0
            if requested_pattern != current_pattern:
                current_pattern = requested_pattern
                if current_mode == "path":
                    path_origins = np.array([control.position for control in controls])
                    path_time = 0.0
            q_posture = q_home.copy()
            for index, control in joint_controls.items():
                q_posture[index] = control.value
            if current_mode == "path":
                for i, control in enumerate(controls):
                    control.position = path_origins[i] + path_displacement(
                        current_pattern, path_time, i
                    )
            obstacle_centers = np.array([c.position for c in obstacle_controls])
            ref = {
                "frame_pos": np.array([c.position for c in controls]),
                "frame_rmat": Rotation.from_quat(
                    np.array([c.wxyz for c in controls]), scalar_first=True
                ).as_matrix(),
                "q_home": q_home,
                "q_posture": q_posture,
                "obstacle": SphereObstacles(
                    obstacle_centers,
                    np.full(args.num_obstacles, args.radius),
                    np.array([c.value for c in obstacle_enabled]),
                ),
            }
            if pair_checker is not None:
                ref["pair_distance"] = pair_distances
                ref["pair_jacobian"] = pair_jacobian
            solve_started = time.perf_counter()
            previous_state = state
            candidate, _, info = step(state, ref, 0.01)
            q = np.asarray(candidate.q)
            solve_times.append(time.perf_counter() - solve_started)
            step_fraction = 1.0
            if pair_checker is not None:
                from primitive_pair_collision import guard_step

                geometry_started = time.perf_counter()
                q, pair_distances, pair_jacobian, step_fraction = guard_step(
                    pair_checker,
                    np.asarray(previous_state.q),
                    q,
                    pair_distances,
                    pair_jacobian,
                )
                geometry_times.append(time.perf_counter() - geometry_started)
                state = (
                    previous_state if step_fraction == 0 else candidate._replace(q=q)
                )
            else:
                state = candidate
            centers = np.asarray(sphere_pos(q))
            clearance = (
                np.linalg.norm(
                    centers[:, None, :] - obstacle_centers[None, :, :], axis=2
                )
                - radii[:, None]
                - args.radius
            )
            if step_count % 5 == 0:
                actual_positions = np.asarray(frame_tfs(q))[:, :3, 3]
                for i in range(len(controls)):
                    desired_history[i].append(ref["frame_pos"][i].copy())
                    actual_history[i].append(actual_positions[i].copy())
            viewer_started = time.perf_counter()
            with server.atomic():
                visual_updates.update(q)
                for handles, checkbox in (
                    (desired_paths, show_desired_traces),
                    (actual_paths, show_actual_traces),
                ):
                    for handle in handles:
                        if handle is not None:
                            handle.visible = checkbox.value
                if show_spheres.value:
                    for marker, pos in zip(markers, centers, strict=True):
                        marker.position = pos
                        if not spheres_visible:
                            marker.visible = True
                elif spheres_visible:
                    for marker in markers:
                        marker.visible = False
                spheres_visible = show_spheres.value
                if step_count % 5 == 0:
                    for i in range(len(controls)):
                        if len(desired_history[i]) < 2:
                            continue
                        for history, handles, name, color, checkbox in (
                            (
                                desired_history[i],
                                desired_paths,
                                "desired",
                                (245, 135, 35),
                                show_desired_traces,
                            ),
                            (
                                actual_history[i],
                                actual_paths,
                                "actual",
                                (40, 150, 240),
                                show_actual_traces,
                            ),
                        ):
                            points = np.asarray(history)
                            segments = np.stack((points[:-1], points[1:]), axis=1)
                            if handles[i] is None:
                                handles[i] = server.scene.add_line_segments(
                                    f"/paths/{name}/{i}",
                                    segments,
                                    colors=color,
                                    thickness=0.005,
                                    visible=checkbox.value,
                                )
                            else:
                                handles[i].points = segments
            viewer_times.append(time.perf_counter() - viewer_started)
            work_times.append(time.perf_counter() - started)
            if started - last_status >= 0.5 and periods:
                pair_status = (
                    f"Selected primitive-pair clearance: {1000 * pair_distances.min():.1f} mm; step fraction: {step_fraction:.3f}  \n"
                    if pair_checker is not None
                    else "Self-collision pairs are not constrained.  \n"
                )
                status.content = (
                    f"Approximate moving-link clearance: {1000 * clearance.min():.0f} mm  \n"
                    f"Mode: {current_mode}; traces: orange target, blue actual  \n"
                    f"{pair_status}"
                    f"QP converged: {bool(info.qp_converged)}  \n"
                    f"Solver median: {1000 * np.median(solve_times):.2f} ms; "
                    f"pair check: {1000 * np.median(geometry_times) if geometry_times else 0.0:.2f} ms; "
                    f"viewer update: {1000 * np.median(viewer_times):.2f} ms; "
                    f"loop work median/p95: {1000 * np.median(work_times):.2f}/{1000 * np.percentile(work_times, 95):.2f} ms; "
                    f"loop rate: {1 / np.median(periods):.0f} Hz  \n"
                    "Fixed-root external obstacles and unselected self-collision pairs are not constrained."
                )
                last_status = started
            time.sleep(max(0.0, 0.01 - (time.perf_counter() - started)))
            if current_mode == "path":
                path_time += 0.01
            step_count += 1
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--frames", nargs=2, required=True)
    parser.add_argument("--active-prefix", nargs="+", required=True)
    parser.add_argument("--control-joints", nargs="*", default=[])
    parser.add_argument(
        "--self-pair",
        nargs=2,
        action="append",
        default=[],
        metavar=("LINK_A", "LINK_B"),
    )
    parser.add_argument("--radius", type=float, default=0.08)
    parser.add_argument("--num-obstacles", type=int, default=1)
    parser.add_argument("--mode", choices=("teleop", "path"), default="teleop")
    parser.add_argument("--path-pattern", choices=("sine", "circle"), default="sine")
    parser.add_argument("--port", type=int, default=8080)
    main(parser.parse_args())
