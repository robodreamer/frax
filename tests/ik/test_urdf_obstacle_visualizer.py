import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip("viser")
yourdfpy = pytest.importorskip("yourdfpy")

path = Path(__file__).parents[2] / "examples/ik/urdf_obstacle_exploration.py"
spec = importlib.util.spec_from_file_location("urdf_obstacle_exploration", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("pattern", ["sine", "circle"])
def test_path_starts_at_home_and_moves_both_targets(pattern):
    np.testing.assert_allclose(module.path_displacement(pattern, 0.0, 0), [0, 0, 0])
    quarter_period = 1.0 / (4 * 0.12)
    left = module.path_displacement(pattern, quarter_period, 0)
    right = module.path_displacement(pattern, quarter_period, 1)
    np.testing.assert_allclose(left[1], 0.12)
    np.testing.assert_allclose(right[1], -0.12)
    np.testing.assert_allclose(left[2], 0.12 if pattern == "circle" else 0.0)


def test_active_joint_frames_match_urdf_transforms():
    origin = np.eye(4)
    origin[:3, :3] = Rotation.from_euler("xyz", [0.3, -0.2, 0.1]).as_matrix()
    origin[:3, 3] = [0.4, 0.5, 0.6]
    joints = [
        yourdfpy.Joint(
            name="rot", type="revolute", origin=origin, axis=np.array([0.0, 0.0, 1.0])
        ),
        yourdfpy.Joint(
            name="slide",
            type="prismatic",
            origin=origin,
            axis=np.array([1.0, 0.0, 0.0]),
        ),
    ]
    frames = [SimpleNamespace() for _ in joints]

    class Visual:
        _joint_map_values = joints
        _joint_frames = frames

        def update_cfg(self, _):
            for joint, frame in zip(joints, frames, strict=True):
                frame.position = joint.origin[:3, 3].copy()
                frame.wxyz = Rotation.from_matrix(joint.origin[:3, :3]).as_quat(
                    scalar_first=True
                )

    robot = SimpleNamespace(joint_name_to_index={"rot": 0, "slide": 1})
    visual = module.ActiveJointVisualizer(
        Visual(), robot, {"rot", "slide"}, np.zeros(2), [0, 1]
    )
    assert not visual.fallback
    q = np.array([0.7, 0.2])
    visual.update(q)
    for i, (joint, frame) in enumerate(zip(joints, frames, strict=True)):
        expected, _ = yourdfpy.URDF._forward_kinematics_joint(None, joint, float(q[i]))
        actual = np.eye(4)
        actual[:3, :3] = Rotation.from_quat(frame.wxyz, scalar_first=True).as_matrix()
        actual[:3, 3] = frame.position
        np.testing.assert_allclose(actual, expected, atol=1e-12)
