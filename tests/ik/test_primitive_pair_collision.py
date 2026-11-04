import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

from frax import Robot
from frax.utils import urdfpy

path = Path(__file__).parents[2] / "examples/ik/primitive_pair_collision.py"
spec = importlib.util.spec_from_file_location("primitive_pair_collision", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def checker(tmp_path):
    path = tmp_path / "primitives.urdf"
    path.write_text("""<robot name="primitive_pair">
      <link name="root">
        <inertial><mass value="1"/><inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial>
        <collision><geometry><sphere radius="0.5"/></geometry></collision>
      </link>
      <link name="moving">
        <inertial><mass value="1"/><inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial>
        <collision><geometry><sphere radius="0.5"/></geometry></collision>
      </link>
      <joint name="slide" type="prismatic"><parent link="root"/><child link="moving"/>
        <origin xyz="1.5 0 0"/><axis xyz="1 0 0"/>
        <limit lower="-4" upper="4" effort="1" velocity="1"/></joint>
    </robot>""")
    robot = Robot(path)
    return module.PrimitivePairChecker(
        urdfpy.URDF.load(path), robot, [("root", "moving")]
    )


def test_primitive_distance_jacobian_matches_finite_difference(checker):
    q = np.array([0.1])
    distance, jacobian = checker.evaluate(q)
    eps = 1e-6
    finite_diff = (checker.evaluate(q + eps)[0] - checker.evaluate(q - eps)[0]) / (
        2 * eps
    )
    np.testing.assert_allclose(distance, [0.6], atol=1e-10)
    np.testing.assert_allclose(jacobian[:, 0], finite_diff, atol=1e-8)


def test_guard_checks_interior_and_rejects_worsening_penetration(checker):
    q = np.array([0.0])
    d, jac = checker.evaluate(q)
    # Both endpoints are separated, but the path between them intersects.
    accepted, distance, _, fraction = module.guard_step(
        checker, q, np.array([-3.0]), d, jac
    )
    assert fraction < 1.0
    assert distance.min() >= -1e-6
    assert accepted[0] > -0.5
    q = np.array([-0.6])
    d, jac = checker.evaluate(q)
    accepted, _, _, fraction = module.guard_step(checker, q, np.array([-0.8]), d, jac)
    assert fraction == 0.0
    np.testing.assert_array_equal(accepted, q)
