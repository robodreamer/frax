import numpy as np

from frax.utils import urdfpy
from frax.utils.urdf_parser import merge_fixed_links


def test_fixed_chain_reparents_descendant_joint(tmp_path):
    # Declare the child fixed joint first to exercise a merge that proceeds
    # from the tip of the chain toward the root.
    path = tmp_path / "branched.urdf"
    path.write_text("""<robot name="fixed_chain">
      <link name="root"/><link name="a"/><link name="b"/><link name="tip"/><link name="side"/>
      <joint name="b_fixed" type="fixed"><parent link="a"/><child link="b"/>
        <origin xyz="0 2 0"/></joint>
      <joint name="tip_joint" type="revolute"><parent link="b"/><child link="tip"/>
        <origin xyz="0 0 3"/><axis xyz="0 0 1"/>
        <limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
      <joint name="side_joint" type="revolute"><parent link="b"/><child link="side"/>
        <origin xyz="0 0 4"/><axis xyz="0 0 1"/>
        <limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
      <joint name="a_fixed" type="fixed"><parent link="root"/><child link="a"/>
        <origin xyz="1 0 0"/></joint>
    </robot>""")
    merged = merge_fixed_links(urdfpy.URDF.load(path), [])
    joints = {joint.name: joint for joint in merged.joints}
    assert set(joints) == {"tip_joint", "side_joint"}
    for name, z in (("tip_joint", 3), ("side_joint", 4)):
        assert joints[name].parent == "root"
        np.testing.assert_allclose(joints[name].origin[:3, 3], [1, 2, z])
