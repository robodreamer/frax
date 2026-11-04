# Fixed-base URDF obstacle exploration

`examples/ik/urdf_obstacle_exploration.py` loads a local URDF, tracks two
user-selected frames, and adds a movable spherical obstacle to differential IK.
Install the source checkout with the `[examples]` extra before running it.

For example, replace the frame and joint names with names from your model:

```bash
python examples/ik/urdf_obstacle_exploration.py \
  --urdf /path/to/robot.urdf \
  --frames left_tool_frame right_tool_frame \
  --active-prefix left_arm_ right_arm_ torso_ \
  --control-joints torso_pitch_joint \
  --self-pair base_link leg_link \
  --num-obstacles 3 --mode path --path-pattern circle \
  --port 8080
```

Open `http://localhost:8080`. The Mode dropdown switches between teleoperation
and a repeating path, and the Path pattern dropdown selects `sine` or `circle`.
The corresponding CLI flags set their initial values. In teleop mode, the frame
gizmos set hand pose targets. Orange and blue lines show the recent commanded
and actual frame paths; their visibility can be controlled independently with
the trace checkboxes. Reset scene returns the robot, targets, obstacles, joint
sliders, and traces to their initial state while keeping the selected mode and
trace visibility settings.
The red gizmos move the obstacles, whose avoidance can be toggled separately.
Optional joint sliders set posture targets in radians. Joint targets are weighted
objectives, so they can trade off with hand tracking and constraints. The initial
configuration is zero clipped to the URDF joint limits; choose a suitable seed
before using a different model.

The obstacle model encloses each URDF sphere, box, or cylinder primitive in a
sphere. It can be substantially more conservative than the original shapes;
mesh collision primitives are not supported by this example. The viewer can
show these approximate spheres. The displayed clearance and obstacle constraint
cover moving-link spheres only. Repeat `--self-pair LINK_A LINK_B` to constrain
selected self-collision link pairs using their original box, cylinder, and sphere
primitives. The example checks every geometry combination for each selected pair,
adds velocity constraints, and backtracks a proposed step if sampled poses would
penetrate or worsen an existing penetration. Sampling is not continuous collision
detection; unselected pairs are unchecked. Fixed-root external obstacles, balance,
wheel motion and dynamics are also not constrained. FRAX's inequalities are
elastic, so a converged solve does not guarantee collision-free motion.

Joint names selected by `--active-prefix` may move; all other joints are held
at their initial values. Mimic joints and their drivers must stay frozen because
this exploration does not implement their coupling in FRAX. The live timing
readout measures the solver and server loop, not network or browser rendering
latency. This example is for local visualization and algorithm exploration.
