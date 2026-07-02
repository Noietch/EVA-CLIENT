# UR5e + DH AG95 Assets

UR5e arm meshes and kinematic parameters:
- Source: https://github.com/UniversalRobots/Universal_Robots_ROS2_Description
- Branch checked during vendoring: rolling
- Commit: e2d047f87148396d7a5e3fddbe589c25a5b26578
- Vendored paths: `meshes/ur5e/visual`, `meshes/ur5e/collision`
- Upstream license: BSD-3-Clause, see upstream `LICENSE`

DH AG95 gripper URDF macro and meshes:
- Source: https://github.com/ian-chuang/dh_ag95_gripper_ros2
- Branch checked during vendoring: humble
- Commit: fc4f80fdfb3acae5626df4359aec1401cb71a9a3
- Vendored paths: `meshes/ag95/visual`, `meshes/ag95/collision`
- Description package license: Apache-2.0 according to `dh_ag95_description/package.xml`
- Repository root license: MIT

The final `urdf/ur5e.urdf` is flattened for EVA Client and uses local mesh paths. UR5e
links and joints are generated from the pinned Universal Robots checkout YAML configs,
and AG95 links and joints are generated from the pinned AG95 xacro.
