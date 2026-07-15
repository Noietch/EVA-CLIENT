import ast
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
R1_LITE_DIR = REPO_ROOT / "examples" / "hardware" / "r1_lite"

ALLOWED_FILES = {
    "README.md",
    "cat_operator_button.py",
    "fake_node.py",
    "kill_hardware.sh",
    "reset_torso.py",
    "run_fake_node.sh",
    "run_hardware.sh",
}

REMOVED_HELPERS = {
    "cat_collect_gate.sh",
    "cat_start_teleop.sh",
    "deploy_cat_collect_gate.sh",
    "eva-switch-ros2-lan",
    "torso_values_2026-07-01.md",
}


def test_r1lite_directory_keeps_only_current_entrypoints():
    files = {path.name for path in R1_LITE_DIR.iterdir() if path.is_file()}
    dirs = {
        path.name
        for path in R1_LITE_DIR.iterdir()
        if path.is_dir() and path.name != "__pycache__"
    }

    assert files == ALLOWED_FILES
    assert dirs == {"fastdds"}
    assert files.isdisjoint(REMOVED_HELPERS)


def test_r1lite_fastdds_directory_contains_sanitized_profile_template():
    fastdds_dir = R1_LITE_DIR / "fastdds"
    files = {path.name for path in fastdds_dir.iterdir() if path.is_file()}

    assert files == {"fastdds_super_client.example.xml"}
    template = fastdds_dir / "fastdds_super_client.example.xml"
    root = ET.parse(template).getroot()
    xml = template.read_text()

    assert any(element.text == "UDPv4" for element in root.iter() if element.tag.endswith("type"))
    assert any(
        element.text == "16777216"
        for element in root.iter()
        if element.tag.endswith("receiveBufferSize")
    )
    assert '<discoveryProtocol>SUPER_CLIENT</discoveryProtocol>' in xml
    assert "__EVA_LOCAL_IP__" in xml
    assert "__DISCOVERY_SERVER_GUID_PREFIX__" in xml
    assert "__DISCOVERY_SERVER_IP__" in xml
    assert "__DISCOVERY_SERVER_PORT__" in xml


def test_shell_entrypoints_have_valid_bash_syntax():
    scripts = [
        R1_LITE_DIR / "run_hardware.sh",
        R1_LITE_DIR / "kill_hardware.sh",
        R1_LITE_DIR / "run_fake_node.sh",
    ]

    subprocess.run(["bash", "-n", *map(str, scripts)], check=True)


def test_python_entrypoints_parse_without_importing_ros_dependencies():
    for path in R1_LITE_DIR.glob("*.py"):
        ast.parse(path.read_text(), filename=str(path))


def test_run_hardware_starts_r1lite_and_cat_through_eva_hil_bootstrap():
    script = (R1_LITE_DIR / "run_hardware.sh").read_text()

    assert 'R1LITE_HOST="${R1LITE_HOST:-r1lite@192.168.12.12}"' in script
    assert 'CAT_HOST="${CAT_HOST:-cat@192.168.12.13}"' in script
    assert (
        'R1LITE_BODY_BOOTSTRAP_REMOTE="${R1LITE_BODY_BOOTSTRAP_REMOTE:-/tmp/eva_r1lite_body_no_teleop_bootstrap}"'
        in script
    )
    assert (
        'R1LITE_BODY_SESSION_SOURCE="${R1LITE_BODY_SESSION_SOURCE:-/home/r1lite/galaxea/install/startup_config/share/startup_config/sessions.d/ATCStandard/R1LITEBody.d}"'
        in script
    )
    assert (
        'R1LITE_BODY_SESSION_REMOTE="${R1LITE_BODY_SESSION_REMOTE:-/tmp/eva_r1lite_body_no_teleop.d}"'
        in script
    )
    assert 'R1LITE_START_CMD="${R1LITE_START_CMD:-${R1LITE_BODY_BOOTSTRAP_REMOTE}}"' in script
    assert "write_r1lite_body_bootstrap_script" in script
    assert "install_r1lite_body_bootstrap_script" in script
    assert 'cp "${R1LITE_BODY_SESSION_SOURCE}/${yaml}" "${R1LITE_BODY_SESSION_REMOTE}/${yaml}"' in script
    assert 'for yaml in hdas.yaml mobiman.yaml ros2_discovery.yaml system.yaml tools.yaml; do' in script
    assert 'tmuxp load "${R1LITE_BODY_SESSION_REMOTE}/${yaml}" -d' in script
    assert (
        "R1LITE_BODY_SESSION_SOURCE='${R1LITE_BODY_SESSION_SOURCE}' "
        "R1LITE_BODY_SESSION_REMOTE='${R1LITE_BODY_SESSION_REMOTE}'"
        in script
    )
    assert (
        'start_remote_tmux "${R1LITE_HOST}" "${R1LITE_SESSION}" '
        '"R1LITE_BODY_SESSION_SOURCE=\'${R1LITE_BODY_SESSION_SOURCE}\' '
        "R1LITE_BODY_SESSION_REMOTE='${R1LITE_BODY_SESSION_REMOTE}' ${R1LITE_START_CMD}\""
        in script
    )
    assert "cd ~/workspace/robot && bash start_robot.sh" not in script
    assert "R1LITEBodyTakeover.d" not in script
    assert "start_r1lite_tabletop_gello_teleop.sh" not in script
    assert 'CAT_HIL_TELEOP_REMOTE="${CAT_HIL_TELEOP_REMOTE:-/tmp/eva_cat_hil_teleop_bootstrap}"' in script
    assert 'CAT_FASTRTPS_PROFILE="${CAT_FASTRTPS_PROFILE:-/tmp/eva_cat_fastdds_super_client.xml}"' in script
    assert 'CAT_ROS2_LOCAL_IP="${CAT_ROS2_LOCAL_IP:-192.168.12.13}"' in script
    assert 'R1LITE_DISCOVERY_SERVER_IP="${R1LITE_DISCOVERY_SERVER_IP:-192.168.12.12}"' in script
    assert "write_cat_fastdds_profile" in script
    assert "install_cat_fastdds_profile" in script
    assert "<discoveryProtocol>SUPER_CLIENT</discoveryProtocol>" in script
    assert "<address>${cat_ros2_local_ip}</address>" in script
    assert "<address>${server_ip}</address>" in script
    assert "<port>${server_port}</port>" in script
    assert "write_cat_hil_teleop_script" in script
    assert "install_cat_hil_teleop_script" in script
    assert 'install_cat_fastdds_profile "${CAT_HOST}" "${CAT_FASTRTPS_PROFILE}"' in script
    assert "/tabletop/hdas/feedback_arm_left:=/eva/hil/input_joint_state_arm_left" in script
    assert "/tabletop/hdas/feedback_arm_right:=/eva/hil/input_joint_state_arm_right" in script
    assert 'export FASTRTPS_DEFAULT_PROFILES_FILE="${CAT_FASTRTPS_PROFILE}"' in script
    assert "startup_config/script/prestart/fastrtps_profiles.xml" not in script
    assert "bash /home/cat/start_tele.sh" not in script
    assert "cat_start_teleop.sh" not in script
    assert "CAT_GATE" not in script
    assert "EVA_CAT_GATE_FILE" not in script
    assert "CAT_START_TELEOP_REMOTE" not in script
    assert "cat_gate_command" not in script
    assert "install_cat_start_teleop_script" not in script
    assert "publish_if_collect_enabled" not in script
    assert "motion_target/target_joint_state_arm_left" not in script
    assert "motion_target/target_joint_state_arm_right" not in script


def test_run_hardware_copies_and_starts_operator_button_node():
    script = (R1_LITE_DIR / "run_hardware.sh").read_text()

    assert (
        'CAT_OPERATOR_BUTTON_LOCAL="${CAT_OPERATOR_BUTTON_LOCAL:-${SCRIPT_DIR}/cat_operator_button.py}"'
        in script
    )
    assert (
        'CAT_OPERATOR_BUTTON_REMOTE="${CAT_OPERATOR_BUTTON_REMOTE:-/tmp/eva_cat_operator_button.py}"'
        in script
    )
    assert (
        'EVA_OPERATOR_BUTTON_TOPIC="${EVA_OPERATOR_BUTTON_TOPIC:-/eva/operator_button}"'
        in script
    )
    assert (
        'install_remote_script "${CAT_HOST}" "${CAT_OPERATOR_BUTTON_LOCAL}" '
        '"${CAT_OPERATOR_BUTTON_REMOTE}"'
    ) in script
    assert 'python3 "${CAT_OPERATOR_BUTTON_PATH}"' in script
    assert "patch_gello" not in script
    assert "patch_joy" not in script


def test_cat_operator_button_maps_joy_edges_to_eva_events():
    script = (R1_LITE_DIR / "cat_operator_button.py").read_text()

    assert "RIGHT_JOY_BUTTONS = {" in script
    assert "LEFT_JOY_BUTTONS = {" in script
    assert '2: "x"' in script
    assert '3: "y"' in script
    assert '4: "left_gripper_open"' in script
    assert '5: "left_gripper_close"' in script
    assert '4: "right_gripper_open"' in script
    assert '5: "right_gripper_close"' in script
    assert "JoyButtonEdgeAdapter" in script
    assert "self.previous_buttons = current" in script
    assert "self.pub = self.create_publisher(String, output_topic, 10)" in script
    assert "message.data = label" in script


def test_kill_hardware_runs_remote_vendor_kill_scripts():
    script = (R1_LITE_DIR / "kill_hardware.sh").read_text()

    assert 'R1LITE_HOST="${R1LITE_HOST:-r1lite@192.168.12.12}"' in script
    assert 'CAT_HOST="${CAT_HOST:-cat@192.168.12.13}"' in script
    assert "START_R1LITE" in script
    assert "START_CAT" in script
    assert 'R1LITE_KILL_CMD="${R1LITE_KILL_CMD:-bash ~/workspace/robot/kill_robot.sh}"' in script
    assert 'CAT_KILL_CMD="${CAT_KILL_CMD:-bash ~/kill_tele.sh}"' in script
    assert "EVA_CAT_GATE_FILE" not in script
    assert "cat_gate_off" not in script
    assert 'run_remote_kill "${R1LITE_HOST}" "${R1LITE_KILL_CMD}"' in script
    assert 'run_remote_kill "${CAT_HOST}" "${CAT_KILL_CMD}"' in script
    assert "pgrep -f" not in script
    assert "kill -KILL" not in script


def test_run_fake_node_forces_local_ros2_discovery():
    script = (R1_LITE_DIR / "run_fake_node.sh").read_text()

    assert "source .venv/bin/activate" in script
    assert 'unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY' in script
    assert "unset FASTRTPS_DEFAULT_PROFILES_FILE ROS_DISCOVERY_SERVER RMW_IMPLEMENTATION" in script
    assert "export ROS_LOCALHOST_ONLY=1" in script
    assert "python examples/hardware/r1_lite/fake_node.py" in script
    assert "--role cameras" in script
    assert "--role robot" in script
    assert "wait -n" in script
    assert '--rate "${PUBLISH_RATE:-30}"' in script
    assert "--obs-endpoint" not in script
    assert "--action-endpoint" not in script


def test_run_hardware_keeps_real_r1lite_fastdds_environment_external():
    script = (R1_LITE_DIR / "run_hardware.sh").read_text()

    assert "ROS_LOCALHOST_ONLY=1" not in script
    assert "ROS_DISCOVERY_SERVER" not in script
    assert "RMW_IMPLEMENTATION" not in script


def test_readme_documents_only_current_r1lite_helpers():
    readme = (R1_LITE_DIR / "README.md").read_text()

    assert "run_hardware.sh" in readme
    assert "kill_hardware.sh" in readme
    assert "run_fake_node.sh" in readme
    assert "cat_operator_button.py" in readme
    assert "CAT gate" not in readme
    assert "run_hardware.sh gate on" not in readme
    assert "/tmp/eva_collect_gate" not in readme
    assert "/eva/hil/input_joint_state_arm_left" in readme
    assert "/motion_target/target_joint_state_arm_left" in readme
    assert "net.core.rmem_max=16777216" in readme
    assert "<receiveBufferSize>16777216</receiveBufferSize>" in readme
    assert "ss -u -a -m -p" in readme
    for removed in REMOVED_HELPERS:
        assert removed not in readme
