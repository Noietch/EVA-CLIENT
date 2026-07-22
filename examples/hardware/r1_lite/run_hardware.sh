#!/usr/bin/env bash
set -euo pipefail

R1LITE_HOST="${R1LITE_HOST:-r1lite@192.168.12.12}"
CAT_HOST="${CAT_HOST:-cat@192.168.12.13}"
R1LITE_SESSION="${R1LITE_SESSION:-eva_r1lite_robot}"
CAT_SESSION="${CAT_SESSION:-eva_r1lite_teleop}"
START_R1LITE="${START_R1LITE:-1}"
START_CAT="${START_CAT:-1}"
R1LITE_BODY_BOOTSTRAP_REMOTE="${R1LITE_BODY_BOOTSTRAP_REMOTE:-/tmp/eva_r1lite_body_no_teleop_bootstrap}"
R1LITE_BODY_SESSION_SOURCE="${R1LITE_BODY_SESSION_SOURCE:-/home/r1lite/galaxea/install/startup_config/share/startup_config/sessions.d/ATCStandard/R1LITEBody.d}"
R1LITE_BODY_SESSION_REMOTE="${R1LITE_BODY_SESSION_REMOTE:-/tmp/eva_r1lite_body_no_teleop.d}"
R1LITE_START_CMD="${R1LITE_START_CMD:-${R1LITE_BODY_BOOTSTRAP_REMOTE}}"
CAT_SETUP_CMD="${CAT_SETUP_CMD:-source /opt/ros/humble/setup.bash && source /home/cat/galaxea/install/setup.bash}"
CAT_HIL_TELEOP_REMOTE="${CAT_HIL_TELEOP_REMOTE:-/tmp/eva_cat_hil_teleop_bootstrap}"
CAT_FASTRTPS_PROFILE="${CAT_FASTRTPS_PROFILE:-/tmp/eva_cat_fastdds_super_client.xml}"
CAT_ROS2_LOCAL_IP="${CAT_ROS2_LOCAL_IP:-192.168.12.13}"
R1LITE_DISCOVERY_SERVER_IP="${R1LITE_DISCOVERY_SERVER_IP:-192.168.12.12}"
R1LITE_DISCOVERY_SERVER_PORT="${R1LITE_DISCOVERY_SERVER_PORT:-11811}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAT_OPERATOR_BUTTON_LOCAL="${CAT_OPERATOR_BUTTON_LOCAL:-${SCRIPT_DIR}/cat_operator_button.py}"
CAT_OPERATOR_BUTTON_REMOTE="${CAT_OPERATOR_BUTTON_REMOTE:-/tmp/eva_cat_operator_button.py}"
EVA_OPERATOR_ACTION_TOPIC="${EVA_OPERATOR_ACTION_TOPIC:-/eva/operator_action}"
CAT_OPERATOR_BUTTON="${CAT_OPERATOR_BUTTON:-1}"

install_remote_script() {
  local host="$1"
  local local_path="$2"
  local remote_path="$3"

  scp -q "${local_path}" "${host}:${remote_path}"
  ssh "${host}" chmod 755 "${remote_path}"
}

write_r1lite_body_bootstrap_script() {
  cat <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

R1LITE_BODY_SESSION_SOURCE="${R1LITE_BODY_SESSION_SOURCE:-/home/r1lite/galaxea/install/startup_config/share/startup_config/sessions.d/ATCStandard/R1LITEBody.d}"
R1LITE_BODY_SESSION_REMOTE="${R1LITE_BODY_SESSION_REMOTE:-/tmp/eva_r1lite_body_no_teleop.d}"
R1LITE_START_DIR="${R1LITE_START_DIR:-/home/r1lite/galaxea/install/startup_config/share/startup_config/script/boot}"

rm -rf "${R1LITE_BODY_SESSION_REMOTE}"
mkdir -p "${R1LITE_BODY_SESSION_REMOTE}"

for yaml in hdas.yaml mobiman.yaml ros2_discovery.yaml system.yaml tools.yaml; do
  cp "${R1LITE_BODY_SESSION_SOURCE}/${yaml}" "${R1LITE_BODY_SESSION_REMOTE}/${yaml}"
done

export START_DIR="${R1LITE_START_DIR}"

for yaml in hdas.yaml mobiman.yaml ros2_discovery.yaml system.yaml tools.yaml; do
  tmuxp load "${R1LITE_BODY_SESSION_REMOTE}/${yaml}" -d
done

printf '[eva-r1lite] started body sessions without vendor tabletop teleop\n'
while true; do
  sleep 3600
done
EOF
}

install_r1lite_body_bootstrap_script() {
  local host="$1"
  local remote_path="$2"

  write_r1lite_body_bootstrap_script | ssh "${host}" "cat > '${remote_path}' && chmod 755 '${remote_path}'"
}

write_cat_fastdds_profile() {
  local cat_ros2_local_ip="$1"
  local server_ip="$2"
  local server_port="$3"

  cat <<EOF
<?xml version="1.0" encoding="UTF-8" ?>
<dds>
  <profiles xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles">
    <transport_descriptors>
      <transport_descriptor>
        <transport_id>udp_transport</transport_id>
        <type>UDPv4</type>
        <interfaceWhiteList>
          <address>${cat_ros2_local_ip}</address>
        </interfaceWhiteList>
      </transport_descriptor>
    </transport_descriptors>
    <participant profile_name="super_client_profile" is_default_profile="true">
      <rtps>
        <userTransports>
          <transport_id>udp_transport</transport_id>
        </userTransports>
        <useBuiltinTransports>false</useBuiltinTransports>
        <builtin>
          <discovery_config>
            <discoveryProtocol>SUPER_CLIENT</discoveryProtocol>
            <discoveryServersList>
              <RemoteServer prefix="44.53.00.5f.45.50.52.4f.53.49.4d.41">
                <metatrafficUnicastLocatorList>
                  <locator>
                    <udpv4>
                      <address>${server_ip}</address>
                      <port>${server_port}</port>
                    </udpv4>
                  </locator>
                </metatrafficUnicastLocatorList>
              </RemoteServer>
            </discoveryServersList>
          </discovery_config>
        </builtin>
      </rtps>
    </participant>
  </profiles>
</dds>
EOF
}

install_cat_fastdds_profile() {
  local host="$1"
  local remote_path="$2"

  write_cat_fastdds_profile \
    "${CAT_ROS2_LOCAL_IP}" \
    "${R1LITE_DISCOVERY_SERVER_IP}" \
    "${R1LITE_DISCOVERY_SERVER_PORT}" \
    | ssh "${host}" "cat > '${remote_path}' && chmod 644 '${remote_path}'"
}

write_cat_hil_teleop_script() {
  cat <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

CAT_FASTRTPS_PROFILE="${CAT_FASTRTPS_PROFILE:-/tmp/eva_cat_fastdds_super_client.xml}"
CAT_GELLO_CMD="${CAT_GELLO_CMD:-python3 /home/cat/galaxea/install/mobiman/lib/DynamixelTeleop/GelloTeleop/gello_teleop.py}"
CAT_JOYCON_CMD="${CAT_JOYCON_CMD:-ros2 launch joycon_evdev_publisher joy_evdev_launch.py}"
CAT_OPERATOR_BUTTON_PATH="${CAT_OPERATOR_BUTTON_PATH:-/tmp/eva_cat_operator_button.py}"
EVA_OPERATOR_ACTION_TOPIC="${EVA_OPERATOR_ACTION_TOPIC:-/eva/operator_action}"
CAT_OPERATOR_BUTTON="${CAT_OPERATOR_BUTTON:-1}"

export FASTRTPS_DEFAULT_PROFILES_FILE="${CAT_FASTRTPS_PROFILE}"

bash -lc "${CAT_JOYCON_CMD}" &
joycon_pid="$!"

bash -lc "${CAT_GELLO_CMD} --ros-args -r /tabletop/hdas/feedback_arm_left:=/eva/hil/input_joint_state_arm_left -r /tabletop/hdas/feedback_arm_right:=/eva/hil/input_joint_state_arm_right" &
gello_pid="$!"

if [[ "${CAT_OPERATOR_BUTTON}" == "1" ]]; then
  python3 "${CAT_OPERATOR_BUTTON_PATH}" --topic "${EVA_OPERATOR_ACTION_TOPIC}" &
  button_pid="$!"
  wait -n "${joycon_pid}" "${gello_pid}" "${button_pid}"
else
  wait -n "${joycon_pid}" "${gello_pid}"
fi
EOF
}

install_cat_hil_teleop_script() {
  local host="$1"
  local remote_path="$2"

  write_cat_hil_teleop_script | ssh "${host}" "cat > '${remote_path}' && chmod 755 '${remote_path}'"
}

start_remote_tmux() {
  local host="$1"
  local session="$2"
  local command="$3"
  local command_b64

  command_b64="$(printf '%s' "${command}" | base64 | tr -d '\n')"
  ssh "${host}" bash -s -- "${session}" "${host}" "${command_b64}" <<'EOF'
set -euo pipefail
session="$1"
host="$2"
command_b64="$3"
command="$(printf '%s' "${command_b64}" | base64 -d)"

if tmux has-session -t "${session}" 2>/dev/null; then
  echo "[r1lite-run] ${host}:${session} already running"
else
  tmux new-session -d -s "${session}" "${command}"
  echo "[r1lite-run] started ${host}:${session}"
fi
EOF
}

if [[ "${START_R1LITE}" == "1" ]]; then
  install_r1lite_body_bootstrap_script "${R1LITE_HOST}" "${R1LITE_BODY_BOOTSTRAP_REMOTE}"
  start_remote_tmux "${R1LITE_HOST}" "${R1LITE_SESSION}" "R1LITE_BODY_SESSION_SOURCE='${R1LITE_BODY_SESSION_SOURCE}' R1LITE_BODY_SESSION_REMOTE='${R1LITE_BODY_SESSION_REMOTE}' ${R1LITE_START_CMD}"
fi

if [[ "${START_CAT}" == "1" ]]; then
  install_remote_script "${CAT_HOST}" "${CAT_OPERATOR_BUTTON_LOCAL}" "${CAT_OPERATOR_BUTTON_REMOTE}"
  install_cat_fastdds_profile "${CAT_HOST}" "${CAT_FASTRTPS_PROFILE}"
  install_cat_hil_teleop_script "${CAT_HOST}" "${CAT_HIL_TELEOP_REMOTE}"
  start_remote_tmux "${CAT_HOST}" "${CAT_SESSION}" "${CAT_SETUP_CMD}; CAT_FASTRTPS_PROFILE='${CAT_FASTRTPS_PROFILE}' CAT_OPERATOR_BUTTON_PATH='${CAT_OPERATOR_BUTTON_REMOTE}' EVA_OPERATOR_ACTION_TOPIC='${EVA_OPERATOR_ACTION_TOPIC}' CAT_OPERATOR_BUTTON='${CAT_OPERATOR_BUTTON}' '${CAT_HIL_TELEOP_REMOTE}'"
fi
