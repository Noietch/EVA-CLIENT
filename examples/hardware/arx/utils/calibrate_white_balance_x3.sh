#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../.."
source .venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}:$PWD/src"

CAM_HIGH_SELECTOR="${CAM_HIGH_SELECTOR:-CP0HC530000Z}"
CAM_LEFT_WRIST_SELECTOR="${CAM_LEFT_WRIST_SELECTOR:-CV2L360000CL}"
CAM_RIGHT_WRIST_SELECTOR="${CAM_RIGHT_WRIST_SELECTOR:-CV2R1610003Z}"

ORBBEC_RESOLUTION="${ORBBEC_RESOLUTION:-640x480}"
ORBBEC_FPS="${ORBBEC_FPS:-30}"
ORBBEC_COLOR_FORMAT="${ORBBEC_COLOR_FORMAT:-MJPG}"
ORBBEC_TIMEOUT_MS="${ORBBEC_TIMEOUT_MS:-1000}"
ORBBEC_WB_OUTPUT="${ORBBEC_WB_OUTPUT:-examples/hardware/arx/utils/white_balance.yaml}"
ORBBEC_WB_IMAGE_DIR="${ORBBEC_WB_IMAGE_DIR:-work_dirs/arx_orbbec_white_balance/$(date +%Y%m%d_%H%M%S)}"
ORBBEC_WB_CAMERA="${ORBBEC_WB_CAMERA:-}"

need_camera=1
for arg in "$@"; do
  case "${arg}" in
    -h|--help|--list)
      need_camera=0
      ;;
  esac
done

if [[ "${#}" -gt 0 ]]; then
  case "$1" in
    1|2|3|all)
      ORBBEC_WB_CAMERA="$1"
      shift
      ;;
  esac
fi

if [[ "${need_camera}" == "1" && -z "${ORBBEC_WB_CAMERA}" &&
      "${ORBBEC_WB_NO_PROMPT:-0}" != "1" && -t 0 ]]; then
  echo "[arx-wb] Select camera: 1=cam_high, 2=cam_left_wrist, 3=cam_right_wrist"
  read -r -p "[arx-wb] Press Enter to calibrate all cameras: " ORBBEC_WB_CAMERA
fi

selected_label="all"
selected_disabled=()
if [[ "${need_camera}" == "1" ]]; then
  case "${ORBBEC_WB_CAMERA}" in
    ""|all)
      ;;
    1)
      selected_label="cam_high"
      selected_disabled=(--disabled-camera cam_left_wrist --disabled-camera cam_right_wrist)
      ;;
    2)
      selected_label="cam_left_wrist"
      selected_disabled=(--disabled-camera cam_high --disabled-camera cam_right_wrist)
      ;;
    3)
      selected_label="cam_right_wrist"
      selected_disabled=(--disabled-camera cam_high --disabled-camera cam_left_wrist)
      ;;
    *)
      echo "[arx-wb] ERROR: ORBBEC_WB_CAMERA must be empty, all, 1, 2, or 3" >&2
      exit 1
      ;;
  esac
fi

if [[ "${need_camera}" == "1" ]] &&
   pgrep -af "examples/hardware/arx/node.py" >/dev/null 2>&1; then
  echo "[arx-wb] WARN: examples/hardware/arx/node.py is running and may occupy Orbbec cameras." >&2
  echo "[arx-wb] WARN: stop run_hardware.sh / node.py before calibration." >&2
fi

args=(
  --orbbec-camera "cam_high=${CAM_HIGH_SELECTOR}"
  --orbbec-camera "cam_left_wrist=${CAM_LEFT_WRIST_SELECTOR}"
  --orbbec-camera "cam_right_wrist=${CAM_RIGHT_WRIST_SELECTOR}"
  --orbbec-resolution "${ORBBEC_RESOLUTION}"
  --orbbec-fps "${ORBBEC_FPS}"
  --orbbec-color-format "${ORBBEC_COLOR_FORMAT}"
  --orbbec-timeout-ms "${ORBBEC_TIMEOUT_MS}"
  --output "${ORBBEC_WB_OUTPUT}"
  --image-dir "${ORBBEC_WB_IMAGE_DIR}"
  "${selected_disabled[@]}"
)

if [[ -n "${EVA_CONFIG:-}" ]]; then
  args+=(--eva-config "${EVA_CONFIG}")
fi

if [[ "${ORBBEC_WB_NO_PROMPT:-0}" == "1" ]]; then
  args+=(--no-prompt)
fi

if [[ "${need_camera}" == "1" ]]; then
  echo "[arx-wb] selected camera: ${selected_label}"
  echo "[arx-wb] calibration images: ${ORBBEC_WB_IMAGE_DIR}"
  echo "[arx-wb] white-balance yaml: ${ORBBEC_WB_OUTPUT}"
fi

python examples/hardware/arx/utils/calibrate_white_balance.py "${args[@]}" "$@"
