#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

REPOSITORY_ROOT="$PWD"
SDK_RELATIVE_DIR="examples/hardware/i2rt/SDK/i2rt"
DEFAULT_SDK_DIR="$REPOSITORY_ROOT/$SDK_RELATIVE_DIR"
SDK_DIR="${I2RT_SDK_DIR:-$DEFAULT_SDK_DIR}"
PROJECT_DIR="$PWD/examples/hardware/i2rt"
VENV_DIR="${I2RT_VENV_DIR:-$PROJECT_DIR/.venv}"
SDK_PATCHES=(
  "$PROJECT_DIR/patches/i2rt-sdk-safe-shutdown.patch"
  "$PROJECT_DIR/patches/i2rt-sdk-eva-visual.patch"
)
I2RT_REPOSITORY="${I2RT_REPOSITORY:-https://github.com/i2rt-robotics/i2rt.git}"
I2RT_REVISION="5d47b358bafb30c65e397f2ece506550a0db4594"
I2RT_RELEASE="v1.2.4"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it with:" >&2
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

for package in build-essential python3-dev "linux-headers-$(uname -r)"; do
  if ! dpkg-query -W -f='${Status}' "$package" 2>/dev/null | grep -q 'install ok installed'; then
    echo "Missing system package: $package" >&2
    echo "Install prerequisites with:" >&2
    echo "  sudo apt update && sudo apt install build-essential python3-dev linux-headers-$(uname -r)" >&2
    exit 1
  fi
done

if ! git -C "$SDK_DIR" rev-parse --git-dir >/dev/null 2>&1; then
  if [[ "$SDK_DIR" == "$DEFAULT_SDK_DIR" ]]; then
    git submodule update --init --recursive -- "$SDK_RELATIVE_DIR"
  else
    mkdir -p "$(dirname "$SDK_DIR")"
    git -c http.version=HTTP/1.1 clone --filter=blob:none "$I2RT_REPOSITORY" "$SDK_DIR"
  fi
fi

SDK_HEAD="$(git -C "$SDK_DIR" rev-parse HEAD)"
if [[ "$SDK_HEAD" != "$I2RT_REVISION" ]]; then
  echo "I2RT SDK is at $SDK_HEAD, expected latest stable $I2RT_RELEASE ($I2RT_REVISION)." >&2
  echo "Run: git submodule update --init --recursive -- $SDK_RELATIVE_DIR" >&2
  exit 1
fi
echo "I2RT SDK pinned: $I2RT_RELEASE ($I2RT_REVISION)"

for sdk_patch in "${SDK_PATCHES[@]}"; do
  if git -C "$SDK_DIR" apply --unidiff-zero --reverse --check "$sdk_patch" 2>/dev/null; then
    echo "I2RT SDK patch already applied: $(basename "$sdk_patch")"
  elif git -C "$SDK_DIR" apply --unidiff-zero --check "$sdk_patch"; then
    git -C "$SDK_DIR" apply --unidiff-zero "$sdk_patch"
    echo "Applied I2RT SDK patch: $(basename "$sdk_patch")"
  else
    echo "SDK patch $(basename "$sdk_patch") does not apply to $I2RT_RELEASE" >&2
    exit 1
  fi
done

UV_PROJECT_ENVIRONMENT="$VENV_DIR" uv sync --project "$PROJECT_DIR"

"$VENV_DIR/bin/python" - <<'PY'
import i2rt
import pyrealsense2 as rs
import zmq

print(f"i2rt installed: {i2rt.__file__}")
print(f"pyrealsense2 available: {rs.__file__}")
print(f"pyzmq available: {zmq.__file__}")
PY

echo
echo "I2RT SDK environment is ready: $VENV_DIR"
echo "Detected CAN interfaces:"
find /sys/class/net -maxdepth 1 -type l -name 'can*' -printf '  %f\n' 2>/dev/null || true
echo
echo "Optional boot-time CAN setup (system-wide):"
echo "  sudo sh $SDK_DIR/devices/install_devices.sh"
