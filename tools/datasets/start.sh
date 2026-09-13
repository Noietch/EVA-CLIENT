#!/usr/bin/env bash
# Start the EVA collection dataset browser in the foreground.
#
# Usage: tools/datasets/start.sh [additional tools.datasets.app arguments]
# Inputs: EVA_DATA_ROOT (default: datasets/data_collection), EVA_DATASET_HOST,
# EVA_DATASET_PORT, EVA_DATASET_READ_ONLY (0 or 1), and EVA_DATASET_PYTHON
# (default: the repository .venv).
# Output: a Flask service listening on the configured host and port.
# Side effects: the application may create missing catalog directories and
# generated image previews below the selected data root.
# Stages: resolve repository paths, validate Python, then replace this process
# with the dataset application.

set -euo pipefail

repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
data_root="${EVA_DATA_ROOT:-${repository_root}/datasets/data_collection}"
python_bin="${EVA_DATASET_PYTHON:-${repository_root}/.venv/bin/python}"
host="${EVA_DATASET_HOST:-0.0.0.0}"
port="${EVA_DATASET_PORT:-8418}"
read_only="${EVA_DATASET_READ_ONLY:-0}"

if [[ ! -x "${python_bin}" ]]; then
  echo "Dataset Python is not executable: ${python_bin}" >&2
  exit 1
fi

command=(
  "${python_bin}" -m tools.datasets.app
  --plans-root "${data_root}/task_sets"
  --assets-root "${data_root}/assets"
  --collection-root "${data_root}"
  --host "${host}"
  --port "${port}"
)

case "${read_only}" in
  0) ;;
  1) command+=(--read-only) ;;
  *)
    echo "EVA_DATASET_READ_ONLY must be 0 or 1, got: ${read_only}" >&2
    exit 1
    ;;
esac

exec "${command[@]}" "$@"
