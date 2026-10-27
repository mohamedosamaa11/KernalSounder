#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "${ROOT}"

if [[ -f "${ROOT}/venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/venv/bin/activate"
fi

exec python3 "${ROOT}/kernal_sounder.py"
