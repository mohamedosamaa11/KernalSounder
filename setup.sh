#!/usr/bin/env bash
# KernalSounder — Ubuntu / Debian setup (works on a 4 GB RAM VM).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
echo "==> KernalSounder setup in ${ROOT}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This script is for Linux. On Windows run setup.bat instead."
  exit 1
fi

sudo apt-get update
sudo apt-get install -y \
  python3 python3-pip python3-venv python3-tk \
  libportaudio2 espeak-ng libespeak-ng1 mbrola mbrola-us1 \
  flac xdg-utils

cd "${ROOT}"
[[ -d venv ]] || python3 -m venv venv
# shellcheck disable=SC1091
source "${ROOT}/venv/bin/activate"
python -m pip install --upgrade pip
pip install -r requirements.txt
python download_models.py

mkdir -p "${HOME}/KernalSounderWorkspace"
chmod +x "${ROOT}/run.sh" || true

echo
echo "Setup complete."
echo "  Start:              ./run.sh            (say \"Hey Nova\")"
echo "  Start with console: ./run.sh --dashboard"
echo "  Start at login:     ./run.sh --install-autostart"
