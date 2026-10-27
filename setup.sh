#!/usr/bin/env bash
# KernalSounder — Ubuntu setup for a 4 GB RAM VM.
# Installs system packages, a Python venv, and the small Vosk model only.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
MODEL_NAME="vosk-model-small-en-us-0.15"
MODEL_ZIP="${MODEL_NAME}.zip"
MODEL_URL="https://alphacephei.com/vosk/models/${MODEL_ZIP}"
MODEL_DIR="${ROOT}/models/${MODEL_NAME}"

echo "==> KernalSounder setup"
echo "    project: ${ROOT}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This setup script is for Ubuntu/Linux. Copy the project into the VM and run it there."
  exit 1
fi

sudo apt-get update
sudo apt-get install -y \
  python3 \
  python3-pip \
  python3-venv \
  python3-tk \
  python3-dev \
  portaudio19-dev \
  espeak-ng \
  libespeak-ng1 \
  wget \
  unzip \
  procps \
  coreutils

cd "${ROOT}"
if [[ ! -d venv ]]; then
  python3 -m venv venv
fi
# shellcheck disable=SC1091
source "${ROOT}/venv/bin/activate"
python -m pip install --upgrade pip
pip install -r "${ROOT}/requirements.txt"

mkdir -p "${ROOT}/models"
if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "==> Downloading lightweight Vosk model (${MODEL_NAME})"
  cd "${ROOT}/models"
  wget -q --show-progress -O "${MODEL_ZIP}" "${MODEL_URL}"
  unzip -q -o "${MODEL_ZIP}"
  rm -f "${MODEL_ZIP}"
else
  echo "==> Vosk model already present"
fi

mkdir -p "${HOME}/KernalSounderWorkspace"
chmod +x "${ROOT}/run.sh" || true

echo
echo "Setup complete."
echo "Start with:  ./run.sh"
echo "Workspace:   ${HOME}/KernalSounderWorkspace"
echo "Model:       ${MODEL_DIR}"
