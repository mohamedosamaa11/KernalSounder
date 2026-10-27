#!/usr/bin/env bash
# KernalSounder — Ubuntu setup for a 4 GB RAM VM.
# Google Web Speech needs internet, PyAudio, and the flac encoder.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

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
  flac \
  espeak-ng \
  libespeak-ng1 \
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
pip install SpeechRecognition

mkdir -p "${HOME}/KernalSounderWorkspace"
chmod +x "${ROOT}/run.sh" || true

echo
echo "Setup complete."
echo "Start with:  ./run.sh"
echo "Workspace:   ${HOME}/KernalSounderWorkspace"
echo "STT:         SpeechRecognition + Google Web Speech API (needs internet)"
