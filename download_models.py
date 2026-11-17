"""
Download the small offline models KernalSounder uses.

    python download_models.py            # wake word + whisper tiny.en
    python download_models.py --whisper base.en

Wake word: openWakeWord v0.5.1 release assets (about 4 MB).
Speech-to-text: faster-whisper CTranslate2 weights (tiny.en is about 75 MB).
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

from config import OWW_DIR, OWW_FEATURE_MODELS, OWW_RELEASE_URL, WAKE_MODEL, WHISPER_DIR, WHISPER_MODEL

PRETRAINED_WAKE_MODELS = (
    "hey_jarvis_v0.1.onnx",
    "alexa_v0.1.onnx",
    "hey_mycroft_v0.1.onnx",
)


def _fetch(url: str, dest: Path) -> None:
    if dest.is_file() and dest.stat().st_size > 0:
        print(f"  ok      {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  fetch   {dest.name}")
    with urllib.request.urlopen(url, timeout=60) as response, open(tmp, "wb") as fh:
        while True:
            block = response.read(1 << 16)
            if not block:
                break
            fh.write(block)
    tmp.replace(dest)


def download_wake_models(extra: str) -> None:
    print("Wake-word models ->", OWW_DIR)
    names = list(OWW_FEATURE_MODELS) + list(PRETRAINED_WAKE_MODELS)
    if extra.endswith(".onnx") and "/" not in extra and "\\" not in extra and extra not in names:
        names.append(extra)
    for name in names:
        try:
            _fetch(OWW_RELEASE_URL + name, OWW_DIR / name)
        except Exception as exc:
            print(f"  failed  {name}: {exc}")


def download_whisper(model: str) -> None:
    target = WHISPER_DIR / model
    print("Whisper model ->", target)
    if (target / "model.bin").is_file():
        print(f"  ok      {model}")
        return
    from faster_whisper.utils import download_model

    download_model(model, output_dir=str(target))
    print(f"  ok      {model}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--whisper", default=WHISPER_MODEL, help="tiny.en (default), base.en, small.en ...")
    parser.add_argument("--skip-whisper", action="store_true")
    args = parser.parse_args()

    download_wake_models(WAKE_MODEL)
    if not args.skip_whisper:
        try:
            download_whisper(args.whisper)
        except Exception as exc:
            print(f"  failed  whisper {args.whisper}: {exc}")
            return 1
    print("Models ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
