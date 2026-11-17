"""
KernalSounder — synthesize short clips with the speech engine the OS already has.

Windows: System.Speech (SAPI voices), driven by a fixed PowerShell script.
Linux:   espeak-ng (or espeak) with several voice variants.
Texts come only from the app's own phrase lists; nothing is passed to a shell.
Used by personalize.py to "replay" look-alike phrases while fine-tuning.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from config import IS_WINDOWS, SAMPLE_RATE

_PS_SCRIPT = r"""
param($listFile, $outDir)
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo 16000, ([System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen), ([System.Speech.AudioFormat.AudioChannel]::Mono)
$voices = @($s.GetInstalledVoices() | Where-Object { $_.Enabled -and $_.VoiceInfo.Culture.Name -like 'en*' })
if ($voices.Count -eq 0) { $voices = @($s.GetInstalledVoices()) }
foreach ($line in Get-Content -LiteralPath $listFile -Encoding UTF8) {
  $parts = $line.Split('|', 3)
  $v = $voices[[int]$parts[1] % $voices.Count]
  $s.SelectVoice($v.VoiceInfo.Name)
  $s.Rate = [int]$parts[2].Split('|')[0]
  $s.SetOutputToWaveFile((Join-Path $outDir ($parts[0] + '.wav')), $fmt)
  $s.Speak($parts[2].Split('|', 2)[1])
}
$s.SetOutputToNull()
"""

_ESPEAK_VOICES = ("en-us", "en-gb", "en-us+m3", "en-us+f2", "en-gb+m5", "en-us+f4", "en-gb+f3", "en-us+m7")


def _read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        rate, channels = w.getframerate(), w.getnchannels()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    if rate != SAMPLE_RATE:
        n = int(len(data) * SAMPLE_RATE / rate)
        data = np.interp(np.linspace(0, len(data) - 1, n), np.arange(len(data)), data)
    return np.asarray(data, dtype=np.int16)


def synthesize(jobs: Sequence[Tuple[str, int, int]], timeout: float = 240.0) -> List[np.ndarray]:
    """jobs: (text, voice_index, rate -10..10). Returns clips (missing ones skipped)."""
    clean = [(t.replace("|", " ").replace("\n", " ").strip(), int(v), max(-10, min(10, int(r))))
             for t, v, r in jobs if t.strip()]
    if not clean:
        return []
    with tempfile.TemporaryDirectory(prefix="ks_tts_") as tmp:
        out = Path(tmp)
        if IS_WINDOWS:
            script = out / "speak.ps1"
            script.write_text(_PS_SCRIPT, encoding="utf-8")
            listing = out / "jobs.txt"
            listing.write_text("\n".join(f"c{i:04d}|{v}|{r}|{t}" for i, (t, v, r) in enumerate(clean)),
                               encoding="utf-8")
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                            "-File", str(script), str(listing), str(out)],
                           capture_output=True, timeout=timeout, creationflags=flags)
        else:
            exe = shutil.which("espeak-ng") or shutil.which("espeak")
            if not exe:
                return []
            for i, (text, voice, rate) in enumerate(clean):
                subprocess.run([exe, "-v", _ESPEAK_VOICES[voice % len(_ESPEAK_VOICES)],
                                "-s", str(160 + rate * 12), "-w", str(out / f"c{i:04d}.wav"), text],
                               capture_output=True, timeout=30)
        clips = []
        for i in range(len(clean)):
            path = out / f"c{i:04d}.wav"
            if path.is_file():
                try:
                    clips.append(_read_wav(path))
                except Exception:
                    continue
        return clips
