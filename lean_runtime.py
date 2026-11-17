"""
KernalSounder — keep native libraries small. Import this before numpy,
onnxruntime or faster-whisper.

Measured on a 16-core laptop (Windows 11):
  * numpy's OpenBLAS reserved ~480 MB and started one thread per core,
    though the app only does tiny matrix products         -> 1 thread
  * CTranslate2's Intel MKL backend reserved ~650 MB more  -> CT2_USE_MKL=0
    (same transcripts, slightly faster for whisper tiny)
  * OpenMP worker threads spin-waited after each call,
    burning ~30 % of a core while idle                     -> passive wait
Together: private memory 1.4 GB -> ~0.3 GB, idle CPU ~30 % -> ~0 %.
Values already set in the environment are left alone.
"""

import os

_LEAN_ENV = {
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "MKL_DISABLE_FAST_MM": "1",
    "CT2_USE_MKL": "0",
    "OMP_WAIT_POLICY": "PASSIVE",
    "KMP_BLOCKTIME": "0",
    "TOKENIZERS_PARALLELISM": "false",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}

for _key, _value in _LEAN_ENV.items():
    os.environ.setdefault(_key, _value)
