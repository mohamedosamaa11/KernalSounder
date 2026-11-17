# Models

| Folder | What | In git? |
| --- | --- | --- |
| `custom/hey_nova.npz` | the default "Hey Nova" wake-word head trained by `train_wake_word.py` | yes (~400 KB) |
| `custom/hi_sounder.npz` | the earlier "Hi Sounder" head (`--wake-model hi_sounder.npz`) | yes (~400 KB) |
| `custom/*_samples/` | your own voice recordings for training | **no** (private) |
| `openwakeword/` | melspectrogram + embedding feature models, optional pretrained wake words | no, `download_models.py` |
| `whisper/tiny.en/` | faster-whisper speech-to-text weights (~75 MB) | no, `download_models.py` |
| `piper/` | multi-speaker TTS voice, only used for training (~75 MB) | no, downloaded by the trainer |

Run `python download_models.py` once after cloning (the setup scripts do it).
