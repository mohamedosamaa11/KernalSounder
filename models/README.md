# Models

KernalSounder no longer ships a local STT model. Speech-to-text uses
`SpeechRecognition` and Google's Web Speech API (`recognize_google()`).

The Ubuntu VM needs:

- Internet access
- A working microphone (VirtualBox: enable audio input)
- `flac` (installed by `setup.sh`) so audio can be sent to Google
