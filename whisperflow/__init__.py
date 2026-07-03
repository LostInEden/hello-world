"""whisperflow-local: a local, offline Wispr Flow clone for Windows + NVIDIA GPU.

Pipeline: global hotkey -> mic capture -> Silero VAD -> faster-whisper (CUDA)
-> optional local-LLM cleanup (Ollama) -> text injection into the focused app.
"""

__version__ = "0.1.0"
