"""Application orchestrator: wires the hotkey, recorder, STT, cleanup, and injector.

Flow per utterance:
    hotkey down  -> start recording
    hotkey up    -> stop recording, then (on a worker thread) transcribe ->
                    clean up -> inject into the focused app.

Transcription runs off the hotkey-listener thread so key events keep flowing,
and a lock serializes access to the Whisper model.
"""

from __future__ import annotations

import threading
from typing import Any, Dict

from .audio import Recorder
from .cleanup import Cleaner
from .config import load_config
from .hotkey import HotkeyListener
from .inject import Injector
from .transcribe import Transcriber


class App:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.recorder = Recorder(
            sample_rate=config["audio"]["sample_rate"],
            device=config["audio"]["device"],
        )
        self.transcriber = Transcriber(config["transcription"])
        self.cleaner = Cleaner(config["cleanup"])
        self.injector = Injector(config["injection"])

        self._model_lock = threading.Lock()
        self._min_seconds = 0.3  # ignore accidental taps shorter than this

        self.listener = HotkeyListener(
            combo=config["hotkey"]["combo"],
            on_activate=self._on_start,
            on_deactivate=self._on_stop,
            mode=config["hotkey"]["mode"],
        )

    # --- hotkey callbacks (run on the listener thread; keep them quick) ---

    def _on_start(self) -> None:
        print("\n🎙️  Recording... (release hotkey to transcribe)", flush=True)
        self.recorder.start()

    def _on_stop(self) -> None:
        audio = self.recorder.stop()
        seconds = self.recorder.duration(audio)
        if seconds < self._min_seconds:
            print("⏹️  Too short, ignored.", flush=True)
            return
        print(f"⏹️  Captured {seconds:.1f}s, processing...", flush=True)
        threading.Thread(target=self._process, args=(audio,), daemon=True).start()

    # --- heavy lifting (worker thread) ---

    def _process(self, audio) -> None:  # noqa: ANN001
        with self._model_lock:
            print("📝  Transcribing...", flush=True)
            raw = self.transcriber.transcribe(audio)
            if not raw:
                print("🤷  No speech detected.", flush=True)
                return
            print(f"    raw: {raw}", flush=True)

            if self.cleaner.enabled:
                print("🤖  Cleaning up...", flush=True)
                final = self.cleaner.clean(raw)
            else:
                final = raw

        print(f"⌨️   Inserting: {final}", flush=True)
        self.injector.inject(final)

    # --- lifecycle ---

    def run(self) -> None:
        combo = self.config["hotkey"]["combo"]
        mode = self.config["hotkey"]["mode"]
        verb = "Hold" if mode == "push_to_talk" else "Press"
        print("=" * 60, flush=True)
        print("  whisperflow-local is running.", flush=True)
        print(f"  {verb} [{combo}] to dictate into the focused app.", flush=True)
        if self.cleaner.enabled and not self.cleaner.is_available():
            print(
                "  ⚠️  Cleanup is enabled but Ollama isn't responding — you'll get\n"
                "     the raw transcript until `ollama serve` is running.",
                flush=True,
            )
        print("  Press Ctrl+C here to quit.", flush=True)
        print("=" * 60, flush=True)

        self.listener.start()
        try:
            self.listener.join()
        except KeyboardInterrupt:
            print("\nShutting down.", flush=True)
        finally:
            self.listener.stop()


def main(config_path: str | None = "config.yaml") -> None:
    config = load_config(config_path)
    App(config).run()
