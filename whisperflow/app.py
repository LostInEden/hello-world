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
import time
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
        released_at = time.perf_counter()
        audio = self.recorder.stop()
        seconds = self.recorder.duration(audio)
        if seconds < self._min_seconds:
            print("⏹️  Too short, ignored.", flush=True)
            return
        print(f"⏹️  Captured {seconds:.1f}s, processing...", flush=True)
        threading.Thread(target=self._process, args=(audio, released_at), daemon=True).start()

    # --- heavy lifting (worker thread) ---

    def _process(self, audio, released_at: float) -> None:  # noqa: ANN001
        with self._model_lock:
            raw = self.transcriber.transcribe(audio)
            stt_done = time.perf_counter()
            if not raw:
                print("🤷  No speech detected.", flush=True)
                return
            print(f"    raw: {raw}", flush=True)

            if self.cleaner.enabled:
                final = self.cleaner.clean(raw)
            else:
                final = raw
            clean_done = time.perf_counter()

        print(f"⌨️   Inserting: {final}", flush=True)
        self._inject_safely(final)
        done = time.perf_counter()

        print(
            f"⏱️   {done - released_at:.2f}s release-to-text "
            f"(stt {stt_done - released_at:.2f}s, "
            f"cleanup {clean_done - stt_done:.2f}s, "
            f"inject {done - clean_done:.2f}s)",
            flush=True,
        )

    def _inject_safely(self, text: str) -> None:
        """Inject text once the hotkey is physically released, with the
        listener suspended so our own synthetic keystrokes don't re-trigger it.

        Without the wait, a fast transcription can race the user's fingers:
        Ctrl+V lands while Alt is still held and the target app sees Ctrl+Alt+V.
        """
        deadline = time.time() + 2.0
        while self.listener.target_down() and time.time() < deadline:
            time.sleep(0.02)

        self.listener.suspend()
        try:
            self.injector.inject(text)
        finally:
            self.listener.resume()

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

        if self.config["transcription"].get("warmup", True):
            print("[stt] warming up...", flush=True)
            with self._model_lock:
                self.transcriber.warmup()
            print("[stt] ready.", flush=True)

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
