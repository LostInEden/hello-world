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

from typing import Optional

from .audio import Recorder
from .cleanup import Cleaner
from .config import load_config, save_override
from .hotkey import HotkeyListener
from .inject import Injector
from .transcribe import Transcriber


class App:
    def __init__(self, config: Dict[str, Any], config_path: Optional[str] = None):
        self.config = config
        self.config_path = config_path
        self.paused = False
        self.recorder = Recorder(
            sample_rate=config["audio"]["sample_rate"],
            device=config["audio"]["device"],
        )
        self.transcriber = Transcriber(config["transcription"])
        self.cleaner = Cleaner(config["cleanup"])
        self.injector = Injector(config["injection"])

        self._model_lock = threading.Lock()
        self._min_seconds = 0.3  # ignore accidental taps shorter than this
        self._inflight = 0
        self._inflight_lock = threading.Lock()

        self.overlay = None
        if config.get("overlay", {}).get("enabled", True):
            try:
                from .overlay import Overlay

                self.overlay = Overlay(config["overlay"], level_provider=self.recorder.level)
            except Exception as exc:
                print(f"[overlay] disabled ({exc})", flush=True)

        self.listener = HotkeyListener(
            combo=config["hotkey"]["combo"],
            on_activate=self._on_start,
            on_deactivate=self._on_stop,
            mode=config["hotkey"]["mode"],
        )

    def _set_overlay(self, state: str) -> None:
        if self.overlay is not None:
            self.overlay.set_state(state)

    # --- tray-facing controls (called from the tray thread) ---

    def pause(self, paused: bool) -> None:
        """Stop reacting to the hotkey without shutting anything down."""
        self.paused = paused
        if paused:
            self.listener.suspend()
            self.listener.force_release()  # finish any in-progress recording
            print("⏸️   Dictation paused.", flush=True)
        else:
            self.listener.resume()
            print("▶️   Dictation resumed.", flush=True)

    def list_cleanup_models(self) -> list:
        return self.cleaner.list_models()

    def set_cleanup_model(self, model: Optional[str]) -> None:
        """Switch the cleanup model (None disables cleanup); persists to config."""
        if model:
            self.cleaner.model = model
            self.cleaner.enabled = True
            print(f"🤖  Cleanup model: {model}", flush=True)
        else:
            self.cleaner.enabled = False
            print("🤖  Cleanup disabled (raw transcripts).", flush=True)

        updates: Dict[str, Any] = {"cleanup": {"enabled": self.cleaner.enabled}}
        if model:
            updates["cleanup"]["model"] = model
        try:
            save_override(self.config_path, updates)
        except Exception as exc:
            print(f"[config] could not persist choice ({exc})", flush=True)

    def shutdown(self) -> None:
        """Stop the app from another thread (tray Quit)."""
        if self.overlay is not None:
            self.overlay.close()
        else:
            self.listener.stop()

    # --- hotkey callbacks (run on the listener thread; keep them quick) ---

    def _on_start(self) -> None:
        print("\n🎙️  Recording... (release hotkey to transcribe)", flush=True)
        self._set_overlay("recording")
        self.recorder.start()

    def _on_stop(self) -> None:
        released_at = time.perf_counter()
        audio = self.recorder.stop()
        seconds = self.recorder.duration(audio)
        if seconds < self._min_seconds:
            print("⏹️  Too short, ignored.", flush=True)
            self._set_overlay("hidden")
            return
        print(f"⏹️  Captured {seconds:.1f}s, processing...", flush=True)
        self._set_overlay("processing")
        with self._inflight_lock:
            self._inflight += 1
        threading.Thread(target=self._process, args=(audio, released_at), daemon=True).start()

    # --- heavy lifting (worker thread) ---

    def _process(self, audio, released_at: float) -> None:  # noqa: ANN001
        try:
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
        finally:
            # Hide the indicator only when no other utterance is recording
            # (state moved on) or still being processed (inflight > 0).
            with self._inflight_lock:
                self._inflight -= 1
                idle = self._inflight == 0
            if idle and self.overlay is not None and self.overlay._state == "processing":
                self._set_overlay("hidden")

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
            if not self.paused:  # don't undo a tray-menu pause
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

        tray = None
        if self.config.get("tray", {}).get("enabled", True):
            try:
                from .tray import Tray

                tray = Tray(self, hotkey_label=combo)
                tray.start()
            except Exception as exc:
                print(f"[tray] disabled ({exc})", flush=True)

        if self.cleaner.enabled:
            # Load the cleanup model into Ollama in parallel with the Whisper
            # warm-up, so the first dictation doesn't hit a cold LLM.
            threading.Thread(target=self.cleaner.warmup, daemon=True, name="ollama-warmup").start()

        if self.config["transcription"].get("warmup", True):
            print("[stt] warming up...", flush=True)
            with self._model_lock:
                self.transcriber.warmup()
            print("[stt] ready.", flush=True)

        self.listener.start()
        try:
            if self.overlay is not None:
                # tkinter must own the main thread; hotkey + workers run on
                # their own threads and poke the overlay via set_state().
                try:
                    self.overlay.run()
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    print(f"[overlay] failed ({exc}); continuing without it.", flush=True)
                    self.overlay = None
                    self.listener.join()
            else:
                self.listener.join()
        except KeyboardInterrupt:
            print("\nShutting down.", flush=True)
        finally:
            if tray is not None:
                tray.stop()
            if self.overlay is not None:
                self.overlay.close()
            self.listener.stop()


def main(config_path: str | None = "config.yaml") -> None:
    config = load_config(config_path)
    App(config, config_path=config_path).run()
