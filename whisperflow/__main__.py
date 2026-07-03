"""Command-line entry point.

    python -m whisperflow            # start dictation
    python -m whisperflow doctor     # health-check CUDA / audio / Ollama
    python -m whisperflow devices    # list audio input devices
    python -m whisperflow --config other.yaml
"""

from __future__ import annotations

import argparse
import sys

from .config import load_config


def _cmd_devices() -> int:
    from .audio import list_devices

    print(list_devices())
    return 0


def _cmd_doctor(config_path: str | None) -> int:
    config = load_config(config_path)
    ok = True

    print("whisperflow-local doctor")
    print("-" * 40)

    # 1. Audio
    try:
        import sounddevice as sd

        default_in = sd.query_devices(kind="input")
        print(f"[ok] microphone: {default_in['name']}")
    except Exception as exc:
        ok = False
        print(f"[!!] no usable input device: {exc}")

    # 2. GPU / CUDA via CTranslate2 (what faster-whisper actually uses)
    try:
        import ctranslate2

        cuda_count = ctranslate2.get_cuda_device_count()
        if cuda_count > 0:
            print(f"[ok] CUDA devices visible to CTranslate2: {cuda_count}")
        else:
            ok = False
            print("[!!] CTranslate2 sees 0 CUDA devices — transcription will run on CPU.")
            print("     Install: pip install nvidia-cublas-cu12 nvidia-cudnn-cu12")
    except Exception as exc:
        ok = False
        print(f"[!!] could not query CUDA via CTranslate2: {exc}")

    # 3. faster-whisper import
    try:
        import faster_whisper  # noqa: F401

        print("[ok] faster-whisper importable")
    except Exception as exc:
        ok = False
        print(f"[!!] faster-whisper not importable: {exc}")

    # 4. Ollama (only if cleanup enabled)
    if config["cleanup"]["enabled"]:
        from .cleanup import Cleaner

        cleaner = Cleaner(config["cleanup"])
        if cleaner.is_available():
            print(f"[ok] Ollama reachable at {cleaner.host} (model: {cleaner.model})")
        else:
            print(f"[!!] Ollama not reachable at {cleaner.host} — cleanup will be skipped.")
            print("     Start it with `ollama serve` and `ollama pull " f"{cleaner.model}`.")
    else:
        print("[--] cleanup disabled in config; skipping Ollama check")

    print("-" * 40)
    print("All good — run `python -m whisperflow` to start." if ok else "Some checks failed (see above).")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="whisperflow", description="Local Wispr Flow clone")
    parser.add_argument("command", nargs="?", default="run", choices=["run", "doctor", "devices"])
    parser.add_argument("--config", default="config.yaml", help="path to config YAML")
    args = parser.parse_args(argv)

    if args.command == "devices":
        return _cmd_devices()
    if args.command == "doctor":
        return _cmd_doctor(args.config)

    from .app import main as run_app

    run_app(args.config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
