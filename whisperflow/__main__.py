"""Command-line entry point.

    python -m whisperflow            # start dictation
    python -m whisperflow doctor     # health-check CUDA / audio / Ollama
    python -m whisperflow devices    # list audio input devices
    python -m whisperflow bench      # load the model, verify GPU use, measure latency
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
            # Seeing the device is not enough: inference also needs the cuBLAS
            # and cuDNN DLLs, which live in the pip nvidia-* packages on Windows.
            if sys.platform == "win32":
                import ctypes

                from . import transcribe  # noqa: F401 - registers pip CUDA DLL dirs

                for dll in ("cublas64_12.dll", "cudnn64_9.dll"):
                    try:
                        ctypes.WinDLL(dll)
                        print(f"[ok] {dll} loadable")
                    except OSError:
                        ok = False
                        print(f"[!!] {dll} not loadable — GPU inference will fail.")
                        print("     Install: pip install nvidia-cublas-cu12 nvidia-cudnn-cu12")
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


def _gpu_memory_mb() -> int | None:
    """Currently used VRAM on GPU 0 in MiB via nvidia-smi, or None."""
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip().splitlines()
        return int(out[0])
    except Exception:
        return None


def _cmd_bench(config_path: str | None, wav_path: str | None) -> int:
    """Load the model per config, confirm where it actually lives (VRAM delta),
    and measure transcription + cleanup latency. Pass --wav for real speech."""
    import time

    import numpy as np

    config = load_config(config_path)

    if wav_path:
        import wave

        with wave.open(wav_path) as w:
            if w.getsampwidth() != 2:
                print("--wav must be 16-bit PCM")
                return 1
            sr = w.getframerate()
            data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
            data = data.reshape(-1, w.getnchannels()).mean(axis=1) / 32768.0
        if sr != 16000:
            x = np.linspace(0, len(data) - 1, int(len(data) * 16000 / sr))
            data = np.interp(x, np.arange(len(data)), data)
        audio = data.astype(np.float32)
        print(f"benchmarking with {wav_path} ({len(audio)/16000:.1f}s of audio)")
    else:
        rng = np.random.default_rng(0)
        audio = (rng.standard_normal(16000 * 5) * 0.02).astype(np.float32)
        print("benchmarking with 5s of synthetic noise (pass --wav speech.wav for a realistic run)")

    vram_before = _gpu_memory_mb()

    from .transcribe import Transcriber

    stt_config = dict(config["transcription"])
    if not wav_path:
        stt_config["vad_filter"] = False  # VAD would strip pure noise and skip the decode

    t0 = time.perf_counter()
    try:
        transcriber = Transcriber(stt_config)
    except RuntimeError as exc:
        print(f"[!!] {exc}")
        return 1
    print(f"model load: {time.perf_counter() - t0:.1f}s")

    vram_after = _gpu_memory_mb()
    if vram_before is not None and vram_after is not None:
        delta = vram_after - vram_before
        print(f"VRAM in use: {vram_after} MiB (+{delta} MiB since before load)")
        if transcriber.device_used == "cuda" and delta < 100:
            print("[!!] device says cuda but VRAM barely moved — check the fallback warning above")
    elif transcriber.device_used == "cuda":
        print("[--] nvidia-smi not available; can't confirm VRAM usage independently")
    print(f"model device: {transcriber.device_used} ({transcriber.compute_type_used})")

    transcriber.warmup()
    times = []
    text = ""
    for i in range(3):
        t0 = time.perf_counter()
        text = transcriber.transcribe(audio)
        times.append(time.perf_counter() - t0)
        print(f"  transcribe run {i + 1}: {times[-1]:.2f}s")
    print(f"STT latency for {len(audio)/16000:.1f}s of audio: best {min(times):.2f}s")
    if wav_path and text:
        print(f"  transcript: {text}")

    if config["cleanup"]["enabled"]:
        from .cleanup import Cleaner

        cleaner = Cleaner(config["cleanup"])
        if cleaner.is_available():
            sample = (
                "um so I think we should uh meet on Monday no wait Tuesday and "
                "we need three things first the report second the slides and third the demo"
            )
            t0 = time.perf_counter()
            cleaned = cleaner.clean(sample)
            dt = time.perf_counter() - t0
            print(f"cleanup latency ({cleaner.model}): {dt:.2f}s")
            print(f"  in:  {sample}")
            print(f"  out: {cleaned}")
        else:
            print(f"[--] Ollama not reachable at {cleaner.host}; skipping cleanup bench")
    return 0


def _ensure_output_sink() -> None:
    """Under pythonw (WhisperFlow.bat) there is no console, so print() is a
    silent no-op. Route output to whisperflow.log instead, so timings and
    transcripts stay inspectable."""
    if sys.stdout is None or sys.stderr is None:
        try:
            log = open("whisperflow.log", "a", encoding="utf-8", buffering=1)
            if sys.stdout is None:
                sys.stdout = log
            if sys.stderr is None:
                sys.stderr = log
        except OSError:
            pass


def main(argv: list[str] | None = None) -> int:
    _ensure_output_sink()
    parser = argparse.ArgumentParser(prog="whisperflow", description="Local Wispr Flow clone")
    parser.add_argument("command", nargs="?", default="run", choices=["run", "doctor", "devices", "bench"])
    parser.add_argument("--config", default="config.yaml", help="path to config YAML")
    parser.add_argument("--wav", default=None, help="16-bit PCM WAV file for `bench` (real speech)")
    args = parser.parse_args(argv)

    if args.command == "devices":
        return _cmd_devices()
    if args.command == "doctor":
        return _cmd_doctor(args.config)
    if args.command == "bench":
        return _cmd_bench(args.config, args.wav)

    from .app import main as run_app

    run_app(args.config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
