# whisperflow-local

A local, offline clone of [Wispr Flow](https://wisprflow.ai) — system-wide AI
voice dictation — built for **Windows 10/11 with a dedicated NVIDIA GPU (CUDA)**.

Hold a hotkey, speak into any text field in any app, release, and your cleaned-up
words are typed where your cursor is. Everything runs on your machine: speech-to-text
with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) on your GPU, and an
optional "make it sound written" cleanup pass through a small local LLM via
[Ollama](https://ollama.com). No cloud, no subscription, no data leaving your PC.

> See [`RESEARCH.md`](./RESEARCH.md) for the deep-dive on how Wispr Flow actually
> works and why this stack was chosen (fully cited).

## What it does (base feature parity with Wispr Flow)

| Wispr Flow | This clone |
|---|---|
| Global push-to-talk hotkey (Hold Mode) | ✅ Hold-to-record via a configurable global hotkey (pynput) |
| System-wide dictation into any app | ✅ Injects text into the focused field (clipboard paste) |
| Local/near-realtime transcription | ✅ faster-whisper on CUDA (`large-v3-turbo` by default) |
| Voice-activity detection | ✅ Silero VAD (bundled in faster-whisper via `vad_filter`) |
| AI cleanup: filler removal, punctuation, lists, self-corrections | ✅ Local LLM cleanup pass via Ollama (e.g. `gemma3:4b`) |
| Custom vocabulary | ⏳ Roadmap (initial-prompt biasing) |
| Command Mode (transform selected text) | ⏳ Roadmap |

## The pipeline

```
[hold hotkey]
     │
     ▼
 mic capture ──► Silero VAD ──► faster-whisper (CUDA) ──► local LLM cleanup (Ollama) ──► paste into focused app
 (sounddevice)   (endpoint)      speech → raw text         raw text → clean text          (clipboard + Ctrl+V)
```

---

## Requirements

- **Windows 10/11**
- **NVIDIA GPU with CUDA** (dedicated card recommended; 6 GB+ VRAM comfortable for `large-v3-turbo`)
- **Python 3.10–3.12**
- **[Ollama](https://ollama.com/download)** (only needed for the AI cleanup step — optional)

## Setup

### 1. Clone and create a virtual environment

```powershell
git clone <this-repo> whisperflow-local
cd whisperflow-local
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2. Install Python dependencies

```powershell
pip install -r requirements.txt
```

### 3. Install CUDA libraries for faster-whisper

faster-whisper needs cuBLAS and cuDNN 9 for CUDA 12. The easiest way on Windows is
to grab them from PyPI (they get picked up automatically):

```powershell
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

If you hit `Could not load library cudnn_ops64_9.dll`, see
[faster-whisper's GPU notes](https://github.com/SYSTRAN/faster-whisper#gpu) — you
either need the pip packages above on your `PATH`, or the NVIDIA CUDA Toolkit installed.

### 4. (Optional) Install Ollama + pull a cleanup model

The dictation works without this — you just get the raw transcript. For the
Wispr-Flow-style "sounds like I wrote it" cleanup, install Ollama and pull a small model:

```powershell
# after installing Ollama from https://ollama.com/download
ollama pull gemma3:4b
```

### 5. Configure

```powershell
copy config.example.yaml config.yaml
```

Edit `config.yaml` to taste (hotkey, model size, cleanup on/off). Defaults are sensible.

---

## Usage

Run a quick health check first (verifies CUDA, audio devices, and Ollama):

```powershell
python -m whisperflow doctor
```

List your microphones if you need to pick a specific one:

```powershell
python -m whisperflow devices
```

Benchmark the pipeline — this loads the model with your config, confirms it
actually landed on the GPU (checks VRAM via `nvidia-smi`, not just logs), and
measures STT + cleanup latency:

```powershell
python -m whisperflow bench
# or, with a real recording for a realistic number:
python -m whisperflow bench --wav my_speech.wav
```

Start dictating:

```powershell
python -m whisperflow
```

Then, in **any** application:

1. Put your cursor in a text field.
2. **Hold** the hotkey (default: **Ctrl + Alt**) and speak.
3. **Release** the hotkey.
4. Your cleaned-up text is typed at the cursor.

After every utterance the terminal prints a latency breakdown, e.g.
`⏱️ 0.84s release-to-text (stt 0.52s, cleanup 0.29s, inject 0.03s)`. If you're
over ~1s for short utterances, try `compute_type: int8_float16` or a smaller
cleanup model (`qwen2.5:3b`) — STT and the LLM share VRAM.

Press **Ctrl + C** in the terminal to quit.

> **Tip:** the first run downloads the Whisper model (a few hundred MB to ~1.5 GB
> depending on model) and loads it into VRAM. Subsequent starts are fast.

---

## Configuration reference

See [`config.example.yaml`](./config.example.yaml) — every option is documented inline.
Highlights:

- **`hotkey.combo`** — e.g. `ctrl+alt`, `ctrl+shift+space`, `f9`. Modifier-only combos
  (like the default `ctrl+alt`) are the most conflict-free for push-to-talk.
- **`hotkey.mode`** — `push_to_talk` (hold) or `toggle` (press once to start, again to stop).
- **`transcription.model`** — `large-v3-turbo` (default, fast+accurate),
  `large-v3` (most accurate), `distil-large-v3` (English, faster), or `small`/`medium`
  for low VRAM.
- **`transcription.compute_type`** — `float16` (default), `int8_float16` or `int8`
  to cut VRAM usage.
- **`cleanup.enabled`** — turn the LLM cleanup pass on/off.
- **`cleanup.model`** — any Ollama model, e.g. `gemma3:4b`, `qwen2.5:3b`, `llama3.1:8b`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Could not load library cudnn_ops64_9.dll` | `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12` and restart the shell |
| Falls back to CPU / very slow | Run `python -m whisperflow doctor`; confirm `device: cuda` in config and that the CUDA libs are installed |
| Nothing gets typed | Some apps block synthetic paste; try `injection.method: type`. Run the terminal as admin if the hotkey doesn't fire globally |
| Cleanup does nothing | Ensure `ollama serve` is running and the model is pulled; check `cleanup.enabled: true` |
| Hotkey not detected in elevated apps | Global hooks can't see keystrokes sent to higher-privilege windows unless this app also runs elevated |

## Development

The test suite mocks the model, audio, and Ollama, so it runs anywhere:

```powershell
pip install pytest
python -m pytest tests/
```

## License

MIT — see [`LICENSE`](./LICENSE). Not affiliated with Wispr Flow.
