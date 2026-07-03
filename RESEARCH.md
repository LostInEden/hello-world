# How Wispr Flow works, and how to clone it locally

Deep-research findings behind `whisperflow-local`. Target machine: **Windows 10/11
with a dedicated NVIDIA GPU (CUDA)**. Every non-obvious claim below was
cross-checked against multiple sources and adversarially verified; sources are
listed at the end and cited inline as `[n]`.

---

## TL;DR

Wispr Flow's magic is **not** raw transcription — Whisper-class speech-to-text is
commodity now. The differentiator is a **second, LLM-based cleanup stage** that
runs after transcription: a fine-tuned Meta Llama model strips filler words, fixes
punctuation, formats lists, and applies spoken self-corrections, with the whole
speech→text→LLM pipeline landing **under ~700 ms p99** [1][3].

That is fully reproducible offline. The recommended local stack for your hardware:

| Layer | Pick | Why |
|---|---|---|
| **STT** | **faster-whisper** (`large-v3-turbo`) on CUDA | Up to ~4× faster than reference Whisper at equal accuracy, low VRAM, leaves headroom for the LLM [2] |
| **VAD** | **Silero VAD** (bundled in faster-whisper) | <1 ms per 30 ms chunk on CPU — stays off the GPU entirely [4] |
| **Cleanup LLM** | **Ollama** running a ~3–8B model (`gemma3:4b`, `qwen2.5:3b`, `llama3.1:8b`) | Small models are plenty for text cleanup; matches what Wispr itself uses (fine-tuned Llama) [1][5][6] |
| **Activation** | Global **push-to-talk hotkey** (hold to record) | The exact model Wispr uses ("Hold Mode") [1] |
| **Injection** | **Clipboard paste** (Ctrl+V), keystroke fallback | Proven pattern in existing clones [7][8] |
| **Starting point** | **Fork [Handy](https://github.com/cjpais/Handy)** or **[WhisperWriter](https://github.com/savbell/whisper-writer)** | Both already ship hotkey + VAD + GPU-Whisper + injection [8][9] |

---

## 1. What Wispr Flow actually does

Wispr Flow is a **system-wide dictation layer**, not a standalone editor. It runs in
the background across macOS 12+, Windows 10+, iPhone (iOS 18.3+) and Android (13+),
and inserts text into whatever field is focused — Slack, Gmail, docs, your IDE [1][3].

**The pipeline** (verified against the vendor site + a Baseten engineering case study):

```
hotkey / push-to-talk  →  audio capture  →  VAD  →  STT (cloud)  →  LLM post-process  →  inject into focused field
```

**Core UX features:**

- **Hold Mode (push-to-talk):** press and hold the bound key (default `fn`), speak,
  release to insert. A double-tap toggle gives hands-free mode [1].
- **The LLM cleanup pass** — this is the whole point. Wispr goes "beyond basic
  dictation by cleaning up filler words, formatting lists, catching punctuation, and
  understanding corrections in real time" [1]. Baseten's case study confirms this
  stage is a **fine-tuned Meta Llama LLM** running after STT, end-to-end **<700 ms
  p99** [3]. Marketed as "4× faster than typing" (220 vs 45 wpm) — vendor framing [1].
- **Command Mode:** highlight text, hold a shortcut, speak an instruction ("make this
  shorter", "translate to French", "fix the tone") — the LLM rewrites the selection in
  place. A distinct, opt-in mode [1].
- **Personal dictionary:** auto-learns terms. "When you correct a spelling, Flow adds
  it automatically to your personal dictionary" [1].

> ⚠️ One popular belief is that Wispr infers punctuation from **pauses and tone**.
> Our verification **refuted** this (0–3 votes) — it's not substantiated. Punctuation
> almost certainly comes from the STT model and/or the cleanup LLM, not prosody
> analysis. Wispr's exact STT vendor/model is **not publicly disclosed**; only the
> Llama cleanup stage is confirmed [3].

---

## 2. Speech-to-text: which local model

For a **Windows + NVIDIA/CUDA** machine, the accuracy-vs-latency sweet spot is
**faster-whisper**, a reimplementation of OpenAI Whisper on the CTranslate2 inference
engine. It is "up to 4 times faster than openai/whisper for the same accuracy while
using less memory" [2].

**faster-whisper's own reproducible benchmark** (13-min audio, `large-v2`, RTX 3070 Ti
8 GB, CUDA 12.4) [2]:

| Engine / precision | Time | VRAM |
|---|---|---|
| openai/whisper fp16, beam 5 | 2m23s | 4708 MB |
| faster-whisper **int8** | **59 s** | **2926 MB** |
| faster-whisper batched fp16 (batch 8) | **17 s** | 6090 MB |

The int8 path realizes the ~4× headline and cuts VRAM under 3 GB — leaving room to
**co-host the cleanup LLM on the same GPU**.

**The rest of the field:**

- **Whisper `large-v3-turbo`** (OpenAI, Oct 2024): decoder-reduced speed variant,
  used in near-realtime tutorials — a great default for dictation [5].
- **Distil-Whisper:** ~6× faster, ~49% smaller, within ~1% WER of Whisper — but
  **English-only**, and the smallest `distil-small.en` is within ~4% WER (not 1%) [10].
- **NVIDIA Parakeet V3** (`parakeet-tdt-0.6b-v3`): notably strong on **CPU** (~5×
  real-time on a mid-range i5) with auto language detection [8]. Great if you ever want
  a CPU fallback, but on a dedicated GPU faster-whisper wins.
- **openai/whisper** (reference): slower, more VRAM — no reason to use it here.
- **whisper.cpp:** excellent for CPU/Apple Silicon; on NVIDIA, faster-whisper is better.

**Recommendation:** `large-v3-turbo` in fp16 for the best feel; drop to
`int8_float16`/`int8` or a smaller model on low-VRAM cards; lock `language: en` to
avoid misdetection latency.

> **Open question the sources didn't settle:** none of the recommended local stacks
> demonstrate true **word-by-word streaming**; they're all "record → transcribe a
> chunk". For push-to-talk dictation that's fine. Real streaming (partial hypotheses)
> would need NVIDIA Parakeet/Canary streaming (NeMo) or a `whisper-streaming` fork.

---

## 3. The cleanup LLM (the part that "feels smart")

Wispr's cleanup is a fine-tuned Llama [3]. You reproduce the *class* of tool locally
with a small model. Two backends are demonstrated in the wild:

- **Ollama** (`local-whisper` project): "If you have Ollama installed, you can enable
  LLM-powered text cleanup... the text is sent to a local LLM that fixes punctuation,
  removes filler words, and formats numbered lists — all on-device," defaulting to
  `gemma3:4b` ("small, fast, good at text cleanup"), gated to text >50 chars [6].
- **llama.cpp** (`llama-cpp-python`): a separate tutorial does the same
  rambling→coherent rewrite with a Hermes-3-Llama-3.1-8B GGUF and no Ollama [5].

**Ollama vs alternatives for this use case:**

| Tool | Best for |
|---|---|
| **Ollama** | Simplest turnkey on Windows: one installer, `ollama pull`, HTTP API. **Recommended default.** |
| **llama.cpp / LM Studio** | More control over quantization/params; LM Studio adds a GUI |
| **vLLM** | High-throughput serving — overkill for a single-user dictation tool |

**Model size:** a **~3–8B** model is plenty for cleanup (it's a rewriting task, not
reasoning). `gemma3:4b` is a proven default; `qwen2.5:3b` is faster; `llama3.1:8b` is
higher quality if VRAM allows. This is exactly the tradeoff `whisperflow-local` exposes
via `cleanup.model` [5][6].

---

## 4. Windows system integration

Every piece here is a solved pattern in existing clones [7][8]:

- **Global hotkey / push-to-talk:** WhisperWriter defines default `ctrl+shift+space`
  and four recording modes: `continuous`, `voice_activity_detection`,
  `press_to_toggle`, `hold_to_record` (= push-to-talk) [7]. Handy uses Rust's `rdev`
  for the same [8]. `whisperflow-local` implements hold-to-record and toggle via pynput.
- **Microphone capture:** `cpal` (Handy) / PyAudio (WhisperWriter). We use
  `sounddevice` (16 kHz mono float32 — Whisper's native format).
- **VAD / endpointing:** **Silero VAD** is the accuracy-favored default — "one audio
  chunk (30+ ms) takes less than 1ms to be processed on a single CPU thread" (~30×
  real-time), a ~1–2 MB model [4]. faster-whisper bundles it (`vad_filter=True`), so we
  get it for free. `webrtcvad` is a lighter alternative.
- **Text injection:** clipboard-paste (copy + Ctrl+V) is the primary method in Handy;
  `enigo`/keystroke simulation is the fallback [8]. We do exactly that (paste with
  clipboard restore; `type` fallback for apps that block synthetic paste).

---

## 5. Existing open-source projects to fork

You almost certainly should **not** build from zero. Ranked:

1. **[Handy](https://github.com/cjpais/Handy)** — *strongest fork target.* MIT, Rust +
   Tauri (React/TS UI), fully offline, **Windows/macOS/Linux**. Already ships global
   push-to-talk (`rdev`), Silero VAD (`vad-rs`), GPU-accelerated Whisper
   (Small/Medium/Turbo/Large) **plus** NVIDIA Parakeet V3, and clipboard/keystroke
   injection. Self-describes as "the most forkable one." **The one gap: it does not ship
   the LLM cleanup stage** — you'd add an Ollama call post-transcription [8].
2. **[WhisperWriter](https://github.com/savbell/whisper-writer)** — *best if you want
   Python.* System-wide dictation, local **faster-whisper on CUDA** (`device: cuda`,
   models tiny→large-v3), four recording modes, default `ctrl+shift+space`. Easiest base
   to bolt an Ollama cleanup step onto — which is essentially what this repo is [7][9].
3. **[OpenWhispr](https://github.com/OpenWhispr/openwhispr)** — MIT, Electron; global
   hotkey auto-paste at cursor; local whisper.cpp + Parakeet (sherpa-onnx); BYOK cloud.
   Explicitly an "alternative to WisprFlow" [11].
4. **[push-to-talk](https://github.com/yixin0829/push-to-talk)** — Windows-focused
   Python; hotkey → transcribe → **refine** → auto-insert; pitched as cheaper than Wispr
   Flow/Superwhisper [7].
5. **[local-whisper](https://github.com/luisalima/local-whisper)** — macOS
   (whisper.cpp), but the **clearest reference for the STT→Ollama cleanup wiring**
   (`gemma3:4b`, >50-char gate). Its STT layer doesn't port to Windows/NVIDIA, but its
   cleanup pattern does — and it's the pattern `whisperflow-local` follows [6].

---

## 6. Implementation plan

**Fast path:** fork **Handy** (you get 1–3 and 5 for free) and add step 4.
**DIY path (this repo):** a clean Python implementation you fully control.

| Step | What | This repo |
|---|---|---|
| 1 | Capture mic audio on a global push-to-talk hotkey | `hotkey.py` + `audio.py` |
| 2 | Gate speech with Silero VAD | `vad_filter=True` in `transcribe.py` |
| 3 | Transcribe with faster-whisper on CUDA | `transcribe.py` |
| 4 | Post-process through a local ~3–8B LLM (Ollama) with a cleanup prompt | `cleanup.py` |
| 5 | Inject via clipboard paste (keystroke fallback) | `inject.py` |
| 6 | *(later)* Command Mode + custom dictionary | roadmap |

The latency bar to aim for is Wispr's **<700 ms p99** STT→LLM [3]. On a dedicated GPU,
`large-v3-turbo` + a 3–4B cleanup model is a realistic target — though **running STT
and the LLM concurrently on one GPU was not benchmarked in the sources**, so measure
your own VRAM/latency and drop to int8 or a smaller cleanup model if needed.

---

## Caveats

- Several performance numbers are **first-party/vendor benchmarks** (faster-whisper's
  own figures; Wispr's "4× faster than typing" marketing; Distil-Whisper's HuggingFace
  numbers, which are English-only). Reproducible, but framed favorably.
- Wispr's internal architecture is only **partially public**: the Llama cleanup stage is
  confirmed [3], but the STT model/vendor and exact punctuation mechanism are not.
- Open-source projects move fast — model lists, cuDNN versions (8 vs 9), and platform
  support may have drifted since these sources were captured.
- The recommended stack and this repo's pipeline are **syntheses**, not independently
  benchmarked end-to-end on the exact target hardware.

## Open questions

1. Achievable end-to-end latency for fully-local faster-whisper + Silero VAD + Ollama on
   a specific mid-range NVIDIA GPU — can it hit <700 ms p99 with STT **and** a 3–8B LLM
   sharing one GPU?
2. Which local STT gives true low-latency **streaming** (partial hypotheses) vs chunked
   record-then-transcribe?
3. What STT model/vendor does Wispr actually use, and how much of its perceived accuracy
   is STT vs the LLM cleanup?
4. How is punctuation actually produced (the "pauses/tone" hypothesis was refuted)?

---

## Sources

1. Wispr Flow — Features (official) — https://wisprflow.ai/features
2. SYSTRAN/faster-whisper (GitHub, benchmarks) — https://github.com/SYSTRAN/faster-whisper
3. Baseten — Wispr Flow engineering case study — https://www.baseten.co/resources/customers/wispr-flow
4. snakers4/silero-vad (GitHub) — https://github.com/snakers4/silero-vad
5. "Turn rambling into writing with Whisper and a local LLM" (Medium) — https://medium.com/design-bootcamp/build-with-genai-turn-rambling-into-writing-with-whisper-and-local-llm-394e8dd5b83f
6. luisalima/local-whisper (GitHub) — https://github.com/luisalima/local-whisper
7. yixin0829/push-to-talk (GitHub) — https://github.com/yixin0829/push-to-talk
8. cjpais/Handy (GitHub) — https://github.com/cjpais/Handy
9. savbell/whisper-writer (GitHub) — https://github.com/savbell/whisper-writer
10. huggingface/distil-whisper (GitHub) — https://github.com/huggingface/distil-whisper
11. OpenWhispr/openwhispr (GitHub) — https://github.com/OpenWhispr/openwhispr
12. Wispr Flow docs — https://docs.wisprflow.ai
13. ionio.ai — 2025 edge STT benchmark — https://www.ionio.ai/blog/2025-edge-speech-to-text-model-benchmark-whisper-vs-competitors

*Generated by a fan-out deep-research pass: 5 search angles → 22 sources fetched → 53
claims extracted → 25 adversarially verified (23 confirmed, 2 refuted).*
