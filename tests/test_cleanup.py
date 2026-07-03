"""Cleaner: Ollama happy path, unreachable fallback, and gating."""

from unittest import mock

import requests

from whisperflow.cleanup import Cleaner, is_faithful_rewrite


CONFIG = {
    "enabled": True,
    "host": "http://localhost:11434",
    "model": "gemma3:4b",
    "min_chars": 40,
    "timeout": 5,
    "temperature": 0.2,
    "keep_alive": "10m",
    "num_ctx": 2048,
}

LONG_RAW = "um so I think we should uh meet on Monday no wait Tuesday okay"


def _response(payload, status=200):
    resp = mock.Mock()
    resp.status_code = status
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


def test_clean_happy_path():
    cleaner = Cleaner(CONFIG)
    with mock.patch("whisperflow.cleanup.requests.post") as post:
        post.return_value = _response({"response": "I think we should meet on Tuesday."})
        assert cleaner.clean(LONG_RAW) == "I think we should meet on Tuesday."
        payload = post.call_args.kwargs["json"]
        assert payload["model"] == "gemma3:4b"
        assert LONG_RAW in payload["prompt"]  # transcript embedded in the template
        assert payload["stream"] is False
        assert payload["keep_alive"] == "10m"
        assert payload["options"]["num_ctx"] == 2048


def test_localhost_is_rewritten_to_ipv4_loopback():
    # "localhost" resolves IPv6-first on Windows and costs ~2s per request
    cleaner = Cleaner({**CONFIG, "host": "http://localhost:11434"})
    assert cleaner.host == "http://127.0.0.1:11434"
    cleaner = Cleaner({**CONFIG, "host": "http://192.168.1.50:11434"})
    assert cleaner.host == "http://192.168.1.50:11434"  # non-localhost untouched


def test_ollama_down_returns_raw_text():
    cleaner = Cleaner(CONFIG)
    with mock.patch(
        "whisperflow.cleanup.requests.post",
        side_effect=requests.ConnectionError("refused"),
    ):
        assert cleaner.clean(LONG_RAW) == LONG_RAW


def test_timeout_returns_raw_text():
    cleaner = Cleaner(CONFIG)
    with mock.patch(
        "whisperflow.cleanup.requests.post",
        side_effect=requests.Timeout("too slow"),
    ):
        assert cleaner.clean(LONG_RAW) == LONG_RAW


def test_short_text_skips_llm():
    cleaner = Cleaner(CONFIG)
    with mock.patch("whisperflow.cleanup.requests.post") as post:
        assert cleaner.clean("short one") == "short one"
        post.assert_not_called()


def test_disabled_skips_llm():
    cleaner = Cleaner({**CONFIG, "enabled": False})
    with mock.patch("whisperflow.cleanup.requests.post") as post:
        assert cleaner.clean(LONG_RAW) == LONG_RAW
        post.assert_not_called()


def test_empty_llm_response_returns_raw_text():
    cleaner = Cleaner(CONFIG)
    with mock.patch("whisperflow.cleanup.requests.post") as post:
        post.return_value = _response({"response": "   "})
        assert cleaner.clean(LONG_RAW) == LONG_RAW


def test_quoted_llm_response_is_unwrapped():
    cleaner = Cleaner(CONFIG)
    with mock.patch("whisperflow.cleanup.requests.post") as post:
        post.return_value = _response({"response": '"I think we should meet on Tuesday."'})
        assert cleaner.clean(LONG_RAW) == "I think we should meet on Tuesday."


def test_echoed_template_label_is_stripped():
    cleaner = Cleaner(CONFIG)
    with mock.patch("whisperflow.cleanup.requests.post") as post:
        post.return_value = _response({"response": "Edited: I think we should meet on Tuesday."})
        assert cleaner.clean(LONG_RAW) == "I think we should meet on Tuesday."


def test_model_answering_the_transcript_is_rejected():
    """If the model replies to the dictation instead of editing it, keep the raw text."""
    cleaner = Cleaner(CONFIG)
    raw = "hey how are you doing today I was wondering if you're free for lunch"
    with mock.patch("whisperflow.cleanup.requests.post") as post:
        post.return_value = _response(
            {"response": "I'm doing great, thanks for asking! Unfortunately I don't eat lunch."}
        )
        assert cleaner.clean(raw) == raw


def test_faithful_rewrite_accepts_edits():
    raw = "um so I think we should uh meet on Monday no wait Tuesday and we need three things first the report second the slides and third the demo"
    edited = "I think we should meet on Tuesday. We need three things:\n1. The report\n2. The slides\n3. The demo"
    assert is_faithful_rewrite(raw, edited)  # list numbers and case changes are fine


def test_faithful_rewrite_rejects_answers():
    raw = "hey how are you doing today"
    answer = "I'm doing great, thank you! How can I help you today?"
    assert not is_faithful_rewrite(raw, answer)


def test_faithful_rewrite_rejects_runaway_expansion():
    raw = "write a short note about the meeting for the team okay thanks"
    expansion = (
        "Dear team, I hope this message finds you well. I wanted to take a moment "
        "to summarize our recent meeting and outline the key action items we discussed "
        "so that everyone is aligned going forward and nobody misses a deadline."
    )
    assert not is_faithful_rewrite(raw, expansion)


def test_faithful_rewrite_rejects_empty():
    assert not is_faithful_rewrite("some raw text", "")


def test_warmup_asks_ollama_to_load_the_model():
    cleaner = Cleaner(CONFIG)
    with mock.patch("whisperflow.cleanup.requests.post") as post:
        post.return_value = _response({"response": ""})
        cleaner.warmup()
        payload = post.call_args.kwargs["json"]
        assert payload["model"] == "gemma3:4b"
        assert payload["prompt"] == ""  # empty prompt = load only
        assert payload["keep_alive"] == "10m"
        # must match clean()'s options or Ollama restarts the runner
        assert payload["options"]["num_ctx"] == 2048


def test_warmup_survives_ollama_down():
    cleaner = Cleaner(CONFIG)
    with mock.patch(
        "whisperflow.cleanup.requests.post",
        side_effect=requests.ConnectionError("refused"),
    ):
        cleaner.warmup()  # must not raise


def test_warmup_skipped_when_disabled():
    cleaner = Cleaner({**CONFIG, "enabled": False})
    with mock.patch("whisperflow.cleanup.requests.post") as post:
        cleaner.warmup()
        post.assert_not_called()


def test_is_available_true_false():
    cleaner = Cleaner(CONFIG)
    with mock.patch("whisperflow.cleanup.requests.get") as get:
        get.return_value = _response({}, status=200)
        assert cleaner.is_available()
    with mock.patch(
        "whisperflow.cleanup.requests.get",
        side_effect=requests.ConnectionError("refused"),
    ):
        assert not cleaner.is_available()
