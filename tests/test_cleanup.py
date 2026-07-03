"""Cleaner: Ollama happy path, unreachable fallback, and gating."""

from unittest import mock

import requests

from whisperflow.cleanup import Cleaner


CONFIG = {
    "enabled": True,
    "host": "http://localhost:11434",
    "model": "gemma3:4b",
    "min_chars": 40,
    "timeout": 5,
    "temperature": 0.2,
    "keep_alive": "10m",
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
        assert payload["prompt"] == LONG_RAW
        assert payload["stream"] is False
        assert payload["keep_alive"] == "10m"


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
        post.return_value = _response({"response": '"Cleaned text."'})
        assert cleaner.clean(LONG_RAW) == "Cleaned text."


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
