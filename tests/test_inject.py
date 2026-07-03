"""Injector: paste sequence with clipboard restore, and type mode."""

from unittest import mock

import pytest

pytest.importorskip("pynput")  # needs a display server on Linux

from whisperflow.inject import Injector  # noqa: E402


def make(method="paste", restore=True):
    injector = Injector({"method": method, "restore_clipboard": restore})
    injector._keyboard = mock.MagicMock()
    return injector


def test_paste_copies_text_and_restores_clipboard():
    injector = make()
    with mock.patch("whisperflow.inject.pyperclip") as clip:
        clip.paste.return_value = "previous contents"
        injector.inject("hello world")
        assert clip.copy.call_args_list == [
            mock.call("hello world"),
            mock.call("previous contents"),
        ]
        injector._keyboard.pressed.assert_called_once()


def test_paste_without_restore_leaves_clipboard():
    injector = make(restore=False)
    with mock.patch("whisperflow.inject.pyperclip") as clip:
        injector.inject("hello")
        clip.paste.assert_not_called()
        clip.copy.assert_called_once_with("hello")


def test_paste_survives_unreadable_clipboard():
    injector = make()
    with mock.patch("whisperflow.inject.pyperclip") as clip:
        clip.paste.side_effect = RuntimeError("clipboard busy")
        injector.inject("hello")
        # restores to empty string when the old contents couldn't be read
        assert clip.copy.call_args_list[-1] == mock.call("")


def test_type_mode_uses_keyboard_type():
    injector = make(method="type")
    with mock.patch("whisperflow.inject.pyperclip") as clip:
        injector.inject("typed text")
        injector._keyboard.type.assert_called_once_with("typed text")
        clip.copy.assert_not_called()


def test_empty_text_is_a_noop():
    injector = make()
    with mock.patch("whisperflow.inject.pyperclip") as clip:
        injector.inject("")
        clip.copy.assert_not_called()
