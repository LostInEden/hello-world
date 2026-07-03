"""Windows CUDA DLL path registration (simulated on any platform)."""

import os

from whisperflow import transcribe


def test_noop_on_non_windows(monkeypatch):
    monkeypatch.setattr(transcribe.sys, "platform", "linux", raising=False)
    path_before = os.environ.get("PATH", "")
    transcribe._ensure_windows_cuda_dlls()
    assert os.environ.get("PATH", "") == path_before


def test_prepends_nvidia_bin_dirs_on_windows(monkeypatch, tmp_path):
    bin_dir = tmp_path / "nvidia" / "cublas" / "bin"
    bin_dir.mkdir(parents=True)
    (tmp_path / "nvidia" / "empty_pkg").mkdir()  # no bin/ -> must be skipped

    monkeypatch.setattr(transcribe.sys, "platform", "win32", raising=False)
    import site

    monkeypatch.setattr(site, "getsitepackages", lambda: [str(tmp_path)], raising=False)
    monkeypatch.setattr(site, "getusersitepackages", lambda: str(tmp_path / "nope"), raising=False)

    transcribe._ensure_windows_cuda_dlls()
    assert os.environ["PATH"].startswith(str(bin_dir) + os.pathsep)
