@echo off
rem Launch WhisperFlow silently in the background (no console window).
rem Control it from the system tray icon next to the clock.
cd /d "%~dp0"
start "" pythonw -m whisperflow
