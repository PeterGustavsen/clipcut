@echo off
rem Luminar Videos: Claude Code direkt im Programmordner starten (doppelklicken).
cd /d "%~dp0"
where claude >nul 2>nul || (echo Claude Code ist noch nicht installiert: https://claude.com/claude-code & pause & exit /b 1)
echo Luminar Videos - sag Claude einfach auf Deutsch, was fuer ein Video du willst.
echo.
claude
