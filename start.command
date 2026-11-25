#!/bin/bash
# Luminar Videos auf dem Mac starten: im Finder doppelklicken.
# Beim ersten Start werden die Python-Pakete in .venv installiert (dauert ein paar Minuten).
cd "$(dirname "$0")"

if ! command -v ffmpeg >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then
    echo "ffmpeg fehlt - installiere es mit Homebrew ..."
    brew install ffmpeg
  else
    echo "ffmpeg fehlt. Erst Homebrew installieren (https://brew.sh), dann: brew install ffmpeg"
    read -n 1 -s -r -p "Taste druecken zum Schliessen"; exit 1
  fi
fi

if [ ! -x .venv/bin/python ]; then
  echo "Erster Start: Python-Pakete werden installiert ..."
  python3 -m venv .venv || { echo "python3 fehlt: https://www.python.org/downloads/ installieren"; read -n 1 -s -r; exit 1; }
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r requirements.txt
fi

.venv/bin/python app.py
