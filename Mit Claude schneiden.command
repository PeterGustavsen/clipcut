#!/bin/bash
# Luminar Videos: Claude Code direkt im Programmordner starten (im Finder doppelklicken).
# Claude liest dann automatisch die Anleitung (CLAUDE.md) und kennt alle Funktionen.
cd "$(dirname "$0")"

if ! command -v claude >/dev/null 2>&1; then
  echo "Claude Code ist noch nicht installiert."
  echo "Installieren: https://claude.com/claude-code  (danach diese Datei nochmal doppelklicken)"
  read -n 1 -s -r -p "Taste druecken zum Schliessen"; exit 1
fi

if [ ! -x .venv/bin/python ]; then
  echo "Tipp: zuerst einmal start.command doppelklicken (installiert die Pakete und oeffnet das Programmfenster)."
  echo
fi

echo "Luminar Videos - sag Claude einfach auf Deutsch, was fuer ein Video du willst."
echo "Zum Beispiel: Mach aus den Videos im Ordner Downloads/Europatag ein Europa-Video mit Karten-Intro."
echo
exec claude
