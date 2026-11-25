# Luminar Videos

Ein Videoschnitt-Programm im Stil von CapCut, das man zusammen mit **Claude Code** benutzt: Du sagst Claude, was für ein Video
du willst, und Claude schneidet es. Das Programm läuft ganz auf deinem eigenen Rechner (Mac oder Windows), ohne Anmeldung und ohne Cloud.

Das kann es: Schnitte im Takt der Musik, weiche Übergänge wie in CapCut, Film-Looks (Farbfilter für das ganze Video),
professionelle Schriften (auch leicht durchsichtig), **Text hinter einer Person oder hinter einem Gebäude**, **großer 3D-Text aus Milchglas, der wie ein echtes Schild in der Szene steht** (z. B. hinter der Schule bei einem Drohnenflug), Hintergrund entfernen, automatische Untertitel,
Zeitlupe, Keyframes, Masken und Green Screen.

## Einrichten auf dem Mac (einmalig)
1. **Homebrew** installieren: https://brew.sh (den Befehl von der Seite ins Terminal kopieren).
2. Im Terminal: `brew install python ffmpeg`
3. **Claude Code** installieren: https://claude.com/claude-code (du brauchst dafür ein Claude-Konto, z. B. Pro).
4. Diesen Ordner herunterladen (grüner Knopf „Code“ → „Download ZIP“) und entpacken, z. B. in „Dokumente“.
5. Im Terminal einmal: `chmod +x ~/Documents/clipcut-main/*.command` (an den Ordnerort anpassen).

## Benutzen
- **Programm öffnen:** im Ordner auf `start.command` doppelklicken. Der erste Start dauert ein paar Minuten (Pakete werden installiert).
  Falls der Mac sagt „nicht verifizierter Entwickler“: Rechtsklick → Öffnen → Öffnen.
- **Videos und Musik** in den Ordner `media/` legen (oder ins Programmfenster ziehen).
- **Mit Claude schneiden:** im Ordner auf `Mit Claude schneiden.command` doppelklicken (startet Claude Code direkt im richtigen Ordner;
  einmalig wie oben `chmod +x` bzw. Rechtsklick → Öffnen). Dann einfach auf Deutsch sagen, was du willst - die Videos dürfen
  irgendwo liegen (z. B. in Downloads), Claude sucht sie und kopiert sie ins Programm. Beispiel:
  > Mach aus den Videos in media ein 60-Sekunden-Video für den Europatag, Querformat, ruhige Übergänge, Musik `musik.mp3`,
  > am Anfang der Titel „Unsere Schule in Europa“ leicht durchsichtig hinter der Person, Film-Look „kino“.

  Oder: „Stell beim Drohnenflug einen großen 3D-Schriftzug ‚LESSING GYMNASIUM‘ hinter die Schule, der in der Szene stehen bleibt.“

  Claude baut das Video Schritt für Schritt, und das offene Programmfenster zeigt jede Änderung live. Fertiges Video: Export im Programm
  oder Claude fragen („rendere das Video“).

## Windows
`pip install -r requirements.txt`, ffmpeg installieren (`winget install Gyan.FFmpeg`), dann `start.bat` doppelklicken.

## Datenschutz
Schneiden, Hintergrund entfernen und Rendern laufen lokal. Wenn Claude sich Clips „anschaut“ (Kontaktbögen), werden einzelne
Standbilder an Anthropic geschickt. Bei Aufnahmen von Schülerinnen und Schülern vorher die Einwilligungen bzw. die Regeln der Schule klären.

## Für Entwickler
`python clipcut.py selftest` (Logik) und `python clipcut.py rendertest` (kurzer Test-Export). Schriften in `fonts/`: Montserrat,
Playfair Display, Anton (SIL Open Font License).
