# ClipCut — agent API

ClipCut is a CapCut-style video editor that Claude Code drives. Typical users here are teachers and students making school videos
(project days, Europe day, school trips, interviews, presentations). They speak German; answer in German, in plain words, no jargon.
`project.json` in this folder IS the timeline. Edit it with any tool and the open editor reloads it within 2 s,
flashes what changed, jumps the playhead there and shows the last `agent.jsonl` message as a 🤖 banner.
Media lives in `media/` and is referenced by basename only. Runs on macOS (`start.command`) and Windows (`start.bat`), all local.

## How to make a school video (the default unless the user asks for something else)
Goal: calm, clean, PROFESSIONAL - like a good documentary or an image film, not a TikTok meme edit.
1. LOOK at the footage first: `python clipcut.py look media/x.mp4 --n 12` for every clip (contact sheet image -> read it). Note what
   each clip shows (people, buildings, speech, landscape) and its orientation. Ask the user only what you cannot see: length, format
   (16:9 for projector/YouTube = canvas [1920,1080]; 9:16 for Instagram/TikTok = [1080,1920]), music, title text.
2. Build the timeline step by step with `op` commands (the user watches the editor) or write project.json directly:
   - shots of 2-5 s, cut on the beat of the music (`python clipcut.py beats media/song.mp3` -> beats/strong/bpm)
   - transitions: mostly hard cuts, plus `dissolve` / `fade` 0.4-0.8 s between topics, `fadeblack` 0.8 s at the start/end.
     Avoid flashes (`fadewhite`), shake, zoom punches, glitch and laser eyes unless asked.
   - slow push-in feels cinematic: segment `zoom: 1.0, zoomTo: 1.08`.
   - a look for the whole video (`filter` at project level): `clean` (bright sunny/outdoor/drone footage - kino makes it murky),
     `kino` (cinematic, darker, for indoor/evening/interviews), `film_warm`, `teal_orange`, `soft`, `doku_bw` (black & white).
     Render one frame with the look and LOOK at it before committing to it. Keep skin natural, never `phonk`/`noir`/`grunge` for school videos.
   - text: font `montserrat` (modern) or `playfair` (elegant serif) - these have no meme outline; `opacity` 0.8-0.9, `shadow: true`,
     `anim_in`/`anim_out` `fade` 0.4-0.6 s. Title size 0.08-0.13, lower third 0.035-0.045 at y 0.85. Keep text inside the frame
     (long words at size > 0.12 overflow a 9:16 canvas).
   - **text behind a person** (CapCut "text behind subject"): big title with `behind: true`, then
     `python clipcut.py op behind <textid>` -> RVM cuts out the person in every shot under the text and puts the cut-out on top.
     Place the text so the person's head/shoulders overlap part of it (e.g. y 0.35-0.45).
   - **text behind a building** (or tower, skyline, trees): `behind: "building"` / `python clipcut.py op behind <textid> building`
     -> with sky in the shot (>= 4 % of the first frame): sky segmentation (models/skyseg.onnx, 176 MB), everything that is NOT sky
     stands in front; put the text IN the sky so the roof/tower/statue cuts into the lower part of the letters.
     Without sky (drone from above, street level): depth (models/depth_small.onnx = Depth Anything V2 small, 99 MB), near/far split
     by Otsu per frame - the building in front covers the text, what lies behind it shows it. Put the text just BEHIND the roof edge
     (in the yard/street/houses behind the building). With a drone flying towards the building the roof edge moves down the frame,
     so a text placed just below the first frame's roof edge is a reveal: it rises out from behind the roof (Lessing drone test
     2026-11-25: 1920x1080, text y 0.43 size 0.1, edge at y ~0.37 -> ~0.57). Keep it off people you want to show (crowds in a yard).
     Both models download once automatically. Cost: ~25 s per second of footage on a laptop CPU (depth every 3rd frame) - tell the
     user it takes a few minutes. LOOK at a frame first (`look`) to find the edge, then check rendered frames.
     After moving/retiming the text or changing the shots under it, run `op behind` again (cut-outs are cached in media/).
   - **big 3D text that stays in the scene** (After Effects "3D camera tracker" look - THE showpiece for school/drone videos,
     use it when the user wants a title "behind the school", "standing in the scene", "3D", "like After Effects"):
     `python clipcut.py op worldtext #N "LESSING GYMNASIUM" y=0.37 w=0.95` -> worldtext.py bakes the shot: huge milky,
     slightly transparent frosted-glass letters with a 3D extrusion (upwards: seen from above you see their top faces), a light
     glass rim and a soft contact shadow. A real 3D camera tracker (like After Effects) rebuilds the camera path and stands the
     text as an upright sign in the scene behind the building, so when the drone/camera moves the text stays where it is and
     changes perspective like a real sign; building, people and trees in front cover it (depth). Best with a moving camera
     (drone, gimbal, handheld push-in); a camera that only pans/stands gets an exact 2D follow instead (the log says so).
     Placement (fractions of the shot's FIRST frame): `y` = bottom of the letters - put it just behind the building's back edge
     so the building/roof cuts or touches the foot of the letters, never over the people/action the video is about;
     `w` = width (0.8-0.95 = big); `x` = centre. Options: `opacity` 0.8, `depth` 0.22 (extrusion), `style` frost|solid,
     `font` montserrat|playfair|impact. Recipe: `look` the shot, pick y from the first frame (where is the back edge of the
     building, where are the people), run it (~50 s per second of footage; the log prints the solve error - mean under 1 px is good), then LOOK at 4 frames spread over the shot
     (`ffmpeg -i media/out.mp4 -vf "select='eq(n,0)+eq(n,80)+eq(n,160)+eq(n,230)',scale=800:-2,tile=2x2" -frames:v 1 -fps_mode passthrough tmp/s.png`):
     the text must not cover faces/people and must read clearly. Re-run with new values any time (it re-bakes from the original
     footage, kept in segment.world.src); `op worldtext #N none` = remove. Cut the shot to its final length BEFORE (the bake
     covers the shot's in..out). Checked 2026-11-25: Lessing drone push-in (solve 0.46 px; independent SIFT 3-view check
     around the text <= 1.7 px over the whole 8 s - the old 2D tracker drifted 20+ px), Brandenburger Tor handheld.
   - music: audio item with `fadeIn` 1, `fadeOut` 2, volume ~0.6 under speech (speech clips keep `volume` 1).
   - interviews / speech: `python clipcut.py` auto captions via the Text tab (faster-whisper) or `/api/transcribe`; set captions to
     `montserrat`, size 0.045, `box: true` or `shadow: true`.
3. Render: `python clipcut.py render project.json media/fertig.mp4`, then LOOK at frames
   (`ffmpeg -ss T -i media/fertig.mp4 -frames:v 1 tmp/f.png`) before telling the user it is done.
Privacy: footage of students stays on this computer; only the contact sheets you read leave it. Never upload media anywhere.

## Fully automatic edit (fast music montages)
```
python clipcut.py tiktok media/SONG.mp3 media/a.mp4 media/b.mp4 ... --style montage --look kino [--text "UNSER EUROPA"] [--every N] [--pace 0.4]
```
Styles: `montage` (calm, best for school), `ironic` (speech hook then a beat-cut montage with punches and flashes), `phonk`,
`velocity` (speed ramps), `aura` (slow, b/w vignette), `trending` (cut like measured TikTok edits). `--pace` = seconds between
visible steps. In the UI: "Auto edit" (timeline toolbar). After an auto edit, tone it down for school use (remove zoom/shake/flash).
Clip choice = best moments (`motion.py`, motion + loudness), every clip at most once per edit (an edit with N shots needs N clips).
`--dumb` = simple round-robin.

## Face check (main person) — only active when `face_ref.json` exists
```
python clipcut.py faces media/a.mp4 media/b.mp4 ...   learn the main person = the face in most clips -> face_ref.json + _star_faces.png (LOOK at it)
python clipcut.py check media/x.mp4                   {ranges, visible_pct, sheet}: green/red contact sheet (LOOK at it)
python clipcut.py look media/x.mp4 --n 12             plain contact sheet of any clip
```
With `face_ref.json`, auto edits only cut from ranges where that person is clearly visible (good shots, crop centred on the face,
smooth transitions on the face). Use it when the video is about ONE person (a portrait, an interview). For group/class videos
delete `face_ref.json` (the UI's "Make edit" learns one automatically on the first run - delete it again if that is not wanted).
`python clipcut.py review [project.json]` = first/middle/last frame of every shot at the rendered crop, red = no face / cut by the edge.

## Watchable manual edits
```
python clipcut.py op say "Ich schneide jetzt auf den Beat"     banner only
python clipcut.py op split 3.2                                 split the main track at timeline second 3.2
python clipcut.py op set #4 zoom=1 zoomTo=1.08 filter=kino     SEL = #index on main track or any item id; values are JSON
python clipcut.py op text "EUROPA" 3.0 2.0                     text at 3.0 s for 2 s (then `op set <id> font="montserrat" opacity=0.85 ...`)
python clipcut.py op behind <textid> [building]                put that text behind the person / building (cut-outs on top)
python clipcut.py op worldtext #N "TEXT" y=0.37 w=0.95         big 3D frosted text pinned in the scene behind a building (bakes the shot)
python clipcut.py op worldtext #N none                         remove it again (back to the original footage)
python clipcut.py op transition #3 dissolve 0.6                transition after clip #3
python clipcut.py op move #6 1 | op delete #2
python clipcut.py beats media/SONG.mp3                         {bpm, beats, strong (accents), drop (s)}
python clipcut.py render project.json media/out.mp4
python clipcut.py scenes media/x.mp4 [--threshold 0.35]       shot cut times
```
Pace yourself: one `op` every ~0.5 s reads as a human editing. Timeline seconds, not source seconds, for `split`/`text`.
HTTP equivalents while the editor runs on :8777: POST /api/tiktok {song, clips, style, text, pace}, /api/say {msg},
/api/beats {media}, /api/autoedit, /api/scenes, /api/transcribe, /api/render <project>, /api/project <project>;
GET /api/media, /api/project (-> {mtime, project, say}).

## project.json schema (seconds, floats)
```
{ canvas: [w,h] | null (= first clip), fps: 30, fill: contain|cover|blur, filter: <look> (whole video),
  video:   [segment]   gapless main track, order = play order
  overlay: [overlay]   picture-in-picture, free start
  audio:   [audio]     music / extracted audio, free start
  text:    [text]
  beats: [timeline s], strong: [timeline s], bpm,
  export: {short: 1080|720|480 (short side, 0 = canvas), crf: 14|16|20 (default 16)}, adjust: {...} (whole video) }
segment = {id, media, in, out, speed, reverse, freeze, dur, rotation, flip, volume, fadeIn, fadeOut, transition: {type, dur},
           zoom (punch-in factor), zoomTo (animate to), shake (0..1), filter (look), opacity (0..1),
           cx, cy (fill=cover only: crop centre as source fractions, default 0.5),
           keyframes: [{t, scale, x, y, rotation, opacity}], speedCurve: [{t, speed}],
           adjust: {bri, con, sat, temp, sharp, vig, grain} each -1..1, effect: glitch|rgbsplit|pixelate|blur|mirror|strobe|invert,
           crop: {x, y, w, h}, stab: true (stabilise shaky phone footage), denoise: true, pitch: -12..12,
           eyes: true (glowing eyes effect), slide: bottom|left|right|top (cut-out of the person slides in over the previous shot),
           world: {text, x, y, w, opacity, depth, style, font, src: {media, in, out}} (set by `op worldtext`; media = the baked clip)}
           transition goes INTO the next clip and overlaps it (timeline shrinks by dur). freeze=true: still frame `in` for `dur` s.
           transition types: none fade dissolve fadeblack fadewhite wipeleft/right/up/down slideleft/... smoothleft/right
           circleopen/close radial zoomin pixelize hblur distance squeezeh diagtl hlslice
           motion-blur transitions (transitions.py): pullin pullout whipleft/right/up/down spin glitch shake blur flicker (dur 0.3-0.5)
           looks: none | kino film_warm teal_orange clean soft doku_bw (professional) | bw vintage cool warm | capcut_silber capcut_demut
           capcut_bw capcut_matte capcut_siegel capcut_terror (CapCut LUTs in luts/) | phonk vhs aura dark noir moody dusk grunge flash neg (edit/meme looks)
overlay = {id, media, start, in, out, speed, x, y, scale, rotation, flip, volume, opacity, keyframes, mask, chroma, adjust, effect, crop}
          x,y centre as canvas fraction, scale = width fraction; chroma = {color "#00ff00", similarity 0.3, blend 0.1} (green screen);
          mask = {type: rect|circle|linear|mirror, x, y, w, h, feather, invert}. Overlays tagged `_behind` belong to a behind-text.
audio   = {id, media, start, in, out, speed, volume, fadeIn, fadeOut, denoise, pitch}
text    = {id, text, start, dur, x, y, size, color, opacity (0..1), behind (true = person | "building"), font, box, boxcolor, shadow, outline (bool),
           anim_in, anim_out, hl, hlcolor, hlscale, sticker}
          size = font px / canvas height. "\n" in `text` = new line.
          fonts: montserrat, playfair (bundled in fonts/, no outline by default) | arial impact georgia comic times (system fonts, black outline;
          on the Mac from /System/Library/Fonts/Supplemental, on Linux Anton/DejaVu stand in).
          anim_in / anim_out = {type: fade|pop|typewriter|slide|shake, dur}.
          hl: "WORD" (or list) = that word in `hlcolor` (#ffd60a) at `hlscale` x; sticker: {media, word, scale, rot} = image after a word.
```
Keyframes: `t` = seconds after the item starts; scale multiplies zoom/scale, x/y shift by canvas fraction, rotation adds degrees,
opacity is absolute; linear. `python clipcut.py op set #1 'keyframes=[{"t":0,"scale":1},{"t":3,"scale":1.08}]'` (values are JSON).
Preview approximations: looks, brightness/contrast/saturation, blur; temperature/sharpen/vignette/grain/chroma/stabilise/denoise are export-only.
Encode: libx264 crf 16, aq-mode 3.

## Copy an existing edit (TikTok / mp4 reference)
```
python clipcut.py copy https://www.tiktok.com/@user/video/ID  media/clip1.mp4 ...  [--text "..."] [--pace 0.4] [--even]
```
Downloads the reference (yt-dlp), cuts the user's clips at its exact cut times with its slow-mo, flashes and sound. `--even` snaps
the cuts to the beat. UI: Effects tab > "Copy an edit".

## Running it
macOS: double-click `start.command` (installs into `.venv` on the first run; needs `brew install python ffmpeg`).
Windows: `pip install -r requirements.txt`, ffmpeg on PATH, `start.bat`. Or `python server.py` -> editor on http://127.0.0.1:8777.
On the Mac use `.venv/bin/python clipcut.py ...` (or activate the venv) so the installed packages are found.
Checks: `python clipcut.py selftest` (logic), `python clipcut.py rendertest` (2 s render through every feature),
`python copy_edit.py`, `python trends.py`, `python transitions.py`, `python worldtext.py selftest`.
