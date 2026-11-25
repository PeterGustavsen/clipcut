"""clipcut - CapCut-style clip ops on top of ffmpeg.

  scenes  IN [--threshold 0.35] [--split DIR]      shot detection, optional split into clips
  search  IN SUBS.srt QUERY OUT [--keep] [--pad S] text-search cut (drop or keep matching cues)
  audio   IN OUT                                   extract audio, format from OUT suffix
  rotate  IN OUT DEG                               rotate 90/180/270 (re-encode) or any angle
  combine OUT IN... [--transition fade] [--dur S]  join clips, overlapping xfade transitions
  beats   AUDIO                                    beat grid as JSON {bpm, beats:[s...]}
  render  PROJECT.json OUT.mp4                     render a project file (media paths relative to media/)
  autoedit SONG OUT.json CLIP... [--every N] [--portrait]   beat-synced cut of CLIPs to SONG -> project file
  tiktok  SONG CLIP... [--style auto|ironic|phonk|velocity|aura|montage|trending] [--text "WORDS"] [--hook S] [--pace 0.4] [--dumb]
                                                   full automatic "edit" -> project.json, step by step (watchable in the editor)
  trends  refresh [--query "phonk edit"] [--n 10] | show   fingerprint popular TikTok edits (media/_trends/) -> style "trending"
  op      say MSG | split T | set SEL k=v.. | text "T" START DUR | delete SEL | move SEL IDX | transition IDX TYPE DUR | eyes SEL | behind TEXTID [building] | worldtext SEL "TEXT" [x= y= w= opacity= depth= style=frost|solid font= until= fade= stars=1] | worldtext SEL none
                                                   | lowerthird "Name" "Rolle" START [DUR] | captions SEL | translate FILE.json [mode=add|replace]
                                                   | mapintro LAT LON "Label" [dur=7 sub="Köln"]
                                                   one visible edit on project.json (SEL = #index or id); the editor shows it live
  faces   CLIP...                                  build/refresh the star reference from these clips (-> face_ref.json + _star_faces.png)
  check   CLIP                                     where the star is on screen: ranges + contact sheet with green/red frames
  look    CLIP [--n 12]                            contact sheet PNG of a clip so you can see what is in it
  review  [PROJECT.json]                           first/middle/last frame of every shot at the rendered crop + star box -> %TEMP%\review_<name>.png
  selftest                                         run logic checks (no ffmpeg needed)

Agent API: project.json (schema below `combine`) is the whole timeline. Edit it, the open editor reloads it live.
"""
import argparse, json, os, re, shutil, subprocess, sys, tempfile
from pathlib import Path

def find_tool(name):
    bundled = Path(getattr(sys, "_MEIPASS", Path(__file__).parent)) / f"{name}.exe"  # PyInstaller build ships ffmpeg
    if bundled.is_file():
        return str(bundled)
    # winget updates PATH only for processes started after the install; fall back to its package folder
    hits = sorted(Path.home().glob(f"AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg*/*/bin/{name}.exe"))
    return shutil.which(name) or (str(hits[-1]) if hits else name)


FF, FP = find_tool("ffmpeg"), find_tool("ffprobe")


def graph_file_opt(path):
    """ffmpeg option that reads the filtergraph from a file: '-/filter_complex FILE' (ffmpeg >= 7; 9 dropped the old form),
    '-filter_complex_script FILE' on 6.x (Ubuntu 24.04's VPS build). Version probed once."""
    global _FF_MAJOR
    if _FF_MAJOR is None:
        try:
            m = re.search(r"ffmpeg version n?(\d+)", subprocess.run([FF, "-version"], capture_output=True, text=True).stdout)
            _FF_MAJOR = int(m.group(1)) if m else 7
        except Exception:
            _FF_MAJOR = 7
    return ["-/filter_complex", path] if _FF_MAJOR >= 7 else ["-filter_complex_script", path]


_FF_MAJOR = None


def run(args, cwd=None):
    print(" ".join(map(str, args)), file=sys.stderr)
    return subprocess.run(args, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=cwd,
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))  # no console flash under pythonw


IMAGES = {".png", ".jpg", ".jpeg", ".webp"}


def probe(path):
    out = run([FP, "-v", "error", "-show_entries", "format=duration:stream=codec_type,codec_name,width,height,r_frame_rate",
               "-of", "json", path]).stdout
    d = json.loads(out)
    v = next((s for s in d["streams"] if s["codec_type"] == "video"), None)
    a = next((s for s in d["streams"] if s["codec_type"] == "audio"), None)
    return {"dur": float(d["format"].get("duration") or 0), "video": v, "audio": a}  # images have no duration


# ---------- scenes (CapCut: compress_shot_detect, threshold 0.35) ----------
def scenes(inp, threshold=0.35):
    # ponytail: ffmpeg's scene score instead of CapCut's 96x96 CNN; swap in PySceneDetect/TransNet if cuts are missed
    err = run([FF, "-hide_banner", "-i", inp, "-vf", f"scale=96:96,select='gt(scene,{threshold})',showinfo",
               "-an", "-f", "null", "-"]).stderr
    return [float(t) for t in re.findall(r"pts_time:([\d.]+)", err)]


def split_at(inp, cuts, outdir):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    bounds = [0.0, *cuts, probe(inp)["dur"]]
    for i, (s, e) in enumerate(zip(bounds, bounds[1:])):
        run([FF, "-y", "-ss", f"{s}", "-to", f"{e}", "-i", inp, f"{outdir}/clip_{i:03d}{Path(inp).suffix}"])


# ---------- search cut (CapCut: TranscriptEdit delete words -> cut ranges) ----------
def parse_srt(text):
    ts = lambda h, m, s, ms: int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000
    cues = []
    for block in re.split(r"\n\s*\n", text.strip().replace("\r", "")):
        m = re.search(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)", block)
        if m:
            body = block[m.end():].strip()
            cues.append((ts(*m.groups()[:4]), ts(*m.groups()[4:]), re.sub(r"<[^>]+>", "", body)))
    return cues


def merge(ranges):
    out = []
    for s, e in sorted(ranges):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [tuple(r) for r in out]


def keep_ranges(cues, query, total, keep=False, pad=0.0):
    hits = merge((max(0, s - pad), min(total, e + pad)) for s, e, t in cues if re.search(query, t, re.I))
    if keep:
        return hits
    out, pos = [], 0.0
    for s, e in hits:
        if s > pos:
            out.append((pos, s))
        pos = e
    if pos < total:
        out.append((pos, total))
    return out


def render_ranges(inp, ranges, out):
    if not ranges:
        sys.exit("nothing left to render")
    has_a = probe(inp)["audio"] is not None
    parts, labels = [], ""
    for i, (s, e) in enumerate(ranges):
        parts.append(f"[0:v]trim={s}:{e},setpts=PTS-STARTPTS[v{i}]")
        labels += f"[v{i}]"
        if has_a:
            parts.append(f"[0:a]atrim={s}:{e},asetpts=PTS-STARTPTS[a{i}]")
            labels += f"[a{i}]"
    parts.append(f"{labels}concat=n={len(ranges)}:v=1:a={int(has_a)}[v]" + ("[a]" if has_a else ""))
    run([FF, "-y", "-i", inp, "-filter_complex", ";".join(parts), "-map", "[v]", *(["-map", "[a]"] if has_a else []), out])


# ---------- transcription (CapCut: ASR -> TranscriptEdit word ranges) ----------
_whisper, _whisper_lock = None, __import__("threading").Lock()


def transcribe(path, model="small"):
    """Word-level timestamps via faster-whisper (local, CPU int8)."""
    global _whisper
    from faster_whisper import WhisperModel
    with _whisper_lock:  # one model, one job at a time
        if _whisper is None:
            # ponytail: CPU int8; device="cuda" needs the cuBLAS/cuDNN wheels, add them if long clips are too slow
            _whisper = WhisperModel(model, device="cpu", compute_type="int8")
        # decode with ffmpeg ourselves: faster-whisper's own decoder breaks on av versions without metadata_errors (2026-10-21)
        import numpy as np
        pcm = subprocess.run([FF, "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
                             capture_output=True, check=True).stdout
        segs, info = _whisper.transcribe(np.frombuffer(pcm, np.float32), word_timestamps=True, vad_filter=True)
        words = [{"s": round(w.start, 3), "e": round(w.end, 3), "w": w.word.strip()} for s in segs for w in s.words]
    return {"language": info.language, "words": words}


# ---------- audio extraction (CapCut: CommonMediaRunner extract_audio) ----------
CODECS = {".mp3": "libmp3lame", ".m4a": "aac", ".aac": "aac", ".wav": "pcm_s16le", ".flac": "flac", ".ogg": "libvorbis", ".opus": "libopus"}
COPY_OK = {".m4a": {"aac", "alac"}, ".aac": {"aac"}, ".mp3": {"mp3"}, ".flac": {"flac"}, ".opus": {"opus"}, ".ogg": {"vorbis", "opus"}}


def extract_audio(inp, out):
    ext = Path(out).suffix.lower()
    if ext not in CODECS:
        sys.exit(f"unsupported audio suffix {ext}; use one of {', '.join(CODECS)}")
    a = probe(inp)["audio"]
    if not a:
        sys.exit("input has no audio stream")
    codec = "copy" if a["codec_name"] in COPY_OK.get(ext, ()) else CODECS[ext]  # lossless stream copy when possible
    run([FF, "-y", "-i", inp, "-vn", "-map", "0:a:0", "-c:a", codec, out])


# ---------- rotate (CapCut: RotateSegment / clip.rotation) ----------
def rotate(inp, out, deg):
    deg %= 360
    vf = {90: "transpose=1", 180: "transpose=1,transpose=1", 270: "transpose=2"}.get(deg)
    if vf is None:  # free angle: CapCut keeps canvas size, so do we
        vf = f"rotate={deg}*PI/180:fillcolor=black"
    run([FF, "-y", "-i", inp, "-vf", vf, "-c:a", "copy", out] if deg else [FF, "-y", "-i", inp, "-c", "copy", out])


# ---------- combine + transitions (CapCut: combineSegment + transitions[is_overlap]) ----------
#   Project = {canvas:[w,h]|None, fps, video:[segment], overlay:[overlay], audio:[audio], text:[text]}  (seconds)
#   Project also: fill = contain|cover|blur (how clips fill the canvas), filter = look preset for the whole video
#   segment = {media, in, out, speed, reverse, freeze, dur, rotation, flip, volume, fadeIn, fadeOut, transition:{type, dur},
#              zoom, zoomTo, shake, filter}   zoom = punch-in factor (1 = none), zoomTo = animate to this factor over the clip,
#              shake = 0..1 camera jitter, filter = look preset (see FILTERS)
#             transition = into the NEXT segment (CapCut: transition material referenced by the outgoing segment)
#             freeze = still of frame `in` held for `dur` seconds (CapCut FreezeInfo)
#   overlay = {media, start, in, out, speed, x, y, scale, rotation, flip, volume}   x,y = centre in canvas fractions,
#             scale = width as canvas fraction; alpha (VP9 WebM cut-outs) is kept
#   audio   = {media, start, in, out, speed, volume, fadeIn, fadeOut}
#   text    = {text, start, dur, x, y, size, color, box}                          size = font px / canvas height
#   ponytail: seconds as floats, not CapCut's int64 microseconds; fine until frame-exact multi-hour timelines matter
def fades(x, length):
    fi, fo = min(x.get("fadeIn", 0), length), min(x.get("fadeOut", 0), length)
    return (f",afade=t=in:d={fi:.3f}" if fi else "") + (f",afade=t=out:st={length - fo:.3f}:d={fo:.3f}" if fo else "")


def seg_len(s):
    if len(s.get("speedCurve") or []) > 1 and not s.get("freeze"):
        return sum(seg_len(p) for p in expand_speed(s))
    return s["dur"] if s.get("freeze") else (s["out"] - s["in"]) / s.get("speed", 1)


def geometry(s):
    rot = s.get("rotation", 0) % 360
    geo = ROT.get(rot) if rot in ROT else f"rotate={rot}*PI/180:ow=rotw({rot}*PI/180):oh=roth({rot}*PI/180):c=none,"
    c = s.get("crop")  # CapCut Crop: {x, y, w, h} fractions of the source frame
    crop = f"crop=iw*{c.get('w', 1):.4f}:ih*{c.get('h', 1):.4f}:iw*{c.get('x', 0):.4f}:ih*{c.get('y', 0):.4f}," if c and (c.get("w", 1) < 1 or c.get("h", 1) < 1) else ""
    return crop + geo + ("hflip," if s.get("flip") else "")


# ---------- CapCut parity 2026-10-16: Adjust, effects, chroma key, stabilize, audio fx, export settings ----------
ADJ = {"bri": 0, "con": 0, "sat": 0, "temp": 0, "sharp": 0, "vig": 0, "grain": 0}  # CapCut "Adjust" sliders, -1..1, 0 = untouched


def adjust_chain(a):
    """ffmpeg chain for an `adjust` dict (segment or whole project)."""
    a = {**ADJ, **{k: float(v or 0) for k, v in (a or {}).items()}}
    out = ""
    if a["bri"] or a["con"] or a["sat"]:
        out += f"eq=brightness={a['bri'] * 0.4:.3f}:contrast={1 + a['con']:.3f}:saturation={max(0, 1 + a['sat']):.3f},"
    if a["temp"]:  # warm = lower kelvin white point (ffmpeg colortemperature: 6500 neutral, 3500 = orange, 9500 = blue; checked 2026-10-16)
        out += f"colortemperature=temperature={round(6500 - 2800 * a['temp'])}:mix=1,"
    if a["sharp"] > 0:
        out += f"unsharp=5:5:{a['sharp'] * 1.5:.2f},"
    if a["vig"] > 0:
        out += f"vignette=angle={a['vig'] * 1.2:.3f},"
    if a["grain"] > 0:
        out += f"noise=alls={round(a['grain'] * 40)}:allf=t,"
    return out


# CapCut "Effects" library, one per clip (`effect`), after zoom/shake/look. mirror uses labels: still a valid fragment inside the chain.
EFFECTS_FF = {
    "glitch": "rgbashift=rh=6:bh=-6:gv=2,noise=alls=18:allf=t,",
    "rgbsplit": "rgbashift=rh=10:bh=-10,",
    "pixelate": "pixelize=w=24:h=24,",
    "blur": "gblur=sigma=12,",
    "mirror": "crop=iw/2:ih:0:0,split[ml][mr];[mr]hflip[mrf];[ml][mrf]hstack,",
    "strobe": "eq=brightness='0.4*lt(mod(n,6),2)':eval=frame,",
    "invert": "negate,",
}


def afx(x):
    """Audio extras on a clip / audio item: denoise (CapCut noise reduction) and pitch in semitones (voice changer)."""
    out = ",afftdn=nf=-25" if x.get("denoise") else ""
    if x.get("pitch"):
        out += f",rubberband=pitch={2 ** (float(x['pitch']) / 12):.4f}"
    return out


def stabilized(src, a, b, media_dir):
    """CapCut Stabilize: vidstab two-pass over source seconds a..b -> media/_stab_<hash>.mp4 (cached)."""
    import hashlib
    src = Path(src)
    key = hashlib.md5(f"{src.name}|{src.stat().st_mtime}|{a}|{b}".encode()).hexdigest()[:12]
    out = Path(media_dir) / f"_stab_{key}.mp4"
    if out.exists():
        return out
    with tempfile.TemporaryDirectory() as tmp:  # trf file is cwd-relative: no drive-letter colon to escape in the filtergraph
        cut = ["-ss", f"{a}", "-t", f"{b - a}", "-i", str(src)]
        run([FF, "-y", *cut, "-vf", "vidstabdetect=shakiness=6:accuracy=15:result=trf", "-an", "-f", "null", "-"], cwd=tmp)
        run([FF, "-y", *cut, "-vf", "vidstabtransform=input=trf:smoothing=20:zoom=4:interpol=bicubic,unsharp=5:5:0.6",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "17", "-c:a", "aac", str(out)], cwd=tmp)
    return out


# CapCut "filters"/"effects" as ffmpeg chains. Names are what the UI and trends.py use.
FILTERS = {
    "none": "", "phonk": "eq=contrast=1.35:saturation=0.75:brightness=-0.03,unsharp=5:5:1.2,noise=alls=10:allf=t",
    "bw": "hue=s=0,eq=contrast=1.3", "vintage": "curves=preset=vintage,noise=alls=8:allf=t",
    "cool": "colorbalance=bs=0.18:rs=-0.05,eq=contrast=1.1", "warm": "colorbalance=rs=0.12:bs=-0.1,eq=saturation=1.15",
    "vhs": "noise=alls=22:allf=t,eq=saturation=1.3:contrast=0.95,unsharp=3:3:0.6", "aura": "eq=contrast=1.4:saturation=0.5:gamma=0.9,vignette=PI/4",
    "flash": "eq=brightness=0.35", "neg": "negate",
    "dark": "curves=all='0/0 0.4/0.27 1/0.93',eq=saturation=0.55:contrast=1.2,vignette=PI/3.5",  # CapCut "dark"/moody: crushed shadows, desaturated, vignette
    # CapCut's own filter LUTs (REA on the effect cache, luts.py converts the 512x512 lookup PNGs -> .cube; render copies luts/ into the graph dir)
    "capcut_bw": "lut3d=capcut_a.cube", "capcut_matte": "lut3d=capcut_b.cube",
    # 2026-10-20 from the user's own CapCut drafts: Silber + Demut (SkinFilter bg LUT; the skin LUT needs CapCut's skin segmentation,
    # ponytail: background LUT on everything), Archaic Terror + Finsteres Siegel (32^3 float cube from texture/filter.cube.vf)
    "capcut_silber": "lut3d=capcut_silber.cube", "capcut_demut": "lut3d=capcut_demut.cube",
    "capcut_terror": "lut3d=capcut_terror.cube", "capcut_siegel": "lut3d=capcut_siegel.cube",
    # dark edit looks: crushed shadows + tint, like CapCut's Noir / Moody / Dusk / Grunge filters
    "noir": "curves=all='0/0 0.25/0.12 0.6/0.55 1/0.95',eq=saturation=0.35:contrast=1.25,colorbalance=bs=0.08:rs=-0.06,vignette=PI/3.2",
    "moody": "curves=all='0/0 0.3/0.18 1/0.92',colorbalance=bs=0.1:rs=-0.06:rh=0.08:bh=-0.05,eq=saturation=0.8:contrast=1.15,vignette=PI/3.5",
    "dusk": "eq=brightness=-0.08:contrast=1.3:saturation=0.7,colorbalance=bs=0.2:rs=-0.1:bm=0.08,vignette=PI/3",
    "grunge": "curves=all='0/0 0.35/0.2 1/0.95',eq=contrast=1.45:saturation=0.6,noise=alls=16:allf=t,unsharp=5:5:1.0,vignette=PI/3.5",
    # professional / film looks (school & documentary videos): subtle, keep skin natural
    "kino": "curves=all='0/0.03 0.25/0.21 0.75/0.79 1/0.97',colorbalance=bs=0.06:rs=-0.03:rh=0.05:bh=-0.04,eq=saturation=0.92:contrast=1.06,vignette=PI/5",
    "teal_orange": "colorbalance=rs=-0.06:bs=0.1:rm=0.03:bm=-0.02:rh=0.08:gh=0.02:bh=-0.08,eq=saturation=1.05:contrast=1.08",
    "film_warm": "curves=all='0/0.04 0.5/0.52 1/0.96',colorbalance=rs=0.06:gs=0.02:bs=-0.05:rh=0.04:bh=-0.05,eq=saturation=0.95,noise=alls=4:allf=t",
    "clean": "eq=contrast=1.06:saturation=1.08:brightness=0.01,unsharp=5:5:0.4",
    "soft": "curves=all='0/0.05 1/0.97',eq=contrast=0.96:saturation=0.9,gblur=sigma=0.4",
    "doku_bw": "hue=s=0,curves=all='0/0.02 0.3/0.26 0.7/0.76 1/0.98',noise=alls=5:allf=t",
}
LUTS = Path(getattr(sys, "_MEIPASS", Path(__file__).parent)) / "luts"
FONTS = {"arial": "arialbd.ttf", "impact": "impact.ttf", "georgia": "georgiab.ttf", "comic": "comicbd.ttf", "times": "timesbd.ttf"}
# professional fonts shipped in fonts/ (SIL OFL, from Google Fonts via fontsource): same file on Windows, Mac and Linux
FONTS_BUNDLED = {"montserrat": "Montserrat-SemiBold.ttf", "playfair": "PlayfairDisplay-Bold.ttf"}
# macOS: /System/Library/Fonts/Supplemental has the same faces under their full names
FONTS_MAC = {"arial": "Arial Bold.ttf", "impact": "Impact.ttf", "georgia": "Georgia Bold.ttf", "comic": "Comic Sans MS Bold.ttf", "times": "Times New Roman Bold.ttf"}
# Linux (VPS) has no Windows fonts: fonts/ in the repo (Anton = Impact look-alike, SIL OFL) and the DejaVu system fonts stand in.
# Without this, drawtext silently fell back to plain DejaVu Sans and every caption looked generic (user 2026-10-21).
FONTS_LINUX = {"impact": "Anton-Regular.ttf", "arial": "DejaVuSans-Bold.ttf", "georgia": "DejaVuSerif-Bold.ttf", "comic": "Anton-Regular.ttf", "times": "DejaVuSerif-Bold.ttf"}
FONT_DIRS = [Path(getattr(sys, "_MEIPASS", Path(__file__).parent)) / "fonts", Path("/System/Library/Fonts/Supplemental"), Path("/Library/Fonts"),
             Path("/usr/share/fonts/truetype/dejavu"), Path("/usr/share/fonts/truetype/liberation")]


def font_file(name):
    """drawtext fontfile= value: /Windows/Fonts/<ttf> (drive-relative, no ':' to escape) on Windows, else the first match in FONT_DIRS."""
    if name in FONTS_BUNDLED:
        p = FONT_DIRS[0] / FONTS_BUNDLED[name]
        if os.name == "nt":  # drive-relative like /Windows/Fonts: no ':' to escape in the filtergraph
            return p.resolve().as_posix()[2:] if p.resolve().drive.lower() == Path.cwd().drive.lower() else p.resolve().as_posix().replace(":", "\\:")
        return str(p)
    name = name if name in FONTS else "arial"
    if os.name == "nt" or Path("/Windows/Fonts", FONTS[name]).exists():
        return "/Windows/Fonts/" + FONTS[name]
    for cand in (FONTS_MAC.get(name), FONTS_LINUX.get(name), FONTS[name]):
        for d in FONT_DIRS:
            if cand and (d / cand).exists():
                return str(d / cand)
    return None  # drawtext picks fontconfig's default sans


def face_on_canvas(s, fill, W, H, sw, sh):
    """Where the segment's crop centre cx/cy (source fractions = his face) lands on the canvas, as fractions. (0.5, 0.5) when unknown."""
    if fill != "cover" or s.get("cx") is None or not sw or not sh:
        return 0.5, 0.5
    if s.get("rotation", 0) % 180 == 90:
        sw, sh = sh, sw
    k = max(W / sw, H / sh)
    out = []
    for c, src, dst in ((float(s.get("cx", 0.5)), sw * k, W), (float(s.get("cy", 0.5)), sh * k, H)):
        crop = min(max(src * c - dst / 2, 0), src - dst)
        out.append(round(min(max((src * c - crop) / dst, 0), 1), 4))
    return out[0], out[1]


def fx(s, W, H, D, face=(0.5, 0.5)):
    """Effects after the clip is on the canvas: zoom punch / animated zoom (around his face, clamped to the frame), shake, look preset. D = clip length."""
    out, z0, z1 = "", s.get("zoom", 1) or 1, s.get("zoomTo")
    at = f"x='clip(iw*{face[0]}-{W}/2,0,iw-{W})':y='clip(ih*{face[1]}-{H}/2,0,ih-{H})'"
    if z1 and abs(z1 - z0) > 1e-3:  # animated zoom: scale per frame, crop back to the canvas
        out += f"scale=eval=frame:w='iw*({z0}+({z1}-{z0})*min(t/{D:.4f},1))':h=-2,crop={W}:{H}:{at},"
    elif z0 != 1:
        out += f"scale=iw*{z0}:-2,crop={W}:{H}:{at},"
    if s.get("shake"):
        a = max(1, round(s["shake"] * H * 0.025))
        out += f"scale=iw+{2 * a}:-2,crop={W}:{H}:x='(iw-ow)/2+{a}*exp(-n/8)*sin(n*12.9)':y='(ih-oh)/2+{a}*exp(-n/8)*cos(n*9.7)',"  # impulse on the cut, decays in ~0.3 s (user: shake read as late)
    if FILTERS.get(s.get("filter") or "none"):
        out += FILTERS[s["filter"]] + ","
    out += adjust_chain(s.get("adjust")) + EFFECTS_FF.get(s.get("effect") or "", "")
    return out


def fit_chain(fill, W, H, s=None):
    if fill == "cover":  # TikTok/Reels: fill the frame, crop the sides around the segment's crop centre cx/cy (source fractions, default 0.5)
        cx, cy = (min(1, max(0, float((s or {}).get(k) or 0.5))) for k in ("cx", "cy"))
        return f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H}:clip(iw*{cx:.4f}-{W}/2\\,0\\,iw-{W}):clip(ih*{cy:.4f}-{H}/2\\,0\\,ih-{H}),"
    return f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,"


# ---------- captions: keyword highlight (CapCut "highlight" style, user 2026-10-21: captions must not look generic) ----------
#   text.hl = "DÖNER" | ["DÖNER", "SÖDER"]: those words get hlcolor (default gold) and hlscale (default 1.2 x) inside the caption.
#   Every word is then its own drawtext, laid out with the font's real advance widths (Pillow); without Pillow the widths are estimated.
#   text.sticker = {media (transparent webm, e.g. "Döner Emoji.webm"), word, scale (x word height, 1.1), rot (deg, -15)}: an emoji that sits
#   right after that word; the layout reserves the slot, render + preview put the webm there (pops in with the caption).
HL_COLOR, HL_SCALE, LINE_H, STICKER_SCALE, STICKER_ROT = "#ffd60a", 1.2, 1.15, 1.1, -15
_PUNCT = ".,!?;:\"'()"


def hl_words(t):
    hl = t.get("hl") or []
    if isinstance(hl, str):
        hl = hl.replace(",", " ").split()
    return {w.strip(_PUNCT).casefold() for w in hl if w.strip(_PUNCT)}


def _font(name, size):
    try:
        from PIL import ImageFont
        f = font_file(name or "arial")
        return ImageFont.truetype(f, size) if f else None
    except Exception:
        return None


def text_layout(t, H):
    """Per-word layout: (lines, size) with lines = [[{w, x0, wd, size, hl}]], x0/wd in px relative to the line centre at the
    item's base size (text animations scale the whole caption around that centre). Used by the renderer, `word_box` and the UI."""
    size = max(8, round(t.get("size", 0.06) * H))
    hs, hl = max(size, round(size * float(t.get("hlscale") or HL_SCALE))), hl_words(t)
    fonts = {s: _font(t.get("font"), s) for s in (size, hs)}
    width = lambda s, px: fonts[px].getlength(s) if fonts[px] else 0.55 * px * len(s)
    st, placed, lines = t.get("sticker") or {}, False, []
    for ln in t["text"].split("\n"):
        words, x, sp = [], 0.0, width(" ", size)
        for w in ln.split(" "):
            is_hl = bool(hl) and w.strip(_PUNCT).casefold() in hl
            px = hs if is_hl else size
            wd = width(w, px)
            words.append({"w": w, "x0": x, "wd": wd, "size": px, "hl": is_hl})
            x += wd + sp
            if st.get("media") and not placed and w.strip(_PUNCT).casefold() == str(st.get("word", "")).strip(_PUNCT).casefold():
                spx = round(px * float(st.get("scale") or STICKER_SCALE))  # the sticker takes a word slot right after its word
                words.append({"w": "", "x0": x, "wd": spx, "size": spx, "hl": False, "sticker": True})
                x, placed = x + spx + sp, True
        total = x - sp
        for d in words:
            d["x0"] -= total / 2
        lines.append(words)
    return lines, size


def word_box(t, W, H, word):
    """Canvas-fraction box {x, y, w, h} (centre + size) of the first `word` of text item t at full size; None when absent.
    Lets stickers sit next to a caption word (`op sticker`)."""
    lines, size = text_layout(t, H)
    for li, ws in enumerate(lines):
        for d in ws:
            if d["w"].strip(_PUNCT).casefold() == word.strip(_PUNCT).casefold():
                cy = float(t.get("y", 0.85)) * H + (li - (len(lines) - 1) / 2) * LINE_H * size
                cx = float(t.get("x", 0.5)) * W + d["x0"] + d["wd"] / 2
                return {"x": round(cx / W, 4), "y": round(cy / H, 4), "w": round(d["wd"] / W, 4), "h": round(d["size"] / H, 4)}
    return None


def sticker_overlay(t, W, H):
    """The overlay item (render + UI) for text item t's `sticker`: its webm in the slot after the word, popping in with a pop-in caption."""
    st = t.get("sticker") or {}
    lines, size = text_layout(t, H)
    for li, ws in enumerate(lines):
        for d in ws:
            if d.get("sticker"):
                cy = float(t.get("y", 0.85)) * H + (li - (len(lines) - 1) / 2) * LINE_H * size
                cx = float(t.get("x", 0.5)) * W + d["x0"] + d["wd"] / 2
                o = {"id": f"{t.get('id', 't')}_st", "media": st["media"], "start": t["start"], "in": 0, "out": round(t["dur"], 3), "speed": 1,
                     "x": round(cx / W, 4), "y": round(cy / H, 4), "scale": round(d["wd"] / W, 4), "rotation": float(st.get("rot", STICKER_ROT)),
                     "flip": False, "volume": 0, "opacity": 1, "_sticker": t.get("id")}
                if (t.get("anim_in") or {}).get("type") == "pop":
                    o["keyframes"] = [{"t": 0, "scale": 0.3, "opacity": 0}, {"t": 0.1, "scale": 1.15, "opacity": 1}, {"t": 0.18, "scale": 1, "opacity": 1}]
                return o
    return None


def _dt(t, font, textfile, size, color, x, y, alpha):
    outline = t.get("outline", (t.get("font") or "arial") not in FONTS_BUNDLED)  # pro fonts: clean, no meme outline (set outline: true to force)
    look = (f":box=1:boxcolor={t.get('boxcolor') or 'black'}@0.55:boxborderw={size // 4}" if t.get("box")
            else f":borderw={max(1, round(size / 22))}:bordercolor=black@0.35" if t.get("soft")  # broadcast captions: hairline + soft shadow
            else f":borderw={max(3, round(size / 14))}:bordercolor=black@0.9" if outline else "")
    if t.get("soft"):
        look += f":shadowx=0:shadowy={max(1, round(size / 16))}:shadowcolor=black@0.55"  # outline scales with the size (CapCut)
    if t.get("shadow"):
        look += f":shadowx={max(1, size // 14)}:shadowy={max(1, size // 14)}:shadowcolor=black@0.7"
    fsize, alpha_e, dx, dy, _ = text_anim(t, size)
    op = float(t.get("opacity", 1) if t.get("opacity") is not None else 1)
    if op < 1:  # slightly transparent text (fill, outline and shadow together)
        alpha_e = f"'{op:.3f}*{alpha_e.strip(chr(39))}'" if alpha_e else f"{op:.3f}"
    return (f"drawtext={'fontfile=' + font if font else 'font=sans'}:textfile={textfile}:expansion=none:fontsize={fsize}"
            f":fontcolor={color}{look}:x='{x}{dx}':y='{y}{dy}'" + (f":alpha={alpha_e}" if alpha_e else "")
            + f":enable='between(t,{t['start']},{t['start'] + t['dur']})'")


def drawtext(t, H, emit=None):
    """drawtext filter(s) for text item t (t['_i'] = index). emit(name, content) stores the text files: textfile= keeps ':' and
    quotes out of the graph. With `hl` every word is drawn on its own so the keyword can differ in colour and size."""
    emit = emit or (lambda n, c: None)
    size = max(8, round(t.get("size", 0.06) * H))
    # fontfile is drive-relative and textfile cwd-relative, so the filtergraph never contains a ':' path to escape
    font, i = font_file(t.get("font") or "arial"), t["_i"]
    if not hl_words(t) and not (t.get("sticker") or {}).get("media"):
        emit(f"t{i}.txt", t["text"])
        xe = f"{t.get('x', 0.5)}*w" + ("" if t.get("align") == "left" else "-text_w/2")  # align left: x = left edge (lower thirds)
        return _dt(t, font, f"t{i}.txt", size, t.get("color", "#ffffff"), xe, f"{t.get('y', 0.85)}*h-text_h/2", None)
    lines, _ = text_layout(t, H)
    k, out = text_anim(t, size)[4], []
    for li, ws in enumerate(lines):
        ly = (li - (len(lines) - 1) / 2) * LINE_H * size
        for j, d in enumerate(ws):
            if not d["w"]:
                continue
            emit(f"t{i}_{li}_{j}.txt", d["w"])
            out.append(_dt(t, font, f"t{i}_{li}_{j}.txt", d["size"], (t.get("hlcolor") or HL_COLOR) if d["hl"] else t.get("color", "#ffffff"),
                           f"{t.get('x', 0.5)}*w+({d['x0']:.1f})*{k}", f"{t.get('y', 0.85)}*h+({ly:.1f})*{k}-text_h/2", None))
    return ",".join(out)


# ---------- CapCut parity for "edit" Reels: keyframes, curve speed, masks, text animations ----------
#   keyframes  = [{t, scale, x, y, rotation, opacity}]  on segments and overlays; t = seconds after the item starts
#                (timeline seconds). scale multiplies the clip's zoom/scale, x/y shift it (canvas fraction),
#                rotation adds degrees, opacity is absolute 0..1. Linear interpolation, held outside the first/last key.
#   opacity    = static 0..1 on segments/overlays (keyframes override it)
#   speedCurve = [{t, speed}] t = 0..1 of the source range in..out. Rendered as constant-speed pieces (CapCut "Curve").
#   mask       = {type: rect|circle|linear|mirror, x, y, w, h, feather, invert} on overlays, fractions of the overlay frame
#   text.anim_in / anim_out = {type: fade|pop|typewriter|slide|shake, dur}
KF_DEFAULT = {"scale": 1, "x": 0, "y": 0, "rotation": 0, "opacity": 1}
CURVES = {  # CapCut curve-speed presets, speeds at evenly spaced points of the clip
    "montage": [1, 1, 2.6, 0.4, 2.6, 1, 1], "hero": [1, 1.8, 0.3, 0.3, 1.8, 1], "bullet": [1, 1, 4, 0.15, 4, 1, 1],
    "jumpcut": [1, 0.4, 3, 0.4, 3, 0.4, 1], "flashin": [5, 2, 1, 1, 1], "flashout": [1, 1, 1, 2, 5],
}


def pl_expr(keys, tvar="t"):
    """ffmpeg expression, piecewise linear through [(t, v)] in `tvar`, held outside the first/last key."""
    keys = sorted(keys)
    expr = f"{keys[-1][1]}"
    for (t0, v0), (t1, v1) in reversed(list(zip(keys, keys[1:]))):
        expr = f"if(lt({tvar},{t1}),{v0}+({v1}-{v0})*clip(({tvar}-{t0})/{max(t1 - t0, 1e-6)},0,1),{expr})"
    return expr


def kf_expr(s, field, tvar="t"):
    """ffmpeg expression (piecewise linear in `tvar`) for one keyframe field, or None when it stays at the default."""
    d = KF_DEFAULT[field]
    keys = sorted((float(k["t"]), float(k[field])) for k in s.get("keyframes") or [] if field in k)
    if not keys:
        v = s.get("opacity", 1) if field == "opacity" else d
        return None if v == d else f"{v}"
    return pl_expr(keys, tvar)


def eye_glow(i, inp, k, W, H):
    """Laser eyes: overlay the generated RGBA light layer (input k, quarter res, see fxface.glow_video) onto [inp] -> [v{i}]."""
    return [f"[{k}:v]format=rgba,scale={W}:{H}:flags=bicubic,format=yuva420p,settb=AVTB[g{i}]",
            f"[{inp}][g{i}]overlay=format=auto:shortest=1,format=yuv420p,settb=AVTB[v{i}]"]


def kf_main(i, s, W, H, out=None):
    """Filter lines for keyframes on main-track clip i: [vp{i}] (canvas-sized) -> transformed, composited on black -> [v{i}]."""
    S, X, Y, R = (kf_expr(s, f) for f in ("scale", "x", "y", "rotation"))
    O = kf_expr(s, "opacity", "T")
    chain = "format=yuva420p,"
    if O:  # ponytail: geq is per-pixel (slow-ish); only used when opacity actually animates. MUST run before the
        chain += f"geq=lum='lum(X,Y)':cb='cb(X,Y)':cr='cr(X,Y)':a='alpha(X,Y)*({O})',"  # per-frame scale: geq crashes on size changes
    if R:
        chain += f"rotate=a='({R})*PI/180':c=black@0,"
    if S:
        chain += f"scale=eval=frame:w='{W}*({S})':h='{H}*({S})',"
    return [f"[vp{i}]split[kb{i}][kf{i}]", f"[kb{i}]drawbox=c=black:t=fill[kbb{i}]", f"[kf{i}]{chain}format=yuva420p[kff{i}]",
            f"[kbb{i}][kff{i}]overlay=x='(W-{W}*({S or 1}))/2+({X or 0})*W':y='(H-{H}*({S or 1}))/2+({Y or 0})*H',format=yuv420p[{out or f'v{i}'}]"]


def mask_expr(m):
    """Alpha 0..1 expression for a CapCut-style mask in normalized frame coordinates (X/W, Y/H)."""
    typ, x, y, w, h = m.get("type", "rect"), m.get("x", 0.5), m.get("y", 0.5), m.get("w", 0.5), m.get("h", 0.5)
    fe, u, v = max(float(m.get("feather", 0) or 0), 0.002), "(X/W)", "(Y/H)"
    if typ == "circle":
        e = f"clip((1-hypot(({u}-{x})/({w}/2),({v}-{y})/({h}/2)))*{w}/2/{fe},0,1)"
    elif typ == "linear":  # everything below the line y, soft edge = feather
        e = f"clip(({v}-{y})/{fe}+0.5,0,1)"
    elif typ == "mirror":  # band between two horizontal lines
        e = f"clip(({h}/2-abs({v}-{y}))/{fe},0,1)"
    else:
        e = f"clip(({w}/2-abs({u}-{x}))/{fe},0,1)*clip(({h}/2-abs({v}-{y}))/{fe},0,1)"
    return f"(1-{e})" if m.get("invert") else e


def pip_chain(o, ow, oh, st, W, H):
    """Overlay (PiP) extras: mask + keyframes. Returns (filter chain to append after the static geometry, x expr, y expr)."""
    tv, chain = f"(t-{st})", ""
    S, X, Y, R = (kf_expr(o, f, tv) for f in ("scale", "x", "y", "rotation"))
    O = kf_expr(o, "opacity", f"(T-{st})")
    ck = o.get("chroma")  # CapCut chroma key: {color, similarity, blend}
    if ck and ck.get("color"):
        chain += f"chromakey=0x{str(ck['color']).lstrip('#')}:{float(ck.get('similarity', 0.3)):.3f}:{float(ck.get('blend', 0.1)):.3f},"
    alpha = [a for a in ([mask_expr(o["mask"])] if o.get("mask") and o["mask"].get("type", "rect") != "none" else []) + ([f"({O})"] if O else [])]
    if alpha:
        chain += f"geq=lum='lum(X,Y)':cb='cb(X,Y)':cr='cr(X,Y)':a='alpha(X,Y)*{'*'.join(alpha)}',"
    bw, bh = ow, oh
    if R:  # fixed square box so the frame size doesn't change with the angle
        bw = bh = int((ow * ow + oh * oh) ** 0.5) // 2 * 2 + 2
        chain += f"rotate=a='({R})*PI/180':ow={bw}:oh={bh}:c=none,"
    if S:
        chain += f"scale=eval=frame:w='iw*({S})':h='ih*({S})',"
    cx, cy = f"{o.get('x', 0.5)}*W+({X or 0})*W", f"{o.get('y', 0.5)}*H+({Y or 0})*H"
    return chain, f"'{cx}-{bw}*({S or 1})/2'", f"'{cy}-{bh}*({S or 1})/2'"


def expand_speed(s, steps=3):
    """speedCurve -> list of constant-speed pieces (speed ramps approximated with `steps` pieces per curve interval)."""
    pts = sorted((float(p["t"]), float(p["speed"])) for p in s.get("speedCurve") or [])
    if len(pts) < 2 or s.get("freeze"):
        return [s]
    if pts[0][0] > 0:
        pts.insert(0, (0.0, pts[0][1]))
    if pts[-1][0] < 1:
        pts.append((1.0, pts[-1][1]))
    a, D = s["in"], s["out"] - s["in"]
    pieces, off = [], 0.0
    for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
        for k in range(steps):
            f0, f1 = t0 + (t1 - t0) * k / steps, t0 + (t1 - t0) * (k + 1) / steps
            if f1 - f0 < 1e-6:
                continue
            sp = max(0.05, (v0 + (v1 - v0) * (k + 0.5) / steps))
            p = {**s, "in": round(a + D * f0, 4), "out": round(a + D * f1, 4), "speed": round(sp, 4), "speedCurve": None,
                 "transition": {"type": "none", "dur": 0}, "fadeIn": 0, "fadeOut": 0}
            if s.get("keyframes"):  # each piece sees its own stretch of the keyframe timeline
                p["keyframes"] = [{**kf, "t": kf["t"] - off} for kf in s["keyframes"]]
            off += (p["out"] - p["in"]) / sp
            pieces.append(p)
    pieces[0]["fadeIn"], pieces[-1]["fadeOut"] = s.get("fadeIn", 0), s.get("fadeOut", 0)
    pieces[-1]["transition"] = s.get("transition") or {"type": "none", "dur": 0}
    if s.get("reverse"):
        pieces.reverse()
    return pieces


def text_anim(t, size):
    """drawtext expression pieces for anim_in/anim_out: (fontsize expr, alpha expr | None, x offset, y offset, scale factor expr)."""
    S, E = t["start"], t["start"] + t["dur"]
    ai, ao = t.get("anim_in") or {}, t.get("anim_out") or {}
    di, do = min(float(ai.get("dur", 0.4) or 0.4), t["dur"] / 2), min(float(ao.get("dur", 0.4) or 0.4), t["dur"] / 2)
    pi, po = f"clip((t-{S})/{di:.4f},0,1)", f"clip(({E}-t)/{do:.4f},0,1)"  # 0->1 while entering, 1->0 while leaving
    alpha, fs, dx, dy = [], [], "", ""
    for typ, p, sign in ((ai.get("type"), pi, 1), (ao.get("type"), po, -1)):
        if typ in ("fade", "slide") or (typ == "typewriter" and sign < 0):  # typewriter-out = fade (ponytail)
            alpha.append(p)
        if typ == "pop":
            fs.append(f"(0.25+0.75*{p})")
        if typ == "slide":
            dy += f"+({sign})*(1-{p})*h*0.08"
        if typ == "shake":
            dx += f"+(1-{p})*sin(t*90)*h*0.012"
    fsize = f"'{size}*{'*'.join(fs)}'" if fs else f"{size}"
    return fsize, (f"'{'*'.join(alpha)}'" if alpha else None), dx, dy, ("*".join(fs) or "1")  # last = scale factor expr (per-word layout)


def typewriter(t):
    """Expand a typewriter-in text item into prefix items that switch on one after the other."""
    if (t.get("anim_in") or {}).get("type") != "typewriter" or len(t["text"]) < 2:
        return [t]
    n, di = len(t["text"]), min(float(t["anim_in"].get("dur", 0.6) or 0.6), t["dur"] / 2)
    out, step = [], di / n
    for k in range(1, n + 1):
        start = t["start"] + (k - 1) * step
        end = t["start"] + k * step if k < n else t["start"] + t["dur"]
        out.append({**t, "text": t["text"][:k], "start": round(start, 4), "dur": round(end - start, 4),
                    "anim_in": None, "anim_out": t.get("anim_out") if k == n else None})
    return out


def joint_offsets(lens, ts):
    """Overlap model (CapCut is_overlap): a t-second transition makes both clips share t seconds.
    Returns (xfade offset per joint, total length)."""
    offs, total = [], lens[0]
    for d, t in zip(lens[1:], ts):
        offs.append(round(total - t, 3))
        total += d - t
    return offs, round(total, 3)


def atempo(speed):
    parts = []  # atempo takes 0.5..2 per instance; chain for the rest
    while speed > 2:
        parts.append("atempo=2"); speed /= 2
    while speed < 0.5:
        parts.append("atempo=0.5"); speed /= 0.5
    return ",".join(parts + [f"atempo={speed:.6f}"])


ROT = {0: "", 90: "transpose=1,", 180: "hflip,vflip,", 270: "transpose=2,"}


def render(project, media_dir, out):
    segs = [p for s in project["video"] for p in expand_speed(s)]  # curve speed -> constant-speed pieces
    if not segs:
        raise ValueError("timeline is empty")
    # server passes media_dir and only basenames are honoured (no escaping it); CLI passes None + real paths
    tmp_obj = tempfile.TemporaryDirectory(); tmp = tmp_obj.name  # graph file + pre-rendered transition clips
    path = (lambda m: str((Path(media_dir) / Path(m).name).resolve())) if media_dir else (lambda m: str(Path(m).resolve()))
    path = (lambda m, p=path: m if str(m).startswith(tmp) else p(m))
    overlays, texts = project.get("overlay", []), [t for t in project.get("text", []) if t.get("text", "").strip()]
    stick = [t for t in texts if (t.get("sticker") or {}).get("media")]  # caption stickers = overlays placed by the text layout
    names = {x["media"] for x in segs + overlays + project.get("audio", [])} | {t["sticker"]["media"] for t in stick}
    info = {m: probe(path(m)) for m in names}
    if project.get("canvas"):
        W, H = project["canvas"]
    else:
        v = info[segs[0]["media"]]["video"]
        W, H = (v["height"], v["width"]) if segs[0].get("rotation", 0) % 180 == 90 else (v["width"], v["height"])
    W, H, fps = W // 2 * 2, H // 2 * 2, project.get("fps", 30)
    overlays = overlays + [o for o in (sticker_overlay(t, W, H) for t in stick) if o]
    fill = project.get("fill", "contain")
    lens = [seg_len(s) for s in segs]
    # frame-exact lengths: clip i spans frames round(start*fps)..round(end*fps) of the cumulative timeline, so 19 clips of
    # 0.464 s do not each lose their fractional frame (that drifted every cut ~1 frame early and the whole edit 3 frames short)
    bounds, acc = [0], 0.0
    for L in lens:
        acc += L; bounds.append(round(acc * fps))
    exact = [f"tpad=stop_mode=clone:stop_duration=0.5,trim=end_frame={max(1, b - a)},setpts=PTS-STARTPTS," for a, b in zip(bounds, bounds[1:])]
    lens = [max(1, b - a) / fps for a, b in zip(bounds, bounds[1:])]
    ts = []
    for s, a, b in zip(segs, lens, lens[1:]):
        tr = s.get("transition") or {}
        n = round(min(tr.get("dur", 0), a / 2, b / 2) * fps) if tr.get("type", "none") != "none" else 0
        ts.append(n / fps if n >= 2 else 0)  # whole frames; a 1-frame xfade into a 2-frame shot ended the whole video chain early

    args, f, n_in = [FF, "-y"], [], 0

    def add_input(media):
        nonlocal n_in
        v = (info.get(media) or {}).get("video") or {}
        # ffmpeg's native vp9 decoder drops the alpha plane; libvpx keeps it (bg-removed cut-outs)
        still = ["-loop", "1", "-framerate", str(fps)] if Path(media).suffix.lower() in IMAGES else []  # logo / lower third PNG
        args.extend((["-c:v", "libvpx-vp9"] if v.get("codec_name") == "vp9" else []) + still + ["-i", path(media)])
        n_in += 1
        return n_in - 1

    for i, s in enumerate(segs):
        if s.get("stab") and not s.get("freeze"):  # stabilized copy of exactly this slice; everything else is relative to the shot start
            st = stabilized(path(s["media"]), s["in"], s["out"], media_dir or Path(s["media"]).parent)
            nm = st.name if media_dir else str(st)
            info[nm] = probe(str(st))
            s = {**s, "media": nm, "in": 0, "out": s["out"] - s["in"]}
        k = add_input(s["media"])
        vi = info[s["media"]]["video"] or {}
        sp, fit, post = s.get("speed", 1), fit_chain(fill, W, H, s), fx(s, W, H, lens[i], face_on_canvas(s, fill, W, H, vi.get("width"), vi.get("height")))
        cut = f"trim=start={s['in']}:duration=0.5" if s.get("freeze") else f"trim={s['in']}:{s['out']}"
        if fill == "blur":  # CapCut "blur background": blurred cover copy behind the contained clip
            f.append(f"[{k}:v]{cut},setpts=PTS-STARTPTS,{geometry(s)}split[bg{i}][fg{i}]")
            f.append(f"[bg{i}]{fit_chain('cover', W, H, s)}boxblur=24[bgb{i}]")
            f.append(f"[fg{i}]scale={W}:{H}:force_original_aspect_ratio=decrease[fgs{i}]")
            f.append(f"[bgb{i}][fgs{i}]overlay=(W-w)/2:(H-h)/2[fit{i}]")
            src, fit = f"[fit{i}]", ""
        else:
            src = f"[{k}:v]{cut},setpts=PTS-STARTPTS," + geometry(s)
            f.append(f"{src}{fit}".rstrip(",") + f"[fit{i}]"); src, fit = f"[fit{i}]", ""
        kf = s.get("keyframes") or s.get("opacity", 1) != 1
        eyes = isinstance(s.get("eyes"), list) and len(s["eyes"]) > 0
        lab = f"vp{i}" if kf else f"ve{i}" if eyes else f"v{i}"  # keyframed clips get one more stage (kf_main), glowing eyes one more (eye_glow); both end in [v{i}]
        if s.get("freeze"):
            # still of frame `in` only (2026-10-21: before, the first 0.5 s of footage played before the hold, so a freeze right
            # before an internal cut held the NEXT scene; the preview always held `in`)
            f.append(f"{src}fps={fps},trim=end_frame=1,setpts=PTS-STARTPTS,"
                     f"tpad=stop_mode=clone:stop_duration={lens[i]},trim=0:{lens[i]:.6f},{post}setsar=1,format=yuv420p,settb=AVTB[{lab}]")
        else:
            rev = "reverse,setpts=PTS-STARTPTS," if s.get("reverse") else ""  # ponytail: reverse buffers the clip in RAM
            f.append(f"{src}{rev}setpts=PTS/{sp},fps={fps},{exact[i]}{post}setsar=1,format=yuv420p,settb=AVTB[{lab}]")
        if kf:
            f.extend(kf_main(i, s, W, H, f"ve{i}" if eyes else None))
        if eyes:
            import fxface
            glow = fxface.glow_video(s, W, H, fps, bounds[i + 1] - bounds[i], media_dir or Path(s["media"]).parent)
            gname = glow.name if media_dir else str(glow)
            info[gname] = probe(str(glow))
            g = add_input(gname)
            f.extend(eye_glow(i, f"ve{i}", g, W, H))
        if info[s["media"]]["audio"] and s.get("volume", 1) > 0 and not s.get("freeze"):
            rev = "areverse," if s.get("reverse") else ""
            f.append(f"[{k}:a]atrim={s['in']}:{s['out']},asetpts=PTS-STARTPTS,{rev}{atempo(sp)},volume={s.get('volume', 1)}"
                     f"{afx(s)}{fades(s, lens[i])},aformat=sample_rates=48000:channel_layouts=stereo[a{i}]")
        else:  # muted / silent / frozen clip still needs an audio leg so joints line up
            f.append(f"anullsrc=r=48000:cl=stereo,atrim=0:{lens[i]:.6f}[a{i}]")

    import transitions
    custom = {i: segs[i - 1]["transition"]["type"] for i, t in enumerate(ts, start=1) if t and segs[i - 1]["transition"]["type"] in transitions.TYPES}
    trclip = {}
    if custom:  # motion-blur transitions (pull in, whip, spin ...): numpy renders the two overlap windows, the clip is spliced in below
        import numpy as np
        pre, maps = list(f), []  # ponytail: this prepass decodes every input again; add per-input -ss if it ever hurts
        for i in range(len(segs)):
            legs, n_i = [], round(lens[i] * fps)
            if i in custom:
                legs.append((f"h{i}", 0, round(ts[i - 1] * fps)))
            if i + 1 in custom:
                legs.append((f"t{i}", n_i - round(ts[i] * fps), n_i))
            pre.append(f"[a{i}]anullsink")
            if not legs:
                pre.append(f"[v{i}]nullsink"); continue
            pre.append(f"[v{i}]split={len(legs)}" + "".join(f"[p{l}]" for l, _, _ in legs))
            for l, a, b in legs:
                pre.append(f"[p{l}]trim=start_frame={a}:end_frame={b},setpts=PTS-STARTPTS[{l}]")
                maps += ["-map", f"[{l}]", "-f", "rawvideo", "-pix_fmt", "bgr24", f"{l}.raw"]
        Path(tmp, "pre.txt").write_text(";\n".join(pre), encoding="utf-8")
        run(args + graph_file_opt("pre.txt") + maps, cwd=tmp)

        def window(l, n):
            w = np.fromfile(Path(tmp, f"{l}.raw"), np.uint8).reshape(-1, H, W, 3)[:n]
            return w if len(w) == n else np.concatenate([w, np.repeat(w[-1:], n - len(w), 0)])

        for i, typ in custom.items():
            n = round(ts[i - 1] * fps)
            frames = transitions.render(typ, window(f"t{i - 1}", n), window(f"h{i}", n))
            clip = str(Path(tmp, f"tr{i}.mp4"))
            ff = subprocess.Popen([FF, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
                                   "-c:v", "libx264", "-preset", "medium", "-qp", "0", "-pix_fmt", "yuv444p", clip],  # lossless: the final encode is the only loss
                                  stdin=subprocess.PIPE, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            for fr in frames:
                ff.stdin.write(np.ascontiguousarray(fr).tobytes())
            ff.stdin.close(); ff.wait()
            info[clip] = {"video": {}, "audio": None}
            trclip[i] = add_input(clip)

    offs, total = joint_offsets(lens, ts)
    v, a = "[v0]", "[a0]"
    for i, (off, t) in enumerate(zip(offs, ts), start=1):
        if t and i in trclip:  # the pre-rendered transition clip replaces A's tail and B's head (same overlap model)
            f.append(f"{v}trim=end_frame={round(off * fps)},setpts=PTS-STARTPTS[ta{i}]")
            f.append(f"[{trclip[i]}:v]fps={fps},setsar=1,format=yuv420p,settb=AVTB[tt{i}]")
            f.append(f"[v{i}]trim=start_frame={round(t * fps)},setpts=PTS-STARTPTS[tb{i}]")
            f.append(f"[ta{i}][tt{i}][tb{i}]concat=n=3:v=1:a=0[vx{i}]")
            f.append(f"{a}[a{i}]acrossfade=d={t:.3f}[ax{i}]")
        elif t:
            typ = segs[i - 1]["transition"]["type"]
            f.append(f"{v}[v{i}]xfade=transition={typ}:duration={t:.3f}:offset={off}[vx{i}]")
            f.append(f"{a}[a{i}]acrossfade=d={t:.3f}[ax{i}]")
        else:
            f.append(f"{v}[v{i}]concat=n=2:v=1:a=0[vx{i}]")
            f.append(f"{a}[a{i}]concat=n=2:v=0:a=1[ax{i}]")
        v, a = f"[vx{i}]", f"[ax{i}]"

    mix = []  # extra audio legs: overlay sound + audio track

    def audio_leg(k, x):
        ms, sp = int(x.get("start", 0) * 1000), x.get("speed", 1)
        f.append(f"[{k}:a]atrim={x['in']}:{x['out']},asetpts=PTS-STARTPTS,{atempo(sp)},volume={x.get('volume', 1)}"
                 f"{afx(x)}{fades(x, seg_len(x))},aformat=sample_rates=48000:channel_layouts=stereo,adelay={ms}:all=1[x{k}]")
        mix.append(f"[x{k}]")

    back = [t for t in texts if t.get("behind")]  # text behind the person: drawn now, its cut-out overlays (fxface.behind) come on top
    if back:
        texts = [t for t in texts if not t.get("behind")]
        emit0 = lambda name, content: Path(tmp, name).write_text(content, encoding="utf-8")
        back = [x for t in back for x in typewriter(t)]
        f.append(v + ",".join(drawtext({**t, "_i": f"b{i}"}, H, emit0) for i, t in enumerate(back)) + "[vbt]")
        v = "[vbt]"
    for j, o in enumerate(o for o in overlays if o["start"] < total):  # CapCut overlay ("picture in picture") track
        k, sp, st = add_input(o["media"]), o.get("speed", 1), o["start"]
        ow = max(2, round(W * o.get("scale", 0.5)) // 2 * 2)
        vi = info[o["media"]]["video"] or {"width": 16, "height": 9}
        oh = max(2, round(ow * vi["height"] / vi["width"]) // 2 * 2)
        box = (oh, ow) if o.get("rotation", 0) % 180 == 90 else (ow, oh)  # frame size after the static transpose
        extra, ox, oy = pip_chain(o, *box, st, W, H)  # mask + keyframes
        fd = o.get("fade") or {}  # alpha fade in/out (lower thirds, logo) - cheap, unlike opacity keyframes (geq)
        fades_ = "".join(f"fade=t={d}:st={t0:.3f}:d={dd:.3f}:alpha=1," for d, t0, dd in
                         (("in", st, float(fd.get("in", 0))), ("out", st + seg_len(o) - float(fd.get("out", 0)), float(fd.get("out", 0)))) if dd > 0)
        gl = o.get("glass")  # glassmorphism: the video behind the card's mask is blurred (fades with the card)
        if gl and gl.get("mask"):
            km, en, sg = add_input(gl["mask"]), f"between(t,{st},{st + seg_len(o)})", float(gl.get("blur", 26)) * H / 1080
            f.append(f"{v}split[ga{j}][gb{j}]")
            f.append(f"[gb{j}]gblur=sigma={sg:.1f}:enable='{en}'[gc{j}]")
            f.append(f"[{km}:v]trim=0:{seg_len(o):.3f},setpts=PTS-STARTPTS+{st}/TB,scale={W}:{H},format=gray[gm{j}]")
            f.append(f"[gc{j}][gm{j}]alphamerge,format=yuva420p,{fades_}format=yuva420p[gg{j}]")
            f.append(f"[ga{j}][gg{j}]overlay=eof_action=pass:enable='{en}'[gv{j}]")
            v = f"[gv{j}]"
        f.append(f"[{k}:v]trim={o['in']}:{o['out']},setpts=(PTS-STARTPTS)/{sp}+{st}/TB,format=yuva420p,"
                 f"scale={ow}:{oh},{geometry(o)}format=yuva420p,{extra}{fades_}format=yuva420p[o{j}]")
        f.append(f"{v}[o{j}]overlay=x={ox}:y={oy}:eof_action=pass"
                 f":enable='between(t,{st},{st + seg_len(o)})'[ov{j}]")
        v = f"[ov{j}]"
        if info[o["media"]]["audio"] and o.get("volume", 1) > 0:
            audio_leg(k, o)

    for x in project.get("audio", []):
        if info[x["media"]]["audio"]:
            audio_leg(add_input(x["media"]), x)
    if mix:
        f.append(a + "".join(mix) + f"amix=inputs={len(mix) + 1}:normalize=0:duration=first[amix]")
        a = "[amix]"

    with tmp_obj as tmp:
        for c in LUTS.glob("*.cube") if LUTS.is_dir() else []:  # lut3d looks reference the cube by name, relative to the graph dir
            shutil.copy(c, tmp)
        glob = (FILTERS.get(project.get("filter") or "none") or "") + "," + adjust_chain(project.get("adjust"))
        glob = ",".join(x for x in glob.split(",") if x)
        if glob:
            f.append(f"{v}{glob}[vf]"); v = "[vf]"
        ex = project.get("export") or {}  # CapCut export: resolution (short side) + quality
        short = int(ex.get("short") or 0)
        if short and short < min(W, H):
            f.append(f"{v}scale=" + (f"{short}:-2" if W <= H else f"-2:{short}") + "[vs]"); v = "[vs]"
        crf = str(int(ex.get("crf") or 16))  # 2026-10-20: crf 20 veryfast smeared skin texture after dark looks (sharpness 26 -> 12 on a moody shot)
        if texts:  # captions/titles on top of everything
            texts = [x for t in texts for x in typewriter(t)]
            emit = lambda name, content: Path(tmp, name).write_text(content, encoding="utf-8")
            f.append(v + ",".join(drawtext({**t, "_i": i}, H, emit) for i, t in enumerate(texts)) + "[vt]")
            v = "[vt]"
        Path(tmp, "graph.txt").write_text(";\n".join(f), encoding="utf-8")
        run(args + graph_file_opt("graph.txt") + ["-map", v, "-map", a, "-c:v", "libx264", "-preset", "medium",  # -/opt FILE: value from a file (graphs can top the 32k arg limit)
                    "-crf", crf, "-maxrate", "24M", "-bufsize", "48M", "-x264-params", "aq-mode=3",  # aq-mode 3 keeps detail in dark areas; cap: grainy looks (phonk/vhs) otherwise hit 70 Mbit/s
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
                    "-t", f"{total}", str(Path(out).resolve())], cwd=tmp)
    return total


def combine(out, inputs, transition="fade", t=0.5):
    segs = [{"media": p, "in": 0, "out": probe(p)["dur"], "transition": {"type": transition if t else "none", "dur": t}}
            for p in inputs]
    render({"video": segs}, None, out)


# ---------- beats (CapCut "auto beat" / music sync: beat markers on the audio track) ----------
def beats(path, hop=512):
    """Beat grid via librosa (Ellis dynamic-programming beat tracker, same family CapCut's auto-beat uses).
    Returns {bpm, beats:[s], strong:[s]} - `strong` = beats whose onset energy is above the median (drops/accents)."""
    import librosa, numpy as np
    wav = Path(tempfile.gettempdir()) / f"clipcut_{abs(hash(path))}.wav"
    run([FF, "-y", "-i", path, "-vn", "-ac", "1", "-ar", "22050", "-t", "600", str(wav)])  # ponytail: first 10 min, mono
    try:
        y, sr = librosa.load(str(wav), sr=22050, mono=True)
    finally:
        wav.unlink(missing_ok=True)
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    tempo, frames = librosa.beat.beat_track(onset_envelope=onset, sr=sr, hop_length=hop, units="frames")
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop)
    energy = onset[np.clip(frames, 0, len(onset) - 1)]
    strong = times[energy > np.percentile(energy, 70)] if len(energy) else []  # top 30 % = accents
    # drop = the beat where loudness over the next 4 beats jumps most vs. the previous 4 (first 45 s)
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rb = [float(rms[int(a / hop * sr):max(int(a / hop * sr) + 1, int(b / hop * sr))].mean()) for a, b in zip(times, times[1:])]
    drop, best = None, 0.0
    for i in range(4, len(rb) - 4):
        if times[i] > 45:
            break
        jump = sum(rb[i:i + 4]) / (sum(rb[i - 4:i]) + 1e-6)
        if jump > best:
            best, drop = jump, float(times[i])
    if best < 1.15:  # no real drop: use the first accent after 2 s
        drop = next((float(t) for t in strong if t > 2), float(times[0]) if len(times) else 0.0)
    else:  # the 4-beat window max can sit a beat or two late: step back to the first loud beat, then snap to the kick itself
        i = int(np.argmin(np.abs(times - drop)))
        hi = sum(rb[i:i + 4]) / 4
        while i > 0 and rb[i - 1] > 0.6 * hi:
            i -= 1
        fine = librosa.feature.rms(y=y, frame_length=1024, hop_length=221)[0]  # 10 ms grid
        a, b = max(1, int((times[i] - 0.4) * sr / 221)), min(len(fine) - 1, int((times[i] + 0.3) * sr / 221))
        drop = float((a + int(np.argmax(np.diff(fine[a - 1:b + 1])))) * 221 / sr) if b > a else float(times[i])
    return {"bpm": round(float(np.atleast_1d(tempo)[0]), 2), "beats": [round(float(t), 3) for t in times],
            "strong": [round(float(t), 3) for t in strong], "drop": round(drop, 3)}


def beat_cut(segs, grid, every=1):
    """Split gapless main-track segments at every N-th beat (timeline seconds). Pure, so the UI and CLI share it."""
    out, t0 = [], 0.0
    marks = grid[::max(1, every)]
    for s in segs:
        L = seg_len(s)
        cuts = [m - t0 for m in marks if t0 + 0.1 < m < t0 + L - 0.1]
        pos, src = 0.0, s["in"]
        for c in [*cuts, L]:
            piece = {**s, "id": None, "transition": {"type": "none", "dur": 0.5}}
            if s.get("freeze"):
                piece["dur"] = c - pos
            else:
                piece["in"], piece["out"] = src, src + (c - pos) * s.get("speed", 1)
                src = piece["out"]
            out.append(piece); pos = c
        out[-1]["transition"] = s.get("transition", {"type": "none", "dur": 0.5})
        t0 += L
    return out


def autoedit(clips, grid, every=2, song=None, min_len=0.25):
    """TikTok beat-sync template: one clip slice per N beats, clips round-robin, each slice continues where the
    clip's previous slice stopped so the footage never repeats. clips = [{media, dur}], grid = beat times (s)."""
    marks = [m for m in grid[::max(1, every)]]
    marks = [0.0, *[m for m in marks if m > min_len]]
    segs, cursor = [], {c["media"]: 0.0 for c in clips}
    for k, (a, b) in enumerate(zip(marks, marks[1:])):
        c = clips[k % len(clips)]
        length = b - a
        if cursor[c["media"]] + length > c["dur"]:  # ran out of footage: start that clip over
            cursor[c["media"]] = 0.0
        if length > c["dur"]:
            continue
        s, e = cursor[c["media"]], cursor[c["media"]] + length
        segs.append({"media": c["media"], "in": round(s, 3), "out": round(e, 3), "speed": 1, "rotation": 0, "flip": False,
                     "volume": 0 if song else 1, "transition": {"type": "none", "dur": 0.5}})
        cursor[c["media"]] = e
    total = marks[-1] if len(marks) > 1 else 0
    proj = {"canvas": None, "fps": 30, "video": segs, "overlay": [], "text": [], "beats": grid,
            "audio": [{"media": song, "start": 0, "in": 0, "out": total, "speed": 1, "volume": 1, "fadeOut": 1}] if song else []}
    return proj


# ---------- agent steps: write project.json + agent.jsonl so the open editor shows each edit ----------
ROOT = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).parent


def uid():
    return __import__("uuid").uuid4().hex[:8]


def write_atomic(path, text):
    """The editor polls project.json while we write it; replace() makes every read see a complete file."""
    import os
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def say(msg, root=ROOT):
    with open(Path(root) / "agent.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"t": __import__("time").time(), "msg": msg}, ensure_ascii=False) + "\n")


def apply_steps(steps, root=ROOT, pace=0.4):
    import time
    for msg, P in steps:
        write_atomic(Path(root) / "project.json", json.dumps(P))
        say(msg, root)
        print(msg, file=sys.stderr)
        time.sleep(pace)


def tl_layout(P):
    """Timeline start of every main-track segment (same overlap rule as the renderer / UI)."""
    segs, t, out = P["video"], 0.0, []
    for i, s in enumerate(segs):
        if i:
            tr = segs[i - 1].get("transition") or {}
            if tr.get("type", "none") != "none":
                t -= min(tr.get("dur", 0), seg_len(segs[i - 1]) / 2, seg_len(s) / 2)
        out.append(t); t += seg_len(s)
    return out


def op(args, root=ROOT):
    """Small visible edits for an agent driving the editor. Returns a message."""
    pj = Path(root) / "project.json"
    P = json.loads(pj.read_text(encoding="utf-8")) if pj.exists() else {"canvas": None, "fps": 30, "video": [], "overlay": [], "audio": [], "text": []}
    cmd, rest = args[0], args[1:]
    uid = lambda: __import__("uuid").uuid4().hex[:8]

    def find(sel):
        for k in ("video", "overlay", "audio", "text"):
            for i, x in enumerate(P[k]):
                if x.get("id") == sel or (sel.startswith("#") and k == "video" and i == int(sel[1:])):
                    return k, i, x
        raise SystemExit(f"no item {sel}")

    if cmd == "say":
        msg = " ".join(rest)
    elif cmd == "eyes":  # CapCut-style glowing eyes (fxface.py)
        import fxface
        k, i, x = find(rest[0]); x[cmd] = True
        msg = fxface.bake(P, Path(root) / "media", root) or f"No face found for {cmd} on {rest[0]}"
    elif cmd == "worldtext":  # big 3D frosted text pinned in the scene behind a building (worldtext.py), baked into the shot
        import worldtext
        k, i, x = find(rest[0])
        if k != "video":
            raise SystemExit("worldtext needs a main-track shot (#index or id)")
        if rest[1:2] == ["none"]:
            worldtext.remove(x)
            msg = f"3D-Text von Shot {rest[0]} entfernt"
        else:
            spec = {"text": rest[1]} if len(rest) > 1 and "=" not in rest[1] else {}
            for kv in rest[1 + bool(spec):]:
                key, val = kv.split("=", 1)
                try:
                    spec[key] = json.loads(val)
                except ValueError:
                    spec[key] = val
            out = worldtext.apply(x, Path(root) / "media", spec, log=lambda m: print(m, file=sys.stderr))
            msg = f"3D-Text '{x['world']['text']}' steht jetzt in der Szene (Shot {rest[0]}, {Path(out).name})"
    elif cmd == "behind":  # text behind the person: RVM cut-out of the shots under the text goes on top of it
        import fxface
        k, i, x = find(rest[0])
        if k != "text":
            raise SystemExit("behind needs a text id")
        x["behind"] = "building" if rest[1:2] == ["building"] else True
        n = fxface.behind(P, x, Path(root) / "media")
        what = "dem Gebäude" if x["behind"] == "building" else "der Person"
        msg = f"Text '{x['text'][:30]}' steht jetzt hinter {what} ({n} Ausschnitt(e))" if n else "Kein passender Shot unter dem Text gefunden"
    elif cmd == "split":
        t, starts = float(rest[0]), tl_layout(P)
        i = max((i for i, st in enumerate(starts) if st <= t), default=None)
        if i is None or t - starts[i] >= seg_len(P["video"][i]) - 0.05 or t - starts[i] < 0.05:
            raise SystemExit("nothing to split there")
        s, local = P["video"][i], t - starts[i]
        right = {**s, "id": uid()}
        if s.get("freeze"):
            right["dur"], s["dur"] = s["dur"] - local, local
        else:
            cut = s["in"] + local * s.get("speed", 1)
            right["in"], s["out"] = cut, cut
        s["transition"] = {"type": "none", "dur": 0.5}
        P["video"].insert(i + 1, right)
        msg = f"Split clip #{i} at {t:.2f}s"
    elif cmd == "set":
        k, i, x = find(rest[0])
        for kv in rest[1:]:
            key, val = kv.split("=", 1)
            try:
                x[key] = json.loads(val)
            except ValueError:
                x[key] = val
        msg = f"Set {' '.join(rest[1:])} on {rest[0]}"
    elif cmd == "text":
        P["text"].append({"id": uid(), "text": rest[0], "start": float(rest[1]), "dur": float(rest[2]), "x": 0.5, "y": 0.22, "size": 0.09,
                          "color": "#ffffff", "font": "montserrat", "shadow": True, "box": False})
        msg = f"Text '{rest[0]}' at {float(rest[1]):.2f}s"
    elif cmd == "sticker":  # op sticker <textid> <WORD> <media.webm> [scale=1.1] [rot=-15] | op sticker <textid> none: emoji right after a caption word
        k, i, x = find(rest[0])
        if rest[1] == "none":
            x.pop("sticker", None); msg = f"Sticker removed from {rest[0]}"
        else:
            x["sticker"] = {"media": rest[2], "word": rest[1], **{a: float(b) for a, b in (kv.split("=", 1) for kv in rest[3:])}}
            if not sticker_overlay(x, *(P.get("canvas") or [1080, 1920])):
                raise SystemExit(f"no word {rest[1]!r} in text {rest[0]}")
            msg = f"Sticker {rest[2]} after '{rest[1]}' of {rest[0]}"
    elif cmd == "lowerthird":  # op lowerthird "Name" "Rolle" START [DUR] : branded lower third (brand.py, school colours + logo)
        import brand
        name, role, st = rest[0], (rest[1] if len(rest) > 1 else ""), float(rest[2] if len(rest) > 2 else 0)
        dur = float(rest[3]) if len(rest) > 3 else 4.5
        W, H = P.get("canvas") or [int(probe(str(Path(root) / "media" / P["video"][0]["media"]))["video"][k]) for k in ("width", "height")]
        g = uid()
        art, mask = brand.lowerthird_glass(name, role, W, H)  # glassmorphism card in the school colours + logo
        art.save(Path(root) / "media" / f"_lt_{g}.png"); mask.save(Path(root) / "media" / f"_lt_{g}_mask.png")
        P["overlay"].append({"id": g, "media": f"_lt_{g}.png", "start": st, "in": 0, "out": dur, "speed": 1, "x": 0.5, "y": 0.5,
                             "scale": 1, "volume": 0, "lt": {"name": name, "role": role}, "fade": {"in": 0.45, "out": 0.4},
                             "glass": {"mask": f"_lt_{g}_mask.png", "blur": 26}})
        msg = f"Bauchbinde '{name}' at {st:.2f}s"
    elif cmd == "logo":  # op logo [corner=tr|tl|br|bl] [start=0] [dur=<all>] [size=0.05] : school logo (brand.py), white + gold
        import brand
        kw = dict(kv.split("=", 1) for kv in rest)
        W, H = P.get("canvas") or [int(probe(str(Path(root) / "media" / P["video"][0]["media"]))["video"][k]) for k in ("width", "height")]
        total = sum(seg_len(v) for v in P["video"]) or 10
        st = float(kw.get("start", 0)); dur = float(kw.get("dur", total - st))
        h = round(float(kw.get("size", 0.042)) * min(W, H))
        h = min(h, round(0.3 * W / 9.5))  # the wide logo is ~9.5x as wide as tall: at most 30 % of the frame width
        g = uid()
        im = brand.logo(h, wide=True, shadow=True)
        im.save(Path(root) / "media" / f"_logo_{g}.png")
        sc = im.width / W
        c = kw.get("corner", "tr")
        mx, my = 0.035, 0.035 * W / H
        x = 1 - mx - sc / 2 if c[1] == "r" else mx + sc / 2
        y = my + h / H / 2 if c[0] == "t" else 1 - my - h / H / 2
        P["overlay"].append({"id": g, "media": f"_logo_{g}.png", "start": st, "in": 0, "out": dur, "speed": 1, "x": round(x, 4),
                             "y": round(y, 4), "scale": round(sc, 4), "opacity": 0.92, "volume": 0, "logo": True, "fade": {"in": 0.6, "out": 0.6}})
        msg = f"Logo {c} from {st:.1f}s"
    elif cmd == "mapintro":  # op mapintro LAT LON ["Label"] [sub=Köln-Porz] [dur=10] : Europe -> school map flight, first shot
        import mapintro
        lat, lon = float(rest[0]), float(rest[1])
        label = rest[2] if len(rest) > 2 and "=" not in rest[2] else ""
        kw = dict(kv.split("=", 1) for kv in rest[2:] if "=" in kv)
        W, H = P.get("canvas") or [int(probe(str(Path(root) / "media" / P["video"][0]["media"]))["video"][k]) for k in ("width", "height")]
        g = uid()
        dur = float(kw.get("dur", 10))
        mapintro.render(Path(root) / "media" / f"_map_{g}.mp4", lat, lon, label, kw.get("sub", ""), dur, W, H, P.get("fps", 30),
                        log=lambda m: print(m, file=sys.stderr))
        P["video"].insert(0, {"id": g, "media": f"_map_{g}.mp4", "in": 0, "out": dur, "speed": 1, "volume": 0,
                              "transition": {"type": "dissolve", "dur": 0.8}})
        msg = f"Map intro ({dur:.0f} s) in front of the first shot"
    elif cmd == "captions":  # op captions SEL [size=0.045] | op captions SEL none : subtitles from the speech of one main-track shot
        k, i, x = find(rest[0])
        if k != "video":
            raise SystemExit("captions needs a main-track shot (#index or id)")
        P["text"] = [t for t in P["text"] if t.get("cap") != x["id"]]
        if rest[1:2] == ["none"]:
            msg = f"Captions removed from {rest[0]}"
        else:
            kw = dict(kv.split("=", 1) for kv in rest[1:])
            r = transcribe(str(Path(root) / "media" / x["media"]))
            cw, ch = P.get("canvas") or [int(probe(str(Path(root) / "media" / x["media"]))["video"][k]) for k in ("width", "height")]
            maxc = 42 if cw >= ch else 28  # characters per cue line: landscape / portrait
            size = float(kw.get("size", min(0.045, 0.9 * cw / (maxc * 0.6) / ch)))  # a full line fits 90 % of the width
            sp, t0 = float(x.get("speed", 1) or 1), tl_layout(P)[i]
            words = [w for w in r["words"] if x["in"] <= w["s"] < x["out"]]
            cues, cur = [], []
            for j, w in enumerate(words):  # cue = up to ~42 characters / 3 s, broken at pauses and sentence ends
                cur.append(w)
                txt = " ".join(c["w"] for c in cur)
                nxt = words[j + 1] if j + 1 < len(words) else None
                if (nxt is None or len(txt) + len(nxt["w"]) > maxc or nxt["s"] - w["e"] > 0.6 or w["e"] - cur[0]["s"] > 3.0
                        or w["w"][-1:] in ".!?"):
                    cues.append((cur[0]["s"], w["e"], txt)); cur = []
            new = []
            for a, b, txt in cues:
                ts = t0 + (a - x["in"]) / sp
                new.append({"id": uid(), "text": txt, "start": round(ts, 3), "dur": round(max(0.6, (b - a) / sp + 0.15), 3),
                            "x": 0.5, "y": 0.9, "size": size, "font": "montserrat", "color": "#ffffff",
                            "soft": True, "cap": x["id"], "lang": r["language"]})
            P["text"] += new
            print(json.dumps([{"id": t["id"], "text": t["text"]} for t in new], ensure_ascii=False))
            msg = f"{len(new)} captions ({r['language']}) on {rest[0]}"
    elif cmd == "translate":  # op translate FILE.json [mode=add|replace] : {"<caption id>": "translation", ...}
        tr = json.loads(Path(rest[0]).read_text(encoding="utf-8"))
        mode = dict(kv.split("=", 1) for kv in rest[1:]).get("mode", "add")
        P["text"] = [t for t in P["text"] if t.get("tr_of") not in tr]  # re-translating replaces the old line
        n, add = 0, []
        for t in P["text"]:
            if t["id"] not in tr:
                continue
            n += 1
            if mode == "replace":
                t["orig"] = t.get("orig", t["text"]); t["text"] = tr[t["id"]]
            else:  # second language: its own line under the caption, smaller, in the school's light gold
                t["y"] = 0.875
                add.append({**t, "id": uid(), "text": tr[t["id"]], "y": 0.875 + t["size"] * 1.25, "size": round(t["size"] * 0.8, 4),
                            "color": "#f6dc9a", "tr_of": t["id"], "cap": t.get("cap")})
        P["text"] += add
        msg = f"Translated {n} captions ({mode})"
    elif cmd == "delete":
        k, i, x = find(rest[0]); P[k].pop(i); msg = f"Deleted {rest[0]}"
    elif cmd == "move":
        k, i, x = find(rest[0]); P[k].pop(i); P[k].insert(int(rest[1]), x); msg = f"Moved {rest[0]} to position {rest[1]}"
    elif cmd == "transition":
        i = int(rest[0].lstrip("#")); P["video"][i]["transition"] = {"type": rest[1], "dur": float(rest[2]) if len(rest) > 2 else 0.3}
        msg = f"Transition {rest[1]} after clip #{i}"
    else:
        raise SystemExit(f"unknown op {cmd}")
    write_atomic(pj, json.dumps(P))
    say(msg, root)
    return msg


def hook_len(path, max_s=4.5, words=None):
    """Hook ends where the speaker's sentence ends: last word finishing before max_s (+ a breath). None if no speech."""
    words = words if words is not None else transcribe(path)["words"]
    ends = [w["e"] for w in words if w["e"] <= max_s]
    if not ends:
        return None
    sent = [w["e"] for w in words if w["e"] <= max_s and w["w"].rstrip().endswith((".", "!", "?"))]
    return round((sent[-1] if sent else ends[-1]) + 0.15, 2)


# ---------- star check (is the main person on screen?) ----------
def face_ref(root=ROOT):
    p = Path(root) / "face_ref.json"
    return json.loads(p.read_text(encoding="utf-8"))["emb"] if p.exists() else None


def face_scan(path, cache_dir=None):
    import faces
    cache_dir = Path(cache_dir or Path(path).parent.parent / "transcripts")
    cache_dir.mkdir(exist_ok=True)
    c = cache_dir / f"{Path(path).name}.{int(Path(path).stat().st_mtime)}.faces.json"
    if not c.exists():
        c.write_text(json.dumps(faces.scan(path)), encoding="utf-8")
    return json.loads(c.read_text(encoding="utf-8"))


def star_ranges(path, root=ROOT, cache_dir=None):
    """Seconds where the reference person is on screen, or None when no reference exists yet."""
    import faces
    ref = face_ref(root)
    if ref is None:
        return None
    return faces.visible(face_scan(path, cache_dir), ref)


def star_track(path, root=ROOT, cache_dir=None):
    """[[t, x, y, w, h]] source-fraction boxes of the star per 1/4 s sample (only where he matches), or None without a reference."""
    import faces
    ref = face_ref(root)
    if ref is None:
        return None
    import cv2  # frame size as decoded (phone clips with rotation metadata come out portrait, ffprobe reports the coded landscape size)
    cap = cv2.VideoCapture(str(path)); ok, img = cap.read(); cap.release()
    v = probe(path)["video"]
    return faces.track(face_scan(path, cache_dir), ref, (img.shape[1], img.shape[0]) if ok else (v["width"], v["height"]))


def scene_cuts(path, cache_dir=None):
    """Internal shot cuts of a clip (clipcut.scenes), cached next to the face/moment caches."""
    cache_dir = Path(cache_dir or Path(path).parent.parent / "transcripts")
    cache_dir.mkdir(exist_ok=True)
    c = cache_dir / f"{Path(path).name}.{int(Path(path).stat().st_mtime)}.scenes.json"
    if not c.exists():
        c.write_text(json.dumps(scenes(path)), encoding="utf-8")
    return json.loads(c.read_text(encoding="utf-8"))


def build_face_ref(clips, root=ROOT, cache_dir=None):
    import faces
    scans = [face_scan(c, cache_dir) for c in clips]
    emb, members = faces.reference(scans)
    Path(root, "face_ref.json").write_text(json.dumps({"emb": emb, "clips": [Path(c).name for c in clips]}), encoding="utf-8")
    sheet = faces.crops(scans, members, clips, Path(root) / "_star_faces.png")
    return {"clips_with_star": len({m[0] for m in members}), "faces": len(members), "sheet": sheet}


def clip_infos(clips, root=ROOT, dumb=False, cache_dir=None):
    infos = [{"media": Path(c).name, "dur": probe(c)["dur"]} for c in clips]
    for c, i in zip(clips, infos):  # only use moments where the star is on screen (if a reference exists)
        r = star_ranges(c, root, cache_dir)
        if r is not None:
            i["ranges"] = r
        if not dumb and r != []:  # motion/loudness per 1/8 s -> trends picks gestures on the beat (motion.Picker); --dumb = round-robin
            try:
                import motion
                i["moments"] = motion.moments(c, cache_dir)
                if r is not None:
                    i["star"] = star_track(c, root, cache_dir)  # face per 1/4 s: Picker verifies every window and frames the crop on him
                i["cuts"] = scene_cuts(c, cache_dir)  # a window never spans an internal cut
            except Exception as e:
                print(f"moments: {c}: {e}", file=sys.stderr)
    return infos


def with_review(steps, infos, root=ROOT):
    """Look at the planned shots (review.py), re-pick the ones without a clearly framed star, then bake the face effects the planner
    flagged (fxface: eyes / sun); both become steps before 'Done'."""
    import copy
    out = steps[:-1]
    P = steps[-1][1]
    try:
        import review
        P, msg, _ = review.repair(P, infos, Path(root) / "media", root)
        out.append((msg, P))
    except Exception as e:  # review is a safety net, never a reason to not deliver the edit
        print(f"review: {e}", file=sys.stderr)
    try:
        import fxface
        P = copy.deepcopy(P)
        msg = fxface.bake(P, Path(root) / "media", root)
        if msg:
            out.append((msg, P))
    except Exception as e:
        print(f"fx: {e}", file=sys.stderr)
    return out + [(steps[-1][0], P)]


def copy_edit(ref, clips, text=None, pace=0.4, root=ROOT, dumb=False, even=False):
    """Re-create a TikTok edit (URL or mp4) with your own clips: same cuts/flashes/punches/slow-mo, its soundtrack as the song."""
    import copy_edit as ce
    R = ce.reference(ref)
    infos = clip_infos(clips, root, dumb)
    steps = with_review(ce.plan(R, infos, text=text, even=even), infos, root)
    apply_steps(steps, root, pace)
    return steps[-1][1]


def tiktok(song, clips, style="auto", text=None, hook=None, every=None, pace=0.4, root=ROOT, grid=None, infos=None, dumb=False, look=None, drop=None):
    import trends
    grid = grid or beats(song)
    if drop:
        grid = {**grid, "drop": drop}
    if infos is None:
        infos = clip_infos(clips, root, dumb)
    if hook is None and style in ("auto", "ironic"):
        try:
            hook = hook_len(clips[0])
        except Exception as e:  # no whisper model / no audio: template default
            print(f"hook: {e}", file=sys.stderr)
    steps = with_review(trends.plan(style, infos, grid, Path(song).name, text=text, hook=hook, every=every, look=look), infos, root)
    steps = level_hook(steps, root)
    apply_steps(steps, root, pace)
    return steps[-1][1]


def loudness(path, start=0, dur=None):
    """Integrated loudness (LUFS, EBU R128) of a window of a media file."""
    r = subprocess.run([FF, "-v", "info", "-ss", str(start)] + (["-t", str(dur)] if dur else []) + ["-i", str(path), "-vn", "-af", "ebur128", "-f", "null", "-"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    m = re.findall(r"I:\s+(-?[\d.]+) LUFS", r.stderr)
    return float(m[-1]) if m else None


def level_hook(steps, root, speech=-15.0, song=-12.0, under=-19.0):
    """Hook speech vs. the song after the drop (user 2026-10-19: the drop was way too loud after the hook): gain the hook clip to ~-16 LUFS
    and the song to ~-9 LUFS, so the drop still hits (+7 dB) without slapping. Returns steps with one more step when something changed."""
    import copy, math
    P = copy.deepcopy(steps[-1][1])
    v, a = P.get("video", []), P.get("audio", [])
    if not (v and a and v[0].get("volume") == 1 and len(a) >= 2):  # hooked plans only: hook clip audible, ducked song + song from the drop
        return steps
    h, sduck, srest = v[0], a[0], a[-1]
    media = Path(root) / "media"
    lh = loudness(media / h["media"], h["in"], max(0.5, h["out"] - h["in"]))
    lu = loudness(media / sduck["media"], sduck["in"], max(0.5, sduck["out"] - sduck["in"]))
    ls = loudness(media / srest["media"], srest["in"], min(8.0, srest["out"] - srest["in"]))
    if lh is None or ls is None or lu is None:
        return steps
    gain = lambda target, measured, lo, hi: round(min(hi, max(lo, 10 ** ((target - measured) / 20))), 3)
    h["volume"], sduck["volume"], srest["volume"] = gain(speech, lh, 0.5, 4.0), gain(under, lu, 0.05, 4.0), gain(song, ls, 0.2, 1.5)
    # user 2026-10-19: the music under the hook must be clearly audible (-19 LUFS, 4 dB under the speech) and the drop only ~7 dB louder
    return steps + [(f"Levels: hook speech {lh:.0f} -> {speech:.0f} LUFS, music under it {lu:.0f} -> {under:.0f} LUFS, "
                     f"song after the drop {ls:.0f} -> {song:.0f} LUFS", P)]


def selftest():
    cues = parse_srt("1\n00:00:01,000 --> 00:00:02,500\nhello um world\n\n2\n00:00:04,000 --> 00:00:05,000\n<i>Um</i> yes\n\n"
                     "3\n00:00:05,000 --> 00:00:06,000\nok\n")
    assert cues[1] == (4.0, 5.0, "Um yes"), cues
    assert keep_ranges(cues, r"\bum\b", 10) == [(0.0, 1.0), (2.5, 4.0), (5.0, 10)]
    assert keep_ranges(cues, r"\bum\b", 10, keep=True) == [(1.0, 2.5), (4.0, 5.0)]
    assert keep_ranges(cues, r"um|ok", 6) == [(0.0, 1.0), (2.5, 4.0)]  # adjacent hits merge, nothing after 6s
    W = [{"s": 0, "e": 0.4, "w": "Was"}, {"s": 0.5, "e": 2.26, "w": "Welt?"}, {"s": 2.3, "e": 2.78, "w": "Alles."}, {"s": 5, "e": 6, "w": "Und"}]
    assert hook_len(None, words=W) == 2.93 and hook_len(None, words=W[:1]) == 0.55 and hook_len(None, words=W[3:]) is None
    assert joint_offsets([5, 5, 5], [1, 1]) == ([4, 8], 13)
    assert fit_chain("cover", 1080, 1920, {"cx": 0.8, "cy": 2}) == "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920:clip(iw*0.8000-1080/2\\,0\\,iw-1080):clip(ih*1.0000-1920/2\\,0\\,ih-1920),"
    assert "iw*0.5000-10/2" in fit_chain("cover", 10, 10) and "pad=" in fit_chain("contain", 10, 10)
    assert joint_offsets([3, 4], [0]) == ([3], 7)
    assert seg_len({"in": 2, "out": 6, "speed": 2}) == 2
    assert atempo(4) == "atempo=2,atempo=2.000000" and atempo(1) == "atempo=1.000000"
    segs = [{"media": "a", "in": 0, "out": 4, "speed": 1, "transition": {"type": "fade", "dur": 0.5}}]
    cut = beat_cut(segs, [1, 2, 3, 3.95], every=1)
    assert [(s["in"], s["out"]) for s in cut] == [(0, 1), (1, 2), (2, 3), (3, 4)], cut  # 3.95 too close to the end
    assert cut[-1]["transition"]["type"] == "fade" and cut[0]["transition"]["type"] == "none"
    cut = beat_cut([{"media": "a", "in": 2, "out": 6, "speed": 2}], [0.5, 1.5])
    assert [(s["in"], s["out"]) for s in cut] == [(2, 3), (3, 5), (5, 6)], cut
    p = autoedit([{"media": "x", "dur": 3}, {"media": "y", "dur": 10}], [0, 0.5, 1, 1.5, 2, 2.5, 3], every=2, song="s.mp3")
    assert [(s["media"], s["in"], s["out"]) for s in p["video"]] == [("x", 0, 1), ("y", 0, 1), ("x", 1, 2)], p["video"]
    assert p["audio"][0]["out"] == 3 and p["video"][0]["volume"] == 0
    import tempfile as _tf
    with _tf.TemporaryDirectory() as d:
        Path(d, "project.json").write_text(json.dumps({"canvas": None, "fps": 30, "video": [{"id": "aaaaaaaa", "media": "x", "in": 0, "out": 4, "speed": 2,
            "transition": {"type": "fade", "dur": 0.5}}, {"id": "bbbbbbbb", "media": "x", "in": 0, "out": 2, "speed": 1}], "overlay": [], "audio": [], "text": []}))
        assert tl_layout(json.loads(Path(d, "project.json").read_text())) == [0.0, 1.5]
        op(["split", "1"], d); P = json.loads(Path(d, "project.json").read_text())
        assert [(x["in"], x["out"]) for x in P["video"]] == [(0, 2), (2, 4), (0, 2)], P["video"]
        op(["set", "#1", "zoom=1.3", "filter=bw"], d); op(["text", "YO", "0.5", "1"], d); op(["delete", "#0"], d)
        P = json.loads(Path(d, "project.json").read_text())
        assert P["video"][0]["zoom"] == 1.3 and P["video"][0]["filter"] == "bw" and P["text"][0]["text"] == "YO" and len(P["video"]) == 2
        assert len(Path(d, "agent.jsonl").read_text(encoding="utf-8").splitlines()) == 4
    # keyframes / curve speed / masks / text animations (pure string + timing checks; `rendertest` runs them through ffmpeg)
    kf = {"keyframes": [{"t": 0, "scale": 1}, {"t": 1, "scale": 2, "opacity": 0.5}]}
    assert kf_expr(kf, "scale") == "if(lt(t,1.0),1.0+(2.0-1.0)*clip((t-0.0)/1.0,0,1),2.0)" and kf_expr(kf, "x") is None
    assert pl_expr([(0, 5)], "T") == "5" and pl_expr([(1, 2), (0, 1)]).startswith("if(lt(t,1),1+(2-1)*clip((t-0)/1,0,1),2)")
    eg = eye_glow(3, "ve3", 4, 400, 800)
    assert eg[0].startswith("[4:v]format=rgba,scale=400:800") and "[ve3][g3]overlay" in eg[1] and eg[1].endswith("[v3]"), eg
    assert adjust_chain({"bri": 0.5, "temp": 1}) == "eq=brightness=0.200:contrast=1.000:saturation=1.000,colortemperature=temperature=3700:mix=1," and adjust_chain(None) == ""
    assert geometry({"crop": {"x": 0.1, "y": 0, "w": 0.5, "h": 1}}).startswith("crop=iw*0.5000:ih*1.0000:iw*0.1000:ih*0.0000,") and geometry({"crop": {"w": 1, "h": 1}}) == ""
    assert afx({"denoise": True, "pitch": 12}) == ",afftdn=nf=-25,rubberband=pitch=2.0000" and afx({}) == ""
    assert face_on_canvas({"cx": 0.5, "cy": 0.2}, "cover", 1080, 1920, 1080, 1920) == (0.5, 0.2) and face_on_canvas({}, "cover", 1, 1, 1, 1) == (0.5, 0.5)
    assert face_on_canvas({"cx": 0.1, "cy": 0.5}, "cover", 1080, 1920, 1920, 1080) == (0.316, 0.5)  # landscape source: crop clamps at the left edge
    assert "x='clip(iw*0.3-1080/2,0,iw-1080)'" in fx({"zoom": 1.3}, 1080, 1920, 1, (0.3, 0.2))
    assert kf_expr(kf, "opacity", "T") == "0.5" and kf_expr({"opacity": 0.3}, "opacity") == "0.3" and kf_expr({}, "opacity") is None
    assert len(kf_main(0, kf, 100, 100)) == 4 and "[v0]" in kf_main(0, kf, 100, 100)[-1]
    assert mask_expr({"type": "circle", "invert": True}).startswith("(1-clip(") and "abs" in mask_expr({"type": "rect"})
    cs = {"in": 0, "out": 3, "speed": 1, "speedCurve": [{"t": 0, "speed": 1}, {"t": 0.5, "speed": 3}, {"t": 1, "speed": 1}],
          "transition": {"type": "fade", "dur": 0.3}, "keyframes": [{"t": 1, "x": 0.1}]}
    pcs = expand_speed(cs)
    assert len(pcs) == 6 and abs(pcs[-1]["out"] - 3) < 1e-9 and pcs[-1]["transition"]["type"] == "fade" and pcs[0]["transition"]["type"] == "none"
    assert abs(seg_len(cs) - sum((p["out"] - p["in"]) / p["speed"] for p in pcs)) < 1e-9 and seg_len(cs) < 3  # faster than 1x overall
    assert pcs[1]["keyframes"][0]["t"] < 1  # keyframe times shift with each piece
    tw = typewriter({"text": "abc", "start": 1, "dur": 2, "anim_in": {"type": "typewriter", "dur": 0.6}, "anim_out": {"type": "fade", "dur": 0.2}})
    assert [x["text"] for x in tw] == ["a", "ab", "abc"] and abs(tw[-1]["start"] + tw[-1]["dur"] - 3) < 1e-6 and tw[-1]["anim_out"]["type"] == "fade"
    fs, al, dx, dy, k = text_anim({"start": 0, "dur": 2, "anim_in": {"type": "pop", "dur": 0.3}, "anim_out": {"type": "slide", "dur": 0.3}}, 40)
    assert fs.startswith("'40*(0.25") and "clip((2-t)" in al and dy.startswith("+(-1)*") and dx == "" and k.startswith("(0.25")
    assert text_anim({"start": 0, "dur": 1}, 40)[4] == "1"
    ht = {"id": "t1", "text": "DÖNER MACHT\nSCHÖNER", "start": 0, "dur": 1, "x": 0.5, "y": 0.8, "size": 0.07, "font": "impact", "hl": "döner", "anim_in": {"type": "pop", "dur": 0.1}}
    lines, size = text_layout(ht, 1920)
    assert size == 134 and [len(l) for l in lines] == [2, 1] and lines[0][0]["hl"] and not lines[0][1]["hl"] and lines[0][0]["size"] == 161
    assert lines[0][0]["x0"] < 0 < lines[0][1]["x0"] and abs(lines[1][0]["x0"] + lines[1][0]["wd"] / 2) < 1e-6  # centred lines
    files = {}
    dt = drawtext({**ht, "_i": 3}, 1920, lambda n, c: files.__setitem__(n, c))
    assert dt.count("drawtext=") == 3 and files == {"t3_0_0.txt": "DÖNER", "t3_0_1.txt": "MACHT", "t3_1_0.txt": "SCHÖNER"} and f"fontcolor={HL_COLOR}" in dt and "*(0.25+0.75*" in dt
    assert drawtext({**ht, "hlcolor": "#ff0000", "_i": 4}, 1920).count("fontcolor=#ff0000") == 1  # only the keyword gets the colour
    assert drawtext({**ht, "hl": None, "_i": 0}, 1920, lambda n, c: files.__setitem__(n, c)).count("drawtext=") == 1 and files["t0.txt"] == ht["text"]
    wb = word_box(ht, 1080, 1920, "DÖNER")
    assert wb and wb["x"] < 0.5 and abs(wb["y"] - (0.8 - LINE_H * 134 / 2 / 1920)) < 1e-3 and word_box(ht, 1080, 1920, "PIZZA") is None
    hs = {**ht, "sticker": {"media": "e.webm", "word": "döner", "scale": 1.0}}
    so = sticker_overlay(hs, 1080, 1920)
    assert so["_sticker"] == "t1" and so["y"] == wb["y"] and so["out"] == 1 and so["rotation"] == -15 and so["scale"] == round(161 / 1080, 4) and so["keyframes"]
    d1, m1 = word_box(hs, 1080, 1920, "DÖNER"), word_box(hs, 1080, 1920, "MACHT")
    assert d1["x"] + d1["w"] / 2 < so["x"] - so["scale"] / 2 and so["x"] + so["scale"] / 2 < m1["x"] - m1["w"] / 2  # the slot sits between the words
    assert word_box(hs, 1080, 1920, "MACHT")["x"] > word_box(ht, 1080, 1920, "MACHT")["x"] and sticker_overlay({**hs, "sticker": {"media": "e.webm", "word": "x"}}, 1080, 1920) is None
    assert drawtext({**hs, "hl": None, "_i": 5}, 1920).count("drawtext=") == 3  # the slot itself draws nothing
    print("selftest ok")


def rendertest():
    """Render a 2 s synthetic clip through keyframes, curve speed, mask and text animations; asserts ffmpeg exit 0."""
    d = Path(tempfile.gettempdir()) / "clipcut_rendertest"
    d.mkdir(exist_ok=True)
    run([FF, "-y", "-f", "lavfi", "-i", "testsrc=s=320x240:r=30:d=2", "-f", "lavfi", "-i", "sine=f=440:d=2",
         "-pix_fmt", "yuv420p", "-c:v", "libx264", "-shortest", str(d / "src.mp4")])
    P = {"canvas": [320, 240], "fps": 30, "fill": "contain",
         "adjust": {"vig": 0.5, "temp": 0.3}, "export": {"short": 200, "crf": 23},
         "video": [{"media": "src.mp4", "in": 0, "out": 1, "speed": 1, "speedCurve": [{"t": 0, "speed": 0.5}, {"t": 1, "speed": 2}],
                    "keyframes": [{"t": 0, "scale": 1, "opacity": 0}, {"t": 0.5, "scale": 1.3, "x": 0.1, "rotation": 10, "opacity": 1}],
                    "eyes": [[0, 0.4, 0.4, 0.6, 0.4], [0.5, 0.42, 0.42, 0.62, 0.42]], "eye_fx": {"color": "#40c0ff", "angle": 20},
                    "adjust": {"bri": 0.1, "con": 0.2, "sat": -0.3, "sharp": 0.5, "grain": 0.3}, "effect": "glitch", "crop": {"x": 0.1, "y": 0.1, "w": 0.8, "h": 0.8},
                    "stab": True, "denoise": True, "pitch": 3},
                   {"media": "src.mp4", "in": 1, "out": 2, "speed": 1, "transition": {"type": "pullin", "dur": 0.3}, "opacity": 0.8, "filter": "dark",
                    "eyes": [[0, 0.3, 0.5, 0.5, 0.5]], "effect": "mirror"},
                   {"media": "src.mp4", "in": 0.2, "out": 1, "speed": 1, "transition": {"type": "whipleft", "dur": 0.2}},
                   {"media": "src.mp4", "in": 1, "out": 1.6, "speed": 1}],
         "overlay": [{"media": "src.mp4", "start": 0.2, "in": 0, "out": 1, "scale": 0.4, "x": 0.5, "y": 0.5, "rotation": 90, "chroma": {"color": "#00ff00", "similarity": 0.3, "blend": 0.1},
                      "mask": {"type": "circle", "feather": 0.05}, "keyframes": [{"t": 0, "scale": 0.5, "opacity": 0.2}, {"t": 0.8, "scale": 1, "rotation": 45}]}],
         "audio": [], "text": [{"text": "POP", "start": 0.1, "dur": 1, "font": "impact", "shadow": True, "box": True, "boxcolor": "#204080", "anim_in": {"type": "pop", "dur": 0.3}, "anim_out": {"type": "shake", "dur": 0.3}},
                               {"text": "DÖNER MACHT\nSCHÖNER", "start": 0.5, "dur": 1, "y": 0.6, "font": "impact", "shadow": True, "hl": "DÖNER", "hlcolor": "#ffd60a", "hlscale": 1.3, "sticker": {"media": "src.mp4", "word": "SCHÖNER"}, "anim_in": {"type": "pop", "dur": 0.2}, "anim_out": {"type": "fade", "dur": 0.2}},
                               {"text": "type this", "start": 0.3, "dur": 1, "y": 0.3, "anim_in": {"type": "typewriter", "dur": 0.5}, "anim_out": {"type": "slide", "dur": 0.3}}]}
    total = render(P, d, str(d / "out.mp4"))
    got = probe(str(d / "out.mp4"))["dur"]
    assert abs(got - total) < 0.3, (got, total)
    print(f"rendertest ok: {d / 'out.mp4'} {got:.2f}s")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("scenes"); s.add_argument("inp"); s.add_argument("--threshold", type=float, default=0.35); s.add_argument("--split")
    s = sp.add_parser("search"); s.add_argument("inp"); s.add_argument("subs"); s.add_argument("query"); s.add_argument("out")
    s.add_argument("--keep", action="store_true"); s.add_argument("--pad", type=float, default=0.0)
    s = sp.add_parser("audio"); s.add_argument("inp"); s.add_argument("out")
    s = sp.add_parser("rotate"); s.add_argument("inp"); s.add_argument("out"); s.add_argument("deg", type=float)
    s = sp.add_parser("combine"); s.add_argument("out"); s.add_argument("inputs", nargs="+")
    s.add_argument("--transition", default="fade"); s.add_argument("--dur", type=float, default=0.5)
    s = sp.add_parser("beats"); s.add_argument("inp")
    s = sp.add_parser("render"); s.add_argument("project"); s.add_argument("out")
    s = sp.add_parser("autoedit"); s.add_argument("song"); s.add_argument("out"); s.add_argument("clips", nargs="+")
    s.add_argument("--every", type=int, default=2); s.add_argument("--portrait", action="store_true")
    s = sp.add_parser("tiktok"); s.add_argument("song"); s.add_argument("clips", nargs="+"); s.add_argument("--style", default="auto")
    s.add_argument("--text"); s.add_argument("--hook", type=float); s.add_argument("--every", type=int); s.add_argument("--pace", type=float, default=0.4)
    s.add_argument("--dumb", action="store_true", help="old round-robin clip choice instead of motion/beat scoring")
    s.add_argument("--look", help="whole-video look (FILTERS key) instead of the style's default")
    s.add_argument("--drop", type=float, help="song drop in seconds (overrides the beat analysis)")
    s = sp.add_parser("copy", help="re-create a TikTok edit (URL or mp4) with your clips"); s.add_argument("ref"); s.add_argument("clips", nargs="+")
    s.add_argument("--text"); s.add_argument("--pace", type=float, default=0.4); s.add_argument("--dumb", action="store_true")
    s.add_argument("--even", action="store_true", help="snap the reference's cuts to its beat/half-beat grid (within 90 ms), merge shots < 0.2 s")
    s = sp.add_parser("review"); s.add_argument("project", nargs="?", default=str(ROOT / "project.json"))
    s = sp.add_parser("trends"); s.add_argument("sub", choices=["refresh", "show"]); s.add_argument("--query", default="phonk edit"); s.add_argument("--n", type=int, default=10)
    s = sp.add_parser("op"); s.add_argument("args", nargs="+")
    s = sp.add_parser("faces"); s.add_argument("clips", nargs="+")
    s = sp.add_parser("check"); s.add_argument("inp")
    s = sp.add_parser("look"); s.add_argument("inp"); s.add_argument("--n", type=int, default=12)
    sp.add_parser("selftest"); sp.add_parser("rendertest")
    a = p.parse_args()
    try:
        if a.cmd == "scenes":
            cuts = scenes(a.inp, a.threshold)
            print(json.dumps(cuts))
            if a.split:
                split_at(a.inp, cuts, a.split)
        elif a.cmd == "search":
            cues = parse_srt(Path(a.subs).read_text(encoding="utf-8-sig"))
            render_ranges(a.inp, keep_ranges(cues, a.query, probe(a.inp)["dur"], a.keep, a.pad), a.out)
        elif a.cmd == "audio":
            extract_audio(a.inp, a.out)
        elif a.cmd == "rotate":
            rotate(a.inp, a.out, int(a.deg) if a.deg.is_integer() else a.deg)
        elif a.cmd == "combine":
            combine(a.out, a.inputs, a.transition, a.dur)
        elif a.cmd == "beats":
            print(json.dumps(beats(a.inp)))
        elif a.cmd == "render":
            proj = json.loads(Path(a.project).read_text(encoding="utf-8"))
            md = Path(a.project).parent / "media"
            print(render(proj, md if md.is_dir() else ROOT / "media", a.out))  # project files elsewhere still mean media/ here
        elif a.cmd == "autoedit":  # paths are real files; the project file points at them by basename (media/)
            grid = beats(a.song)["beats"]
            proj = autoedit([{"media": Path(c).name, "dur": probe(c)["dur"]} for c in a.clips], grid, a.every, Path(a.song).name)
            if a.portrait:
                proj["canvas"] = [1080, 1920]
            Path(a.out).write_text(json.dumps(proj), encoding="utf-8")
            print(f"{len(proj['video'])} slices, {proj['audio'][0]['out']:.1f}s -> {a.out}")
        elif a.cmd == "tiktok":
            P = tiktok(a.song, a.clips, a.style, a.text, a.hook, a.every, a.pace, dumb=a.dumb, look=a.look, drop=a.drop)
            print(f"{len(P['video'])} clips, {P['audio'][-1]['start'] + P['audio'][-1]['out'] - P['audio'][-1]['in']:.1f}s -> project.json")
        elif a.cmd == "copy":
            P = copy_edit(a.ref, a.clips, a.text, a.pace, dumb=a.dumb, even=a.even)
            print(f"{len(P['video'])} shots, {P['audio'][0]['out']:.1f}s -> project.json")
        elif a.cmd == "trends":  # what do popular TikTok edits look like right now -> style 'trending'
            import tiktok_trends
            print(json.dumps(tiktok_trends.refresh(a.query, a.n) if a.sub == "refresh" else tiktok_trends.show(), indent=1))
        elif a.cmd == "op":
            print(op(a.args))
        elif a.cmd == "faces":
            print(json.dumps(build_face_ref(a.clips)))
        elif a.cmd == "check":
            import faces
            r = star_ranges(a.inp)
            if r is None:
                sys.exit("no face_ref.json yet: run  clipcut.py faces CLIP...")
            d = probe(a.inp)["dur"]
            print(json.dumps({"ranges": r, "visible_pct": round(100 * sum(b - x for x, b in r) / d), "dur": round(d, 2),
                              "sheet": faces.contact(a.inp, 12, str(Path(tempfile.gettempdir()) / (Path(a.inp).stem + "_check.png")), r)}))
        elif a.cmd == "review":  # what will the Reel actually show? first/middle/last frame per shot at the rendered crop, star box drawn
            import review
            proj = json.loads(Path(a.project).read_text(encoding="utf-8"))
            md = Path(a.project).parent / "media"
            r = review.review(proj, md if md.is_dir() else ROOT / "media", out=str(Path(tempfile.gettempdir()) / f"review_{Path(a.project).stem}.png"))
            print(json.dumps(r, ensure_ascii=False))
        elif a.cmd == "look":
            import faces
            print(faces.contact(a.inp, a.n, str(Path(tempfile.gettempdir()) / (Path(a.inp).stem + "_look.png"))))
        elif a.cmd == "rendertest":
            rendertest()
        else:
            selftest()
    except subprocess.CalledProcessError as e:
        sys.exit(e.stderr[-2000:])


if __name__ == "__main__":
    main()
