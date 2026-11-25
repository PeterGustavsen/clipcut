"""Luminar Videos editor: python server.py  ->  http://127.0.0.1:8777"""
import json, os, re, shutil, sys, time, traceback
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import threading

import clipcut, trends

ROOT = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).parent  # exe: data next to it
STATIC = Path(getattr(sys, "_MEIPASS", ROOT))  # index.html + onnx model live inside the bundle
MEDIA = ROOT / "media"
MEDIA.mkdir(exist_ok=True)
PROJECT = ROOT / "project.json"
AGENT = ROOT / "agent.jsonl"
clipcut.ROOT = ROOT


def last_say():
    if not AGENT.exists():
        return None
    lines = AGENT.read_text(encoding="utf-8").splitlines()
    return json.loads(lines[-1]) if lines else None


def grid_for(name):
    src = media_path(name)
    cache = TRANSCRIPTS / f"{src.name}.{int(src.stat().st_mtime)}.beats.json"
    if not cache.exists():
        cache.write_text(json.dumps(clipcut.beats(str(src))), encoding="utf-8")
    return json.loads(cache.read_text(encoding="utf-8"))
TRANSCRIPTS = ROOT / "transcripts"
TRANSCRIPTS.mkdir(exist_ok=True)
IMAGES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_probe = {}


def media_path(name):
    p = MEDIA / Path(str(name)).name  # basename only: requests can't reach outside media/
    if not p.is_file():
        raise FileNotFoundError(f"no such media: {name}")
    return p


def info(name):
    p = media_path(name)
    key = (p.name, p.stat().st_mtime)
    if key not in _probe:
        i = clipcut.probe(str(p))
        v = i["video"] or {}
        is_song = p.suffix.lower() in {".mp3", ".m4a", ".aac", ".wav", ".flac", ".ogg", ".opus"}  # cover art is not a video track
        _probe[key] = {"name": p.name, "dur": i["dur"], "audio": bool(i["audio"]), "video": bool(i["video"]) and not is_song,
                       "width": v.get("width"), "height": v.get("height")}
    return _probe[key]


def job_dir(job):
    if not re.fullmatch(r"[a-z0-9]{8,32}", job):
        raise ValueError("bad job id")
    return ROOT / "tmp" / job


def unique(name):
    stem, suf = Path(name).stem, Path(name).suffix
    stem = re.sub(r"[^\w\- ]", "_", stem)[:60] or "clip"
    p, n = MEDIA / f"{stem}{suf}", 1
    while p.exists():
        p, n = MEDIA / f"{stem}_{n}{suf}", n + 1
    return p


class H(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(STATIC), **k)

    def log_message(self, *a):
        pass

    def send_json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        try:
            self.get()
        except Exception as e:
            traceback.print_exc()
            self.send_json({"error": str(e)}, 500)

    def get(self):
        u = urlparse(self.path)
        if u.path == "/api/media":
            return self.send_json([info(p.name) for p in sorted(MEDIA.iterdir()) if p.is_file() and not p.name.startswith("_")])  # _eyes_* = render caches
        if u.path == "/api/project":  # mtime lets the UI notice edits made by Claude / the CLI and reload live
            if not PROJECT.exists():
                return self.send_json({"mtime": 0, "project": None, "say": last_say()})
            return self.send_json({"mtime": self.stamp(), "project": json.loads(PROJECT.read_text(encoding="utf-8")), "say": last_say()})
        if u.path == "/api/project-mtime":
            return self.send_json(self.stamp())
        if u.path.startswith("/media/"):
            return self.send_media(unquote(u.path[7:]))
        if re.fullmatch(r"/fonts/[\w-]+\.ttf", u.path):
            return super().do_GET()
        if u.path in ("/", "/index.html", "/rvm_mobilenetv3_fp32.onnx"):
            return super().do_GET()
        self.send_error(404)

    def send_media(self, name):
        # <video> needs Range requests to seek; SimpleHTTPRequestHandler doesn't do them
        try:
            p = media_path(name)
        except FileNotFoundError:
            return self.send_error(404)
        size = p.stat().st_size
        m = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        start, end = 0, size - 1
        if m and (m[1] or m[2]):
            start, end = (int(m[1]), int(m[2]) if m[2] else size - 1) if m[1] else (size - int(m[2]), size - 1)
            end = min(end, size - 1)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", self.guess_type(str(p)))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with open(p, "rb") as f:
            f.seek(start)
            left = end - start + 1
            try:
                while left > 0:
                    chunk = f.read(min(1 << 20, left))
                    self.wfile.write(chunk)
                    left -= len(chunk)
            except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                pass  # browser cancelled the range, normal while seeking

    def transcript(self, name):
        src = media_path(name)
        cache = TRANSCRIPTS / f"{src.name}.{int(src.stat().st_mtime)}.json"
        if not cache.exists():
            cache.write_text(json.dumps(clipcut.transcribe(str(src))), encoding="utf-8")
        return json.loads(cache.read_text(encoding="utf-8"))

    def words(self, name):
        return self.transcript(name)["words"]

    def stamp(self):  # newest of project.json / agent.jsonl: either one changing means "show something new"
        return max((f.stat().st_mtime for f in (PROJECT, AGENT) if f.exists()), default=0)

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length", 0))
        try:
            if u.path == "/api/upload":
                q = parse_qs(u.query)
                p = unique(q.get("name", ["clip.mp4"])[0])
                with open(p, "wb") as f:
                    while n > 0:
                        chunk = self.rfile.read(min(1 << 20, n))
                        if not chunk:
                            break
                        f.write(chunk)
                        n -= len(chunk)
                if p.suffix.lower() in IMAGES:  # CapCut treats photos as still clips; ours become 5 s videos (PNG: 30 s transparent webm)
                    png = p.suffix.lower() == ".png"
                    mp4 = unique(p.stem + (".webm" if png else ".mp4"))
                    try:
                        clipcut.run([clipcut.FF, "-y", "-loop", "1", "-i", str(p), "-t", "30" if png else "5", "-r", "30",
                                     "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=" + ("yuva420p" if png else "yuv420p")]
                                    + (["-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-b:v", "0", "-crf", "30", "-row-mt", "1", "-deadline", "realtime", "-cpu-used", "8"]
                                       if png else ["-c:v", "libx264"]) + [str(mp4)])
                    finally:
                        p.unlink()
                    p = mp4
                return self.send_json(info(p.name))
            if u.path == "/api/frames":  # one matted frame of a background-removal job
                q = parse_qs(u.query)
                d, ext = job_dir(q["job"][0]), q.get("ext", ["jpg"])[0]
                if ext not in ("jpg", "png"):
                    raise ValueError("bad frame type")
                d.mkdir(parents=True, exist_ok=True)
                (d / f"{int(q['i'][0]):06d}.{ext}").write_bytes(self.rfile.read(n))
                return self.send_json({"ok": True})
            body = json.loads(self.rfile.read(n) or b"{}")
            if u.path == "/api/frames-done":
                d, src = job_dir(body["job"]), media_path(body["media"])
                if body.get("alpha"):  # transparent cut-out for the overlay track: VP9 keeps the alpha channel
                    out, frames = unique(f"{src.stem}_cutout.webm"), "%06d.png"
                    enc = ["-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-b:v", "0", "-crf", "28", "-row-mt", "1",
                           "-deadline", "realtime", "-cpu-used", "8", "-c:a", "libopus", "-b:a", "160k"]
                else:
                    out, frames = unique(f"{src.stem}_nobg.mp4"), "%06d.jpg"
                    enc = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
                           "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart"]
                try:
                    clipcut.run([clipcut.FF, "-y", "-framerate", str(body["fps"]), "-i", str(d / frames),
                                 "-ss", str(body["in"]), "-to", str(body["out"]), "-i", str(src),
                                 "-map", "0:v", "-map", "1:a?", *enc, "-shortest", str(out)])
                finally:
                    shutil.rmtree(d, ignore_errors=True)
                return self.send_json(info(out.name))
            if u.path == "/api/transcribe":
                return self.send_json(self.transcript(body["media"]))
            if u.path == "/api/project":
                clipcut.write_atomic(PROJECT, json.dumps(body))
                return self.send_json({"mtime": self.stamp()})
            if u.path == "/api/eyes":  # {seg, canvas, fill} -> {eyes: [[t, x1, y1, x2, y2]]} face-tracked eye centres for the laser eyes
                import fxface
                return self.send_json({"eyes": fxface.eyes(body["seg"], {"canvas": body.get("canvas"), "fill": body.get("fill", "cover")}, MEDIA)})
            if u.path == "/api/beats":
                return self.send_json(grid_for(body["media"]))
            if u.path == "/api/faces":  # {clips:[names]} -> learn the star from these clips
                return self.send_json(clipcut.build_face_ref([str(media_path(c)) for c in body["clips"]], ROOT, TRANSCRIPTS))
            if u.path == "/api/check":  # {media} -> ranges where the star is on screen
                r = clipcut.star_ranges(str(media_path(body["media"])), ROOT, TRANSCRIPTS)
                return self.send_json({"ranges": r, "pct": round(100 * sum(b - a for a, b in r) / info(body["media"])["dur"]) if r else None})
            if u.path == "/api/say":
                clipcut.say(body["msg"], ROOT)
                return self.send_json({"ok": True})
            if u.path == "/api/tiktok":  # {song, clips, style, text, hook, every, pace} -> steps play out in a thread
                grid = grid_for(body["song"])
                infos = [{"media": c, "dur": info(c)["dur"]} for c in body["clips"]]
                if clipcut.face_ref(ROOT) is None and len(body["clips"]) > 1:  # first run: learn who the star is
                    clipcut.build_face_ref([str(media_path(c)) for c in body["clips"]], ROOT, TRANSCRIPTS)
                for i in infos:
                    r = clipcut.star_ranges(str(media_path(i["media"])), ROOT, TRANSCRIPTS)
                    if r is not None:
                        i["ranges"] = r
                    if not body.get("dumb") and r != []:  # motion/loudness moments -> trends picks gestures on the beat (motion.Picker)
                        try:
                            import motion
                            i["moments"] = motion.moments(str(media_path(i["media"])), TRANSCRIPTS)
                            if r is not None:
                                i["star"] = clipcut.star_track(str(media_path(i["media"])), ROOT, TRANSCRIPTS)  # face per 1/4 s: verified windows, crop on him
                            i["cuts"] = clipcut.scene_cuts(str(media_path(i["media"])), TRANSCRIPTS)
                        except Exception as e:
                            print(f"moments: {i['media']}: {e}")
                hook = body.get("hook") or None
                if hook is None and body.get("style", "auto") in ("auto", "ironic"):
                    try:
                        hook = clipcut.hook_len(str(media_path(body["clips"][0])), words=self.words(body["clips"][0]))
                    except Exception as e:
                        print("hook:", e)
                steps = trends.plan(body.get("style", "auto"), infos, grid, media_path(body["song"]).name,
                                    text=body.get("text") or None, hook=hook, every=body.get("every") or None)
                def go(steps=steps):
                    clipcut.apply_steps(clipcut.with_review(steps, infos, ROOT), ROOT, float(body.get("pace", 0.6)))  # review.py re-picks weak shots first
                threading.Thread(target=go, daemon=True).start()
                return self.send_json({"steps": len(steps)})
            if u.path == "/api/autoedit":  # {song, clips:[name], every} -> project (ids assigned here)
                src = media_path(body["song"])
                g = grid_for(src.name); grid = g["beats"]
                proj = clipcut.autoedit([{"media": c, "dur": info(c)["dur"]} for c in body["clips"]], grid,
                                        int(body.get("every", 2)), src.name)
                for k in ("video", "audio"):
                    for x in proj[k]:
                        x["id"] = os.urandom(4).hex()
                proj["bpm"] = g["bpm"]
                return self.send_json(proj)
            if u.path == "/api/copy":  # {ref: tiktok url | mp4 name, clips, text, pace, dumb} -> same edit with your footage, built step by step
                import copy_edit
                ref = body["ref"].strip()
                infos = [{"media": c, "dur": info(c)["dur"]} for c in body["clips"]]
                for i in infos:
                    r = clipcut.star_ranges(str(media_path(i["media"])), ROOT, TRANSCRIPTS)
                    if r is not None:
                        i["ranges"] = r
                    if not body.get("dumb") and r != []:
                        try:
                            import motion
                            i["moments"] = motion.moments(str(media_path(i["media"])), TRANSCRIPTS)
                            if r is not None:
                                i["star"] = clipcut.star_track(str(media_path(i["media"])), ROOT, TRANSCRIPTS)
                            i["cuts"] = clipcut.scene_cuts(str(media_path(i["media"])), TRANSCRIPTS)
                        except Exception as e:
                            print(f"moments: {i['media']}: {e}")
                def go():
                    try:
                        R = copy_edit.reference(ref if ref.startswith("http") else str(media_path(ref)), log=lambda m: clipcut.say(str(m)[:200], ROOT))
                        steps = copy_edit.plan(R, infos, text=body.get("text") or None, even=bool(body.get("even")))  # even: cuts snapped to the beat grid
                        clipcut.apply_steps(clipcut.with_review(steps, infos, ROOT), ROOT, float(body.get("pace", 0.6)))
                    except BaseException as e:
                        clipcut.say(f"Copy edit: {e}", ROOT)
                threading.Thread(target=go, daemon=True).start()
                return self.send_json({"started": True})
            if u.path == "/api/trends":  # {sub: refresh|show, query} -> refresh runs in a thread, progress via the agent toast
                import tiktok_trends
                if body.get("sub") == "refresh":
                    def go():
                        try:
                            tiktok_trends.refresh(body.get("query") or "phonk edit", log=lambda m: clipcut.say(str(m)[:200], ROOT))
                        except BaseException as e:  # SystemExit carries the "no trend videos" message
                            clipcut.say(f"Trends: {e}", ROOT)
                    threading.Thread(target=go, daemon=True).start()
                    return self.send_json({"started": True})
                return self.send_json(tiktok_trends.show())
            if u.path == "/api/trend-urls":  # {urls: "one per line"} -> media/_trends/urls.txt (appended, deduplicated)
                import tiktok_trends
                tiktok_trends.TREND_DIR.mkdir(parents=True, exist_ok=True)
                have = tiktok_trends.URLS.read_text(encoding="utf-8").split() if tiktok_trends.URLS.exists() else []
                new = [l.strip() for l in str(body.get("urls", "")).splitlines() if l.strip().startswith("http") and l.strip() not in have]
                with open(tiktok_trends.URLS, "a", encoding="utf-8") as fh:
                    fh.write("".join(l + "\n" for l in new))
                return self.send_json({"added": len(new), "total": len(have) + len(new)})
            if u.path == "/api/scenes":
                return self.send_json(clipcut.scenes(str(media_path(body["media"])), float(body.get("threshold", 0.35))))
            if u.path == "/api/extract-audio":
                src = media_path(body["media"])
                out = unique(f"{src.stem}_audio.{body.get('format', 'm4a')}")
                clipcut.extract_audio(str(src), str(out))
                return self.send_json(info(out.name))
            if u.path == "/api/render":
                kind = "compound" if body.get("compound") else "export"
                out = unique(f"{kind}_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
                clipcut.render(body, MEDIA, str(out))
                return self.send_json(info(out.name))
            self.send_error(404)
        except SystemExit as e:  # clipcut CLI helpers exit with a message
            self.send_json({"error": str(e)}, 400)
        except Exception as e:
            traceback.print_exc()
            msg = e.stderr[-1500:] if hasattr(e, "stderr") and e.stderr else str(e)
            self.send_json({"error": msg}, 500)


class Server(ThreadingHTTPServer):
    allow_reuse_address = False  # on Windows reuse lets a second server silently share a busy port
    daemon_threads = True


PORT = 8777

if __name__ == "__main__":
    print(f"Luminar Videos editor on http://127.0.0.1:{PORT}")
    Server(("127.0.0.1", PORT), H).serve_forever()
