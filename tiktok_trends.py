"""tiktok_trends.py - how are popular TikTok "edits" actually cut right now? Measure, view-weight, turn into a template.

fetch_candidates(query, n)  -> [{url, views, title}]      TikTok video URLs + view counts (see the ponytail note inside)
download(cands, n, dest)    -> [paths]                     yt-dlp into media/_trends/ (UNTRUSTED files: only ffmpeg/opencv read them)
fingerprint(path)           -> knobs dict                  cuts per beat, flashes, zoom punches, slow-mo, look, text presence
template(fps)               -> view-weighted average       -> media/_trends/trend_template.json, used by trends.py as style "trending"
refresh(query, n) / show()  <- `python clipcut.py trends refresh|show`
python tiktok_trends.py [CLIP]  -> selftest: fingerprint one local clip and build a template from it.
"""
import json, re, subprocess, sys, time, urllib.parse, urllib.request
from pathlib import Path
# ponytail: Windows console is cp1252, TikTok titles have emoji
if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(errors='replace')

import numpy as np

ROOT = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).parent  # exe: media/ lives next to it
TREND_DIR = ROOT / "media" / "_trends"
TEMPLATE = TREND_DIR / "trend_template.json"
URLS = TREND_DIR / "urls.txt"  # one TikTok video URL per line; the Chrome extension or the user can fill it
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36",
      "Accept-Language": "de-DE,de;q=0.9,en;q=0.8"}
VIDEO_URL = re.compile(r"https?://(?:www\.)?tiktok\.com/@[\w.\-]+/video/\d+")


def _ytdlp(*args, timeout=180):
    """yt-dlp in-process (works inside the PyInstaller exe too); mimics subprocess.run's result for the 3 call shapes used here:
    -J url  /  --flat-playlist -J url  /  -f FMT --max-filesize N -o TMPL url.   # ponytail: timeout is socket-level only"""
    import io, types
    args = list(args); url = args[-1]
    opts = {"quiet": True, "no_warnings": True, "noplaylist": True, "socket_timeout": 20, "paths": {"home": str(ROOT)}}
    if "--flat-playlist" in args:
        opts["extract_flat"] = True; opts["noplaylist"] = False
    if "--playlist-end" in args:
        opts["playlistend"] = int(args[args.index("--playlist-end") + 1])
    if "-f" in args:
        opts["format"] = args[args.index("-f") + 1]
    if "--max-filesize" in args:
        opts["max_filesize"] = int(float(args[args.index("--max-filesize") + 1].rstrip("mM")) * 1024 * 1024)
    if "-o" in args:
        opts["outtmpl"] = args[args.index("-o") + 1]
    err = io.StringIO()
    try:
        import yt_dlp
        class _Log:  # yt-dlp >= 2026 hashes the logger; SimpleNamespace is unhashable
            def debug(self, m): pass
            def info(self, m): pass
            def warning(self, m): pass
            def error(self, m): err.write(str(m) + "\n")
        with yt_dlp.YoutubeDL({**opts, "logger": _Log()}) as y:
            info = y.extract_info(url, download="-J" not in args)
            out = json.dumps(y.sanitize_info(info)) if "-J" in args else ""
        return types.SimpleNamespace(returncode=0, stdout=out, stderr=err.getvalue())
    except Exception as e:
        return types.SimpleNamespace(returncode=1, stdout="", stderr=err.getvalue() + str(e))


def _get(url, timeout=25):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read().decode("utf-8", "replace")


# ---------- 1. source list ----------
def fetch_candidates(query="phonk edit", n=30, log=print):
    """TikTok video URLs for `query`, with view counts when yt-dlp can read them.
    # ponytail: TikTok markup changes; swap source here. Status 2026-10-14:
    #   (a) yt-dlp --flat-playlist on /tag/<x>  -> "No working app info is available" (extractor marked broken); /search unsupported
    #       plain urllib GET of /tag/<x> returns 200 but the rehydration JSON has no items (loaded by a signed XHR)
    #   (b) yt-dlp -J on a single /@user/video/ID URL works -> view_count, duration, download
    #   so URLs come from: media/_trends/urls.txt (paste from the Chrome extension / browser) + Bing/DDG HTML results for
    #   site:tiktok.com <query>. Creative Center API answers {"code":40101,"msg":"no permission"} without signed headers.
    """
    urls = []
    if URLS.exists():
        urls += VIDEO_URL.findall(URLS.read_text(encoding="utf-8", errors="replace"))
        log(f"{len(urls)} URLs from {URLS.name}")
    tag = query.split()[-1].replace("#", "")
    r = _ytdlp("--flat-playlist", "-J", "--playlist-end", str(n), f"https://www.tiktok.com/tag/{tag}", timeout=90)
    if r.returncode == 0:
        try:
            es = json.loads(r.stdout).get("entries") or []
            got = [{"url": e.get("url") or e.get("webpage_url"), "views": e.get("view_count") or 0, "title": e.get("title") or ""} for e in es if e]
            if got:
                log(f"{len(got)} from tiktok.com/tag/{tag} via yt-dlp"); return got[:n]
        except json.JSONDecodeError:
            pass
    log(f"tiktok.com/tag/{tag} via yt-dlp: {(r.stderr or '').strip().splitlines()[-1:] or 'nothing'}")
    q = urllib.parse.quote(f'site:tiktok.com {query}')
    for name, u in [("bing", f"https://www.bing.com/search?q={q}&setlang=de&count=50"), ("ddg", f"https://html.duckduckgo.com/html/?q={q}")]:
        try:
            found = set(VIDEO_URL.findall(urllib.parse.unquote(_get(u))))
            log(f"{len(found)} URLs from {name}"); urls += sorted(found)
        except Exception as e:  # 2 sources, one try each; never loop on a blocked network
            log(f"{name}: {e}")
    urls = list(dict.fromkeys(urls))[:n]
    cands, fails = [], 0
    for u in urls:  # yt-dlp -J per video for view_count (~2 s each)
        if fails >= 3:
            log("3 yt-dlp failures in a row, stopping metadata lookups"); break
        r = _ytdlp("-J", u, timeout=60)
        if r.returncode != 0:
            fails += 1; log(f"yt-dlp -J failed: {u} {(r.stderr or '')[-120:].strip()}"); continue
        fails = 0
        d = json.loads(r.stdout)
        cands.append({"url": u, "views": d.get("view_count") or 0, "title": (d.get("title") or "")[:80], "dur": d.get("duration")})
    return cands


def download(cands, n=10, dest=TREND_DIR, log=print):
    dest.mkdir(parents=True, exist_ok=True)
    paths = []
    for c in sorted(cands, key=lambda c: -(c.get("views") or 0))[:n]:
        vid = c["url"].rstrip("/").rsplit("/", 1)[-1]
        have = [q for q in dest.glob(f"{vid}.*") if q.suffix != ".json"]  # the .json sidecar is not the video
        if not have:
            r = _ytdlp("-f", "mp4/best", "--max-filesize", "60m", "-o", str(dest / "%(id)s.%(ext)s"), c["url"], timeout=240)
            if r.returncode != 0:
                log(f"download failed: {c['url']} {(r.stderr or '')[-120:].strip()}"); continue
            have = [q for q in dest.glob(f"{vid}.*") if q.suffix != ".json"]  # the .json sidecar is not the video
        if have:
            (dest / f"{vid}.json").write_text(json.dumps(c), encoding="utf-8")
            paths.append(have[0]); log(f"{c.get('views', 0):>10} views  {have[0].name}  {c.get('title', '')}")
    return paths


# ---------- 2. fingerprint ----------
def fingerprint(path, log=print):
    import cv2, clipcut
    path = str(path)
    dur = clipcut.probe(path)["dur"]
    cuts = [t for t in clipcut.scenes(path) if t < dur]
    try:
        grid = clipcut.beats(path)
    except Exception:  # no audio track
        grid = {"bpm": 0, "beats": [], "strong": [], "drop": 0}
    W, H = 320, 180
    p = subprocess.Popen([clipcut.FF, "-v", "error", "-i", path, "-vf", f"scale={W}:{H}", "-pix_fmt", "bgr24", "-f", "rawvideo", "-"],
                         stdout=subprocess.PIPE, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    prev = None
    diff, bright, flash, zoom, text, sat, con, grain, warm = [], [], [], [], [], [], [], [], []
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    while True:
        raw = p.stdout.read(W * H * 3)
        if len(raw) < W * H * 3:
            break
        img = np.frombuffer(raw, np.uint8).reshape(H, W, 3)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gi = g.astype(np.int16)
        bright.append(float(g.mean())); con.append(float(g.std()) / 128)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV); sat.append(float(hsv[..., 1].mean()) / 255)
        grain.append(float(np.abs(gi - cv2.GaussianBlur(g, (0, 0), 1.5)).mean()))
        warm.append(float(img[..., 2].mean() - img[..., 0].mean()))
        # text: big bright blobs that are wider than tall (impact captions) - ponytail: no OCR, white text only
        m = cv2.morphologyEx((g > 215).astype(np.uint8), cv2.MORPH_CLOSE, kern)
        nb, _, st, _ = cv2.connectedComponentsWithStats(m)
        text.append(any(0.004 * W * H < a < 0.12 * W * H and 1.5 * h < w < 0.85 * W and 0.03 * H < h < 0.25 * H and 0.25 < a / (w * h) < 0.75
                        or w > 0.6 * W and 0.05 * H < h < 0.3 * H and a / (w * h) > 0.6  # caption banner (dark text on a white box)
                        for x, y, w, h, a in st[1:]))
        d = float(np.abs(gi - prev).mean()) if prev is not None else 0.0
        diff.append(d)
        flash.append(prev is not None and bright[-1] - bright[-2] > 50 and bright[-1] > 170)
        z = False
        if prev is not None and 3 < d < 40:  # zoom punch = same picture, suddenly bigger
            base = d
            for s in (1.06, 1.12, 1.2, 1.3):
                big = cv2.resize(prev.astype(np.uint8), None, fx=s, fy=s)
                y0, x0 = (big.shape[0] - H) // 2, (big.shape[1] - W) // 2
                if float(np.abs(big[y0:y0 + H, x0:x0 + W].astype(np.int16) - gi).mean()) < 0.6 * base:
                    z = True; break
        zoom.append(z)
        prev = gi
    p.wait()
    n = len(diff)
    fps = n / dur if dur else 30
    if n < 10:
        raise ValueError(f"{path}: no frames decoded")
    diff = np.array(diff); keep = diff < 40  # drop cut frames from the speed stats
    win = max(1, int(fps / 2))
    wdt = [(i / fps, diff[i:i + win][keep[i:i + win]].mean()) for i in range(0, n - win, win) if keep[i:i + win].any()]
    wd = [w for _, w in wdt]
    med = float(np.median(wd)) if wd else 0
    dup = float((diff[1:] < 0.8).mean())  # 0.5x without frame blending = every 2nd frame identical
    slow = float(np.mean([w < 0.35 * med for w in wd])) if med else 0
    fast = float(np.mean([w > 2.5 * med for w in wd])) if med else 0
    events = lambda flags: sum(1 for i, f in enumerate(flags) if f and not (i and flags[i - 1]))
    beats_in = [b for b in grid["beats"] if b < dur]
    strong = [b for b in grid["strong"] if b < dur]
    synced = sum(1 for c in cuts if beats_in and min(abs(c - b) for b in beats_in) < 0.08)
    first_cut = cuts[0] if cuts else dur
    bw = float(np.mean(sat)) < 0.08
    look = ("bw" if bw else "phonk" if np.mean(con) > 0.42 and np.mean(sat) < 0.3 else
            "warm" if np.mean(warm) > 12 else "cool" if np.mean(warm) < -12 else "none")
    fp = {"dur": round(dur, 2), "bpm": grid["bpm"], "n_cuts": len(cuts), "n_beats": len(beats_in), "shot_s": round(float(np.median(np.diff([0, *cuts, dur]))), 2),
          "cut_every_beats": int(np.clip(round(len(beats_in) / max(1, len(cuts))), 1, 4)),
          "beat_sync_pct": round(synced / max(1, len(cuts)), 2),
          "flash_on_accents_pct": round(min(1.0, events(flash) / max(1, len(strong))), 2),
          "zoom_punch_pct": round(min(1.0, events(zoom) / max(1, len(cuts))), 2),
          "slowmo_pct": round(max(dup, slow), 2), "speed_pct": round(fast, 2),
          "hook_s": round(first_cut, 2) if cuts and 1.5 <= first_cut <= 4.5 else 0.0,  # long first shot = talking hook
          "contrast": round(float(np.mean(con)), 3), "saturation": round(float(np.mean(sat)), 3), "grain": round(float(np.mean(grain)), 2),
          "bw": bw, "look": look, "text_pct": round(float(np.mean(text)), 2)}
    ev = lambda flags: [round(i / fps, 3) for i, f in enumerate(flags) if f and not (i and flags[i - 1])]
    fp["events"] = {"cuts": [round(c, 3) for c in cuts], "flashes": ev(flash), "zooms": ev(zoom),  # copy_edit.py replays these 1:1
                    "slow": [[round(t, 2), round(t + win / fps, 2)] for t, w in wdt if med and w < 0.35 * med],
                    "beats": [round(b, 3) for b in beats_in], "strong": [round(b, 3) for b in strong], "bpm": grid["bpm"]}
    log(f"{Path(path).name}: " + str({k: v for k, v in fp.items() if k != 'events'}))
    return fp


# ---------- 3. template ----------
NUM = ["cut_every_beats", "beat_sync_pct", "flash_on_accents_pct", "zoom_punch_pct", "slowmo_pct", "speed_pct", "hook_s", "contrast",
       "saturation", "grain", "text_pct", "shot_s"]


def template(fps, weights=None):
    """View-weighted average of the knobs. # ponytail: raw views as weights, one 50M-view video dominates - that is the point."""
    w = np.array([float(x or 0) for x in (weights or [1] * len(fps))]); w = w if w.sum() > 0 else np.ones(len(fps))
    k = {key: round(float(np.average([f[key] for f in fps], weights=w)), 3) for key in NUM}
    k["cut_every_beats"] = int(np.clip(round(k["cut_every_beats"]), 1, 4))
    votes = {}
    for f, wi in zip(fps, w):
        votes[f["look"]] = votes.get(f["look"], 0) + wi
    k["look"] = max(votes, key=votes.get) if votes else "phonk"
    k["bw"] = k["look"] == "bw"
    k["shake_on_accents"] = k["zoom_punch_pct"] > 0.15
    return k


def refresh(query="phonk edit", n=10, log=print):
    TREND_DIR.mkdir(parents=True, exist_ok=True)
    cands = fetch_candidates(query, max(n * 3, 20), log)
    paths = download(cands, n, TREND_DIR, log) if cands else []
    local = [p for p in TREND_DIR.glob("*.mp4") if p not in paths]
    if local:
        log(f"{len(local)} local mp4(s) in {TREND_DIR.name} (dropped in by hand) included")
    paths = [*paths, *local]
    if not paths:
        raise SystemExit(f"no trend videos: drop mp4s into {TREND_DIR} or URLs into {URLS}")
    fps, views = [], []
    for p in paths:
        meta = TREND_DIR / f"{p.stem}.json"
        try:
            fps.append(fingerprint(p, log))
        except Exception as e:
            log(f"{p.name}: {e}"); continue
        v = json.loads(meta.read_text(encoding="utf-8")).get("views", 0) if meta.exists() else 0
        views.append(v); fps[-1]["source"] = p.name; fps[-1]["views"] = v
    # ponytail: tag pages mix audio posts / single-shot memes with real edits; only average things that are cut like an edit
    edits = [f for f in fps if f["shot_s"] < 2.0 and f["n_cuts"] >= 4] or fps
    knobs = template(edits, [f["views"] for f in edits] if any(f["views"] for f in edits) else None)
    out = {"created": time.strftime("%Y-%m-%d %H:%M"), "query": query, "knobs": knobs, "sources": fps,
           "used": [f["source"] for f in edits]}
    TEMPLATE.write_text(json.dumps(out, indent=1), encoding="utf-8")
    log(f"-> {TEMPLATE} from {len(fps)} video(s): {knobs}")
    return out


def show():
    if not TEMPLATE.exists():
        return {"error": f"no template yet: python clipcut.py trends refresh (or drop mp4s into {TREND_DIR})"}
    return json.loads(TEMPLATE.read_text(encoding="utf-8"))


def selftest(clip=None):
    clip = clip or next(iter(sorted((ROOT / "media").glob("*slidet*.mp4"))), None)
    assert clip, "need a clip"
    fp = fingerprint(clip)
    assert 1 <= fp["cut_every_beats"] <= 4 and 0 <= fp["text_pct"] <= 1 and fp["look"] in ("bw", "phonk", "warm", "cool", "none"), fp
    k = template([fp, {**fp, "zoom_punch_pct": 1.0, "look": "bw"}], [1, 3])
    assert k["look"] == "bw" and k["bw"] and abs(k["zoom_punch_pct"] - (fp["zoom_punch_pct"] + 3) / 4) < 0.01, k
    print("tiktok_trends selftest ok")


if __name__ == "__main__":
    selftest(sys.argv[1] if len(sys.argv) > 1 else None)
