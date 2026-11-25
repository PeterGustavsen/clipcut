"""fxface.py - CapCut-style face effects for the auto edits: glowing eyes, a spinning image behind the head, text behind a person.

eyes(seg, P, media_dir, root)   -> [[t, x1, y1, x2, y2], ...]  eye centres (canvas fractions, t = s after the shot starts) from YuNet
                                   landmarks on the rendered crop; clipcut.render draws a soft glow there, index.html previews it.
cutout(src, a, b, fps, out)     -> transparent VP9 webm of source seconds a..b (RVM matting via onnxruntime, same model as the
                                   browser's "Remove background"; cached in media/ as <stem>_<a>-<b>_cutout.webm).
sun(P, i, media_dir, root)      -> adds two overlays for main-track shot i: `Sonne.webm` (disc 1.7 face heights) spinning 0.75 turns/s and
                                   FOLLOWING his head (x/y/scale keyframes from track()), then the cut-out of the same slice aligned to the
                                   cover crop on top (3 layers: clip, sun, cut-out).
bake(P, media_dir, root)        -> resolves `eyes: true` / `sun: true` flags left by the planners; returns a step message.
python fxface.py [CLIP [A B]]  -> mattes a 0.3 s slice and prints the webm + an eye track (needs the RVM model + face_ref.json).
"""
import copy, json, shutil, subprocess, sys, tempfile
from pathlib import Path

import cv2, numpy as np

ROOT = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).parent
MODEL = Path(getattr(sys, "_MEIPASS", Path(__file__).parent)) / "rvm_mobilenetv3_fp32.onnx"
# sky segmentation (U2Net trained on skies, 176 MB, too big for git): downloaded on first use. Foreground = everything that is NOT sky,
# so text "behind a building" sits in the sky and buildings, towers, trees and people stand in front of it.
SKY_MODEL = Path(getattr(sys, "_MEIPASS", Path(__file__).parent)) / "models" / "skyseg.onnx"
SKY_URL = "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx"


# no sky in the shot (drone from above, street level without horizon): monocular depth instead (Depth Anything V2 small, 99 MB).
# The building in front is nearer than what is behind it; the near/far split is Otsu on the frame's depth (2026-11-25, Lessing drone shot).
DEPTH_MODEL = SKY_MODEL.parent / "depth_small.onnx"
DEPTH_URL = "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx"


def _session(path, url, what):
    import onnxruntime as ort, urllib.request
    if not path.exists():
        print(f"downloading the {what} model (once) ...", file=sys.stderr)
        path.parent.mkdir(exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _sky_session():
    return _session(SKY_MODEL, SKY_URL, "sky (176 MB)")


def depth_map(sess, img):
    """Relative nearness 0..1 (1 = nearest) at the frame's size."""
    h, w = img.shape[:2]
    H, W = 518, max(14, round(518 * w / h / 14) * 14)
    x = cv2.cvtColor(cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    x = ((x - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)).transpose(2, 0, 1)[None].astype(np.float32)
    d = cv2.resize(sess.run(None, {sess.get_inputs()[0].name: x})[0][0], (w, h), interpolation=cv2.INTER_CUBIC)
    return (d - d.min()) / (d.max() - d.min() + 1e-6)


def sky_alpha(sess, img):
    """Foreground alpha 0..1 (1 = building/tree/person, 0 = sky) at the frame's size."""
    x = cv2.cvtColor(cv2.resize(img, (320, 320), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    x = ((x - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)).transpose(2, 0, 1)[None].astype(np.float32)
    sky = np.clip(sess.run(None, {sess.get_inputs()[0].name: x})[0][0, 0], 0, 1)
    sky = cv2.resize(sky, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    return np.clip((0.5 - sky) / 0.3 + 0.5, 0, 1)  # soft edge around the 0.5 boundary
SUN = "Sonne disc.webm"  # square crop of the PNG with the disc exactly centred, so rotation never wobbles (user 2026-10-16)
STEP = 0.1  # s between eye samples (render interpolates)
# Laser eyes look, after CapCut 9.5's "eyelight" package (REA 2026-10-16, effect cache 7399483006278307077): animated light bars on each
# eye with a hue LUT, tone map 1-exp(-x), saturation up, directional 9-tap blur, additive composite; plus its DeepGlow (8 blur passes,
# Screen blend, tint multiply). Here: white-hot core + tinted halo + wobbling beam, 3-scale Gaussian bloom, tone map, screen blend.
EYE_FX = {"color": "#ff2a2a", "beam": True, "angle": 72, "strength": 1.0}  # angle: beam direction, degrees clockwise from screen-right


def _rgb(c):
    c = str(c).lstrip("#")
    return np.array([int(c[i:i + 2], 16) / 255 for i in (0, 2, 4)], np.float32)


def glow_video(seg, W, H, fps, n, media_dir):
    """Laser-eyes light layer for one shot: n RGBA frames at quarter res -> media/_eyes_<hash>.mov (PNG, lossless, cached).
    Composited like CapCut's mix.frag: out = shot * (1 - a) + light (premultiplied over), so it reads on bright backgrounds too.
    The renderer scales it to WxH and overlays it; the editor draws the same look live on its canvas."""
    import hashlib, math
    o = {**EYE_FX, **(seg.get("eye_fx") or {})}
    key = hashlib.md5(json.dumps([seg["eyes"], o, W, H, fps, n, 3], sort_keys=True).encode()).hexdigest()[:12]
    out = Path(media_dir) / f"_eyes_{key}.mov"
    if out.exists():
        return out
    q = 4
    w, h = max(2, W // q // 2 * 2), max(2, H // q // 2 * 2)
    E = np.array(seg["eyes"], np.float32)
    tint, k0 = _rgb(o["color"]), float(o.get("strength") or 1)
    ang = math.radians(float(o.get("angle", 72)))
    ca, sa = math.cos(ang), math.sin(ang)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    import clipcut
    ff = subprocess.Popen([clipcut.FF, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
                           "-c:v", "png", "-pix_fmt", "rgba", str(out)], stdin=subprocess.PIPE, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    for f in range(n):
        t = f / fps
        p = [float(np.interp(t, E[:, 0], E[:, j])) for j in range(1, 5)]
        eyes = [(p[0] * w, p[1] * h), (p[2] * w, p[3] * h)]
        d = max(3.0, math.hypot(eyes[1][0] - eyes[0][0], eyes[1][1] - eyes[0][1]))
        d = min(d, 0.18 * w)  # close-ups: the light stays eye-sized instead of flooding the frame
        core, glow = np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)
        for cx, cy in eyes:
            dx, dy = xx - cx, yy - cy
            r2 = dx * dx + dy * dy
            core += np.exp(-r2 / (2 * (0.2 * d) ** 2))
            glow += np.exp(-r2 / (2 * (0.45 * d) ** 2))
            glow += 0.7 * np.exp(-(dx / (1.3 * d)) ** 2 - (dy / (0.1 * d)) ** 2)  # lens-flare streak across the eye
            if o.get("beam", True):
                al, pe = dx * ca + dy * sa, np.abs(-dx * sa + dy * ca)
                wd = 0.28 * d * (1 + al / (5 * d)) * (1 + 0.1 * np.sin(t * 41 + al / (0.6 * d)))  # widens with distance, edges wobble like CapCut's sequence
                glow += 1.3 * (al > 0) * np.exp(-(pe / np.maximum(wd, 0.5)) ** 2) * np.exp(-al / (6 * d))
        light = core + glow
        bloom = sum(wgt * cv2.GaussianBlur(light, (0, 0), sig * d) for wgt, sig in ((0.45, 0.5), (0.25, 1.1), (0.12, 2.2)))  # DeepGlow-style multi-pass
        v = core[..., None] + tint * (glow + 0.8 * bloom)[..., None]
        pulse = k0 * (1 + 0.12 * math.sin(2 * math.pi * 2.2 * t))
        L = 1 - np.exp(-2.5 * pulse * v)  # CapCut tone map, premultiplied light 0..1
        L[..., 0] = np.roll(L[..., 0], 1, axis=1); L[..., 2] = np.roll(L[..., 2], -1, axis=1)  # chromatic aberration (ca)
        A = np.clip(L.max(axis=2), 0, 1)
        C = np.clip(L / np.maximum(A, 1e-3)[..., None], 0, 1)  # unpremultiplied colour; overlay does shot*(1-A) + C*A = shot*(1-A) + L
        rgba = np.dstack([C, A])
        ff.stdin.write((rgba * 255).round().astype(np.uint8).tobytes())
    ff.stdin.close(); ff.wait()
    if ff.returncode:
        raise RuntimeError("ffmpeg failed writing the eye glow")
    return out


def _canvas(P):
    CW, CH = P.get("canvas") or [1080, 1920]
    return CW, CH


def _frame(cap, t):
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, img = cap.read()
    return img if ok else None


def eyes(seg, P, media_dir, root=ROOT):
    import clipcut, faces, review
    ref = clipcut.face_ref(root)
    CW, CH = _canvas(P)
    W, H = round(review.DET_H * CW / CH) // 2 * 2, review.DET_H
    cap = cv2.VideoCapture(str(Path(media_dir) / seg["media"]))
    L, sp, out = clipcut.seg_len(seg), seg.get("speed", 1), []
    t = 0.0
    while t <= L + 1e-6:
        img = _frame(cap, seg["in"] + t * sp)
        if img is not None:
            img = review.framed(img, W, H, seg, P.get("fill", "contain"), seg.get("zoom") or 1)
            det, rec = faces._nets(W, H)
            _, found = det.detect(img)
            best = None
            for row in (found if found is not None else []):
                if ref is None or faces._cos(ref, rec.feature(rec.alignCrop(img, row)).flatten()) > faces.THR:
                    if best is None or row[2] * row[3] > best[2] * best[3]:
                        best = row
            if best is not None:  # YuNet landmarks: right eye, left eye, nose, mouth corners
                out.append([round(t, 3), *(round(float(v) / d, 4) for v, d in zip(best[4:8], (W, H, W, H)))])
        t += STEP
    cap.release()
    return out


def sun_disc(media_dir):
    """media/Sonne disc.webm from media/Sonne.webm: square crop centred on the disc (made once, cached)."""
    import clipcut
    src, out = Path(media_dir) / "Sonne.webm", Path(media_dir) / SUN
    if out.exists() or not src.exists():
        return out if out.exists() else None
    v = clipcut.probe(str(src))["video"]
    w, h = v["width"], v["height"]
    d = round(0.964 * w) // 2 * 2
    x, y = (w - d) // 2, max(0, round(0.474 * h - d / 2))
    clipcut.run([clipcut.FF, "-y", "-c:v", "libvpx-vp9", "-i", str(src), "-vf", f"crop={d}:{d}:{x}:{y},format=yuva420p", "-c:v", "libvpx-vp9",
                 "-pix_fmt", "yuva420p", "-b:v", "0", "-crf", "30", "-deadline", "realtime", "-cpu-used", "8", "-auto-alt-ref", "0", "-an", str(out)])
    return out


def cutout(src, a, b, fps, out, kind="person"):
    """RVM matting of src seconds a..b -> transparent webm `out` (skipped when it exists).
    kind="building": everything except the sky (sky segmentation) instead of the people."""
    import onnxruntime as ort
    out = Path(out)
    if out.exists():
        return out
    if kind == "building":
        return _sky_cutout(src, a, b, fps, out)
    sess = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    cap = cv2.VideoCapture(str(src))
    rec = [np.zeros((1, 1, 1, 1), np.float32)] * 4
    n = max(1, round((b - a) * fps))
    with tempfile.TemporaryDirectory() as tmp:
        for k in range(n):
            img = _frame(cap, a + k / fps)
            if img is None:
                break
            h0, w0 = img.shape[:2]
            s = min(1, 1280 / max(h0, w0))
            dr = 0.5 if max(h0, w0) * s > 640 else 1.0
            m = int(8 / dr)  # the model's pools need dims*dr divisible by 8
            w, h = max(m, round(w0 * s / m) * m), max(m, round(h0 * s / m) * m)
            x = cv2.cvtColor(cv2.resize(img, (w, h)), cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1)[None] / 255
            fgr, pha, *rec = sess.run(None, {"src": x, "r1i": rec[0], "r2i": rec[1], "r3i": rec[2], "r4i": rec[3],
                                             "downsample_ratio": np.array([dr], np.float32)})
            rgba = np.concatenate([fgr[0][::-1].transpose(1, 2, 0), pha[0].transpose(1, 2, 0)], 2)  # BGR + alpha
            cv2.imwrite(f"{tmp}/{k:06d}.png", (cv2.resize(rgba, (w0, h0)) * 255).clip(0, 255).astype(np.uint8))
        cap.release()
        import clipcut
        tmp_out = out.with_suffix(".tmp.webm")
        clipcut.run([clipcut.FF, "-y", "-framerate", str(fps), "-i", f"{tmp}/%06d.png", "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p",
                     "-b:v", "0", "-crf", "28", "-row-mt", "1", "-deadline", "realtime", "-cpu-used", "8", "-an", str(tmp_out)])
        shutil.move(tmp_out, out)
    return out


def _sky_cutout(src, a, b, fps, out):
    """Building cut-out (everything in front of the background): sky segmentation when the shot has sky (>= 4 % of the first frame), else the depth split."""
    import clipcut
    sky, prev, T = _sky_session(), None, None
    cap = cv2.VideoCapture(str(src))
    first = _frame(cap, a)
    use_depth = first is not None and float((sky_alpha(sky, first) < 0.5).mean()) < 0.04
    sess = _session(DEPTH_MODEL, DEPTH_URL, "depth (99 MB)") if use_depth else sky
    print(f"building cut-out via {'depth' if use_depth else 'sky'}", file=sys.stderr)
    with tempfile.TemporaryDirectory() as tmp:
        for k in range(max(1, round((b - a) * fps))):
            img = _frame(cap, a + k / fps)
            if img is None:
                break
            if use_depth:
                if k % 3 == 0:  # ponytail: depth every 3rd frame - a drone/handheld edge moves < 1 px per frame; per frame if edges lag
                    d = depth_map(sess, img)
                t, _ = cv2.threshold((d * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                T = t / 255 if T is None else 0.8 * T + 0.2 * t / 255  # smooth the split point over time (no pumping edge)
                al = np.clip((d - T) / 0.04 + 0.5, 0, 1)
            else:
                al = sky_alpha(sess, img)
            prev = al if prev is None or prev.shape != al.shape else 0.6 * al + 0.4 * prev  # ponytail: light temporal smoothing against edge flicker
            cv2.imwrite(f"{tmp}/{k:06d}.png", np.dstack([img, (prev * 255).astype(np.uint8)]))
        cap.release()
        tmp_out = out.with_suffix(".tmp.webm")
        clipcut.run([clipcut.FF, "-y", "-framerate", str(fps), "-i", f"{tmp}/%06d.png", "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p",
                     "-b:v", "0", "-crf", "24", "-row-mt", "1", "-deadline", "realtime", "-cpu-used", "8", "-an", str(tmp_out)])
        shutil.move(tmp_out, out)
    return out


SUN_DISC = 1.0, 0.5       # the disc fills the square asset, centred (the raw 4:5 PNG had it at 0.964 wide / 0.474 down -> see sun_disc())
SUN_SIZE = 1.7            # disc diameter in face heights (a halo round the head, not a billboard)
SUN_TURNS = 0.75          # turns per second


def track(seg, P, media_dir, ref):
    """[(t, fx, fy, fh)]: the star's face centre + height (canvas fractions) every STEP s of the shot, lightly smoothed."""
    import clipcut, review
    CW, CH = _canvas(P)
    W, H = round(review.DET_H * CW / CH) // 2 * 2, review.DET_H
    cap = cv2.VideoCapture(str(Path(media_dir) / seg["media"]))
    L, sp, raw, t = clipcut.seg_len(seg), seg.get("speed", 1), [], 0.0
    while t <= L + 1e-6:
        img = _frame(cap, seg["in"] + t * sp)
        if img is not None:
            box, _ = review.star_in(review.framed(img, W, H, seg, P.get("fill", "contain")), ref)
            if box is not None:
                x, y, w, h = box
                raw.append((round(t, 3), (x + w / 2) / W, (y + h / 2) / H, h / H))
        t += STEP
    cap.release()
    out = []  # ponytail: 3-sample moving average kills YuNet jitter; a Kalman filter if it still wobbles
    for k, r in enumerate(raw):
        win = raw[max(0, k - 1):k + 2]
        out.append((r[0], *(round(sum(v[j] for v in win) / len(win), 4) for j in (1, 2, 3))))
    return out


def sun(P, i, media_dir, root=ROOT):
    """Overlays for shot i: the sun spinning behind his head and following it (keyframes from the face track), the cut-out of the
    slice on top. False when the star is not tracked in at least 2 samples or other people are in the frame."""
    import clipcut, faces, review
    seg, media_dir = P["video"][i], Path(media_dir)
    sun_disc(media_dir)
    if not (media_dir / SUN).exists():
        raise FileNotFoundError(f"no {SUN} in media/ (drop a PNG named Sonne.png into the editor, it becomes a transparent webm)")
    CW, CH = _canvas(P)
    cap = cv2.VideoCapture(str(media_dir / seg["media"]))
    img = _frame(cap, (seg["in"] + seg["out"]) / 2)
    cap.release()
    if img is None:
        return False
    iw, ih = img.shape[1], img.shape[0]
    W, H = round(review.DET_H * CW / CH) // 2 * 2, review.DET_H
    _, found = faces._nets(W, H)[0].detect(review.framed(img, W, H, seg, P.get("fill", "contain")))
    if found is not None and len(found) > 1:  # RVM cuts out everybody, the sun would hide behind the others: no sun in group shots
        return False
    seg.pop("zoom", None); seg.pop("zoomTo", None); seg.pop("shake", None)  # the cut-out must sit exactly on the clip
    tr = track(seg, P, media_dir, clipcut.face_ref(root))
    if len(tr) < 2:
        return False
    L, st = clipcut.seg_len(seg), clipcut.tl_layout(P)[i]
    fps = P.get("fps", 30)
    # user 2026-10-20: for a frame or two the sun sat ON him -> those were the transition frames: the shot's first/last frames are
    # replaced by the zoom/flash window, the cut-out no longer matches them. Overlays start after the incoming window and end before the outgoing one.
    def _win(s_prev, s_next):
        tr = (s_prev.get("transition") or {})
        n = round(min(tr.get("dur", 0), clipcut.seg_len(s_prev) / 2, clipcut.seg_len(s_next) / 2) * fps) if tr.get("type", "none") != "none" else 0
        return n / fps if n >= 2 else 0.0
    d_in = _win(P["video"][i - 1], seg) if i else 0.0
    d_out = _win(seg, P["video"][i + 1]) if i + 1 < len(P["video"]) else 0.0
    name = f"{Path(seg['media']).stem}_{seg['in']:.2f}-{seg['out']:.2f}_cutout.webm"
    cutout(media_dir / seg["media"], seg["in"], seg["out"], fps, media_dir / name)
    # where the source frame sits on the canvas under the cover crop (clipcut.fit_chain): scale k, crop offset x0/y0
    k = max(CW / iw, CH / ih); sw, sh = iw * k, ih * k
    cx, cy = (min(1, max(0, float(seg.get(c) or 0.5))) for c in ("cx", "cy"))
    x0, y0 = min(sw - CW, max(0, sw * cx - CW / 2)), min(sh - CH, max(0, sh * cy - CH / 2))
    t0, fx0, fy0, fh0 = tr[0]
    dfrac, cfrac = SUN_DISC
    scale = SUN_SIZE * fh0 * CH / dfrac / CW                    # overlay width as a canvas fraction
    head = lambda fy, fh: fy - 0.15 * fh                        # head centre sits a bit above the face box centre
    ydisc = (0.5 - cfrac) * scale * CW * 1.25 / CH              # frame centre is this much below the disc centre
    keys = [{"t": round(t - d_in, 3), "x": round(fx - fx0, 4), "y": round(head(fy, fh) - head(fy0, fh0), 4), "scale": round(fh / fh0, 4),
             "rotation": round(360 * SUN_TURNS * t, 1)} for t, fx, fy, fh in tr if t >= d_in - 1e-6]
    if not keys:
        return False
    Lv = L - d_in - d_out  # visible (plain) part of the shot
    if keys[-1]["t"] < Lv - 1e-3:  # keep spinning to the end of the shot, hold the last position
        keys.append({**keys[-1], "t": round(Lv, 3), "rotation": round(360 * SUN_TURNS * (Lv + d_in), 1)})
    sp = seg.get("speed", 1)
    P["overlay"] += [
        {"id": clipcut.uid(), "media": SUN, "start": round(st + d_in, 3), "in": 0, "out": round(Lv, 3), "speed": 1, "x": round(fx0, 4),
         "y": round(head(fy0, fh0) + ydisc, 4), "scale": round(scale, 4), "rotation": 0, "flip": False, "volume": 0, "opacity": 1,
         "_sun": seg["id"], "keyframes": keys},
        {"id": clipcut.uid(), "media": name, "start": round(st + d_in, 3), "in": round(d_in * sp, 3), "out": round(seg["out"] - seg["in"] - d_out * sp, 3), "speed": sp,
         "x": round((sw / 2 - x0) / CW, 4), "y": round((sh / 2 - y0) / CH, 4), "scale": round(sw / CW, 4), "rotation": 0, "flip": False,
         "volume": 0, "opacity": 1, "_sun": seg["id"]},
    ]
    return True


SLIDE = 0.4               # s: the cut-out flies in over the previous shot, lands exactly on the cut, then the background is there
SLIDE_FROM = {"bottom": (0, 1.1), "left": (-1.1, 0), "right": (1.1, 0), "top": (0, -1.1)}


def slide(P, i, media_dir, root=ROOT):
    """CapCut-style slide-in (user 2026-10-20): the RVM cut-out of the SLIDE s before shot i's in-point slides in from `seg['slide']`
    (bottom/left/right/top) over the previous shot and lands where he stands in shot i, which then starts plain on the cut.
    The cut into shot i must be a hard cut (the planner keeps it 'none')."""
    import clipcut
    seg, media_dir = P["video"][i], Path(media_dir)
    if not i or seg.get("freeze"):
        return False
    CW, CH = _canvas(P)
    sp, fps = seg.get("speed", 1), P.get("fps", 30)
    a = round(max(0.0, seg["in"] - SLIDE * sp), 3)
    if seg["in"] - a < 0.1:
        return False
    cap = cv2.VideoCapture(str(media_dir / seg["media"]))
    img = _frame(cap, a)
    cap.release()
    if img is None:
        return False
    iw, ih = img.shape[1], img.shape[0]
    seg.pop("zoom", None); seg.pop("zoomTo", None); seg.pop("shake", None)  # the cut-out must land exactly on the plain shot
    b = round(seg["in"] + 2 * sp / fps, 3)  # the cut-out runs 2 frames INTO the plain shot: it sits on its own pixels there, so a
    name = f"{Path(seg['media']).stem}_{a:.2f}-{b:.2f}_cutout.webm"  # 1-frame rounding of the real cut never shows the old shot bare
    cutout(media_dir / seg["media"], a, b, fps, media_dir / name)
    k = max(CW / iw, CH / ih); sw, sh = iw * k, ih * k
    cx, cy = (min(1, max(0, float(seg.get(c) or 0.5))) for c in ("cx", "cy"))
    x0, y0 = min(sw - CW, max(0, sw * cx - CW / 2)), min(sh - CH, max(0, sh * cy - CH / 2))
    dx, dy = SLIDE_FROM.get(seg.get("slide") or "bottom", SLIDE_FROM["bottom"])
    st = clipcut.tl_layout(P)[i]
    d = (seg["in"] - a) / sp
    P["overlay"].append({"id": clipcut.uid(), "media": name, "start": round(st - d, 3), "in": 0, "out": round(b - a, 3), "speed": sp,
                         "x": round((sw / 2 - x0) / CW, 4), "y": round((sh / 2 - y0) / CH, 4), "scale": round(sw / CW, 4), "rotation": 0,
                         "flip": False, "volume": 0, "opacity": 1, "_slide": seg["id"],
                         # linear keys: fly in, 3 % overshoot, settled at 80 % and HELD STILL until past the cut (user 2026-10-20:
                         # "something moving when the background comes" = keys that reached 0 on the cut frame itself)
                         "keyframes": [{"t": 0, "x": dx, "y": dy}, {"t": round(d * 0.6, 3), "x": round(-dx * 0.03, 4), "y": round(-dy * 0.03, 4)},
                                       {"t": round(d * 0.8, 3), "x": 0, "y": 0}, {"t": round((b - a) / sp, 3), "x": 0, "y": 0}]})
    return True


def behind(P, t, media_dir):
    """Text t behind the person (CapCut "text behind subject"; t.behind == "building": behind everything that is not sky): for every main-track shot under the text, the RVM cut-out of exactly
    that overlap goes on top as an overlay tagged `_behind` = t's id; render + preview draw `behind` texts before the overlays.
    Returns the number of cut-outs. ponytail: plain cover/contain shots only - zoom/shake/keyframes on those shots are removed,
    because the cut-out could not follow them."""
    import clipcut
    media_dir = Path(media_dir)
    P["overlay"] = [o for o in P.get("overlay", []) if o.get("_behind") != t["id"]]
    CW, CH = _canvas(P)
    fill, fps, n = P.get("fill", "contain"), P.get("fps", 30), 0
    a0, b0 = t["start"], t["start"] + t["dur"]
    for i, (seg, st) in enumerate(zip(P["video"], clipcut.tl_layout(P))):
        L = clipcut.seg_len(seg)
        a, b = max(a0, st), min(b0, st + L)
        if b - a < 0.05 or seg.get("freeze"):
            continue
        sp = seg.get("speed", 1)
        sa, sb = round(seg["in"] + (a - st) * sp, 3), round(seg["in"] + (b - st) * sp, 3)
        cap = cv2.VideoCapture(str(media_dir / seg["media"]))
        img = _frame(cap, sa)
        cap.release()
        if img is None:
            continue
        for k in ("zoom", "zoomTo", "shake", "keyframes"):
            seg.pop(k, None)
        iw, ih = img.shape[1], img.shape[0]
        kind = "building" if t.get("behind") == "building" else "person"
        name = f"{Path(seg['media']).stem}_{sa:.2f}-{sb:.2f}_{'skycut' if kind == 'building' else 'cutout'}.webm"
        cutout(media_dir / seg["media"], sa, sb, fps, media_dir / name, kind)
        if fill == "cover":  # same geometry as clipcut.fit_chain
            k = max(CW / iw, CH / ih); sw, sh = iw * k, ih * k
            cx, cy = (min(1, max(0, float(seg.get(c) or 0.5))) for c in ("cx", "cy"))
            x0, y0 = min(sw - CW, max(0, sw * cx - CW / 2)), min(sh - CH, max(0, sh * cy - CH / 2))
            x, y, scale = (sw / 2 - x0) / CW, (sh / 2 - y0) / CH, sw / CW
        else:  # contain / blur: the frame sits centred, fitted inside the canvas
            x, y, scale = 0.5, 0.5, min(CW / iw, CH / ih) * iw / CW
        P["overlay"].append({"id": clipcut.uid(), "media": name, "start": round(a, 3), "in": 0, "out": round(sb - sa, 3), "speed": sp,
                             "x": round(x, 4), "y": round(y, 4), "scale": round(scale, 4), "rotation": 0, "flip": False,
                             "volume": 0, "opacity": 1, "_behind": t["id"]})
        n += 1
    return n


def bake(P, media_dir, root=ROOT):
    """Resolve planner flags: `eyes: true` -> eye track, `sun: true` -> overlays. Returns a step message ('' when nothing to do)."""
    ne = ns = nsl = 0
    for i, seg in enumerate(P["video"]):
        if isinstance(seg.get("slide"), str):
            P["overlay"] = [o for o in P.get("overlay", []) if o.get("_slide") != seg["id"]]
            try:
                ok = slide(P, i, media_dir, root)
            except Exception as e:
                print(f"slide: {e}", file=sys.stderr); ok = False
            if ok:
                nsl += 1
            else:
                seg["slide"] = False
        if seg.get("eyes") is True:
            tr = eyes(seg, P, media_dir, root)
            if tr:
                seg["eyes"], ne = tr, ne + 1
            else:
                seg.pop("eyes")
        if seg.get("sun") is True:
            P["overlay"] = [o for o in P.get("overlay", []) if o.get("_sun") != seg["id"]]
            try:
                ok = sun(P, i, media_dir, root)
            except Exception as e:  # no sun asset / no onnxruntime: say so, keep the edit
                print(f"sun: {e}", file=sys.stderr); ok = False
            seg["sun"] = ok; ns += ok
    if False:  # ponytail: old rule 'every edit gets the sun at least once' switched off for school videos
        order = sorted((i for i, s in enumerate(P["video"]) if not s.get("freeze") and s.get("volume", 1) == 0 and s.get("sun") is not False),
                       key=lambda i: (P["video"][i].get("zoom", 1) < 1.3, i))
        for i in order:
            try:
                ok = sun(P, i, media_dir, root)
            except Exception as e:
                print(f"sun: {e}", file=sys.stderr); ok = False
            P["video"][i]["sun"] = ok
            if ok:
                ns = 1; break
    parts = [f"image spinning behind the head on {ns} shot(s)" if ns else "", f"laser eyes on {ne} accent(s)" if ne else "",
             f"cut-out slides in over the old shot on {nsl} cut(s)" if nsl else ""]
    return ", ".join(p for p in parts if p)


if __name__ == "__main__":
    import clipcut
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted((ROOT / "media").glob("*.mp4"))[0]
    a = float(sys.argv[2]) if len(sys.argv) > 2 else 5.4
    b = float(sys.argv[3]) if len(sys.argv) > 3 else a + 0.3
    P = {"canvas": [1080, 1920], "fps": 30, "fill": "cover", "video": [{"id": "selftest", "media": src.name, "in": a, "out": b, "speed": 1, "cx": 0.5, "cy": 0.5, "sun": True, "eyes": True}],
         "overlay": [], "audio": [], "text": []}
    out = Path(tempfile.gettempdir()) / "fxface_selftest_cutout.webm"
    out.unlink(missing_ok=True)
    print(cutout(src, a, b, 30, out), clipcut.probe(str(out))["video"]["codec_name"])
    msg = bake(P, ROOT / "media", ROOT)
    assert P["video"][0].get("eyes") and len(P["video"][0]["eyes"]) >= 2 and all(0 < v < 1 for e in P["video"][0]["eyes"] for v in e[1:]), P["video"][0].get("eyes")
    assert P["video"][0]["sun"] is True and [o["media"] for o in P["overlay"]][0] == SUN and P["overlay"][1]["scale"] >= 1 - 1e-6, P["overlay"]  # cover: the cut-out is never smaller than the canvas
    ks = P["overlay"][0]["keyframes"]
    assert len(ks) >= 2 and ks[-1]["rotation"] > 0 and all({"x", "y", "scale"} <= k.keys() for k in ks) and abs(ks[-1]["t"] - 0.3) < 1e-3, ks
    print(msg, json.dumps(P["overlay"], ensure_ascii=False)[:400])
    print("fxface selftest ok")
