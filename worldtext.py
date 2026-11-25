"""worldtext.py - big 3D "frosted glass" text that STAYS IN PLACE in the scene (After Effects 3D camera tracker look), 2026-11-25.

The text stands behind a building like a real, huge sign: it is pinned to the background around it (Lucas-Kanade points on
far pixels only, similarity transform per frame relative to the first frame, then the whole camera path is smoothed), so when
the camera/drone moves, the text moves, turns and grows exactly like that part of the scene. Everything nearer than the
background - the building, people on the roof, trees - covers it (depth matte: fxface.depth_map, Otsu near/far split).

Style "frost" (default): milky, slightly transparent glass letters - the scene behind each letter is blurred and lightened,
a 3D extrusion in milky grey, a thin light rim on the top edges and a soft drop shadow. Style "solid": white letters + grey
extrusion. Result = a new clip `media/<stem>_<a>-<b>_world<tag>.mp4` (source audio kept) that replaces the shot (`op worldtext`).

python worldtext.py SRC A B "TEXT" [--x 0.5] [--y 0.37] [--w 0.95] [--opacity 0.8] [--font montserrat] [--depth 0.22] [--style frost]
python worldtext.py selftest
  x, y  = bottom centre of the text in the FIRST frame (fractions of the source frame) - put it just behind the building edge
  w     = text width as a fraction of the frame width in the first frame
  depth = extrusion as a fraction of the letter height
"""
import argparse, json, sys, tempfile, zlib
from pathlib import Path

import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent
VERSION = "frost-6"  # bump when the look changes, so cached bakes are redone
SIDE = 0.45  # extrusion brightness (frost): lower = more 3D contrast against the milky face
BUNDLED = {"montserrat": "Montserrat-SemiBold.ttf", "playfair": "PlayfairDisplay-Bold.ttf", "impact": "Anton-Regular.ttf"}


def _font(font, px):
    for cand in (BUNDLED.get(font), BUNDLED["montserrat"]):
        if cand and (ROOT / "fonts" / cand).exists():
            return ImageFont.truetype(str(ROOT / "fonts" / cand), px)
    return ImageFont.load_default(px)


def layers(text, font="montserrat", depth=0.22, px=360):
    """Float32 masks in sprite space: face, side (extrusion), shade (side darkness 0..1), rim (light top edges), shadow.
    A mild perspective is baked in (bottom 8 % narrower: seen from above, the top of a standing sign is nearer)."""
    f = _font(font, px)
    l, t, r, b = f.getbbox(text)
    ext = max(2, round(px * depth))
    pad = ext + px // 5
    W, H = r - l + 2 * pad, b - t + 2 * pad

    def mask(dx=0.0, dy=0.0):
        m = Image.new("L", (W, H), 0)
        ImageDraw.Draw(m).text((pad - l + dx, pad - t + dy), text, font=f, fill=255)
        return np.asarray(m, np.float32) / 255

    face = mask()
    side, shade = np.zeros_like(face), np.zeros_like(face)
    for k in range(ext, 0, -1):  # extrusion UP and a bit right: from a drone/eye level above you see the letters' top faces
        m = mask(k * 0.3, -k)
        side = np.maximum(side, m)
        shade = np.where(m > 0.5, 1 - k / ext, shade)
    side = np.clip(side - face, 0, 1)
    up = np.roll(face, max(1, px // 45), axis=0)  # top edges: face where the face shifted down is empty
    rim = cv2.GaussianBlur(np.clip(face - up, 0, 1), (0, 0), px / 160)
    er = cv2.erode(face, np.ones((max(3, px // 60),) * 2, np.uint8))
    rim = np.maximum(rim, 0.45 * cv2.GaussianBlur(np.clip(face - er, 0, 1), (0, 0), px / 240))  # thin glass edge all round
    shadow = cv2.GaussianBlur(mask(ext * 0.2, px * 0.06), (0, 0), px / 10)
    stack = [face, side, shade, rim, shadow]
    k = 0.04 * W
    P = cv2.getPerspectiveTransform(np.float32([[0, 0], [W, 0], [W, H], [0, H]]), np.float32([[0, 0], [W, 0], [W - k, H], [k, H]]))
    return np.dstack([cv2.warpPerspective(c, P, (W, H), flags=cv2.INTER_LINEAR) for c in stack]), (W, H)


def compose(img, L, fg, opacity=0.8, style="frost"):
    """img BGR uint8, L = warped layers (H, W, 5) float, fg = 1 where the scene is in front of the text."""
    vis = (1 - fg)[..., None] * opacity
    face, side, shade, rim, shadow = (L[..., i:i + 1] for i in range(5))
    x = img.astype(np.float32) / 255
    x = x * (1 - 0.2 * shadow * vis)  # soft contact shadow on the scene
    if style == "solid":
        side_col = 0.55 + 0.25 * shade
        face_col = 0.97
    else:  # frosted glass: the scene behind the letter, blurred and lifted towards milky white
        blur = cv2.GaussianBlur(x, (0, 0), max(2.0, img.shape[0] / 90))
        side_col = blur * 0.35 + (SIDE + 0.3 * shade) * 0.65  # top faces catch the light, deeper parts darker
        face_col = blur * 0.3 + 0.7
    x = x * (1 - side * vis) + side_col * side * vis
    x = x * (1 - face * vis) + face_col * face * vis
    x = x + rim * vis * 0.55  # light catches the top edges
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def track(frames_iter, n, box0, FW, FH, log=print):
    """Pass 1: [(tx, ty, log scale, rotation)] per frame (frame 0 -> frame k), smoothed. frames_iter yields (k, gray, fg)."""
    lp = dict(winSize=(21, 21), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    prev_g = p_cur = p0 = None
    M = np.eye(3, dtype=np.float32)
    raw = []
    for k, g, fg in frames_iter:
        if k:
            p_new, st, _ = cv2.calcOpticalFlowPyrLK(prev_g, g, p_cur, None, **lp)
            p_back, st2, _ = cv2.calcOpticalFlowPyrLK(g, prev_g, p_new, None, **lp)  # forward-backward check
            fb = np.linalg.norm((p_back - p_cur).reshape(-1, 2), axis=1)
            q = np.clip(p_new.reshape(-1, 2).round().astype(int), 0, [FW - 1, FH - 1])
            # drop points the building now covers: at an occluding edge they slide along it and drag the text with the roof
            ok = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 0.5) & (fg[q[:, 1], q[:, 0]] < 0.2)
            p0, p_cur = p0[ok], p_new[ok]
            if len(p0) >= 8:
                A, inl = cv2.estimateAffinePartial2D(p0, p_cur, method=cv2.RANSAC, ransacReprojThreshold=2.0)
                if A is not None:
                    M = np.vstack([A, [0, 0, 1]]).astype(np.float32)
                    p0, p_cur = p0[inl.ravel() == 1], p_cur[inl.ravel() == 1]
        if p0 is None or len(p0) < 150:  # top up: background points around where the text stands now, far pixels only
            # anchor = what lies right behind the letters (the eye judges the text against exactly that; with parallax a
            # farther or nearer plane would make it visibly slide over the scene). Base-only anchoring was tried: the foot
            # of a text "behind" a building stands on ground the building hides, so it has nothing visible to hold on to.
            box = cv2.perspectiveTransform(box0, M).reshape(-1, 2)
            x0, y0 = box.min(0); x1, y1 = box.max(0)
            mx, my = (x1 - x0) * 0.15, (y1 - y0) * 0.3
            roi = np.zeros(g.shape, np.uint8)
            roi[int(np.clip(y0 - my, 0, FH)):int(np.clip(y1 + my, 0, FH)), int(np.clip(x0 - mx, 0, FW)):int(np.clip(x1 + mx, 0, FW))] = 1
            roi &= (fg < 0.3).astype(np.uint8)  # the building moves differently (parallax): never track it
            pts = cv2.goodFeaturesToTrack(g, 400, 0.01, 8, mask=roi * 255)
            if pts is None or len(pts) < 40:  # plain sky / wall behind the letters: follow the whole background instead
                far = (fg < 0.3).astype(np.uint8) * 255
                pts = cv2.goodFeaturesToTrack(g, 400, 0.01, 8, mask=far)
            if pts is not None:
                pts = pts.astype(np.float32)
                new0 = cv2.perspectiveTransform(pts, np.linalg.inv(M))
                p0 = new0 if p0 is None else np.concatenate([p0, new0])
                p_cur = pts if p_cur is None else np.concatenate([p_cur, pts])
        prev_g = g
        raw.append((M[0, 2], M[1, 2], np.log(np.sqrt(abs(np.linalg.det(M[:2, :2])))), np.arctan2(M[1, 0], M[0, 0])))
        if k % 30 == 0:
            log(f"world text: tracking {k}/{n}, {0 if p0 is None else len(p0)} points")
    # like a tracker's smooth pass: median 7 against jumps, then Gaussian sigma 3 frames
    R = np.array(raw, np.float64)
    if len(R) > 7:
        pad = np.pad(R, ((3, 3), (0, 0)), mode="reflect", reflect_type="odd")  # odd reflection keeps the motion going at the ends
        R = np.median(np.stack([pad[i:i + len(R)] for i in range(7)]), 0)
        ker = np.exp(-0.5 * (np.arange(-9, 10) / 3.0) ** 2); ker /= ker.sum()
        R = np.stack([np.convolve(np.pad(R[:, j], 9, mode="reflect", reflect_type="odd"), ker, "valid") for j in range(4)], 1)
    return R


def _M(tx, ty, ls, rot):
    s = np.exp(ls)
    return np.float32([[s * np.cos(rot), -s * np.sin(rot), tx], [s * np.sin(rot), s * np.cos(rot), ty], [0, 0, 1]])


def bake(src, a, b, text, out, x=0.5, y=0.37, w=0.95, opacity=0.8, font="montserrat", depth=0.22, style="frost", log=print):
    import clipcut, fxface
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n = max(1, round((b - a) * fps))
    dsess = fxface._session(fxface.DEPTH_MODEL, fxface.DEPTH_URL, "depth (99 MB)")
    L0, (SW, SH) = layers(text, font, depth)
    first = fxface._frame(cap, a)
    if first is None:
        raise ValueError(f"cannot read {src} at {a}s")
    FH, FW = first.shape[:2]
    sw = w * FW / SW
    place = np.float32([[sw, 0, x * FW - sw * SW / 2], [0, sw, y * FH - sw * SH], [0, 0, 1]])  # bottom centre at (x, y)
    box0 = cv2.perspectiveTransform(np.float32([[[0, 0]], [[SW, 0]], [[SW, SH]], [[0, SH]]]), place)
    fgs = []

    def frames():
        T = d = None
        for k in range(n):
            img = first if k == 0 else fxface._frame(cap, a + k / fps)
            if img is None:
                return
            if k % 3 == 0:  # depth every 3rd frame (edges move < 1 px per frame)
                d = fxface.depth_map(dsess, img)
                t, _ = cv2.threshold((d * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                T = t / 255 if T is None else 0.8 * T + 0.2 * t / 255
            fg = np.clip((d - T) / 0.04 + 0.5, 0, 1)
            fgs.append((fg * 255).astype(np.uint8))
            yield k, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), fg

    R = track(frames(), n, box0, FW, FH, log)
    bake.path, bake.place, bake.box0 = R, place, box0
    with tempfile.TemporaryDirectory() as tmp:
        for k, p in enumerate(R):
            img = first if k == 0 else fxface._frame(cap, a + k / fps)
            if img is None:
                break
            H = _M(*p) @ place
            L = np.dstack([cv2.warpPerspective(L0[..., i], H, (FW, FH), flags=cv2.INTER_LINEAR) for i in range(5)])
            cv2.imwrite(f"{tmp}/{k:06d}.png", compose(img, L, fgs[k].astype(np.float32) / 255, opacity, style))
            if k % 60 == 0:
                log(f"world text: rendering {k}/{len(R)}")
        cap.release()
        clipcut.run([clipcut.FF, "-y", "-framerate", str(fps), "-i", f"{tmp}/%06d.png", "-ss", str(a), "-t", str(b - a), "-i", str(src),
                     "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p",
                     "-c:a", "aac", "-b:a", "192k", "-shortest", str(out)])
    return out


KEYS = ("x", "y", "w", "opacity", "font", "depth", "style")


def apply(seg, media_dir, spec, log=print):
    """`op worldtext`: bake the world text into main-track shot `seg`. The untouched slice is kept in seg['world']['src'], so a
    second call (new text / position) re-bakes from the original footage, not from the already baked clip."""
    media_dir = Path(media_dir)
    w = seg.get("world") or {}
    src = w.get("src") or {"media": seg["media"], "in": seg["in"], "out": seg["out"]}
    spec = {**{k: v for k, v in w.items() if k != "src"}, **spec}
    if not spec.get("text"):
        raise ValueError("worldtext needs a text")
    a, b = float(src["in"]), float(src["out"])
    tag = zlib.crc32((VERSION + json.dumps(spec, sort_keys=True, ensure_ascii=False)).encode()) % 10 ** 6  # new look -> new bake
    out = media_dir / f"{Path(src['media']).stem}_{a:.2f}-{b:.2f}_world{tag}.mp4"
    if not out.exists():
        bake(media_dir / src["media"], a, b, spec["text"], out, **{k: spec[k] for k in KEYS if k in spec}, log=log)
    seg.update({"media": out.name, "in": 0, "out": round(b - a, 3), "world": {**spec, "src": src}})
    for k in ("zoom", "zoomTo", "shake", "keyframes", "speedCurve", "reverse", "speed"):  # pinned to the footage: no extra camera moves
        seg.pop(k, None)
    seg["speed"] = 1
    return out


def remove(seg):
    """Undo `op worldtext`: back to the original slice."""
    src = (seg.pop("world", None) or {}).get("src")
    if src:
        seg.update(src)


def selftest():
    """Synthetic camera move: a textured background slides + zooms by a known amount; the tracker must recover it and the
    pinned text must follow (< 1.5 px error) while a near 'building' band moving much faster is ignored (parallax); the frost
    style must look milky (scene visible, not opaque white) and occluded letters must leave the image untouched."""
    rng = np.random.default_rng(1)
    tex = cv2.GaussianBlur((rng.random((900, 1600)) * 255).astype(np.uint8), (0, 0), 2)
    tex = cv2.normalize(tex, None, 0, 255, cv2.NORM_MINMAX)
    FW, FH, n = 640, 360, 40
    truth = []

    def frames():
        for k in range(n):
            s, tx, ty = 1 + 0.004 * k, -3.0 * k, 1.5 * k
            A = np.float32([[s, 0, tx], [0, s, ty]])
            g = cv2.warpAffine(tex, A, (FW, FH), flags=cv2.INTER_LINEAR)
            near = cv2.warpAffine(tex[::-1].copy(), np.float32([[1, 0, -9 * k], [0, 1, 0]]), (FW, FH))  # building: much faster
            fg = np.zeros((FH, FW), np.float32); fg[250:] = 1
            truth.append(A)
            yield k, np.where(fg > 0.5, near, g).astype(np.uint8), fg

    box0 = np.float32([[[150, 120]], [[450, 120]], [[450, 200]], [[150, 200]]])
    R = track(frames(), n, box0, FW, FH, log=lambda m: None)
    err = 0.0
    for k in (10, 25, n - 1):
        p_true = cv2.perspectiveTransform(box0, np.vstack([truth[k], [0, 0, 1]]).astype(np.float32))
        p_est = cv2.perspectiveTransform(box0, _M(*R[k]))
        err = max(err, float(np.abs(p_true - p_est).max()))
    assert err < 1.5, f"tracking error {err:.2f} px"
    L, (W, H) = layers("TEST", px=120)
    assert L.shape[2] == 5 and L[..., 0].max() > 0.9 and L[..., 1].max() > 0.9 and L[..., 3].max() > 0.05, "layers"
    img = np.full((H, W, 3), 60, np.uint8)
    fg = np.zeros((H, W), np.float32); fg[:, : W // 2] = 1  # left half of the text is behind the building
    o = compose(img, L, fg, 0.8, "frost")
    face = L[..., 0] > 0.9
    cols = np.arange(W)[None]
    left, right = face & (cols < W // 2), face & (cols >= W // 2)
    assert np.abs(o[left].astype(int) - 60).max() <= 1, "occluded letters must not show"
    v = float(o[right].mean())
    assert 140 < v < 245, f"frosted face should be milky, not opaque white or invisible: {v:.0f}"
    print(f"worldtext selftest ok (tracking error {err:.2f} px, frost face {v:.0f}/255)")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    if sys.argv[1:2] == ["selftest"]:
        selftest(); sys.exit()
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("a", type=float); ap.add_argument("b", type=float); ap.add_argument("text")
    ap.add_argument("--out"); ap.add_argument("--x", type=float, default=0.5); ap.add_argument("--y", type=float, default=0.37)
    ap.add_argument("--w", type=float, default=0.95); ap.add_argument("--opacity", type=float, default=0.8)
    ap.add_argument("--font", default="montserrat"); ap.add_argument("--depth", type=float, default=0.22)
    ap.add_argument("--style", default="frost", choices=["frost", "solid"])
    o = ap.parse_args()
    src = Path(o.src)
    out = o.out or str(src.with_name(f"{src.stem}_{o.a:.2f}-{o.b:.2f}_world.mp4"))
    print(bake(src, o.a, o.b, o.text, out, o.x, o.y, o.w, o.opacity, o.font, o.depth, o.style))
