"""worldtext.py - big 3D "frosted glass" text that STAYS IN PLACE in the scene (After Effects 3D camera tracker look), 2026-11-25.

The text stands behind a building like a real, huge sign: a 3D camera tracker (Lucas-Kanade tracks -> essential matrix ->
incremental PnP + bundle adjustment with focal length and lens distortion, scipy) rebuilds the drone/camera path and the scene
as ~3000 3D points; the sign is then an upright plane standing at the depth of the scene behind the letters (ground plane =
"up"), projected into every frame. So it stays exactly where it stands while the camera flies in, with real perspective.
A camera that only turns / stands still has no depth: then one homography per frame (exact for pure rotation). Everything
nearer than the background - the building, people on the roof, trees - covers it (depth matte: fxface.depth_map, Otsu split).

Style "frost" (default): milky, slightly transparent glass letters - the scene behind each letter is blurred and lightened,
a 3D extrusion in milky grey, a thin light rim on the top edges and a soft drop shadow. Style "solid": white letters + grey
extrusion. Result = a new clip `media/<stem>_<a>-<b>_world<tag>.mp4` (source audio kept) that replaces the shot (`op worldtext`).

python worldtext.py SRC A B "TEXT" [--x 0.5] [--y 0.37] [--w 0.95] [--opacity 0.8] [--font montserrat] [--depth 0.22] [--style frost]
                                  [--until SECONDS --fade 1.5]   (text dissolves and is gone at `until`, then plain footage)
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
VERSION = "frost-8-solid3d"  # bump when the look changes, so cached bakes are redone
SIDE = 0.58  # extrusion brightness at the back (frost): lower = more 3D contrast against the milky face
BUNDLED = {"montserrat": "Montserrat-SemiBold.ttf", "playfair": "PlayfairDisplay-Bold.ttf", "impact": "Anton-Regular.ttf"}


def _font(font, px):
    for cand in (BUNDLED.get(font), BUNDLED["montserrat"]):
        if cand and (ROOT / "fonts" / cand).exists():
            return ImageFont.truetype(str(ROOT / "fonts" / cand), px)
    return ImageFont.load_default(px)


def _art(text, font="montserrat", depth=0.22, px=360, stars=False):
    """The sign's canvas: (W, H, ext, mask(dx, dy) -> letters (+ stars) float mask, star_mask or None). stars = the ring of
    12 golden five-pointed stars of the European flag around the text (the text sits in the middle, like the flag)."""
    f = _font(font, px)
    l, t, r, b = f.getbbox(text) if text.strip() else (0, 0, 1, 1)
    ext = max(2, round(px * depth))
    pad = ext + px // 5
    tw, th = r - l, b - t
    sr = px * 0.42  # star radius
    if not stars:
        W, H = tw + 2 * pad, th + 2 * pad
        ox, oy = pad - l, pad - t
        ring = None
    else:
        R = max(tw * 0.5 * 1.18 + sr, px * 1.6)
        W = H = int(2 * (R + sr) + 2 * pad)
        ox, oy = W / 2 - tw / 2 - l, H / 2 - th / 2 - t
        ring = [(W / 2 + R * np.cos(-np.pi / 2 + k * np.pi / 6), H / 2 + R * np.sin(-np.pi / 2 + k * np.pi / 6)) for k in range(12)]

    def star_mask(dx=0.0, dy=0.0):
        m = Image.new("L", (W, H), 0)
        d = ImageDraw.Draw(m)
        for cx, cy in ring:
            pts = [(cx + dx + (sr if k % 2 == 0 else sr * 0.382) * np.cos(-np.pi / 2 + k * np.pi / 5),
                    cy + dy + (sr if k % 2 == 0 else sr * 0.382) * np.sin(-np.pi / 2 + k * np.pi / 5)) for k in range(10)]
            d.polygon(pts, fill=255)
        return np.asarray(m, np.float32) / 255

    def mask(dx=0.0, dy=0.0):
        m = Image.new("L", (W, H), 0)
        if text.strip():
            ImageDraw.Draw(m).text((ox + dx, oy + dy), text, font=f, fill=255)
        m = np.asarray(m, np.float32) / 255
        return np.maximum(m, star_mask(dx, dy)) if ring else m

    return W, H, ext, mask, (star_mask if ring else None)


def layers(text, font="montserrat", depth=0.22, px=360, stars=False):
    """Float32 masks in sprite space: face, side (extrusion), shade (side darkness 0..1), rim (light top edges), shadow.
    A mild perspective is baked in (bottom 8 % narrower: seen from above, the top of a standing sign is nearer)."""
    W, H, ext, mask, _ = _art(text, font, depth, px, stars)
    face = mask()
    side, shade = np.zeros_like(face), np.zeros_like(face)
    ep = max(2, round(px * min(depth, 0.25)))  # painted extrusion stays moderate; the canvas (pad) follows `depth` like sprite()
    for k in range(ep, 0, -1):  # extrusion UP and a bit right: from a drone/eye level above you see the letters' top faces
        m = mask(k * 0.3, -k)
        side = np.maximum(side, m)
        shade = np.where(m > 0.5, 1 - k / ep, shade)
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


def sprite(text, font="montserrat", depth=0.22, px=360, stars=False):
    """For the real 3D path: the flat letters (face) and the contact shadow in the SAME canvas as layers() (so placement is
    identical), plus mip levels (each half size) so the letters are never sampled with gaps when they are small on screen.
    Returns ([face levels], [shadow levels], [gold star levels] | None), ext/H = extrusion depth / canvas height."""
    W, H, ext, mask, star_mask = _art(text, font, depth, px, stars)
    face = mask()
    shadow = cv2.GaussianBlur(mask(0, px * 0.06), (0, 0), px / 10)
    gold = star_mask() if star_mask else None
    fp, sp, gp = [face], [shadow], ([gold] if gold is not None else None)
    while min(fp[-1].shape) > 64:
        fp.append(cv2.pyrDown(fp[-1])); sp.append(cv2.pyrDown(sp[-1]))
        if gp:
            gp.append(cv2.pyrDown(gp[-1]))
    return (fp, sp, gp), ext / H


def _sample(pyr, Hc, ux, uy, size):
    """Letters from the canvas mip level that fits the on-screen size, sampled ONCE straight into the output pixels
    (ux, uy = their undistorted coordinates) - no second resampling for the lens."""
    W, H = size
    q = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    s = cv2.perspectiveTransform(q[None], Hc)[0]
    scale = min(np.linalg.norm(s[1] - s[0]) / W, np.linalg.norm(s[2] - s[1]) / H, np.linalg.norm(s[3] - s[2]) / W)
    lv = int(np.clip(np.floor(np.log2(1 / max(scale, 1e-6))), 0, len(pyr) - 1))
    D = np.diag([2.0 ** lv, 2.0 ** lv, 1.0])
    Hi = np.linalg.inv(Hc @ D)
    zx = Hi[0, 0] * ux + Hi[0, 1] * uy + Hi[0, 2]
    zy = Hi[1, 0] * ux + Hi[1, 1] * uy + Hi[1, 2]
    zw = Hi[2, 0] * ux + Hi[2, 1] * uy + Hi[2, 2]
    return cv2.remap(pyr[lv], (zx / zw).astype(np.float32), (zy / zw).astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def render3d(spr, size, front, back, und):
    """True 3D extruded letters: the face at the sign plane plus slices every ~1 px back to `back` (the sign moved back along
    its normal by the extrusion depth), so the sides are exactly what this camera would see - from above the tops, from the
    side the flanks. Returns the 5 layers of compose() in output pixels: face, side, shade (1 = front edge, 0 = back),
    rim (crisp glass edges), shadow."""
    (fp, sp, gp), (W, H) = (spr if len(spr) == 3 else (*spr, None)), size
    FH, FW = und.shape[:2]
    canvas = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    allq = np.r_[front, back]
    x0, y0 = np.floor(allq.min(0)).astype(int) - 60
    x1, y1 = np.ceil(allq.max(0)).astype(int) + 60
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, FW), min(y1, FH)
    L = np.zeros((FH, FW, 6), np.float32)
    if x1 <= x0 or y1 <= y0:
        return L
    ux, uy = und[y0:y1, x0:x1, 0], und[y0:y1, x0:x1, 1]
    Hf = lambda t: cv2.getPerspectiveTransform(canvas, (front + t * (back - front)).astype(np.float32)).astype(np.float64)
    face = _sample(fp, Hf(0), ux, uy, (W, H))
    n = int(np.clip(np.ceil(np.abs(back - front).max() * 1.2), 1, 64))
    side = np.zeros_like(face); dep = np.ones_like(face); gs = np.zeros_like(face)
    for j in range(n, 0, -1):  # back to front: the frontmost slice decides the depth seen at a pixel
        m = _sample(fp, Hf(j / n), ux, uy, (W, H))
        side = np.maximum(side, m)
        dep = np.where(m > 0.5, j / n, dep)
        if gp:
            gs = np.where(m > 0.5, _sample(gp, Hf(j / n), ux, uy, (W, H)), gs)
    side = np.clip(side - face, 0, 1)
    k3 = np.ones((3, 3), np.uint8)
    sil = np.maximum(face, side)
    rim = np.maximum(np.clip(face - cv2.erode(face, k3), 0, 1), 0.5 * np.clip(sil - cv2.erode(sil, k3), 0, 1))
    gold = np.where(face > 0.5, _sample(gp, Hf(0), ux, uy, (W, H)), gs) if gp else np.zeros_like(face)  # star parts
    L[y0:y1, x0:x1] = np.dstack([face, side, 1 - dep, rim, _sample(sp, Hf(0), ux, uy, (W, H)), gold])
    return L


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
        side_col = blur * 0.12 + (SIDE + 0.22 * shade) * 0.88  # clean milky glass, a bit darker towards the back
        face_col = blur * 0.3 + 0.7
    if L.shape[2] > 5:  # golden stars of the European flag: gold glass, the sides a deeper gold
        g = L[..., 5:6]
        gface = (blur if style != "solid" else x) * 0.12 + np.float32([0.10, 0.78, 1.0]) * 0.88
        gside = np.float32([0.05, 0.50, 0.80]) * (0.75 + 0.25 * shade)
        face_col = face_col * (1 - g) + gface * g
        side_col = side_col * (1 - g) + gside * g
    x = x * (1 - side * vis) + side_col * side * vis
    x = x * (1 - face * vis) + face_col * face * vis
    x = x + rim * vis * 0.55  # light catches the top edges
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def tracks(grays):
    lp = dict(winSize=(21, 21), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    n, (H, W) = len(grays), grays[0].shape
    T = []  # list of arrays (n, 2) with nan
    pts = np.zeros((0, 1, 2), np.float32); ids = np.zeros(0, int)
    for k, g in enumerate(grays):
        if k and len(pts):
            p1, st, _ = cv2.calcOpticalFlowPyrLK(grays[k - 1], g, pts, None, **lp)
            p2, st2, _ = cv2.calcOpticalFlowPyrLK(g, grays[k - 1], p1, None, **lp)
            fb = np.linalg.norm((p2 - pts).reshape(-1, 2), axis=1)
            inb = (p1[:, 0, 0] >= 0) & (p1[:, 0, 0] < W) & (p1[:, 0, 1] >= 0) & (p1[:, 0, 1] < H)
            ok = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 0.5) & inb
            pts, ids = p1[ok], ids[ok]
        for i, p in zip(ids, pts.reshape(-1, 2)):
            T[i][k] = p
        if len(pts) < 1500:
            mask = np.full((H, W), 255, np.uint8)
            for p in pts.reshape(-1, 2):
                cv2.circle(mask, (int(p[0]), int(p[1])), 10, 0, -1)
            new = cv2.goodFeaturesToTrack(g, 2000 - len(pts), 0.01, 10, mask=mask)
            if new is not None:
                nid = np.arange(len(T), len(T) + len(new))
                for i, p in zip(nid, new.reshape(-1, 2)):
                    a = np.full((n, 2), np.nan, np.float32); a[k] = p; T.append(a)
                pts = np.concatenate([pts, new.astype(np.float32)]); ids = np.concatenate([ids, nid])
    return np.array(T)  # (m, n, 2)

def triangulate(K, P1, P2, x1, x2):
    X = cv2.triangulatePoints(K @ P1, K @ P2, x1.T.astype(np.float64), x2.T.astype(np.float64))
    return (X[:3] / X[3]).T

def proj(K, R, t, X):
    x = (K @ (R @ X.T + t.reshape(3, 1))).T
    return x[:, :2] / x[:, 2:3], x[:, 2]

def _angle(K, Pa, Pb, xa, xb):
    Ki = np.linalg.inv(K)
    da = (Pa[:, :3].T @ (Ki @ np.c_[xa, np.ones(len(xa))].T)).T
    db = (Pb[:, :3].T @ (Ki @ np.c_[xb, np.ones(len(xb))].T)).T
    c = (da * db).sum(1) / np.linalg.norm(da, axis=1) / np.linalg.norm(db, axis=1)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def _tri_check(K, Pa, Pb, xa, xb, thr=2.0):
    X = triangulate(K, Pa, Pb, xa, xb)
    ok = np.isfinite(X).all(1)
    for P, x in ((Pa, xa), (Pb, xb)):
        p, z = proj(K, P[:, :3], P[:, 3], np.nan_to_num(X))
        ok &= (z > 0) & (np.linalg.norm(p - x, axis=1) < thr)
    return X, ok


def _pnp(K, X, x, guess):
    """Pose from 3D-2D matches, started from the previous frame's pose (tiny motion between frames): Levenberg-Marquardt on
    all points, then on the points that fit (error < max(1 px, 3 x median)), twice. RANSAC was tried: on this mostly flat
    aerial scene its minimal samples are degenerate and it returned wild poses."""
    if len(X) < 12:
        return None
    rv, tv = guess[0].copy().reshape(3, 1), guess[1].copy().reshape(3, 1)
    keep = np.ones(len(X), bool)
    for _ in range(4):
        rv, tv = cv2.solvePnPRefineLM(X[keep], x[keep], K, None, rv, tv)
        p, _ = cv2.projectPoints(X, rv, tv, K, None)
        e = np.linalg.norm(p.reshape(-1, 2) - x, axis=1)
        keep = e < max(1.0, 3 * np.median(e))
    if keep.sum() < max(10, 0.3 * len(X)) or np.median(e[keep]) > 2.0:
        return None
    return rv.ravel(), tv.ravel()


def _P(pose):
    return np.hstack([cv2.Rodrigues(pose[0])[0], pose[1].reshape(3, 1)])


def _rot(rv):
    """Rodrigues for many vectors at once: (N,3) -> (N,3,3)."""
    th = np.linalg.norm(rv, axis=1)[:, None, None] + 1e-12
    k = rv / th[:, :, 0]
    Kx = np.zeros((len(rv), 3, 3))
    Kx[:, 0, 1], Kx[:, 0, 2], Kx[:, 1, 0], Kx[:, 1, 2], Kx[:, 2, 0], Kx[:, 2, 1] = -k[:, 2], k[:, 1], k[:, 2], -k[:, 0], -k[:, 1], k[:, 0]
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx


def bundle(K, dist, poses, X, T, vis, frames, log=print, iters=40):
    """Bundle adjustment (scipy least_squares, sparse Jacobian, soft-L1 loss) over `frames` (frame 0 stays fixed) and all
    3D points seen there; also refines the focal length."""
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix
    frames = [k for k in frames if k > 0]
    pts = np.where(~np.isnan(X[:, 0]))[0]
    fi = {k: j for j, k in enumerate(frames)}; pi = {p: j for j, p in enumerate(pts)}
    obs_k, obs_p = [], []
    for k in [0] + frames:
        p = pts[vis[pts, k]]
        obs_k += [k] * len(p); obs_p += list(p)
    obs_k, obs_p = np.array(obs_k), np.array(obs_p)
    uv = T[obs_p, obs_k]
    nf, npt = len(frames), len(pts)
    x0 = np.concatenate([[K[0, 0], dist[0], dist[1]], np.concatenate([np.r_[poses[k][0], poses[k][1]] for k in frames]) if nf else [], X[pts].ravel()])
    cam_j = np.array([fi.get(k, -1) for k in obs_k]); pt_j = np.array([pi[p] for p in obs_p])
    cx, cy = K[0, 2], K[1, 2]
    O = 3

    def res(v):
        f, k1, k2 = v[:3]
        C = np.vstack([np.zeros((1, 6)), v[O:O + 6 * nf].reshape(nf, 6)])  # row 0 = fixed frame 0
        P = v[O + 6 * nf:].reshape(npt, 3)
        R = _rot(C[:, :3])
        Xc = np.einsum("nij,nj->ni", R[cam_j + 1], P[pt_j]) + C[cam_j + 1, 3:]
        a, b = Xc[:, 0] / Xc[:, 2], Xc[:, 1] / Xc[:, 2]
        r2 = a * a + b * b
        d = 1 + k1 * r2 + k2 * r2 * r2  # radial lens distortion (wide drone/phone lenses)
        return np.concatenate([(f * a * d + cx - uv[:, 0]), (f * b * d + cy - uv[:, 1])])

    m = len(obs_k)
    A = lil_matrix((2 * m, len(x0)), dtype=np.uint8)
    rows = np.arange(m)
    for r0 in (0, m):
        for c in range(O):
            A[r0 + rows, c] = 1
        sel = cam_j >= 0
        for c in range(6):
            A[r0 + rows[sel], O + 6 * cam_j[sel] + c] = 1
        for c in range(3):
            A[r0 + rows, O + 6 * nf + 3 * pt_j + c] = 1
    r = least_squares(res, x0, jac_sparsity=A, loss="soft_l1", f_scale=1.0, x_scale="jac", method="trf", tr_solver="lsmr", max_nfev=iters)
    v = r.x
    K = K.copy(); K[0, 0] = K[1, 1] = v[0]
    for j, k in enumerate(frames):
        poses[k] = (v[O + 6 * j:O + 3 + 6 * j], v[O + 3 + 6 * j:O + 6 + 6 * j])
    X[pts] = v[O + 6 * nf:].reshape(npt, 3)
    e = np.abs(r.fun).reshape(2, -1)
    e = np.hypot(e[0], e[1])
    pe = np.bincount(pt_j, e, npt) / np.maximum(np.bincount(pt_j, None, npt), 1)
    X[pts[pe > 2.5]] = np.nan  # points that don't fit (moving people, bad tracks) are dropped
    log(f"bundle: {nf + 1} frames, {npt} points, f={v[0]:.0f}, k1={v[1]:.3f}, k2={v[2]:.3f}, median error {np.median(e):.2f}px")
    return K, np.array([v[1], v[2], 0, 0, 0])


def solve(T, f, W, H, log=print, step=3):
    """Incremental camera solve with bundle adjustment: init from frame 0 + first frame with enough parallax, PnP frame by
    frame, new tracks triangulated once their rays differ by > 3 deg, global bundle adjustment every 60 frames and at the end."""
    cv2.setRNGSeed(0)
    m, n, _ = T.shape
    T = T.astype(np.float64)
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], np.float64)
    vis = ~np.isnan(T[..., 0])
    T0, dist = T, np.zeros(5)

    def und(K, dist):  # undistorted pixel tracks (pinhole) for the PnP / triangulation steps
        u = T0.copy()
        u[vis] = cv2.undistortPoints(T0[vis].reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)
        return u
    init = None
    for k1 in range(max(2, n // 10), n, max(1, n // 30)):
        c = np.where(vis[:, 0] & vis[:, k1])[0]
        if len(c) < 60: break
        E, inl = cv2.findEssentialMat(T[c, 0], T[c, k1], K, cv2.RANSAC, 0.999, 1.0)
        if E is None or E.shape != (3, 3): continue
        cnt, R, t, inl2 = cv2.recoverPose(E, T[c, 0], T[c, k1], K, mask=inl.copy())
        ci = c[inl2.ravel() > 0]
        if len(ci) < 50: continue
        P1 = np.hstack([R, t])
        Xi, ok = _tri_check(K, np.eye(3, 4), P1, T[ci, 0], T[ci, k1])
        ang = np.median(_angle(K, np.eye(3, 4), P1, T[ci, 0], T[ci, k1])) if len(ci) else 0
        if ok.sum() >= 50 and ang > 1.0:  # below 1 deg the camera only turned: no depth to recover
            init = (k1, ci[ok], Xi[ok], cv2.Rodrigues(R)[0].ravel(), t.ravel())
            if ang > 2.0: break
    if init is None:
        raise ValueError("world text: the camera barely moves (or too little texture) - no 3D solve possible")
    k1, ci, Xi, r1, t1 = init
    X = np.full((m, 3), np.nan); X[ci] = Xi
    poses = [None] * n
    poses[0] = (np.zeros(3), np.zeros(3)); poses[k1] = (r1, t1)
    K, dist = bundle(K, dist, poses, X, T0, vis, [k1], log); T = und(K, dist)
    first = np.argmax(vis, 1)
    done = [0]
    for k in range(1, n):
        ok = vis[:, k] & ~np.isnan(X[:, 0])
        p = _pnp(K, X[ok], T[ok, k], poses[k - 1])
        poses[k] = p if p is not None else poses[k - 1]
        if p is None: log(f"frame {k}: pose lost, kept previous")
        done.append(k)
        cand = np.where(vis[:, k] & np.isnan(X[:, 0]) & (first < k))[0]
        if len(cand):
            Pk = _P(poses[k])
            for a in np.unique(first[cand]):
                ii = cand[first[cand] == a]
                Pa = _P(poses[a])
                ii = ii[_angle(K, Pa, Pk, T[ii, a], T[ii, k]) > 3.0]
                if len(ii):
                    Xn, okn = _tri_check(K, Pa, Pk, T[ii, a], T[ii, k], 2.0)
                    X[ii[okn]] = Xn[okn]
        if k % 60 == 0 or k == n - 1:  # 30 / every 2nd frame: same accuracy, 2.5x slower
            K, dist = bundle(K, dist, poses, X, T0, vis, done[::step] + ([k] if k % step else []), log, iters=20 if k < n - 1 else 40); T = und(K, dist)
            # frames skipped by the bundle: re-PnP them against the refined points
            for j in done:
                if j % step and j != k:
                    o = vis[:, j] & ~np.isnan(X[:, 0])
                    p = _pnp(K, X[o], T[o, j], poses[j])
                    if p is not None: poses[j] = p
    Ps = [_P(p) for p in poses]
    errs = []
    for k in range(n):
        ok = vis[:, k] & ~np.isnan(X[:, 0])
        pp, _ = proj(K, Ps[k][:, :3], Ps[k][:, 3], X[ok])
        errs.append(np.median(np.linalg.norm(pp - T[ok, k], axis=1)) if ok.sum() else 99)
    return K, dist, Ps, X, np.array(errs)



def sign(K, dist, poses, X, box0, fg0, back=0.0):
    """The text as a real, upright sign in the solved 3D scene: its base stands at the depth of the scene behind the letters in
    frame 0 (median of the 3D points there, near pixels excluded), upright = normal of the dominant plane (ground and flat
    roofs), its frame-0 projection = the 2D box the user placed. Returns the 4 corners (TL, TR, BR, BL) per frame (n, 4, 2);
    with back > 0 also the 4 corners of the same sign moved back (away from the camera) by back x its height: (n, 8, 2)."""
    good = np.where(~np.isnan(X[:, 0]))[0]
    Xg = X[good]
    p0, _ = cv2.projectPoints(Xg, np.zeros(3), np.zeros(3), K, dist)
    p0 = p0.reshape(-1, 2)
    b = box0.reshape(4, 2)
    (x0, y0), (x1, y1) = b.min(0), b.max(0)
    mx, my = (x1 - x0) * 0.1, (y1 - y0) * 0.5
    H, W = fg0.shape
    q = np.clip(p0.round().astype(int), 0, [W - 1, H - 1])
    near = (p0[:, 0] > x0 - mx) & (p0[:, 0] < x1 + mx) & (p0[:, 1] > y0) & (p0[:, 1] < y1 + my) & (fg0[q[:, 1], q[:, 0]] < 0.3)
    Z = np.median(Xg[near, 2]) if near.sum() >= 8 else np.median(Xg[:, 2])
    # dominant roughly-horizontal plane (ground / flat roofs) by RANSAC -> "up". Only normals within 60 deg of the image
    # vertical count (a far wall facing the camera is not the ground); distance relative to depth (far points are noisier)
    rng = np.random.default_rng(0)
    best, up = 0, np.array([0.0, -1.0, 0.0])  # fallback: image-up of frame 0 (sign faces the first frame)
    for _ in range(600):
        s = Xg[rng.choice(len(Xg), 3, replace=False)]
        nrm = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(nrm) < 1e-12: continue
        nrm /= np.linalg.norm(nrm)
        if abs(nrm[1]) < 0.5: continue
        cnt = (np.abs((Xg - s[0]) @ nrm) < 0.01 * Xg[:, 2]).sum()
        if cnt > best: best, up = cnt, nrm
    if best < 0.15 * len(Xg): up = np.array([0.0, -1.0, 0.0])
    ray = lambda u: Z * np.r_[cv2.undistortPoints(np.float64([[u]]), K, dist).ravel(), 1.0]
    BL, BR = ray(b[3]), ray(b[2])
    C = (BL + BR) / 2
    if (cv2.projectPoints(C + 0.01 * Z * up, np.zeros(3), np.zeros(3), K, dist)[0].ravel()[1] >
            cv2.projectPoints(C, np.zeros(3), np.zeros(3), K, dist)[0].ravel()[1]): up = -up  # must point up in the image
    # the base stays exactly where the user put it in frame 0; "up" is made perpendicular to it. A slightly wrong ground
    # estimate (handheld phone, few ground points) then only leans the sign a little back/forward - it never turns it crooked
    dh = (BR - BL) / np.linalg.norm(BR - BL)
    up = up - (up @ dh) * dh; up /= np.linalg.norm(up)
    q = cv2.undistortPoints(np.float64([[(b[0] + b[1]) / 2]]), K, dist).ravel()[1]  # top edge where the user put it in frame 0
    h = (q * C[2] - C[1]) / (up[1] - q * up[2])  # solves (C + h up)_y / (C + h up)_z = q
    corners = np.float64([BL + h * up, BR + h * up, BR, BL])
    sign.geo = (BL, BR, up)  # for checks: the ground line the sign stands on
    if back > 0:
        nb = np.cross(dh, up); nb *= np.sign(nb @ C) or 1.0  # sign normal, pointing away from the first camera
        corners = np.r_[corners, corners + back * abs(h) * nb]
    # pinhole (undistorted) pixels: corners often leave the frame when the camera comes close, and the lens polynomial is
    # only valid inside it - bake() bends the rendered text with the exact per-pixel lens map instead
    return np.stack([cv2.projectPoints(corners, rv, tv, K, None)[0].reshape(-1, 2) for rv, tv in poses])


def _smooth(C, sigma=1.5):
    """Light temporal smoothing of the corner paths (removes sub-pixel solver jitter, keeps every real move)."""
    R = C.reshape(len(C), -1).astype(np.float64)
    if len(R) < 7:
        return C
    r = int(3 * sigma)
    ker = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2); ker /= ker.sum()
    R = np.stack([np.convolve(np.pad(R[:, j], r, mode="reflect", reflect_type="odd"), ker, "valid") for j in range(R.shape[1])], 1)
    return R.reshape(C.shape)


def _pan(T, box):
    """Fallback for a camera that only turns or stands still (no 3D possible): one homography per frame from the tracks -
    exact for pure rotation. Direct to frame 0 while enough tracks reach back, else chained from the previous frame."""
    n = T.shape[1]
    vis = ~np.isnan(T[..., 0])
    Hs = [np.eye(3)]
    for k in range(1, n):
        c0, c1 = vis[:, 0] & vis[:, k], vis[:, k - 1] & vis[:, k]
        H = None
        if c0.sum() >= 50:
            H, _ = cv2.findHomography(T[c0, 0], T[c0, k], cv2.RANSAC, 1.5)
        if H is None and c1.sum() >= 8:
            H, _ = cv2.findHomography(T[c1, k - 1], T[c1, k], cv2.RANSAC, 1.5)
            H = None if H is None else H @ Hs[-1]
        Hs.append(H if H is not None else Hs[-1])
    return np.stack([cv2.perspectiveTransform(box.reshape(-1, 1, 2).astype(np.float64), H).reshape(4, 2) for H in Hs])


def place(grays, box0, fg0, log=print, back=0.0):
    """Tracks + 3D camera solve (frames scaled to <= 1280 px wide) + the sign. Returns (corners per frame in full-size pinhole
    pixels, K at full size, lens distortion, per-frame solve error or None when the fallback for a non-moving camera ran)."""
    s = min(1.0, 1280 / grays[0].shape[1])
    if s < 1:
        grays = [cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) for g in grays]
        fg0 = cv2.resize(fg0, (grays[0].shape[1], grays[0].shape[0]))
    H, W = grays[0].shape
    log("world text: tracking points ...")
    T = tracks(grays)
    log(f"world text: 3D camera solve ({T.shape[0]} tracks) ...")
    try:
        K, dist, Ps, X, err = solve(T, 0.7 * W, W, H, log=lambda m: log("world text: " + m))
    except ValueError as e:
        log(f"{e} -> following the whole image instead (exact when the camera only turns or stands still)")
        return _smooth(_pan(T, box0 * s)) / s, np.eye(3), np.zeros(5), None
    poses = [(cv2.Rodrigues(P[:, :3])[0].ravel(), P[:, 3]) for P in Ps]
    C = _smooth(sign(K, dist, poses, X, box0 * s, fg0, back)) / s
    K = K.copy(); K[:2] /= s
    return C, K, dist, err


def bake(src, a, b, text, out, x=0.5, y=0.37, w=0.95, opacity=0.8, font="montserrat", depth=0.4, style="frost", until=None,
         fade=1.5, stars=False, log=print):
    """until = seconds into the shot when the text is gone (it dissolves - fades and softens - over `fade` seconds before);
    after that the footage runs on untouched and is neither tracked nor solved (fast, and the camera may do anything)."""
    import clipcut, fxface
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n_all = max(1, round((b - a) * fps))
    n = n_all if until is None else max(2, min(n_all, int(np.ceil(float(until) * fps)) + 1))
    dsess = fxface._session(fxface.DEPTH_MODEL, fxface.DEPTH_URL, "depth (99 MB)")
    spr, back = sprite(text, font, depth, stars=bool(stars))  # real 3D letters (+ star ring), needs the 3D solve
    SH, SW = spr[0][0].shape
    imgs = []
    for k in range(n):
        img = fxface._frame(cap, a) if k == 0 else cap.read()[1]  # sequential reads after the seek: exact frames, fast
        if img is None:
            break
        imgs.append(img)
    if not imgs:
        raise ValueError(f"cannot read {src} at {a}s")
    FH, FW = imgs[0].shape[:2]
    fgs, T = [], None
    for k, img in enumerate(imgs):
        if k % 3 == 0:  # depth every 3rd frame (edges move < 1 px per frame)
            d = fxface.depth_map(dsess, img)
            t, _ = cv2.threshold((d * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            T = t / 255 if T is None else 0.8 * T + 0.2 * t / 255
        fgs.append((np.clip((d - T) / 0.04 + 0.5, 0, 1) * 255).astype(np.uint8))
        if k % 60 == 0:
            log(f"world text: depth {k}/{len(imgs)}")
    sw = w * FW / SW
    box0 = np.float32([[x * FW - sw * SW / 2, y * FH - sw * SH], [x * FW + sw * SW / 2, y * FH - sw * SH],
                       [x * FW + sw * SW / 2, y * FH], [x * FW - sw * SW / 2, y * FH]])  # bottom centre at (x, y)
    C, Kf, dist, err = place([cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in imgs], box0, fgs[0] / 255.0, log, back)
    L0 = None if C.shape[1] == 8 else layers(text, font, depth, stars=bool(stars))[0]  # no 3D solve: painted (same canvas)
    if err is not None:
        log(f"world text: camera solved, reprojection error mean {err.mean():.2f} px, worst frame {err.max():.2f} px")
    bake.corners, bake.err, bake.K, bake.dist = C, err, Kf, dist
    canvas = np.float32([[0, 0], [SW, 0], [SW, SH], [0, SH]])
    # lens map: for every real (distorted) pixel, where it lies in the pinhole image the text is drawn into
    grid = np.stack(np.meshgrid(np.arange(FW, dtype=np.float32), np.arange(FH, dtype=np.float32)), -1).reshape(-1, 1, 2)
    und = cv2.undistortPoints(grid, Kf, dist, P=Kf).reshape(FH, FW, 2)
    m = int(np.ceil(max(0, -und[..., 0].min(), -und[..., 1].min(), und[..., 0].max() - FW, und[..., 1].max() - FH))) + 2
    mapx, mapy = und[..., 0] + m, und[..., 1] + m
    with tempfile.TemporaryDirectory() as tmp:
        for k, img in enumerate(imgs):
            if L0 is None:
                L = render3d(spr, (SW, SH), C[k, :4], C[k, 4:], und)
            else:
                Hk = cv2.getPerspectiveTransform(canvas, (C[k] + m).astype(np.float32))
                L = np.dstack([cv2.remap(cv2.warpPerspective(L0[..., i], Hk, (FW + 2 * m, FH + 2 * m), flags=cv2.INTER_LINEAR),
                                         mapx, mapy, cv2.INTER_LINEAR) for i in range(5)])
            if until is not None:  # dissolve: fade + soften towards `until`
                t = k / fps
                al = float(np.clip((float(until) - t) / max(float(fade), 1e-3), 0, 1))
                al = al * al * (3 - 2 * al)  # smoothstep
                if al < 1:
                    sig = (1 - al) * FH / 120
                    L = cv2.GaussianBlur(L, (0, 0), sig).reshape(L.shape) * al if sig > 0.3 else L * al
            cv2.imwrite(f"{tmp}/{k:06d}.png", compose(img, L, fgs[k].astype(np.float32) / 255, opacity, style))
            if k % 60 == 0:
                log(f"world text: rendering {k}/{len(imgs)}")
        imgs.clear()
        for k in range(n, n_all):  # after the text is gone: the footage as it is
            ok, img = cap.read()
            if not ok:
                break
            cv2.imwrite(f"{tmp}/{k:06d}.png", img)
        cap.release()
        clipcut.run([clipcut.FF, "-y", "-framerate", str(fps), "-i", f"{tmp}/%06d.png", "-ss", str(a), "-t", str(b - a), "-i", str(src),
                     "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p",
                     "-c:a", "aac", "-b:a", "192k", "-shortest", str(out)])
    return out


KEYS = ("x", "y", "w", "opacity", "font", "depth", "style", "until", "fade", "stars")


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
    """Synthetic drone shot with a known camera (forward + down + turn, lens distortion, 0.2 px tracking noise): ground, a
    flat-roofed building and far background as 3D points. The camera solve + sign placement must put the sign's corners where
    the true camera puts them (< 3 px in every frame, sign ~1200 px wide); the frost style must look milky and occluded letters stay untouched."""
    rng = np.random.default_rng(1)
    W, H, n, f = 1280, 720, 60, 900.0
    Kt = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]]); dt = np.array([-0.05, 0.01, 0, 0, 0])
    P = np.r_[np.c_[rng.uniform(-30, 30, 1500), np.full(1500, 10.0), rng.uniform(15, 80, 1500)],  # ground (y down = 10)
              np.c_[rng.uniform(-12, 12, 400), np.full(400, 2.0), rng.uniform(18, 30, 400)],  # roof
              np.c_[rng.uniform(-60, 60, 300), rng.uniform(-15, 10, 300), np.full(300, 120.0)]]  # far background
    tilt = cv2.Rodrigues(np.array([0.35, 0.0, 0.0]))[0]  # looking down
    poses = []
    for k in range(n):
        R = cv2.Rodrigues(np.array([0.002 * k, 0.003 * k, 0.0]))[0] @ tilt
        c = np.array([0.05 * k, 0.03 * k, 0.25 * k])  # camera centre: forward and a bit down/right
        poses.append((cv2.Rodrigues(R)[0].ravel(), -R @ c))
    T = np.full((len(P), n, 2), np.nan)
    for k, (rv, tv) in enumerate(poses):
        z = (cv2.Rodrigues(rv)[0] @ P.T).T[:, 2] + tv[2]
        p = cv2.projectPoints(P, rv, tv, Kt, dt)[0].reshape(-1, 2) + rng.normal(0, 0.2, (len(P), 2))
        ok = (z > 1) & (p[:, 0] >= 0) & (p[:, 0] < W) & (p[:, 1] >= 0) & (p[:, 1] < H)
        T[ok, k] = p[ok]
    T = T[(~np.isnan(T[..., 0])).sum(1) >= 2]  # synthetic tracks never vanish and come back, as with LK
    R0, t0 = cv2.Rodrigues(poses[0][0])[0], poses[0][1]  # truth in frame-0 camera coordinates, like the solve
    P = (R0 @ P.T).T + t0
    poses = [(cv2.Rodrigues(cv2.Rodrigues(rv)[0] @ R0.T)[0].ravel(), tv - cv2.Rodrigues(rv)[0] @ R0.T @ t0) for rv, tv in poses]
    K, dist, Ps, X, e = solve(T, 0.7 * W, W, H, log=lambda m: None)
    est = [(cv2.Rodrigues(Q[:, :3])[0].ravel(), Q[:, 3]) for Q in Ps]
    box0 = np.float32([[240, 120], [1040, 120], [1040, 260], [240, 260]])
    fg0 = np.zeros((H, W))
    Ct = sign(Kt, dt, poses, P, box0, fg0)
    Ce = sign(K, dist, est, X, box0, fg0)
    err = float(np.abs(Ct - Ce).max())
    assert err < 3.0, f"sign corners off by {err:.2f} px (solve error {e.mean():.2f} px)"
    # camera that only turns (tripod pan): no 3D solve, the homography fallback must follow exactly
    rots = [cv2.Rodrigues(np.array([0.0, 0.004 * k, 0.001 * k]))[0] @ tilt for k in range(n)]
    Tp = np.full((len(P), n, 2), np.nan)
    for k, R in enumerate(rots):
        z = (R @ P.T).T[:, 2]
        p = cv2.projectPoints(P, cv2.Rodrigues(R)[0], np.zeros(3), Kt, None)[0].reshape(-1, 2) + rng.normal(0, 0.2, (len(P), 2))
        ok = (z > 1) & (p[:, 0] >= 0) & (p[:, 0] < W) & (p[:, 1] >= 0) & (p[:, 1] < H)
        Tp[ok, k] = p[ok]
    try:
        solve(Tp, 0.7 * W, W, H, log=lambda m: None)
        raise AssertionError("a turning-only camera must not give a 3D solve")
    except ValueError:
        pass
    Cp = _pan(Tp, box0)
    truth = [cv2.perspectiveTransform(box0.reshape(-1, 1, 2).astype(np.float64), Kt @ R @ rots[0].T @ np.linalg.inv(Kt)).reshape(4, 2) for R in rots]
    perr = float(np.abs(Cp - np.stack(truth)).max())
    assert perr < 2.0, f"pan fallback off by {perr:.2f} px"
    # real 3D extrusion: back face 20 px higher on screen (seen from above) -> sides above the letters, none below, crisp face
    (fp, sp, _), bk = sprite("TEST", px=120)
    sh, sw_ = fp[0].shape
    fr_q = np.float32([[100, 100], [100 + sw_, 100], [100 + sw_, 100 + sh], [100, 100 + sh]])
    gx, gy = np.meshgrid(np.arange(800, dtype=np.float32), np.arange(400, dtype=np.float32))
    L3 = render3d(((fp, sp)), (sw_, sh), fr_q, fr_q - [0, 20], np.dstack([gx, gy]))
    col = L3[..., 1].sum(1)
    top, bot = np.nonzero(L3[..., 0].sum(1) > 0)[0][[0, -1]]
    assert col[top - 15:top].sum() > 50 and col[bot + 3:].sum() < 1, "3D sides must show above the letters only"
    assert L3[..., 0].max() > 0.99 and np.isin(np.round(L3[..., 0][L3[..., 0] > 0], 1), [1.0]).mean() > 0.6, "face not crisp"
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
    print(f"worldtext selftest ok (3D sign corners {err:.2f} px, pan fallback {perr:.2f} px, frost face {v:.0f}/255)")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    if sys.argv[1:2] == ["selftest"]:
        selftest(); sys.exit()
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("a", type=float); ap.add_argument("b", type=float); ap.add_argument("text")
    ap.add_argument("--out"); ap.add_argument("--x", type=float, default=0.5); ap.add_argument("--y", type=float, default=0.37)
    ap.add_argument("--w", type=float, default=0.95); ap.add_argument("--opacity", type=float, default=0.8)
    ap.add_argument("--font", default="montserrat"); ap.add_argument("--depth", type=float, default=0.4)
    ap.add_argument("--style", default="frost", choices=["frost", "solid"])
    ap.add_argument("--until", type=float, help="seconds into the shot when the text has dissolved (then plain footage)")
    ap.add_argument("--fade", type=float, default=1.5, help="dissolve duration in seconds")
    ap.add_argument("--stars", action="store_true", help="ring of 12 golden stars (European flag) around the text")
    o = ap.parse_args()
    src = Path(o.src)
    out = o.out or str(src.with_name(f"{src.stem}_{o.a:.2f}-{o.b:.2f}_world.mp4"))
    print(bake(src, o.a, o.b, o.text, out, o.x, o.y, o.w, o.opacity, o.font, o.depth, o.style, o.until, o.fade, o.stars))
