"""CapCut-style edit transitions that ffmpeg's xfade cannot do (motion-blurred zoom / whip / spin / glitch ...).

Recipe source: CapCut 9.5 effect cache, VerticalZoomNear.frag (effect 6724226861666144779): progress eased with
easeInOutQuint, first half magnifies A up to 2x, second half B comes in from 0.67x, both with a 21-tap radial blur
whose spread grows with the eased progress (±28 % at full ease). Everything else here follows the same construction.

render(typ, A, B) takes the two overlap windows as (n, H, W, 3) BGR uint8 arrays and returns n frames: the frames
replace A's tail and B's head on the timeline (CapCut is_overlap), so frame 0 ~ A and frame n-1 ~ B.
"""
import math

import cv2
import numpy as np

K = 7            # taps per motion blur; the shader uses 21, 7 is indistinguishable after x264
BORDER = cv2.BORDER_REFLECT


def ease(p):     # easeInOutQuint from the shader
    return 16 * p ** 5 if p < 0.5 else 1 + 16 * (p - 1) ** 5


def _avg(frames):
    return np.clip(sum(f.astype(np.float32) for f in frames) / len(frames), 0, 255).astype(np.uint8)


def radial(img, mag, spread, angle=0.0, aspread=0.0, shift=(0, 0)):
    """Zoom/rotate about the centre with motion blur: average of K taps spread over mag*(1±spread), angle±aspread."""
    h, w = img.shape[:2]
    taps = []
    for k in range(K):
        u = (2 * k / (K - 1) - 1) if K > 1 else 0
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle + u * aspread, mag * (1 + u * spread))
        m[:, 2] += shift
        taps.append(cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=BORDER))
    return _avg(taps)


def pullin(A, B):
    n = len(A)
    out = []
    for j in range(n):
        tt = ease((j + 0.5) / n)
        if tt < 0.5:     # zoomNear: uv*(1-amount) -> magnification 1/(1-amount), blur spread 0.28*tt
            out.append(radial(A[j], 1 / (1 - tt), 0.28 * tt))
        else:            # uv*(2-amount) -> 1/(2-amount) rising to 1, blur fades with 1-tt
            out.append(radial(B[j], 1 / (2 - tt), 0.28 * (1 - tt)))
    return out


def pullout(A, B):       # time-reversed pull-in: continuous zoom out
    return pullin(B[::-1], A[::-1])[::-1]


def _whip(A, B, axis, forward):
    n, h, w = A.shape[:3]
    span = w if axis == 1 else h
    strip = np.concatenate([A, B] if forward else [B, A], axis=axis + 1)   # A|B side by side, the window slides across
    out = []
    for j in range(n):
        p = (j + 0.5) / n
        e = ease(p)
        pos = e * span if forward else (1 - e) * span
        blur = 0.12 * span * math.sin(math.pi * e)      # tap spread follows the slide speed
        taps = []
        for k in range(K):
            u = (2 * k / (K - 1) - 1)
            x = int(round(min(max(pos + u * blur, 0), span)))
            taps.append(strip[j, :, x:x + w] if axis == 1 else strip[j, x:x + h])
        out.append(_avg(taps))
    return out


def whipleft(A, B): return _whip(A, B, 1, True)
def whipright(A, B): return _whip(A, B, 1, False)
def whipup(A, B): return _whip(A, B, 0, True)
def whipdown(A, B): return _whip(A, B, 0, False)


def spin(A, B):
    n = len(A)
    out = []
    for j in range(n):
        e = ease((j + 0.5) / n)
        ang, mag, spread = 180 * e, 1 + math.sin(math.pi * e), 25 * math.sin(math.pi * e)
        src, a = (A[j], ang) if e < 0.5 else (B[j], ang - 180)   # B continues the same rotation, landing upright
        out.append(radial(src, mag, 0.0, a, spread))
    return out


def glitch(A, B):
    n, h, w = A.shape[:3]
    out = []
    for j in range(n):
        p = (j + 0.5) / n
        g = math.sin(math.pi * p) ** 0.5
        rng = np.random.default_rng(j)
        src, other = (A[j], B[j]) if p < 0.5 else (B[j], A[j])
        img = src.copy()
        dx = int(g * 0.03 * w)
        img[:, :, 2] = np.roll(src[:, :, 2], dx, axis=1)       # RGB split (BGR order: index 2 = red)
        img[:, :, 0] = np.roll(src[:, :, 0], -dx, axis=1)
        for _ in range(int(g * 8)):                               # displaced bands, some showing the other clip
            y0, bh = int(rng.integers(0, h)), int(rng.integers(h // 40, h // 8))
            sh = int(rng.integers(-w // 10, w // 10))
            band = other if rng.random() < 0.4 else img
            img[y0:y0 + bh] = np.roll(band[y0:y0 + bh], sh, axis=1)
        out.append(img)
    return out


def shake(A, B):
    n, h, w = A.shape[:3]
    out = []
    for j in range(n):
        p = (j + 0.5) / n
        g = math.sin(math.pi * p)
        rng = np.random.default_rng(100 + j)
        d = rng.uniform(-1, 1, 2) * g * 0.05 * np.array([w, h])
        out.append(radial(A[j] if p < 0.5 else B[j], 1 + 0.08 * g, 0.03 * g, shift=d))
    return out


def blur(A, B):
    n, h, w = A.shape[:3]
    out = []
    for j in range(n):
        p = (j + 0.5) / n
        s = max(0.5, 0.03 * w * math.sin(math.pi * p))
        a = cv2.GaussianBlur(A[j], (0, 0), s).astype(np.float32)
        b = cv2.GaussianBlur(B[j], (0, 0), s).astype(np.float32)
        out.append(np.clip(a * (1 - p) + b * p, 0, 255).astype(np.uint8))
    return out


def flicker(A, B):
    n = len(A)
    out = []
    for j in range(n):
        p = (j + 0.5) / n
        g = math.sin(math.pi * p)
        src = B[j] if (p < 0.5) == (j % 3 == 2) else A[j]       # every third frame peeks at the other clip
        out.append(np.clip(src.astype(np.float32) * (1 + 0.7 * g * (j % 2)), 0, 255).astype(np.uint8))
    return out


TYPES = {f.__name__: f for f in (pullin, pullout, whipleft, whipright, whipup, whipdown, spin, glitch, shake, blur, flicker)}


def render(typ, A, B):
    A, B = np.ascontiguousarray(A), np.ascontiguousarray(B)
    return TYPES[typ](A, B)


if __name__ == "__main__":   # self-check: every type keeps shape/dtype, pull-in starts on A and lands on B
    n, h, w = 8, 64, 36
    A = np.zeros((n, h, w, 3), np.uint8); A[..., 2] = 200
    B = np.zeros((n, h, w, 3), np.uint8); B[..., 0] = 200
    for name in TYPES:
        fr = render(name, A, B)
        assert len(fr) == n and all(f.shape == (h, w, 3) and f.dtype == np.uint8 for f in fr), name
    fr = render("pullin", A, B)
    assert fr[0][..., 2].mean() > 150 and fr[-1][..., 0].mean() > 150
    assert abs(ease(0.5) - 0.5) < 1e-9 and ease(0) == 0 and abs(ease(1) - 1) < 1e-9
    print("transitions ok:", ", ".join(TYPES))
