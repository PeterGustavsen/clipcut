"""motion.py - best-moment detection: cut where the star DOES something, not where he just stands.

moments(media)                     -> [{"t", "motion", "loud"}] at 8 fps, cached in transcripts/<name>.<mtime>.moments.json
    motion = mean |frame diff| inside the face box grown to the upper body (face cache from faces.py, else whole frame), 0..1
    loud   = RMS of the clip's own audio (speech = hook material), 0..1
Picker(clips, skip)                -> drop-in for trends.Footage: take(src_len, beats=[offsets], hook=False, accent=False, avoid=[(media, s, e)])
    scores every window inside the star-visible ranges: motion peaks aligned with the slice's beat offsets
    (cross-correlation with a bump kernel) + mean motion, loudness for the hook slot, penalties for re-used footage,
    for the clip of the previous slice (hard), for clips of the last 3 slices (mild) and per earlier use of the clip (variety).
    With clip["star"] (clipcut.star_track: [[t, x, y, w, h]] source fractions at 4 fps) a window is only valid when he is
    detected in >= 90 % of its samples incl. the first and last one plus a 0.25 s margin; face height < 7 % of the frame is
    rejected, >= 15 % preferred, faces touching the frame edge are penalised. After take() `picker.frame` = {cx, cy} = mean
    face centre of the window (the cover crop centre, see clipcut.fit_chain).
    With clip["cuts"] (clipcut.scene_cuts) no window spans an internal cut; starts on a motion onset score higher, starts on
    a blur peak (very high frame diff = whip) lower.
python motion.py CLIP CLIP  -> selftest on real footage (needs face_ref.json for ranges, else whole clips).
"""
import json, subprocess, sys
from pathlib import Path

import numpy as np

FPS = 8
W, H = 160, 90  # ponytail: fixed 16:9 raster for every clip, motion energy does not care about aspect


def _ff(name):
    import clipcut  # same folder; lazy so trends.py can import this module without ffmpeg being found yet
    return clipcut.FF if name == "ffmpeg" else clipcut.FP


def _face_boxes(media, cache_dir):
    """{t: [x,y,w,h]} largest face per sampled frame from faces.py's cache (coords in the <=640px-wide frame)."""
    c = Path(cache_dir) / f"{Path(media).name}.{int(Path(media).stat().st_mtime)}.faces.json"
    if not c.exists():
        return {}
    # ponytail: largest face, not the SFace match against face_ref; inside the visible ranges it is him anyway
    return {s["t"]: max(s["faces"], key=lambda f: f["box"][2] * f["box"][3])["box"] for s in json.loads(c.read_text(encoding="utf-8")) if s["faces"]}


def _frames(media):
    out = subprocess.run([_ff("ffmpeg"), "-v", "error", "-i", str(media), "-vf", f"fps={FPS},scale={W}:{H},format=gray", "-f", "rawvideo", "-"],
                         capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
    n = len(out) // (W * H)
    return np.frombuffer(out[:n * W * H], np.uint8).reshape(n, H, W).astype(np.int16)


def _loud(media, n):
    sr = 8000
    r = subprocess.run([_ff("ffmpeg"), "-v", "error", "-i", str(media), "-vn", "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"],
                       capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    pcm = np.frombuffer(r.stdout, np.int16).astype(np.float32)
    hop = sr // FPS
    rms = np.array([np.sqrt(np.mean(pcm[i * hop:(i + 1) * hop] ** 2)) if len(pcm[i * hop:(i + 1) * hop]) else 0.0 for i in range(n)])
    return rms


def _norm(x):
    top = np.percentile(x, 95) if len(x) else 0
    return np.clip(x / top, 0, 1) if top > 0 else np.zeros_like(x)


def moments(media, cache_dir=None):
    media = Path(media)
    cache_dir = Path(cache_dir or media.parent.parent / "transcripts")
    cache_dir.mkdir(exist_ok=True)
    c = cache_dir / f"{media.name}.{int(media.stat().st_mtime)}.moments.json"
    if c.exists():
        return json.loads(c.read_text(encoding="utf-8"))
    fr = _frames(media)
    n = len(fr)
    boxes = _face_boxes(media, cache_dir)
    if boxes:
        import cv2
        cap = cv2.VideoCapture(str(media))
        ow, oh = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT); cap.release()
        sw = min(ow, 640); sh = oh * sw / ow  # faces.py scans a <=640 px wide copy
        ts = np.array(sorted(boxes))
    motion = np.zeros(n)
    for i in range(1, n):
        d = np.abs(fr[i] - fr[i - 1])
        if boxes:
            k = ts[np.argmin(np.abs(ts - i / FPS))]
            if abs(k - i / FPS) <= 0.5:
                x, y, w, h = boxes[k]
                # upper body: 3 face widths wide, half a face above, ~2.5 faces below (gesturing hands)
                x0, x1 = int(max(0, x - w) * W / sw), int(min(sw, x + 2 * w) * W / sw)
                y0, y1 = int(max(0, y - 0.5 * h) * H / sh), int(min(sh, y + 3 * h) * H / sh)
                if x1 - x0 > 2 and y1 - y0 > 2:
                    d = d[y0:y1, x0:x1]
        motion[i] = d.mean()
    motion[1:] = np.minimum(motion[1:], np.percentile(motion[1:], 98) if n > 2 else motion[1:])  # a cut inside the clip is not a gesture
    out = [{"t": round(i / FPS, 3), "motion": round(float(m), 3), "loud": round(float(l), 3)}
           for i, (m, l) in enumerate(zip(_norm(motion), _norm(_loud(media, n))))]
    c.write_text(json.dumps(out), encoding="utf-8")
    return out


FACE_FPS = 4  # faces.py samples every 0.25 s
MIN_FACE, GOOD_FACE = 0.07, 0.15  # face height / frame height


class Picker:
    """Drop-in for trends.Footage, but every slice is chosen by score instead of round-robin."""

    def __init__(self, clips, skip=None, reuse=False):
        self.clips = [c for c in clips if c.get("moments")]
        self.reuse = reuse  # user 2026-10-21: every clip at most ONCE is a hard rule; reuse=True only for tests / explicit opt-in
        self.recent = []  # media of the last slices, newest last
        self.frame = {}  # {cx, cy} of the last take()
        self.used = {}
        for c in self.clips:
            m = np.array([x["motion"] for x in c["moments"]])
            c["_raw"] = m
            c["_motion"] = np.convolve(m, [0.25, 0.5, 0.25], "same") if len(m) > 2 else m
            c["_loud"] = np.array([x["loud"] for x in c["moments"]])
            self.used[c["media"]] = np.zeros(len(m), bool)
            if c.get("star") is not None:  # 4 fps arrays: detected, face height, centre, touching the frame edge
                n4 = int(np.ceil(len(m) / FPS * FACE_FPS)) + 2
                det, fh, fx, fy, edge = np.zeros(n4, bool), np.zeros(n4), np.zeros(n4), np.zeros(n4), np.zeros(n4, bool)
                for t, x, y, w, h in c["star"]:
                    k = int(round(t * FACE_FPS))
                    if k < n4:
                        det[k], fh[k], fx[k], fy[k] = True, h, x + w / 2, y + h / 2
                        edge[k] = x <= 0.01 or y <= 0.01 or x + w >= 0.99 or y + h >= 0.99
                c["_face"] = (det, fh, fx, fy, edge)
        if skip:  # the hook used the start of clip 0
            self.mark(clips[0]["media"], 0, skip)

    @property
    def prev(self):
        return self.recent[-1] if self.recent else None

    def mark(self, media, s, e):
        u = self.used.get(media)
        if u is not None:
            u[int(s * FPS):int(np.ceil(e * FPS))] = True

    def _face_terms(self, c, starts, L):
        """(valid, size 0..1, edge fraction, cx, cy) per start from the 4 fps star track; valid = he is in >= 90 % of the samples
        incl. first/last and one sample (0.25 s) of margin on both sides, face >= MIN_FACE."""
        det, fh, fx, fy, edge = c["_face"]
        cs = lambda a: np.concatenate([[0], np.cumsum(a)])
        j0 = starts // (FPS // FACE_FPS)  # sample at/before the start
        j1 = np.minimum(-(-(starts + L) // (FPS // FACE_FPS)), len(det) - 1)  # sample at/after the end
        n = j1 - j0 + 1
        hits = cs(det)[j1 + 1] - cs(det)[j0]
        ends = det[j0] & det[j1] & det[np.maximum(j0 - 1, 0)] & det[np.minimum(j1 + 1, len(det) - 1)]
        size = (cs(fh)[j1 + 1] - cs(fh)[j0]) / np.maximum(hits, 1)
        valid = ends & (hits >= 0.9 * n) & (size >= MIN_FACE)
        edge_f = (cs(edge)[j1 + 1] - cs(edge)[j0]) / np.maximum(hits, 1)
        cx = (cs(fx)[j1 + 1] - cs(fx)[j0]) / np.maximum(hits, 1)
        cy = (cs(fy)[j1 + 1] - cs(fy)[j0]) / np.maximum(hits, 1)
        return valid, np.clip((size - MIN_FACE) / (GOOD_FACE - MIN_FACE), 0, 1), edge_f, cx, cy

    def take(self, src_len, beats=(0.0,), hook=False, accent=False, media=None, avoid=()):
        L = max(1, int(round(src_len * FPS)))
        kern = np.zeros(L)
        for o in beats:  # motion peak ~60 ms after the beat = the gesture is seen landing on the hit
            i = np.arange(L); kern += np.exp(-0.5 * ((i / FPS - o - 0.06) / 0.12) ** 2)
        kern /= kern.sum() or 1
        best, pick = -1e9, None
        # user 2026-10-20/21: every clip at most ONCE per edit, hard. Passes: fresh clips with the face gate, then fresh without it;
        # a re-used clip only with reuse=True (the planner caps the shot count at the clip count instead, see trends.plan)
        for strict, fresh in ((True, True), (False, True)) + (((True, False), (False, False)) if self.reuse or media else ()):
            for c in self.clips:
                if media and c["media"] != media:
                    continue
                if fresh and not hook and c["media"] in self.recent:
                    continue
                mo, lo, u = c["_motion"], c["_loud"], self.used[c["media"]]
                if len(mo) < L:
                    continue
                ranges = c.get("ranges") or [(0.0, c["dur"])]
                starts = [i for a, b in ranges for i in range(int(np.ceil(a * FPS)), min(int(b * FPS), len(mo)) - L + 1)]  # ranges are padded past the end
                if not starts:
                    continue
                starts = np.array(starts)
                ok = np.ones(len(starts), bool)
                for cut in c.get("cuts") or []:  # never span an internal cut (+-1 sample of slack)
                    ok &= ~((starts / FPS < cut + 0.1) & ((starts + L) / FPS > cut - 0.1))
                for m_, a, b in avoid:  # windows the review rejected
                    if m_ == c["media"]:
                        ok &= ~((starts / FPS < b + 0.25) & ((starts + L) / FPS > a - 0.25))
                face = c.get("_face") is not None and strict
                if face:
                    valid, size, edge_f, cx, cy = self._face_terms(c, starts, L)
                    ok &= valid
                if not ok.any():
                    continue
                starts = starts[ok]
                win = starts[:, None] + np.arange(L)[None, :]
                align = (mo[win] * kern).sum(1)
                mean_m = mo[win].mean(1)
                loud = lo[win].mean(1)
                uwide = np.convolve(u, np.ones(FPS), "same") > 0  # +-0.5 s around used footage
                reuse = u[win].mean(1) + 0.3 * uwide[win].mean(1)
                score = (0.6 if hook else 1.0) * (1.5 if accent else 1.0) * align + 0.3 * mean_m + (1.5 if hook else 0.1) * loud - 1.2 * reuse
                score = score - 5.0 * (u[win].mean(1) > 0.5)  # footage already shown is out, whatever the variety penalties say
                if face:
                    score = score + 0.5 * size[ok] - 0.5 * edge_f[ok]
                raw = c["_raw"]
                onset = np.clip((mo[starts] - mo[np.maximum(starts - 2, 0)]) / 0.3, 0, 1)  # motion rising into the cut = clean start
                nb = np.maximum(raw[np.maximum(starts - 1, 0)], raw[np.minimum(starts + 1, len(raw) - 1)])
                blur = (raw[starts] >= 0.9) & (raw[starts] >= nb)  # local max of very high frame diff = whip/blur
                score = score + 0.25 * onset - 0.4 * blur
                if not hook:
                    if c["media"] == self.prev:
                        score = score - 0.6  # back-to-back same clip reads as a jump cut
                    elif c["media"] in self.recent[-3:]:
                        score = score - 0.5
                    score = score - 0.35 * self.recent.count(c["media"])  # variety: every earlier use of this clip costs (user 2026-10-19: more different clips)
                if hook:
                    score = score + 0.3 * (starts == 0)  # ponytail: hook_len was measured from t=0 (sentence end); keep it there if we can
                j = int(np.argmax(score))
                if score[j] > best:
                    fr = {"cx": round(float(cx[ok][j]), 3), "cy": round(float(cy[ok][j]), 3)} if face else {}
                    best, pick = score[j], (c["media"], starts[j] / FPS, fr)
            if pick is not None:
                break
        if pick is None:
            left = [c["media"] for c in self.clips if c["media"] not in self.recent]
            if not left and not self.reuse:
                raise ValueError(f"every clip once: all {len(self.clips)} clips are used - download more clips or cut every 2nd beat")
            raise ValueError("no usable footage long enough for one slice; use --every 1 or more clips")
        m, s, self.frame = pick
        self.mark(m, s, s + src_len); self.recent.append(m)
        return m, round(float(s), 3), round(float(s + src_len), 3)


def selftest(paths):
    import clipcut
    # synthetic: face gate, scene cuts, cx from the track, avoid
    mo = [{"t": i / 8, "motion": 0.5, "loud": 0.1} for i in range(80)]  # 10 s
    star = [[k / 4, 0.6, 0.2, 0.1, 0.2] for k in range(8, 24)] + [[k / 4, 0.1, 0.2, 0.02, 0.04] for k in range(28, 40)]  # him 2-6 s big, 7-10 s tiny
    C = [{"media": "a.mp4", "dur": 10, "ranges": [(0, 10)], "moments": mo, "star": star, "cuts": [4.0]}]
    pk = Picker(C, reuse=True)
    for _ in range(3):
        m, s, e = pk.take(0.75)
        assert 2.25 <= s and e <= 5.75 and not (s < 4.1 and e > 3.9), (s, e)  # inside 2-6 s minus margin, never across the cut at 4 s
        assert pk.frame == {"cx": 0.65, "cy": 0.3}, pk.frame
    m, s, e = Picker(C).take(0.75, avoid=[("a.mp4", 2.0, 4.0)])
    assert s >= 4.1, s
    pk = Picker(C, reuse=True)
    ws = [pk.take(0.75) for _ in range(8)]  # more slices than face-valid windows: the lenient pass still returns usable footage
    assert all(2.25 <= s and e <= 5.75 for _, s, e in ws[:3]) and all(0 <= s and e <= 10 for _, s, e in ws), ws
    pk = Picker(C); pk.take(0.75)  # every clip once (2026-10-21): the second slice from the only clip is refused ...
    try:
        pk.take(0.75); assert False, "re-used a clip"
    except ValueError as e:
        assert "every clip once" in str(e), e
    assert pk.take(0.75, media="a.mp4")[0] == "a.mp4"  # ... unless that clip is asked for explicitly (review re-pick of a weak shot)
    B = [{"media": "x.mp4", "dur": 10, "moments": mo}, {"media": "y.mp4", "dur": 10, "moments": mo}]
    pk = Picker(B, reuse=True); seq = [pk.take(0.5)[0] for _ in range(6)]
    assert not any(a == b for a, b in zip(seq, seq[1:])), seq  # back-to-back penalty alternates the clips
    # real footage
    clips = []
    for p in paths:
        c = {"media": Path(p).name, "dur": clipcut.probe(p)["dur"], "moments": moments(p), "cuts": clipcut.scene_cuts(p)}
        r = clipcut.star_ranges(p)
        if r:
            c["ranges"], c["star"] = r, clipcut.star_track(p)
        clips.append(c)
        print(f"{c['media']}: {len(c['moments'])} samples, ranges={c.get('ranges')}, cuts={c['cuts']}, motion peak at "
              f"{max(c['moments'], key=lambda x: x['motion'])['t']}s, loudest at {max(c['moments'], key=lambda x: x['loud'])['t']}s")
    pk = Picker(clips, reuse=True)
    hook = pk.take(2.0, hook=True, media=clips[0]["media"])
    wins = [hook] + [pk.take(0.65, accent=k % 4 == 0) for k in range(12)]
    for m, s, e in wins:
        c = next(c for c in clips if c["media"] == m)
        assert any(a - 1e-6 <= s and e <= b + 1e-6 for a, b in (c.get("ranges") or [(0, c["dur"])])), (m, s, e, c.get("ranges"))
        assert not any(s < cut < e for cut in c["cuts"]), (m, s, e, c["cuts"])
    assert len(set(wins)) > 1 and len({m for m, _, _ in wins}) == len(clips), wins
    print("hook:", hook, "frame:", pk.frame, "\nslices:", [(m[:12], s) for m, s, _ in wins[1:]])
    print("motion selftest ok")


if __name__ == "__main__":
    selftest(sys.argv[1:] or sorted(str(p) for p in Path(__file__).parent.glob("media/*zockt*.mp4"))
             + sorted(str(p) for p in Path(__file__).parent.glob("media/*slidet*.mp4")))
