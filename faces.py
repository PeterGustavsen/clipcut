"""faces.py - is the right person on screen? YuNet detector + SFace embeddings (OpenCV, CPU, ~5 ms/frame).

scan(video, step)        -> [{"t": s, "faces": [{"box": [x,y,w,h], "emb": [128 floats]}]}]   (sampled frames)
reference(scans)         -> embedding of the person who shows up most (the main person of the project)
visible(scan, ref, thr)  -> merged [(a, b)] seconds where that person is on screen
track(scan, ref, size)   -> [[t, x, y, w, h]] of the star's box per sampled frame as fractions of the source frame (motion.Picker, review.py)
contact(video, n, out)   -> contact-sheet PNG so an agent can look at a clip
"""
import json, sys
from pathlib import Path

import cv2, numpy as np


def imwrite(path, img):  # cv2.imwrite cannot handle non-ASCII paths on Windows
    ok, buf = cv2.imencode('.png', img)
    Path(path).write_bytes(buf.tobytes())

MODELS = Path(getattr(sys, "_MEIPASS", Path(__file__).parent)) / "models"
THR = 0.37  # SFace cosine; opencv_zoo recommends 0.363, slightly stricter keeps strangers out
_det = _rec = None


def _nets(w, h):
    global _det, _rec
    if _det is None:
        _det = cv2.FaceDetectorYN.create(str(MODELS / "yunet.onnx"), "", (w, h), 0.7, 0.3, 5000)
        _rec = cv2.FaceRecognizerSF.create(str(MODELS / "sface.onnx"), "")
    _det.setInputSize((w, h))
    return _det, _rec


def frames(video, step=0.25, max_w=640):
    cap = cv2.VideoCapture(str(video))
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 30)
    t = 0.0
    while t < dur:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, img = cap.read()
        if not ok:
            break
        if img.shape[1] > max_w:
            img = cv2.resize(img, (max_w, round(img.shape[0] * max_w / img.shape[1])))
        yield round(t, 3), img
        t += step
    cap.release()


def scan(video, step=0.25):
    out = []
    for t, img in frames(video, step):
        det, rec = _nets(img.shape[1], img.shape[0])
        _, found = det.detect(img)
        faces = []
        for row in (found if found is not None else []):
            emb = rec.feature(rec.alignCrop(img, row)).flatten()
            faces.append({"box": [int(v) for v in row[:4]], "emb": [round(float(v), 5) for v in emb]})
        out.append({"t": t, "faces": faces})
    return out


def _cos(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def reference(scans, thr=THR):
    """Greedy clustering of every face seen; the biggest cluster is the star. Returns (embedding, members) ."""
    embs = [(k, f["emb"], s["t"], f["box"]) for k, sc in enumerate(scans) for s in sc for f in s["faces"]]
    clusters = []  # [centroid, [members]]
    for k, e, t, box in embs:
        for c in clusters:
            if _cos(c[0], e) > thr:
                c[1].append((k, t, box, e)); c[0] = np.mean([m[3] for m in c[1]], axis=0); break
        else:
            clusters.append([np.asarray(e, dtype=float), [(k, t, box, e)]])
    if not clusters:
        raise ValueError("no faces found in any clip")
    best = max(clusters, key=lambda c: len({m[0] for m in c[1]}) * 1000 + len(c[1]))  # most clips, then most frames
    return [round(float(v), 5) for v in best[0]], best[1]


def visible(sc, ref, thr=THR, step=0.25, min_len=0.5, pad=0.1):
    hits = [s["t"] for s in sc if s["faces"] and max(_cos(ref, f["emb"]) for f in s["faces"]) > thr]
    ranges = []
    for t in hits:
        if ranges and t - ranges[-1][1] <= step * 1.5:
            ranges[-1][1] = t
        else:
            ranges.append([t, t])
    return [(round(max(0, a - pad), 2), round(b + step + pad, 2)) for a, b in ranges if b + step - a >= min_len]


def track(sc, ref, size, thr=THR):
    """Star box per sample where he matches, as source-frame fractions. size = (width, height) of the video (scan ran <=640 px wide)."""
    sw = min(size[0], 640); sh = size[1] * sw / size[0]
    out = []
    for s in sc:
        hits = [f for f in s["faces"] if _cos(ref, f["emb"]) > thr]
        if hits:
            x, y, w, h = max(hits, key=lambda f: f["box"][2] * f["box"][3])["box"]
            out.append([s["t"], round(x / sw, 4), round(y / sh, 4), round(w / sw, 4), round(h / sh, 4)])
    return out


def contact(video, n=12, out=None, marks=None):
    """n frames side by side (3 rows) with timestamps; `marks` = ranges to tint green (person visible)."""
    cap = cv2.VideoCapture(str(video))
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 30)
    tiles = []
    for i in range(n):
        t = dur * (i + 0.5) / n
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, img = cap.read()
        if not ok:
            continue
        img = cv2.resize(img, (240, round(img.shape[0] * 240 / img.shape[1])))
        if marks is not None:
            on = any(a <= t <= b for a, b in marks)
            cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), (0, 200, 0) if on else (0, 0, 220), 6)
        cv2.putText(img, f"{t:.1f}s", (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        tiles.append(img)
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    cols = 4
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    w = max(r.shape[1] for r in rows)
    sheet = np.vstack([cv2.copyMakeBorder(r, 0, 0, 0, w - r.shape[1], cv2.BORDER_CONSTANT) for r in rows])
    out = out or str(Path(video).with_suffix(".contact.png"))
    imwrite(out, sheet)
    return out


def crops(scans, members, videos, out, n=12):
    """Face crops of cluster members (one per clip first) so a human/agent can confirm who the star is."""
    seen, tiles = set(), []
    for k, t, box, _ in sorted(members, key=lambda m: (m[0] in seen, m[0])):
        if len(tiles) >= n:
            break
        if k in seen and len(seen) < len(videos):
            continue
        seen.add(k)
        cap = cv2.VideoCapture(str(videos[k])); cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000); ok, img = cap.read(); cap.release()
        if not ok:
            continue
        sc = img.shape[1] / 640 if img.shape[1] > 640 else 1
        x, y, w, h = [int(v * sc) for v in box]
        m = int(0.3 * h)
        crop = img[max(0, y - m):y + h + m, max(0, x - m):x + w + m]
        if crop.size:
            tiles.append(cv2.resize(crop, (160, 160)))
    sheet = np.hstack(tiles) if tiles else np.zeros((160, 160, 3), np.uint8)
    imwrite(str(out), sheet)
    return str(out)


if __name__ == "__main__":  # python faces.py CLIP...  -> who is the star, contact sheet of their face
    vids = sys.argv[1:]
    scans = [scan(v) for v in vids]
    ref, members = reference(scans)
    print(json.dumps({"clips_with_star": len({m[0] for m in members}), "faces": len(members)}))
    print(crops(scans, members, vids, Path(vids[0]).parent / "_star_faces.png"))
