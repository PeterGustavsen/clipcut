"""review.py - look at the planned shots the way the Reel will show them, and re-pick the ones where the star is not clearly framed.

review(project, media_dir, root, out) -> {"ok", "failed", "slices": [{"i", "ok", "reason", "media", "in"}], "sheet"}
    first / middle / last frame of every main-track shot at the rendered crop (cover crop around cx/cy + zoom punch, see
    clipcut.fit_chain), star detector on each, face box drawn, tiled into one contact sheet; red border = no face / not the
    star / face < 7 % of the frame / face cut by the frame edge.
repair(project, infos, media_dir, root) -> (project, message, result)
    review, then ask motion.Picker for the next-best window for every failed shot (the rejected windows are avoided),
    up to 3 rounds. Message: "Review: N of M shots re-picked (reasons)".
python review.py [project.json]  -> JSON + sheet path (%TEMP%/review_<name>.png)
"""
import copy, json, sys, tempfile
from collections import Counter
from pathlib import Path

import cv2, numpy as np

import faces
from motion import MIN_FACE

ROOT = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).parent
DET_H, TILE_H = 640, 220  # detector runs on a 640 px tall canvas frame; sheet tiles are 220 px tall


def framed(img, W, H, seg, fill="cover", z=1.0):
    """One source frame the way the renderer shows it: cover crop around cx/cy (or contain), then the zoom punch around his face."""
    h, w = img.shape[:2]
    cx = cy = 0.5; x = y = 0
    if fill == "cover":
        k = max(W / w, H / h)
        img = cv2.resize(img, (max(W, round(w * k)), max(H, round(h * k))))
        h, w = img.shape[:2]
        cx, cy = (min(1, max(0, float(seg.get(k) or 0.5))) for k in ("cx", "cy"))
        x, y = round(min(w - W, max(0, w * cx - W / 2))), round(min(h - H, max(0, h * cy - H / 2)))  # crop centred on (cx, cy), clamped
        img = img[y:y + H, x:x + W]
    else:
        k = min(W / w, H / h)
        img = cv2.resize(img, (max(2, round(w * k)), max(2, round(h * k))))
        h, w = img.shape[:2]
        img = cv2.copyMakeBorder(img, (H - h) // 2, H - h - (H - h) // 2, (W - w) // 2, W - w - (W - w) // 2, cv2.BORDER_CONSTANT)
    if z and abs(z - 1) > 1e-3:  # zoom punch around his face (= clipcut.fx with face_on_canvas), clamped to the frame
        zw, zh = round(W / z), round(H / z)
        fx, fy = (0.5, 0.5) if fill != "cover" else ((w * cx - x) / W, (h * cy - y) / H)
        x0, y0 = round(min(W - zw, max(0, W * fx - zw / 2))), round(min(H - zh, max(0, H * fy - zh / 2)))
        img = cv2.resize(img[y0:y0 + zh, x0:x0 + zw], (W, H))
    return img


def star_in(img, ref, thr=faces.THR):
    """(box or None, reason '' when fine) for the star in one canvas-sized frame."""
    det, rec = faces._nets(img.shape[1], img.shape[0])
    _, found = det.detect(img)
    best = None
    for row in (found if found is not None else []):
        if ref is None or faces._cos(ref, rec.feature(rec.alignCrop(img, row)).flatten()) > thr:
            if best is None or row[2] * row[3] > best[2] * best[3]:
                best = row[:4]
    if best is None:
        return None, "not the star" if found is not None and len(found) else "no face"
    x, y, w, h = [int(v) for v in best]
    H, W = img.shape[:2]
    if h / H < MIN_FACE:
        return (x, y, w, h), f"face < {MIN_FACE:.0%}"
    if x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1:
        return (x, y, w, h), "cut by the edge"
    return (x, y, w, h), ""


def review(P, media_dir, root=ROOT, out=None, only=None):
    import clipcut
    ref = clipcut.face_ref(root)
    CW, CH = P.get("canvas") or [1080, 1920]
    W, H = round(DET_H * CW / CH) // 2 * 2, DET_H
    fill = P.get("fill", "contain")
    caps, res, groups = {}, [], []
    for i, seg in enumerate(P["video"]):
        if only is not None and i not in only:
            continue
        m = seg["media"]
        if m not in caps:
            caps[m] = cv2.VideoCapture(str(Path(media_dir) / Path(m).name))
        cap = caps[m]
        times = [seg["in"]] if seg.get("freeze") else [seg["in"], (seg["in"] + seg["out"]) / 2, max(seg["in"], seg["out"] - 0.04)]
        z0, z1 = seg.get("zoom") or 1, seg.get("zoomTo") or seg.get("zoom") or 1
        tiles, reasons = [], []
        for k, t in enumerate(times):
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, img = cap.read()
            if not ok:
                reasons.append("no frame"); continue
            img = framed(img, W, H, seg, fill, z1 if k == len(times) - 1 and len(times) > 1 else z0)
            box, why = star_in(img, ref)
            reasons.append(why)
            if box:
                x, y, w, h = box
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 230) if why else (0, 210, 0), 3)
            tiles.append(cv2.resize(img, (round(TILE_H * W / H), TILE_H)))
        bad = [r for r in reasons if r]
        reason = Counter(bad).most_common(1)[0][0] if bad else ""
        res.append({"i": i, "ok": not bad, "reason": reason, "media": m, "in": seg["in"]})
        if out and tiles:
            g = np.hstack([cv2.copyMakeBorder(t, 0, 0, 0, 3, cv2.BORDER_CONSTANT, value=(30, 30, 30)) for t in tiles])
            g = cv2.copyMakeBorder(g, 22, 4, 4, 4, cv2.BORDER_CONSTANT, value=(0, 0, 200) if bad else (0, 150, 0))
            cv2.putText(g, f"#{i} {Path(m).stem[:18]} {seg['in']:.2f}-{seg['out']:.2f} {reason}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            groups.append(g)
    for c in caps.values():
        c.release()
    if out and groups:
        w = max(g.shape[1] for g in groups); h = max(g.shape[0] for g in groups)
        groups = [cv2.copyMakeBorder(g, 0, h - g.shape[0], 0, w - g.shape[1], cv2.BORDER_CONSTANT) for g in groups]
        cols = 4
        rows = [np.hstack(groups[i:i + cols]) for i in range(0, len(groups), cols)]
        sheet = np.vstack([cv2.copyMakeBorder(r, 0, 0, 0, rows[0].shape[1] - r.shape[1], cv2.BORDER_CONSTANT) for r in rows])
        faces.imwrite(out, sheet)
    failed = [r for r in res if not r["ok"]]
    return {"ok": not failed, "failed": len(failed), "slices": res, "sheet": out}


def repair(P, infos, media_dir, root=ROOT, tries=3, out=None):
    from motion import Picker
    P = copy.deepcopy(P)
    out = out or str(Path(tempfile.gettempdir()) / "review_project.png")
    res = review(P, media_dir, root, out)
    before = [x for x in res["slices"] if not x["ok"]]
    reasons = ", ".join(f"{v}x {k}" for k, v in Counter(x["reason"] for x in before).items())
    clips = [c for c in infos if c.get("moments")]
    n = len(res["slices"])
    if not before:
        return P, f"Review: all {n} shots show the star clearly", res
    if not clips:
        return P, f"Review: {len(before)} of {n} shots weak ({reasons}), nothing to re-pick from without moments", res
    tried, changed, failed = {}, set(), before
    for _ in range(tries):
        pk = Picker(copy.deepcopy(clips))
        for s in P["video"]:
            if not s.get("freeze"):
                pk.mark(s["media"], s["in"], s["out"])
        idx = []
        for x in failed:
            i, seg = x["i"], P["video"][x["i"]]
            if seg.get("freeze"):
                continue
            tried.setdefault(i, []).append((seg["media"], seg["in"], seg["out"]))
            # every clip once (user 2026-10-20): all other shots count as used, the real neighbour last for the back-to-back penalty
            pk.recent = [t["media"] for j, t in enumerate(P["video"]) if j != i and not t.get("freeze") and j != i - 1] + ([P["video"][i - 1]["media"]] if i else [])
            hook = bool(seg.get("volume"))  # trends' speech hook: same clip, loudest window
            try:
                m, s_, e_ = pk.take(seg["out"] - seg["in"], hook=hook, media=seg["media"] if hook else None, avoid=tried[i])
            except ValueError:
                try:  # no unused clip left (every clip once): another window of the same clip is still that clip once
                    m, s_, e_ = pk.take(seg["out"] - seg["in"], hook=hook, media=seg["media"], avoid=tried[i])
                except ValueError:
                    continue
            seg.pop("cx", None); seg.pop("cy", None)
            seg.update({"media": m, "in": s_, "out": e_, **pk.frame})
            if i + 1 < len(P["video"]) and P["video"][i + 1].get("freeze"):  # trends' freeze-frame ending follows its shot
                P["video"][i + 1].update({"media": m, "in": e_, "out": e_, **pk.frame})
            idx.append(i); changed.add(i)
        if not idx:
            break
        failed = [x for x in review(P, media_dir, root, None, only=idx)["slices"] if not x["ok"]]
        if not failed:
            break
    res = review(P, media_dir, root, out)
    after = [x for x in res["slices"] if not x["ok"]]
    msg = f"Review: {len(changed)} of {n} shots re-picked ({reasons})" + (f", {len(after)} still weak" if after else ", all good now")
    return P, msg, res


if __name__ == "__main__":
    pj = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "project.json")
    P = json.loads(pj.read_text(encoding="utf-8"))
    r = review(P, pj.parent / "media" if (pj.parent / "media").is_dir() else ROOT / "media", out=str(Path(tempfile.gettempdir()) / f"review_{pj.stem}.png"))
    print(json.dumps(r, ensure_ascii=False))
