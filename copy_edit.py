"""copy_edit.py - paste a TikTok edit link (or any mp4): the same cuts, flashes, zoom punches, slow-mo and sound, with your footage.

reference(url_or_path) -> {"media": "ref_<id>.m4a", "dur", "look", "events": {...}}   downloads (yt-dlp), extracts the sound into media/, fingerprints
plan(ref, clips, text, even) -> [(message, project)]                                 same shape as trends.plan, pure, so `python copy_edit.py` self-checks it
quantize(cuts, beats)  -> (cuts, note)   `even`: snap cuts to the reference's beat/half-beat grid (<= 90 ms), merge shots < 0.2 s
"""
import copy, shutil, subprocess, sys
from pathlib import Path

from trends import _id, _seg, Footage, CANVAS

ROOT = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).parent  # exe: media/ next to it
MEDIA = ROOT / "media"


def reference(src, log=print):
    import clipcut, tiktok_trends
    src = str(src)
    if src.startswith("http"):
        m = tiktok_trends.VIDEO_URL.match(src)
        if not m:
            raise ValueError("not a TikTok video URL: " + src)
        paths = tiktok_trends.download([{"url": m.group(0), "views": 0}], 1, log=log)
        if not paths:
            raise ValueError("download failed: " + src)
        src = str(paths[0])
    stem = Path(src).stem
    fp = tiktok_trends.fingerprint(src, log)
    sound = MEDIA / f"ref_{stem}.m4a"
    if not sound.exists():  # the reference's own soundtrack becomes the song track
        tmp = sound.with_suffix(".tmp.m4a")
        subprocess.run([clipcut.FF, "-y", "-v", "error", "-i", src, "-vn", "-map", "0:a:0", "-c:a", "aac", "-b:a", "192k", str(tmp)], check=True,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        shutil.move(tmp, sound)
    return {"media": sound.name, "dur": fp["dur"], "look": fp["look"], "source": Path(src).name, "events": fp["events"]}


def quantize(cuts, beats, tol=0.09, min_shot=0.2, bpm=None):
    """Even lengths: every cut (not first/last) snaps to the nearest beat or half-beat within `tol`; shots shorter than
    `min_shot` merge into a neighbour (the off-grid cut goes). Returns (cuts, note for the step message).
    With `bpm` the grid is regular (period 60/bpm, phase fitted to the detected beats): librosa's beat times jitter by a
    hop (~23 ms) and snapping to them would make even shots uneven."""
    import math
    if bpm and beats:
        T = 60 / bpm
        ph = T * math.atan2(sum(math.sin(2 * math.pi * b / T) for b in beats), sum(math.cos(2 * math.pi * b / T) for b in beats)) / (2 * math.pi)
        beats = [ph + k * T for k in range(-1, int((max(beats) + T) / T) + 2) if ph + k * T >= 0]
    grids = [sorted(beats), sorted((a + b) / 2 for a, b in zip(beats, beats[1:]))]  # full beats first
    if not grids[0]:
        return cuts, ", no beat grid in the reference so the cuts stay"
    on, snapped, n = {cuts[0], cuts[-1]}, [cuts[0]], 0
    for c in cuts[1:-1]:
        for grid in grids:
            g = round(min(grid, key=lambda x: abs(x - c)), 3) if grid else None
            if g is not None and abs(g - c) <= tol:
                n += g != c; c = g; on.add(g); break
        snapped.append(round(c, 3))
    snapped.append(cuts[-1])
    out, merged = [snapped[0]], 0
    for c in snapped[1:]:
        p = out[-1]
        if c - p >= min_shot or (len(out) == 1 and c == snapped[-1]):
            out.append(c)
        else:
            merged += 1
            if len(out) == 1 or (p in on and c not in on and c != snapped[-1]):
                continue  # drop this cut: the short shot joins the next one
            out[-1] = c  # drop the previous cut: it joins the previous one
    return out, f", even: {len(cuts) - 2} cuts, {n} snapped to the beat" + (f", {merged} shots < {min_shot} s merged" if merged else "")


def plan(ref, clips, text=None, even=False):
    E, dur = ref["events"], ref["dur"]
    cuts = [0.0, *[c for c in E["cuts"] if 0.15 < c < dur - 0.15], dur]
    cuts = [c for i, c in enumerate(cuts) if i == 0 or c - cuts[i - 1] > 0.12]  # ponytail: merge sub-3-frame doubles
    cuts, note = quantize(cuts, E.get("beats", []), bpm=E.get("bpm")) if even else (cuts, "")
    if any("ranges" in c for c in clips):
        clips = [c for c in clips if c.get("ranges")] or clips
    if any(c.get("moments") for c in clips):
        from motion import Picker
        foot = Picker(clips)
    else:
        foot = Footage(clips)
    avail = len({c["media"] for c in (foot.clips if hasattr(foot, "clips") else clips)})
    if len(cuts) - 1 > avail:  # user 2026-10-21: every clip once, hard -> the copy stops where the clips run out (sound ends with it)
        cuts, note = cuts[:avail + 1], note + f"; every clip once: only {avail} clips, stopping after {avail} of the reference's {len(cuts) - 1} shots"
        dur = cuts[-1]
    steps, P = [], {"canvas": CANVAS, "fps": 30, "fill": "cover", "filter": ref["look"], "video": [], "overlay": [], "audio": [], "text": [],
                    "beats": E.get("beats", []), "strong": E.get("strong", []), "bpm": E.get("bpm", 0)}
    step = lambda msg: steps.append((msg, copy.deepcopy(P)))
    step(f"Copying '{ref.get('source', ref['media'])}': {len(cuts) - 1} shots, {len(E['flashes'])} flashes, {len(E['zooms'])} zoom punches, look '{ref['look']}'" + note)
    P["audio"].append({"id": _id(), "media": ref["media"], "start": 0, "in": 0, "out": round(dur, 3), "speed": 1, "volume": 1})
    step("Sound: the reference's own soundtrack")
    near = lambda ts, t, w: any(abs(x - t) < w for x in ts)
    acc = nsun = neyes = 0
    for a, b in zip(cuts, cuts[1:]):
        slow = sum(min(b, y) - max(a, x) for x, y in E["slow"] if y > a and x < b) > 0.5 * (b - a)
        sp = 0.5 if slow else 1.0
        accent = any(a - 0.05 <= x < b - 0.05 for x in E.get("strong", []))  # the shot in which the hit lands (cuts often sit a few frames before the beat)
        m, s, e = foot.take((b - a) * sp, beats=[(x - a) * sp for x in E.get("beats", []) if a <= x < b], accent=accent)
        seg = _seg(m, s, e, speed=sp, **foot.frame)  # cx/cy: cover crop centred on his face (motion.Picker)
        if near(E["zooms"], a, 0.2):
            seg["zoom"], seg["shake"] = 1.3, 0.3
        if accent:
            acc += 1
        P["video"].append(seg)
    step(f"{len(P['video'])} shots cut at the reference's exact cut times" + (", slow-mo where it is slow" if any(s["speed"] < 1 for s in P["video"]) else ""))
    # flashes: a fadewhite INTO the next shot overlaps it and shortens the timeline -> lengthen the outgoing shot by the same amount
    fl = 0
    for seg, b in zip(P["video"], cuts[1:-1]):
        if near(E["flashes"], b, 0.15):
            seg["transition"] = {"type": "fadewhite", "dur": 0.08}
            seg["out"] = round(seg["out"] + 0.08 * seg["speed"], 3); fl += 1
    if fl:
        step(f"White flash on {fl} of the cuts, like the original")
    if any("zoom" in s for s in P["video"]):
        step(f"Zoom punch + shake on {sum('zoom' in s for s in P['video'])} shots")
    if text:
        words = text.split()
        pops = [b for b in E.get("strong", []) if b >= 0.5][:len(words)] or [0.5]
        beat = 60 / E["bpm"] if E.get("bpm") else 0.5
        for k, (w, b) in enumerate(zip(words, pops)):
            nxt = pops[k + 1] if k + 1 < len(pops) else b + beat * 2
            P["text"].append({"id": _id(), "text": w, "start": b, "dur": round(min(beat * 2, nxt - b), 3), "x": 0.5, "y": 0.22, "size": 0.09,
                              "color": "#ffffff", "font": "impact", "box": False})
        step("Text pops on the accents: " + " / ".join(words[:len(pops)]))
    steps.append(("Done - press play", copy.deepcopy(P)))
    return steps


def selftest():
    ref = {"media": "ref_x.m4a", "dur": 6.0, "look": "phonk", "source": "x.mp4",
           "events": {"cuts": [1.0, 2.0, 2.05, 3.0, 4.0, 5.0], "flashes": [2.0, 4.0], "zooms": [3.02], "slow": [[4.0, 4.5], [4.5, 5.0]],
                      "beats": [i * 0.5 for i in range(12)], "strong": [1.0, 3.0], "bpm": 120}}
    clips = [{"media": "a.mp4", "dur": 20, "ranges": [(0, 20)]}, {"media": "b.mp4", "dur": 9}] + [{"media": f"c{k}.mp4", "dur": 9, "ranges": [(0, 9)]} for k in range(5)]
    steps = plan(ref, clips, text="UNSER EUROPA")
    P = steps[-1][1]
    V = P["video"]
    assert len(V) == 6, [(s["in"], s["out"]) for s in V]  # 2.05 merged into 2.0
    assert P["audio"][0]["media"] == "ref_x.m4a" and P["audio"][0]["out"] == 6.0 and P["filter"] == "phonk"
    assert V[4]["speed"] == 0.5 and all(s["speed"] == 1 for s in V[:4]), [s["speed"] for s in V]  # 4-5 s is slow
    assert V[3]["zoom"] == 1.3 and "zoom" not in V[2], V[3]
    assert V[1]["transition"]["type"] == "fadewhite" and abs(V[1]["out"] - V[1]["in"] - 1.08) < 1e-6, V[1]  # flash at 2.0 -> shot 1-2 flashes into shot 2
    assert V[0]["transition"]["type"] == "none"
    assert not any(s.get("eyes") or s.get("sun") for s in V), V  # school version: no face effects on the accents
    tl = sum((s["out"] - s["in"]) / s["speed"] for s in V) - 0.08 * sum(s["transition"]["type"] == "fadewhite" for s in V)
    assert abs(tl - 6.0) < 1e-6, tl  # flashes do not drift the cut times
    assert [t["text"] for t in P["text"]] == ["UNSER", "EUROPA"] and P["text"][0]["start"] == 1.0
    # even: snap to beat / half-beat within 90 ms, merge < 0.2 s shots (off-grid cut goes), first/last untouched
    c, note = quantize([0.0, 1.06, 2.0, 2.15, 3.3, 3.42, 5.0, 5.08, 6.0], [i * 0.5 for i in range(13)])
    assert c == [0.0, 1.0, 2.0, 3.25, 3.5, 5.0, 6.0], (c, note)  # 2.15 joins 2.0's shot; 3.3/3.42 -> 3.25/3.5; 5.08 -> 5.0 = double, dropped
    assert "4 snapped" in note and "2 shots" in note, note
    assert quantize([0.0, 1.0, 2.0], [])[0] == [0.0, 1.0, 2.0]
    jit = [0.0, 0.52, 0.98, 1.52, 1.98, 2.52, 2.98]  # detector jitter around a 120 BPM grid: shots come out 0.5 s each, not 0.46/0.54
    assert quantize([0.0, 0.47, 1.02, 1.49, 2.03, 3.0], jit, bpm=120)[0] == [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    assert quantize([0.0, 0.47, 1.02, 1.49, 2.03, 3.0], jit)[0] == [0.0, 0.52, 0.98, 1.52, 1.98, 3.0]
    S = plan(ref, clips, even=True)
    assert "even:" in S[0][0] and len(S[-1][1]["video"]) == 6, S[0][0]
    # cx/cy from the Picker land on the segments
    import math
    mo = [{"t": i / 8, "motion": (1 + math.sin(i)) / 2, "loud": 0.1} for i in range(160)]
    C = [{"media": "a.mp4", "dur": 20, "ranges": [(0, 20)], "moments": mo, "star": [[k / 4, 0.6, 0.2, 0.1, 0.2] for k in range(81)], "cuts": []}]
    S = plan(ref, C)
    V = S[-1][1]["video"]
    assert all((s["cx"], s["cy"]) == (0.65, 0.3) for s in V), V[0]
    assert len(V) == 1 and "every clip once" in S[0][0] and S[-1][1]["audio"][0]["out"] == 1.0, (S[0][0], V)  # one clip -> one shot, sound cut with it
    print("copy_edit selftest ok")


if __name__ == "__main__":
    selftest()
