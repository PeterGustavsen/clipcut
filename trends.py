"""trends.py - "edit"-genre templates (TikTok / Reels fan edits) for ClipCut.

plan(style, clips, grid, song, ...) -> [(message, project)]  : one snapshot per visible editing step, last = result.
Pure (no ffmpeg), so `python trends.py` self-checks it.

Styles (what people call them on TikTok/CapCut in 2026):
  ironic    speech hook from the first clip (normal speed, own audio, song ducked) -> bass drop -> hard phonk montage
  phonk     cut on every beat, zoom punch alternating, flash + shake on accents, grainy high-contrast look
  velocity  slow-mo (0.5x) for two beats, snap to 2x for one beat, cuts land on beats
  aura      cinematic: one clip per 4 beats at 0.6x with a slow zoom, bw/vignette "aura" look, text on accents
  montage   plain beat sync every 2 beats, cover fill, cool grade
  trending  knobs measured from currently popular TikTok edits (tiktok_trends.py -> media/_trends/trend_template.json)

Clip choice: if clips carry "moments" (motion.moments) slices are picked by motion.Picker (gesture on the beat, no re-use,
loud window for the hook); otherwise Footage round-robin (= `--dumb`).
"""
import copy, json, sys
from pathlib import Path
from uuid import uuid4


def _id():
    return uuid4().hex[:8]

STYLES = ["ironic", "phonk", "velocity", "aura", "montage", "trending"]
CANVAS = [1080, 1920]
TREND_FILE = (Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).parent) / "media" / "_trends" / "trend_template.json"  # exe: media/ next to it


def trend_knobs(template=None):
    """Knobs for style 'trending' (see tiktok_trends.NUM): template dict, else the json tiktok_trends.refresh wrote."""
    if template is None:
        if not TREND_FILE.exists():
            raise FileNotFoundError(f"no trend template yet: python clipcut.py trends refresh  (or drop mp4s into {TREND_FILE.parent})")
        template = json.loads(TREND_FILE.read_text(encoding="utf-8"))
    return template.get("knobs", template)


def _spread(k, pct):
    """Deterministic 'pct of the items': True for item k when the running total crosses an integer."""
    return int(k * pct) != int((k + 1) * pct)


def _seg(media, s, e, **kw):
    d = {"id": _id(), "media": media, "in": round(s, 3), "out": round(e, 3), "speed": 1, "rotation": 0, "flip": False, "volume": 0,
         "transition": {"type": "none", "dur": 0.5}}
    d.update(kw)
    return d


class Footage:
    """Round-robin over usable pieces (clip `ranges` = where the star is on screen, else the whole clip).
    Each piece continues where it stopped so nothing repeats until it must; pieces of different clips alternate."""
    frame = {}  # motion.Picker sets {cx, cy} (crop centre on the face) after every take(); round-robin has no idea where he is

    def __init__(self, clips, skip=None):
        from itertools import zip_longest
        per = [[[c["media"], a, b, a] for a, b in (c.get("ranges") or [(0.0, c["dur"])])] for c in clips]
        self.pieces = [p for group in zip_longest(*per) for p in group if p]
        if skip:  # the hook used the start of clip 0
            for p in self.pieces:
                if p[0] == clips[0]["media"] and p[1] < skip:
                    p[3] = min(p[2], skip)
        self.k = 0

    def take(self, src_len, **_):  # beats/hook/accent/media hints are only used by motion.Picker
        for _ in range(len(self.pieces)):
            p = self.pieces[self.k % len(self.pieces)]; self.k += 1
            media, a, b, cur = p
            if b - a < src_len:
                continue
            if cur + src_len > b:
                cur = a
            p[3] = cur + src_len
            return media, cur, cur + src_len
        raise ValueError("no usable footage long enough for one slice; use --every 1 or more clips")


def _marks(beats, every, start, end):
    m = [b for b in beats[::max(1, every)] if start <= b <= end]
    return m if m and abs(m[0] - start) < 1e-6 else [start, *m]


def plan(style, clips, grid, song, text=None, hook=None, every=None, pace_note=True, template=None, look=None):
    if style == "auto":
        style = "ironic" if clips[0].get("speech", True) else "phonk"
    T = trend_knobs(template) if style == "trending" else None
    hooked = style == "ironic" or (T is not None and T.get("hook_s", 0) >= 1.5)  # trending with a talking first shot
    beats, strong, bpm = grid["beats"], set(grid.get("strong", [])), grid.get("bpm", 120)
    drop = grid.get("drop") or (sorted(strong)[0] if strong else beats[min(8, len(beats) - 1)])
    steps, P = [], {"canvas": CANVAS, "fps": 30, "fill": "cover", "filter": "none", "video": [], "overlay": [], "audio": [], "text": [],
                    "beats": [], "strong": [], "bpm": bpm}

    def step(msg):
        steps.append((msg, copy.deepcopy(P)))

    lim = {c["media"]: c.get("ranges") or [(0.0, c["dur"])] for c in clips}

    def extend(s, t):  # a transition INTO the next shot overlaps it and shortens the timeline: lengthen the outgoing shot so the cut stays on the beat
        old = s["out"]
        end = next((hi for lo, hi in lim.get(s["media"], []) if lo - 1e-6 <= s["in"] and old <= hi + 1e-6), old + t * s["speed"])
        s["out"] = round(min(old + t * s["speed"], end), 3)  # never past the clip or past where his face leaves the frame
        got = round((s["out"] - old) / s["speed"], 3)
        if got < t - 0.0015:  # no room: shorter transition, or none below two frames (3-decimal rounding of out must not kill a 4-frame one)
            s["transition"] = {"type": s["transition"]["type"], "dur": got} if got >= 0.065 else {"type": "none", "dur": 0.5}
        return got

    def extend_in(s, t):  # the incoming half of a transition: lengthen the shot without touching its own transition
        old = s["out"]
        end = next((hi for lo, hi in lim.get(s["media"], []) if lo - 1e-6 <= s["in"] and old <= hi + 1e-6), old + t * s["speed"])
        s["out"] = round(min(old + t * s["speed"], end), 3)  # ponytail: no room = up to dur/2 drift on this one cut, rare

    look = look or {"ironic": "phonk", "phonk": "phonk", "velocity": "none", "aura": "aura", "montage": "cool", "trending": T and T.get("look", "phonk")}[style]
    P["filter"] = look
    step(f"Style '{style}': 9:16 canvas, cover fill, look '{look}', {bpm:.0f} BPM" +
         (f" (trend: cut every {T['cut_every_beats']} beat(s), {T['zoom_punch_pct']:.0%} zoom punches, {T['flash_on_accents_pct']:.0%} flashes)" if T else ""))
    if any("ranges" in c for c in clips):
        usable = [c for c in clips if c.get("ranges")]
        secs = sum(b - a for c in usable for a, b in c["ranges"])
        clips = usable or clips
        step(f"Face check: star on screen in {len(usable)} of {len(clips) if usable else len(clips)} clips, {secs:.0f}s usable footage")

    smart = any(c.get("moments") for c in clips)
    if smart:
        from motion import Picker
        foot = Picker(clips)
        step("Moments: slices land on gestures/head movement that hit the beat, no footage re-used, loudest window for the hook")
    # --- hook: first clip talks, song sneaks in under it, drop lands exactly when the hook ends
    hook_len, hook_in = 0.0, 0.0
    if hooked:
        c0 = clips[0]
        hook_len = hook if hook else min(3.5, max(1.5, round((T or {}).get("hook_s") or c0["dur"] * 0.5, 2)))
        hook_len = min(hook_len, c0["dur"])
        if smart:
            try:  # loudest visible window of clip 0 (keeps t=0 when it is loud enough: the hook length was measured from there)
                _, hook_in, _ = foot.take(hook_len, hook=True, media=c0["media"])
            except ValueError:  # no gated window of that length: keep t=0, but the hook clip still counts as used (every clip once;
                foot.recent.append(c0["media"]); foot.mark(c0["media"], 0.0, hook_len)  # 2026-10-21: it came back as montage shot 13)
        P["video"].append(_seg(c0["media"], hook_in, hook_in + hook_len, volume=1, **(foot.frame if smart else {})))
        song_in = max(0.0, drop - hook_len)
        P["audio"].append({"id": _id(), "media": song, "start": 0, "in": round(song_in, 3), "out": round(drop, 3), "speed": 1, "volume": 0.25, "fadeIn": 0.5})
        step(f"Hook: {hook_len:.1f}s of '{c0['media']}' from {hook_in:.1f}s with its own audio, song ducked underneath")
        # the drop (= the kick itself, snapped off the grid) is the first cut; grid beats within 0.25 s of it would make a 2-frame shot
        tl_beats = [round(hook_len, 3)] + [round(b - drop + hook_len, 3) for b in beats if b >= drop + 0.25]
        P["beats"], P["strong"] = tl_beats, [round(hook_len, 3)] + [round(b - drop + hook_len, 3) for b in strong if b >= drop + 0.25]
        song_rest_in = drop
    else:
        first = beats[0] if beats else 0
        tl_beats = [round(b - first, 3) for b in beats]
        P["beats"], P["strong"] = tl_beats, [round(b - first, 3) for b in strong]
        song_rest_in = first
    strong_tl = set(P["strong"])
    on_accent = lambda t: any(abs(t - x) < 0.02 for x in strong_tl)  # slice starts accumulate 3-decimal rounding, so no exact ==

    # --- montage slices on the beat grid
    n = every or {"ironic": 1, "phonk": 1, "velocity": 1, "aura": 4, "montage": 2, "trending": T and T["cut_every_beats"]}[style]
    end = min(tl_beats[-1] if tl_beats else 0, hook_len + {"aura": 20, "velocity": 16}.get(style, 14))
    marks = _marks(tl_beats, n, hook_len, end)
    if not smart:
        foot = Footage(clips, skip=hook_len if hooked else None)
    # user 2026-10-21: every clip at most ONCE, hard: the montage gets at most one shot per clip not used by the hook. Fewer clips than
    # beats -> a shorter edit, never a re-used clip (download more clips for a longer one)
    avail = len({c["media"] for c in (foot.clips if smart else clips) if c.get("ranges") != []} - ({clips[0]["media"]} if hooked else set()))
    if avail < 1:
        raise ValueError("every clip once: nothing left for the montage after the hook - add more clips")
    capped = len(marks) - 1 > avail
    marks = marks[:avail + 1]
    slow_every = T and T.get("slowmo_pct", 0) >= 0.2 and max(2, round(1 / T["slowmo_pct"]))  # trending: every k-th slice 2 beats at 0.5x
    slices = []
    if style == "velocity" or slow_every:
        slow_every = slow_every or 3
        k, i = 0, 0
        while i < len(marks) - 1 and len(slices) < avail:
            if k % slow_every != slow_every - 1 or i + 2 > len(marks) - 1:  # fast slice: one beat (velocity: at 2x)
                a, b = marks[i], marks[i + 1]; sp, i = (2.0 if style == "velocity" else 1.0), i + 1
            else:  # slow slice: two beats at 0.5x
                a, b = marks[i], marks[i + 2]; sp, i = 0.5, i + 2
            m, s, e = foot.take((b - a) * sp, beats=[(x - a) * sp for x in tl_beats if a <= x < b], accent=on_accent(a))
            slices.append(_seg(m, s, e, speed=sp, zoomTo=1.12 if sp < 1 else None, zoom=1.0 if sp < 1 else 1.1, **foot.frame))
            k += 1
    else:
        sp = 0.6 if style == "aura" else 1.0
        for a, b in zip(marks, marks[1:]):
            m, s, e = foot.take((b - a) * sp, beats=[(x - a) * sp for x in tl_beats if a <= x < b], accent=on_accent(a))
            slices.append(_seg(m, s, e, speed=sp, **foot.frame))
    for s in slices:
        if s.get("zoomTo") is None:
            s.pop("zoomTo", None)
    P["video"].extend(slices)
    step(f"{len(slices)} slices cut to {'every beat' if n == 1 else f'every {n} beats'} from {len(clips)} clip(s)")
    if capped:
        step(f"Every clip once: only {avail} unused clips, so the montage stops after {len(slices)} shots ({marks[-1] - hook_len:.1f}s) - add clips for a longer edit")

    # --- accents: zoom punch, flash, shake on strong beats
    t = hook_len
    starts = []
    for s in P["video"][1 if hooked else 0:]:
        starts.append(t); t += (s["out"] - s["in"]) / s["speed"]
    body = P["video"][1 if hooked else 0:]
    if style in ("ironic", "phonk", "trending"):
        punch, flash = (T["zoom_punch_pct"], T["flash_on_accents_pct"]) if T else (0.5, 1.0)
        shake = 0.35 if (T is None or T.get("shake_on_accents", True)) else 0
        acc = nsun = neyes = 0
        for k, (s, st) in enumerate(zip(body, starts)):
            if on_accent(st):
                s["zoom"], s["shake"] = 1.3, shake
                acc += 1
            elif _spread(k, punch):
                s["zoom"] = 1.12
        step(f"Zoom punch on {punch:.0%} of the cuts, 1.3x punch{' + shake' if shake else ''} on the accents")
        # user 2026-10-20: one or two slide-ins per edit: his cut-out flies in over the old shot and lands on the cut, then the
        # background appears (fxface.slide bakes it). Hard cuts, not on accents, at about 1/3 and 2/3 of the montage.
        dirs = ["bottom", "left", "right", "top"]
        for j, want in enumerate((len(body) // 3, 2 * len(body) // 3)):
            for k in range(want, len(body) - 1):  # first usable cut from there on (v7: a 4-cut window found only one)
                if 0 < k and not on_accent(starts[k + 1]) and not body[k + 1].get("sun") and not body[k].get("sun"):
                    body[k + 1]["slide"] = dirs[j % 4]
                    break
        acc = 0
        # user 2026-10-20: CapCut's zoom-in ("pull in", VerticalZoomNear recipe) between the clips, centred on the beat:
        # 4 frames on every cut, 6 frames alternating with the white flash on the accents
        for k, (s, st) in enumerate(zip(body, starts)):
            nxt = starts[k + 1] if k + 1 < len(starts) else None
            if nxt is None or body[k + 1].get("slide"):  # slide-ins need a hard cut
                continue
            if on_accent(nxt):
                if _spread(acc, flash):
                    s["transition"] = {"type": "fadewhite", "dur": 0.08} if acc % 2 == 0 else {"type": "pullin", "dur": 0.2}
                acc += 1
            else:
                s["transition"] = {"type": "pullin", "dur": 0.133}
            if s["transition"]["type"] != "none":
                # centred on the beat: half of the overlap eats the outgoing shot, half the incoming (CapCut); the incoming shot must grow by
                # the other half too, else every cut drifts dur/2 early (v6: 4 frames by the end). Pull-in switches A->B in the middle of
                # an even window, so it is pulled one frame earlier: the first frame of the new shot IS the beat frame (frame-checked v6)
                nudge = 1 / 30 if s["transition"]["type"] == "pullin" else 0
                extend(s, s["transition"]["dur"] / 2 - nudge)
                if k + 1 < len(body):
                    extend_in(body[k + 1], s["transition"]["dur"] / 2 + nudge)
        step(f"CapCut zoom-in between all clips, white flash on every other accent ({flash:.0%} of the accents)")
        if hooked and P["video"]:
            P["video"][0]["transition"] = {"type": "fadewhite", "dur": 0.1}
            extend(P["video"][0], 0.05)
            if body:
                extend_in(body[0], 0.05)
            step("Flash on the drop")
    elif style == "aura":
        for s in body:
            s["zoomTo"] = 1.15
        step("Slow push-in on every shot")
    elif style == "velocity":
        for k, (s, st) in enumerate(zip(body, starts)):
            if s["speed"] > 1:
                s["shake"] = 0.25
        step("Shake on the fast snaps")

    # --- end: freeze the last frame for a beat, fade out
    last = P["video"][-1]
    beat = 60 / bpm if bpm else 0.5
    P["video"].append({"id": _id(), "media": last["media"], "in": last["out"], "out": last["out"], "freeze": True, "dur": round(beat * 2, 3),
                       "speed": 1, "rotation": 0, "flip": False, "volume": 0,
                       "transition": {"type": "none", "dur": 0.5}, **{k: last[k] for k in ("cx", "cy") if k in last}})
    total = round(t + beat * 2, 3)
    P["audio"].append({"id": _id(), "media": song, "start": round(hook_len, 3), "in": round(song_rest_in, 3), "out": round(song_rest_in + total - hook_len, 3),
                       "speed": 1, "volume": 1, "fadeOut": 0.8})
    step(f"Song from the drop, freeze frame ending, total {total:.1f}s")

    # --- text pops on accents (impact font), one word per accent
    if text:
        words = text.split()
        pops = [b for b in P["strong"] if b >= hook_len][:len(words)] or [hook_len]
        for k, (w, b) in enumerate(zip(words, pops)):
            nxt = pops[k + 1] if k + 1 < len(pops) else b + beat * 2
            P["text"].append({"id": _id(), "text": w, "start": b, "dur": round(min(beat * 2, nxt - b), 3), "x": 0.5, "y": 0.22, "size": 0.09, "color": "#ffffff",
                              "font": "impact", "box": False})
        step(f"Text pops on accents: {' / '.join(words[:len(pops)])}")
    steps.append(("Done — press play", copy.deepcopy(P)))
    return steps


def selftest():
    clips = [{"media": "a.mp4", "dur": 8}, {"media": "b.mp4", "dur": 5}] + [{"media": f"c{k}.mp4", "dur": 6} for k in range(30)]
    grid = {"bpm": 120, "beats": [i * 0.5 for i in range(80)], "strong": [4, 6, 8, 10], "drop": 4.0}
    T0 = {"cut_every_beats": 1, "zoom_punch_pct": 0.5, "flash_on_accents_pct": 1.0, "slowmo_pct": 0.0, "hook_s": 0.0, "look": "phonk"}
    for st in STYLES:
        steps = plan(st, clips, grid, "song.mp3", text="UNSER EUROPA", template=T0 if st == "trending" else None)
        P = steps[-1][1]
        assert len(steps) >= 4 and P["video"] and P["audio"], st
        for s in P["video"]:
            assert s["out"] >= s["in"] and s["in"] >= 0, (st, s)
            d = next(c for c in clips if c["media"] == s["media"])["dur"]
            assert s["out"] <= d + 1e-6, (st, s)
        assert all(a["out"] >= a["in"] for a in P["audio"]), st
    R = plan("phonk", [{"media": "a.mp4", "dur": 8, "ranges": [(1, 3), (5, 8)]}, {"media": "b.mp4", "dur": 5, "ranges": []}], grid, "s.mp3")[-1][1]
    body = [s for s in R["video"] if not s.get("freeze")]
    assert all(s["media"] == "a.mp4" and (1 <= s["in"] and s["out"] <= 3 + 1e-6 or 5 <= s["in"] and s["out"] <= 8 + 1e-6) for s in body), body
    P = plan("ironic", clips, grid, "song.mp3", hook=2)[-1][1]
    assert P["video"][0]["volume"] == 1 and abs(P["video"][0]["out"] - 2.05) < 1e-6  # hook 2 s + the 0.1 s drop flash it overlaps
    assert P["audio"][0]["in"] == 2.0 and P["audio"][0]["out"] == 4.0, P["audio"][0]  # song ducked, drop at hook end
    assert P["audio"][1]["start"] == 2 and P["audio"][1]["in"] == 4.0
    assert P["beats"][0] == 2.0  # first timeline beat = drop = end of hook
    assert P["video"][1]["zoom"] == 1.3 and P["video"][1]["shake"]  # drop beat is an accent
    fx = [("sun" in s, "eyes" in s) for s in P["video"] if s.get("zoom") == 1.3]
    assert fx and not any(a or b for a, b in fx), fx  # school version: no sun, no eyes
    V = plan("velocity", clips, grid, "s.mp3")[-1][1]["video"]
    assert {s["speed"] for s in V if not s.get("freeze")} == {0.5, 2.0}
    # trending: knobs drive cut rate, punches, flashes, slow-mo mix, hook
    T = {"cut_every_beats": 2, "zoom_punch_pct": 0.0, "flash_on_accents_pct": 0.0, "slowmo_pct": 0.34, "hook_s": 2.0, "look": "bw", "shake_on_accents": False}
    P = plan("trending", clips, grid, "s.mp3", template=T)[-1][1]
    body = [s for s in P["video"][1:] if not s.get("freeze")]
    assert P["filter"] == "bw" and P["video"][0]["volume"] == 1 and abs(P["video"][0]["out"] - 2.05) < 1e-6, P["video"][0]
    assert {s["speed"] for s in body} == {0.5, 1.0} and not any(s.get("zoom") == 1.12 for s in body), body
    assert not any(s["transition"]["type"] == "fadewhite" for s in body) and not any(s.get("shake") for s in body)
    assert all((s["out"] - s["in"]) / s["speed"] >= 0.99 for s in body), [(s["in"], s["out"], s["speed"]) for s in body]  # 2 beats = 1 s at 120 BPM
    # moments: Picker is used when clips carry them, windows stay inside the ranges
    import math
    mo = [{"t": i / 8, "motion": (1 + math.sin(i)) / 2, "loud": 1.0 if 24 <= i < 48 else 0.1} for i in range(64)]
    mo2 = [{"t": i / 8, "motion": (1 + math.cos(i / 3)) / 2, "loud": 0.2} for i in range(160)]
    C = [{"media": "a.mp4", "dur": 8, "ranges": [(1, 3), (5, 8)], "moments": mo}, {"media": "b.mp4", "dur": 20, "ranges": [(0, 20)], "moments": mo2}]
    S = plan("ironic", C + [{**C[1], "media": f"b{k}.mp4"} for k in range(8)], grid, "s.mp3", hook=1.5)
    assert any("Moments" in m for m, _ in S), [m for m, _ in S]
    P = S[-1][1]
    assert P["video"][0]["media"] == "a.mp4" and 5 <= P["video"][0]["in"] and P["video"][0]["out"] <= 8 + 1e-6, P["video"][0]  # hook = loudest visible window (3-6 s loud, (5,8) visible)
    for s in P["video"][1:]:
        if not s.get("freeze"):
            r = next(c for c in C if c["media"] == s["media"][:1] + ".mp4")["ranges"]
            assert any(a - 1e-6 <= s["in"] and s["out"] <= b + 1e-6 for a, b in r), s
    wins = {(s["media"], s["in"]) for s in P["video"] if not s.get("freeze")}
    assert len(wins) == len([s for s in P["video"] if not s.get("freeze")]), "re-used footage"
    P = plan("ironic", C + [{**C[1], "media": f"b{k}.mp4"} for k in range(8)], grid, "s.mp3", hook=3.5)[-1][1]  # no 3.5 s window inside a.mp4's ranges
    assert P["video"][0]["media"] == "a.mp4" and P["video"][0]["in"] == 0 and not any(s["media"] == "a.mp4" for s in P["video"][1:]), [s["media"] for s in P["video"]]  # hook clip never in the montage
    # star track: windows only where he is in frame (5-8 s of a.mp4 at 20 %, b.mp4 everywhere at 30 % and left of centre), crop centre on him
    C[0]["star"] = [[k / 4, 0.4, 0.3, 0.1, 0.2] for k in range(20, 33)]; C[0]["cuts"] = []
    C[1]["star"] = [[k / 4, 0.1, 0.3, 0.15, 0.3] for k in range(81)]; C[1]["cuts"] = [10.0]
    S = plan("phonk", C + [{**C[1], "media": f"b{k}.mp4"} for k in range(8)], grid, "s.mp3")
    P = S[-1][1]
    body = [s for s in P["video"] if not s.get("freeze")]
    assert len(body) == 10 and len({s["media"] for s in body}) == 10 and any("Every clip once" in m for m, _ in S), [s["media"] for s in body]  # 28 beats, 10 clips -> 10 shots
    for s in P["video"]:
        if not s.get("freeze"):
            assert (s["cx"], s["cy"]) == ((0.45, 0.4) if s["media"] == "a.mp4" else (0.175, 0.45)), s
            assert s["media"].startswith("b") or 5.25 <= s["in"] and s["out"] <= 8 + 1e-6, s  # starts gated by the star track, ends may extend into the ranges for a transition
            assert not (s["media"].startswith("b") and s["in"] < 10 < s["out"]), s
    assert P["video"][-1]["freeze"] and P["video"][-1]["cx"] == P["video"][-2]["cx"]
    print("trends selftest ok")


if __name__ == "__main__":
    selftest()
