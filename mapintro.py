"""mapintro.py - map intro: Europe with the ring of 12 stars, then a smooth flight down to the school (pin + school logo).

Map data: data/europe.json (Natural Earth, public domain - countries 1:50m, rivers, cities 1:10m), no internet needed.
Look: deep navy sea, EU member states in blue, the target country outlined in gold, rivers, city names fading in while
zooming, at the end a gold pin with a pulse and the school logo (brand.py) next to it.

python mapintro.py OUT.mp4 LAT LON ["Label"] [--sub "Köln-Porz"] [--dur 7] [--w 1920 --h 1080] [--fps 30]
python mapintro.py selftest
In the editor: `python clipcut.py op mapintro LAT LON "Label" [sub=...] [dur=7]` (puts it in front of the first shot).
"""
import argparse, json, math, subprocess, sys, tempfile
from pathlib import Path

import cv2, numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).parent
DE = {"Cologne": "Köln", "Munich": "München", "Vienna": "Wien", "Rome": "Rom", "Prague": "Prag", "Warsaw": "Warschau",
      "Brussels": "Brüssel", "Lisbon": "Lissabon", "Copenhagen": "Kopenhagen", "Athens": "Athen", "Bucharest": "Bukarest",
      "Nuremberg": "Nürnberg", "Hanover": "Hannover", "Moscow": "Moskau", "Kiev": "Kiew", "Kyiv": "Kiew", "Belgrade": "Belgrad",
      "Venice": "Venedig", "Milan": "Mailand", "Naples": "Neapel", "Florence": "Florenz", "Geneva": "Genf", "Zurich": "Zürich",
      "The Hague": "Den Haag", "Antwerp": "Antwerpen", "Gothenburg": "Göteborg", "Brunswick": "Braunschweig", "Turin": "Turin",
      "Luxembourg": "Luxemburg", "Valletta": "Valletta", "Nicosia": "Nikosia", "Sofia": "Sofia", "Duesseldorf": "Düsseldorf"}
SEA, LAND, EUL, BORDER, GOLD, RIVER = (0x36, 0x22, 0x0e), (0x52, 0x3b, 0x23), (0x8a, 0x5c, 0x30), (0xcc, 0xb6, 0x9f), (0x2b, 0xc0, 0xff), (0xd6, 0xa7, 0x6f)  # BGR
R_EARTH = 6378137.0


def merc(lon, lat):
    lat = np.clip(lat, -85, 85)
    return R_EARTH * np.radians(lon), R_EARTH * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))


def ease(t):
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


def star(d, cx, cy, r, fill):
    pts = []
    for k in range(10):
        a = -np.pi / 2 + k * np.pi / 5
        rr = r if k % 2 == 0 else r * 0.382
        pts.append((cx + rr * np.cos(a), cy + rr * np.sin(a)))
    d.polygon(pts, fill=fill)


def render(out, lat, lon, label="", sub="", dur=7.0, W=1920, H=1080, fps=30, country=None, log=print):
    import brand
    D = json.loads((ROOT / "data" / "europe.json").read_text(encoding="utf-8"))
    eu = set(D["eu"])
    tx, ty = merc(lon, lat)
    if country is None:  # the country the target lies in (point in polygon)
        for c in D["countries"]:
            for p in c["p"]:
                a = np.float32(p)
                if cv2.pointPolygonTest(a.reshape(-1, 1, 2), (float(lon), float(lat)), False) >= 0:
                    country = c["a3"]
    polys = [(c["a3"], np.stack(merc(np.float64(p)[:, 0], np.float64(p)[:, 1]), 1)) for c in D["countries"] for p in c["p"]]
    rivers = [np.stack(merc(np.float64(r["l"])[:, 0], np.float64(r["l"])[:, 1]), 1) for r in D["rivers"]]
    places = [(DE.get(p["name"], p["name"]), *merc(p["lon"], p["lat"]), p["pop"]) for p in D["places"]
              if abs(p["lat"] - lat) > 0.02 or abs(p["lon"] - lon) > 0.02]  # the target itself gets the pin, not a label
    u = min(W, H * 16 / 9) / 1920  # 1 unit = 1 px at 1920 wide
    font = lambda w, px: brand.F(w, px)
    w0, w1 = 4600e3, 170e3  # visible width (m at the target's latitude): Europe -> region (neighbour cities still in view)
    c0 = np.array(merc(11.0, 51.5))
    T = np.array([tx, ty])
    k_lat = 1 / np.cos(np.radians(lat))
    t_zoom0, t_zoom1 = dur * 0.27, dur * 0.80
    logo = brand.logo(round(58 * u), wide=True, shadow=True)
    n = int(round(dur * fps))
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(n):
            t = i / fps
            z = ease((t - t_zoom0) / (t_zoom1 - t_zoom0))
            width = math.exp(math.log(w0 * 0.92) + (math.log(w1) - math.log(w0 * 0.92)) * z) if t >= t_zoom0 else w0 * (1 - 0.08 * t / t_zoom0)
            # centre follows the zoom so the target drifts in smoothly (van Wijk-like: move with the log-zoom)
            m = (math.log(w0) - math.log(width)) / (math.log(w0) - math.log(w1))
            scale = W / (width * k_lat)  # px per mercator metre
            Tv = T + np.array([0.13 * W / scale, 0.0]) * min(1.0, max(0.0, m))  # end: pin left of centre, logo beside it
            ctr = c0 + (Tv - c0) * min(1.0, max(0.0, m)) ** 0.7
            P = lambda xy: np.stack([(xy[..., 0] - ctr[0]) * scale + W / 2, (ctr[1] - xy[..., 1]) * scale + H / 2], -1)
            img = np.zeros((H, W, 3), np.uint8); img[:] = SEA
            yy, xx = np.mgrid[0:H, 0:W]
            vig = 1 - 0.35 * (((xx - W / 2) / W) ** 2 + ((yy - H / 2) / H) ** 2) * 2
            hl = ease((t - t_zoom0 * 0.9) / (dur * 0.25))  # target country highlight
            for a3, xy in polys:
                q = P(xy)
                if q[:, 0].max() < -50 or q[:, 0].min() > W + 50 or q[:, 1].max() < -50 or q[:, 1].min() > H + 50:
                    continue
                col = EUL if a3 in eu else LAND
                if a3 == country:
                    col = tuple(int(c + (d - c) * 0.35 * hl) for c, d in zip(col, (0xa8, 0x76, 0x40)))
                cv2.fillPoly(img, [np.int32(q * 16)], col, cv2.LINE_AA, shift=4)
            for a3, xy in polys:
                q = P(xy)
                if q[:, 0].max() < -50 or q[:, 0].min() > W + 50:
                    continue
                cv2.polylines(img, [np.int32(q * 16)], True, BORDER, max(1, round(1.2 * u)), cv2.LINE_AA, shift=4)
            rw = max(1, round(u * (1 + 2.5 * z)))
            for xy in rivers:
                q = P(xy)
                if q[:, 0].max() < 0 or q[:, 0].min() > W or q[:, 1].max() < 0 or q[:, 1].min() > H:
                    continue
                cv2.polylines(img, [np.int32(q * 16)], False, RIVER, rw, cv2.LINE_AA, shift=4)
            if country and hl > 0:  # gold outline of the target country
                ov = img.copy()
                for a3, xy in polys:
                    if a3 == country:
                        cv2.polylines(ov, [np.int32(P(xy) * 16)], True, GOLD, max(2, round(2.6 * u)), cv2.LINE_AA, shift=4)
                img = cv2.addWeighted(ov, hl, img, 1 - hl, 0)
            img = (img.astype(np.float32) * vig[..., None]).clip(0, 255).astype(np.uint8)
            im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGBA")
            lay = Image.new("RGBA", im.size, (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
            # city names: the bigger the city, the earlier it shows up while zooming in
            boxes = []
            for name, px_, py_, pop in sorted(places, key=lambda p: -p[3]):  # big cities first, no overlapping labels
                q = P(np.array([px_, py_]))
                if not (0 < q[0] < W and 0 < q[1] < H):
                    continue
                need = 1.2e6 if width > 1500e3 else 6e5 if width > 500e3 else 2.5e5
                if pop < need:
                    continue
                a = int(200 * ease((t - t_zoom0) / (dur * 0.1))) if width < 2500e3 else 0
                if a <= 0:
                    continue
                f_ = font("medium", 20 * u)
                bb = d.textbbox((q[0] + 8 * u, q[1] - 12 * u), name, font=f_)
                bb = (bb[0] - 10 * u, bb[1] - 4 * u, bb[2] + 4 * u, bb[3] + 4 * u)
                if any(bb[0] < o[2] and o[0] < bb[2] and bb[1] < o[3] and o[1] < bb[3] for o in boxes):
                    continue
                boxes.append(bb)
                r = max(2, round(3 * u))
                d.ellipse([q[0] - r, q[1] - r, q[0] + r, q[1] + r], fill=(255, 255, 255, a))
                d.text((q[0] + 8 * u, q[1] - 12 * u), name, font=f_, fill=(230, 236, 242, a))
            # ring of 12 stars over Europe at the start
            sa = ease(t / 0.8) * (1 - ease((t - t_zoom0 * 0.8) / 0.7))
            if sa > 0:
                rr, s = 0.30 * H * (1 + 0.04 * t), 0.032 * H
                for k in range(12):
                    ang = -np.pi / 2 + k * np.pi / 6 + t * 0.05
                    star(d, W / 2 + rr * np.cos(ang), H / 2 + rr * np.sin(ang), s, (255, 204, 0, int(255 * sa)))
            # pin + label at the end
            pa = ease((t - t_zoom1 + 0.25) / 0.5)
            if pa > 0:
                q = P(T)
                drop = (1 - pa) * 40 * u
                pr = 11 * u
                pulse = ((t - t_zoom1) * 0.8) % 1 if t > t_zoom1 else 0
                if t > t_zoom1:
                    R = pr + pulse * 55 * u
                    d.ellipse([q[0] - R, q[1] - R, q[0] + R, q[1] + R], outline=(255, 192, 43, int(180 * (1 - pulse))), width=max(2, round(3 * u)))
                d.ellipse([q[0] - pr, q[1] - pr - drop, q[0] + pr, q[1] + pr - drop], fill=(255, 192, 43, int(255 * pa)),
                          outline=(255, 255, 255, int(255 * pa)), width=max(2, round(3 * u)))
                la = ease((t - t_zoom1) / 0.6)
                if la > 0:
                    x0, y0 = q[0] + 34 * u, q[1] - logo.height / 2
                    lg = logo.copy(); lg.putalpha(lg.getchannel("A").point(lambda v: int(v * la)))
                    lay.alpha_composite(lg, (int(x0), int(y0 - 14 * u)))
                    if sub:
                        d.text((x0 + 8 * u, y0 + logo.height - 6 * u), sub, font=font("medium", 26 * u), fill=(255, 255, 255, int(220 * la)))
                    if label and label.upper() != "LESSING-GYMNASIUM":  # another school: its name under the logo line
                        d.text((x0 + 8 * u, y0 - 50 * u), label, font=font("bold", 34 * u), fill=(255, 255, 255, int(255 * la)))
            im.alpha_composite(lay)
            im.convert("RGB").save(f"{tmp}/{i:05d}.png")
            if i % 60 == 0:
                log(f"map intro: {i}/{n}")
        import clipcut
        clipcut.run([clipcut.FF, "-y", "-framerate", str(fps), "-i", f"{tmp}/%05d.png", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
                     "-shortest", "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac", str(out)])
    return out


def selftest():
    import os
    out = Path(tempfile.gettempdir()) / "mapintro_selftest.mp4"
    render(out, 50.87, 7.05, "Lessing-Gymnasium", "Köln-Porz", dur=4.0, W=480, H=270, fps=10, log=lambda m: None)
    cap = cv2.VideoCapture(str(out)); frames = []
    while True:
        ok, f = cap.read()
        if not ok: break
        frames.append(f)
    assert len(frames) >= 39, len(frames)
    first, last = frames[8], frames[-1]  # 0.8 s: the star ring is fully in
    gold = lambda f: ((f[..., 2] > 180) & (f[..., 1] > 120) & (f[..., 0] < 120)).mean()  # gold, anti-aliased at this small size
    assert gold(first) > 0.002, "the 12 gold stars should show at the start"
    assert np.abs(first.astype(int) - last.astype(int)).mean() > 8, "the map must zoom (first and last frame differ)"
    assert gold(last) > 0.0005, "pin / gold outline at the end"
    cap.release(); os.remove(out)
    print("mapintro selftest ok")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    if sys.argv[1:2] == ["selftest"]:
        selftest(); sys.exit()
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("lat", type=float); ap.add_argument("lon", type=float); ap.add_argument("label", nargs="?", default="")
    ap.add_argument("--sub", default=""); ap.add_argument("--dur", type=float, default=7.0)
    ap.add_argument("--w", type=int, default=1920); ap.add_argument("--h", type=int, default=1080); ap.add_argument("--fps", type=int, default=30)
    o = ap.parse_args()
    print(render(o.out, o.lat, o.lon, o.label, o.sub, o.dur, o.w, o.h, o.fps))
