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


OVERPASS = ["https://maps.mail.ru/osm/tools/overpass/api/interpreter", "https://overpass.private.coffee/api/interpreter",
            "https://overpass-api.de/api/interpreter"]


def fetch_place(lat, lon, log=print):
    """Streets, buildings, the river and the school building around (lat, lon) from OpenStreetMap (Overpass), cached in
    data/places/. Bundled: Köln / Lessing-Gymnasium. Returns the place dict or None (offline: the intro just has no streets)."""
    for f in sorted((ROOT / "data" / "places").glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        if abs(d["target"]["lat"] - lat) < 0.03 and abs(d["target"]["lon"] - lon) < 0.03:
            return d
    import urllib.parse, urllib.request
    kx, ky = 111320 * np.cos(np.radians(lat)), 110540
    bb = lambda r: f"({lat - r / ky:.4f},{lon - r / kx:.4f},{lat + r / ky:.4f},{lon + r / kx:.4f})"

    def q(ql):
        for url in OVERPASS:
            try:
                req = urllib.request.Request(url, urllib.parse.urlencode({"data": ql}).encode(), {"User-Agent": "clipcut-mapintro/1.0"})
                return [e for e in json.loads(urllib.request.urlopen(req, timeout=180).read())["elements"] if e.get("geometry")]
            except Exception as e:  # busy mirror -> next one
                log(f"map intro: {url.split('/')[2]} failed ({e.__class__.__name__})")
        raise RuntimeError("no Overpass server reachable")

    try:
        log("map intro: loading streets and buildings from OpenStreetMap (once) ...")
        major = q(f'[out:json][timeout:170];way["highway"~"^(motorway|trunk|primary|secondary)$"]{bb(30000)};out geom;')
        minor = q(f'[out:json][timeout:170];way["highway"~"^(tertiary|residential|unclassified|living_street)$"]{bb(3500)};out geom;')
        bld = q(f'[out:json][timeout:170];way["building"]{bb(2500)};out geom;')
        water = q(f'[out:json][timeout:170];way["waterway"="river"]{bb(30000)};out geom;')
    except Exception as e:
        log(f"map intro: no street detail ({e})")
        return None
    S = lambda g: [v for pnt in g for v in (round(pnt["lon"] * 1e5), round(pnt["lat"] * 1e5))]
    rank = {"motorway": 3, "trunk": 3, "primary": 2, "secondary": 1}
    near = min(bld, key=lambda e: (np.mean([g["lat"] for g in e["geometry"]]) - lat) ** 2
               + ((np.mean([g["lon"] for g in e["geometry"]]) - lon) * np.cos(np.radians(lat))) ** 2) if bld else None
    d = {"source": "© OpenStreetMap contributors (ODbL)", "scale": 1e5, "target": {"lat": lat, "lon": lon, "poly": S(near["geometry"]) if near else None},
         "roads_major": [[rank[e["tags"]["highway"]]] + S(e["geometry"]) for e in major], "roads_minor": [S(e["geometry"]) for e in minor],
         "buildings": [S(e["geometry"]) for e in bld], "water": [S(e["geometry"]) for e in water], "landmarks": []}
    (ROOT / "data" / "places").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "places" / f"{lat:.3f}_{lon:.3f}.json").write_text(json.dumps(d, separators=(",", ":")), encoding="utf-8")
    return d


def _geo(flat):
    a = np.float64(flat).reshape(-1, 2) / 1e5
    return np.stack(merc(a[:, 0], a[:, 1]), 1)


class _Layer:
    """Many polylines/polygons in mercator metres + their bounding boxes, so each frame only draws what is on screen."""

    def __init__(self, geoms):
        self.g = geoms
        self.bb = np.array([[q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()] for q in geoms]) if geoms else np.zeros((0, 4))

    def visible(self, x0, y0, x1, y1):
        if not len(self.g):
            return []
        m = (self.bb[:, 2] >= x0) & (self.bb[:, 0] <= x1) & (self.bb[:, 3] >= y0) & (self.bb[:, 1] <= y1)
        return [self.g[i] for i in np.nonzero(m)[0]]


def render(out, lat, lon, label="", sub="", dur=10.0, W=1920, H=1080, fps=30, country=None, log=print):
    import brand
    D = json.loads((ROOT / "data" / "europe.json").read_text(encoding="utf-8"))
    PL = fetch_place(lat, lon, log)
    if PL and PL["target"].get("poly"):  # land the pin on the real school building
        a = np.float64(PL["target"]["poly"]).reshape(-1, 2) / 1e5
        lon, lat = float(a[:, 0].mean()), float(a[:, 1].mean())
    roads = {r: _Layer([_geo(x[1:]) for x in (PL or {}).get("roads_major", []) if x[0] == r]) for r in (1, 2, 3)}
    minor = _Layer([_geo(x) for x in (PL or {}).get("roads_minor", [])])
    blds = _Layer([_geo(x) for x in (PL or {}).get("buildings", [])])
    water = _Layer([_geo(x) for x in (PL or {}).get("water", [])])
    marks = [(m["name"], _geo(m["poly"])) for m in (PL or {}).get("landmarks", [])]
    school = _geo(PL["target"]["poly"]) if PL and PL["target"].get("poly") else None
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
    w0, w1 = 4600e3, (2.6e3 if PL else 170e3)  # visible width (m): Europe -> the school's quarter (-> region without street data)
    c0 = np.array(merc(11.0, 51.5))
    T = np.array([tx, ty])
    k_lat = 1 / np.cos(np.radians(lat))
    t_zoom0, t_zoom1 = dur * 0.22, dur * 0.80
    logo = brand.logo(round(58 * u), wide=True, shadow=True)
    n = int(round(dur * fps))
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(n):
            t = i / fps
            uz = min(1.0, max(0.0, (t - t_zoom0) / (t_zoom1 - t_zoom0)))
            z = ease(uz ** 0.8)  # a bit quicker over the continent, slower down in the streets
            width = math.exp(math.log(w0 * 0.92) + (math.log(w1) - math.log(w0 * 0.92)) * z) if t >= t_zoom0 else w0 * (1 - 0.08 * t / t_zoom0)
            # centre follows the zoom so the target drifts in smoothly (van Wijk-like: move with the log-zoom)
            m = (math.log(w0) - math.log(width)) / (math.log(w0) - math.log(w1))
            scale = W / (width * k_lat)  # px per mercator metre
            Tv = T + np.array([0.13 * W / scale, 0.0]) * min(1.0, max(0.0, m))  # end: pin left of centre, logo beside it
            # the target's screen offset shrinks smoothly to 0: it never leaves the frame, even over a 1000x zoom
            ctr = Tv + (c0 - Tv) * (width / w0) * (1 - ease(min(1.0, max(0.0, m))))
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
            ramp = lambda a, b: ease((math.log(a) - math.log(width)) / (math.log(a) - math.log(b)))  # 0 at width a -> 1 at width b
            osm_w = ramp(200e3, 50e3) if len(water.g) else 0  # OSM river replaces the coarse map river when zooming in
            view = (ctr[0] - W / 2 / scale, ctr[1] - H / 2 / scale, ctr[0] + W / 2 / scale, ctr[1] + H / 2 / scale)

            def layer(geoms, closed, col, width_px, alpha, fill=None):
                nonlocal img
                vis = geoms.visible(*view) if isinstance(geoms, _Layer) else geoms
                if alpha <= 0.01 or not vis:
                    return
                ov = img.copy()
                pts = [np.int32(P(g) * 16) for g in vis]
                if fill is not None:
                    cv2.fillPoly(ov, pts, fill, cv2.LINE_AA, shift=4)
                if width_px:
                    cv2.polylines(ov, pts, closed, col, max(1, round(width_px)), cv2.LINE_AA, shift=4)
                img = cv2.addWeighted(ov, alpha, img, 1 - alpha, 0)

            if osm_w > 0:  # the Rhine at its real width (~350 m), at least a visible line
                layer(water, False, RIVER, max(rw, 350 * scale), osm_w)
            layer(roads[3], False, (0xe8, 0xdc, 0xcf), 2.2 * u + 18 * scale, 0.55 * ramp(400e3, 80e3))
            layer(roads[2], False, (0xe0, 0xd2, 0xc2), 1.6 * u + 12 * scale, 0.45 * ramp(150e3, 30e3))
            layer(roads[1], False, (0xd8, 0xc8, 0xb6), 1.2 * u + 9 * scale, 0.40 * ramp(60e3, 12e3))
            layer(minor, False, (0xd0, 0xbc, 0xa6), 0.9 * u + 6 * scale, 0.35 * ramp(20e3, 4e3))
            layer(blds, True, (0xc8, 0xa6, 0x80), 0.8 * u if scale > 0.3 else 0, 0.75 * ramp(25e3, 4e3), fill=(0xa6, 0x7c, 0x52))
            for name, g in marks:  # landmarks (Kölner Dom): gold outline while the city is in view
                layer([g], True, GOLD, 2 * u, ramp(250e3, 50e3), fill=(0x6f, 0xb4, 0xe6))
            for xy in ([] if osm_w >= 1 else rivers):
                q = P(xy)
                if q[:, 0].max() < 0 or q[:, 0].min() > W or q[:, 1].max() < 0 or q[:, 1].min() > H:
                    continue
                if osm_w > 0:
                    ov = img.copy(); cv2.polylines(ov, [np.int32(q * 16)], False, RIVER, rw, cv2.LINE_AA, shift=4)
                    img = cv2.addWeighted(ov, 1 - osm_w, img, osm_w, 0)
                else:
                    cv2.polylines(img, [np.int32(q * 16)], False, RIVER, rw, cv2.LINE_AA, shift=4)
            if school is not None:  # the school building lights up gold before the pin lands
                layer([school], True, GOLD, 2.5 * u, ease((t - t_zoom1 + 0.8) / 0.8), fill=(0x4a, 0xa6, 0xe0))
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
                wshow = min(3000e3, max(120e3, 3000e3 * (pop / 4e6) ** 0.6))  # width at which this city is fully shown
                a = int(200 * ease(t / 1.2) * ramp(wshow * 3, wshow) * (1 - ramp(40e3, 10e3)))
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
            for name, g in marks:
                q = P(g).mean(0)
                a = int(230 * ramp(250e3, 50e3) * (1 - ramp(15e3, 5e3)))
                if a > 0 and 0 < q[0] < W and 0 < q[1] < H:
                    d.text((q[0] + 12 * u, q[1] - 30 * u), name, font=font("bold", 22 * u), fill=(255, 214, 120, a))
            if PL and osm_w > 0:  # licence: OpenStreetMap data must be credited where it is shown
                d.text((W - 16 * u, H - 14 * u), "Kartendaten © OpenStreetMap-Mitwirkende, Natural Earth", font=font("medium", 15 * u),
                       fill=(255, 255, 255, int(170 * osm_w)), anchor="rd")
            # ring of 12 stars over Europe at the start
            grow = (w0 / width) ** 0.42  # the ring opens up with the zoom: the camera flies through it
            rr, s = 0.30 * H * (1 + 0.03 * t) * grow, 0.032 * H * grow ** 0.7
            sa = ease(t / 0.8) * (1 - ease((rr - 0.5 * W) / (0.45 * W)))
            if sa > 0:
                for k in range(12):
                    ang = -np.pi / 2 + k * np.pi / 6 + t * 0.32  # slow, visible turn
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
    assert gold(last) > 0.0002, "pin / gold school building at the end"
    cap.release(); os.remove(out)
    print("mapintro selftest ok")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    if sys.argv[1:2] == ["selftest"]:
        selftest(); sys.exit()
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("lat", type=float); ap.add_argument("lon", type=float); ap.add_argument("label", nargs="?", default="")
    ap.add_argument("--sub", default=""); ap.add_argument("--dur", type=float, default=10.0)
    ap.add_argument("--w", type=int, default=1920); ap.add_argument("--h", type=int, default=1080); ap.add_argument("--fps", type=int, default=30)
    o = ap.parse_args()
    print(render(o.out, o.lat, o.lon, o.label, o.sub, o.dur, o.w, o.h, o.fps))
