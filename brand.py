"""brand.py - the school's look: logo (re-drawn, sharp at any size) and lower thirds in the school colours.

Lessing-Gymnasium Köln (Porz-Zündorf): colours from lessing-gymnasium.eu (blue #375e80) and the gold of its logo; the logo
(roof line + two slabs over "LESSING-GYMNASIUM" in a light geometric sans) only exists online as tiny rasters, so it is
re-drawn here at any size. Another school: change BRAND (and _icon) - everything else uses these values.

python brand.py logo OUT.png [--wide] [--tagline] [--h 400] [--color #ffffff]   the logo as a transparent PNG
python brand.py lowerthird "Name" "Rolle" OUT.png [--w 1920 --h 1080]          one lower third (full-frame transparent PNG)
python brand.py selftest
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).parent
BRAND = {"name": "LESSING-GYMNASIUM", "lines": ("LESSING-", "GYMNASIUM"), "tagline": "KOMPETENT  ENGAGIERT  MENSCHLICH",
         "blue": "#375e80", "gold": "#ffc02b", "light": "#ffffff"}
FONT = {"light": "Montserrat-Light.ttf", "medium": "Montserrat-Medium.ttf", "bold": "Montserrat-SemiBold.ttf"}


def F(weight, px):
    return ImageFont.truetype(str(ROOT / "fonts" / FONT[weight]), max(1, int(px)))


def rgb(h):
    return tuple(int(h.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))


def _icon(d, x, y, s, col):
    """The roof icon in a 640 x 300 design box at (x, y), scale s: a thin roof line rising to the right and bending into a
    tall slab, a second shorter slab with a rounded top beside it (the school's curved building)."""
    P = lambda px, py: (x + px * s, y + py * s)
    roof = [P(18, 190), P(430, 22)] + [P(430 + 100 * t, 22 - 12 * t + 2 * t * t) for t in np.linspace(0, 1, 16)[1:]]
    d.line(roof, fill=col, width=max(1, round(9 * s)), joint="curve")
    d.rounded_rectangle([P(514, 12), P(556, 296)], radius=22 * s, fill=col, corners=(False, True, False, False))
    d.rounded_rectangle([P(580, 46), P(626, 281)], radius=30 * s, fill=col, corners=(False, True, False, False))


def icon(h, col=None):
    """Just the roof icon, h px tall (RGBA)."""
    S = 4
    s = h * S / 296
    im = Image.new("RGBA", (round(640 * s), round(300 * s)), (0, 0, 0, 0))
    _icon(ImageDraw.Draw(im), 0, 0, s, rgb(col or BRAND["gold"]) + (255,))
    im = im.crop(im.getbbox())
    return im.resize((max(1, round(im.width / S)), max(1, round(im.height / S))), Image.LANCZOS)


def logo(h=400, wide=False, color=None, icon_col=None, tagline=False, shadow=False):
    """RGBA logo, h = pixel height. Stacked (icon over two text lines, like the school's documents) or wide (one text line,
    icon right, optional tagline - like the website header). Drawn at 4x and scaled down (clean edges)."""
    color, ic = rgb(color or BRAND["light"]) + (255,), rgb(icon_col or BRAND["gold"]) + (255,)
    S = 4
    probe = ImageDraw.Draw(Image.new("L", (1, 1)))
    if not wide:
        im = Image.new("RGBA", (660 * S, 440 * S), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
        _icon(d, 0, 0, S, ic)
        l, t, r, b = probe.textbbox((0, 0), BRAND["lines"][1], font=F("light", 112 * S))
        f = F("light", 112 * S * (610 * S) / (r - l))  # the long line spans the logo width, like the original
        for txt, bottom in ((BRAND["lines"][0], 299), (BRAND["lines"][1], 411)):
            l, t, r, b = probe.textbbox((0, 0), txt, font=f)
            d.text((16 * S - l, bottom * S - b), txt, font=f, fill=color)
    else:
        f = F("light", 200 * S)
        l, t, r, b = probe.textbbox((0, 0), BRAND["name"], font=f)
        cap = b - t
        isc = cap * 1.25 / 296  # icon about 1.25 cap heights tall
        tg = F("light", cap * 0.5)
        im = Image.new("RGBA", (int(r - l + 60 * S + 640 * isc), int(cap * 2.2)), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
        base = int(cap * 0.3)
        d.text((2 * S - l, base - t), BRAND["name"], font=f, fill=color)
        _icon(d, r - l + 30 * S, base + cap - 296 * isc, isc, ic)
        if tagline:
            tl, tt, tr, tb = d.textbbox((0, 0), BRAND["tagline"], font=tg)
            d.text((2 * S - tl, base + cap * 1.3 - tt), BRAND["tagline"], font=tg, fill=color)
    im = im.crop(im.getbbox())
    im = im.resize((max(1, round(im.width * h / im.height)), h), Image.LANCZOS)
    if shadow:  # soft dark halo: the thin white letters stay readable over bright footage
        p = max(2, round(h * 0.25))
        out = Image.new("RGBA", (im.width + 2 * p, im.height + 2 * p), (0, 0, 0, 0))
        a = Image.new("L", out.size, 0); a.paste(im.getchannel("A"), (p, p + max(1, h // 30)))
        sh = Image.new("RGBA", out.size, (0, 0, 0, 0)); sh.putalpha(a.filter(ImageFilter.GaussianBlur(max(1, h / 10))).point(lambda v: int(v * 0.6)))
        out.alpha_composite(sh); out.alpha_composite(im, (p, p))
        im = out
    return im


def lowerthird(name, role, W=1920, H=1080):
    """Full-frame transparent RGBA lower third, bottom left: gold accent bar, name in white on Lessing blue (gold roof icon at
    its end), role in blue on white, a soft shadow. Sized from the frame (works in 9:16 too); captions stay free below."""
    S = 2
    im = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    u = min(H, W * 9 / 16 * 1.6) * S / 1080  # 1 unit = 1 px at 1080p landscape; portrait: from the width
    x0 = round((0.05 if W >= H else 0.06) * W * S)
    fn, fr = F("bold", 44 * u), F("medium", 27 * u)
    nl, nt, nr, nb = d.textbbox((0, 0), name, font=fn)
    rl, rt, rr, rb = d.textbbox((0, 0), role or " ", font=fr)
    ph1, ph2, pad, bar = round(80 * u), round(50 * u), round(28 * u), round(8 * u)
    ic = icon(round(30 * u))
    w1 = (nr - nl) + 2 * pad + ic.width + round(22 * u)
    w2 = (rr - rl) + 2 * pad if role else 0
    yb = round((0.80 if W >= H else 0.74) * H * S)  # bottom of the role panel (captions sit at 0.9)
    y1 = yb - (ph2 if role else 0) - ph1
    sh = Image.new("RGBA", im.size, (0, 0, 0, 0))
    ImageDraw.Draw(sh).rectangle([x0, y1 + round(8 * u), x0 + bar + max(w1, w2) + round(8 * u), yb + round(8 * u)], fill=(0, 0, 0, 80))
    im.alpha_composite(sh.filter(ImageFilter.GaussianBlur(16 * u)))
    blue, gold = rgb(BRAND["blue"]), rgb(BRAND["gold"])
    d.rectangle([x0 + bar, y1, x0 + bar + w1, y1 + ph1 - 1], fill=blue + (245,))
    d.text((x0 + bar + pad - nl, y1 + (ph1 - (nb - nt)) / 2 - nt), name, font=fn, fill=(255, 255, 255, 255))
    im.alpha_composite(ic, (x0 + bar + w1 - pad - ic.width, y1 + (ph1 - ic.height) // 2))
    if role:
        d.rectangle([x0 + bar, y1 + ph1, x0 + bar + w2, yb - 1], fill=(255, 255, 255, 240))
        d.text((x0 + bar + pad - rl, y1 + ph1 + (ph2 - (rb - rt)) / 2 - rt), role, font=fr, fill=blue + (255,))
    d.rectangle([x0, y1, x0 + bar - 1, yb - 1], fill=gold + (255,))
    return im.resize((W, H), Image.LANCZOS)


def lowerthird_glass(name, role, W=1920, H=1080):
    """Glassmorphism lower third in the school colours: one rounded frosted-glass card (the video behind it is blurred at
    render time through the returned mask) tinted Lessing blue, a light top sheen and hairline rim, gold accent bar, name in
    white, role in soft white, the gold roof logo on the right. Returns (RGBA card art, L mask of the glass area)."""
    S = 2
    u = (H if W >= H else W * 1.3) * S / 1080  # portrait: a bit larger relative to the narrow frame
    probe = ImageDraw.Draw(Image.new("L", (1, 1)))
    fn, fr = F("bold", 44 * u), F("medium", 27 * u)
    nl, nt, nr, nb = probe.textbbox((0, 0), name, font=fn)
    rl, rt, rr, rb = probe.textbbox((0, 0), role or " ", font=fr)
    pad, gap, rad = round(30 * u), round(12 * u), round(18 * u)
    ic = None
    tw = max(nr - nl, (rr - rl) if role else 0)
    ch = round(pad * 0.8 + (nb - nt) + ((gap + rb - rt) if role else 0) + pad * 0.8)
    ic = logo(round(ch * 0.62))  # the school logo (white text, gold roof) at the right end of the card
    sep = round(28 * u)
    cw = round(pad + 8 * u + pad * 0.8 + tw + sep * 2 + ic.width + pad)
    x0 = round((0.05 if W >= H else 0.06) * W * S)
    yb = round((0.82 if W >= H else 0.76) * H * S)  # captions stay free below (y 0.9)
    y0 = yb - ch
    box = [x0, y0, x0 + cw, yb]
    mask = Image.new("L", (W * S, H * S), 0)
    ImageDraw.Draw(mask).rounded_rectangle(box, radius=rad, fill=255)
    art = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
    sh = Image.new("RGBA", art.size, (0, 0, 0, 0))  # soft shadow outside the card only
    ImageDraw.Draw(sh).rounded_rectangle([box[0], box[1] + 10 * u, box[2], box[3] + 10 * u], radius=rad, fill=(0, 0, 0, 70))
    sh = sh.filter(ImageFilter.GaussianBlur(18 * u))
    sh.putalpha(Image.fromarray((np.asarray(sh)[..., 3] * (1 - np.asarray(mask) / 255)).astype(np.uint8)))
    art.alpha_composite(sh)
    blue, gold = rgb(BRAND["blue"]), rgb(BRAND["gold"])
    tint = Image.new("RGBA", art.size, tuple(round(c * 0.75 + 255 * 0.25 * 0.6) for c in blue) + (0,))  # a touch lighter
    ta = np.asarray(mask, np.float32) / 255 * 80  # light Lessing-blue veil: the blurred video stays visible (glass)
    yy = (np.arange(H * S)[:, None] - y0) / max(ch, 1)
    sheen = np.clip(1 - yy * 1.6, 0, 1) * (np.asarray(mask) > 0) * 38  # light falls on the top of the glass
    tint.putalpha(Image.fromarray(ta.astype(np.uint8)))
    art.alpha_composite(tint)
    sh2 = Image.new("RGBA", art.size, (255, 255, 255, 0)); sh2.putalpha(Image.fromarray(sheen.astype(np.uint8)))
    art.alpha_composite(sh2)
    d = ImageDraw.Draw(art)
    d.rounded_rectangle(box, radius=rad, outline=(255, 255, 255, 95), width=max(1, round(1.6 * u)))  # hairline rim
    ax = x0 + pad
    d.rounded_rectangle([ax, y0 + pad * 0.8, ax + round(7 * u), yb - pad * 0.8], radius=round(3.5 * u), fill=gold + (255,))
    tx = ax + round(7 * u) + round(pad * 0.8)
    d.text((tx - nl, y0 + pad * 0.8 - nt), name, font=fn, fill=(255, 255, 255, 255))
    if role:
        d.text((tx - rl, y0 + pad * 0.8 + (nb - nt) + gap - rt), role, font=fr, fill=(255, 255, 255, 215))
    lx = box[2] - pad - ic.width
    d.line([(lx - sep, y0 + ch * 0.22), (lx - sep, yb - ch * 0.22)], fill=(255, 255, 255, 90), width=max(1, round(1.4 * u)))
    art.alpha_composite(ic, (lx, y0 + (ch - ic.height) // 2))
    return art.resize((W, H), Image.LANCZOS), mask.resize((W, H), Image.LANCZOS)


def selftest():
    lg = logo(200)
    a = np.asarray(lg)[..., 3]
    assert lg.height == 200 and 250 < lg.width < 350 and a.max() == 255 and 0.08 < (a > 128).mean() < 0.6, (lg.size, (a > 128).mean())
    px = np.asarray(lg)[..., :3][a > 200].astype(int)
    assert (np.abs(px - rgb(BRAND["gold"])).sum(1) < 40).mean() > 0.15, "icon should be gold"
    wd = logo(80, wide=True)
    assert wd.width > 6 * wd.height, wd.size
    lt = np.asarray(lowerthird("Annika Fink", "Lehrerin", 1920, 1080))
    ys, xs = np.nonzero(lt[..., 3] > 200)
    assert xs.min() > 60 and xs.max() < 960 and 700 < ys.min() and ys.max() < 870, (xs.min(), xs.max(), ys.min(), ys.max())
    print("brand selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("logo"); s.add_argument("out"); s.add_argument("--wide", action="store_true"); s.add_argument("--h", type=int, default=400)
    s.add_argument("--color"); s.add_argument("--tagline", action="store_true")
    s = sp.add_parser("lowerthird"); s.add_argument("name"); s.add_argument("role"); s.add_argument("out")
    s.add_argument("--w", type=int, default=1920); s.add_argument("--h", type=int, default=1080)
    sp.add_parser("selftest")
    o = ap.parse_args()
    if o.cmd == "logo":
        logo(o.h, o.wide, o.color, tagline=o.tagline).save(o.out); print(o.out)
    elif o.cmd == "lowerthird":
        lowerthird(o.name, o.role, o.w, o.h).save(o.out); print(o.out)
    else:
        selftest()
