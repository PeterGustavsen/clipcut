"""CapCut filter LUTs -> ffmpeg .cube files.

CapCut filters are GPUImage-style lookup PNGs: 512x512 = 8x8 tiles of 64x64, tile index (row-major) = blue level,
x inside the tile = red, y = green (same layout as the app's Resources/models/colorcard/lookup.png identity map).
`png_to_cube(png, cube)` writes a LUT_3D_SIZE 64 cube that ffmpeg's lut3d applies 1:1; `blend` scales it toward identity
(= CapCut's filter intensity slider).
"""
import sys
from pathlib import Path

import cv2
import numpy as np


def png_to_cube(png, cube, intensity=1.0):
    img = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"cannot read {png}")
    img = img[:, :, :3][:, :, ::-1].astype(np.float32) / 255  # BGR -> RGB
    h, w = img.shape[:2]
    assert h == w == 512, f"{png}: expected 512x512, got {w}x{h}"
    n = 64
    lut = np.zeros((n, n, n, 3), np.float32)  # [b, g, r]
    for b in range(n):
        ty, tx = (b // 8) * n, (b % 8) * n
        lut[b] = img[ty:ty + n, tx:tx + n]  # rows = green, cols = red
    if intensity != 1:
        ident = np.stack(np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")[::-1], -1) / (n - 1)  # [b,g,r] -> (r,g,b)
        lut = ident + (lut - ident) * intensity
    lines = ["# from CapCut filter " + Path(png).name, f"LUT_3D_SIZE {n}", "DOMAIN_MIN 0 0 0", "DOMAIN_MAX 1 1 1"]
    lines += [f"{r:.5f} {g:.5f} {b:.5f}" for r, g, b in lut.reshape(-1, 3)]  # red fastest, then green, then blue
    Path(cube).write_text("\n".join(lines) + "\n", encoding="ascii")
    return cube


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit("usage: luts.py IN.png OUT.cube [intensity]")
    png_to_cube(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 1.0)
    print("wrote", sys.argv[2])
