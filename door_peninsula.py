# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "matplotlib", "pillow", "rasterio", "requests", "rich"]
# ///
"""Door County Peninsula terrain wallpaper.

Downloads USGS 3DEP 1/3 arc-second elevation data automatically and caches it.
Horizontal E-W profile slices reveal the Niagara Escarpment and lake margins.

Usage:
    uv run drumlin_wallpaper.py [-o out/] [--cache .cache/]
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge as rio_merge
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from rich.console import Console
from scipy import ndimage as ndi

FRAME_W, FRAME_H = 2560, 1440

# Larger map, lower resolution
BBOX = (-88.3, 44.3, -86.8, 45.5)
# Smaller map, higher resolution
# BBOX = (-88, 44.7, -86.8, 45.5)

SEA_LEVEL = 177.0  # Lake Michigan / Green Bay surface elevation (m)
COLORS = [(0, "#090e0a"), (0, "#e8f8ee"), (0.45, "#4ab870"), (1.0, "#1b4d2e")]
RISE_ROWS = 1.0
CAPTION_TITLE = "DOOR PENINSULA"
CAPTION_SUB = "The Cape Cod of the Midwest   ·   USGS 3DEP 1/3 arc-second"
TEXT_COLOR = "#a0c8b0"
DEM_SHORT_SIDE = 3000  # output pixel budget (resampled from native ~10m source)
S3_ROOT = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/current"


# --------------------------------------------------------------------------
# 1. Download / cache DEM
# --------------------------------------------------------------------------
def _tile_names(bbox) -> list[tuple[int, int]]:
    """Return (lat, lon) NW-corner pairs for every 1°×1° 3DEP tile that
    intersects *bbox*. lat/lon are positive integers (e.g. 45, 88 → n45w088)."""
    west, south, east, north = bbox
    lats = range(math.ceil(south), math.ceil(north) + 1)
    lons = range(math.ceil(abs(east)), math.ceil(abs(west)) + 1)
    return [(lat, lon) for lat in lats for lon in lons]


def _download_tile(lat: int, lon: int, cache_dir: Path) -> Path:
    name = f"n{lat:02d}w{lon:03d}"
    cached = cache_dir / f"{name}.tif"
    if not cached.exists():
        url = f"{S3_ROOT}/{name}/USGS_13_{name}.tif"
        resp = requests.get(url, timeout=(30, 600), stream=True)
        resp.raise_for_status()
        with cached.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    return cached


def get_elevation(bbox, cache_dir: Path, px_w: int = 0, px_h: int = 0) -> np.ndarray:
    west, south, east, north = bbox
    if not px_w or not px_h:
        px_w, px_h = _bbox_pixels(bbox)
    bbox_tag = f"{west}_{south}_{east}_{north}".replace("-", "m").replace(".", "p")
    cached = cache_dir / f"dem_{bbox_tag}_{px_w}x{px_h}.npy"
    if cached.exists():
        return np.load(cached)

    tile_paths = [_download_tile(lat, lon, cache_dir)
                  for lat, lon in _tile_names(bbox)]
    datasets = [rasterio.open(p) for p in tile_paths]
    nodata = datasets[0].nodata
    try:
        merged, _ = rio_merge(
            datasets,
            bounds=(west, south, east, north),
            res=((east - west) / px_w, (north - south) / px_h),
            resampling=Resampling.bilinear,
        )
    finally:
        for ds in datasets:
            ds.close()

    elev = merged[0].astype(float)
    if nodata is not None:
        elev[elev == nodata] = np.nan
    np.save(cached, elev)
    return elev


# --------------------------------------------------------------------------
# 2. Rendering
# --------------------------------------------------------------------------
def _bbox_pixels(
        bbox: tuple[float, float, float, float],
        short_side: int = DEM_SHORT_SIDE
    ) -> tuple[int, int]:
    west, south, east, north = bbox
    lat_mid = (south + north) / 2
    ground_w = (east - west) * np.cos(np.radians(lat_mid))
    ground_h = north - south
    aspect = ground_w / ground_h
    if aspect >= 1:
        return int(short_side * aspect), short_side
    else:
        return short_side, int(short_side / aspect)


def _layout(elev: np.ndarray, height_frac=0.95, dx=-0.25, dy=0):
    h, w = elev.shape
    scale = FRAME_H * height_frac / h
    ox = (FRAME_W - w * scale) / 2 + FRAME_W * dx
    oy = (FRAME_H - h * scale) / 2 + FRAME_H * dy
    return scale, ox, oy


def _new_canvas():
    fig = plt.figure(figsize=(FRAME_W / 100, FRAME_H / 100), dpi=100, facecolor="k")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("k")
    ax.set_xlim(0, FRAME_W)
    ax.set_ylim(FRAME_H, 0)
    ax.axis("off")
    return fig, ax


def _finish(fig, ax, path: Path, black_cutoff=16 / 255):
    # Bloom pass — line art only, no text
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(float) / 255
    plt.close(fig)

    glow = (0.55 * ndi.gaussian_filter(img, (5, 5, 0))
            + 0.35 * ndi.gaussian_filter(img, (22, 22, 0)))
    out = np.clip(1 - (1 - img) * (1 - glow), 0, 1)
    out[out.max(-1) < black_cutoff] = 0

    # Composite text on top of bloomed result
    fig2, ax2 = _new_canvas()
    ax2.imshow(out, extent=[0, FRAME_W, FRAME_H, 0], aspect="auto", interpolation="none")
    ax2.text(FRAME_W - 110, FRAME_H - 100, CAPTION_SUB, color=TEXT_COLOR, fontsize=13,
             ha="right", va="bottom", zorder=10_000)
    ax2.text(FRAME_W - 110, FRAME_H - 124, CAPTION_TITLE, color=TEXT_COLOR, fontsize=60,
             ha="right", va="bottom", fontweight="bold", fontfamily="serif", zorder=10_000)
    fig2.canvas.draw()
    u8 = np.asarray(fig2.canvas.buffer_rgba())[..., :3]
    plt.close(fig2)
    Image.fromarray(u8).save(path, optimize=True)


def _make_cmap(e_lo: float, e_hi: float) -> LinearSegmentedColormap:
    t_sea = float(np.clip((SEA_LEVEL - e_lo) / (e_hi - e_lo), 0.02, 0.96))
    water = COLORS[0][1]
    land = [(t_sea + (1.0 - t_sea) * t, c) for t, c in COLORS[1:]]
    stops = [(0.0, water), (t_sea, water), *land]
    return LinearSegmentedColormap.from_list("terrain", stops)


def render_profiles(elev: np.ndarray, path: Path, row_spacing_px=20):
    filled = np.where(np.isnan(elev), np.nanmedian(elev), elev)
    smooth = ndi.gaussian_filter(filled, sigma=4)

    h, w = smooth.shape
    scale, ox, oy = _layout(smooth)
    xs = ox + np.arange(w) * scale
    n_rows = max(50, int(h * scale / row_spacing_px))

    e_lo, e_hi = np.nanpercentile(smooth, [2, 98])
    erange = e_hi - e_lo
    cmap = _make_cmap(e_lo, e_hi)
    amp = h * scale / n_rows * RISE_ROWS / erange  # px per metre

    fig, ax = _new_canvas()
    for k, r in enumerate(np.linspace(0, h - 1, n_rows).astype(int)):
        row = np.clip(smooth[r], e_lo, e_hi)
        base = oy + r * scale
        y = base - (row - e_lo) * amp  # higher elevation -> up on screen

        ax.fill_between(xs, y + 1.5, FRAME_H, color="k", lw=0, zorder=2 * k)

        t = (row - e_lo) / erange
        colors = cmap(t)
        pts = np.array([xs, y]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        seg_colors = (colors[:-1] + colors[1:]) / 2
        ax.add_collection(LineCollection(segs, colors=seg_colors, lw=1.3,
                                         capstyle="round", zorder=2 * k + 1))

    _finish(fig, ax, path)


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-o", "--outdir", type=Path, default=Path(".") / "gallery")
    p.add_argument("--cache", type=Path, default=Path(".cache"),
                   help="Directory for cached DEM tiles")
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)

    console = Console()

    with console.status("[bold green]Downloading 3DEP source tiles...", spinner="dots"):
        elev = get_elevation(BBOX, args.cache)
    out_path = args.outdir / Path(__file__).name.replace(".py", ".png")
    with console.status("[bold green]Rendering profiles...", spinner="dots"):
        render_profiles(elev, out_path)
    console.print(f"[bold green]Saved to {out_path}")


if __name__ == "__main__":
    main()
