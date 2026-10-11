"""Lake Mendota bathymetry wallpaper.

The dataset for this wallpaper can be found with the following DOI.
Contained within is a shapefile containing the bathymetry polygons.

DOI: https://doi.org/10.6073/pasta/d2169aa8f69918a5b5644f8248345807

Usage:
    uv run lake_mendota.py path/to/yld_bathymetry.shp -o out/
"""

# /// script
# dependencies = [
#   "matplotlib==3.11.2",
#   "numpy==2.5.4",
#   "rich==15.0.0",
#   "pyshp==3.1.6",
#   "pillow==12.3.0",
#   "scipy==1.18.1",
# ]
# ///

from __future__ import annotations

import argparse
from pathlib import Path
from rich.console import Console

import matplotlib
from matplotlib.collections import LineCollection

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shapefile
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

FT_TO_M = 0.3048
FRAME_W, FRAME_H = 3840, 2160

CMAP = LinearSegmentedColormap.from_list(
    "deep",
    [(0, "#0b2a3a"), (0.35, "#11677e"), (0.7, "#3cc4d6"), (1, "#e8fbff")],
)
CAPTION_TITLE = "LAKE MENDOTA"
CAPTION_SUB = "NTL-LTER 10-ft survey contours   ·   max 25 m"


# --------------------------------------------------------------------------
# 1. Shapefile -> grid of depth bands
# --------------------------------------------------------------------------
def rasterize_bands(shp_path: Path, cell: float = 5.0, pad: float = 100.0):
    """Return (band, cell) where band[y, x] is the band's upper depth in ft,
    or -1 on land. Rings are XOR-ed so holes (shoals, islands) cut out."""
    r = shapefile.Reader(str(shp_path))
    x0, y0, x1, y1 = r.bbox
    x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
    w, h = int((x1 - x0) / cell), int((y1 - y0) / cell)

    def ring_mask(pts):
        im = Image.new("1", (w, h), 0)
        ImageDraw.Draw(im).polygon(
            [((x - x0) / cell, (y1 - y) / cell) for x, y in pts], fill=1
        )
        return np.array(im, bool)

    band = np.full((h, w), -1, int)
    for shape, rec in zip(r.shapes(), r.records()):
        mask = np.zeros((h, w), bool)
        parts = [*shape.parts, len(shape.points)]
        for a, b in zip(parts[:-1], parts[1:]):
            mask ^= ring_mask(shape.points[a:b])
        band[mask] = rec["DEPTH"]
    return band, cell


# --------------------------------------------------------------------------
# 2. Depth bands -> smooth depth surface (metres)
# --------------------------------------------------------------------------
def depth_surface(band: np.ndarray, smooth_cells: float = 7.0) -> np.ndarray:
    """Interpolate within each band by relative distance to the shallower
    and deeper band edges, then apply a masked Gaussian smooth."""
    lake = band >= 0
    depth_ft = np.full(band.shape, np.nan)
    for b in np.unique(band[lake]):
        inside = band == b
        d_up = ndi.distance_transform_edt(~(band < b))  # to shallower / shore
        if (band > b).any():
            d_dn = ndi.distance_transform_edt(~(band > b))  # to deeper
            depth_ft[inside] = b + 10 * d_up[inside] / (d_up[inside] + d_dn[inside])
        else:  # deepest band: gentle rise into the hole
            depth_ft[inside] = b + 3 * d_up[inside] / d_up[inside].max()

    num = ndi.gaussian_filter(np.where(lake, depth_ft, 0), smooth_cells)
    den = ndi.gaussian_filter(lake.astype(float), smooth_cells)
    out = num / np.maximum(den, 1e-6) * FT_TO_M
    out[~lake] = np.nan
    return out


# --------------------------------------------------------------------------
# 3. Rendering
# --------------------------------------------------------------------------
def _layout(depth: np.ndarray, height_frac=0.64, dx=0, dy=-0.03):
    h, w = depth.shape
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
    # Bloom pass — profiles only, no text
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(float) / 255
    plt.close(fig)

    glow = 0.55 * ndi.gaussian_filter(img, (5, 5, 0)) + 0.35 * ndi.gaussian_filter(img, (22, 22, 0))
    out = np.clip(1 - (1 - img) * (1 - glow), 0, 1)
    out[out.max(-1) < black_cutoff] = 0  # OLED: force true #000

    # Composite text on top of the bloomed result
    fig2, ax2 = _new_canvas()
    ax2.imshow(out, extent=[0, FRAME_W, FRAME_H, 0], aspect="auto", interpolation="none")
    ax2.text(FRAME_W - 110, FRAME_H - 100, CAPTION_SUB, color="#4a6670", fontsize=13,
             ha="right", va="bottom", zorder=10_000)
    ax2.text(FRAME_W - 110, FRAME_H - 124, CAPTION_TITLE, color="#4a6670", fontsize=80,
             ha="right", va="bottom", fontweight="bold", fontfamily="serif", zorder=10_000)
    fig2.canvas.draw()
    u8 = np.asarray(fig2.canvas.buffer_rgba())[..., :3]
    plt.close(fig2)
    Image.fromarray(u8).save(path, optimize=True)


def render_profiles(depth, path: Path, n_rows=88, dip_rows=4.0):
    h, w = depth.shape
    scale, ox, oy = _layout(depth)
    xs = ox + np.arange(w) * scale
    dmax = np.nanmax(depth)
    amp = h * scale / n_rows * dip_rows / dmax  # px per metre

    fig, ax = _new_canvas()
    for k, r in enumerate(np.linspace(0, h - 1, n_rows).astype(int)):
        d = depth[r]
        wet = ~np.isnan(d)
        if wet.sum() < 3:
            continue
        base = oy + r * scale
        y = base + d * amp
        # contiguous wet runs (a row can cross the lake more than once)
        edges = np.flatnonzero(np.diff(np.r_[0, wet.astype(int), 0]))
        for s0, s1 in zip(edges[::2], edges[1::2]):
            xx, yy = xs[s0:s1], y[s0:s1]
            # painter's algorithm, north -> south: black fill below each
            # line hides the rows behind it
            ax.fill_between(xx, yy + 1.5, FRAME_H, color="k", lw=0, zorder=2 * k)
            t = d[s0:s1] / dmax  # per-point depth, normalized 0→1
            colors = CMAP(1.0 - 0.3 * t)  # shallow→near-white, deep→cyan
            pts = np.array([xx, yy]).T.reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            seg_colors = (colors[:-1] + colors[1:]) / 2  # average over each segment
            ax.add_collection(LineCollection(segs, colors=seg_colors, lw=1.7,
                                             capstyle="round", zorder=2 * k + 1))
    _finish(fig, ax, path)


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("shapefile", type=Path, help="yld_bathymetry.shp")
    p.add_argument("-o", "--outdir", type=Path, default=Path(".") / "out")
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    console = Console()

    with console.status("[bold green]Creating a raster image from the shapefile...", spinner="dots"):
        band, _ = rasterize_bands(args.shapefile)
    with console.status("[bold green]Interpolating depth surface from rasterized bands...", spinner="dots"):
        depth = depth_surface(band)
    out_path = args.outdir / "mendota_profiles.png"
    with console.status("[bold green]Rendering profiles...", spinner="dots"):
        render_profiles(depth, out_path)
    console.print(f"[bold green]Saved to {out_path}")


if __name__ == "__main__":
    main()
