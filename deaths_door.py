# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "matplotlib", "pillow", "rasterio", "requests", "rich"]
# ///
"""Death's Door (Port des Morts) bathymetry wallpaper.

Downloads NOAA Great Lakes Lake Michigan bathymetry data automatically and caches it.
Horizontal E-W profile slices reveal the underwater topography of the strait.

Usage:
    uv run deaths_door.py [-o out/] [--cache .cache/]
"""
from __future__ import annotations

import argparse
import json
import re
import tarfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
import rasterio
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from rich.console import Console
from scipy import ndimage as ndi

BATHY_URL = (
    "https://www.ngdc.noaa.gov/mgg/greatlakes/michigan/data/geotiff/michigan_lld.geotiff.tar.gz"
)

WISCSHIP_URL = "https://www.wisconsinshipwrecks.org"
_HEADERS = {"User-Agent": "black-earth-wallpaper/1.0 (educational/non-commercial)"}

FRAME_W, FRAME_H = 3840, 2160
BBOX = (-87.3, 45.2, -86.7, 45.47)
CAPTION_TITLE = "PORT DES MORTS"
CAPTION_SUB = "Beware the shoals   ·   Wisconsin Historical Society Shipwreck Logs   ·   NOAA Great Lakes Bathymetry"
TEXT_COLOR = "#4a6670"
WRECK_COLOR = "#ffe080"
CMAP = LinearSegmentedColormap.from_list("depth", ["#04202e", "#d8eff5"])


def get_bathymetry(bbox, cache_dir: Path) -> np.ndarray:
    west, south, east, north = bbox
    bbox_tag = f"{west}_{south}_{east}_{north}".replace("-", "m").replace(".", "p")
    clipped = cache_dir / f"bathy_{bbox_tag}.npy"
    if clipped.exists():
        return np.load(clipped)

    tiff_path = cache_dir / "michigan_lld.tif"
    if not tiff_path.exists():
        tarball = cache_dir / "michigan_lld.geotiff.tar.gz"
        if not tarball.exists():
            resp = requests.get(BATHY_URL, timeout=300, stream=True)
            resp.raise_for_status()
            with open(tarball, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
        with tarfile.open(tarball, "r:gz") as tf:
            member = next(m for m in tf.getmembers() if m.name.endswith((".tif", ".tiff")))
            member.name = tiff_path.name
            tf.extract(member, cache_dir)

    with rasterio.open(tiff_path) as src:
        window = rasterio.windows.from_bounds(west, south, east, north, src.transform)
        data = src.read(1, window=window).astype(float)
        if src.nodata is not None:
            data[data == src.nodata] = np.nan

    np.save(clipped, data)
    return data


def _dms_to_dec(hem: str, deg: str, mins: str) -> float:
    val = int(deg) + float(mins) / 60
    return -val if hem in ("S", "W") else val


def _scrape_wrecks(cache_file: Path) -> None:
    from rich.progress import track

    resp = requests.get(f"{WISCSHIP_URL}/vessel", headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    ids = list(dict.fromkeys(re.findall(r"/Vessel/Details/(\d+)", resp.text)))

    coord_re = re.compile(
        r"Latitude[^NS]*([NS])\s*(\d+)&deg;&nbsp;([\d.]+)'.*?"
        r"Longitude[^EW]*([EW])\s*(\d+)&deg;&nbsp;([\d.]+)'",
        re.DOTALL,
    )
    wrecks = []
    for vid in track(ids, description="Scraping Wisconsin Shipwrecks..."):
        time.sleep(0.5)
        try:
            r = requests.get(
                f"{WISCSHIP_URL}/Vessel/Details/{vid}",
                headers=_HEADERS,
                timeout=30,
            )
            r.raise_for_status()
            m = coord_re.search(r.text)
            if m:
                lat = _dms_to_dec(m.group(1), m.group(2), m.group(3))
                lon = _dms_to_dec(m.group(4), m.group(5), m.group(6))
                wrecks.append({"id": int(vid), "lat": lat, "lon": lon})
        except Exception:
            pass

    cache_file.write_text(json.dumps(wrecks, indent=2))


def get_wrecks(bbox, cache_dir: Path) -> np.ndarray:
    """Return (N, 2) array of [lon, lat] within bbox, scraped from wisconsinshipwrecks.org."""
    west, south, east, north = bbox
    cache_file = cache_dir / "wisconsin_shipwrecks.json"
    if not cache_file.exists():
        _scrape_wrecks(cache_file)
    wrecks = json.loads(cache_file.read_text())
    pts = [
        [w["lon"], w["lat"]] for w in wrecks
        if west <= w["lon"] <= east and south <= w["lat"] <= north
    ]
    return np.array(pts) if pts else np.empty((0, 2))


def _layout(depth: np.ndarray, height_frac=0.95, dx=0, dy=0):
    h, w = depth.shape
    west, south, east, north = BBOX
    lat_mid = (south + north) / 2
    # The GeoTIFF stores equal degrees per pixel in both axes.  At 45°N a
    # degree of longitude spans only cos(lat) as much ground as a degree of
    # latitude, so we compress the x-axis by that factor to get true proportions.
    y_scale = FRAME_H * height_frac / h
    x_scale = y_scale * np.cos(np.radians(lat_mid))
    ox = (FRAME_W - w * x_scale) / 2 + FRAME_W * dx
    oy = (FRAME_H - h * y_scale) / 2 + FRAME_H * dy
    return x_scale, y_scale, ox, oy


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

    glow = (0.55 * ndi.gaussian_filter(img, (5, 5, 0))
            + 0.35 * ndi.gaussian_filter(img, (22, 22, 0)))
    out = np.clip(1 - (1 - img) * (1 - glow), 0, 1)
    out[out.max(-1) < black_cutoff] = 0  # OLED: force true #000

    # Composite text on top of the bloomed result
    fig2, ax2 = _new_canvas()
    ax2.imshow(out, extent=[0, FRAME_W, FRAME_H, 0], aspect="auto", interpolation="none")
    ax2.text(FRAME_W - 110, FRAME_H - 100, CAPTION_SUB, color=WRECK_COLOR, fontsize=13,
             ha="right", va="bottom", zorder=10_000)
    ax2.text(FRAME_W - 110, FRAME_H - 124, CAPTION_TITLE, color=WRECK_COLOR, fontsize=80,
             ha="right", va="bottom", fontweight="bold", fontfamily="serif", zorder=10_000)
    fig2.canvas.draw()
    u8 = np.asarray(fig2.canvas.buffer_rgba())[..., :3]
    plt.close(fig2)
    Image.fromarray(u8).save(path, optimize=True)


def render_profiles(depth: np.ndarray, path: Path, wrecks: np.ndarray | None = None, n_rows=88, dip_rows=4.0):
    h, w = depth.shape
    x_scale, y_scale, ox, oy = _layout(depth)
    xs = ox + np.arange(w) * x_scale
    dmax = np.nanmax(np.abs(depth))
    amp = h * y_scale / n_rows * dip_rows / dmax  # px per metre

    fig, ax = _new_canvas()
    for k, r in enumerate(np.linspace(0, h - 1, n_rows).astype(int)):
        d = depth[r]
        wet = ~np.isnan(d)
        if wet.sum() < 3:
            continue
        base = oy + r * y_scale
        y = base - d * amp
        edges = np.flatnonzero(np.diff(np.r_[0, wet.astype(int), 0]))
        for s0, s1 in zip(edges[::2], edges[1::2]):
            xx, yy = xs[s0:s1], y[s0:s1]
            ax.fill_between(xx, yy + 1.5, FRAME_H, color="k", lw=0, zorder=2 * k)
            t = np.abs(d[s0:s1]) / dmax
            range = t.max() - t.min()
            colors = CMAP((t.max() - t) / range)  # shallow → near-white, deep → dark navy
            colors[d[s0:s1] >= 0] = (0, 0, 0, 1)  # land / surface → black
            pts = np.array([xx, yy]).T.reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            seg_colors = (colors[:-1] + colors[1:]) / 2
            ax.add_collection(LineCollection(segs, colors=seg_colors, lw=1.7,
                                             capstyle="round", zorder=2 * k + 1))
    if wrecks is not None and len(wrecks):
        west, south, east, north = BBOX
        wx = ox + (wrecks[:, 0] - west) / (east - west) * w * x_scale
        wr = (north - wrecks[:, 1]) / (north - south) * (h - 1)
        wy = oy + wr * y_scale
        ax.scatter(wx, wy, s=180, facecolors="none", edgecolors=WRECK_COLOR,
                   linewidths=1.2, zorder=2 * n_rows + 1)
        ax.scatter(wx, wy, s=30, color=WRECK_COLOR, linewidths=0, zorder=2 * n_rows + 2)

    _finish(fig, ax, path)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).parent
    p.add_argument("-o", "--outdir", type=Path, default=here / "gallery")
    p.add_argument("--cache", type=Path, default=here / ".cache",
                   help="Directory for cached bathymetry data")
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)

    console = Console()

    with console.status("[bold green]Downloading bathymetry data...", spinner="dots"):
        depth = get_bathymetry(BBOX, args.cache)
    wrecks = get_wrecks(BBOX, args.cache)
    out_path = args.outdir / Path(__file__).name.replace(".py", ".png")
    with console.status("[bold green]Rendering profiles...", spinner="dots"):
        render_profiles(depth, out_path, wrecks=wrecks)
    console.print(f"[bold green]Saved to {out_path}")


if __name__ == "__main__":
    main()
