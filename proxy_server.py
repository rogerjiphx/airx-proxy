"""
proxy_server.py — Elevation tile proxy for the AirX-style Roblox flight sim.

WHY THIS EXISTS:
Roblox's HttpService can fetch URLs but Luau cannot decode PNG images.
Elevation data on the web is served as "terrain-RGB" PNG tiles (elevation
encoded in pixel colors). This tiny server sits between Roblox and the
tile source: it fetches the PNG, decodes it, downsamples it to a small
grid, and returns plain JSON that Luau can parse in one line.

Data source: AWS Open Data "Terrain Tiles" (Terrarium format) — free,
no API key, global coverage.
Encoding: elevation_meters = (R * 256 + G + B / 256) - 32768

RUN LOCALLY:
    pip install flask requests pillow numpy
    python proxy_server.py
    # serves on http://YOUR_LAN_IP:8020

DEPLOY (so Roblox game servers can reach it):
    Roblox servers can't reach your localhost — deploy this to any free
    host (Render, Railway, Fly.io free tiers all work) and put the public
    URL in Config.lua. For Studio playtesting only, you can use a tunnel
    like `ssh -R` port forwarding or a service that gives your local
    server a public URL.

ENDPOINT:
    GET /elevation/<z>/<x>/<y>?grid=33
    -> {"z":10,"x":183,"y":409,"grid":33,
        "heights":[[...33 rows of 33 floats...]],
        "colors":[[...33 rows of 33 "rrggbb" hex strings...]] (or null)}

    The colors grid is sampled from an OpenStreetMap raster tile at the
    SAME z/x/y (it uses the identical slippy-map addressing scheme), so
    ground tiles can be colored to roughly match real land use/roads
    instead of a fake elevation-based color ramp. If that fetch fails,
    "colors" is null and the caller should fall back to its own coloring.

    NOTE: tile.openstreetmap.org is OSM's free demo tile server, meant for
    light/interactive use (see https://operations.osmfoundation.org/policies/tiles/).
    Fine for solo testing; for a real published game with real traffic,
    switch COLOR_TILE_URL to a paid provider (Mapbox/MapTiler/Stadia) with
    an API key instead.
"""

from flask import Flask, jsonify, request
from PIL import Image
import numpy as np
import requests
import io

app = Flask(__name__)

TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
COLOR_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
COLOR_TILE_HEADERS = {"User-Agent": "airx-replica-hobby-project (personal/educational use)"}

# simple in-memory cache so repeated requests for the same tile are instant
_cache = {}
CACHE_MAX = 2000


def decode_terrarium(png_bytes):
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    arr = np.asarray(img, dtype=np.float64)  # 256x256x3
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    return (r * 256.0 + g + b / 256.0) - 32768.0  # meters


def decode_color_grid(png_bytes, grid):
    """Downsample a 256x256 map tile to grid x grid average hex colors,
    aligned the same way as decode_terrarium's height grid."""
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    arr = np.asarray(img, dtype=np.float64)  # 256x256x3
    size = arr.shape[0]
    edges = np.linspace(0, size, grid + 1).astype(int)

    colors = []
    for i in range(grid):
        r0, r1 = edges[i], max(edges[i] + 1, edges[i + 1])
        row = []
        for j in range(grid):
            c0, c1 = edges[j], max(edges[j] + 1, edges[j + 1])
            mean = arr[r0:r1, c0:c1].reshape(-1, 3).mean(axis=0)
            row.append("%02x%02x%02x" % (int(mean[0]), int(mean[1]), int(mean[2])))
        colors.append(row)
    return colors


@app.route("/elevation/<int:z>/<int:x>/<int:y>")
def elevation(z, x, y):
    grid = min(max(int(request.args.get("grid", 33)), 2), 65)

    key = (z, x, y, grid)
    if key in _cache:
        return jsonify(_cache[key])

    resp = requests.get(TILE_URL.format(z=z, x=x, y=y), timeout=15)
    if resp.status_code != 200:
        return jsonify({"error": f"tile fetch failed: {resp.status_code}"}), 502

    heights = decode_terrarium(resp.content)  # 256x256

    # downsample to grid x grid by striding (fast, good enough for terrain)
    idx = np.linspace(0, 255, grid).astype(int)
    small = heights[np.ix_(idx, idx)]

    colors = None
    try:
        color_resp = requests.get(
            COLOR_TILE_URL.format(z=z, x=x, y=y), timeout=10, headers=COLOR_TILE_HEADERS)
        if color_resp.status_code == 200:
            colors = decode_color_grid(color_resp.content, grid)
    except requests.RequestException:
        colors = None  # ground still renders fine without real colors

    payload = {
        "z": z, "x": x, "y": y, "grid": grid,
        "heights": [[round(float(v), 1) for v in row] for row in small],
        "colors": colors,
    }

    if len(_cache) > CACHE_MAX:
        _cache.clear()
    _cache[key] = payload
    return jsonify(payload)


@app.route("/health")
def health():
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8020)
