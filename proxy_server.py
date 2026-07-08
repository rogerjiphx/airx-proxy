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
    -> {"z":10,"x":183,"y":409,"grid":33,"heights":[[...33 rows of 33 floats...]]}
"""

from flask import Flask, jsonify, request
from PIL import Image
import numpy as np
import requests
import io

app = Flask(__name__)

TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"

# simple in-memory cache so repeated requests for the same tile are instant
_cache = {}
CACHE_MAX = 2000


def decode_terrarium(png_bytes):
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    arr = np.asarray(img, dtype=np.float64)  # 256x256x3
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    return (r * 256.0 + g + b / 256.0) - 32768.0  # meters


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

    payload = {
        "z": z, "x": x, "y": y, "grid": grid,
        "heights": [[round(float(v), 1) for v in row] for row in small],
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
