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

ENDPOINTS:
    GET /elevation/<z>/<x>/<y>?grid=33
    -> {"z":10,"x":183,"y":409,"grid":33,
        "heights":[[...33 rows of 33 floats, meters...]],
        "colors":[[...33 rows of 33 "rrggbb" hex strings...]] (or null)}

    "colors" is sampled from Esri World Imagery — real satellite/aerial
    photography, not a cartographic map. Deliberately switched away from
    OpenStreetMap's standard basemap tiles: that style draws highways as
    bright stylized lines ON TOP of the land coloring, so any grid cell
    that happened to cross a drawn road picked up the road's rendering
    color instead of real ground color (visible as odd colored streaks
    cutting across terrain that has no actual road there). Satellite
    imagery has no such overlay — what you sample is what the ground
    actually looks like. "colors" is null only if that fetch fails; the
    caller should fall back to its own elevation-based coloring.

    GET /features?south=<lat>&west=<lon>&north=<lat>&east=<lon>
    -> {"bounds": {...}, "buildings": [{"points":[[lat,lon],...],"heightM":9.0}],
        "roads": [{"points":[[lat,lon],...],"class":"residential"}],
        "taxiways": [{"points":[[lat,lon],...],"kind":"taxiway"}]}

    Real OpenStreetMap building footprints, road centerlines, and airport
    taxiway/apron paths within a bounding box, via the Overpass API.
    Bounding boxes are snapped to a coarse grid and cached indefinitely
    in-memory, because Overpass's free public instance has strict rate
    limits (~10k req/day, 2 concurrent slots) — nowhere near enough for
    1:1 per-player-movement queries. Callers should query a modest area
    at once (e.g. once per near-ring refresh, not per tile) rather than
    firing many small bbox requests.

    NOTE: Esri World Imagery (colors) and overpass-api.de (features) are
    both free public services with fair-use limits, not meant for heavy
    production traffic (see https://dev.overpass-api.de/overpass-doc/en/preface/commons.html
    for Overpass; Esri's World Imagery is free for non-commercial/light
    use but has no published hard rate limit the way OSM tiles do).
    Fine for solo testing; a real published game with real traffic needs
    a paid imagery provider and a self-hosted Overpass instance instead.
"""

from flask import Flask, Response, jsonify, request
import json
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from requests.adapters import HTTPAdapter
import numpy as np
import requests
import io
import math
import colorsys

app = Flask(__name__)

# One shared Session = TLS connection reuse. Without this, EVERY tile/color/
# Overpass fetch performed a full fresh TLS handshake — dozens per batch
# request — and on Render's free-tier CPU (0.1 vCPU) handshake crypto alone
# blew past gunicorn's request timeout under load (measured live: 16-tile
# batches dying at exactly the ~30s worker timeout). With pooling, each
# worker pays the handshake once per host and reuses the connection.
_http = requests.Session()
_http.mount("https://", HTTPAdapter(pool_connections=8, pool_maxsize=32))

TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
# Esri World Imagery — real satellite/aerial photography, free, no API key.
# NOTE: Esri's tile path order is z/y/x (row before column), NOT the usual
# z/x/y — easy to get backwards, so it's spelled out explicitly here.
COLOR_TILE_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
COLOR_TILE_HEADERS = {"User-Agent": "airx-replica-hobby-project (personal/educational use)"}
# Multiple independent public Overpass mirrors, RACED IN PARALLEL (see
# fetch_overpass_raced below) rather than tried one after another — a
# dense-city query can legitimately take 10-25s on a healthy mirror, and
# serial fallback (try #1, THEN #2, THEN #3, each paying its own timeout)
# stacks those delays into 30-40+s, which lands right on Render's default
# 30s gunicorn worker timeout and was the direct cause of intermittent 502s
# even when a mirror would have succeeded if given the time. Racing bounds
# total latency to whichever mirror finishes first instead of the sum.
#
# overpass.osm.ch was dropped from this list after being confirmed
# (2026-07-09) to return HTTP 200 with an EMPTY elements array for a bbox
# known to contain thousands of buildings — a stale/broken replica, not an
# outage. That's worse than an honest failure: a wrong-but-"successful"
# empty response would get cached indefinitely by _features_cache below,
# permanently hiding real buildings/roads for that area.
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
OVERPASS_HEADERS = {"User-Agent": "airx-replica-hobby-project (personal/educational use)"}
MAJOR_HIGHWAYS = "motorway|trunk|primary|secondary|tertiary|residential|unclassified"
# The "majorroads" feature type: arterial classes only, no residential/
# unclassified. Exists so the Roblox client can query ROADS across a much
# larger radius than buildings (the AirX look: road network visible across
# the whole terrain, buildings only near the player) without the query
# exploding — a 12km metro bbox of ALL road classes is thousands of ways,
# but arterials alone stay bounded.
MAJOR_ONLY_HIGHWAYS = "motorway|trunk|primary|secondary|tertiary"

# Per-category caps on the Overpass query itself (via "out geom N;"), not
# just on what Roblox ends up keeping. Confirmed directly (2026-07-09):
# dense downtown Phoenix, uncapped, returned 4.9MB and took 13-28s from a
# single mirror; the SAME bbox with these caps returned 1.6MB in 3.6s.
# Roblox already re-sorts by distance and hard-caps what it keeps
# (Config.MAX_BUILDINGS_LOADED/MAX_ROADS_LOADED), so fetching, JSON-
# encoding, and transferring thousands of far-away elements just to
# discard most of them client-side was pure waste — these caps are set
# comfortably above Roblox's own caps so its nearest-N selection still has
# plenty of real candidates.
BUILDING_OUT_CAP = 800
ROAD_OUT_CAP = 400
MAJORROAD_OUT_CAP = 600  # majorroads queries cover much larger bboxes
TAXIWAY_OUT_CAP = 300
LAKE_OUT_CAP = 150   # natural=water polygons
RIVER_OUT_CAP = 150  # waterway=river/canal centerlines

# In-memory caches so repeated requests are instant (and, for /features,
# so we don't hammer Overpass's rate-limited free tier).
#
# MEMORY DISCIPLINE — these two rules exist because the service OOM-crash-
# looped in production (repeated "used over 512MB" kills, 2026-07-10):
#   1. Cache values are compact JSON STRINGS, never Python object trees —
#      a dense-city /features payload is ~300KB as JSON but MEGABYTES as
#      nested Python lists (every float is a 24-byte object). Cached
#      strings are also returned directly, skipping re-serialization.
#   2. Caps are sized for 512MB SPLIT ACROSS WORKER PROCESSES, each of
#      which holds its own copy of every cache. Run 2 workers, not 3.
_cache = {}          # (z,x,y,grid) -> JSON string of one tile payload (~4KB each)
CACHE_MAX = 800
_features_cache = {} # snapped bbox+types -> JSON string (~50-500KB each)
FEATURES_CACHE_MAX = 60
FEATURES_GRID_DEG = 0.02  # ~2km; bboxes snap to this grid before querying/caching


def decode_terrarium(png_bytes):
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    arr = np.asarray(img, dtype=np.float64)  # 256x256x3
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    return (r * 256.0 + g + b / 256.0) - 32768.0  # meters


def sample_map_tile_raw(png_bytes, grid):
    """Block-average a 256x256 map tile down to grid x grid raw (r,g,b) in
    0..1, aligned the same way as decode_terrarium's height grid."""
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    arr = np.asarray(img, dtype=np.float64) / 255.0  # 256x256x3
    size = arr.shape[0]
    edges = np.linspace(0, size, grid + 1).astype(int)

    raw = [[None] * grid for _ in range(grid)]
    for i in range(grid):
        r0, r1 = edges[i], max(edges[i] + 1, edges[i + 1])
        for j in range(grid):
            c0, c1 = edges[j], max(edges[j] + 1, edges[j + 1])
            mean = arr[r0:r1, c0:c1].reshape(-1, 3).mean(axis=0)
            raw[i][j] = (float(mean[0]), float(mean[1]), float(mean[2]))
    return raw


def boosted_hex(rgb01):
    """Deliberately STYLIZED grading, not faithful satellite color: strong
    saturation push + brightness lift turns muted photographic land tones
    into vivid painterly greens/golds (matching AirX's look, per the
    user's reference screenshots — an earlier gentle 1.25x kept things
    realistic but drab; the stylized look was explicitly chosen)."""
    h, s, v = colorsys.rgb_to_hsv(*rgb01)
    s = min(1.0, s * 1.75)
    v = min(1.0, v * 1.12 + 0.06)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return "%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))


def snap_bbox(south, west, north, east, grid_deg=FEATURES_GRID_DEG):
    """Round a bbox outward to a coarse fixed grid, so nearby requests
    (e.g. successive near-ring refreshes as a plane drifts slowly) hit the
    same cache entry and the same Overpass query instead of each firing
    their own slightly-different request."""
    s = math.floor(south / grid_deg) * grid_deg
    w = math.floor(west / grid_deg) * grid_deg
    n = math.ceil(north / grid_deg) * grid_deg
    e = math.ceil(east / grid_deg) * grid_deg
    return round(s, 6), round(w, 6), round(n, 6), round(e, 6)


def estimate_building_height(tags):
    """height/building:levels tag if present, else a type-based default —
    same fallback convention used by OSMBuildings/Simple3DBuildingsV1."""
    height = tags.get("height")
    if height:
        try:
            return float("".join(c for c in height if c.isdigit() or c == "."))
        except ValueError:
            pass
    levels = tags.get("building:levels")
    if levels:
        try:
            return float(levels) * 3.0
        except ValueError:
            pass
    btype = tags.get("building", "")
    if btype in ("house", "detached", "residential", "garage", "shed", "hut"):
        return 6.0
    if btype in ("industrial", "warehouse", "hangar"):
        return 8.0
    if btype in ("skyscraper", "office", "commercial", "apartments", "retail"):
        return 24.0
    return 9.0


OVERPASS_TIMEOUT_SECONDS = 15


def _fetch_overpass_one(mirror_url, query):
    resp = _http.post(mirror_url, data={"data": query}, timeout=OVERPASS_TIMEOUT_SECONDS, headers=OVERPASS_HEADERS)
    resp.raise_for_status()
    return resp.json()


# HEDGE_DELAY_SECONDS + OVERPASS_TIMEOUT_SECONDS is the worst-case total
# latency (primary fails right at the hedge boundary, backup then gets its
# own full timeout budget) — this MUST stay safely under Render's gunicorn
# worker timeout (30s default). An earlier version used 7 + 25 = 32s,
# which exceeded it: gunicorn killed the worker mid-request, and the
# client saw a clean HTTP 500 (confirmed live, 2026-07-09) instead of
# either a fast success or an honest "all mirrors failed" 502. Verified
# ONLY against Flask's own dev server locally before that shipped, which
# has no such timeout — this class of bug only shows up under gunicorn.
# 5 + 15 = 20s worst case, leaving real margin below the 30s ceiling.
HEDGE_DELAY_SECONDS = 5


def fetch_overpass_hedged(query):
    """Try the primary mirror ALONE first. Only fall back to the backup
    mirror if the primary hasn't finished within HEDGE_DELAY_SECONDS, or if
    it fails outright. Returns (data, None) on success, (None,
    last_exception) if nothing worked.

    This replaced an earlier version that always fired BOTH mirrors in
    parallel on every single request. That fixed the original problem
    (stacked serial timeouts causing 502s) but had a real downside: it
    doubled real load on both mirrors for every query, including the
    common case where the primary was healthy and would have succeeded
    fine alone. Repeated 502s kept happening even after that fix — the
    likely explanation is Overpass's OWN rate-limiting reacting to the
    sustained doubled request volume from always-parallel racing (public
    Overpass mirrors run ~2 concurrent-slot limits; hammering both on every
    query eats into that fast). Hedging keeps the "don't wait out a dead
    mirror's full timeout" benefit while touching the backup only when the
    primary actually needs help.

    Deliberately not a `with ThreadPoolExecutor(...) as pool:` block —
    that form's shutdown(wait=True) on exit blocks until EVERY submitted
    future finishes, including ones we've already decided to stop waiting
    on. shutdown(wait=False) lets an abandoned thread finish in the
    background and simply never be read."""
    pool = ThreadPoolExecutor(max_workers=len(OVERPASS_URLS))
    last_error = None
    try:
        primary = pool.submit(_fetch_overpass_one, OVERPASS_URLS[0], query)
        done, _pending = wait([primary], timeout=HEDGE_DELAY_SECONDS)

        if primary in done:
            try:
                return primary.result(), None
            except (requests.RequestException, ValueError) as ex:
                last_error = ex
            # primary already finished (with a failure) — try the backup
            # alone, sequentially, no need to race against a resolved future
            if len(OVERPASS_URLS) < 2:
                return None, last_error
            try:
                return _fetch_overpass_one(OVERPASS_URLS[1], query), None
            except (requests.RequestException, ValueError) as ex:
                return None, ex

        # primary is still running after the hedge delay — give it a
        # backup to race against instead of waiting out its full timeout
        if len(OVERPASS_URLS) < 2:
            try:
                return primary.result(), None
            except (requests.RequestException, ValueError) as ex:
                return None, ex

        backup = pool.submit(_fetch_overpass_one, OVERPASS_URLS[1], query)
        for future in as_completed([primary, backup]):
            try:
                return future.result(), None
            except (requests.RequestException, ValueError) as ex:
                last_error = ex
                continue
        return None, last_error
    finally:
        pool.shutdown(wait=False)


ALL_FEATURE_TYPES = frozenset({"buildings", "roads", "majorroads", "taxiways", "water", "airport"})


@app.route("/features")
def features():
    try:
        south = float(request.args["south"])
        west = float(request.args["west"])
        north = float(request.args["north"])
        east = float(request.args["east"])
    except (KeyError, ValueError):
        return jsonify({"error": "south/west/north/east query params required"}), 400

    # Optional "?types=taxiways" (comma-separated) so a caller that only
    # needs ONE category doesn't force Overpass to also scan+return the
    # others. TaxiwayBuilder.lua's airport-radius query used to fetch
    # buildings+roads it never even reads (see its own code — only
    # response.taxiways is ever used) purely because this endpoint always
    # queried all three; that was wasted Overpass load on every single
    # airport visit for zero benefit. Defaults to all three (unchanged
    # behavior for CityFeatures.lua, which genuinely wants all of them).
    requested = request.args.get("types")
    if requested:
        want = {t.strip() for t in requested.split(",") if t.strip()} & ALL_FEATURE_TYPES
        if not want:
            want = ALL_FEATURE_TYPES
    else:
        want = ALL_FEATURE_TYPES

    bbox_key = snap_bbox(south, west, north, east)
    cache_key = bbox_key + (tuple(sorted(want)),)
    if cache_key in _features_cache:
        return Response(_features_cache[cache_key], mimetype="application/json")

    s, w, n, e = bbox_key
    # Each requested way-type gets its OWN "out geom N;" so the cap applies
    # per category (buildings can't crowd taxiways out of the budget) —
    # see BUILDING_OUT_CAP/ROAD_OUT_CAP/TAXIWAY_OUT_CAP above for why these
    # exist. All requested "out" blocks land in one combined response's
    # "elements" array.
    query_parts = ["[out:json][timeout:20];\n"]
    if "buildings" in want:
        query_parts.append(f'way["building"]({s},{w},{n},{e});\n')
        query_parts.append(f"out geom {BUILDING_OUT_CAP};\n")
    if "roads" in want:
        query_parts.append(f'way["highway"~"^({MAJOR_HIGHWAYS})$"]({s},{w},{n},{e});\n')
        query_parts.append(f"out geom {ROAD_OUT_CAP};\n")
    elif "majorroads" in want:
        # arterials only, for wide-radius queries (elif: if a caller asked
        # for both, "roads" already includes every majorroads class)
        query_parts.append(f'way["highway"~"^({MAJOR_ONLY_HIGHWAYS})$"]({s},{w},{n},{e});\n')
        query_parts.append(f"out geom {MAJORROAD_OUT_CAP};\n")
    if "airport" in want:
        # the full airport ground plan: taxiway centerlines PLUS apron
        # polygons and the aerodrome perimeter polygon (the green infield
        # base + gray tarmac areas of the AirX airport look)
        query_parts.append(f'way["aeroway"~"^(taxiway|apron|aerodrome)$"]({s},{w},{n},{e});\n')
        query_parts.append(f"out geom {TAXIWAY_OUT_CAP};\n")
    elif "taxiways" in want:
        query_parts.append(f'way["aeroway"~"^(taxiway|apron)$"]({s},{w},{n},{e});\n')
        query_parts.append(f"out geom {TAXIWAY_OUT_CAP};\n")
    if "water" in want:
        # lakes/ponds as polygons, rivers/canals as centerlines
        query_parts.append(f'way["natural"="water"]({s},{w},{n},{e});\n')
        query_parts.append(f"out geom {LAKE_OUT_CAP};\n")
        query_parts.append(f'way["waterway"~"^(river|canal)$"]({s},{w},{n},{e});\n')
        query_parts.append(f"out geom {RIVER_OUT_CAP};\n")
    query = "".join(query_parts)

    data, err = fetch_overpass_hedged(query)
    if data is None:
        return jsonify({"error": f"all overpass mirrors failed: {err}"}), 502

    buildings, roads, taxiways, lakes, rivers = [], [], [], [], []
    aprons, aerodromes = [], []
    for el in data.get("elements", []):
        if el.get("type") != "way" or "geometry" not in el:
            continue
        tags = el.get("tags", {})
        points = [[pt["lat"], pt["lon"]] for pt in el["geometry"] if "lat" in pt and "lon" in pt]
        if len(points) < 2:
            continue

        if "building" in tags:
            buildings.append({"points": points, "heightM": estimate_building_height(tags)})
        elif "aeroway" in tags:
            kind = tags.get("aeroway", "taxiway")
            if kind == "apron":
                aprons.append({"points": points})
            elif kind == "aerodrome":
                aerodromes.append({"points": points})
            else:
                taxiways.append({"points": points, "kind": kind})
        elif tags.get("natural") == "water":
            lakes.append({"points": points})
        elif "waterway" in tags:
            rivers.append({"points": points, "kind": tags.get("waterway", "river")})
        elif "highway" in tags:
            roads.append({"points": points, "class": tags.get("highway", "residential")})

    payload = {
        "bounds": {"south": s, "west": w, "north": n, "east": e},
        "buildings": buildings,
        "roads": roads,
        "taxiways": taxiways,
        "lakes": lakes,
        "rivers": rivers,
        "aprons": aprons,
        "aerodromes": aerodromes,
    }

    payload_json = json.dumps(payload, separators=(",", ":"))
    if len(_features_cache) > FEATURES_CACHE_MAX:
        _features_cache.clear()
    _features_cache[cache_key] = payload_json
    return Response(payload_json, mimetype="application/json")


# Raw satellite tile bytes, keyed (z,x,y) — ~15KB each, so this is cheap
# memory even at 200 entries. Cached for the same reason _elev_cache is:
# batch prefetching should never fetch the same tile twice.
_color_cache = {}
COLOR_CACHE_MAX = 200


def _fetch_color_tile(z, x, y):
    """Satellite tile PNG bytes, cached; None on any failure (terrain then
    falls back to elevation-based coloring for that tile)."""
    key = (z, x, y)
    if key in _color_cache:
        return _color_cache[key]
    try:
        resp = _http.get(
            COLOR_TILE_URL.format(z=z, x=x, y=y), timeout=10, headers=COLOR_TILE_HEADERS)
    except requests.RequestException:
        return None
    if resp.status_code != 200:
        return None
    data = resp.content
    if len(_color_cache) > COLOR_CACHE_MAX:
        _color_cache.clear()
    _color_cache[key] = data
    return data


# Decoded 256x256 elevation arrays (float32 meters), keyed (z,x,y). Exists
# because SEAM STITCHING (below) needs each tile's east/south neighbors:
# without this cache every tile request would fetch 4 PNGs; with it, a
# ring of adjacent tiles reuses each decoded PNG ~4 times, so total AWS
# fetches stay roughly what they were before stitching. Kept deliberately
# small: float32 256x256 = 256KB each, and every gunicorn worker process
# holds its own copy of this cache — 80 x 256KB x 2 workers ~= 40MB.
_elev_cache = {}
ELEV_CACHE_MAX = 80


def _get_elevation_array(z, x, y):
    """Decoded elevation for one tile, cached. Returns None on any failure
    (caller degrades gracefully). x wraps around the antimeridian; y out of
    range (past the poles) returns None."""
    n = 2 ** z
    if y < 0 or y >= n:
        return None
    x = x % n
    key = (z, x, y)
    if key in _elev_cache:
        return _elev_cache[key]
    try:
        resp = _http.get(TILE_URL.format(z=z, x=x, y=y), timeout=15)
    except requests.RequestException:
        return None
    if resp.status_code != 200:
        return None
    arr = decode_terrarium(resp.content).astype(np.float32)
    if len(_elev_cache) > ELEV_CACHE_MAX:
        _elev_cache.clear()
    _elev_cache[key] = arr
    return arr


def build_tile_payload(z, x, y, grid):
    """One tile's payload (heights + colors) as a compact JSON STRING,
    cached as such (see cache memory notes above). Returns None if the
    tile's own elevation fetch failed.

    SEAM STITCHING: a slippy tile's 256 pixel rows/cols cover [edge, next
    edge) — pixel 255 is one pixel SHORT of the tile's east/south boundary;
    the true boundary value is the NEIGHBOR tile's pixel 0. The old
    downsample (linspace 0..255 on one tile) put each tile's edge vertices
    at pixel 255 while the adjacent tile put its matching vertices at ITS
    pixel 0 — different pixels, different heights, and on slopes that
    one-pixel disagreement (~5-10m) opened visible vertical gaps along
    every tile border (reported live: "gaps between chunks"). Building a
    257x257 grid whose last row/col come from the neighbors makes shared
    edge vertices sample the SAME pixel on both sides — mathematically
    identical heights, seamless borders."""
    cache_key = (z, x, y, grid)
    if cache_key in _cache:
        return _cache[cache_key]

    # main tile + 3 stitch neighbors + color, all concurrently — on a free
    # host these are the latency, not the CPU
    with ThreadPoolExecutor(max_workers=5) as pool:
        main_f = pool.submit(_get_elevation_array, z, x, y)
        east_f = pool.submit(_get_elevation_array, z, x + 1, y)
        south_f = pool.submit(_get_elevation_array, z, x, y + 1)
        diag_f = pool.submit(_get_elevation_array, z, x + 1, y + 1)
        color_f = pool.submit(_fetch_color_tile, z, x, y)
        main = main_f.result()
        east = east_f.result()
        south = south_f.result()
        diag = diag_f.result()
        color_bytes = color_f.result()

    if main is None:
        return None

    full = np.empty((257, 257), dtype=np.float64)
    full[:256, :256] = main
    # missing neighbors (ocean-edge 404s, transient failures) degrade to
    # duplicating this tile's own edge — worst case is the OLD seam
    # behavior on that one border, never a hard failure
    full[:256, 256] = east[:, 0] if east is not None else main[:, 255]
    full[256, :256] = south[0, :] if south is not None else main[255, :]
    if diag is not None:
        full[256, 256] = diag[0, 0]
    elif east is not None:
        full[256, 256] = east[255, 0]
    else:
        full[256, 256] = main[255, 255]

    # grid samples spanning 0..256 INCLUSIVE — edge vertices land exactly
    # on tile boundaries (this is what makes adjacent tiles agree)
    idx = np.linspace(0, 256, grid).astype(int)
    small = full[np.ix_(idx, idx)]

    colors = None
    if color_bytes is not None:
        raw = sample_map_tile_raw(color_bytes, grid)
        colors = [[boosted_hex(px) for px in row] for row in raw]

    payload = {
        "z": z, "x": x, "y": y, "grid": grid,
        "heights": [[round(float(v), 1) for v in row] for row in small],
        "colors": colors,
    }
    payload_json = json.dumps(payload, separators=(",", ":"))

    if len(_cache) > CACHE_MAX:
        _cache.clear()
    _cache[cache_key] = payload_json
    return payload_json


@app.route("/elevation/<int:z>/<int:x>/<int:y>")
def elevation(z, x, y):
    grid = min(max(int(request.args.get("grid", 33)), 2), 65)
    payload_json = build_tile_payload(z, x, y, grid)
    if payload_json is None:
        return jsonify({"error": "tile fetch failed"}), 502
    return Response(payload_json, mimetype="application/json")


# Hard cap on tiles per batch request — bounds worst-case latency and
# response size for one HTTP call.
MAX_BATCH_TILES = 24


@app.route("/elevation_batch")
def elevation_batch():
    """Many tiles in ONE request: /elevation_batch?tiles=15/100/200,15/101/200&grid=9
    Exists because the Roblox client used to make one HTTP round-trip per
    tile — 100+ requests to fill its terrain ring on a cold spawn, each
    paying full round-trip latency and queueing behind the others. One
    batched request amortizes that across 16+ tiles, and the server fans
    the per-tile work out across threads. Failed tiles come back as
    {"z","x","y","error":true} entries rather than failing the whole batch."""
    grid = min(max(int(request.args.get("grid", 9)), 2), 65)
    specs = []
    for item in request.args.get("tiles", "").split(","):
        parts = item.strip().split("/")
        if len(parts) != 3:
            continue
        try:
            specs.append((int(parts[0]), int(parts[1]), int(parts[2])))
        except ValueError:
            continue
    if not specs:
        return jsonify({"error": "tiles=z/x/y,z/x/y,... required"}), 400
    specs = specs[:MAX_BATCH_TILES]

    # PREFETCH, deduplicated across the whole batch: adjacent tiles share
    # most of their stitch-neighbors (a 4x4 block of tiles needs only 25
    # unique elevation PNGs, not 16*4=64), so gather the unique set first
    # and fetch each exactly once through one bounded pool. The naive
    # alternative — fanning out build_tile_payload calls, each spawning its
    # own 5-way fetch — created thread/connection stampedes that blew past
    # the gunicorn worker timeout on Render's small CPU.
    elev_needed = set()
    color_needed = set()
    for (z, x, y) in specs:
        if (z, x, y, grid) in _cache:
            continue
        elev_needed.update({(z, x, y), (z, x + 1, y), (z, x, y + 1), (z, x + 1, y + 1)})
        color_needed.add((z, x, y))
    if elev_needed or color_needed:
        with ThreadPoolExecutor(max_workers=8) as pool:
            for (z, x, y) in elev_needed:
                pool.submit(_get_elevation_array, z, x, y)
            for (z, x, y) in color_needed:
                pool.submit(_fetch_color_tile, z, x, y)
            # pool context exit waits for all fetches; everything below
            # hits the warm caches

    # assemble the response by SPLICING cached JSON strings — payloads are
    # cached pre-serialized (see memory notes on _cache) and never exist
    # as Python object trees here
    fragments = []
    for (z, x, y) in specs:
        try:
            payload_json = build_tile_payload(z, x, y, grid)
        except Exception:
            payload_json = None
        if payload_json is None:
            fragments.append(json.dumps({"z": z, "x": x, "y": y, "error": True}, separators=(",", ":")))
        else:
            fragments.append(payload_json)
    body = '{"grid":%d,"tiles":[%s]}' % (grid, ",".join(fragments))
    return Response(body, mimetype="application/json")


@app.route("/health")
def health():
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8020)
