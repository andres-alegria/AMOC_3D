"""
Deep-water formation block diagram — subpolar North Atlantic.

    python3 block3d.py OUT.png [--svg OUT.svg] [--fast] [--no-viking] [--no-arrows]

A block diagram (parallel/axonometric camera, as in the MRI 25 range) of the
GEBCO 2024 seafloor between the Labrador Sea and the Greenland–Scotland Ridge,
with VIKING20X currents drawn as ribbons floating at their depth and the
Fig. 1.1 (v6) schematic arrows lifted to their depth class.

Inputs (../data/):
    block_5km.bil       GEBCO 2024 sub-ice warped to LAEA 60N/32.5W, 5 km (gdalwarp)
    land_block.geojson  Natural Earth 10 m land, same CRS, clipped to the block
    ice_block.geojson   Natural Earth 10 m glaciated areas, same
and the VIKING20X layers (viking20x_layers.npz) and Basemap_v4.geojson, copied
into data/ from Ch_1_Figure_1/Basemap.

World coordinates: x east, y north, in km on the LAEA plane; z metres (negative
below the sea surface). Everything is drawn back to front (painter's order on
view depth), so lines inside the water column hide correctly behind ridges.
"""
import sys, json, os, math
import numpy as np
from osgeo import osr, gdal
osr.UseExceptions(); gdal.UseExceptions()
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
import matplotlib.patheffects as pe
from matplotlib.path import Path
from matplotlib.patches import PathPatch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
# Fig. 1.1 inputs, copied into data/ on 2 Nov 2026 from the figure's Basemap folder
# (_Figures/Ch_1_Figure_1/Basemap: VIKING20X_Currents/data/ and Basemap_v4/), so the repo stands alone
VIKING_NPZ = os.path.join(DATA, "viking20x_layers.npz")          # + viking20x_layers_meta.json beside it
ARROWS = os.path.join(DATA, "Basemap_v4.geojson")
PROJ4 = "+proj=laea +lat_0=60 +lon_0=-32.5 +datum=WGS84 +units=m"

# --- the block -----------------------------------------------------------------
DEM_Y0 = -1350.0                                     # km, south edge of block_5km.bil (the gdalwarp -te)
X0, X1, Y0, Y1 = -1900.0, 1900.0, -1050.0, 1250.0    # km, the block drawn; Y0 moved north 2 Nov (was -1350):
#   the front wall now cuts north of the Grand Banks, so the deep return flow leaves through 2,750 m of water
#   instead of beside the shelf plateau that hid it. Corners 47.3° N, centre 50.6° N. Y0 >= DEM_Y0.
CELL = 5.0                                           # km
ZBASE = -5600.0                                      # floor of the block, m
CLEAR = 500.0                                        # m a deep line keeps above the floor, so the painter's order cannot bury it

# --- the camera — adjust the view here ------------------------------------------
W, H_PX = 2400, 1600          # canvas, px
THETA = math.radians(14)      # turn of the block: 0 = square-on (south face in front), 45 = diamond
PHI = math.radians(41)        # elevation of the eye above the horizon
EXAG = 85.0                  # vertical exaggeration (1 = true scale)
MARGIN = (60, 130, 60, 50)    # l, t, r, b, px
RASTER_SCALE = 2              # SVG export: the seafloor raster is this many times the canvas resolution

# --- palette (light page, Fig. 1.1 family) — adjust colours here ----------------
PAL = dict(
    paper="#FFFFFF",
    # seafloor depth ramp, shallow → deep (greys-blue so the flows carry the colour)
    floor=[(0, "#E3E9ED"), (-500, "#CDD8DF"), (-1500, "#A9BAC5"), (-3000, "#7F95A4"), (-4500, "#5B7080"), (-5600, "#4A5C6B")],
    water="#9CC4DA", water_alpha=0.20,             # the translucent sea surface
    face_water="#D9E8F0", face_water_alpha=0.5, face_crust="#B9BFC4", face_crust_w="#A9B0B6",   # the cut faces (water column half transparent)
    land="#C8D0D8", ice="#EEF3F6", coast="#9AA6AE",
    surf=("#F3CFC8", "#E2907E"),                    # VIKING surface ribbons, slow → fast
    mid=("#BFD2DE", "#6F93AA"),                     # VIKING intermediate ribbons
    deep=("#9DB9CE", "#2E5F80"),                    # VIKING deep ribbons
    arrow_surf="#D04028", arrow_deep="#003048", arrow_mid="#88C0D0",
    plume="#7E6E9D",
    conv=("#D04028", "#7E6E9D", "#003048"),        # the sinking ribbons: surface red → purple → deep navy (Fig. 1.1 depth panel)
    osnap="#26333C",                                # the observing array: ink, so it reads as measurement, not flow
    ink="#26333C", ink_soft="#64798A",
)
FONT = "Open Sans"
LABEL_PT = dict(land=10.5, sea=10.5, feature=8.5, flow=9.5, obs=9.0)   # label sizes by kind — adjust here

# --- VIKING ribbons — adjust density and look here --------------------------------
BANDS = dict(                                 # name: (z metres, colour pair, speed percentile floor, separation km, min length km)
    surface=(-200.0, "surf", 48, 56, 170),   # 2 Nov: 200 m down (was 0) so the land plateaus and shelves cover the ribbons
    deep=(-2000.0, "deep", 60, 46, 150),
)
DRAW_MID = False                              # the 500–1200 m band, off by default (clutter)
RIBBON_W = (0.9, 3.4)                          # px, slow → fast
ARROW_W = {1: 5.0, 2: 9.0, 3: 14.0}            # px per Thickness class

# --- deep-water formation sites — adjust here ----------------------------------------
# The purple "deepwater formation" clouds of Fig. 1.1. Each site is an outline in lon/lat (a hand-drawn
# polygon, or a circle given as (lon, lat, radius km)) and a bottom depth. Outlines are clipped to water
# deeper than PLUME_MIN_DEPTH, so a cloud drawn around Cape Farewell never covers Greenland's plateau.
PLUMES = [
    dict(name="Labrador–Irminger", zb=-1800.0,      # one cloud around southern Greenland, as in Fig. 1.1 (2 Nov)
         outline=[(-57.5, 58.5), (-56.5, 60.5), (-53.0, 61.2), (-49.0, 60.2), (-45.0, 59.2), (-41.0, 60.8),
                  (-38.0, 63.2), (-34.0, 63.8), (-30.5, 62.0), (-31.5, 59.0), (-35.5, 57.3), (-41.0, 56.3),
                  (-46.0, 55.4), (-51.0, 55.2), (-55.0, 56.3)]),
    dict(name="Iceland Sea", zb=-1600.0, outline=(-15.0, 68.5, 130)),
]
PLUME_MIN_DEPTH = -700.0        # m: the clouds exist only over floor deeper than this
PLUME_SMOOTH = 4                # smoothing passes on the clipped outline
WISPS = dict(n=0, height=2600.0, spread=0.38, amp=0.07, lw=(2.4, 0.5))   # heat-loss wisps above each plume (n=0: off, dropped 2 Nov): count, height m (exaggerated), spread/amplitude as fraction of radius, line width px thick → thin
CONV_FADE = (0.08, 0.92)        # along a Convection arrow: fraction where the descent starts / ends
CONV_SPLIT = -60.0              # m: shallower than this the sinking ribbon is drawn above the sea surface
DRAW_LAND = False               # Natural Earth land/ice plates on the surface (off since 2 Nov: the bathymetry is the subject)
DRAW_WATER = False              # translucent sea surface + water column on the cut faces (off since 2 Nov)
# the inflow / outflow arrows that cross the front (south) face protrude from the box by `length` km, so the
# reader sees what enters and leaves the subpolar basin; `leader` px is the label tick from the stub tip
FRONT_STUB = dict(length=520.0, leader=22, drop=14, match_outflow=True, head_gap=0.6)   # leader px sideways/down from the tip to the label; match_outflow: the inflow stub reaches as far south as the outflow tip; head_gap: space (× line width) between the inflow tip head and the ribbon
SURF_SEED = -250.0              # m: VIKING surface ribbons are seeded only over floor deeper than this (they sit at BANDS z)
FRONT_LABELS = dict(Red="Warm surface flow", Blue="Cold deep return flow")   # labels pointing at the stubs
# where one Fig. 1.1 arrow runs into another of a different thickness class, the width is tapered over the
# last `taper` km of the arriving arrow to the width of the one it joins, and the head at the joint is dropped
# (the flow continues). A joint = an arrow end within `dist` km and `dz` m of another arrow's vertex.
JOIN = dict(dist=45.0, taper=300.0, dz=400.0)
FRONT_LABEL_SIDE = dict(Red=1)  # force a label side (+1 right, -1 left); classes not listed go to the side away from the ribbon
# deep (Blue) arrows: Fig. 1.1 drew them on a flat map, running over the shelves. In 3D a boundary current
# belongs at the slope foot, so each vertex of the arrows listed in `ids` is walked downslope (toward the
# deepest floor on a ring of radius `look` km, in `step` km moves) until the floor is deeper than `target`;
# a vertex that finds no such water within `max_km` stays put, and unreached vertices at either end are
# trimmed. Only the Labrador / Newfoundland boundary-current arrows are listed: the overflow arrows
# (Denmark Strait, Iceland–Scotland) descend the sills on purpose and keep their drawn paths.
DEEP_SNAP = dict(ids=(27, 29, 31), target=-2500.0, step=4.0, look=30.0, max_km=250.0)

# --- the observing array ------------------------------------------------------------
OSNAP_CSV = os.path.join(DATA, "osnap_section.csv")
OSNAP = dict(lw=1.4, dash=(0, (5, 3)), mooring_every=110.0, mooring_lw=0.9, mooring_alpha=0.75)   # surface line + hairline "moorings" every N km

# --- labels: lon, lat, text, kind (land / sea / feature / flow / obs) — adjust or add here ----
LABELS = [
    (-44.5, 70.5, "Greenland", "land"), (-18.5, 64.9, "Iceland", "land"), (-62.0, 54.5, "Labrador", "land"),
    (-3.5, 56.5, "Scotland", "land"), (9.0, 62.0, "Norway", "land"), (-55.5, 60.0, "Labrador\nSea", "sea"),
    (-36.5, 62.5, "Irminger\nSea", "sea"), (-24.0, 60.0, "Iceland Basin", "sea"), (-6.0, 69.0, "Nordic Seas", "sea"),
    (-26.5, 68.0, "Denmark Strait", "feature"),
    (-45.5, 57.0, "Deep-water\nconversion", "flow"),
    (-52.5, 51.5, "OSNAP West", "obs"), (-14.5, 56.6, "OSNAP East", "obs"),
]

# ===================================================================================
# camera
# ===================================================================================
SCALE, ZK, ORIGIN_X, ORIGIN_Y = 1.0, 1.0, 0.0, 0.0


def to_view(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    a = x * math.cos(THETA) - y * math.sin(THETA)
    d = x * math.sin(THETA) + y * math.cos(THETA)
    return a, d


def project(x, y, z):
    a, d = to_view(x, y)
    sx = ORIGIN_X + a * SCALE
    sy = ORIGIN_Y - d * math.sin(PHI) * SCALE - np.asarray(z, float) * ZK
    return sx, sy


def fit():
    """Place the block (sea surface and floor) inside the canvas margins."""
    global SCALE, ZK, ORIGIN_X, ORIGIN_Y
    SCALE, ZK, ORIGIN_X, ORIGIN_Y = 1.0, EXAG * math.cos(PHI) / 1000.0, 0.0, 0.0
    cx = np.array([X0, X1, X1, X0]); cy = np.array([Y0, Y0, Y1, Y1])
    sx0, sy0 = project(cx, cy, np.zeros(4))
    sx1, sy1 = project(cx, cy, np.full(4, ZBASE))
    x0 = min(sx0.min(), sx1.min()); x1 = max(sx0.max(), sx1.max())
    y0 = min(sy0.min(), sy1.min()); y1 = max(sy0.max(), sy1.max())
    l, t, r, b = MARGIN
    s = min((W - l - r) / (x1 - x0), (H_PX - t - b) / (y1 - y0))
    SCALE, ZK = s, ZK * s
    ORIGIN_X = l + (W - l - r - s * (x1 - x0)) / 2 - s * x0
    ORIGIN_Y = t + (H_PX - t - b - s * (y1 - y0)) / 2 - s * y0


# ===================================================================================
# data
# ===================================================================================
def load_dem():
    H = np.fromfile(os.path.join(DATA, "block_5km.bil"), dtype=np.float32).reshape(520, 760)
    H = H[::-1].astype(float)                     # row 0 = south
    H = H[int(round((Y0 - DEM_Y0) / CELL)):]      # drop the rows south of the drawn block
    ny, nx = H.shape
    xs = X0 + (np.arange(nx) + 0.5) * CELL
    ys = Y0 + (np.arange(ny) + 0.5) * CELL
    return H, xs, ys


def lonlat_grid(xs, ys):
    """lon/lat of every cell centre, and the local east/north unit vectors in km space."""
    src = osr.SpatialReference(); src.ImportFromProj4(PROJ4)
    dst = osr.SpatialReference(); dst.ImportFromEPSG(4326)
    dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    tr = osr.CoordinateTransformation(src, dst)
    X, Y = np.meshgrid(xs * 1000.0, ys * 1000.0)
    pts = tr.TransformPoints(np.column_stack([X.ravel(), Y.ravel()]).tolist())
    ll = np.array(pts)[:, :2]
    lon = ll[:, 0].reshape(X.shape); lat = ll[:, 1].reshape(X.shape)
    # local north = gradient of latitude on the plane
    dlat_dx = np.gradient(lat, axis=1); dlat_dy = np.gradient(lat, axis=0)
    n = np.hypot(dlat_dx, dlat_dy); nx_, ny_ = dlat_dx / n, dlat_dy / n
    return lon, lat, (nx_, ny_)


def ll_to_block(lon, lat):
    src = osr.SpatialReference(); src.ImportFromEPSG(4326)
    dst = osr.SpatialReference(); dst.ImportFromProj4(PROJ4)
    src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    tr = osr.CoordinateTransformation(src, dst)
    pts = np.array(tr.TransformPoints(np.column_stack([lon, lat]).tolist()))[:, :2] / 1000.0
    return pts[:, 0], pts[:, 1]


def bilinear(F, fx, fy):
    """Sample F[j, i] at fractional indices (fx along axis 1, fy along axis 0)."""
    ny, nx = F.shape
    fx = np.clip(fx, 0, nx - 1.0001); fy = np.clip(fy, 0, ny - 1.0001)
    i0 = np.floor(fx).astype(int); j0 = np.floor(fy).astype(int)
    tx = fx - i0; ty = fy - j0
    return ((F[j0, i0] * (1 - tx) + F[j0, i0 + 1] * tx) * (1 - ty)
            + (F[j0 + 1, i0] * (1 - tx) + F[j0 + 1, i0 + 1] * tx) * ty)


def viking_band(name, lon, lat, north):
    """u, v (km-space components, m/s) and a wet mask of one VIKING band on the block grid."""
    d = np.load(VIKING_NPZ)
    glon, glat = d["lon"], d["lat"]
    u = np.where(d[name + "_ok"], d[name + "_u"], 0.0); v = np.where(d[name + "_ok"], d[name + "_v"], 0.0)
    ok = d[name + "_ok"].astype(float)
    fx = (lon - glon[0]) / 0.25; fy = (glat[0] - lat) / 0.25        # row 0 = north in the npz
    ue, vn, wet = bilinear(u, fx, fy), bilinear(v, fx, fy), bilinear(ok, fx, fy)
    nx_, ny_ = north                                   # east = north rotated clockwise
    ex, ey = ny_, -nx_
    return ue * ex + vn * nx_, ue * ey + vn * ny_, wet > 0.5


# ===================================================================================
# streamlines: integrate everything at once, then thin to an even spacing
# ===================================================================================
def streamlines(U, V, wet, floor, dsep, min_len, speed_floor_pct, step=4.0, max_len=1400.0, seed=7):
    ny, nx = U.shape
    speed = np.hypot(U, V)
    thr = np.percentile(speed[wet & (floor < -200)], speed_floor_pct)
    go = wet & (speed > thr)
    gof = go.astype(float)
    rng = np.random.default_rng(seed)
    # seeds on a jittered grid, denser than the target spacing
    gx = np.arange(0, nx * CELL, dsep * 0.5); gy = np.arange(0, ny * CELL, dsep * 0.5)
    SX, SY = np.meshgrid(gx, gy)
    sx = (SX + rng.uniform(-0.25, 0.25, SX.shape) * dsep).ravel()
    sy = (SY + rng.uniform(-0.25, 0.25, SY.shape) * dsep).ravel()
    nsteps = int(max_len / 2 / step)

    def integrate(sign):
        px, py = sx.copy(), sy.copy()
        alive = np.ones(px.shape, bool)
        out = np.full((nsteps + 1, px.size, 2), np.nan)
        out[0, :, 0], out[0, :, 1] = px, py
        for k in range(1, nsteps + 1):
            fx, fy = px / CELL - 0.5, py / CELL - 0.5
            u, v = bilinear(U, fx, fy), bilinear(V, fx, fy)
            s = np.hypot(u, v) + 1e-9
            mx, my = px + sign * step * 0.5 * u / s, py + sign * step * 0.5 * v / s
            u2, v2 = bilinear(U, mx / CELL - 0.5, my / CELL - 0.5), bilinear(V, mx / CELL - 0.5, my / CELL - 0.5)
            s2 = np.hypot(u2, v2) + 1e-9
            px = px + sign * step * u2 / s2; py = py + sign * step * v2 / s2
            inside = (px > 0) & (px < nx * CELL) & (py > 0) & (py < ny * CELL)
            okk = inside.copy()
            okk[inside] = bilinear(gof, px[inside] / CELL - 0.5, py[inside] / CELL - 0.5) > 0.5
            alive &= okk
            out[k, alive, 0], out[k, alive, 1] = px[alive], py[alive]
        return out

    fwd, bwd = integrate(+1), integrate(-1)
    # keep only seeds that start in flow
    start_ok = bilinear(gof, sx / CELL - 0.5, sy / CELL - 0.5) > 0.5
    # thinning: a test grid at dsep/2; walk out from the seed, stop at an occupied cell
    tcell = dsep * 0.5
    tnx, tny = int(nx * CELL / tcell) + 2, int(ny * CELL / tcell) + 2
    occ = np.full((tny, tnx), -1, int)
    order = rng.permutation(sx.size)
    lines = []
    for idx in order:
        if not start_ok[idx]:
            continue
        halves = []
        for arr in (bwd, fwd):
            pts = arr[:, idx, :]
            pts = pts[~np.isnan(pts[:, 0])]
            keep = 0
            for p in pts:
                ci, cj = int(p[0] / tcell), int(p[1] / tcell)
                o = occ[cj, ci]
                if o != -1 and o != idx:
                    break
                keep += 1
            halves.append(pts[:keep])
        line = np.vstack([halves[0][::-1], halves[1][1:]]) if len(halves[1]) > 1 else halves[0][::-1]
        if len(line) < 2:
            continue
        L = np.sum(np.hypot(np.diff(line[:, 0]), np.diff(line[:, 1])))
        if L < min_len:
            continue
        # claim the cells (with one cell of padding so neighbours keep their distance)
        ci = (line[:, 0] / tcell).astype(int); cj = (line[:, 1] / tcell).astype(int)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ii = np.clip(ci + di, 0, tnx - 1); jj = np.clip(cj + dj, 0, tny - 1)
                free = occ[jj, ii] == -1
                occ[jj[free], ii[free]] = idx
        sp = bilinear(speed, line[:, 0] / CELL - 0.5, line[:, 1] / CELL - 0.5)
        lines.append((line, sp))
    return lines, thr, np.percentile(speed[go], 97)



# ===================================================================================
# visibility: a floating horizon per screen column (the MRI range3d device)
# ===================================================================================
class Horizon:
    """For every screen column, the highest screen point of the occluding surface
    (seafloor + land plates) at each view depth, accumulated front to back."""

    def __init__(self, Hocc, xs, ys, nd=1400):
        cols = np.arange(W) + 0.5
        a = (cols - ORIGIN_X) / SCALE
        _, dcorner = to_view(np.array([X0, X1, X1, X0]), np.array([Y0, Y0, Y1, Y1]))
        self.d = np.linspace(dcorner.min(), dcorner.max(), nd)
        A, D = np.meshgrid(a, self.d, indexing="ij")                      # (W, nd)
        x = A * math.cos(THETA) + D * math.sin(THETA)
        y = -A * math.sin(THETA) + D * math.cos(THETA)
        inside = (x >= X0) & (x <= X1) & (y >= Y0) & (y <= Y1)
        z = bilinear(Hocc, (x - X0) / CELL - 0.5, (y - Y0) / CELL - 0.5)
        sy = ORIGIN_Y - D * math.sin(PHI) * SCALE - z * ZK
        sy = np.where(inside, sy, np.inf)
        self.hmin = np.minimum.accumulate(sy, axis=1)

    def visible(self, x, y, z, tol=0.6):
        sx, sy = project(x, y, z)
        c = np.clip(np.floor(sx).astype(int), 0, W - 1)
        _, d = to_view(x, y)
        k = np.clip(np.searchsorted(self.d, d) - 1, 0, len(self.d) - 1)   # the sample just in front
        front = np.where(k > 0, self.hmin[c, np.maximum(k - 1, 0)], np.inf)
        return sy <= front + tol


def visible_runs(hz, pts, zz, min_pts=2, bridge=5):
    """Split a line into the stretches the camera can see. Hidden gaps shorter than
    `bridge` points are kept visible: a line grazing the horizon flickers otherwise."""
    vis = hz.visible(pts[:, 0], pts[:, 1], zz, tol=1.5)
    i = 0
    while i < len(vis):
        if not vis[i]:
            j = i
            while j < len(vis) and not vis[j]:
                j += 1
            if 0 < i and j < len(vis) and j - i <= bridge:
                vis[i:j] = True
            i = j
        else:
            i += 1
    runs, cur = [], []
    for i, v in enumerate(vis):
        if v:
            cur.append(i)
        elif cur:
            runs.append(cur); cur = []
    if cur:
        runs.append(cur)
    return [(pts[r], zz[r], r[-1] == len(pts) - 1) for r in runs if len(r) >= min_pts]

# ===================================================================================
# drawing helpers
# ===================================================================================
def hex2rgb(h):
    h = h.lstrip("#"); return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def mix(c0, c1, t):
    t = np.clip(np.asarray(t, float), 0, 1)[..., None]
    return hex2rgb(c0) * (1 - t) + hex2rgb(c1) * t


def ramp(stops, z):
    zs = np.array([s[0] for s in stops], float); cs = np.array([hex2rgb(s[1]) for s in stops])
    out = np.zeros(z.shape + (3,))
    for k in range(3):
        out[..., k] = np.interp(z, zs[::-1], cs[::-1, k])
    return out


def ribbon_quads(line, z, width_px, color, depth_key):
    """A polyline (world km) at depth z → per-segment screen quads (solid colour, joins overlap)."""
    sx, sy = project(line[:, 0], line[:, 1], z)
    _, d = to_view(line[:, 0], line[:, 1])
    dx, dy = np.diff(sx), np.diff(sy)
    L = np.hypot(dx, dy) + 1e-9
    nx_, ny_ = -dy / L, dx / L
    w = np.asarray(width_px, float)
    if w.ndim == 0:
        w = np.full(len(sx), float(w))
    w0, w1 = w[:-1] / 2, w[1:] / 2
    ex, ey = dx / L * 0.6, dy / L * 0.6           # tiny overlap along the line hides the joins
    quads = np.stack([
        np.stack([sx[:-1] + nx_ * w0 - ex, sy[:-1] + ny_ * w0 - ey], -1),
        np.stack([sx[1:] + nx_ * w1 + ex, sy[1:] + ny_ * w1 + ey], -1),
        np.stack([sx[1:] - nx_ * w1 + ex, sy[1:] - ny_ * w1 + ey], -1),
        np.stack([sx[:-1] - nx_ * w0 - ex, sy[:-1] - ny_ * w0 - ey], -1),
    ], 1)
    col = np.asarray(color, float)
    if col.ndim == 1:
        col = np.tile(col, (len(quads), 1))
    else:
        col = (col[:-1] + col[1:]) / 2
    return list(quads), list(col), list((d[:-1] + d[1:]) / 2 + depth_key)


def arrowhead_touching(point, towards, z, width_px, gap=0.0):
    """Triangle whose point sits on `point` (block km), set back by `gap` × width_px pixels, with its body on
    the far side from `towards`: a head that just clears the end of a ribbon and points along it."""
    sx, sy = project(np.array([point[0], towards[0]]), np.array([point[1], towards[1]]), z)
    dx, dy = sx[1] - sx[0], sy[1] - sy[0]
    L = math.hypot(dx, dy) + 1e-9
    ux, uy = dx / L, dy / L
    nx_, ny_ = -uy, ux
    tip = np.array([sx[0], sy[0]]) - np.array([ux, uy]) * width_px * gap
    base = tip - np.array([ux, uy]) * width_px * 2.2
    hw = width_px * 1.25
    return np.array([tip, base + np.array([nx_, ny_]) * hw, base - np.array([nx_, ny_]) * hw])


def arrowhead(line, z, width_px):
    """Triangle at the end of a polyline, in screen space."""
    sx, sy = project(line[-2:, 0], line[-2:, 1], z)
    dx, dy = sx[1] - sx[0], sy[1] - sy[0]
    L = math.hypot(dx, dy) + 1e-9
    ux, uy = dx / L, dy / L
    nx_, ny_ = -uy, ux
    tip = np.array([sx[1] + ux * width_px * 1.6, sy[1] + uy * width_px * 1.6])
    base = np.array([sx[1] - ux * width_px * 0.6, sy[1] - uy * width_px * 0.6])
    hw = width_px * 1.25
    return np.array([tip, base + np.array([nx_, ny_]) * hw, base - np.array([nx_, ny_]) * hw])


def clip_to_block(pts):
    """Split a world polyline into the pieces inside the block."""
    inside = (pts[:, 0] > X0 + 2) & (pts[:, 0] < X1 - 2) & (pts[:, 1] > Y0 + 2) & (pts[:, 1] < Y1 - 2)
    pieces, cur = [], []
    for p, ok in zip(pts, inside):
        if ok:
            cur.append(p)
        elif cur:
            pieces.append(np.array(cur)); cur = []
    if cur:
        pieces.append(np.array(cur))
    return [p for p in pieces if len(p) >= 2]


def plume_outline(site, floor, xs, ys):
    """Block-km outline of a formation site: the lon/lat polygon (or circle) clipped to floor < PLUME_MIN_DEPTH,
    as the longest 0.5-contour of the inside-and-deep mask, smoothed. Also returns its centroid and an
    equivalent radius (for the wisps)."""
    import contourpy
    from matplotlib.path import Path
    o = site["outline"]
    if len(o) == 3 and not hasattr(o[0], "__len__"):       # circle: lon, lat, radius km
        cx, cy = ll_to_block(np.array([o[0]]), np.array([o[1]]))
        ang = np.linspace(0, 2 * np.pi, 72, endpoint=False)
        poly = np.column_stack([cx[0] + o[2] * np.cos(ang), cy[0] + o[2] * np.sin(ang)])
    else:
        lon, lat = np.array(o, float).T
        bx, by = ll_to_block(lon, lat)
        poly = smooth(np.vstack([np.column_stack([bx, by]), [[bx[0], by[0]]]]), 2)[:-1]
    X, Y = np.meshgrid(xs, ys)
    inside = Path(poly).contains_points(np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)
    mask = (inside & (floor < PLUME_MIN_DEPTH)).astype(float)
    lines = contourpy.contour_generator(x=xs, y=ys, z=mask).lines(0.5)
    if not lines:
        return None
    ring = max(lines, key=len)
    for _ in range(PLUME_SMOOTH):                           # closed running mean
        ring = (np.roll(ring, 1, 0) + 2 * ring + np.roll(ring, -1, 0)) / 4
    ring = ring[::max(1, len(ring) // 160)]                 # ~160 vertices are plenty for the walls
    c = ring.mean(0); r = float(np.sqrt(abs(0.5 * np.sum(ring[:, 0] * np.roll(ring[:, 1], -1) - np.roll(ring[:, 0], -1) * ring[:, 1])) / np.pi))
    return ring, c, r


def south_crossing(P):
    """First segment of path P (block km) that crosses the front face y = Y0 inside the block's x range."""
    for i in range(len(P) - 1):
        a, b = P[i], P[i + 1]
        if (a[1] - Y0) * (b[1] - Y0) < 0 and X0 < a[0] < X1:
            t = (Y0 - a[1]) / (b[1] - a[1])
            return i, a + t * (b - a)
    return None


def front_split(P, to_y=None):
    """Split a (densified) path at the front face. Returns (inside part ending/starting exactly on the face,
    stub outside the face ordered in the flow direction, inbound flag), or None when the path does not cross
    the face. The stub is FRONT_STUB['length'] km long, or longer when `to_y` is given: it then also continues
    until it reaches that y (the inflow stub is taken as far south as the outflow tip). It follows the drawn
    path while it lasts and continues straight beyond its end (due south if the path was not heading south)."""
    hit = south_crossing(P)
    if hit is None:
        return None
    i, X = hit
    inbound = P[i + 1][1] > Y0
    outside = P[:i + 1][::-1] if inbound else P[i + 1:]
    inside = np.vstack([X, P[i + 1:]]) if inbound else np.vstack([P[:i + 1], X])
    done = lambda L, pt: L >= FRONT_STUB["length"] and (to_y is None or pt[1] <= to_y)
    stub, L, prev = [X], 0.0, X
    for q in outside:
        d = np.hypot(*(q - prev))
        if done(L, prev):
            break
        if L + d >= FRONT_STUB["length"] and (to_y is None or q[1] <= to_y):
            # finish inside this segment at whichever comes later: the length or the target y
            t_len = (FRONT_STUB["length"] - L) / d
            t_y = 1.0 if to_y is None or q[1] == prev[1] else np.clip((to_y - prev[1]) / (q[1] - prev[1]), 0, 1)
            t = max(t_len, t_y) if to_y is not None and prev[1] > to_y else t_len
            pt = prev + (q - prev) * t; stub.append(pt); L += d * t; prev = pt
            if done(L, pt):
                break
            continue
        stub.append(q); L += d; prev = q
    if not done(L, stub[-1]):
        u = stub[-1] - stub[-2] if len(stub) > 1 else np.array([0.0, -1.0]); u = u / max(np.hypot(*u), 1e-9)
        if to_y is not None and stub[-1][1] > to_y and u[1] >= -0.2:
            u = np.array([0.0, -1.0])                       # the drawn path does not head south: go straight south
        ext = max(FRONT_STUB["length"] - L, 0.0)
        if to_y is not None and stub[-1][1] > to_y and u[1] < 0:
            ext = max(ext, (stub[-1][1] - to_y) / -u[1])
        stub.append(stub[-1] + u * ext)
    stub = np.array(stub)
    return inside, (stub[::-1] if inbound else stub), inbound


def outflow_tip_y(features):
    """y (km) of the southernmost outflow stub tip among the deep arrows, or None: the inflow stub is
    extended to this latitude so the two protrusions reach equally far (FRONT_STUB['match_outflow'])."""
    best = None
    for f in features:
        if f["properties"]["Notes"] != "Blue":
            continue
        for part in f["geometry"]["coordinates"]:
            c = np.array(part); bx, by = ll_to_block(c[:, 0], c[:, 1])
            fs = front_split(densify(np.column_stack([bx, by])))
            if fs is not None and not fs[2]:
                y = float(fs[1][-1, 1]); best = y if best is None else min(best, y)
    return best


def densify(pts, step=15.0):
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(np.hypot(*(b - a)) / step))
        for k in range(1, n + 1):
            out.append(a + (b - a) * k / n)
    return np.array(out)


def smooth(pts, n=2):
    for _ in range(n):
        if len(pts) > 4:
            pts = np.vstack([pts[0], (pts[:-2] + 2 * pts[1:-1] + pts[2:]) / 4, pts[-1]])
    return pts


def geojson_polys(path):
    """[(exterior, [holes])] in km."""
    g = json.load(open(path))
    polys = []
    for f in g["features"]:
        geom = f["geometry"]
        rings = geom["coordinates"] if geom["type"] == "Polygon" else [r for p in geom["coordinates"] for r in p]
        if geom["type"] == "Polygon":
            polys.append(([np.array(r)[:, :2] / 1000.0 for r in rings]))
        else:
            for p in geom["coordinates"]:
                polys.append([np.array(r)[:, :2] / 1000.0 for r in p])
    return polys


def poly_patch(ax, rings, z, fc, ec=None, lw=0.0, alpha=1.0, zorder=5):
    verts, codes = [], []
    for r in rings:
        sx, sy = project(r[:, 0], r[:, 1], z)
        v = np.column_stack([sx, sy])
        verts.append(v); codes += [Path.MOVETO] + [Path.LINETO] * (len(v) - 2) + [Path.CLOSEPOLY]
    path = Path(np.vstack(verts), codes)
    ax.add_patch(PathPatch(path, facecolor=fc, edgecolor=ec or "none", linewidth=lw, alpha=alpha, zorder=zorder))


# ===================================================================================
# main render
# ===================================================================================
def render(out_png, out_svg=None, fast=False, viking=True, arrows=True):
    fit()
    H, xs, ys = load_dem()
    floor = np.minimum(H, 0.0)                    # land is a flat plate; the seafloor is the subject
    ny, nx = H.shape
    print(f"camera: scale {SCALE:.3f} px/km, ZK {ZK:.4f} px/m, exaggeration {EXAG:.0f}x")
    hz = Horizon(np.where(H > 0, 15.0, floor), xs, ys)        # land plates occlude too
    deep_items = []           # drawn after the floor, clipped by the horizon: (quads, cols)
    front_items, front_tips = [], []   # arrow stubs outside the front face (drawn after the faces) and their tips for the labels

    fig = plt.figure(figsize=(W / 100, H_PX / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H_PX, 0); ax.axis("off")
    fig.patch.set_facecolor(PAL["paper"])

    # --- A. the seafloor, as shaded quads ---------------------------------------------
    st = 2 if fast else 1
    Hs = floor[::st, ::st]; xg = xs[::st]; yg = ys[::st]
    cell = CELL * st
    # hillshade — light from the upper left of the page (north-west of the block)
    gy, gx = np.gradient(Hs, cell * 1000.0)
    exag_shade = 12.0
    slope = np.arctan(exag_shade * np.hypot(gx, gy)); aspect = np.arctan2(-gx, gy)
    az, alt = math.radians(315), math.radians(45)
    shade = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    shade = np.clip(shade, 0, 1)
    base = ramp(PAL["floor"], Hs)
    col = np.clip(base * (0.62 + 0.55 * shade)[..., None], 0, 1)
    # quads: corners at cell edges
    xe = np.concatenate([xg - cell / 2, [xg[-1] + cell / 2]]); ye = np.concatenate([yg - cell / 2, [yg[-1] + cell / 2]])
    XE, YE = np.meshgrid(xe, ye)
    # corner heights: average of neighbouring cells (pad edges)
    Hp = np.pad(Hs, 1, mode="edge")
    Hc = (Hp[:-1, :-1] + Hp[:-1, 1:] + Hp[1:, :-1] + Hp[1:, 1:]) / 4
    SX, SY = project(XE, YE, Hc)
    q = np.stack([
        np.stack([SX[:-1, :-1], SY[:-1, :-1]], -1), np.stack([SX[:-1, 1:], SY[:-1, 1:]], -1),
        np.stack([SX[1:, 1:], SY[1:, 1:]], -1), np.stack([SX[1:, :-1], SY[1:, :-1]], -1)], 2)
    XC, YC = np.meshgrid(xg, yg)
    _, dq = to_view(XC, YC)
    quads = list(q.reshape(-1, 4, 2)); cols = list(col.reshape(-1, 3)); depth = list(dq.ravel())
    nfloor = len(quads)

    # --- B. things inside the water column, sorted with the floor -----------------------
    lon = lat = north = None
    if viking or arrows:
        lon, lat, north = lonlat_grid(xs, ys)

    def floor_at(x, y):
        return bilinear(floor, np.asarray(x) / CELL - 0.5 - X0 / CELL, np.asarray(y) / CELL - 0.5 - Y0 / CELL)

    bands = dict(BANDS)
    if DRAW_MID:
        bands["intermediate"] = (-850.0, "mid", 55, 44, 150)
    surface_items = []        # drawn above the sea surface: (quads, cols, depth)
    if viking:
        for name, (z, pal, pct, dsep, minlen) in bands.items():
            U, V, wet = viking_band(name, lon, lat, north)
            # local coordinates for the tracer: km from the block's SW corner
            seed = floor < (SURF_SEED if name == "surface" else -50)
            lines, thr, vmax = streamlines(U, V, wet & seed, floor, dsep, minlen, pct)
            print(f"{name}: {len(lines)} ribbons, speed floor {thr:.3f} m/s, p97 {vmax:.3f}")
            c0, c1 = PAL[pal]
            for line, sp in lines:
                wl = line + np.array([X0, Y0])
                t = np.clip(sp / vmax, 0, 1) ** 0.7
                wpx = RIBBON_W[0] + (RIBBON_W[1] - RIBBON_W[0]) * t
                colr = mix(c0, c1, t)
                if z >= 0:
                    zz = np.zeros(len(wl))
                elif name == "surface":
                    zz = np.full(len(wl), z)              # a flat sheet just below the surface; shelves and land hide it
                else:
                    zz = smooth_z(np.maximum(z, floor_at(wl[:, 0], wl[:, 1]) + CLEAR))
                if z >= 0:
                    surface_items.append(ribbon_quads_z(wl, zz, wpx, colr))
                else:
                    for i, (rp, rz, _) in enumerate(visible_runs(hz, wl, zz)):
                        sel = np.isin(wl[:, 0], rp[:, 0])
                        qq, cc, _ = ribbon_quads_z(rp, rz, wpx[sel] if np.ndim(wpx) else wpx, colr[sel])
                        deep_items.append((qq, cc))

    if arrows:
        g = json.load(open(ARROWS))
        in_to_y = outflow_tip_y(g["features"]) if FRONT_STUB["match_outflow"] else None
        recs = []                                           # pass 1: geometry of every arrow piece
        for f in g["features"]:
            p = f["properties"]
            if p["Notes"] not in ("Red", "Blue", "Convection"):
                continue
            for part in f["geometry"]["coordinates"]:
                c = np.array(part)
                bx, by = ll_to_block(c[:, 0], c[:, 1])
                P = np.column_stack([bx, by])
                stub, inbound = None, False
                fs = front_split(densify(P), to_y=in_to_y if p["Notes"] == "Red" else None) if p["Notes"] in ("Red", "Blue") else None
                if fs is not None:
                    P, stub, inbound = fs
                pieces = clip_to_block(P)
                if stub is not None and pieces:          # clip_to_block drops the point on the face: put it back
                    if inbound:
                        pieces[0] = np.vstack([stub[-1], pieces[0]])
                    else:
                        pieces[-1] = np.vstack([pieces[-1], stub[0]])
                for ip, piece in enumerate(pieces):
                    at_face = stub is not None and ((inbound and ip == 0) or (not inbound and ip == len(pieces) - 1))
                    piece = smooth(densify(piece))
                    wpx = 3.0 if p["Type"] == "Eddy" else ARROW_W[int(p["Thickness"])]
                    rec = dict(id=p["id"], notes=p["Notes"], type=p["Type"], w=wpx, at_face=at_face, inbound=inbound,
                               stub=stub, head=p["Type"] == "Current", s=None)
                    if p["Notes"] == "Convection":
                        # the sinking ribbon: starts at the surface in red, descends to the deep level in navy
                        z_deep = smooth_z(np.maximum(-2000.0, floor_at(piece[:, 0], piece[:, 1]) + CLEAR))
                        rec["s"], zz = conv_profile(piece, z_deep)
                        rec["colr"] = ramp3(PAL["conv"], rec["s"])
                    elif p["Notes"] == "Red":
                        if at_face and inbound:          # the inflow: stub outside the face + the path inside, one ribbon
                            piece = np.vstack([stub[:-1], piece]); front_tips.append(("Red", stub[0], stub[1], 0.0))
                        zz = np.zeros(len(piece)); rec["colr"] = hex2rgb(PAL["arrow_surf"])
                    else:
                        if p["id"] in DEEP_SNAP["ids"]:
                            piece = offshore(piece, floor_at)    # boundary currents hug the slope foot, not the shelf
                        zz = smooth_z(np.maximum(-2000.0, floor_at(piece[:, 0], piece[:, 1]) + CLEAR)); rec["colr"] = hex2rgb(PAL["arrow_deep"])
                    rec["piece"], rec["zz"] = piece, zz
                    recs.append(rec)

        join_arrows(recs)                                   # pass 2: taper widths where arrows meet, drop heads there

        for rec in recs:                                    # pass 3: draw
            piece, zz, wv, colr, at_face, inbound, stub = (rec[k] for k in ("piece", "zz", "wv", "colr", "at_face", "inbound", "stub"))
            if rec["notes"] == "Convection":
                k = int(np.argmax(zz < CONV_SPLIT)) if np.any(zz < CONV_SPLIT) else len(zz)
                if k >= 2:   # shallow part, above the sea surface
                    qq, cc, dd = ribbon_quads_z(piece[:k], zz[:k], wv[:k], colr[:k])
                    surface_items.append((qq, cc, dd))
                if k < len(zz) - 1:
                    k0 = max(k - 1, 0)
                    for rp, rz, is_end in visible_runs(hz, piece[k0:], zz[k0:]):
                        sel = np.isin(piece[k0:, 0], rp[:, 0])
                        qq, cc, _ = ribbon_quads_z(rp, rz, wv[k0:][sel], colr[k0:][sel])
                        if is_end and rec["head"]:
                            qq.append(arrowhead(rp, rz[-1], wv[-1])); cc.append(hex2rgb(PAL["conv"][2]))
                        deep_items.append((qq, cc))
            elif rec["notes"] == "Red":
                qq, cc, dd = ribbon_quads_z(piece, zz, wv, colr)
                if rec["head"]:
                    qq.append(arrowhead(piece, zz[-1], wv[-1])); cc.append(colr); dd.append(dd[-1] + 0.01)
                if at_face and inbound and rec["type"] == "Current":
                    # a head outside the stub tip, its point just short of the ribbon, pointing along the flow into the box
                    qq.append(arrowhead_touching(piece[0], piece[1], 0.0, wv[0], FRONT_STUB["head_gap"])); cc.append(colr); dd.append(dd[0] - 0.01)
                surface_items.append((qq, cc, dd))
            else:
                exits = at_face and not inbound          # the outflow leaves through the face: head goes on the stub
                for rp, rz, is_end in visible_runs(hz, piece, zz):
                    sel = np.isin(piece[:, 0], rp[:, 0])
                    qq, cc, _ = ribbon_quads_z(rp, rz, wv[sel], colr)
                    if rec["head"] and is_end and not exits:
                        qq.append(arrowhead(rp, rz[-1], wv[-1])); cc.append(colr)
                    deep_items.append((qq, cc))
                if exits:
                    zs = np.full(len(stub), zz[-1])      # the stub keeps the depth it had at the face
                    qq, cc, _ = ribbon_quads_z(stub, zs, wv[-1], colr)
                    if rec["type"] == "Current":
                        qq.append(arrowhead(stub, zs[-1], wv[-1])); cc.append(colr)
                    front_items.append((qq, cc)); front_tips.append(("Blue", stub[-1], stub[-2], zs[-1]))

    # the observing array: a dashed line on the sea surface, hairline "moorings" down to the floor
    osnap_line = None
    if os.path.exists(OSNAP_CSV):
        for piece in clip_to_block(load_osnap()):
            piece = densify(piece, 8.0)
            osnap_line = piece if osnap_line is None else np.vstack([osnap_line, piece])
            seg = np.hypot(*np.diff(piece, axis=0).T); cum = np.concatenate([[0.0], np.cumsum(seg)])
            for dcum in np.arange(OSNAP["mooring_every"] / 2, cum[-1], OSNAP["mooring_every"]):
                i = int(np.searchsorted(cum, dcum))
                mx, my = piece[i]
                zf = float(floor_at(mx, my))
                if zf > -150:
                    continue
                # a vertical post has no screen direction of its own — give ribbon_quads_z one by nudging x
                pts = np.array([[mx, my], [mx + 0.01, my]]); zz = np.array([zf + 40.0, 0.0])
                for rp, rz, _ in visible_runs(hz, pts, zz):
                    qq, cc, _ = ribbon_quads_z(rp, rz, OSNAP["mooring_lw"], np.append(hex2rgb(PAL["osnap"]), OSNAP["mooring_alpha"]))
                    deep_items.append((qq, cc))

    # formation sites: translucent prisms from the surface down to zb, walls sorted with the floor
    plume_tops, plume_sites = [], []
    for site in PLUMES:
        po = plume_outline(site, floor, xs, ys)
        if po is None:
            continue
        ring, cen, req = po; plume_sites.append((ring, cen, req, site["zb"]))
        sx0, sy0 = project(ring[:, 0], ring[:, 1], 0.0); sx1, sy1 = project(ring[:, 0], ring[:, 1], site["zb"])
        _, dwall = to_view(ring[:, 0], ring[:, 1])
        for i in range(len(ring)):                           # one wall quad per outline segment
            j = (i + 1) % len(ring)
            quads.append(np.array([[sx0[i], sy0[i]], [sx0[j], sy0[j]], [sx1[j], sy1[j]], [sx1[i], sy1[i]]]))
            cols.append(np.append(hex2rgb(PAL["plume"]), 0.11)); depth.append(float((dwall[i] + dwall[j]) / 2))
        plume_tops.append(np.column_stack([sx0, sy0]))

    # paint A+B back to front
    order = np.argsort(-np.asarray(depth))
    cols_rgba = [np.append(c, 1.0) if len(c) == 3 else c for c in cols]
    pc = PolyCollection([quads[i] for i in order], facecolors=[cols_rgba[i] for i in order],
                        edgecolors=[cols_rgba[i] for i in order], linewidths=0.35, zorder=2)
    if out_svg:
        # Illustrator-friendly export: the seafloor (400k quads) becomes one embedded raster at the canvas
        # resolution (× RASTER_SCALE); every line, arrow, label and face stays vector
        fig2 = plt.figure(figsize=(W / 100, H_PX / 100), dpi=100 * RASTER_SCALE)
        ax2 = fig2.add_axes([0, 0, 1, 1]); ax2.set_xlim(0, W); ax2.set_ylim(H_PX, 0); ax2.axis("off")
        ax2.add_collection(pc)
        tmp = os.path.splitext(out_svg)[0] + "_floor.png"
        fig2.savefig(tmp, dpi=100 * RASTER_SCALE, transparent=True); plt.close(fig2)
        ax.imshow(plt.imread(tmp), extent=(0, W, H_PX, 0), zorder=2, interpolation="bilinear")
        print("floor raster", tmp)
    else:
        ax.add_collection(pc)
    if deep_items:
        q3, c3 = [], []
        for qq, cc in deep_items:
            q3 += qq; c3 += cc
        c3 = [np.append(c, 1.0) if len(c) == 3 else c for c in c3]
        ax.add_collection(PolyCollection(q3, facecolors=c3, edgecolors=c3, linewidths=0.35, zorder=2.5))

    # --- C. the cut faces ---------------------------------------------------------------
    def face(xs_, ys_, prof, crust_col):
        # water column: sea surface down to the floor profile; crust: floor down to ZBASE
        sxt, syt = project(xs_, ys_, np.zeros(len(xs_)))
        sxf, syf = project(xs_, ys_, prof)
        sxb, syb = project(xs_, ys_, np.full(len(xs_), ZBASE))
        water = np.vstack([np.column_stack([sxt, syt]), np.column_stack([sxf, syf])[::-1]])
        crust = np.vstack([np.column_stack([sxf, syf]), np.column_stack([sxb, syb])[::-1]])
        ax.add_patch(plt.Polygon(crust, closed=True, fc=crust_col, ec="none", zorder=3))
        if DRAW_WATER:
            ax.add_patch(plt.Polygon(water, closed=True, fc=PAL["face_water"], ec="none", alpha=PAL["face_water_alpha"], zorder=3))
        ax.plot(sxf, syf, color=PAL["ink_soft"], lw=0.8, zorder=3.1)
    # south face (front), west face (left)
    face(np.concatenate([[X0], xs, [X1]]), np.full(nx + 2, Y0), np.concatenate([[floor[0, 0]], floor[0, :], [floor[0, -1]]]), PAL["face_crust"])
    face(np.full(ny + 2, X0), np.concatenate([[Y0], ys, [Y1]]), np.concatenate([[floor[0, 0]], floor[:, 0], [floor[-1, 0]]]), PAL["face_crust_w"])
    if front_items:           # arrow stubs protruding from the front face, over the face
        q4, c4 = [], []
        for qq, cc in front_items:
            q4 += qq; c4 += cc
        c4 = [np.append(c, 1.0) if len(c) == 3 else c for c in c4]
        ax.add_collection(PolyCollection(q4, facecolors=c4, edgecolors=c4, linewidths=0.35, zorder=3.5))
    # block outline
    for (xa, ya), (xb, yb) in [((X0, Y0), (X1, Y0)), ((X0, Y0), (X0, Y1))]:
        for z in (0.0, ZBASE):
            sx, sy = project(np.array([xa, xb]), np.array([ya, yb]), z); ax.plot(sx, sy, color=PAL["ink_soft"], lw=0.6, zorder=6.5)
    for (xa, ya) in [(X0, Y0), (X1, Y0), (X0, Y1)]:
        sx, sy = project(np.array([xa, xa]), np.array([ya, ya]), np.array([0.0, ZBASE])); ax.plot(sx, sy, color=PAL["ink_soft"], lw=0.6, zorder=6.5)

    # --- D. the sea surface, the land plates, then everything on the surface --------------
    cx = np.array([X0, X1, X1, X0]); cy = np.array([Y0, Y0, Y1, Y1])
    sx, sy = project(cx, cy, np.zeros(4))
    if DRAW_WATER:
        ax.add_patch(plt.Polygon(np.column_stack([sx, sy]), closed=True, fc=PAL["water"], ec="none", alpha=PAL["water_alpha"], zorder=4))
    if DRAW_LAND:
        for rings in geojson_polys(os.path.join(DATA, "land_block.geojson")):
            poly_patch(ax, rings, 15.0, PAL["land"], PAL["coast"], 0.5, zorder=5)
        for rings in geojson_polys(os.path.join(DATA, "ice_block.geojson")):
            poly_patch(ax, rings, 16.0, PAL["ice"], None, 0.0, zorder=5.1)
    for top in plume_tops:
        ax.add_patch(plt.Polygon(top, closed=True, fc=PAL["plume"], ec="none", alpha=0.35, zorder=5.5))
    if osnap_line is not None:
        sx, sy = project(osnap_line[:, 0], osnap_line[:, 1], np.zeros(len(osnap_line)))
        ax.plot(sx, sy, color=PAL["osnap"], lw=OSNAP["lw"], ls=OSNAP["dash"], solid_capstyle="butt", zorder=6.2)
    if surface_items:
        q2, c2, d2 = [], [], []
        for qq, cc, dd in surface_items:
            q2 += qq; c2 += cc; d2 += dd
        order = np.argsort(-np.asarray(d2))
        c2 = [np.append(c, 1.0) if len(c) == 3 else c for c in c2]
        ax.add_collection(PolyCollection([q2[i] for i in order], facecolors=[c2[i] for i in order],
                                         edgecolors=[c2[i] for i in order], linewidths=0.35, zorder=6))

    # heat-loss wisps above the plumes (drawn last among scene items; they rise above the surface)
    for ring, (pcx, pcy), r, zb in plume_sites:
        for x, y, u in wisp_paths(pcx, pcy, r):
            pts = np.column_stack([x, y]); segs = np.stack([pts[:-1], pts[1:]], 1)
            lws = WISPS["lw"][0] + (WISPS["lw"][1] - WISPS["lw"][0]) * u[:-1]
            cols = [np.append(hex2rgb(PAL["plume"]), 0.85 * (1 - 0.8 * uu)) for uu in u[:-1]]
            ax.add_collection(LineCollection(segs, colors=cols, linewidths=lws, capstyle="round", zorder=6.4))

    # labels
    for llon, llat, text, kind in LABELS:
        lx, ly = ll_to_block(np.array([llon]), np.array([llat]))
        if not (X0 <= lx[0] <= X1 and Y0 <= ly[0] <= Y1):
            continue
        sx, sy = project(lx, ly, 15.0 if kind == "land" else 0.0)
        style = dict(land=dict(color=PAL["ink"], weight="semibold"), sea=dict(color=PAL["ink_soft"], style="italic"),
                     feature=dict(color=PAL["ink_soft"]), flow=dict(color=PAL["ink"], style="italic"),
                     obs=dict(color=PAL["osnap"], weight="semibold"))[kind]
        txt = ax.text(sx[0], sy[0], text, ha="center", va="center", fontsize=LABEL_PT[kind], family=FONT,
                      linespacing=1.05, zorder=8, **style)
        txt.set_path_effects([pe.withStroke(linewidth=2.2, foreground=PAL["paper"], alpha=0.75)])

    # in/outflow labels: a tick from each stub tip, text on the side away from where the ribbon continues
    for cls, (tx, ty), (nx_, ny_), tz in front_tips:
        sx, sy = project(np.array([tx, nx_]), np.array([ty, ny_]), np.array([tz, tz]))
        side = FRONT_LABEL_SIDE.get(cls, -1 if sx[1] > sx[0] else 1)   # default: the side away from the ribbon
        lx_, ly_ = sx[0] + FRONT_STUB["leader"] * side, sy[0] + FRONT_STUB["drop"]
        ax.plot([sx[0] + 4 * side, lx_], [sy[0] + 3, ly_], color=PAL["ink"], lw=0.6, zorder=8)
        ax.text(lx_ + 4 * side, ly_, FRONT_LABELS[cls], ha="left" if side > 0 else "right",
                va="center", fontsize=LABEL_PT["flow"], family=FONT, color=PAL["ink"], style="italic", zorder=8)

    # --- E. depth scale on the west face, title --------------------------------------------
    for zt in (0, -1000, -2000, -3000, -4000, -5000):
        sx, sy = project(np.array([X0, X0]), np.array([Y0 - 30, Y0]), np.array([zt, zt]))
        ax.plot(sx, sy, color=PAL["ink"], lw=0.7, zorder=7)
        lab = "0 m" if zt == 0 else f"{-zt:,} m"
        ax.text(sx[0] - 6, sy[0], lab, ha="right", va="center", fontsize=9, family=FONT, color=PAL["ink"], zorder=7)
    ax.text(MARGIN[0], 52, "Deep-water formation in the subpolar North Atlantic", fontsize=22, family=FONT,
            weight="semibold", color=PAL["ink"], va="top")
    ax.text(MARGIN[0], 92, f"GEBCO 2024 seafloor, vertical exaggeration {EXAG:.0f}×  ·  VIKING20X 10-year mean currents (2013–2023): "
            f"surface 0–500 m in red, 1,500–2,500 m in blue  ·  schematic flows from Fig. 1.1 v6, sinking in purple  ·  "
            f"dashed: OSNAP observing array  ·  v1 draft",
            fontsize=10.5, family=FONT, color=PAL["ink_soft"], va="top")

    fig.savefig(out_png, dpi=100, facecolor=PAL["paper"])
    print("wrote", out_png)
    if out_svg:
        fig.savefig(out_svg, facecolor=PAL["paper"], metadata={"Creator": "block3d.py"})
        print("wrote", out_svg)


def ramp3(stops, t):
    """Per-vertex colour through three hex stops at t = 0, 0.5, 1."""
    t = np.clip(np.asarray(t, float), 0, 1)
    c = np.array([hex2rgb(s) for s in stops])
    out = np.empty((len(t), 3))
    lo = t < 0.5
    out[lo] = c[0] + (c[1] - c[0]) * (t[lo] / 0.5)[:, None]
    out[~lo] = c[1] + (c[2] - c[1]) * ((t[~lo] - 0.5) / 0.5)[:, None]
    return out


def offshore(piece, floor_fn, **kw):
    """Nudge a deep-arrow path offshore: every vertex over a floor shallower than DEEP_SNAP['target'] walks
    downslope (toward the deepest floor on a ring of radius `look`) until it is over deep water. Vertices
    that cannot get there within `max_km` are left in place. Returns a re-densified, smoothed path."""
    c = dict(DEEP_SNAP); c.update(kw)
    ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    ring = np.column_stack([np.cos(ang), np.sin(ang)]) * c["look"]
    out, ok = [], []
    for x, y in piece:
        x0, y0, moved = x, y, 0.0
        while float(floor_fn(x, y)) > c["target"] and moved < c["max_km"]:
            cand = np.asarray(floor_fn(x + ring[:, 0], y + ring[:, 1]), float)
            k = int(np.argmin(cand))
            if cand[k] >= float(floor_fn(x, y)):
                break                                   # local bowl with nothing deeper around: give up
            x += ring[k, 0] / c["look"] * c["step"]; y += ring[k, 1] / c["look"] * c["step"]; moved += c["step"]
        reached = float(floor_fn(x, y)) <= c["target"]
        if not reached:
            x, y = x0, y0                               # could not reach deep water: keep the drawn position
        if not out or np.hypot(x - out[-1][0], y - out[-1][1]) > 1.0:
            out.append((x, y)); ok.append(reached)
    out, ok = np.array(out), np.array(ok)
    if ok.any():                                        # trim shelf stubs at both ends; keep interior gaps (sills)
        i0, i1 = int(np.argmax(ok)), len(ok) - int(np.argmax(ok[::-1]))
        out = out[i0:i1]
    if len(out) < 2:
        return piece
    return smooth(densify(out), 3)


def join_arrows(recs):
    """Pass over the arrow records: give each a per-vertex width `wv`; where an arrow's end lies within
    JOIN['dist'] km (and JOIN['dz'] m in height) of another arrow's vertex that is not that arrow's own end,
    taper the width over the last JOIN['taper'] km to the other arrow's width and drop the head at the joint."""
    for r in recs:
        r["wv"] = np.full(len(r["piece"]), float(r["w"]))
    for r in recs:
        if r["type"] != "Current":
            continue
        P, zz = r["piece"], r["zz"]
        for end in (0, -1):
            if end == 0 and r["at_face"] and r["inbound"]:
                continue                                    # the inflow stub tip has nothing to join
            E, zE = P[end], zz[end]
            best = None
            for o in recs:
                if o is r or o["type"] != "Current":
                    continue
                d = np.hypot(*(o["piece"] - E).T)
                j = int(np.argmin(d))
                if d[j] >= JOIN["dist"] or abs(o["zz"][j] - zE) >= JOIN["dz"]:
                    continue
                o_ends_here = np.hypot(*(o["piece"][-1] - E)) < JOIN["dist"]
                o_starts_here = np.hypot(*(o["piece"][0] - E)) < JOIN["dist"]
                if o_ends_here or (end == 0 and o_starts_here):
                    continue        # the other arrow arrives here (it adapts, not us), or two arrows start together
                if best is None or d[j] < best[0]:
                    best = (d[j], o["w"])
                    if end == -1 and o_starts_here and d[j] > 1.0:
                        r["bridge"] = (o["piece"][0], float(o["zz"][0]))   # close the gap to the start of the next arrow
            if best is None or abs(best[1] - r["w"]) < 1e-6:
                if best is not None and end == -1:
                    r["head"] = False                       # same width, but the flow continues: no head at the joint
                continue
            w_to = best[1]
            seg = np.hypot(*np.diff(P, axis=0).T)
            cum = np.concatenate([[0.0], np.cumsum(seg)])
            t = cum if end == 0 else cum[-1] - cum          # distance from the joining end
            u = np.clip(t / JOIN["taper"], 0, 1); u = u * u * (3 - 2 * u)
            r["wv"] = np.minimum(r["wv"], w_to + (r["w"] - w_to) * u) if w_to < r["w"] else np.maximum(r["wv"], w_to + (r["w"] - w_to) * u)
            if end == -1:
                r["head"] = False
    for r in recs:                                          # extend arriving arrows to the start of the one they join
        if "bridge" in r:
            (bx, by), bz = r.pop("bridge")
            r["piece"] = np.vstack([r["piece"], [bx, by]]); r["zz"] = np.append(r["zz"], bz); r["wv"] = np.append(r["wv"], r["wv"][-1])
            if r.get("s") is not None:
                r["s"] = np.append(r["s"], r["s"][-1])
            if np.ndim(r.get("colr", 0)) == 2:
                r["colr"] = np.vstack([r["colr"], r["colr"][-1]])


def conv_profile(piece, z_deep):
    """Descent fraction s (0 at the surface end, 1 at depth) and height along a Convection arrow."""
    seg = np.hypot(*np.diff(piece, axis=0).T)
    t = np.concatenate([[0.0], np.cumsum(seg)]) / max(seg.sum(), 1e-9)
    a, b = CONV_FADE
    s = np.clip((t - a) / (b - a), 0, 1)
    s = s * s * (3 - 2 * s)                     # smoothstep
    return s, s * z_deep


def load_osnap():
    """The OSNAP section as block coordinates (km), west to east."""
    d = np.loadtxt(OSNAP_CSV, delimiter=",", skiprows=2)
    bx, by = ll_to_block(d[:, 0], d[:, 1])
    return np.column_stack([bx, by])


def wisp_paths(cx, cy, r):
    """Heat-loss wisps above a plume: screen-space wavy lines rising from the surface (Fig. 1.1 depth panel)."""
    sx0, sy0 = project(cx, cy, 0.0)
    _, sy1 = project(cx, cy, WISPS["height"])
    h = sy0 - sy1
    out = []
    for i in range(WISPS["n"]):
        off = (i - (WISPS["n"] - 1) / 2) * WISPS["spread"] * r * SCALE
        u = np.linspace(0, 1, 40)
        x = sx0 + off + np.sin(u * 2 * np.pi * 1.25 + i * 1.3) * WISPS["amp"] * r * SCALE * (0.4 + 0.6 * u)
        y = sy0 - u * h * (0.8 + 0.2 * (i % 2))
        out.append((x, y, u))
    return out


def smooth_z(zz, n=10):
    """Running mean along a line, so ribbons that follow the floor do not inherit its roughness."""
    zz = np.asarray(zz, float)
    if len(zz) < 3:
        return zz
    k = np.ones(min(n, len(zz)) * 2 + 1); k /= k.sum()
    pad = len(k) // 2
    return np.convolve(np.pad(zz, pad, mode="edge"), k, mode="valid")


def ribbon_quads_z(line, zz, width_px, color):
    """ribbon_quads with a per-vertex z (ribbons that follow the floor)."""
    zz = smooth_z(zz)
    sx, sy = project(line[:, 0], line[:, 1], zz)
    _, d = to_view(line[:, 0], line[:, 1])
    dx, dy = np.diff(sx), np.diff(sy)
    L = np.hypot(dx, dy) + 1e-9
    nx_, ny_ = -dy / L, dx / L
    w = np.asarray(width_px, float)
    if w.ndim == 0:
        w = np.full(len(sx), float(w))
    w0, w1 = w[:-1] / 2, w[1:] / 2
    ex, ey = dx / L * 0.6, dy / L * 0.6
    quads = np.stack([
        np.stack([sx[:-1] + nx_ * w0 - ex, sy[:-1] + ny_ * w0 - ey], -1),
        np.stack([sx[1:] + nx_ * w1 + ex, sy[1:] + ny_ * w1 + ey], -1),
        np.stack([sx[1:] - nx_ * w1 + ex, sy[1:] - ny_ * w1 + ey], -1),
        np.stack([sx[:-1] - nx_ * w0 - ex, sy[:-1] - ny_ * w0 - ey], -1),
    ], 1)
    col = np.asarray(color, float)
    if col.ndim == 1:
        col = np.tile(col, (len(quads), 1))
    else:
        col = (col[:-1] + col[1:]) / 2
    # the painter key: view depth, nudged forward by height above the floor is not needed —
    # lines float above the floor, so equal depth draws them after the quad beneath (stable sort)
    return list(quads), list(col), list((d[:-1] + d[1:]) / 2 - 0.001)


if __name__ == "__main__":
    args = sys.argv[1:]
    out = args[0] if args and not args[0].startswith("--") else os.path.join(HERE, "..", "v1", "block_v1.png")
    svg = args[args.index("--svg") + 1] if "--svg" in args else None
    render(out, svg, fast="--fast" in args, viking="--no-viking" not in args, arrows="--no-arrows" not in args)
