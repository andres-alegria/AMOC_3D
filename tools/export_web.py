"""
Export the block's data for the interactive page (../web/data/).

    python3 export_web.py

Writes
    floor.f32       float32 heightfield, row 0 = south, metres (land clamped to 0)
    meta.json       grid geometry + block extent
    lines.json      VIKING ribbons (both bands), schematic arrows, plume sites — km, z in metres
    land.json / ice.json   Natural Earth polygons in km (from ../data/*.geojson)
Reuses the tracing and lifting code of block3d.py, so the page and the static
render show the same lines.
"""
import os, json, shutil
import numpy as np
import block3d as B

OUT = os.path.join(B.HERE, "..", "web", "data")
os.makedirs(OUT, exist_ok=True)


def floor_fn(floor):
    def f(x, y):
        return B.bilinear(floor, np.asarray(x) / B.CELL - 0.5 - B.X0 / B.CELL, np.asarray(y) / B.CELL - 0.5 - B.Y0 / B.CELL)
    return f


def main():
    H, xs, ys = B.load_dem()
    floor = np.minimum(H, 0.0)
    ny, nx = floor.shape
    floor.astype("<f4").tofile(os.path.join(OUT, "floor.f32"))
    json.dump(dict(nx=nx, ny=ny, x0=B.X0, y0=B.Y0, x1=B.X1, y1=B.Y1, cell=B.CELL, zbase=B.ZBASE,
                   proj=B.PROJ4, source="GEBCO 2024 sub-ice, 5 km average; land clamped to 0"),
              open(os.path.join(OUT, "meta.json"), "w"), indent=1)
    fl = floor_fn(floor)

    lon, lat, north = B.lonlat_grid(xs, ys)
    out = dict(bands={}, arrows=[], plumes=[])
    bands = dict(B.BANDS)
    bands["intermediate"] = (-850.0, "mid", 55, 44, 150)
    for name, (z, pal, pct, dsep, minlen) in bands.items():
        U, V, wet = B.viking_band(name, lon, lat, north)
        lines, thr, vmax = B.streamlines(U, V, wet & (floor < -50), floor, dsep, minlen, pct)
        recs = []
        for line, sp in lines:
            wl = line + np.array([B.X0, B.Y0])
            zz = np.zeros(len(wl)) if z >= 0 else B.smooth_z(np.maximum(z, fl(wl[:, 0], wl[:, 1]) + B.CLEAR))
            recs.append(dict(p=np.round(np.column_stack([wl, zz]), 1).tolist(),
                             s=np.round(sp / vmax, 3).tolist()))
        out["bands"][name] = dict(z=z, vmax=float(vmax), lines=recs)
        print(name, len(recs), "ribbons")

    g = json.load(open(B.ARROWS))
    cls_of = dict(Red="surface", Blue="deep", Convection="conv")
    for f in g["features"]:
        p = f["properties"]
        if p["Notes"] not in cls_of:
            continue
        for part in f["geometry"]["coordinates"]:
            c = np.array(part)
            bx, by = B.ll_to_block(c[:, 0], c[:, 1])
            for piece in B.clip_to_block(np.column_stack([bx, by])):
                piece = B.smooth(B.densify(piece))
                rec = dict(id=p["id"], cls=cls_of[p["Notes"]], type=p["Type"], w=int(p["Thickness"]))
                if p["Notes"] == "Red":
                    zz = np.zeros(len(piece))
                elif p["Notes"] == "Blue":
                    zz = B.smooth_z(np.maximum(-2000.0, fl(piece[:, 0], piece[:, 1]) + B.CLEAR))
                else:   # the sinking ribbon: surface → deep along its length, with the descent fraction s for the colour
                    z_deep = B.smooth_z(np.maximum(-2000.0, fl(piece[:, 0], piece[:, 1]) + B.CLEAR))
                    s, zz = B.conv_profile(piece, z_deep)
                    rec["s"] = np.round(s, 3).tolist()
                rec["p"] = np.round(np.column_stack([piece, zz]), 1).tolist()
                out["arrows"].append(rec)
    print(len(out["arrows"]), "arrow pieces")

    # the OSNAP section: surface line pieces + mooring posts (x, y, floor m)
    out["osnap"] = dict(lines=[], moorings=[])
    for piece in B.clip_to_block(B.load_osnap()):
        piece = B.densify(piece, 8.0)
        out["osnap"]["lines"].append(np.round(piece, 1).tolist())
        seg = np.hypot(*np.diff(piece, axis=0).T); cum = np.concatenate([[0.0], np.cumsum(seg)])
        for dcum in np.arange(B.OSNAP["mooring_every"] / 2, cum[-1], B.OSNAP["mooring_every"]):
            i = int(np.searchsorted(cum, dcum)); mx, my = piece[i]; zf = float(fl(mx, my))
            if zf <= -150:
                out["osnap"]["moorings"].append([round(float(mx), 1), round(float(my), 1), round(zf + 40.0)])
    print(len(out["osnap"]["moorings"]), "moorings")

    out["wisps"] = dict(B.WISPS)
    out["labels"] = []
    for llon, llat, text, kind in B.LABELS:
        lx, ly = B.ll_to_block(np.array([llon]), np.array([llat]))
        if B.X0 <= lx[0] <= B.X1 and B.Y0 <= ly[0] <= B.Y1:
            out["labels"].append(dict(x=round(float(lx[0]), 1), y=round(float(ly[0]), 1), text=text, kind=kind))

    for plon, plat, r, zb in B.PLUMES:
        px, py = B.ll_to_block(np.array([plon]), np.array([plat]))
        out["plumes"].append(dict(x=round(float(px[0]), 1), y=round(float(py[0]), 1), r=r, zb=zb, lon=plon, lat=plat))

    json.dump(out, open(os.path.join(OUT, "lines.json"), "w"), separators=(",", ":"))

    for src, dst in (("land_block.geojson", "land.json"), ("ice_block.geojson", "ice.json")):
        polys = B.geojson_polys(os.path.join(B.DATA, src))
        json.dump([[np.round(r, 1).tolist() for r in rings] for rings in polys],
                  open(os.path.join(OUT, dst), "w"), separators=(",", ":"))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
