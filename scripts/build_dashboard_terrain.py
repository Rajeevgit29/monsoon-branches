"""Build the dashboard's 3D terrain (dashboard/data/terrain.json): ERA5 surface height and land-sea mask at 0.25 degrees.

  python scripts/build_dashboard_terrain.py

Source: the static fields of the 0.25-degree WeatherBench 2 ERA5 archive (the same ERA5 the kit uses, CC BY 4.0), cut to
the forecast window 6-38 N, 65-101 E. Display only: the model and server use the kit's 1-degree fields, never this file.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://storage.googleapis.com/weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
G = 9.80665


def fetch(var: str, shape: tuple[int, ...]) -> np.ndarray:
    import requests
    from numcodecs import Blosc
    key = ".".join("0" * len(shape))
    r = requests.get(f"{BASE}/{var}/{key}", timeout=60)
    r.raise_for_status()
    return np.frombuffer(Blosc().decode(r.content), dtype="<f4").reshape(shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "dashboard" / "data" / "terrain.json"))
    a = ap.parse_args()
    lat, lon = fetch("latitude", (721,)), fetch("longitude", (1440,))
    z, lsm = fetch("geopotential_at_surface", (721, 1440)), fetch("land_sea_mask", (721, 1440))
    if lat[0] > lat[-1]:                                   # store south -> north, like the forecast grid
        lat, z, lsm = lat[::-1], z[::-1], lsm[::-1]
    i = np.where((lat >= 6 - 1e-6) & (lat <= 38 + 1e-6))[0]
    j = np.where((lon >= 65 - 1e-6) & (lon <= 101 + 1e-6))[0]
    h = np.clip(z[np.ix_(i, j)] / G, 0, None)               # metres above sea level; the sea is 0
    m = lsm[np.ix_(i, j)]
    out = {"lat0": float(lat[i[0]]), "lon0": float(lon[j[0]]), "step": 0.25, "nlat": int(len(i)), "nlon": int(len(j)),
           "source": "ERA5 surface geopotential / g and land-sea mask, 0.25 deg, via WeatherBench 2 (CC BY 4.0)",
           "height_m": base64.b64encode(np.round(h).astype("<i2").tobytes()).decode(),
           "land": base64.b64encode(np.round(m * 255).astype("u1").tobytes()).decode()}
    Path(a.out).write_text(json.dumps(out, separators=(",", ":")))
    print(f"{len(i)} x {len(j)} points, {h.max():.0f} m max -> {a.out} ({Path(a.out).stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
