"""Build the dashboard's bundled data: a few held-out example windows and a clipped coastline.

  python scripts/build_dashboard_data.py --data ../data/india_climate_train.nc \
      --coast /path/to/ne_50m_coastline.geojson

Examples come from the 2020-2021 test years (never used for training or checkpoint selection): three input
days plus the ten observed days that follow, stored as base64 float32 so the browser can compare the
forecast with what actually happened. Coastlines (Natural Earth, public domain) are clipped to the grid;
no political boundaries are drawn.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from monsoon_branches.data import load_archive  # noqa: E402

# (id, title, last input day, note, suggested shock)
EXAMPLES = [
    ("konkan-floods", "Konkan floods", "2021-07-19", "Active monsoon; extreme rain on the Konkan coast 21-23 July.", "monsoon_stall"),
    ("amphan", "Cyclone Amphan", "2020-05-17", "Super cyclone over the Bay of Bengal; landfall in West Bengal on 20 May.", "warm_ocean"),
    ("kerala-onset", "Monsoon onset", "2020-05-31", "Monsoon reaches Kerala on 1 June.", "monsoon_stall"),
    ("tauktae", "Cyclone Tauktae", "2021-05-13", "Arabian Sea cyclone tracking up the west coast; landfall in Gujarat on 17 May.", "monsoon_stall"),
    ("heatwave", "North India heatwave", "2020-05-24", "Pre-monsoon heatwave over the northwest plains, 25-28 May.", "warm_ocean"),
    ("western-disturbance", "Western disturbance", "2021-01-02", "Winter trough bringing rain and snow to the north, 4-6 January.", "warm_ocean"),
]


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode("ascii")


def clip_coast(path: Path, lon0=65.0, lon1=101.0, lat0=6.0, lat1=38.0, pad=0.6):
    gj = json.loads(Path(path).read_text())
    lines = []
    for f in gj["features"]:
        g = f["geometry"]
        parts = g["coordinates"] if g["type"] == "MultiLineString" else [g["coordinates"]]
        for part in parts:
            cur = []
            for lon, lat in part:
                if lon0 - pad <= lon <= lon1 + pad and lat0 - pad <= lat <= lat1 + pad:
                    cur.append([round(lon, 2), round(lat, 2)])
                elif cur:
                    if len(cur) > 1:
                        lines.append(cur)
                    cur = []
            if len(cur) > 1:
                lines.append(cur)
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT.parent / "data" / "india_climate_train.nc"))
    ap.add_argument("--coast", default=None)
    ap.add_argument("--out", default=str(ROOT / "dashboard" / "data"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    arch = load_archive(a.data)
    items = []
    for eid, title, last, note, shock in EXAMPLES:
        i = int(np.where(arch.dates == np.datetime64(last))[0][0])
        assert arch.years[i] >= 2020, "examples must come from the held-out test years"
        win = arch.data[i - 2:i + 11]                       # 3 inputs + 10 observed days
        items.append({"id": eid, "title": title, "last_input_day": last, "note": note, "shock": shock,
                      "doy": float(arch.doy[i]),
                      "dates": [str(d) for d in arch.dates[i - 2:i + 11]],
                      "shape": list(win.shape), "b64": b64(win)})
    meta = {"lat": [6.5, 37.5], "lon": [65.5, 100.5], "channels": ["t2m", "tp", "u10", "v10", "msl", "tcwv"],
            "land_sea_mask": b64(arch.static["land_sea_mask"]), "examples": items}
    (out / "examples.json").write_text(json.dumps(meta))
    print(f"examples.json: {len(items)} windows, {(out / 'examples.json').stat().st_size / 1e6:.1f} MB")
    if a.coast:
        lines = clip_coast(Path(a.coast))
        (out / "coast.json").write_text(json.dumps(lines, separators=(",", ":")))
        print(f"coast.json: {len(lines)} polylines, {(out / 'coast.json').stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
