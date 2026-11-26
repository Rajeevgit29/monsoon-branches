"""Build the data behind the dashboard's "Forecast check" view (dashboard/data/check.json).

  python scripts/build_dashboard_check.py --data ../data/india_climate_train.nc --artifacts artifacts

Two parts:
- the date-average reference (the scorer's calendar-day climatology of the training years 2005-2017) for the ten
  forecast days of every bundled example, so the browser can compare the live forecast with both baselines on each event;
- error by forecast day over every test window (2020-2021, a window every 3rd day, as evaluate.py): latitude-weighted
  RMSE of the model, of repeating the last day (persistence) and of the date average (climatology), for the fields the
  dashboard shows, plus heavy-rain CSI (rain > 10 mm/day).
The test years were already scored for the README (reports/eval_test.json); this only breaks the same evaluation down
by day. Nothing here feeds back into training or model selection.
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
from evaluate import ModelRunner, gather  # noqa: E402
from monsoon_branches.data import calendar_slot, load_archive, reference_climatology, window_starts  # noqa: E402
from monsoon_branches.grid import LAT  # noqa: E402

STEPS, HEAVY = 10, 10.0
W_LAT = (np.cos(np.deg2rad(LAT)) / np.cos(np.deg2rad(LAT)).mean())[:, None]   # the scorer's latitude weights


def fields(x):
    """The dashboard's display fields from [..., 6, H, W] arrays in the kit's units."""
    return {"t2m": x[..., 0, :, :] - 273.15, "tp": x[..., 1, :, :], "wind": np.hypot(x[..., 2, :, :], x[..., 3, :, :]),
            "msl": x[..., 4, :, :] / 100.0, "tcwv": x[..., 5, :, :]}


def rmse_by_lead(pred, true):
    """[W, 10, H, W] -> [10]: latitude-weighted RMSE pooled over windows, as the scorer's rmse_lat."""
    return np.sqrt((((pred - true) ** 2) * W_LAT).mean(axis=(0, 2, 3)))


def csi_by_lead(pred, true):
    p, t = pred > HEAVY, true > HEAVY
    h, m, f = (p & t).sum(axis=(0, 2, 3)), (~p & t).sum(axis=(0, 2, 3)), (p & ~t).sum(axis=(0, 2, 3))
    return np.where(h + m + f > 0, h / np.maximum(h + m + f, 1), np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT.parent / "data" / "india_climate_train.nc"))
    ap.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    ap.add_argument("--examples", default=str(ROOT / "dashboard" / "data" / "examples.json"))
    ap.add_argument("--out", default=str(ROOT / "dashboard" / "data" / "check.json"))
    ap.add_argument("--stride", type=int, default=3)
    a = ap.parse_args()

    arch = load_archive(a.data)
    clim_ref = reference_climatology(arch, "train")
    ex = json.loads(Path(a.examples).read_text())["examples"]
    per_example = {}
    for e in ex:
        slots = [calendar_slot(np.datetime64(d)) for d in e["dates"][3:3 + STEPS]]
        per_example[e["id"]] = base64.b64encode(clim_ref[slots].astype("<f4").tobytes()).decode()

    idx = window_starts(arch, "test", STEPS, a.stride)
    inp, tru, doy, slots = gather(arch, idx)
    pred = ModelRunner(a.artifacts)(inp.astype("f4"), doy)
    persist = np.repeat(inp[:, -1:], STEPS, axis=1)
    clim = clim_ref[slots]
    F = {k: fields(v) for k, v in (("model", pred), ("persist", persist), ("clim", clim), ("obs", tru))}
    rmse = {f: {m: [round(float(v), 4) for v in rmse_by_lead(F[m][f], F["obs"][f])] for m in ("model", "persist", "clim")}
            for f in F["obs"]}
    csi = {m: [round(float(v), 4) for v in csi_by_lead(F[m]["tp"], F["obs"]["tp"])] for m in ("model", "persist", "clim")}
    rep = json.loads((ROOT / "reports" / "eval_test.json").read_text())
    out = {"clim": per_example,
           "test": {"years": "2020-2021", "windows": int(len(idx)), "rmse": rmse, "csi": csi,
                    "mean_skill": round(rep["mean_ss"], 4), "mean_csi_skill": round(rep["mean_csi_skill"], 4)}}
    Path(a.out).write_text(json.dumps(out, separators=(",", ":")))
    print(f"{len(per_example)} examples, {len(idx)} test windows -> {a.out} ({Path(a.out).stat().st_size / 1e6:.1f} MB)")
    for f in rmse:
        best = [min(p, c) for p, c in zip(rmse[f]["persist"], rmse[f]["clim"])]
        print(f"  {f:5s} model beats both baselines on days", [k + 1 for k in range(STEPS) if rmse[f]["model"][k] < best[k]])
    print("  CSI model", csi["model"], "\n  CSI persist", csi["persist"], "\n  CSI clim", csi["clim"])


if __name__ == "__main__":
    main()
