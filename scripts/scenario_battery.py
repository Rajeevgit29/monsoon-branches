"""Shocks beyond the two official ones: does the model react in the physically expected direction?

  python scripts/scenario_battery.py --artifacts artifacts --out reports/scenarios.json

Each scenario: a season, a box, a change to the last observed day, and an expectation written down before running
(field, region, days, sign). It runs on evenly spaced windows of that season in the tuning years 2018-2019 (never the
test years). Reported per scenario: the mean change of the expected quantity, the share of windows with the expected
sign, and the official causality checks (localised on day 1, stable, propagating).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from monsoon_branches.data import load_archive, window_starts  # noqa: E402
from monsoon_branches.grid import LAT, LON, box_mask  # noqa: E402
from monsoon_branches.model import Forecaster  # noqa: E402

T2M, TP, U10, V10, MSL, TCWV = range(6)
JJAS, MAMJ, DJF = (6, 7, 8, 9), (3, 4, 5, 6), (12, 1, 2)
CENTRAL_INDIA = (18, 26, 76, 86)

# name, months, box, perturbation(last_day, mask) -> delta, expectation (field, region box or None = shock box, days, sign)
SCENARIOS = [
    ("North-west heatwave, +5 K", MAMJ, (24, 30, 70, 78),
     lambda x, m: {T2M: 5.0 * m}, (T2M, None, (1, 2, 3), +1), "the heat persists for days, then decays"),
    ("Winter cold wave over the north, −5 K", DJF, (26, 32, 72, 84),
     lambda x, m: {T2M: -5.0 * m}, (T2M, None, (1, 2, 3), -1), "the cold persists for days, then decays"),
    ("Dry air over central India, vapour −25%", JJAS, CENTRAL_INDIA,
     lambda x, m: {TCWV: -0.25 * x[TCWV] * m}, (TP, None, (1, 2, 3), -1), "less rain where the column dried"),
    ("Moisture surge on the west coast, vapour +20%", JJAS, (12, 20, 70, 76),
     lambda x, m: {TCWV: 0.20 * x[TCWV] * m}, (TP, None, (1, 2), +1), "more coastal rain"),
    ("Drier Bay of Bengal, vapour −20%", JJAS, (8, 20, 82, 94),
     lambda x, m: {TCWV: -0.20 * x[TCWV] * m}, (TP, None, (1, 2), -1), "less Bay rain (mirror of the warm-ocean shock)"),
    ("Stronger monsoon winds, Arabian Sea +40%", JJAS, (8, 20, 65, 75),
     lambda x, m: {U10: 0.4 * x[U10] * m, V10: 0.4 * x[V10] * m}, (TP, CENTRAL_INDIA, (2, 4, 6, 8), +1),
     "more central-India rain on days 2-8 (mirror of the monsoon stall; ERA5 sign)"),
    ("Depression at the head of the Bay, −6 hPa", JJAS, (18, 22, 86, 92),
     lambda x, m: {MSL: -600.0 * m}, ("vort", None, (1,), +1), "cyclonic (anticlockwise) winds around the low on day 1"),
    ("Depression at the head of the Bay, −6 hPa (rain)", JJAS, (18, 22, 86, 92),
     lambda x, m: {MSL: -600.0 * m}, (TP, "near", (1, 2, 3), +1), "more rain around the low"),
    ("Cloudburst over Mumbai, +100 mm/day", JJAS, (18, 20, 72, 74),
     lambda x, m: {TP: 100.0 * m}, ("decay", None, (1, 3), +1), "the extra rain is local and fades by day 3, no blow-up"),
]


def region(box, shock_box):
    from monsoon_r1.scoring import near_mask
    if box is None:
        return box_mask(shock_box)
    if box == "near":
        return near_mask(shock_box)
    return box_mask(box)


def vorticity(u, v):
    """Relative vorticity (1/s) on the 1-degree grid, centred differences; latitude ascends with row index."""
    a = 6.371e6
    lat = np.deg2rad(LAT)[:, None]
    dvdx = np.gradient(v, axis=-1) / (a * np.cos(lat) * np.deg2rad(1.0))
    dudy = np.gradient(u, axis=-2) / (a * np.deg2rad(1.0)) * np.sign(LAT[1] - LAT[0])
    return dvdx - dudy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT.parent / "data" / "india_climate_train.nc"))
    ap.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    ap.add_argument("--windows", type=int, default=12)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from monsoon_r1.scoring import causality_checks
    torch.set_num_threads(8)
    arch = load_archive(a.data)
    std = arch.data.std(axis=(0, 2, 3)).astype("f4")
    m = Forecaster.load(a.artifacts)
    months = pd.DatetimeIndex(arch.dates).month.values
    rep = []
    for name, season, box, pert, (field, reg, days, sign), why in SCENARIOS:
        idx = window_starts(arch, "tune", 10, 1)
        idx = idx[np.isin(months[idx], season)]
        idx = idx[np.linspace(0, len(idx) - 1, min(a.windows, len(idx))).astype(int)]
        bm = box_mask(box)
        inp = np.stack([arch.data[i - 2:i + 1] for i in idx]).astype("f4")
        x = inp.copy()
        for w in range(len(idx)):
            for ch, d in pert(inp[w, -1], bm.astype("f4")).items():
                x[w, -1, ch] += d
        doy = torch.from_numpy(arch.doy[idx])
        with torch.inference_mode():
            base = m.rollout(torch.from_numpy(inp), doy, 10, calibrate=True).numpy()
            shock = m.rollout(torch.from_numpy(x), doy, 10, calibrate=True).numpy()
        d = shock - base
        rm = region(reg, box)
        if field == "vort":
            val = np.array([(vorticity(shock[w, 0, U10], shock[w, 0, V10]) - vorticity(base[w, 0, U10], base[w, 0, V10]))[rm].mean()
                            for w in range(len(idx))]) * 1e5
            unit = "1e-5/s"
        elif field == "decay":
            day1 = d[:, 0, TP][:, rm].mean(1); day3 = d[:, 2, TP][:, rm].mean(1)
            val = day1 - day3                      # positive = fades
            unit = "mm/day (day 1 − day 3)"
        else:
            val = np.stack([d[:, k - 1, field][:, rm].mean(1) for k in days]).mean(0)
            unit = "K" if field == T2M else "mm/day"
        checks = [causality_checks(base[w], shock[w], box, std) for w in range(len(idx))]
        row = {"scenario": name, "expect": why, "windows": int(len(idx)),
               "mean_change": float(val.mean()), "unit": unit, "share_expected_sign": float(np.mean(np.sign(val) == sign)),
               "direction_ok": bool(np.sign(val.mean()) == sign),
               "localized": float(np.mean([c["localized"] for c in checks])), "stable": float(np.mean([c["stable"] for c in checks])),
               "propagates": float(np.mean([c["propagates"] for c in checks])),
               "max_abs_std": float(np.abs(d / std[None, None, :, None, None]).max())}
        if field in (T2M,):
            row["decay_day1_to_day5"] = float(d[:, 4, T2M][:, rm].mean() / max(abs(d[:, 0, T2M][:, rm].mean()), 1e-6) * sign)
        rep.append(row)
        print(f"{'OK ' if row['direction_ok'] else 'NO '} {name:52s} {row['mean_change']:+8.2f} {unit:24s} expected sign in "
              f"{row['share_expected_sign']:.0%} | local {row['localized']:.0%} stable {row['stable']:.0%} spreads {row['propagates']:.0%}"
              + (f" | day-5 keeps {row['decay_day1_to_day5']:.0%} of day 1" if 'decay_day1_to_day5' in row else ""))
    print(f"{sum(r['direction_ok'] for r in rep)}/{len(rep)} scenarios respond in the expected direction")
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
