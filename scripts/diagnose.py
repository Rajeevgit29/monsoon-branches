"""Where does the forecast error come from? Skill by region, by distance from the domain edge, by season, and rain
bias by intensity, on the tuning years.

  python scripts/diagnose.py --artifacts ../runs/v1 [--split tune]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from monsoon_branches.data import calendar_slot, load_archive, reference_climatology, window_starts  # noqa: E402
from monsoon_branches.grid import CHANNELS, LAT, LON, lat_weights  # noqa: E402
from monsoon_branches.model import Forecaster  # noqa: E402


def regions(lsm, oro):
    la, lo = np.meshgrid(LAT, LON, indexing="ij")
    land, sea = lsm >= 0.5, lsm < 0.5
    return {
        "Western Ghats + Konkan (windward)": land & (la >= 8) & (la <= 21) & (lo <= 76.5) & (oro > 100),
        "Deccan interior (rain shadow)": land & (la >= 12) & (la <= 21) & (lo >= 76.5) & (lo <= 81),
        "Indo-Gangetic plain": land & (la >= 24) & (la <= 30) & (lo >= 75) & (lo <= 88) & (oro < 500),
        "Himalaya + Tibet (oro > 1500 m)": land & (oro > 1500),
        "NE India + Myanmar hills": land & (la >= 22) & (lo >= 89) & (oro <= 1500),
        "NW desert (Thar)": land & (la >= 23) & (la <= 30) & (lo <= 75),
        "Arabian Sea": sea & (lo <= 76),
        "Bay of Bengal": sea & (lo >= 80) & (lo <= 95) & (la >= 8),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT.parent / "data" / "india_climate_train.nc"))
    ap.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    ap.add_argument("--split", default="tune")
    ap.add_argument("--calibrate", action="store_true")
    a = ap.parse_args()
    arch = load_archive(a.data)
    m = Forecaster.load(a.artifacts)
    idx = window_starts(arch, a.split, 10, 1)
    inp = np.stack([arch.data[i - 2:i + 1] for i in idx]); tru = np.stack([arch.data[i + 1:i + 11] for i in idx])
    clim = reference_climatology(arch, "train")[np.array([[calendar_slot(arch.dates[i + k]) for k in range(1, 11)] for i in idx])]
    per = np.repeat(inp[:, -1:], 10, 1)
    torch.set_num_threads(8)
    with torch.inference_mode():
        pred = np.concatenate([m.rollout(torch.from_numpy(inp[s:s + 128]), torch.from_numpy(arch.doy[idx[s:s + 128]]), 10,
                                         calibrate=a.calibrate).numpy() for s in range(0, len(idx), 128)])
    lsm, oro = arch.static["land_sea_mask"], arch.static["orography_m"]
    R = regions(lsm, oro)
    out = {}
    print(f"{len(idx)} windows, {a.split}. Skill = 1 - RMSE/min(persistence, climatology) per region, pooled leads 1/4/8")
    L = [0, 3, 7]
    mse = lambda p: ((p[:, L] - tru[:, L]) ** 2).mean(axis=(0, 1))          # [6, 32, 36]
    mm, mp, mc = mse(pred), mse(per), mse(clim)
    print(f"{'region':36s} cells " + " ".join(f"{c:>6s}" for c in CHANNELS))
    for name, msk in R.items():
        ss = [1 - np.sqrt(mm[c][msk].mean()) / min(np.sqrt(mp[c][msk].mean()), np.sqrt(mc[c][msk].mean())) for c in range(6)]
        out[name] = ss
        print(f"{name:36s} {msk.sum():5d} " + " ".join(f"{s:+.3f}" for s in ss))
    # distance from the lateral boundary
    ii, jj = np.indices((32, 36)); edge = np.minimum.reduce([ii, 31 - ii, jj, 35 - jj])
    print("\nby distance from the domain edge (cells):")
    for lo_, hi_ in ((0, 0), (1, 2), (3, 5), (6, 20)):
        msk = (edge >= lo_) & (edge <= hi_)
        ss = [1 - np.sqrt(mm[c][msk].mean()) / min(np.sqrt(mp[c][msk].mean()), np.sqrt(mc[c][msk].mean())) for c in range(6)]
        rel8 = [np.sqrt(((pred[:, 7, c] - tru[:, 7, c]) ** 2).mean(0)[msk].mean()) / np.sqrt(((clim[:, 7, c] - tru[:, 7, c]) ** 2).mean(0)[msk].mean()) for c in range(6)]
        print(f"  {lo_}-{hi_:<3d} {msk.sum():4d} cells  skill " + " ".join(f"{s:+.3f}" for s in ss) + "   day-8 RMSE/clim " + " ".join(f"{r:.2f}" for r in rel8))
    # rain intensity bias
    print("\nrain: mean forecast vs observed, binned by observed day-1 / day-4 rain (mm/day)")
    for k in (0, 3):
        t, p = tru[:, k, 1].ravel(), pred[:, k, 1].ravel()
        row = []
        for lo_, hi_ in ((0, 1), (1, 5), (5, 10), (10, 25), (25, 50), (50, 1e9)):
            s = (t >= lo_) & (t < hi_)
            row.append(f"[{lo_:g},{hi_ if hi_ < 1e8 else 'inf'}) obs {t[s].mean():5.1f} fc {p[s].mean():5.1f} n={s.sum()}")
        print(f"  day {k + 1}: " + " | ".join(row))
    # field-wide bias
    print("\nmean bias (forecast - observed) at day 1 / 4 / 8: " + " ".join(
        f"{c}=" + "/".join(f"{(pred[:, k, i] - tru[:, k, i]).mean():+.2f}" for k in L) for i, c in enumerate(CHANNELS)))
    Path(ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / f"diagnose_{Path(a.artifacts).name}_{a.split}.json").write_text(json.dumps({k: [float(x) for x in v] for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    main()
