"""Fit the per-lead rain gain on the TUNING years and store it in artifacts/model.npz.

  python scripts/calibrate_rain.py --data ../data/india_climate_train.nc --artifacts artifacts

An MSE-trained rollout damps rain amplitudes as the lead grows, so fewer cells cross the 10 mm/day heavy-rain
threshold than in reality. The gain g(k) = g1 + (g10 - g1) (k - 1) / 9 rescales only the RETURNED rain below the
10 mm/day knee and shifts it above (slope 1, see model.calibrate_rain); the state fed back into the rollout is
unchanged, so temperature, wind, pressure and vapour forecasts are identical.
The two numbers are chosen on a grid to maximise an UNCAPPED version of the forecast score
(30 x skill / 0.12 + 15 x CSI-skill / 0.10 + anti-blur), so the fit cannot exploit the caps of the real scorer.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from monsoon_branches.data import calendar_slot, load_archive, reference_climatology, window_starts  # noqa: E402
from monsoon_branches.model import Forecaster, calibrate_rain  # noqa: E402
from monsoon_branches import scoreboard as SB  # noqa: E402


def objective(t):
    return 30 * t["mean_ss"] / 0.12 + 15 * t["mean_csi_skill"] / 0.10 + t["anti_blur_pts"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT.parent / "data" / "india_climate_train.nc"))
    ap.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    ap.add_argument("--split", default="tune")
    ap.add_argument("--mos", action=argparse.BooleanOptionalAction, default=True,
                    help="also fit the per-lead drift removal and anomaly scaling for the non-rain fields")
    a = ap.parse_args()
    arch = load_archive(a.data)
    m = Forecaster.load(a.artifacts)
    idx = window_starts(arch, a.split, 10, 2)
    inp = np.stack([arch.data[i - 2:i + 1] for i in idx]); tru = np.stack([arch.data[i + 1:i + 11] for i in idx])
    clim = reference_climatology(arch, "train")[np.array([[calendar_slot(arch.dates[i + k]) for k in range(1, 11)] for i in idx])]
    per = np.repeat(inp[:, -1:], 10, 1)
    torch.set_num_threads(8)
    with torch.inference_mode():
        raw = np.concatenate([m.rollout(torch.from_numpy(inp[s:s + 64]), torch.from_numpy(arch.doy[idx[s:s + 64]]), 10).numpy()
                              for s in range(0, len(idx), 64)])
    lead = np.arange(10)
    best = None
    grid1 = np.round(np.arange(1.0, 1.81, 0.1), 2)
    grid10 = np.round(np.arange(0.7, 2.01, 0.1), 2)
    base = SB.forecast_table(raw, tru, per, clim)
    print(f"raw: ss {base['mean_ss']:+.4f} csi-skill {base['mean_csi_skill']:+.4f} blur {base['anti_blur_pts']:.2f} -> {objective(base):.2f}")
    for g1 in grid1:
        for g10 in grid10:
            g = g1 + (g10 - g1) * lead / 9
            p = raw.copy()
            p[:, :, 1] = calibrate_rain(torch.from_numpy(raw[:, :, 1]), torch.from_numpy(g[None, :, None, None].astype("f4"))).numpy()
            t = SB.forecast_table(p, tru, per, clim)
            o = objective(t)
            if best is None or o > best[0]:
                best = (o, g1, g10, t)
    o, g1, g10, t = best
    print(f"best g1={g1} g10={g10}: ss {t['mean_ss']:+.4f} csi-skill {t['mean_csi_skill']:+.4f} blur {t['anti_blur_pts']:.2f} -> {o:.2f}")
    m.rain_gain = [float(g1 + (g10 - g1) * k / 9) for k in range(10)]
    print("rain_gain", [round(x, 3) for x in m.rain_gain])
    if a.mos:
        # drift and long-lead anomaly scaling for the five non-rain fields, on the same tuning windows
        from monsoon_branches.grid import lat_weights
        w = lat_weights()[None, :, None]
        with torch.inference_mode():
            cl = np.stack([m.clim_at(torch.from_numpy(arch.doy[idx] + k)).numpy() for k in range(1, 11)], 1)
        bias = np.zeros((10, 6)); lam = np.ones((10, 6))
        for k in range(10):
            for ch in (0, 2, 3, 4, 5):
                bias[k, ch] = ((raw[:, k, ch] - tru[:, k, ch]) * w).mean() / w.mean()
                pa = raw[:, k, ch] - bias[k, ch] - cl[:, k, ch]; ta = tru[:, k, ch] - cl[:, k, ch]
                lam[k, ch] = float(np.clip((pa * ta * w).sum() / (pa * pa * w).sum(), 0.3, 1.2))
        m.mos = {"bias": bias.tolist(), "lam": lam.tolist()}
        print("anomaly scaling day 1/4/8/10 (t2m,u10,v10,msl,tcwv):",
              " | ".join(",".join(f"{lam[k, ch]:.2f}" for ch in (0, 2, 3, 4, 5)) for k in (0, 3, 7, 9)))
    else:
        m.mos = None
    m.save(Path(a.artifacts), {k: v for k, v in m.meta.items() if k not in ("hidden", "blocks", "rain_gain", "mos")})
    print("wrote calibration to", a.artifacts)


if __name__ == "__main__":
    main()
