"""Scoreboard: run complete 10-day forecasts on held-out windows and score them with the official functions.

  python evaluate.py --data ../data/india_climate_train.nc --split tune --model persistence
  python evaluate.py --data ../data/india_climate_train.nc --split tune --model climatology
  python evaluate.py --data ../data/india_climate_train.nc --split tune --model artifacts          # in-process
  python evaluate.py --data ../data/india_climate_train.nc --split test --url http://localhost:8000  # over HTTP

Reports heavy-rain CSI, skill against the stronger of persistence and climatology by channel and lead,
day-8 variance ratios, validity, the three official shocks on several windows, and (with --url) end-to-end latency.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from monsoon_branches.data import SPLITS, calendar_slot, load_archive, reference_climatology, window_starts
from monsoon_branches.grid import CHANNELS, NC, NLAT, NLON
from monsoon_branches.model import Forecaster
from monsoon_branches import scoreboard as SB

STEPS = 10
ROOT = Path(__file__).parent


def gather(arch, idx):
    """inputs [W,3,6,H,W], truth [W,10,...], doy of the last input day [W], target-day calendar slots [W,10]."""
    inp = np.stack([arch.data[i - 2:i + 1] for i in idx])
    tru = np.stack([arch.data[i + 1:i + 1 + STEPS] for i in idx])
    slots = np.array([[calendar_slot(arch.dates[i + k]) for k in range(1, STEPS + 1)] for i in idx])
    return inp, tru, arch.doy[idx], slots


class ModelRunner:
    def __init__(self, art_dir):
        torch.set_num_threads(4)
        self.m = Forecaster.load(art_dir)

    def __call__(self, inp, doy, pert=None, batch=64):
        x = inp.copy()
        if pert is not None:
            x[:, -1] += pert
            x[:, -1, 1] = np.maximum(x[:, -1, 1], 0)
        out = []
        with torch.inference_mode():
            for s in range(0, len(x), batch):
                out.append(self.m.rollout(torch.from_numpy(x[s:s + batch]), torch.from_numpy(doy[s:s + batch]), STEPS, calibrate=True).numpy())
        return np.concatenate(out)


class HttpRunner:
    def __init__(self, url):
        from monsoon_r1.client import call_simulate
        from monsoon_r1.contract import doy_sincos
        self.url, self.call, self.enc = url, call_simulate, doy_sincos
        self.latency = []

    def __call__(self, inp, doy, pert=None):
        out = []
        for w in range(len(inp)):
            payload = {"init_states": inp[w].tolist(), "doy_sincos": self.enc(int(round(float(doy[w])))), "steps": STEPS,
                       "perturbation": None if pert is None else pert[w].tolist()}
            r = self.call(self.url, payload, allow_private=True)
            if not r.ok:
                raise RuntimeError(f"window {w}: {r.error}")
            self.latency.append(r.wall_s)
            out.append(r.out)
        return np.stack(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=[str(ROOT.parent / "data" / "india_climate_train.nc")],
                    help="the kit archive, plus ../data/era5_2022.nc to score --split test2022")
    ap.add_argument("--split", default="tune", choices=list(SPLITS))
    ap.add_argument("--stride", type=int, default=3, help="days between window starts")
    ap.add_argument("--model", default="artifacts", help="persistence | climatology | artifacts | path to an artifacts dir")
    ap.add_argument("--url", default=None, help="score a running server over HTTP instead")
    ap.add_argument("--shock-windows", type=int, default=12)
    ap.add_argument("--out", default=None, help="write the report JSON here")
    a = ap.parse_args(argv)

    arch = load_archive(a.data)
    idx = window_starts(arch, a.split, STEPS, a.stride)
    if a.url:                                      # HTTP is slow-ish: cap at a representative subset
        idx = idx[:: max(1, len(idx) // 40)]
    inp, tru, doy, slots = gather(arch, idx)
    clim_ref = reference_climatology(arch, "train")
    clim = clim_ref[slots]                         # [W,10,6,H,W]
    persist = np.repeat(inp[:, -1:], STEPS, axis=1)
    std = arch.data.std(axis=(0, 2, 3))            # equals stats.json (all years), the scorer's shock units

    if a.url:
        run, name = HttpRunner(a.url), f"server {a.url}"
    elif a.model == "persistence":
        run, name = (lambda x, d, p=None: np.repeat((x[:, -1] + (0 if p is None else p))[:, None], STEPS, axis=1)), "persistence"
    elif a.model == "climatology":
        run, name = (lambda x, d, p=None: clim), "climatology"
    else:
        path = ROOT / "artifacts" if a.model == "artifacts" else Path(a.model)
        run, name = ModelRunner(path), f"model {path.name}"

    t0 = time.perf_counter()
    pred = run(inp, doy)
    elapsed = time.perf_counter() - t0
    table = SB.forecast_table(pred, tru, persist, clim)
    SB.print_forecast(f"{name} on {a.split} {SPLITS[a.split]}", table)
    print(f"forecast wall time {elapsed:.1f}s for {len(idx)} windows")
    boot = SB.bootstrap_points(table)
    print(f"5-window draws (as graded): forecast pts mean {boot['forecast_pts_mean']:.1f}/45 (p10 {boot['forecast_pts_p10']:.1f}); "
          f"full RMSE pts in {100 * boot['p_full_rmse']:.0f}% of draws, full CSI pts in {100 * boot['p_full_csi']:.0f}%")

    # shocks on evenly spaced windows (cover several seasons); climatology ignores its inputs, so it has none
    sh = {"summary": {"causality_pts_mean": 0.0}}
    if a.model != "climatology" or a.url:
        sw = np.linspace(0, len(idx) - 1, min(a.shock_windows, len(idx))).astype(int)
        from monsoon_r1.scoring import make_shock
        base = pred[sw]
        outs = {}
        for sname in ("warm_ocean", "monsoon_stall", "null"):
            pert = np.stack([make_shock(sname, inp[w, -1], std) for w in sw])
            outs[sname] = run(inp[sw], doy[sw], pert)
        sh = SB.shock_table(base, outs, std)
        print("\nshocks:", json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in sh["summary"].items()}, indent=1))

    report = {"name": name, "split": a.split, "windows": int(len(idx)),
              "mean_ss": table["mean_ss"], "rmse_pts": table["rmse_pts"], "csi_pts": table["csi_pts"],
              "mean_csi_skill": table["mean_csi_skill"], "csi_model": table["csi_model"],
              "csi_persist": table["csi_persist"], "csi_clim": table["csi_clim"],
              "anti_blur_pts": table["anti_blur_pts"], "var_ratio_day8": dict(zip(CHANNELS, map(float, table["var_ratio_day8"]))),
              "pooled_ss": dict(zip(CHANNELS, map(float, table["pooled_ss"]))),
              "per_lead": {str(k): {kk: dict(zip(CHANNELS, map(float, vv))) for kk, vv in v.items()} for k, v in table["per_lead"].items()},
              "beats_persistence_day4": table["beats_persistence_day4"], "invalid_windows": len(table["invalid"]),
              "shocks": sh["summary"], "bootstrap_5_windows": boot}
    if a.url:
        lat = np.array(run.latency)
        report["latency_s"] = {"mean": float(lat.mean()), "p95": float(np.percentile(lat, 95)), "max": float(lat.max())}
        print(f"\nHTTP latency over {len(lat)} calls: mean {lat.mean():.2f}s  p95 {np.percentile(lat, 95):.2f}s  max {lat.max():.2f}s")
    est = table["rmse_pts"] + table["csi_pts"] + table["anti_blur_pts"] + sh["summary"]["causality_pts_mean"]
    report["estimated_auto_pts_of_70"] = est
    print(f"\nestimated automated points: {est:.1f} / 70  (forecast 45 + dynamics 25; repo and brief not included)")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(report, indent=1, default=float))
    return report


if __name__ == "__main__":
    main()
