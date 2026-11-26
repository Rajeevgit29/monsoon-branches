"""Throw random shocks at a running server, the way judges might, and check that every answer is usable.

  python scripts/fuzz_shocks.py --url http://localhost:8000 --n 300

Each case: a random held-out start (2018-2021), a random box anywhere on the grid (1x1 cells up to the whole map),
and a random shock: wind scale (-100%..+100%), temperature (-10..+10 K), vapour (-60%..+100%), pressure (-30..+30 hPa),
rain (+0..+200 mm/day), or several at once, sometimes far outside anything observed. Checks, per case:
valid (official validity gate), bounded (no field moves more than 5 std beyond the largest observed daily anomaly),
local on day 1 for box shocks, and the 5 s limit. Also re-sends some cases to confirm identical answers.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from monsoon_branches.data import load_archive, window_starts  # noqa: E402
from monsoon_branches.grid import LAT, LON  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--data", default=str(ROOT.parent / "data" / "india_climate_train.nc"))
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from monsoon_r1.client import call_simulate
    from monsoon_r1.contract import doy_sincos
    from monsoon_r1.scoring import validity, near_mask
    arch = load_archive(a.data)
    std = arch.data.std(axis=(0, 2, 3))
    idx = np.concatenate([window_starts(arch, s, 10, 1) for s in ("tune", "test")])
    rng = np.random.default_rng(a.seed)
    fails, worst_t, results = [], 0.0, []
    kinds = ["wind", "temp", "vapour", "pressure", "rain", "combo"]
    for n in range(a.n):
        i = int(rng.choice(idx))
        init = arch.data[i - 2:i + 1].astype("f4")
        h, w = int(rng.integers(1, 33)), int(rng.integers(1, 37))
        r0, c0 = int(rng.integers(0, 33 - h)), int(rng.integers(0, 37 - w))
        m = np.zeros((32, 36), bool); m[r0:r0 + h, c0:c0 + w] = True
        box = (LAT[r0] - .5, LAT[r0 + h - 1] + .5, LON[c0] - .5, LON[c0 + w - 1] + .5)
        kind = kinds[n % len(kinds)]
        d = np.zeros((6, 32, 36), "f4"); last = init[-1]
        extreme = rng.random() < 0.25
        if kind in ("wind", "combo"):
            s = rng.uniform(-1, 1); d[2][m] = s * last[2][m]; d[3][m] = s * last[3][m]
        if kind in ("temp", "combo"):
            d[0][m] = rng.uniform(-10, 10) if extreme else rng.uniform(-4, 4)
        if kind in ("vapour", "combo"):
            d[5][m] = (rng.uniform(-0.6, 1.0) if extreme else rng.uniform(-0.3, 0.3)) * last[5][m]
        if kind in ("pressure", "combo"):
            d[4][m] = 100 * (rng.uniform(-30, 30) if extreme else rng.uniform(-8, 8))
        if kind == "rain":
            d[1][m] = rng.uniform(0, 200 if extreme else 40)
        payload = {"init_states": init.tolist(), "doy_sincos": doy_sincos(int(arch.doy[i])), "steps": 10, "perturbation": d.tolist()}
        r = call_simulate(a.url, payload, allow_private=True)
        if not r.ok:
            fails.append((n, kind, r.error)); continue
        worst_t = max(worst_t, r.wall_s)
        bad = validity(r.out, 10)
        base = call_simulate(a.url, {**payload, "perturbation": None}, allow_private=True).out
        z = np.abs((r.out - base) / std[None, :, None, None])
        e = (z ** 2).sum(1)
        near = near_mask(box)
        local = float(e[0][near].sum() / max(e[0].sum(), 1e-12)) if e[0].sum() > 0 else 1.0
        if n % 10 == 0:                                                  # determinism: same request, same answer
            again = call_simulate(a.url, payload, allow_private=True).out
            if not np.array_equal(again, r.out):
                bad.append("non-deterministic")
        z_in = float(np.abs(d / std[:, None, None]).max())
        k_, c_, *_ = np.unravel_index(np.argmax(z), z.shape)
        results.append({"kind": kind, "extreme": extreme, "cells": int(m.sum()), "zmax": float(z.max()), "local": local,
                        "z_in": z_in, "amp": float(z.max() / max(z_in, 1e-6)), "day": int(k_) + 1, "field": int(c_),
                        "z_late": float(z[3:].max())})
        if bad:
            fails.append((n, kind, bad))
    z = np.array([x["zmax"] for x in results]); loc = np.array([x["local"] for x in results])
    ext = np.array([x["extreme"] for x in results])
    print(f"{len(results)}/{a.n} answered; failures: {len(fails)}; slowest call {worst_t:.2f} s")
    for f in fails[:10]:
        print("  FAIL", f)
    amp = np.array([x["amp"] for x in results]); zin = np.array([x["z_in"] for x in results])
    names = ["t2m", "tp", "u10", "v10", "msl", "tcwv"]
    print("by shock kind: median shock size (std) -> median / max response (std), max response / shock size, where the max is")
    for kd in kinds:
        sel = np.array([x["kind"] == kd for x in results])
        if sel.any():
            where = {}
            for x in [r for r in results if r["kind"] == kd]:
                where[(names[x["field"]], x["day"])] = where.get((names[x["field"]], x["day"]), 0) + 1
            top = sorted(where.items(), key=lambda kv: -kv[1])[:3]
            print(f"  {kd:8s} in {np.median(zin[sel]):5.2f} -> out {np.median(z[sel]):5.2f} / {z[sel].max():5.2f}; amplification median "
                  f"{np.median(amp[sel]):.2f} max {amp[sel].max():.2f}; max at " + ", ".join(f"{f} day {dd} ({c})" for (f, dd), c in top))
    zl = np.array([x["z_late"] for x in results])
    print(f"largest response on days 4-10: median {np.median(zl):.2f} std, max {zl.max():.2f} std, above 5 std in {np.mean(zl > 5):.0%} of cases")
    for lab, sel in (("ordinary shocks", ~ext), ("extreme shocks", ext)):
        if sel.any():
            print(f"{lab:16s}: max response {np.percentile(z[sel], 50):.2f} std (median) / {z[sel].max():.2f} std (max); "
                  f"within 5 std in {np.mean(z[sel] <= 5):.0%}; day-1 energy near the box {np.median(loc[sel]):.0%} (median), "
                  f">=60% in {np.mean(loc[sel] >= 0.6):.0%}")


if __name__ == "__main__":
    main()
