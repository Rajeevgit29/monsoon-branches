"""Does the model respond to a shock the way the real atmosphere has? Model shock responses vs observed ERA5 sensitivity.

  python scripts/shock_realism.py --artifacts artifacts [--compare ../runs/other] --out reports/shock_realism.json \
      --fig docs/figures/shock_realism.png

Observed sensitivity (monsoon_branches.sensitivity): June-September days of 2005-2017; regional rain at t+k regressed on
a box-mean driver at t, controlling for the same day's state, with a 90% range from a bootstrap over years.
Model response: the same single-field change added to the last input day of June-September tuning windows (2018-2019),
(shocked - baseline) / applied change. Drivers: Arabian Sea wind (u10, v10 x 0.6, the official monsoon_stall), Bay of
Bengal vapour (tcwv x 1.15) and Bay of Bengal warmth (t2m + 3 K), the two halves of the official warm_ocean shock.
A regression is an association, so the check is whether the model's response falls inside the observed range
(for warmth, only the sign: natural anomalies are a few tenths of a kelvin, too small to pin down a size).
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
from monsoon_branches import sensitivity as SENS  # noqa: E402
from monsoon_branches.data import fit_stats, load_archive, window_starts  # noqa: E402
from monsoon_branches.grid import box_mask  # noqa: E402
from monsoon_branches.model import Forecaster  # noqa: E402

LEADS = SENS.LEADS
HEADLINES = [("wind", "Western Ghats", (1,), "Arabian Sea wind → W Ghats rain, day 1 (mm/day per m/s)"),
             ("wind", "Central India", (2, 4, 6, 8), "Arabian Sea wind → central India rain, days 2–8 (per m/s)"),
             ("vapour", "Bay of Bengal", (1,), "Bay vapour → Bay rain, day 1 (mm/day per kg/m²)"),
             ("warmth", "Bay of Bengal", (1,), "Bay warmth → Bay rain, day 1 (mm/day per K)")]


def modelled(arch, m, R, split="tune"):
    idx = window_starts(arch, split, max(LEADS), 2)
    idx = idx[SENS.jjas_mask(arch.dates[idx])]
    inp = np.stack([arch.data[i - 2:i + 1] for i in idx]).astype("f4")
    doy = torch.from_numpy(arch.doy[idx])
    with torch.inference_mode():
        base = m.rollout(torch.from_numpy(inp), doy, max(LEADS)).numpy()
    out = {}
    for name, d in SENS.DRIVERS.items():
        bm = box_mask(d["box"])
        x = inp.copy()
        if name == "wind":
            x[:, -1, 2] -= 0.4 * inp[:, -1, 2] * bm; x[:, -1, 3] -= 0.4 * inp[:, -1, 3] * bm
            applied = (-0.4 * inp[:, -1, 2][:, bm]).mean()
        elif name == "vapour":
            x[:, -1, 5] += 0.15 * inp[:, -1, 5] * bm
            applied = (0.15 * inp[:, -1, 5][:, bm]).mean()
        else:
            x[:, -1, 0] += 3.0 * bm
            applied = 3.0
        with torch.inference_mode():
            sh = m.rollout(torch.from_numpy(x), doy, max(LEADS)).numpy()
        resp = (sh - base)[:, :, 1].mean(0) / applied
        out[name] = {rn: {k: float(resp[k - 1][msk].mean()) for k in LEADS} for rn, msk in R.items()}
    return out


def headline_values(table):
    return [float(np.mean([table[d][r][k] for k in leads])) for d, r, leads, _ in HEADLINES]


def headline_ranges(rng_):
    out = []
    for d, r, leads, _ in HEADLINES:
        lo = float(np.mean([rng_[d][r][k][0] for k in leads])); hi = float(np.mean([rng_[d][r][k][1] for k in leads]))
        if d in SENS.SIGN_ONLY:
            lo, hi = 0.0, float("inf")
        out.append((lo, hi))
    return out


def evaluate_models(arch, paths):
    st = fit_stats(arch, "train")
    R = SENS.regions(arch.static["land_sea_mask"], arch.static["orography_m"])
    full, rng_ = SENS.observed_intervals(arch, st["clim"], R)
    ranges = headline_ranges(rng_)
    rep = {"headlines": [h[3] for h in HEADLINES], "observed": headline_values(full), "observed_range": ranges, "models": {}}
    for p in paths:
        mt = modelled(arch, Forecaster.load(p), R)
        v = headline_values(mt)
        rep["models"][Path(p).name] = {"values": v, "inside": [bool(lo <= x <= hi) for x, (lo, hi) in zip(v, ranges)], "by_lead": mt}
    rep["observed_by_lead"] = full
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT.parent / "data" / "india_climate_train.nc"))
    ap.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    ap.add_argument("--compare", nargs="*", default=[])
    ap.add_argument("--labels", nargs="*", default=None, help="display names for --artifacts then --compare")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fig", default=None)
    a = ap.parse_args()
    torch.set_num_threads(8)
    arch = load_archive(a.data)
    paths = [a.artifacts] + a.compare
    rep = evaluate_models(arch, paths)
    names = list(rep["models"])
    print("headline rain responses, June-September (observed: same-day state held fixed, 90% range over years)")
    for i, h in enumerate(rep["headlines"]):
        lo, hi = rep["observed_range"][i]
        rng_txt = f"[{lo:+.2f}, {hi:+.2f}]" if np.isfinite(hi) else "(sign: > 0)"
        print(f"  {h:60s} obs {rep['observed'][i]:+.2f} {rng_txt:18s} | " + " | ".join(
            f"{n[:14]} {rep['models'][n]['values'][i]:+.2f}{'' if rep['models'][n]['inside'][i] else ' ✗'}" for n in names))
    for n in names:
        print(f"  {n}: {sum(rep['models'][n]['inside'])}/4 inside the observed range")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rep, indent=1, default=lambda x: None if x == float("inf") else x))
    if a.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.edgecolor": "#e2e2df"})
        titles = ["Arabian Sea wind → W Ghats\nrain, day 1 (per m/s)", "Arabian Sea wind → central\nIndia, days 2–8 (per m/s)",
                  "Bay vapour → Bay rain,\nday 1 (per kg/m²)", "Bay warmth → Bay rain,\nday 1 (per K; sign only)"]
        cols = ["#257de0", "#f47230", "#00b894", "#6a4ce2"]
        labels = a.labels or names
        fig, axes = plt.subplots(2, 2, figsize=(3.6, 2.55), dpi=240)
        for i, ax in enumerate(axes.ravel()):
            lo, hi = rep["observed_range"][i]
            v = [rep["models"][n]["values"][i] for n in names]
            top = max([rep["observed"][i], *v, 0]); bot = min([rep["observed"][i], *v, 0])
            hi_draw = hi if np.isfinite(hi) else top * 1.15 + 0.1
            ax.axhspan(lo, hi_draw, color="#e9e9e6", zorder=0)
            ax.axhline(rep["observed"][i], color="#6b6b67", lw=1.2, ls=(0, (4, 2)), zorder=1)
            ax.bar(range(len(v)), v, color=cols[:len(v)], width=0.6, zorder=2)
            ax.axhline(0, color="#191919", lw=0.6)
            for j, x in enumerate(v):
                ax.text(j, x, f"{x:+.2f}", ha="center", va="bottom" if x >= 0 else "top", fontsize=6.3)
            pad = 0.18 * (max(top, hi_draw) - bot + 1e-6)
            ax.set_ylim(bot - pad, max(top, hi_draw) + pad)
            ax.set_title(titles[i], fontsize=6.8, loc="left"); ax.set_xticks([])
            ax.spines[["top", "right", "bottom"]].set_visible(False)
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cols[:len(names)]]
        handles += [plt.Rectangle((0, 0), 1, 1, color="#e9e9e6"), plt.Line2D([], [], color="#6b6b67", ls=(0, (4, 2)))]
        fig.legend(handles, list(labels) + ["ERA5 observed 90% range", "ERA5 observed"], loc="lower center",
                   ncol=2, frameon=False, fontsize=6)
        fig.tight_layout(rect=(0, 0.13, 1, 1), h_pad=1.2)
        Path(a.fig).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(a.fig); plt.close(fig)


if __name__ == "__main__":
    main()
