"""Figures for the two-page brief, computed from the shipped artifacts on the held-out test years.

  python scripts/make_figures.py --data ../data/india_climate_train.nc --out docs/figures

Writes: skill_by_lead.png (RMSE relative to climatology, leads 1-10, model vs persistence),
shock_energy.png (scorer-style response energy on days 1/4/8 for both official shocks),
demo_konkan.png (baseline / shocked / difference rain and the dry-spell series at one land cell),
and figures.json with the numbers behind them.
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
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from monsoon_branches.data import calendar_slot, load_archive, reference_climatology, window_starts  # noqa: E402
from monsoon_branches.grid import CHANNELS, LAT, LON, BOX_ARABIAN_SEA, BOX_BAY_OF_BENGAL, box_mask, lat_weights  # noqa: E402
from monsoon_branches.model import Forecaster  # noqa: E402

INK, INK2, GRID = "#191919", "#666666", "#e2e2df"
S1, S2, OBS = "#257de0", "#f47230", "#6b6b67"
BLUE = LinearSegmentedColormap.from_list("rain", ["#f5f8fc", "#cce3fd", "#9dc8f7", "#6aabf1", "#338ced", "#2370cd", "#1757ac", "#0e3d7e"])
BRBL = LinearSegmentedColormap.from_list("dif", ["#7a4a12", "#b77c2f", "#e6c99a", "#efefef", "#b6d5f8", "#519ced", "#1757ac"])
ENERGY = LinearSegmentedColormap.from_list("en", ["#fdf6f0", "#fde0cc", "#fbc4a1", "#f8a273", "#f47230", "#d85a1c", "#b04512", "#83320c"])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 8.5, "axes.titlecolor": INK})


def coast(ax, lines):
    for ln in lines:
        a = np.array(ln)
        ax.plot(a[:, 0], a[:, 1], color=INK, lw=0.5, alpha=0.6)


def draw_box(ax, box, **kw):
    la0, la1, lo0, lo1 = box
    m = box_mask(box)
    ii, jj = np.where(m)
    ax.add_patch(plt.Rectangle((LON[jj.min()] - .5, LAT[ii.min()] - .5), jj.max() - jj.min() + 1, ii.max() - ii.min() + 1,
                               fill=False, ec=INK, lw=1.1, **kw))


def field_ax(ax, z, cmap, norm=None, vmin=None, vmax=None, lines=(), title=""):
    im = ax.imshow(z, origin="lower", extent=(65, 101, 6, 38), cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, interpolation="nearest")
    coast(ax, lines)
    ax.set_xlim(65, 101); ax.set_ylim(6, 38); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc="left")
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT.parent / "data" / "india_climate_train.nc"))
    ap.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "figures"))
    ap.add_argument("--split", default="test", help="years for the skill-by-lead figure")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    lines = json.loads((ROOT / "dashboard" / "data" / "coast.json").read_text())
    arch = load_archive(a.data)
    torch.set_num_threads(4)
    m = Forecaster.load(a.artifacts)
    std = arch.data.std(axis=(0, 2, 3))
    w = lat_weights()[:, None]
    nums = {}

    # 1. RMSE relative to climatology by lead, test years
    idx = window_starts(arch, a.split, 10, 2)
    inp = np.stack([arch.data[i - 2:i + 1] for i in idx]); tru = np.stack([arch.data[i + 1:i + 11] for i in idx])
    clim = reference_climatology(arch, "train")[np.array([[calendar_slot(arch.dates[i + k]) for k in range(1, 11)] for i in idx])]
    with torch.inference_mode():
        pred = np.concatenate([m.rollout(torch.from_numpy(inp[s:s + 64]), torch.from_numpy(arch.doy[idx[s:s + 64]]), 10, calibrate=True).numpy()
                               for s in range(0, len(idx), 64)])
    rm = lambda p: np.sqrt((((p - tru) ** 2) * w).mean(axis=(0, 3, 4)))      # [10, 6]
    r_model, r_pers, r_clim = rm(pred), rm(np.repeat(inp[:, -1:], 10, 1)), rm(clim)
    rel_m, rel_p = (r_model / r_clim).mean(1), (r_pers / r_clim).mean(1)
    nums["rel_rmse_model"] = rel_m.tolist(); nums["rel_rmse_persist"] = rel_p.tolist()
    nums["rel_rmse_model_by_channel"] = {c: (r_model[:, k] / r_clim[:, k]).tolist() for k, c in enumerate(CHANNELS)}
    heavy = tru[:, :, 1] >= 50                                                 # cells with > 50 mm/day observed
    nums["heavy_rain_ratio"] = {f"day{k}": float(pred[:, k - 1, 1][heavy[:, k - 1]].mean() / tru[:, k - 1, 1][heavy[:, k - 1]].mean())
                                for k in (1, 4, 8)}
    fig, ax = plt.subplots(figsize=(3.4, 2.1), dpi=220)
    L = np.arange(1, 11)
    ax.axhline(1, color=OBS, lw=1, ls=(0, (4, 3)))
    ax.plot(L, rel_p, color=S2, lw=2, marker="o", ms=3)
    ax.plot(L, rel_m, color=S1, lw=2, marker="o", ms=3)
    ax.text(10.25, rel_p[-1], "Persistence", color=INK, va="center", fontsize=7.5)
    ax.text(10.25, min(rel_m[-1], 1.0) - 0.06, "Model", color=INK, va="center", fontsize=7.5)
    ax.text(10.25, max(rel_m[-1], 1.0) + 0.05, "Climatology", color=INK, va="center", fontsize=7.5)
    ax.set_xlim(0.7, 12.6); ax.set_xticks(L); ax.set_xlabel("lead (days)")
    ax.set_ylabel("RMSE ÷ climatology RMSE\n(mean of 6 fields)")
    ax.grid(axis="y", color=GRID, lw=0.6); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out / "skill_by_lead.png"); plt.close(fig)

    # 2. shock energy maps on a held-out monsoon window
    i = int(np.where(arch.dates == np.datetime64("2021-07-19"))[0][0])
    x0 = arch.data[i - 2:i + 1]
    from monsoon_r1.scoring import make_shock, causality_checks
    def run(x):
        with torch.inference_mode():
            return m.rollout(torch.from_numpy(x[None].astype("f4")), torch.tensor([float(arch.doy[i])]), 10, calibrate=True)[0].numpy()
    base = run(x0)
    fig, axes = plt.subplots(2, 3, figsize=(6.6, 3.45), dpi=220)
    nums["shocks"] = {}
    for r, (name, box) in enumerate((("monsoon_stall", BOX_ARABIAN_SEA), ("warm_ocean", BOX_BAY_OF_BENGAL))):
        x = x0.copy(); x[-1] += make_shock(name, x0[-1], std)
        d = run(x) - base
        e = ((d / std[None, :, None, None]) ** 2).sum(1)
        chk = causality_checks(base, run(x), box, std)
        nums["shocks"][name] = {k: (float(v) if not isinstance(v, bool) else v) for k, v in chk.items()}
        for c, k in enumerate((0, 3, 7)):
            ax = axes[r, c]
            vmax = max(e[k].max(), 1e-6)
            field_ax(ax, e[k], ENERGY, vmin=0, vmax=vmax, lines=lines,
                     title=f"{'Arabian Sea wind −40%' if name == 'monsoon_stall' else 'Warm Bay of Bengal'} · day {k + 1}")
            draw_box(ax, box)
    fig.tight_layout(); fig.savefig(out / "shock_energy.png"); plt.close(fig)

    # 3. demo: Konkan floods window, Arabian Sea wind stall
    x = x0.copy(); x[-1] += make_shock("monsoon_stall", x0[-1], std)
    sh = run(x)
    day = 3                                                     # day 3 after the last input = 2021-07-22
    tb, ts = base[:, 1], sh[:, 1]
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.35), dpi=220)
    vmax = 100
    from matplotlib.colors import PowerNorm
    pn = PowerNorm(0.5, 0, vmax)
    im = field_ax(axes[0], tb[day - 1], BLUE, norm=pn, lines=lines, title=f"Baseline · day {day}")
    field_ax(axes[1], ts[day - 1], BLUE, norm=pn, lines=lines, title=f"Shocked start · day {day}")
    dd = ts[day - 1] - tb[day - 1]; lim = max(np.percentile(np.abs(dd), 99), 0.5)
    im2 = field_ax(axes[2], dd, BRBL, norm=TwoSlopeNorm(0, -lim, lim), lines=lines, title="Shocked − baseline")
    for ax in axes:
        draw_box(ax, BOX_ARABIAN_SEA)
    cb = fig.colorbar(im, ax=axes[:2], orientation="horizontal", fraction=0.035, pad=0.03, shrink=0.7, ticks=[0, 2.5, 10, 25, 50, 100])
    cb.ax.tick_params(labelsize=6.5)
    cb.outline.set_visible(False); cb.set_label("rain (mm/day)", color=INK2, fontsize=7)
    cb2 = fig.colorbar(im2, ax=axes[2], orientation="horizontal", fraction=0.035, pad=0.03, shrink=0.8)
    cb2.ax.tick_params(labelsize=6.5)
    cb2.outline.set_visible(False); cb2.set_label("change (mm/day)", color=INK2, fontsize=7)
    fig.savefig(out / "demo_konkan.png", bbox_inches="tight"); plt.close(fig)

    # dry-spell example: the land cell with the largest change in longest dry spell
    lsm = arch.static["land_sea_mask"] >= 0.5
    def longest(s):
        best = cur = 0
        for v in s:
            cur = cur + 1 if v < 2.5 else 0; best = max(best, cur)
        return best
    db = np.array([[longest(tb[:, a, b]) for b in range(36)] for a in range(32)])
    ds = np.array([[longest(ts[:, a, b]) for b in range(36)] for a in range(32)])
    delta = np.where(lsm, ds - db, 0)
    nums["dry"] = {"cells_longer": int((delta > 0).sum()), "cells_shorter": int((delta < 0).sum()), "max_longer": int(delta.max()),
                   "land_mean_10day_rain_change_mm": float((ts.sum(0) - tb.sum(0))[lsm].mean())}
    # example cell: the largest dry-spell lengthening on land with at least 15 mm of baseline rain (not desert);
    # ties broken by the size of the 10-day rain change
    drain = np.abs(ts.sum(0) - tb.sum(0))
    cand = lsm & (tb.sum(0) >= 15)
    score = np.where(cand, delta * 1000 + drain, -1e9)
    a_, b_ = np.unravel_index(np.argmax(score), delta.shape)
    obs = arch.data[i + 1:i + 11, 1, a_, b_]
    nums["dry"]["cell"] = {"lat": float(LAT[a_]), "lon": float(LON[b_]), "baseline": int(db[a_, b_]), "shocked": int(ds[a_, b_]),
                           "rain_base": tb[:, a_, b_].round(2).tolist(), "rain_shock": ts[:, a_, b_].round(2).tolist(), "rain_obs": obs.round(2).tolist()}
    fig, ax = plt.subplots(figsize=(3.4, 1.9), dpi=220)
    ax.axhline(2.5, color=OBS, lw=0.8, ls=(0, (3, 3)))
    ax.text(9.9, 2.5, "dry day < 2.5 ", color=INK2, fontsize=6.5, va="bottom", ha="right")
    ax.plot(L, obs, color=OBS, lw=1.5, ls=(0, (4, 3)))
    ax.plot(L, tb[:, a_, b_], color=S1, lw=2)
    ax.plot(L, ts[:, a_, b_], color=S2, lw=2)
    ends = sorted([(tb[-1, a_, b_], "Baseline"), (ts[-1, a_, b_], "Shocked"), (obs[-1], "Observed")])
    ymax = max(10, tb[:, a_, b_].max(), ts[:, a_, b_].max(), obs.max()) * 1.1
    lasty = -1e9
    for v, n in ends:
        v = max(v, lasty + ymax * 0.1); lasty = v
        ax.text(10.25, v, n, fontsize=7, va="center", color=INK)
    ax.set_xlim(0.7, 12.4); ax.set_ylim(0, ymax); ax.set_xticks(L); ax.set_xlabel("forecast day")
    ax.set_ylabel("rain (mm/day)"); ax.set_title(f"{LAT[a_]:.1f}°N {LON[b_]:.1f}°E: longest dry spell {db[a_, b_]} → {ds[a_, b_]} days", loc="left")
    ax.grid(axis="y", color=GRID, lw=0.6); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out / "dry_spell_cell.png"); plt.close(fig)

    (out / "figures.json").write_text(json.dumps(nums, indent=1))
    print(json.dumps({k: v for k, v in nums.items() if k != "rel_rmse_model_by_channel"}, indent=1)[:3000])


if __name__ == "__main__":
    main()
