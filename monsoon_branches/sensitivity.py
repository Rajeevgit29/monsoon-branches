"""Observed shock sensitivities (ERA5) and the training constraint that keeps the model's shock responses inside them.
Training/evaluation only; not imported by the server.

Observed sensitivity: for a driver D (a box-mean anomaly on day t) the response of regional rain at lead k is the
coefficient on D(t) in a regression of the regional rain anomaly at t+k on D(t) with CONTROLS for the same day's state
(the region's rain, vapour, pressure and temperature, the box means of the other fields, and D(t-1)), over
June-September days of the training years. Holding today's weather fixed is the observational analogue of a shock
that changes one field. A block bootstrap over years gives a 90% range.
"""
from __future__ import annotations

import numpy as np
import torch

from .grid import BOX_ARABIAN_SEA, BOX_BAY_OF_BENGAL, LAT, LON, box_mask

LEADS = (1, 2, 4, 6, 8)
DRIVERS = {
    "wind": {"box": BOX_ARABIAN_SEA, "ch": 2, "label": "Arabian Sea wind (per m/s)"},
    "vapour": {"box": BOX_BAY_OF_BENGAL, "ch": 5, "label": "Bay of Bengal vapour (per kg/m²)"},
    "warmth": {"box": BOX_BAY_OF_BENGAL, "ch": 0, "label": "Bay of Bengal warmth (per K)"},
}
# constraints used in training: (driver, region, leads)
TARGETS = [("wind", "Western Ghats", (1, 2)), ("wind", "Central India", (2, 4, 6, 8)),
           ("vapour", "Bay of Bengal", (1, 2, 4)), ("warmth", "Bay of Bengal", (1, 2))]
SIGN_ONLY = {"warmth"}           # natural anomalies are a few tenths of a K: trust the sign, not the size


def regions(lsm, oro):
    la, lo = np.meshgrid(LAT, LON, indexing="ij")
    land = lsm >= 0.5
    return {"Western Ghats": land & (la >= 8) & (la <= 21) & (lo <= 76.5) & (oro > 100),
            "Central India": land & (la >= 18) & (la <= 26) & (lo >= 76) & (lo <= 86),
            "Bay of Bengal": (lsm < 0.5) & (lo >= 80) & (lo <= 95) & (la >= 8)}


def jjas_mask(dates) -> np.ndarray:
    import pandas as pd
    return np.isin(pd.DatetimeIndex(dates).month.values, [6, 7, 8, 9])


def _series(data, doy, clim, R):
    """Daily anomaly series needed by the regressions: regional means and box means, [N] each."""
    a = data - clim[doy.astype(int) % 365]
    reg = {rn: {ch: a[:, ch][:, msk].mean(1) for ch in (0, 1, 4, 5)} for rn, msk in R.items()}
    box = {name: {ch: a[:, ch][:, box_mask(d["box"])].mean(1) for ch in range(6)} for name, d in DRIVERS.items()}
    return reg, box


def _regress(reg, box, t, R):
    c = lambda x: x - x.mean()
    out = {}
    for name, d in DRIVERS.items():
        D = c(box[name][d["ch"]][t])
        others = [c(box[name][ch][t]) for ch in range(6) if ch != d["ch"]] + [c(box[name][d["ch"]][t - 1])]
        out[name] = {}
        for rn in R:
            ctrl = [c(reg[rn][ch][t]) for ch in (1, 5, 4, 0)]
            X = np.c_[D, *ctrl, *others]
            out[name][rn] = {k: float(np.linalg.lstsq(X, c(reg[rn][1][t + k]), rcond=None)[0][0]) for k in LEADS}
    return out


def _days(dates, years, n, year_range):
    ok = jjas_mask(dates) & (years >= year_range[0]) & (years <= year_range[1])
    t = np.where(ok)[0]
    return t[(t >= 1) & (t + max(LEADS) < n)]


def observed(data, doy, dates, years, clim, R, year_range=(2005, 2017)):
    reg, box = _series(data, doy, clim, R)
    return _regress(reg, box, _days(dates, years, len(data), year_range), R)


def observed_intervals(arch, clim, R, year_range=(2005, 2017), n_boot=40, seed=0):
    """Point estimates and 5-95% ranges from a bootstrap over whole years (days of a resampled year keep their
    neighbours, so the lagged regressions stay within each year's own sequence)."""
    reg, box = _series(arch.data, arch.doy, clim, R)
    t_all = _days(arch.dates, arch.years, len(arch.data), year_range)
    full = _regress(reg, box, t_all, R)
    yrs = np.unique(arch.years[t_all])
    by_year = {y: t_all[arch.years[t_all] == y] for y in yrs}
    rng = np.random.default_rng(seed)
    boots = [_regress(reg, box, np.concatenate([by_year[y] for y in rng.choice(yrs, len(yrs), replace=True)]), R)
             for _ in range(n_boot)]
    rng_ = {name: {rn: {k: (float(np.percentile([b[name][rn][k] for b in boots], 5)),
                            float(np.percentile([b[name][rn][k] for b in boots], 95))) for k in LEADS} for rn in R}
            for name in DRIVERS}
    return full, rng_


def training_targets(arch, clim, R):
    """[(driver, region, lead, lo, hi)] for the constraint, from the observed 90% ranges."""
    full, rng_ = observed_intervals(arch, clim, R)
    out = []
    for drv, rn, leads in TARGETS:
        for k in leads:
            lo, hi = rng_[drv][rn][k]
            if drv in SIGN_ONLY:
                lo = 0.0
            out.append((drv, rn, k, lo, hi))
    return out, full


# small shocks used inside training (linear regime, comparable with natural anomalies)
def perturb(hist: torch.Tensor, driver: str, masks: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (perturbed history, applied driver change per window [B])."""
    x = hist.clone()
    if driver == "wind":
        m = masks["Arabian Sea box"]
        du, dv = -0.2 * hist[:, -1, 2] * m, -0.2 * hist[:, -1, 3] * m
        x[:, -1, 2] = x[:, -1, 2] + du; x[:, -1, 3] = x[:, -1, 3] + dv
        applied = du[:, m.bool()].mean(1)
    elif driver == "vapour":
        m = masks["Bay box"]
        dq = 0.05 * hist[:, -1, 5] * m
        x[:, -1, 5] = x[:, -1, 5] + dq
        applied = dq[:, m.bool()].mean(1)
    else:
        m = masks["Bay box"]
        x[:, -1, 0] = x[:, -1, 0] + 0.5 * m
        applied = torch.full((hist.shape[0],), 0.5, device=hist.device)
    return x, applied


def sensitivity_loss(model, hist, doy, targets, masks, base=None):
    """Squared distance of the model's rain responses from the observed 90% ranges (zero inside the range),
    each term scaled by the range width."""
    steps = max(t[2] for t in targets)
    if base is None:
        base = model.rollout(hist, doy, steps)
    loss, n, resp = 0.0, 0, {}
    for drv in sorted({t[0] for t in targets}):
        x, applied = perturb(hist, drv, masks)
        d = (model.rollout(x, doy, steps) - base)[:, :, 1]            # [B, steps, H, W]
        for _, rn, k, lo, hi in [t for t in targets if t[0] == drv]:
            r = (d[:, k - 1][:, masks[rn]].mean(1) / applied).mean()
            resp[(drv, rn, k)] = float(r.detach())
            gap = torch.relu(lo - r) + torch.relu(r - hi)
            loss = loss + (gap / max(hi - lo, 1e-3)) ** 2
            n += 1
    return loss / n, resp


def torch_masks(R: dict, device) -> dict:
    m = {rn: torch.as_tensor(v, device=device) for rn, v in R.items()}
    m["Arabian Sea box"] = torch.as_tensor(box_mask(BOX_ARABIAN_SEA), dtype=torch.float32, device=device)
    m["Bay box"] = torch.as_tensor(box_mask(BOX_BAY_OF_BENGAL), dtype=torch.float32, device=device)
    return m
