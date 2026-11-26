"""Training-side data access: the NetCDF archive, year splits, windows and statistics. Not imported by the server.

Splits are by calendar year and are applied BEFORE windows are cut, so no window straddles two splits:
  train 2005-2017 (fit the model, normalisers and climatology), tune 2018-2019 (checkpoint selection),
  test 2020-2021 (reported once, at the end).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .grid import CHANNELS, LAT, LON, NC, NLAT, NLON, T_IN, TP, lat_weights

SPLITS = {"train": (2005, 2017), "tune": (2018, 2019), "test": (2020, 2021), "all": (2005, 2021),
          # with the extra ERA5 years fetched by scripts/fetch_era5_wb2.py (same source and processing as the kit)
          "train_ext": (1979, 2017), "test2022": (2022, 2022)}
MONTH_MID = np.array([16, 45.5, 75, 105.5, 136, 166.5, 197, 228, 258.5, 289, 319.5, 350])   # mid-month day of year


@dataclass
class Archive:
    data: np.ndarray        # [N, 6, 32, 36] float32, physical units
    dates: np.ndarray       # [N] datetime64[D]
    doy: np.ndarray         # [N] day of year, 1-based (Jan 1 = 1), the kit's doy_sincos convention
    years: np.ndarray       # [N]
    static: dict            # land_sea_mask, orography_m: [32, 36]

    @property
    def yyyymmdd(self) -> np.ndarray:
        return np.array([int(str(d).replace("-", "")) for d in self.dates])


def load_archive(path) -> Archive:
    """One NetCDF path, or several (e.g. the kit archive plus fetched ERA5 years), merged in time order."""
    import xarray as xr
    paths = [path] if isinstance(path, (str, Path)) else list(path)
    ds = xr.concat([xr.open_dataset(p) for p in paths], dim="time").sortby("time") if len(paths) > 1 else xr.open_dataset(paths[0])
    if len(paths) > 1:
        ds["static"] = xr.open_dataset(paths[0])["static"]
    assert tuple(str(c) for c in ds["channel"].values) == CHANNELS, ds["channel"].values
    assert np.allclose(ds["lat"].values, LAT), "rows must run south -> north, 6.5..37.5"
    assert np.allclose(ds["lon"].values, LON), "columns must run west -> east, 65.5..100.5"
    data = ds["data"].transpose("time", "channel", "lat", "lon").values.astype("f4")
    assert data.shape[1:] == (NC, NLAT, NLON) and np.isfinite(data).all()
    dates = ds["time"].values.astype("datetime64[D]")
    assert (np.diff(dates).astype(int) == 1).all(), "archive must be daily and gap-free"
    names = [str(c) for c in ds["static_channel"].values]
    st = ds["static"].values.astype("f4")
    static = {n: st[names.index(n)] for n in ("land_sea_mask", "orography_m")}
    import pandas as pd
    idx = pd.DatetimeIndex(dates)
    return Archive(data, dates, idx.dayofyear.values.astype("f4"), idx.year.values, static)


def window_starts(arch: Archive, split: str, horizon: int, stride: int = 1) -> np.ndarray:
    """Indices i of the LAST input day such that days i-2..i+horizon all fall inside the split's years."""
    y0, y1 = SPLITS[split]
    inside = (arch.years >= y0) & (arch.years <= y1)
    lo, hi = np.where(inside)[0][[0, -1]]
    return np.arange(lo + T_IN - 1, hi - horizon + 1, stride)


def phase_climatology(arch: Archive, split: str = "train", smooth_days: int = 15) -> np.ndarray:
    """[365, 6, 32, 36] mean by day-of-year phase (doy mod 365) over the split's years, circular running mean.
    Phase-indexed rather than calendar-indexed because the server only ever sees doy_sincos."""
    y0, y1 = SPLITS[split]
    sel = (arch.years >= y0) & (arch.years <= y1)
    ph = (arch.doy[sel].astype(int)) % 365
    acc = np.zeros((365, NC, NLAT, NLON), "f8")
    cnt = np.zeros(365)
    np.add.at(acc, ph, arch.data[sel])
    np.add.at(cnt, ph, 1)
    acc /= np.maximum(cnt, 1)[:, None, None, None]
    h = smooth_days // 2
    out = np.zeros_like(acc)
    for o in range(-h, h + 1):
        out += np.roll(acc, o, axis=0)
    return (out / (2 * h + 1)).astype("f4")


def fit_phi(arch: Archive, clim: np.ndarray, split: str = "train") -> np.ndarray:
    """Per-cell, per-field AR(1) memory of daily anomalies about the climatology, by season.

    phi = cov(a(t), a(t+1)) / var(a(t)) for day pairs inside the split, grouped by the calendar month of t+1 and pooled
    with the two neighbouring months, clipped to [0, 0.99], then interpolated between mid-month days to a
    [365, 6, 32, 36] phase table. It is the red-noise null model of climate anomalies (Hasselmann 1976)."""
    import pandas as pd
    y0, y1 = SPLITS[split]
    sel = (arch.years >= y0) & (arch.years <= y1)
    a = arch.data - clim[arch.doy.astype(int) % 365]
    month = pd.DatetimeIndex(arch.dates).month.values - 1
    num = np.zeros((12, NC, NLAT, NLON)); den = np.zeros((12, NC, NLAT, NLON))
    for t in np.where(sel[:-1] & sel[1:])[0]:
        num[month[t + 1]] += a[t] * a[t + 1]
        den[month[t + 1]] += a[t] * a[t]
    num = num + np.roll(num, 1, 0) + np.roll(num, -1, 0)
    den = den + np.roll(den, 1, 0) + np.roll(den, -1, 0)
    phi_m = np.clip(num / np.maximum(den, 1e-12), 0.0, 0.99)
    out = np.zeros((365, NC, NLAT, NLON), "f4")
    for j in range(365):
        d = j if j > 0 else 365
        k1 = int(np.searchsorted(MONTH_MID, d)) % 12          # first mid-month at or after d (wrapping)
        k0 = (k1 - 1) % 12
        span = (MONTH_MID[k1] - MONTH_MID[k0]) % 365
        w = ((d - MONTH_MID[k0]) % 365) / span
        out[j] = (1 - w) * phi_m[k0] + w * phi_m[k1]
    return out


def fit_stats(arch: Archive, split: str = "train", clim_split: str = "train") -> dict:
    """Normalisers, AR(1) memory and static maps from the split's years; the climatology from clim_split's years
    (2005-2017 by default even when more years are used for fitting, so anomalies are measured against recent climate)."""
    y0, y1 = SPLITS[split]
    sel = (arch.years >= y0) & (arch.years <= y1)
    x = arch.data[sel]
    lt = np.log1p(np.maximum(x[:, TP], 0))
    d = np.diff(x, axis=0)
    clim = phase_climatology(arch, clim_split)
    return {
        "mean": x.mean(axis=(0, 2, 3)), "std": x.std(axis=(0, 2, 3)),
        "tp_log_mean": np.float32(lt.mean()), "tp_log_std": np.float32(lt.std()),
        "dstd": d.std(axis=(0, 2, 3)),
        "clim": clim,
        "phi": fit_phi(arch, clim, split),
        **arch.static,
    }


def reference_climatology(arch: Archive, split: str = "train") -> np.ndarray:
    """The scorer's climatology reference: the official kit's calendar-day climatology (366 slots, 15-day running
    mean) over the training years."""
    from monsoon_r1.stats import climatology
    y0, y1 = SPLITS[split]
    sel = (arch.years >= y0) & (arch.years <= y1)
    return climatology(arch.data[sel], arch.yyyymmdd[sel])


def calendar_slot(date: np.datetime64) -> int:
    """Same slotting as monsoon_r1.stats._doy_index: 0..365 with Feb 29 in its own slot."""
    import datetime as dt
    d = date.astype(object)
    return (dt.date(2000, d.month, d.day) - dt.date(2000, 1, 1)).days


def lat_w() -> np.ndarray:
    return lat_weights()
