"""Grid, channel and calendar conventions shared by training, evaluation and the server.

Mirrors the official Round 1 contract: 32 x 36 cells of 1 degree, rows south -> north (6.5..37.5 N),
columns west -> east (65.5..100.5 E), channels in the order below, physical units on the wire.
This module is imported by the server, so it holds no data paths or calendar literals.
"""
from __future__ import annotations

import math

import numpy as np

CHANNELS = ("t2m", "tp", "u10", "v10", "msl", "tcwv")
UNITS = ("K", "mm/day", "m/s", "m/s", "Pa", "kg/m2")
NC, NLAT, NLON = 6, 32, 36
T_IN = 3
TP = 1                                     # index of the precipitation channel
LAT = np.arange(6.5, 37.5 + 1e-9, 1.0)
LON = np.arange(65.5, 100.5 + 1e-9, 1.0)
YEAR_DAYS = 365.0                          # the kit's seasonal cycle: doy_sincos = [sin, cos](2 pi doy / 365)

# (lat_min, lat_max, lon_min, lon_max) in degrees, as in the official shocks
BOX_BAY_OF_BENGAL = (8.0, 20.0, 82.0, 94.0)
BOX_ARABIAN_SEA = (8.0, 20.0, 65.0, 75.0)


def box_mask(box) -> np.ndarray:
    la0, la1, lo0, lo1 = box
    return ((LAT >= la0) & (LAT <= la1))[:, None] & ((LON >= lo0) & (LON <= lo1))[None, :]


def lat_weights() -> np.ndarray:
    w = np.cos(np.deg2rad(LAT))
    return w / w.mean()


def doy_from_sincos(s: float, c: float) -> float:
    """Inverse of the kit's encoding: a fractional day of year in [0, 365)."""
    return (math.atan2(float(s), float(c)) * YEAR_DAYS / (2 * math.pi)) % YEAR_DAYS


def sincos(doy: float) -> list[float]:
    a = 2 * math.pi * doy / YEAR_DAYS
    return [math.sin(a), math.cos(a)]
