"""Physics-informed residual CNN and the autoregressive rollout shared by training, evaluation and the server.

One forecast day = statistical base + learned dynamics:

  base:  x(t+1) = clim(t+1) + phi(season, cell, field) * (x(t) - clim(t))
         a per-cell, per-season AR(1) "red-noise" model of how fast each field's anomaly fades back to the
         training-year climatology (fitted on the training years; ocean temperature and pressure remember for
         about a week, rain for a day or two). It never does worse than climatology at long leads.
  CNN:   a correction to that base, from the last three days, static maps (land-sea mask, orography and its
         slopes, latitude, longitude), the target day's season and climatology, and physical diagnostics of the
         latest day computed on the grid: orographic lift (wind . terrain slope) and the moisture it forces up
         the Western Ghats and Himalaya, moisture flux and its convergence (in mm/day), divergence, vorticity,
         the pressure Laplacian (lows), wind speed, and column saturation from Clausius-Clapeyron.

Every layer is a full-resolution 3x3 (or 1x1) convolution and the diagnostics are centred differences: no pooling,
attention or map-wide normalisation, so one day sees a bounded neighbourhood and a boxed shock can only spread as
the rollout feeds predictions back. Rain comes from a non-negative softplus head.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .grid import NC, NLAT, NLON, T_IN, TP, YEAR_DAYS, LAT, LON

TP_SCALE = 4.0                             # mm/day per unit of the rain head
RAIN_KNEE = 10.0                           # mm/day: the heavy-rain threshold; calibration slope is 1 above it
N_PHYS = 10
R_EARTH = 6.371e6
DY = R_EARTH * math.pi / 180.0             # metres per degree of latitude
T2M, U10, V10, MSL, TCWV = 0, 2, 3, 4, 5


def n_inputs(cfg: dict) -> int:
    n = T_IN * NC + 4 + 2 + NC              # dynamic, geo, season, climatology (the original 30)
    if cfg.get("phys"):
        n += 2 + N_PHYS                      # terrain slopes + per-step diagnostics
    return n


def calibrate_rain(tp, g):
    """Piecewise-linear, continuous, monotone: g * tp below the knee, tp + (g - 1) * knee above it."""
    return torch.where(tp <= RAIN_KNEE, tp * g, tp + (g - 1) * RAIN_KNEE)


def _ddx(f: torch.Tensor, dx: torch.Tensor) -> torch.Tensor:
    """Centred eastward derivative of [..., H, W] (one-sided at the edges); dx [H, 1] metres."""
    fp = torch.cat([2 * f[..., :1] - f[..., 1:2], f, 2 * f[..., -1:] - f[..., -2:-1]], -1)
    return (fp[..., 2:] - fp[..., :-2]) / (2 * dx)


def _ddy(f: torch.Tensor) -> torch.Tensor:
    """Centred northward derivative (rows run south -> north)."""
    fp = torch.cat([2 * f[..., :1, :] - f[..., 1:2, :], f, 2 * f[..., -1:, :] - f[..., -2:-1, :]], -2)
    return (fp[..., 2:, :] - fp[..., :-2, :]) / (2 * DY)


def sat_vapour_hpa(t_k: torch.Tensor) -> torch.Tensor:
    """Clausius-Clapeyron (Bolton 1980) saturation vapour pressure over water, hPa."""
    tc = t_k - 273.15
    return 6.112 * torch.exp(17.67 * tc / (tc + 243.5))


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, padding_mode="replicate")
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, padding_mode="replicate")

    def forward(self, x):
        return x + self.c2(F.gelu(self.c1(F.gelu(x))))


class MonsoonNet(nn.Module):
    def __init__(self, n_in: int, hidden: int = 64, blocks: int = 4, rain_diag: bool = False):
        super().__init__()
        self.stem = nn.Conv2d(n_in, hidden, 3, padding=1, padding_mode="replicate")
        self.blocks = nn.Sequential(*[ResBlock(hidden) for _ in range(blocks)])
        self.head = nn.Conv2d(hidden, NC, 1)
        nn.init.zeros_(self.head.weight)   # start exactly at the base forecast
        nn.init.zeros_(self.head.bias)
        if rain_diag:
            # rain diagnosed from the trunk's features and the physics of the PREDICTED day (lift, convergence, ...)
            self.rain = nn.Sequential(nn.Conv2d(hidden + N_PHYS, 32, 3, padding=1, padding_mode="replicate"), nn.GELU(),
                                      nn.Conv2d(32, 32, 3, padding=1, padding_mode="replicate"), nn.GELU(), nn.Conv2d(32, 1, 1))
            nn.init.zeros_(self.rain[-1].weight); nn.init.zeros_(self.rain[-1].bias)

    def features(self, x):
        return F.gelu(self.blocks(self.stem(x)))

    def forward(self, x):
        return self.head(self.features(x))


class Forecaster(nn.Module):
    """Physical units in, physical units out. Normalisation constants, static maps and the AR(1) memory live in buffers."""

    def __init__(self, stats: dict, hidden: int = 64, blocks: int = 4, members: int = 1, base: str = "persistence",
                 phys: bool = False, rain_diag: bool = False):
        super().__init__()
        if base == "ar1" and "phi" not in stats:
            raise ValueError("the ar1 base needs 'phi' in the statistics")
        self.config = {"hidden": hidden, "blocks": blocks, "members": members, "base": base, "phys": bool(phys),
                       "rain_diag": bool(rain_diag)}
        if rain_diag and not phys:
            raise ValueError("rain_diag needs the physics diagnostics (phys=True)")
        f = lambda k: torch.as_tensor(np.asarray(stats[k], dtype="f4"))
        self.register_buffer("mean", f("mean"))                # [6] physical
        self.register_buffer("std", f("std"))                  # [6]
        self.register_buffer("tp_log_mean", f("tp_log_mean"))  # scalar
        self.register_buffer("tp_log_std", f("tp_log_std"))
        self.register_buffer("dstd", f("dstd"))                # [6] std of the day-to-day change
        self.register_buffer("clim", f("clim"))                # [365, 6, 32, 36] physical, phase-indexed
        self.register_buffer("phi", f("phi") if base == "ar1" else torch.zeros(1))   # [365, 6, 32, 36]
        lsm, oro = f("land_sea_mask"), f("orography_m")
        lat = torch.as_tensor(LAT, dtype=torch.float32)[:, None].expand(NLAT, NLON)
        lon = torch.as_tensor(LON, dtype=torch.float32)[None, :].expand(NLAT, NLON)
        self.register_buffer("geo", torch.stack([lsm, oro / 1000.0, (lat - 22.0) / 9.2, (lon - 83.0) / 10.4]))
        dx = (R_EARTH * math.pi / 180.0 * torch.cos(torch.deg2rad(torch.as_tensor(LAT, dtype=torch.float32))))[:, None]
        self.register_buffer("dx", dx)                         # [32, 1] metres per degree of longitude
        self.register_buffer("hx", _ddx(oro, dx))              # terrain slope, m/m, eastward
        self.register_buffer("hy", _ddy(oro))                  # northward
        pm = stats.get("phys_mean", np.zeros(N_PHYS)); ps = stats.get("phys_std", np.ones(N_PHYS))
        self.register_buffer("phys_mean", torch.as_tensor(np.asarray(pm, dtype="f4")))
        self.register_buffer("phys_std", torch.as_tensor(np.asarray(ps, dtype="f4")))
        self.register_buffer("slope_std", torch.as_tensor(float(np.asarray(stats.get("slope_std", 1.0)))))
        n_in = n_inputs(self.config)
        # one net when training; several independently trained nets for the served ensemble (each rolls out on its
        # own fed-back predictions; the rollouts are averaged, so the result stays deterministic)
        self.nets = nn.ModuleList([MonsoonNet(n_in, hidden, blocks, rain_diag) for _ in range(members)])
        self.rain_gain = None                                  # per-lead output calibration, see rollout()
        self.mos = None                                        # per-lead drift and anomaly scaling, see rollout()

    @property
    def net(self) -> MonsoonNet:
        return self.nets[0]

    # ---- normalisation ---------------------------------------------------------------------------------
    def encode(self, s: torch.Tensor) -> torch.Tensor:
        """[..., 6, H, W] physical -> normalised; rain goes through log1p first."""
        shp = [1] * (s.dim() - 3) + [NC, 1, 1]
        z = (s - self.mean.view(shp)) / self.std.view(shp)
        tp = (torch.log1p(s[..., TP:TP + 1, :, :].clamp_min(0)) - self.tp_log_mean) / self.tp_log_std
        return torch.cat([z[..., :TP, :, :], tp, z[..., TP + 1:, :, :]], dim=-3)

    def _phase(self, table: torch.Tensor, doy: torch.Tensor) -> torch.Tensor:
        """Linear interpolation of a phase-indexed [365, 6, H, W] table at fractional day of year [B]."""
        p = torch.remainder(doy, YEAR_DAYS)
        i0 = torch.floor(p).long() % 365
        i1 = (i0 + 1) % 365
        w = (p - torch.floor(p)).view(-1, 1, 1, 1)
        return table[i0] * (1 - w) + table[i1] * w

    def clim_at(self, doy: torch.Tensor) -> torch.Tensor:
        return self._phase(self.clim, doy)

    # ---- physics on the grid ---------------------------------------------------------------------------
    def phys_raw(self, s: torch.Tensor) -> torch.Tensor:
        """Diagnostics of one day [B, 6, H, W] physical -> [B, 10, H, W] in physical units."""
        u, v, t, q, p = s[:, U10], s[:, V10], s[:, T2M], s[:, TCWV].clamp_min(0.1), s[:, MSL]
        w_oro = u * self.hx + v * self.hy                                    # m/s forced up (+) or down (-) the terrain
        lift = q * F.relu(w_oro)                                             # moisture pushed up windward slopes
        qx, qy = q * u, q * v                                                # vapour flux, kg m^-1 s^-1
        mfc = -(_ddx(qx, self.dx) + _ddy(qy)) * 86400.0                      # convergence, kg m^-2 day^-1 = mm/day
        div = (_ddx(u, self.dx) + _ddy(v)) * 1e5
        vort = (_ddx(v, self.dx) - _ddy(u)) * 1e5
        lap_p = (_ddx(_ddx(p, self.dx), self.dx) + _ddy(_ddy(p))) * 1e9      # Pa per (1000 km)^2 / 1000
        speed = torch.sqrt(u * u + v * v + 1e-6)
        crh = torch.log(q) - torch.log(sat_vapour_hpa(t))                     # log column vapour vs saturation
        return torch.stack([w_oro, lift, qx, qy, mfc, div, vort, lap_p, speed, crh], 1)

    def phys_features(self, s: torch.Tensor) -> torch.Tensor:
        z = (self.phys_raw(s) - self.phys_mean.view(1, -1, 1, 1)) / self.phys_std.view(1, -1, 1, 1)
        return z.clamp(-6, 6)

    # ---- one day ahead ---------------------------------------------------------------------------------
    def base(self, last: torch.Tensor, doy_next: torch.Tensor) -> torch.Tensor:
        if self.config["base"] != "ar1":
            return last
        c1, c0 = self.clim_at(doy_next), self.clim_at(doy_next - 1)
        return c1 + self._phase(self.phi, doy_next) * (last - c0)

    def step(self, hist: torch.Tensor, doy_next: torch.Tensor, net: MonsoonNet | None = None) -> torch.Tensor:
        """hist [B,3,6,H,W] physical (oldest first), doy_next [B] = day of year of the day being predicted."""
        b = hist.shape[0]
        z = self.encode(hist).reshape(b, T_IN * NC, NLAT, NLON)
        cl = self.encode(self.clim_at(doy_next))
        a = 2 * math.pi * doy_next / YEAR_DAYS
        seas = torch.stack([torch.sin(a), torch.cos(a)], 1)[:, :, None, None].expand(b, 2, NLAT, NLON)
        parts = [z, self.geo.expand(b, -1, -1, -1), seas, cl]
        if self.config["phys"]:
            slopes = torch.stack([self.hx, self.hy]) / self.slope_std
            parts += [slopes.expand(b, -1, -1, -1), self.phys_features(hist[:, -1])]
        net = net or self.nets[0]
        feat = net.features(torch.cat(parts, 1))
        h = net.head(feat)
        base = self.base(hist[:, -1], doy_next)
        nxt = base + h * self.dstd.view(1, NC, 1, 1)
        logit = h[:, TP:TP + 1]
        if self.config.get("rain_diag"):
            # moisture budget and orography of the predicted day: rain responds to the circulation it falls in
            logit = logit + net.rain(torch.cat([feat, self.phys_features(nxt)], 1))
        if self.config["base"] == "ar1":
            # rain: the head shifts the base amount inside the softplus, so a zero correction returns the base
            b_tp = base[:, TP:TP + 1].clamp_min(0.02) / TP_SCALE
            rain = F.softplus(logit + torch.log(torch.expm1(b_tp))) * TP_SCALE
        else:
            rain = F.softplus(h[:, TP:TP + 1]) * TP_SCALE
        return torch.cat([nxt[:, :TP], rain, nxt[:, TP + 1:]], 1)

    def rain_teacher(self, hist: torch.Tensor, doy_next: torch.Tensor, true_next: torch.Tensor) -> torch.Tensor:
        """Training aid: tomorrow's rain diagnosed from tomorrow's TRUE circulation and moisture (lift, convergence,
        saturation), with the same rain head. Fitting it teaches the head the observed physical dependence of rain
        on the state it falls in, which is what decides the response to a shock."""
        b = hist.shape[0]
        net = self.nets[0]
        z = self.encode(hist).reshape(b, T_IN * NC, NLAT, NLON)
        cl = self.encode(self.clim_at(doy_next))
        a = 2 * math.pi * doy_next / YEAR_DAYS
        seas = torch.stack([torch.sin(a), torch.cos(a)], 1)[:, :, None, None].expand(b, 2, NLAT, NLON)
        slopes = torch.stack([self.hx, self.hy]) / self.slope_std
        x = torch.cat([z, self.geo.expand(b, -1, -1, -1), seas, cl, slopes.expand(b, -1, -1, -1), self.phys_features(hist[:, -1])], 1)
        feat = net.features(x)
        logit = net.head(feat)[:, TP:TP + 1] + net.rain(torch.cat([feat, self.phys_features(true_next)], 1))
        base = self.base(hist[:, -1], doy_next)
        b_tp = base[:, TP:TP + 1].clamp_min(0.02) / TP_SCALE
        return F.softplus(logit + torch.log(torch.expm1(b_tp))) * TP_SCALE

    def rollout(self, init: torch.Tensor, doy_last: torch.Tensor, steps: int, calibrate: bool = False) -> torch.Tensor:
        """init [B,3,6,H,W] physical, doy_last [B] day of year of init[:, -1]. Returns [B,steps,6,H,W]: index 0 is tomorrow.
        Each step drops the oldest day, appends all six predicted fields and advances the seasonal phase by one day.
        calibrate=True applies a per-lead rain calibration, fitted on the tuning years, to the RETURNED rain only (the
        fed-back state is never changed): rain below the 10 mm/day heavy-rain knee is multiplied by the lead's gain g,
        rain above it is shifted by the same amount (slope 1), so moderate rain that an MSE-trained rollout damps is
        restored while large values and large shock responses are not amplified."""
        runs = []
        for net in self.nets:
            hist, outs = init, []
            for k in range(1, steps + 1):
                nxt = self.step(hist, doy_last + k, net)
                outs.append(nxt)
                hist = torch.cat([hist[:, 1:], nxt[:, None]], 1)
            runs.append(torch.stack(outs, 1))
        out = runs[0] if len(runs) == 1 else torch.stack(runs).mean(0)
        if calibrate and self.rain_gain is not None:
            g = torch.ones(steps, dtype=out.dtype)
            n = min(steps, len(self.rain_gain))
            g[:n] = torch.as_tensor(self.rain_gain[:n], dtype=out.dtype)
            g[n:] = float(self.rain_gain[-1])
            out = torch.cat([out[:, :, :TP], calibrate_rain(out[:, :, TP:TP + 1], g.to(out.device).view(1, -1, 1, 1, 1)),
                             out[:, :, TP + 1:]], 2)
        if calibrate and self.mos is not None:
            # per lead and field (not rain): remove the domain-mean drift and scale the anomaly about the climatology by
            # lambda, fitted on the tuning years; a deterministic forecast's long-lead anomalies are too large
            n = min(steps, len(self.mos["lam"]))
            lam = torch.ones(steps, NC, dtype=out.dtype); bias = torch.zeros(steps, NC, dtype=out.dtype)
            lam[:n] = torch.as_tensor(self.mos["lam"][:n], dtype=out.dtype); bias[:n] = torch.as_tensor(self.mos["bias"][:n], dtype=out.dtype)
            lam[n:] = lam[n - 1]; bias[n:] = bias[n - 1]
            lam[:, TP] = 1.0; bias[:, TP] = 0.0
            cl = torch.stack([self.clim_at(doy_last + k) for k in range(1, steps + 1)], 1)
            lam, bias = lam.to(out.device).view(1, steps, NC, 1, 1), bias.to(out.device).view(1, steps, NC, 1, 1)
            out = cl + lam * (out - bias - cl)
        return out

    # ---- artifacts -------------------------------------------------------------------------------------
    def save(self, out_dir: Path, meta: dict | None = None) -> None:
        """Weights -> model.npz (plain arrays + JSON config, no pickle); stats and static maps -> static.npz."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        w = {f"m{i}.{k}": v.detach().cpu().numpy() for i, net in enumerate(self.nets) for k, v in net.state_dict().items()}
        cfg = json.dumps({**(meta or {}), **self.config,
                          "rain_gain": None if self.rain_gain is None else [float(x) for x in self.rain_gain],
                          "mos": self.mos})
        np.savez(out_dir / "model.npz", __config__=np.array(cfg), **w)
        st = dict(mean=self.mean, std=self.std, tp_log_mean=self.tp_log_mean, tp_log_std=self.tp_log_std, dstd=self.dstd,
                  clim=self.clim, land_sea_mask=self.geo[0], orography_m=self.geo[1] * 1000.0,
                  phys_mean=self.phys_mean, phys_std=self.phys_std, slope_std=self.slope_std)
        if self.config["base"] == "ar1":
            st["phi"] = self.phi
        np.savez(out_dir / "static.npz", **{k: v.detach().cpu().numpy() for k, v in st.items()})

    @classmethod
    def load(cls, art_dir: Path) -> "Forecaster":
        art_dir = Path(art_dir)
        st = dict(np.load(art_dir / "static.npz"))
        mz = np.load(art_dir / "model.npz")
        cfg = json.loads(str(mz["__config__"]))
        keys = [k for k in mz.files if k != "__config__"]
        if not keys[0].startswith("m0."):                       # single-member file written before ensembles
            keys_by = {0: {k: k for k in keys}}
        else:
            keys_by = {}
            for k in keys:
                i, name = k.split(".", 1)
                keys_by.setdefault(int(i[1:]), {})[name] = k
        m = cls(st, hidden=cfg["hidden"], blocks=cfg["blocks"], members=len(keys_by),
                base=cfg.get("base", "persistence"), phys=cfg.get("phys", False), rain_diag=cfg.get("rain_diag", False))
        for i, names in keys_by.items():
            m.nets[i].load_state_dict({n: torch.from_numpy(mz[k]) for n, k in names.items()})
        cfg["members"] = len(keys_by)
        m.meta = cfg
        m.rain_gain = cfg.get("rain_gain")
        m.mos = cfg.get("mos")
        return m.eval()
