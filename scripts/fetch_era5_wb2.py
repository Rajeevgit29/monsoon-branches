"""Extend the training archive with more ERA5 years from WeatherBench 2 (training-side only, never used by the server).

  python scripts/fetch_era5_wb2.py --start 1979-01-01 --end 2004-12-31 --out ../data/era5_1979_2004.nc
  python scripts/fetch_era5_wb2.py --verify ../data/india_climate_train.nc        # reproduce kit days exactly first

Same source and the same processing as the official kit (monsoon_r1.build_dataset.wb2_daily): the 1-degree
conservative 6-hourly WeatherBench 2 ERA5, integer-degree points 6..38 N x 65..101 E, daily means of the
00/06/12/18 UTC samples (rain: the four 6 h accumulations ending 06, 12, 18 and 24 UTC, m -> mm), then the mean of
the 4 surrounding integer points for each half-degree cell centre. Chunks are fetched directly over HTTPS
(one chunk = one day of one variable for the whole globe), many in parallel, and cut to the window.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BASE = "https://storage.googleapis.com/weatherbench2/datasets/era5/1959-2023_01_10-6h-360x181_equiangular_with_poles_conservative.zarr"
VARS = ["2m_temperature", "total_precipitation_6hr", "10m_u_component_of_wind", "10m_v_component_of_wind",
        "mean_sea_level_pressure", "total_column_water_vapour"]
T0 = dt.date(1959, 1, 1)
LAT_I = slice(6 + 90, 38 + 90 + 1)        # 33 integer latitudes 6..38 (array is -90..90 ascending)
LON_I = slice(65, 101 + 1)                # 37 integer longitudes 65..101


_local = threading.local()


def _session():
    """One keep-alive HTTPS session per worker thread: far fewer TLS handshakes than a new connection per chunk."""
    if not hasattr(_local, "s"):
        import requests
        _local.s = requests.Session()
        _local.s.mount("https://", requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=1))
    return _local.s


def fetch_chunk(var: str, k: int, tries: int = 20) -> np.ndarray:
    """Chunk k of a variable = the four 6-hourly steps of day T0 + k. Returns [4, 33 lat, 37 lon] float32."""
    from numcodecs import Blosc
    url = f"{BASE}/{var}/{k}.0.0"
    for t in range(tries):
        try:
            r = _session().get(url, timeout=(10, 30))
            r.raise_for_status()
            a = np.frombuffer(Blosc().decode(r.content), dtype="<f4").reshape(4, 360, 181)
            return np.ascontiguousarray(a[:, LON_I, LAT_I].transpose(0, 2, 1))
        except Exception:
            if hasattr(_local, "s"):
                del _local.s                                   # drop a broken connection, open a fresh one
            if t == tries - 1:
                raise
            time.sleep(min(20.0, 1.0 + 1.5 * t))


def half_degree(x):
    return 0.25 * (x[..., :-1, :-1] + x[..., 1:, :-1] + x[..., :-1, 1:] + x[..., 1:, 1:])


def build(days: list[dt.date], workers: int = 128) -> np.ndarray:
    ks = [(d - T0).days for d in days]
    need = {(v, k) for v in VARS for k in ks}
    need |= {(VARS[1], k + 1) for k in ks}                    # rain needs the next day's 00 UTC accumulation
    jobs = sorted(need)
    got = {}
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        for n, (job, arr) in enumerate(zip(jobs, ex.map(lambda j: fetch_chunk(*j), jobs))):
            got[job] = arr
            if (n + 1) % 500 == 0:
                rate = (n + 1) / (time.time() - t0)
                print(f"  {n + 1}/{len(jobs)} chunks, {rate:.1f}/s, eta {(len(jobs) - n - 1) / rate / 60:.0f} min", flush=True)
    out = np.zeros((len(days), 6, 32, 36), "f4")
    for i, k in enumerate(ks):
        for c, v in enumerate(VARS):
            if c == 1:
                a = got[(v, k)]
                day = (a[1:4].sum(axis=0) + got[(v, k + 1)][0]) * 1000.0
            else:
                day = got[(v, k)].mean(axis=0)
            out[i, c] = half_degree(day)
    return out


def save_nc(path: Path, data: np.ndarray, days: list[dt.date], static: np.ndarray, static_names) -> None:
    import xarray as xr
    from monsoon_branches.grid import CHANNELS, LAT, LON
    ds = xr.Dataset({"data": (("time", "channel", "lat", "lon"), data),
                     "static": (("static_channel", "lat", "lon"), static)},
                    coords={"time": np.array(days, dtype="datetime64[ns]"), "channel": list(CHANNELS), "lat": LAT, "lon": LON,
                            "static_channel": list(static_names)})
    ds["data"].attrs["units"] = "t2m K | tp mm/day | u10 m/s | v10 m/s | msl Pa | tcwv kg/m2"
    ds.attrs["source"] = "ERA5 via WeatherBench 2 (1 deg conservative, 6-hourly), processed as monsoon_r1.build_dataset.wb2_daily"
    ds.to_netcdf(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start"); ap.add_argument("--end"); ap.add_argument("--out")
    ap.add_argument("--verify", default=None, help="kit archive to reproduce a few days of")
    ap.add_argument("--static-from", default=str(ROOT.parent / "data" / "india_climate_train.nc"))
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    if a.verify:
        import xarray as xr
        ds = xr.open_dataset(a.verify)
        days = [dt.date(2005, 1, 1), dt.date(2013, 6, 14), dt.date(2021, 7, 19), dt.date(2021, 12, 31)]
        mine = build(days, 32)
        for d, m in zip(days, mine):
            ref = ds["data"].sel(time=np.datetime64(d)).values
            err = np.abs(m - ref).max(axis=(1, 2))
            print(d, "max |diff| per channel:", " ".join(f"{e:.2e}" for e in err))
        return
    s, e = dt.date.fromisoformat(a.start), dt.date.fromisoformat(a.end)
    days = [s + dt.timedelta(i) for i in range((e - s).days + 1)]
    print(f"fetching {len(days)} days x 6 variables from WeatherBench 2 ...", flush=True)
    cache = Path(a.out).parent / "wb2_cache"
    cache.mkdir(parents=True, exist_ok=True)
    parts = []
    for year in sorted({d.year for d in days}):                       # one resumable piece per calendar year
        yd = [d for d in days if d.year == year]
        f = cache / f"{yd[0]}_{yd[-1]}.npy"
        if not f.exists():
            t0 = time.time()
            arr = build(yd, a.workers)
            np.save(f, arr)
            print(f"  {year}: {len(yd)} days in {(time.time() - t0) / 60:.1f} min", flush=True)
        parts.append(np.load(f))
    data = np.concatenate(parts)
    assert np.isfinite(data).all()
    import xarray as xr
    st = xr.open_dataset(a.static_from)
    save_nc(Path(a.out), data, days, st["static"].values, [str(x) for x in st["static_channel"].values])
    print(f"wrote {a.out}: {data.shape}")


if __name__ == "__main__":
    main()
