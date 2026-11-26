# Third-party notices

| What | Where it is used | Licence |
|---|---|---|
| Official Round 1 kit `monsoon_r1` (OfficialAbhinavSingh/yug-monsoon-kit v1.0.0) | Training-side only: scoring functions, schema validator, repo scan, the reference climatology. Installed from the release wheel, not vendored. `serve.py` does not import it. The starter `serve.py` template shaped the server's API plumbing. | MIT |
| ERA5-derived training data (`india_climate_train.nc`) | Training and evaluation; six held-out 2020–2021 windows are bundled in `dashboard/data/examples.json` for the demo | Contains modified Copernicus Climate Change Service information (ERA5, Hersbach et al. 2020), via Google WeatherBench 2 (Rasp et al. 2024). CC BY 4.0 |
| Natural Earth 1:50m coastline | `dashboard/data/coast.json`, clipped to the grid | Public domain |
| ERA5 surface geopotential and land-sea mask, 0.25° | `dashboard/data/terrain.json` (heights in metres and land fraction for the 3D relief; display only, never used by the model) | Contains modified Copernicus Climate Change Service information (ERA5), via Google WeatherBench 2. CC BY 4.0 |
| three.js r160 (`three.module.min.js`, `OrbitControls.js`) | Vendored in `dashboard/vendor/` so the 3D view works offline; licence in `dashboard/vendor/LICENSE-three.txt` | MIT |
| Archivo typeface (Omnibus-Type), Latin subset | Vendored in `dashboard/vendor/archivo-latin.woff2`; licence in `dashboard/vendor/LICENSE-archivo.txt` | SIL Open Font License 1.1 |
| PyTorch, NumPy, FastAPI, Uvicorn, xarray, pandas, matplotlib | Runtime and tooling dependencies, installed from PyPI | BSD / MIT / BSD-style |

Reference projects reviewed for ideas (VAYU, AGNI, MausamSetu, Hirtheesh1/final_project1, Climate-X): no code was copied.
Ideas taken: a three-panel baseline / scenario / difference map with a shared rain scale and a zero-centred difference
scale (VAYU), a sliding-window autoregressive rollout (AGNI), a longest-dry-spell indicator (MausamSetu), and the
region → shock → inspect-a-cell interaction (Hirtheesh1, Climate-X). All were re-implemented from scratch for this grid and API.
