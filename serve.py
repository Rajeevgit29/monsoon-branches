"""Monsoon Branches forecast server (Round 1 contract).

    python serve.py --port 8000

POST /health   -> {"status": "ready", "model_name": ...} once the weights are loaded and warmed up.
POST /simulate -> {"rollout": [steps,6,32,36], "inference_ms": ...}, physical units, index 0 = tomorrow.

Runs fully offline: weights and static maps are read from ./artifacts next to this file, once, at start-up.
The perturbation (a physical-unit delta) is added to init_states[-1] before normalisation, then the same
deterministic rollout runs, so a zero perturbation reproduces the unperturbed forecast exactly.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from monsoon_branches.grid import NC, NLAT, NLON, T_IN, TP, doy_from_sincos
from monsoon_branches.model import Forecaster

ART = Path(__file__).resolve().parent / "artifacts"
MAX_STEPS = 30
T_MIN, T_MAX = 233.65, 332.65          # stay inside the validity gate (-40..+60 C) with a small margin

torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
MODEL = Forecaster.load(ART)
MODEL_NAME = (f"monsoon-branches-{MODEL.config['base']}{'-phys' if MODEL.config.get('phys') else ''}"
              f"{'-sens' if (MODEL.meta.get('sens') or 0) > 0 else ''}-h{MODEL.config['hidden']}b{MODEL.config['blocks']}x{len(MODEL.nets)}")


def forecast(init: np.ndarray, doy_last: float, steps: int) -> np.ndarray:
    with torch.inference_mode():
        x = torch.from_numpy(np.ascontiguousarray(init, dtype=np.float32))[None]
        out = MODEL.rollout(x, torch.tensor([doy_last], dtype=torch.float32), steps, calibrate=True)[0]
        clim = MODEL.clim_at(torch.tensor([doy_last + k for k in range(1, steps + 1)], dtype=torch.float32))
    out, clim = out.numpy().astype(np.float64), clim.numpy().astype(np.float64)
    bad = ~np.isfinite(out)
    if bad.any():                                   # never ship NaN/Inf: fall back to climatology cell by cell
        out[bad] = clim[bad]
    out[:, TP] = np.maximum(out[:, TP], 0.0)
    out[:, 0] = np.clip(out[:, 0], T_MIN, T_MAX)
    out[:, 5] = np.maximum(out[:, 5], 0.0)
    return out


forecast(MODEL.clim_at(torch.tensor([150.0, 151.0, 152.0])).numpy(), 152.0, 10)   # warm-up, outside any scored call
READY = True

app = FastAPI(title="Monsoon Branches")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _arr(x, shape) -> np.ndarray:
    """Nested lists (the contract) or {'b64','shape'} (tolerated)."""
    if isinstance(x, dict):
        a = np.frombuffer(base64.b64decode(x["b64"]), dtype="<f4").reshape([int(s) for s in x["shape"]])
    else:
        a = np.asarray(x, dtype=np.float64)
    if a.shape != shape:
        raise ValueError(f"expected shape {shape}, got {a.shape}")
    return a.astype(np.float64)


@app.post("/health")
@app.get("/health")
def health():
    return {"status": "ready" if READY else "loading", "model_name": MODEL_NAME}


@app.post("/simulate")
async def simulate(request: Request):
    t0 = time.perf_counter()
    try:
        body = json.loads(await request.body())
        init = _arr(body["init_states"], (T_IN, NC, NLAT, NLON))
        steps = int(body.get("steps", 10))
        if not 1 <= steps <= MAX_STEPS:
            raise ValueError(f"steps must be in 1..{MAX_STEPS}")
        s, c = body["doy_sincos"]
        doy_last = doy_from_sincos(s, c)
        pert = body.get("perturbation")
        if pert is not None:
            init[-1] = init[-1] + _arr(pert, (NC, NLAT, NLON))
    except (KeyError, TypeError, ValueError) as e:
        return JSONResponse({"error": f"bad request: {e}"}, status_code=422)
    if not np.isfinite(init).all():                 # defensive: patch missing input cells with climatology
        cl = MODEL.clim_at(torch.tensor([doy_last - 2, doy_last - 1, doy_last], dtype=torch.float32)).numpy()
        init = np.where(np.isfinite(init), init, cl)
    init[:, TP] = np.maximum(init[:, TP], 0.0)
    init[:, 5] = np.maximum(init[:, 5], 0.0)
    out = forecast(init, doy_last, steps)
    ms = (time.perf_counter() - t0) * 1000.0
    payload = json.dumps({"rollout": np.round(out, 4).tolist(), "inference_ms": ms, "model_name": MODEL_NAME})
    return Response(payload, media_type="application/json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    a = ap.parse_args()
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
