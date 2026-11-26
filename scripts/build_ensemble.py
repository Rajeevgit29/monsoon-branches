"""Combine independently trained single-model artifacts into one ensemble artifact.

  python scripts/build_ensemble.py --members ../runs/v1 ../runs/s1 ../runs/s2 --out artifacts
  python scripts/calibrate_rain.py --artifacts artifacts      # refit the rain gain for the ensemble mean

All members must share static.npz (same training years, so the same normalisers and climatology).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from monsoon_branches.model import Forecaster  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--out", default=str(ROOT / "artifacts"))
    a = ap.parse_args()
    ms = [Forecaster.load(p) for p in a.members]
    for m in ms:                                  # compare architectures without the per-file member count
        m.config = {k: v for k, v in m.config.items() if k != "members"}
    s0 = dict(np.load(Path(a.members[0]) / "static.npz"))
    for p in a.members[1:]:
        s = dict(np.load(Path(p) / "static.npz"))
        assert all(np.array_equal(s0[k], s[k]) for k in s0), f"{p}: static.npz differs from {a.members[0]}"
    cfg = ms[0].config
    for m in ms[1:]:
        assert m.config == ms[0].config, f"members differ in architecture: {m.config} vs {ms[0].config}"
    ens = Forecaster(s0, cfg["hidden"], cfg["blocks"], members=sum(len(m.nets) for m in ms),
                     base=cfg["base"], phys=cfg["phys"], rain_diag=cfg.get("rain_diag", False))
    i = 0
    for m in ms:
        for net in m.nets:
            ens.nets[i].load_state_dict(net.state_dict()); i += 1
    meta = {"fit_split": ms[0].meta.get("fit_split"), "members_from": [Path(p).name for p in a.members],
            "seeds": [m.meta.get("seed") for m in ms], "data": ms[0].meta.get("data"), "stages": ms[0].meta.get("stages"),
            "rain_aux": ms[0].meta.get("rain_aux", 0.0), "sens": ms[0].meta.get("sens", 0.0), "sens_band": ms[0].meta.get("sens_band", 1.0),
            "init": ms[0].meta.get("init"), "init_members": [m.meta.get("init") for m in ms]}
    ens.save(Path(a.out), meta)
    print(f"wrote {i}-member ensemble to {a.out}")


if __name__ == "__main__":
    main()
