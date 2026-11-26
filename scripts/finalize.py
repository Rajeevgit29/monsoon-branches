"""Pick the final model and produce every shipped artifact and report, in one go.

  python scripts/finalize.py --candidate phys ../runs/phys_s0 ../runs/phys_s1 ../runs/phys_s2 \
                             --candidate ar1  ../runs/ar1_s0 ../runs/ar1_s1 ../runs/ar1_s2

For each candidate (a list of independently seeded runs with the same configuration):
  1. build the ensemble, 2. fit the rain calibration on 2018-2019, 3. score it on 2018-2019 (as graded: 5-window draws,
  anti-blur, the official shocks) and 4. check its shock responses against observed ERA5 sensitivities.
Selection rule, fixed before looking at any result: the highest expected tuning score
(5-window forecast points + anti-blur + shock points) wins, unless another candidate is within 1.0 point and passes
more of the four observed-sensitivity checks, in which case that one wins.
Then, once, for the winner only: test years 2020-2021 and 2022, the server over HTTP (schema validator, latency),
the repo scan, figures and the brief.
"""
from __future__ import annotations

import argparse
import json
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WS = ROOT.parent
PY = sys.executable
KIT = str(WS / "data" / "india_climate_train.nc")
Y2022 = str(WS / "data" / "era5_2022.nc")


def run(args, log=None):
    print("$", " ".join(str(a) for a in args), flush=True)
    r = subprocess.run([str(a) for a in args], cwd=ROOT, capture_output=True, text=True)
    out = "\n".join(l for l in (r.stdout + r.stderr).splitlines() if "Warning" not in l and "warn(" not in l)
    if log:
        Path(log).write_text(out)
    if r.returncode != 0:
        print(out[-3000:]); raise SystemExit(f"failed: {args}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", nargs="+", action="append", required=True, metavar=("NAME", "RUN"))
    ap.add_argument("--skip-test", action="store_true", help="stop after selection")
    a = ap.parse_args()

    scores = {}
    for name, *runs in a.candidate:
        ens = WS / "runs" / f"ens_{name}"
        run([PY, "scripts/build_ensemble.py", "--members", *runs, "--out", ens])
        run([PY, "scripts/calibrate_rain.py", "--artifacts", ens])
        out = run([PY, "evaluate.py", "--split", "tune", "--model", ens, "--out", ens / "eval_tune.json"])
        print(out.split("shocks:")[0])
        ev = json.loads((ens / "eval_tune.json").read_text())
        run([PY, "scripts/shock_realism.py", "--artifacts", ens, "--out", ens / "realism.json"])
        rz = json.loads((ens / "realism.json").read_text())
        inside = sum(rz["models"][ens.name]["inside"])
        score = ev["bootstrap_5_windows"]["forecast_pts_mean"] + ev["anti_blur_pts"] + ev["shocks"]["causality_pts_mean"]
        scores[name] = {"dir": ens, "score": score, "realism": inside, "mean_ss": ev["mean_ss"], "csi_skill": ev["mean_csi_skill"],
                        "p_full_rmse": ev["bootstrap_5_windows"]["p_full_rmse"], "shock_pts": ev["shocks"]["causality_pts_mean"]}
        print(f"== {name}: expected tuning score {score:.2f}/70, realism {inside}/4, mean skill {ev['mean_ss']:+.3f}")

    best = max(scores, key=lambda n: scores[n]["score"])
    for n, s in scores.items():
        if n != best and s["score"] >= scores[best]["score"] - 1.0 and s["realism"] > scores[best]["realism"]:
            best = n
    print("\nselection:", json.dumps({n: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items() if k != "dir"}
                                       for n, s in scores.items()}, indent=1), "\nwinner:", best)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "selection.json").write_text(json.dumps(
        {"winner": best, "candidates": {n: {k: v for k, v in s.items() if k != "dir"} for n, s in scores.items()}}, indent=1))
    if a.skip_test:
        return

    art = ROOT / "artifacts"
    for f in ("model.npz", "static.npz"):
        shutil.copy(scores[best]["dir"] / f, art / f)
    R = ROOT / "reports"
    shutil.copy(scores[best]["dir"] / "eval_tune.json", R / "eval_tune.json")
    # held-out years, once
    run([PY, "evaluate.py", "--split", "test", "--out", R / "eval_test.json"], R / "eval_test.txt")
    run([PY, "evaluate.py", "--data", KIT, Y2022, "--split", "test2022", "--out", R / "eval_test2022.json"], R / "eval_test2022.txt")
    for m in ("persistence", "climatology"):
        run([PY, "evaluate.py", "--split", "test", "--model", m, "--out", R / f"baseline_{m}.test.json"])
    # the server, as the judge runs it
    srv = subprocess.Popen([PY, "serve.py", "--port", "8765"], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(120):
            try:
                import urllib.request
                urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8765/health", method="POST"), timeout=2)
                break
            except Exception:
                time.sleep(1)
        (R / "schema_validator.txt").write_text(run([PY, "-m", "monsoon_r1.schema_validator", "http://127.0.0.1:8765"]))
        run([PY, "evaluate.py", "--split", "test", "--url", "http://127.0.0.1:8765", "--out", R / "eval_test_http.json"], R / "eval_test_http.txt")
    finally:
        srv.send_signal(signal.SIGINT); srv.wait(timeout=20)
    scan = run([PY, "-c", "from monsoon_r1.repo_scan import scan; import json; print(json.dumps(scan('.'), indent=1))"])
    (R / "repo_scan.json").write_text(scan)
    others = [str(scores[n]["dir"]) for n in scores if n != best]
    labels = [f"final ({best})"] + [Path(o).name.replace("ens_", "") for o in others]
    run([PY, "scripts/shock_realism.py", "--artifacts", art, "--compare", *others, "--labels", *labels,
         "--out", R / "shock_realism.json", "--fig", ROOT / "docs" / "figures" / "shock_realism.png"])
    run([PY, "scripts/make_figures.py", "--out", ROOT / "docs" / "figures"])
    run([PY, "scripts/make_brief.py"])
    print("\n" + (R / "eval_test.txt").read_text()[-2500:])
    print("\n" + (R / "schema_validator.txt").read_text())
    print("repo scan:", json.loads(scan)["points"], "/ 15")


if __name__ == "__main__":
    main()
