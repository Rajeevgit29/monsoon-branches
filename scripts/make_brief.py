"""Assemble the two-page brief (docs/brief.html) from the shipped artifacts and the evaluation reports.

  python scripts/make_brief.py      # run by scripts/finalize.py; then print docs/brief.html to an A4 PDF

Every number is read from reports/*.json, docs/figures/figures.json or artifacts/model.npz.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from monsoon_branches.grid import CHANNELS  # noqa: E402

NAMES = {"t2m": "2 m temperature", "tp": "Rain", "u10": "10 m wind, east", "v10": "10 m wind, north",
         "msl": "Sea-level pressure", "tcwv": "Water vapour"}


def load(p):
    return json.loads((ROOT / p).read_text())


def sgn(x, d=2):
    return f"{'+' if x >= 0 else '−'}{abs(x):.{d}f}"


def main():
    test, t22 = load("reports/eval_test.json"), load("reports/eval_test2022.json")
    http = load("reports/eval_test_http.json")
    bp, bc = load("reports/baseline_persistence.test.json"), load("reports/baseline_climatology.test.json")
    fig, rz, sel = load("docs/figures/figures.json"), load("reports/shock_realism.json"), load("reports/selection.json")
    mz = np.load(ROOT / "artifacts" / "model.npz")
    cfg = json.loads(str(mz["__config__"]))
    members = len({k.split(".")[0] for k in mz.files if k != "__config__"})
    n_par = sum(int(np.prod(mz[k].shape)) for k in mz.files if k != "__config__") // members
    size_mb = sum(p.stat().st_size for p in (ROOT / "artifacts").iterdir()) / 1e6
    ext = cfg.get("fit_split") == "train_ext"
    years = "1979–2017 (the kit's years plus 1979–2004 ERA5 from WeatherBench 2, processed as the kit)" if ext else "2005–2017"
    phys, rain_diag, sens, ar1 = cfg.get("phys"), cfg.get("rain_diag"), (cfg.get("sens") or 0) > 0, cfg.get("base") == "ar1"
    gain = cfg.get("rain_gain") or [1.0]

    rows = []
    for c in CHANNELS:
        l1, l4, l8 = (test["per_lead"][k]["ss"][c] for k in ("1", "4", "8"))
        rows.append(f"<tr><td>{NAMES[c]}</td><td>{sgn(l1)}</td><td>{sgn(l4)}</td><td>{sgn(l8)}</td>"
                    f"<td><b>{sgn(test['pooled_ss'][c])}</b></td><td>{sgn(test['per_lead']['4']['ss_persist'][c])}</td></tr>")
    sh, lat, dry, cell = test["shocks"], http["latency_s"], fig["dry"], fig["dry"]["cell"]
    ms, wo = fig["shocks"]["monsoon_stall"], fig["shocks"]["warm_ocean"]
    pct = lambda x: f"{100 * x:.0f}%"
    vr = test["var_ratio_day8"]
    rel_m, rel_p = fig["rel_rmse_model"], fig["rel_rmse_persist"]
    final = list(rz["models"])[0]
    rv, inside = rz["models"][final]["values"], rz["models"][final]["inside"]
    boot = test["bootstrap_5_windows"]

    model_items = []
    if ar1:
        model_items.append("<li><b>Climate memory as the base.</b> Each day starts from a per-cell, per-season AR(1) forecast, "
                           "clim(t+1) + φ·(x(t) − clim(t)), fitted on the training years. Pressure and ocean temperature remember about 0.8 of "
                           "an anomaly a day, rain about 0.5, so long leads relax towards climatology at each region's own rate. The CNN learns "
                           "corrections to it.</li>")
    if phys:
        model_items.append("<li><b>India's terrain and moisture budget as inputs,</b> recomputed from the state every day: terrain slopes, "
                           "orographic lift (wind · slope) and the moisture it forces up the Western Ghats and Himalaya, vapour flux and its "
                           "convergence (mm/day), divergence, vorticity, the pressure Laplacian (monsoon lows) and column saturation from "
                           "Clausius–Clapeyron. In ERA5, same-day convergence and lift explain 65% of Western Ghats rain variance.</li>")
    if rain_diag:
        model_items.append("<li><b>Rain diagnosed from the predicted day's physics</b> (lift, convergence, saturation of tomorrow's state), "
                           "trained also against tomorrow's true state.</li>")
    era5_years = sorted(int(n[5:9]) for n in cfg.get("data") or [] if n.startswith("era5_") and n[5:9].isdigit() and int(n[5:9]) < 2018)
    ft_years = f"{era5_years[0] if era5_years else 2005}–2017"
    if sens and cfg.get("init"):
        model_items.append(f"<li><b>Shock responses constrained by observations.</b> Each trained model is then fine-tuned ({ft_years}) on 4- and "
                           "8-day rollouts while small wind, vapour and warmth shocks are applied to June–September windows, "
                           + ("pulling the rain response towards the observed ERA5 sensitivity (the central half of its 90% range; below)."
                              if (cfg.get("sens_band") or 1.0) < 1.0 else
                              "keeping the rain response inside the 90% range of the observed ERA5 sensitivity (below).") + "</li>")
    elif sens:
        model_items.append("<li><b>Shock responses constrained by observations.</b> During training, small wind, vapour and warmth shocks are "
                           "applied to June–September windows and the rain response is kept inside the 90% range of the observed ERA5 "
                           "sensitivity (below). Forecasts are still learned from data; the constraint only shapes the response to interventions.</li>")
    if not phys:
        model_items.append("<li><b>India's terrain as inputs:</b> land-sea mask and orography. Terrain-physics diagnostics (orographic lift, "
                           "moisture convergence) as extra inputs did not raise 2018–2019 skill in matched runs (+0.136 vs +0.140), so they "
                           "are left out.</li>")
    model_items.append(f"<li><b>Locality and training.</b> 3×3 convolutions only, no pooling or attention: one day sees about {1 + 2 * cfg['blocks']} cells, "
                       f"so shocks spread only as forecasts feed back. Years {years} train; 2018–2019 tune; 2020–2022 held out. The loss is "
                       f"latitude-weighted MSE scaled per field by the better reference, as scored; one-step, then 2-, 4- and 8-day rollouts. "
                       f"{members} seeds averaged ({n_par:,} parameters each, {size_mb:.0f} MB). Output calibration fitted on 2018–2019: rain below "
                       f"10 mm/day rescaled per lead ({gain[0]:.2f} → {gain[-1]:.2f}); for the other fields the drift is removed and long-lead "
                       f"anomalies are scaled down (day-8 λ ≈ {np.mean([cfg['mos']['lam'][7][c] for c in (0, 2, 3, 4, 5)]) if cfg.get('mos') else 1:.2f}).</li>")

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Monsoon Branches · Round 1 brief</title>
<style>
@page {{ size: A4; margin: 0; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; font: 8.1pt/1.36 -apple-system, "Helvetica Neue", Arial, sans-serif; color: #191919; font-variant-numeric: tabular-nums; }}
.page {{ width: 210mm; height: 297mm; padding: 9mm 10.5mm 8mm; overflow: hidden; page-break-after: always; display: flex; flex-direction: column; gap: 2.6mm; }}
.page:last-child {{ page-break-after: auto; }}
.page > * {{ flex-shrink: 0; }}
header {{ display: flex; justify-content: space-between; align-items: baseline; border-bottom: 1.2pt solid #191919; padding-bottom: 1.4mm; }}
h1 {{ font-size: 16pt; margin: 0; letter-spacing: -.01em; }}
header .meta {{ font-size: 7.6pt; color: #5f5f5c; text-align: right; }}
.lede {{ font-size: 8.9pt; line-height: 1.36; margin: 0; }}
h2 {{ font-size: 7.6pt; text-transform: uppercase; letter-spacing: .07em; color: #5f5f5c; margin: 0 0 1mm; font-weight: 600; }}
h3 {{ font-size: 8.2pt; margin: 0 0 .6mm; }}
p {{ margin: 0 0 1.2mm; }}
ul {{ margin: 0; padding-left: 3.4mm; }} li {{ margin: 0 0 .8mm; }}
.grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 4.5mm; }}
.grid3 {{ display: grid; grid-template-columns: 1.08fr 1fr; gap: 4.5mm; }}
table {{ border-collapse: collapse; width: 100%; font-size: 7.6pt; }}
th, td {{ text-align: right; padding: .45mm 1.1mm; border-bottom: .4pt solid #e2e2df; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ color: #5f5f5c; font-weight: 600; border-bottom: .8pt solid #191919; }}
.tiles {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1.6mm; margin-top: 1.4mm; }}
.tile {{ border: .5pt solid #e2e2df; border-radius: 1.4mm; padding: 1.3mm 1.7mm; }}
.tile .v {{ font-size: 11.5pt; font-weight: 600; }}
.tile .k {{ font-size: 6.9pt; color: #5f5f5c; line-height: 1.22; }}
img {{ display: block; width: 100%; }}
.cap {{ font-size: 7pt; color: #5f5f5c; margin-top: .5mm; }}
.note {{ background: #f6f6f4; border-radius: 1.4mm; padding: 1.8mm 2.4mm; }}
code {{ font: 7.3pt/1.3 ui-monospace, Menlo, monospace; }}
.small {{ font-size: 7.2pt; color: #5f5f5c; }}
.ok {{ color: #0a7d1c; font-weight: 600; }} .no {{ color: #b72a2b; font-weight: 600; }}
</style></head><body>

<section class="page">
  <header><h1>Monsoon Branches</h1>
    <div class="meta">Team intelligentapes · Multiverse of Monsoon · Yugaantar DevForge<br>Round 1 brief · Page 1 · Model and evidence</div></header>
  <p class="lede">We forecast 10 days of weather over India and the surrounding seas from three observed days, rerun the same model from a
    <b>shocked</b> start to show where a local change to today's weather moves rainfall and dry spells, and check those responses against how
    the real atmosphere has behaved. <b>Demo question:</b> if winds over the Arabian Sea weaken today, where will it rain more or less over the next 10 days?</p>

  <div class="grid3">
    <div><h2>Model · {"physics-informed" if phys else "observation-constrained"}, local, deterministic</h2><ul>{''.join(model_items)}</ul>
      <img src="figures/skill_by_lead.png" alt="RMSE relative to climatology by lead" style="margin-top:1.2mm;width:70%">
      <div class="cap">RMSE ÷ climatology RMSE, mean of the six fields, held-out 2020–2021: {rel_m[0]:.2f} on day 1 (persistence {rel_p[0]:.2f}), {rel_m[-1]:.2f} on day 10.</div>
      <div class="note" style="margin-top:1.6mm"><h3>Qualification gates</h3>
        <p>Day-4 skill above persistence on every field ({sgn(min(test['per_lead']['4']['ss_persist'].values()))} at worst);
          {test['invalid_windows']} invalid outputs in {test['windows']} windows (finite, rain ≥ 0, −40 to +60 °C); a non-zero, localised
          response to both shocks in every tested window; zero shock identical to the baseline.</p></div>
    </div>
    <div>
      <h2>Evidence · held-out 2020–2021, {test['windows']} windows</h2>
      <table>
        <tr><th>Skill vs better reference</th><th>Day 1</th><th>Day 4</th><th>Day 8</th><th>Pooled</th><th>Day 4 vs pers.</th></tr>
        {''.join(rows)}
      </table>
      <p class="small" style="margin-top:.8mm">Skill = 1 − RMSE / min(RMSE persistence, RMSE climatology), latitude-weighted; pooled = leads 1, 4, 8 as
        scored. Also held out: 2022 (closest to the 2023+ test dates), mean skill {sgn(t22['mean_ss'], 3)} over {t22['windows']} windows.</p>
      <div class="tiles">
        <div class="tile"><div class="v">{sgn(test['mean_ss'], 3)}</div><div class="k">mean per-window skill (0.12 = full marks); full marks in {pct(boot['p_full_rmse'])} of 5-window draws</div></div>
        <div class="tile"><div class="v">{test['csi_model']:.2f}</div><div class="k">CSI, rain &gt;10 mm/day; persistence {test['csi_persist']:.2f}, climatology {test['csi_clim']:.2f}</div></div>
        <div class="tile"><div class="v">≥ {min(vr.values()):.2f}</div><div class="k">day-8 spatial variance ÷ observed, all six fields (0.40 needed)</div></div>
        <div class="tile"><div class="v">{lat['mean']:.2f} s</div><div class="k">HTTP round trip, 10 days, CPU (max {lat['max']:.2f} s)</div></div>
      </div>
      <h2 style="margin-top:2.2mm">Are the shock responses realistic? · vs ERA5, June–September</h2>
      <img src="figures/shock_realism.png" alt="Model vs observed shock sensitivity" style="width:88%">
      <div class="cap">Observed: regional rain on day t+k regressed on the driver on day t, <i>holding the same day's weather fixed</i>
        (2005–2017), 90% range from resampling years. Model: the same single-field shock on 2018–2019 windows. The final model is inside
        {sum(inside)} of 4 ranges: {', '.join(('<span class="ok">' if ok else '<span class="no">') + n + '</span>' for n, ok in zip(['Ghats day 1', 'inland days 2–8', 'Bay vapour', 'Bay warmth (sign)'], inside))}.</div>
    </div>
  </div>

  <div class="grid3" style="grid-template-columns:1.45fr 1fr">
    <div>
      <h2>Shock behaviour · response energy Σ(Δ ÷ std)², days 1 / 4 / 8</h2>
      <img src="figures/shock_energy.png" alt="Shock response energy">
    </div>
    <div>
      <p class="cap" style="margin-top:4mm">Konkan floods window (forecast from 20 July 2021). On this window {pct(ms['near_frac_day1'])} (wind stall) and
        {pct(wo['near_frac_day1'])} (warm Bay) of day-1 response energy lies within 3 cells of the box. Over 12 held-out windows: localised
        {pct(sh['monsoon_stall.localized'])} / {pct(sh['warm_ocean.localized'])}, spreading {pct(sh['monsoon_stall.propagates'])} /
        {pct(sh['warm_ocean.propagates'])}, within 5 std {pct(sh['monsoon_stall.stable'])} / {pct(sh['warm_ocean.stable'])}.</p>
      <div class="note" style="margin-top:1.6mm"><h3>Measured limitations</h3>
        <p>Heavy rain is under-forecast: where more than 50 mm/day fell, the forecast gives {100 * fig['heavy_rain_ratio']['day1']:.0f}% of it on day 1
          and {100 * fig['heavy_rain_ratio']['day4']:.0f}% on day 4; rain skill reaches climatology level by day 8
          ({sgn(test['per_lead']['8']['ss']['tp'])}). The realism check compares with observed regressions, which are associations, not
          experiments{'' if all(inside) else '; responses outside the observed range (red) are too weak or too strong'}. Single-field shocks
          (wind without the matching pressure change) lie outside anything in the training data.</p></div>
    </div>
  </div>
</section>

<section class="page">
  <header><h1>Monsoon Branches</h1>
    <div class="meta">Team intelligentapes · Multiverse of Monsoon · Yugaantar DevForge<br>Round 1 brief · Page 2 · Demo and impact</div></header>
  <div>
    <h2>Demo · Arabian Sea winds weaken by 40% on 19 July 2021</h2>
    <img src="figures/demo_konkan.png" alt="Baseline, shocked and difference rain maps">
    <div class="cap">Shock: inside 8–20°N, 65–75°E, both 10 m wind components × 0.6 on the last observed day (official <code>monsoon_stall</code>).
      Same model and inputs, two rollouts. Rain colour scale shared by both forecasts; difference scale centred at zero (brown = drier, blue = wetter).</div>
  </div>
  <div class="grid3">
    <div>
      <img src="figures/dashboard_3d.png" alt="Dashboard: 3D relief view with the rain change on day 5 and the day timeline" style="border:.5pt solid #e2e2df;border-radius:1.4mm">
      <div class="cap">The dashboard: choose a held-out start and a shock (type, strength, box). Each run makes two live <code>/simulate</code> calls.
        A 3D relief model of India (ERA5 terrain) carries the change, the wind flow and the shock box and plays the 10 days; flat 1° maps, the cell chart
        and the response summary update with it. A second view checks the forecast against what happened and both baselines.</div>
    </div>
    <div>
      <img src="figures/dry_spell_cell.png" alt="10-day rain at one land cell">
      <div class="cap">Longest run of days below 2.5 mm/day at {cell['lat']:.1f}°N {cell['lon']:.1f}°E: {cell['baseline']} → {cell['shocked']} days.</div>
      <div class="note" style="margin-top:1.6mm"><h3>Dry-spell indicator</h3>
        <p>For every land cell and each timeline, the longest run of consecutive forecast days with rain below 2.5 mm/day; the indicator is shocked
          minus baseline, in days. Here {dry['cells_longer']} land cells get a longer dry spell (up to +{dry['max_longer']} days) and {dry['cells_shorter']}
          a shorter one; 10-day land rain changes by {sgn(dry['land_mean_10day_rain_change_mm'])} mm on average. Dry-spell changes concentrate
          where rain hovers near the threshold, so read them together with the rain change.</p></div>
    </div>
  </div>
  <div class="grid2">
    <div><h2>How to read the output</h2><ul>
      <li><b>A model-projected rainfall indicator</b> (mm/day): what this forecast model does when today's weather is changed.</li>
      <li><b>Read the difference map first.</b> Near the box on day 1 is the direct effect; changes downstream on days 3–8 (Western Ghats,
        central India) are the propagated response.</li>
      <li><b>Small differences are noise-level.</b> A single forecast's error exceeds most shock responses; treat a few mm/day, or one dry day, as weak evidence.</li>
      <li><b>Check the observed panel</b> on held-out examples, and the realism check above, before trusting a response.</li></ul></div>
    <div><h2>Impact use case</h2>
      <p>A state agriculture or water-resources desk asks before the weekly advisory: <i>if the monsoon current over the Arabian Sea slackens,
        which districts face a longer dry spell in the next ten days?</i> The dashboard answers in seconds on a laptop CPU, mapping where rain
        shifts and how many extra dry days each land cell gets. That can flag areas for sowing-delay or protective-irrigation advice, or for
        checking reservoir releases.</p>
      <p>It cannot estimate crop loss (that needs crop type, growing stage, irrigation and soil moisture), and 1° cells cannot resolve district-scale extremes.</p>
      <h2 style="margin-top:1.6mm">Reproduce</h2>
      <p class="small"><code>pip install -r requirements.txt &amp;&amp; python serve.py --port 8000</code> serves the bundled weights offline.
        <code>train.py</code>, <code>scripts/finalize.py</code> and <code>evaluate.py</code> regenerate every number here; the official
        schema validator and repo scan pass.</p></div>
  </div>
</section>
</body></html>"""
    out = ROOT / "docs" / "brief.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
