"""Local scoreboard built on the official scoring functions (monsoon_r1.scoring), aggregated the way the grader
describes: per window, skill pools leads 1/4/8 and averages the six channels; CSI pools the same leads.
Training/evaluation only; not imported by the server."""
from __future__ import annotations

import numpy as np
from monsoon_r1 import scoring as S

from .grid import CHANNELS, NC, TP, BOX_ARABIAN_SEA, BOX_BAY_OF_BENGAL

LEADS = (1, 4, 8)
LI = [k - 1 for k in LEADS]


def forecast_table(pred, true, persist, clim) -> dict:
    """All arrays [W, steps, 6, 32, 36] in physical units."""
    W = pred.shape[0]
    ss_w, csi_sk_w, csi_m, csi_p, csi_c, ratios, invalid = [], [], [], [], [], [], []
    for w in range(W):
        p, t, pe, cl = pred[w, LI], true[w, LI], persist[w, LI], clim[w, LI]
        ss_w.append(float(np.nanmean(S.skill_vs_best_ref(p, t, pe, cl)["ss"])))
        cm, cp, cc = S.csi(p[:, TP], t[:, TP]), S.csi(pe[:, TP], t[:, TP]), S.csi(cl[:, TP], t[:, TP])
        csi_m.append(cm); csi_p.append(cp); csi_c.append(cc)
        csi_sk_w.append(S.csi_skill(cm, cp, cc))
        ratios.append(S.variance_ratio(pred[w, 7], true[w, 7]))
        bad = S.validity(pred[w], pred.shape[1])
        if bad:
            invalid.append((w, bad))
    ratios = np.array(ratios)
    fp = S.forecast_points(ss_w, csi_sk_w)
    # diagnostics: skill per channel and lead, pooled over windows (each lead on its own)
    per = {}
    for k in LEADS:
        r = S.skill_vs_best_ref(pred[:, k - 1], true[:, k - 1], persist[:, k - 1], clim[:, k - 1])
        per[k] = {"ss": r["ss"], "ss_persist": r["ss_persist"], "ss_clim": r["ss_clim"], "rmse": r["rmse"]}
    pooled = S.skill_vs_best_ref(pred[:, LI], true[:, LI], persist[:, LI], clim[:, LI])
    return {
        "windows": W, "mean_ss": fp["mean_ss"], "ss_by_window": np.array(ss_w), "csi_skill_by_window": np.array(csi_sk_w),
        "csi_model": float(np.nanmean(csi_m)), "csi_persist": float(np.nanmean(csi_p)), "csi_clim": float(np.nanmean(csi_c)),
        "mean_csi_skill": fp["mean_csi_skill"], "rmse_pts": fp["rmse_pts"], "csi_pts": fp["csi_pts"],
        "var_ratio_day8": np.nanmean(ratios, axis=0), "anti_blur_pts": S.anti_blur_points(ratios),
        "per_lead": per, "pooled_ss": pooled["ss"], "invalid": invalid,
        "beats_persistence_day4": bool(np.all(per[4]["ss_persist"] > 0)),
        "day4_ss_persist_mean": float(np.mean(per[4]["ss_persist"])),
    }


def bootstrap_points(t: dict, n_windows: int = 5, draws: int = 4000, seed: int = 0) -> dict:
    """The grader scores only 5 windows: the distribution of forecast points over random 5-window draws."""
    rng = np.random.default_rng(seed)
    ss, cs = t["ss_by_window"], t["csi_skill_by_window"]
    pts = []
    for _ in range(draws):
        k = rng.choice(len(ss), n_windows, replace=False)
        f = S.forecast_points(list(ss[k]), list(cs[k]))
        pts.append((f["rmse_pts"], f["csi_pts"]))
    pts = np.array(pts)
    return {"rmse_pts_mean": float(pts[:, 0].mean()), "p_full_rmse": float(np.mean(pts[:, 0] >= 29.99)),
            "csi_pts_mean": float(pts[:, 1].mean()), "p_full_csi": float(np.mean(pts[:, 1] >= 14.99)),
            "forecast_pts_mean": float(pts.sum(1).mean()), "forecast_pts_p10": float(np.percentile(pts.sum(1), 10))}


SHOCKS = {"warm_ocean": BOX_BAY_OF_BENGAL, "monsoon_stall": BOX_ARABIAN_SEA}


def shock_table(base, outs: dict, std) -> dict:
    """base [W,steps,6,H,W]; outs[name] same shape for warm_ocean, monsoon_stall, null."""
    W = base.shape[0]
    rows, pts = [], []
    for w in range(W):
        res = {n: S.causality_checks(base[w], outs[n][w], box, std) for n, box in SHOCKS.items()}
        null_ok = S.null_consistent(base[w], outs["null"][w], std)
        pts.append(S.causality_points(list(res.values()), null_ok))
        rows.append({"null_ok": null_ok, **{n: r for n, r in res.items()}})
    summary = {"causality_pts_mean": float(np.mean(pts)), "causality_pts_min": float(np.min(pts)),
               "null_ok_all": all(r["null_ok"] for r in rows)}
    for n in SHOCKS:
        for key in ("nonzero", "localized", "propagates", "stable", "no_teleport"):
            summary[f"{n}.{key}"] = float(np.mean([r[n][key] for r in rows]))
        summary[f"{n}.near_frac_day1"] = float(np.mean([r[n]["near_frac_day1"] for r in rows]))
    return {"summary": summary, "rows": rows, "points": pts}


def print_forecast(name: str, t: dict) -> None:
    print(f"\n== {name}: {t['windows']} windows ==")
    print(f"mean skill (per-window, leads 1/4/8, 6 ch): {t['mean_ss']:+.3f}  -> RMSE pts {t['rmse_pts']:.1f}/30")
    print(f"heavy-rain CSI  model {t['csi_model']:.3f}  persist {t['csi_persist']:.3f}  clim {t['csi_clim']:.3f}"
          f"  skill {t['mean_csi_skill']:+.3f} -> {t['csi_pts']:.1f}/15")
    print("day-8 variance ratio " + " ".join(f"{c}={r:.2f}" for c, r in zip(CHANNELS, t["var_ratio_day8"]))
          + f" -> {t['anti_blur_pts']:.1f}/10")
    print(f"{'lead':>4} " + " ".join(f"{c:>17}" for c in CHANNELS))
    for k, r in t["per_lead"].items():
        print(f"{k:>4} " + " ".join(f"{a:+.2f} p{b:+.2f} c{c:+.2f}" for a, b, c in zip(r["ss"], r["ss_persist"], r["ss_clim"])))
    print("pooled skill by channel: " + " ".join(f"{c}={s:+.3f}" for c, s in zip(CHANNELS, t["pooled_ss"])))
    print(f"day-4 beats persistence on every channel: {t['beats_persistence_day4']}  (mean {t['day4_ss_persist_mean']:+.3f})"
          f"   invalid windows: {len(t['invalid'])}")
