"""Train the six-field residual CNN: one-step training, then rollout fine-tuning (2, 4, 8 days).

  python train.py --data ../data/india_climate_train.nc --out artifacts

Normalisers, the climatology input and the loss weights are fitted on the training years (2005-2017) only.
Each epoch is scored on the tuning years (2018-2019) with full 10-day rollouts and the official scoring
functions; the best checkpoint by estimated forecast points is written to --out as plain arrays.
The loss is a latitude-weighted MSE per channel, divided by the error of the better reference
(persistence or climatology) pooled over leads 1, 4 and 8, which is how the scorer normalises skill.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from monsoon_branches.data import calendar_slot, fit_stats, load_archive, reference_climatology, window_starts
from monsoon_branches.grid import CHANNELS, NC, lat_weights
from monsoon_branches.model import Forecaster
from monsoon_branches import scoreboard as SB

ROOT = Path(__file__).parent


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def reference_mse(arch, stats, split="train", leads=(1, 4, 8)):
    """Per-channel lat-weighted MSE of the better reference, pooled over the scored leads, on training windows."""
    idx = window_starts(arch, split, max(leads), stride=2)
    w = lat_weights()[:, None]
    clim = stats["clim"]
    mp, mc = np.zeros(NC), np.zeros(NC)
    for k in leads:
        t = arch.data[idx + k]
        p = arch.data[idx]
        c = clim[(arch.doy[idx + k].astype(int)) % 365]
        mp += (((p - t) ** 2) * w).mean(axis=(0, 2, 3))
        mc += (((c - t) ** 2) * w).mean(axis=(0, 2, 3))
    return np.minimum(mp, mc) / len(leads)


def parse_stages(s):
    out = []
    for part in s.split(","):
        k, e = part.split("x")
        out.append((int(k), int(e)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=[str(ROOT.parent / "data" / "india_climate_train.nc")],
                    help="the kit archive, optionally followed by extra ERA5 years from scripts/fetch_era5_wb2.py")
    ap.add_argument("--out", default=str(ROOT / "artifacts"))
    ap.add_argument("--fit-split", default="train", help="years used for fitting (train, or all for a final refit)")
    ap.add_argument("--val-split", default="tune")
    ap.add_argument("--stages", default="1x30,2x8,4x8,8x8", help="rollout-length x epochs, comma separated")
    ap.add_argument("--base", default="ar1", choices=["ar1", "persistence"], help="what the CNN corrects")
    ap.add_argument("--phys", action=argparse.BooleanOptionalAction, default=True, help="terrain and physics diagnostics as inputs")
    ap.add_argument("--rain-diag", action=argparse.BooleanOptionalAction, default=True,
                    help="diagnose rain from the physics of the predicted day")
    ap.add_argument("--rain-aux", type=float, default=0.0,
                    help="weight of the auxiliary loss: rain diagnosed from the TRUE next day's physics (needs --rain-diag)")
    ap.add_argument("--sens", type=float, default=0.0,
                    help="weight of the observed-sensitivity constraint on shock responses (0 = off)")
    ap.add_argument("--sens-every", type=int, default=4, help="apply it every N batches, from the 2-day stage on")
    ap.add_argument("--sens-batch", type=int, default=16)
    ap.add_argument("--sens-band", type=float, default=1.0,
                    help="accepted band around the observed point estimate, as a fraction of the 90%% range (1 = whole range)")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ft-lr", type=float, default=3e-4, help="peak learning rate of the rollout stages")
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--tp-weight", type=float, default=1.0, help="extra weight on the rain loss")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default=None,
                    help="fine-tune a saved single-member run (its normalisers, AR(1) memory and weights); every stage "
                         "then uses --ft-lr, and the architecture flags are taken from the run")
    ap.add_argument("--log", default=None)
    a = ap.parse_args(argv)

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = pick_device(a.device)
    arch = load_archive(a.data)
    stats = fit_stats(arch, a.fit_split)
    ref = reference_mse(arch, stats, a.fit_split)
    cw = 1.0 / ref
    cw[1] *= a.tp_weight
    cw = cw / cw.mean()
    print(f"device {dev}; channel loss weights " + " ".join(f"{c}={x:.2f}" for c, x in zip(CHANNELS, cw)))

    init = None
    if a.init:
        init = Forecaster.load(a.init)
        assert len(init.nets) == 1, "--init expects a single training run, not an ensemble"
        a.hidden, a.blocks, a.base = init.config["hidden"], init.config["blocks"], init.config["base"]
        a.phys, a.rain_diag = init.config["phys"], init.config["rain_diag"]
        init.rain_gain, init.mos = None, None
        print(f"fine-tuning {a.init} (base {a.base}, phys {a.phys}, rain-diag {a.rain_diag})")
    if a.phys and init is None:                         # normalisers for the physics diagnostics, from training days
        probe = Forecaster(stats, a.hidden, a.blocks, base=a.base, phys=True)
        sel = window_starts(arch, a.fit_split, 0, stride=7)
        with torch.no_grad():
            raw = probe.phys_raw(torch.from_numpy(arch.data[sel]))
        stats["phys_mean"] = raw.mean(dim=(0, 2, 3)).numpy()
        stats["phys_std"] = raw.std(dim=(0, 2, 3)).numpy()
        stats["slope_std"] = float(torch.stack([probe.hx, probe.hy]).std())
        print("physics diagnostics (mean, std): " + ", ".join(
            f"{n} {m:.3g}/{s:.3g}" for n, m, s in zip(["w_oro", "lift", "qx", "qy", "mfc", "div", "vort", "lap_p", "speed", "crh"],
                                                      stats["phys_mean"], stats["phys_std"])))
    if init is not None:
        model = init.train().to(dev)
    else:
        model = Forecaster(stats, a.hidden, a.blocks, base=a.base, phys=a.phys, rain_diag=a.rain_diag and a.phys).to(dev)
    n_par = sum(p.numel() for p in model.net.parameters())
    print(f"parameters: {n_par:,}")

    D = torch.from_numpy(arch.data).to(dev)
    DOY = torch.from_numpy(arch.doy).to(dev)
    latw = torch.as_tensor(lat_weights(), dtype=torch.float32, device=dev).view(1, 1, 1, -1, 1)
    cwt = torch.as_tensor(cw, dtype=torch.float32, device=dev).view(1, 1, NC, 1, 1)

    # validation set: full 10-day rollouts on the tuning years
    vidx = window_starts(arch, a.val_split, 10, stride=3)
    v_inp = np.stack([arch.data[i - 2:i + 1] for i in vidx])
    v_tru = np.stack([arch.data[i + 1:i + 11] for i in vidx])
    refclim = reference_climatology(arch, "train")
    v_clim = refclim[np.array([[calendar_slot(arch.dates[i + k]) for k in range(1, 11)] for i in vidx])]
    v_per = np.repeat(v_inp[:, -1:], 10, axis=1)
    v_inp_t = torch.from_numpy(v_inp).to(dev)
    v_doy_t = torch.from_numpy(arch.doy[vidx]).to(dev)

    def validate():
        model.eval()
        with torch.no_grad():
            pred = torch.cat([model.rollout(v_inp_t[s:s + 128], v_doy_t[s:s + 128], 10) for s in range(0, len(vidx), 128)]).cpu().numpy()
        model.train()
        t = SB.forecast_table(pred, v_tru, v_per, v_clim)
        # selection score: the forecast part of the scorer WITHOUT its caps, so selection keeps rewarding real skill
        t["score"] = 30 * t["mean_ss"] / 0.12 + 15 * t["mean_csi_skill"] / 0.10 + t["anti_blur_pts"]
        return t

    if a.sens > 0:
        from monsoon_branches import sensitivity as SENS
        R = SENS.regions(arch.static["land_sea_mask"], arch.static["orography_m"])
        sens_targets, sens_point = SENS.training_targets(arch, stats["clim"], R)
        if a.sens_band < 1.0:                           # aim at the observed central estimate, not just inside the range
            sens_targets = [(d, r, k, lo if d in SENS.SIGN_ONLY else sens_point[d][r][k] - a.sens_band * (sens_point[d][r][k] - lo),
                             sens_point[d][r][k] + a.sens_band * (hi - sens_point[d][r][k])) for d, r, k, lo, hi in sens_targets]
        sens_masks = SENS.torch_masks(R, dev)
        jidx = window_starts(arch, a.fit_split, 8)
        jidx = jidx[SENS.jjas_mask(arch.dates[jidx])]
        print("sensitivity constraint (driver, region, lead, observed 90% range): " +
              "; ".join(f"{d}/{r}/d{k} [{lo:+.2f},{hi:+.2f}]" for d, r, k, lo, hi in sens_targets))
    best, best_state, log = -1e9, None, []
    it = 0
    off_in = torch.arange(-2, 1, device=dev)
    t_start = time.time()
    for stage, (K, epochs) in enumerate(parse_stages(a.stages)):
        idx_all = window_starts(arch, a.fit_split, K)
        lr = a.lr if stage == 0 and init is None else a.ft_lr
        opt = torch.optim.AdamW(model.net.parameters(), lr=lr, weight_decay=a.wd)
        steps_per_epoch = math.ceil(len(idx_all) / a.batch)
        total = epochs * steps_per_epoch
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total, pct_start=0.1 if stage == 0 and init is None else 0.05)
        off_out = torch.arange(1, K + 1, device=dev)
        for ep in range(epochs):
            perm = torch.from_numpy(np.random.permutation(idx_all)).to(dev)
            run = 0.0
            for s in range(0, len(perm), a.batch):
                ii = perm[s:s + a.batch]
                hist = D[ii[:, None] + off_in]
                tgt = D[ii[:, None] + off_out]
                pred = model.rollout(hist, DOY[ii], K)
                loss = (((pred - tgt) ** 2) * latw * cwt).mean() * NC / cwt.sum()
                if a.rain_aux > 0:
                    r_tf = model.rain_teacher(hist, DOY[ii] + 1, tgt[:, 0])
                    aux = (((r_tf - tgt[:, 0, 1:2]) ** 2) * latw[:, 0]).mean() * cwt[0, 0, 1, 0, 0] / cwt.sum() * NC
                    loss = loss + a.rain_aux * aux
                it += 1
                if a.sens > 0 and K >= 2 and it % a.sens_every == 0:
                    jj = torch.from_numpy(np.random.choice(jidx, a.sens_batch, replace=False)).to(dev)
                    l_s, sens_resp = SENS.sensitivity_loss(model, D[jj[:, None] + off_in], DOY[jj], sens_targets, sens_masks)
                    loss = loss + a.sens * l_s
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.net.parameters(), 1.0)
                opt.step(); sched.step()
                run += loss.item() * len(ii)
            t = validate()
            if a.sens > 0 and K >= 2:
                print("   last sensitivity batch: " + " ".join(f"{d[0]}/{d[1][:7]}/d{d[2]} {v:+.2f}" for d, v in sens_resp.items()), flush=True)
            row = {"stage": stage, "K": K, "epoch": ep, "loss": run / len(perm), "val_mean_ss": t["mean_ss"],
                   "val_csi_skill": t["mean_csi_skill"], "val_score": t["score"], "pooled_ss": [float(x) for x in t["pooled_ss"]],
                   "var8": [float(x) for x in t["var_ratio_day8"]], "minutes": (time.time() - t_start) / 60}
            log.append(row)
            flag = ""
            if t["score"] > best:
                best, best_state, flag = t["score"], copy.deepcopy(model.state_dict()), " *"
            print(f"[K={K} ep {ep:2d}] loss {row['loss']:.4f}  val ss {t['mean_ss']:+.3f} csi-skill {t['mean_csi_skill']:+.3f} "
                  f"var8-tp {t['var_ratio_day8'][1]:.2f}  pts {t['rmse_pts']:.1f}+{t['csi_pts']:.1f}+{t['anti_blur_pts']:.1f}"
                  f" | uncapped {t['score']:.1f}{flag}  ({row['minutes']:.1f} min)", flush=True)

    model.load_state_dict(best_state)
    model = model.cpu()
    meta = {"fit_split": a.fit_split, "val_split": a.val_split, "stages": a.stages, "lr": a.lr, "ft_lr": a.ft_lr,
            "data": [Path(p).name for p in a.data],
            "tp_weight": a.tp_weight, "seed": a.seed, "val_score": best, "params": n_par, "rain_aux": a.rain_aux,
            "sens": a.sens, "sens_band": a.sens_band, "init": Path(a.init).name if a.init else None}
    model.save(Path(a.out), meta)
    print(f"saved best checkpoint (val score {best:.2f}) to {a.out}")
    if a.log:
        Path(a.log).parent.mkdir(parents=True, exist_ok=True)
        Path(a.log).write_text(json.dumps({"meta": meta, "log": log}, indent=1))


if __name__ == "__main__":
    main()
