#!/usr/bin/env python3
"""What does each stage of the network actually do? FP32, CPU, no training.

Runs a trained checkpoint over the evaluation pairs and reports:
  1. EPE after every stage (s32, s16, s8, s4, full), overall and by motion size
  2. knockouts: zero one input group of a block, or drop one stage's increment,
     and see how much the final EPE changes
  3. learned blending weights (alpha) of every stage
  4. matching quality of the cost volumes alone (argmax vs ground truth)
  5. response to a pure translation of known size

    .venv/bin/python bench/stage_diag.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))

import eqexp  # noqa: E402

RESULTS = ROOT / "results" / "stage_diag.json"
STAGES = ("s32", "s16", "s8", "s4", "full")
SCALE = {"s32": 32, "s16": 16, "s8": 8, "s4": 4, "full": 1}
BINS = ((0, 2), (2, 10), (10, 40), (40, 1e9))


@torch.no_grad()
def stage_flows(model, img0, img1):
    """All five stage outputs, each upsampled to full resolution in full-res pixels."""
    model.training = True          # only switches the return value; BatchNorm layers stay in eval
    try:
        outs, _ = model(img0, img1)
    finally:
        model.training = False
    size = img0.shape[-2:]
    full = {}
    for name, flow in zip(STAGES, outs):
        if name != "full":
            flow = F.interpolate(flow, size=size, mode="bilinear", align_corners=False) * SCALE[name]
        full[name] = flow
    return full


def batches(data, bs=8):
    for i in range(0, len(data["img0"]), bs):
        yield (torch.from_numpy(data["img0"][i:i + bs]).float() / 255.0,
               torch.from_numpy(data["img1"][i:i + bs]).float() / 255.0,
               torch.from_numpy(data["flow"][i:i + bs]),
               torch.from_numpy(data["valid_noc"][i:i + bs]))


def epe_stats(pred, gt, valid):
    err = torch.sqrt(((pred - gt) ** 2).sum(1))
    mag = torch.sqrt((gt ** 2).sum(1))
    out = {"all": (float(err[valid].sum()), int(valid.sum()))}
    for lo, hi in BINS:
        m = valid & (mag >= lo) & (mag < hi)
        out[f"{lo}-{hi if hi < 1e8 else 'inf'}"] = (float(err[m].sum()), int(m.sum()))
    return out


def merge(total, part):
    for k, (s, n) in part.items():
        a = total.setdefault(k, [0.0, 0])
        a[0] += s
        a[1] += n


def finish(total):
    return {k: (s / n if n else float("nan")) for k, (s, n) in total.items()}


@torch.no_grad()
def final_epe(model, data):
    total = {}
    for img0, img1, gt, valid in batches(data):
        merge(total, epe_stats(model(img0, img1), gt, valid))
    return finish(total)


def knock_group(model, name, index):
    """Context: zero one input group of the concat `name`."""
    orig = model._cat

    def patched(cat_name, groups):
        if cat_name == name:
            groups = [torch.zeros_like(g) if i == index else g for i, g in enumerate(groups)]
        return orig(cat_name, groups)

    model._cat = patched
    return lambda: model.__dict__.pop("_cat", None)


def cost_volume_accuracy(model, data):
    """Does the argmax of the local s16 cost volume point at the true displacement?"""
    captured = {}
    hook = model.corr16.register_forward_hook(lambda m, i, o: captured.__setitem__("cv", o))
    shifts = torch.tensor(model.corr16.shifts, dtype=torch.float32)       # (dy, dx) in s16 cells
    r = model.corr16.r
    hit = within = total = 0
    with torch.no_grad():
        for img0, img1, gt, valid in batches(data):
            model(img0, img1)
            cv = captured["cv"]
            best = shifts[cv.argmax(1)]                                    # [B,H,W,2] (dy,dx)
            gt16 = F.avg_pool2d(gt, 16) / 16.0                             # (u,v) in s16 cells
            v16 = F.avg_pool2d(valid.float().unsqueeze(1), 16).squeeze(1) > 0.99
            in_range = v16 & (gt16.abs().amax(1) <= r)
            err = torch.maximum((best[..., 1] - gt16[:, 0]).abs(), (best[..., 0] - gt16[:, 1]).abs())
            hit += int((err[in_range] <= 0.5).sum())
            within += int(in_range.sum())
            total += int(v16.sum())
    hook.remove()
    return {"argmax_correct_within_range": hit / max(within, 1),
            "pixels_within_search_range": within / max(total, 1), "radius_cells": r}


@torch.no_grad()
def translation_response(model, data, shifts=(0.25, 0.5, 1, 2, 4, 8, 16, 32, 48, 64, 96, 128)):
    """Shift a real frame horizontally by d pixels; the true flow is (d, 0) everywhere."""
    img = torch.from_numpy(data["img0"][::23][:8]).float() / 255.0         # a few different scenes
    b, _, h, w = img.shape
    xs = torch.linspace(-1, 1, w).view(1, 1, w).expand(b, h, w)
    ys = torch.linspace(-1, 1, h).view(1, h, 1).expand(b, h, w)
    rows = []
    for d in shifts:
        grid = torch.stack([xs - 2.0 * d / (w - 1), ys], -1)              # img1(x) = img0(x - d)
        img1 = F.grid_sample(img, grid, mode="bilinear", padding_mode="reflection", align_corners=True)
        flow = model(img, img1)
        margin = int(min(d + 8, w // 2 - 8))
        u = flow[:, 0, 16:-16, margin:w - 16]
        v = flow[:, 1, 16:-16, margin:w - 16]
        rows.append({"shift_px": d, "mean_u": float(u.mean()), "ratio": float(u.mean()) / d,
                     "epe": float(torch.sqrt((u - d) ** 2 + v ** 2).mean())})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="train_out_v1/Phase2_step30000_EMA_best.pth")
    ap.add_argument("--datasets", default="sintel-clean,sintel-final,kitti15")
    args = ap.parse_args()

    full = np.load(eqexp.EVALSET)
    keep = np.isin(full["dataset"], args.datasets.split(","))
    data = {k: full[k][keep] for k in full.files}
    model = eqexp.load_model(args.checkpoint)
    report = {"checkpoint": args.checkpoint, "pairs": int(keep.sum()), "datasets": args.datasets}

    # 1. EPE after every stage
    totals = {s: {} for s in STAGES}
    for img0, img1, gt, valid in batches(data):
        flows = stage_flows(model, img0, img1)
        for s in STAGES:
            merge(totals[s], epe_stats(flows[s], gt, valid))
    report["stage_epe"] = {s: finish(t) for s, t in totals.items()}
    cols = list(report["stage_epe"]["full"])
    print("1. EPE after each stage (flow upsampled to full resolution), by true motion size in px")
    print(f"{'stage':6s} " + " ".join(f"{c:>9s}" for c in cols))
    for s in STAGES:
        print(f"{s:6s} " + " ".join(f"{report['stage_epe'][s][c]:9.3f}" for c in cols))
    share = {c: totals["full"][c][1] / totals["full"]["all"][1] for c in cols}
    print(f"{'pixels':6s} " + " ".join(f"{100 * share[c]:8.1f}%" for c in cols))

    # 2. knockouts
    base = final_epe(model, data)
    knock = {}
    groups = {"update32": ("corr32", "flow", "ctx"), "update16": ("corr16", "corr32_up", "flow", "ctx"),
              "refine_s16": ("f0", "f1", "flow", "ctx"), "refine_s8": ("f0", "f1", "flow", "ctx"),
              "refine_s4": ("f0", "f1", "flow", "ctx"), "full": ("flow", "img0")}
    for cat, names in groups.items():
        for i, g in enumerate(names):
            restore = knock_group(model, cat, i)
            try:
                knock[f"{cat}: zero {g}"] = final_epe(model, data)
            finally:
                restore()
    for alpha in ("alpha_s16", "alpha_s8", "alpha_s4", "alpha_full"):
        p = getattr(model, alpha)
        saved = p.data.clone()
        p.data.zero_()
        knock[f"drop increment of {alpha.replace('alpha_', 'refine_')}"] = final_epe(model, data)
        p.data.copy_(saved)
    report["baseline"] = base
    report["knockouts"] = knock
    print(f"\n2. Knockouts: final EPE when one input or one stage increment is removed (baseline {base['all']:.3f})")
    print(f"{'what is removed':38s} " + " ".join(f"{c:>9s}" for c in cols) + "    delta")
    for name, r in sorted(knock.items(), key=lambda kv: -kv[1]["all"]):
        print(f"{name:38s} " + " ".join(f"{r[c]:9.3f}" for c in cols) + f"  {r['all'] - base['all']:+7.3f}")

    # 3. learned blending weights
    alphas = {n: float(p) for n, p in model.named_parameters() if "alpha" in n}
    report["alphas"] = alphas
    print("\n3. Learned blending weights:", json.dumps({k: round(v, 3) for k, v in alphas.items()}))

    # 4. cost volume matching
    report["cost_volume_s16"] = cost_volume_accuracy(model, data)
    print("\n4. Local cost volume at s16:", json.dumps({k: round(v, 3) for k, v in report["cost_volume_s16"].items()}))

    # 5. translation response
    report["translation"] = translation_response(model, data)
    print("\n5. Pure horizontal translation of a real frame by d pixels")
    print(f"{'d, px':>7s} {'predicted':>10s} {'ratio':>7s} {'EPE':>8s}")
    for r in report["translation"]:
        print(f"{r['shift_px']:7.2f} {r['mean_u']:10.2f} {r['ratio']:7.2f} {r['epe']:8.2f}")

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
