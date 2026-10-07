#!/usr/bin/env python3
"""Per-pair EPE of trained checkpoints: are there pairs where the prediction explodes?

A validation mean can jump while Fl stays the same when a few pairs get huge errors
(C3-chairs, step 25 000, Sintel final). This script evaluates every pair with the
validation protocol (train.predict_full) and reports, per model and dataset, the
distribution of per-pair EPE, the worst pairs, and the largest predicted flow
relative to the largest true flow of the pair.

    .venv/bin/python bench/outliers.py --runs B0=B0,C3=C3_1d_ps_nopos_relu
    (weights: runs/<name>-chairs/chairs_ema_best.pth)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import data  # noqa: E402
import train  # noqa: E402
from model import EdgeFlowNet, ModelConfig  # noqa: E402

RESULTS = ROOT / "results" / "outliers.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="B0=B0,C3=C3_1d_ps_nopos_relu,V1d=V1d_global1d_nosa,V1=V1_global1d_sa")
    ap.add_argument("--stage", default="chairs")
    ap.add_argument("--datasets", default="sintel-clean,sintel-final,kitti15")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    variants = yaml.safe_load((ROOT / "bench" / "variants.yaml").read_text())
    report = {}
    for item in args.runs.split(","):
        name, variant = item.split("=")
        weights = ROOT / "runs" / f"{name}-{args.stage}" / f"{args.stage}_ema_best.pth"
        model = EdgeFlowNet(ModelConfig(**(variants[variant] or {})))
        model.load_state_dict(torch.load(weights, map_location="cpu"))
        model.eval()
        report[name] = {}
        for ds_name in args.datasets.split(","):
            ds = data.make_dataset(ds_name, "all", (256, 320), augment=False)
            rows = []
            with torch.no_grad():
                for i in range(len(ds)):
                    img0, img1, gt, _, valid = ds[i]
                    flow = train.predict_full(model, img0, img1, (256, 320))
                    valid = torch.as_tensor(valid).bool()
                    err = torch.sqrt(((flow - gt) ** 2).sum(0))[valid]
                    gmax = float(torch.sqrt((gt ** 2).sum(0))[valid].max())
                    pmax = float(torch.sqrt((flow ** 2).sum(0)).max())
                    rows.append((float(err.mean()), train.pair_name(ds, i), pmax, gmax, int(valid.sum())))
            epe = np.array([r[0] for r in rows])
            w = np.array([r[4] for r in rows], float)
            order = np.argsort(-epe)
            top_share = float((epe[order[:10]] * w[order[:10]]).sum() / (epe * w).sum())
            report[name][ds_name] = {
                "pairs": len(rows), "epe_pixel_mean": float((epe * w).sum() / w.sum()),
                "pair_epe_median": float(np.median(epe)), "pair_epe_p99": float(np.percentile(epe, 99)),
                "pair_epe_max": float(epe.max()), "share_of_error_in_worst_10_pairs": top_share,
                "pred_over_true_max_flow_max": float(max(r[2] / max(r[3], 1.0) for r in rows)),
                "worst": [{"pair": rows[j][1], "epe": rows[j][0], "max_pred": rows[j][2], "max_true": rows[j][3]}
                          for j in order[:args.top]]}
            r = report[name][ds_name]
            print(f"{name:4s} {ds_name:13s} mean {r['epe_pixel_mean']:6.2f} | pair EPE median {r['pair_epe_median']:5.2f} "
                  f"p99 {r['pair_epe_p99']:6.2f} max {r['pair_epe_max']:6.2f} | worst 10 pairs = "
                  f"{100 * top_share:4.1f}% of error | max pred/true flow {r['pred_over_true_max_flow_max']:.2f}", flush=True)
            for wst in r["worst"][:3]:
                print(f"       {wst['pair']:32s} EPE {wst['epe']:7.2f}  max pred {wst['max_pred']:6.0f}  max true {wst['max_true']:6.0f}")
    RESULTS.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
