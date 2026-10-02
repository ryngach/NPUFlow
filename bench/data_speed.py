#!/usr/bin/env python3
"""Where does the time go when loading training samples? CPU only.

    python bench/data_speed.py --dataset chairs

Prints the cost of each step for single samples, then the throughput of the
DataLoader for several worker counts.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import data  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="chairs")
    ap.add_argument("--workers", default="4,8,16")
    ap.add_argument("--batches", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    import cv2
    print("cores", os.cpu_count(), "| torch threads", torch.get_num_threads(), "| cv2 threads", cv2.getNumThreads())
    ds = data.make_dataset(args.dataset, "all" if args.dataset in ("chairs", "things") else "train", (256, 320), True)
    print(args.dataset, len(ds), "samples at", data.DATA.get(args.dataset.split("-")[0], "?"))

    idx = np.random.RandomState(0).permutation(len(ds))
    t_load = t_aug = 0.0
    n = 40
    for i in idx[:n]:
        t = time.time()
        img0, img1, flow, vnoc, vall = ds.load(int(i))
        t_load += time.time() - t
        t = time.time()
        ds.augmentor(np.ascontiguousarray(img0[..., :3]), np.ascontiguousarray(img1[..., :3]),
                     np.ascontiguousarray(flow.transpose(1, 2, 0)), vall)
        t_aug += time.time() - t
    print(f"single process, cold files: load {1000 * t_load / n:.0f} ms, augment {1000 * t_aug / n:.0f} ms per sample")
    t = time.time()
    for i in idx[:n]:
        ds.load(int(i))
    print(f"same files again (cached): load {1000 * (time.time() - t) / n:.0f} ms per sample")

    pos = n
    for workers in [int(w) for w in args.workers.split(",")]:
        sub = torch.utils.data.Subset(ds, idx[pos:pos + (args.batches + 2) * args.batch_size].tolist())
        pos += len(sub)
        loader = DataLoader(sub, batch_size=args.batch_size, num_workers=workers, drop_last=True)
        t = None
        count = 0
        for b, batch in enumerate(loader):
            if b == 1:
                t = time.time()          # skip worker start-up and the first batch
            elif b > 1:
                count += len(batch[0])
        dt = time.time() - t
        print(f"DataLoader workers {workers:2d}: {count / dt:6.1f} samples/s", flush=True)


if __name__ == "__main__":
    main()
