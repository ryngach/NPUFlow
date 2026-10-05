#!/usr/bin/env python3
"""Build a compact FlyingChairs copy for a small fixed-resolution model.

The original release is 22 872 pairs of 512x384 PPM frames and .flo files (32.5 GB,
68 616 small files). The model trains on 320x256 crops around the "fit" scale, at
which a 512x384 frame becomes 341x256, so this script stores frames and flow at
exactly that scale (2/3) in a few large files, in the same format as
kaggle/build_things.py:

    chairs_NN_flow.npy        float16 [n, H, W, 2], flow in pixels of the stored size
    chairs_NN_img.bin         JPEG bytes of both frames of every sample, back to back
    chairs_NN_index.npy       int64 [n, 2, 2]: (offset, length) of frame 0 and frame 1
    chairs_NN_ids.npy         int32 [n]: original sample number (00001 -> 1)
    meta.json                 including the distribution of motion at the stored scale

Samples are written in the sorted order of the original names, so the deterministic
train/val split in data.py selects the same pairs as with the original files.

Runs on Kaggle (CPU session) with the dataset craljimenez/flyingchairs attached;
the result is the notebook output.

    python kaggle/build_chairs.py --out /kaggle/working
    python kaggle/build_chairs.py --src DIR --out DIR --limit 50      # local test
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

SRC_H, SRC_W = 384, 512
OUT_H, OUT_W = 256, 341          # 2/3: the fit scale for 320x256 crops (height-limited)
SHARD = 2500
JPEG_QUALITY = 95
MARKER = "00001_img1.ppm"


def find_source() -> str:
    """Folder with the original *_img1.ppm files among the Kaggle inputs."""
    level = ["/kaggle/input"]
    for _ in range(8):
        nxt = []
        for d in level:
            if os.path.exists(os.path.join(d, MARKER)):
                return d
            names = sorted(os.listdir(d))
            if len(names) > 64:
                continue
            nxt += [os.path.join(d, n) for n in names if os.path.isdir(os.path.join(d, n))]
        level = nxt
    raise FileNotFoundError(f"{MARKER} not found under /kaggle/input")


def read_flo(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        buf = f.read()
    if np.frombuffer(buf, np.float32, 1)[0] != 202021.25:
        raise ValueError(f"bad .flo magic: {path}")
    w, h = np.frombuffer(buf, np.int32, 2, offset=4)
    return np.frombuffer(buf, np.float32, int(h) * int(w) * 2, offset=12).reshape(int(h), int(w), 2)


def convert(src: str, base: str):
    """One sample -> (JPEG bytes of both frames, float16 flow at the stored size)."""
    jpgs = []
    for k in (1, 2):
        img = cv2.imread(os.path.join(src, f"{base}_img{k}.ppm"), cv2.IMREAD_COLOR)
        if img is None or img.shape[:2] != (SRC_H, SRC_W):
            raise ValueError(f"unexpected frame {base}_img{k}.ppm: {None if img is None else img.shape}")
        small = cv2.resize(img, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
        jpgs.append(cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])[1].tobytes())
    flow = read_flo(os.path.join(src, f"{base}_flow.flo"))
    small = cv2.resize(flow, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
    small *= np.array([OUT_W / SRC_W, OUT_H / SRC_H], np.float32)
    return jpgs, small.astype(np.float16)


class Writer:
    def __init__(self, out_dir, prefix="chairs"):
        self.out, self.prefix = out_dir, prefix
        self.shard = 0
        self.total = 0
        self._reset()

    def _reset(self):
        self.flows, self.index, self.ids = [], [], []
        self.bin = open(os.path.join(self.out, f"{self.prefix}_{self.shard:02d}_img.bin"), "wb")
        self.pos = 0

    def add(self, sample_id, jpg0, jpg1, flow16):
        entry = []
        for jpg in (jpg0, jpg1):
            self.bin.write(jpg)
            entry.append((self.pos, len(jpg)))
            self.pos += len(jpg)
        self.index.append(entry)
        self.flows.append(flow16)
        self.ids.append(sample_id)
        self.total += 1
        if len(self.flows) == SHARD:
            self.flush()

    def flush(self, final=False):
        self.bin.close()
        base = os.path.join(self.out, f"{self.prefix}_{self.shard:02d}")
        if self.flows:
            np.save(base + "_flow.npy", np.stack(self.flows))
            np.save(base + "_index.npy", np.array(self.index, np.int64))
            np.save(base + "_ids.npy", np.array(self.ids, np.int32))
            self.shard += 1
        else:
            os.remove(base + "_img.bin")
        if not final:
            self._reset()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="", help="folder with the original files (found automatically on Kaggle)")
    ap.add_argument("--out", default="/kaggle/working")
    ap.add_argument("--threads", type=int, default=16, help="readers; the Kaggle inputs sit on a network disk")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many samples (testing)")
    args = ap.parse_args()

    src = args.src or find_source()
    bases = sorted(n[:-len("_img1.ppm")] for n in os.listdir(src) if n.endswith("_img1.ppm"))
    if args.limit:
        bases = bases[:args.limit]
    os.makedirs(args.out, exist_ok=True)
    print(f"source: {src} | {len(bases)} samples | out: {args.out}", flush=True)

    writer = Writer(args.out)
    hist_edges = np.array([0, 2, 5, 10, 20, 40, 60, 90, 1e9], np.float32)
    hist = np.zeros(len(hist_edges) - 1, np.int64)
    max_mag = 0.0
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        # map keeps the input order, so the shards follow the sorted names
        for base, (jpgs, flow16) in zip(bases, pool.map(lambda b: convert(src, b), bases)):
            writer.add(int(base), jpgs[0], jpgs[1], flow16)
            mag = np.sqrt((flow16.astype(np.float32) ** 2).sum(-1))
            hist += np.histogram(mag, bins=hist_edges)[0]
            max_mag = max(max_mag, float(mag.max()))
            if writer.total % 1000 == 0:
                el = time.time() - started
                print(f"[{el / 60:6.1f} min] {writer.total}/{len(bases)} samples, {writer.total / el:.1f} samples/s",
                      flush=True)
    writer.flush(final=True)

    share = hist / hist.sum()
    labels = [f"{int(a)}-{'inf' if b > 1e8 else int(b)}" for a, b in zip(hist_edges[:-1], hist_edges[1:])]
    meta = {"source": src, "dataset": "FlyingChairs", "samples": writer.total, "shards": writer.shard,
            "shard_size": SHARD, "stored_hw": [OUT_H, OUT_W], "original_hw": [SRC_H, SRC_W],
            "order": "sorted original names (same split as the original files in data.py)",
            "jpeg_quality": JPEG_QUALITY, "flow_dtype": "float16, pixels of the stored size",
            "image_order": "BGR as decoded by cv2.imdecode",
            "motion_px_at_stored_scale": {"share_of_pixels": dict(zip(labels, [round(float(x), 5) for x in share])),
                                          "max": round(max_mag, 1)},
            "hours": round((time.time() - started) / 3600, 2)}
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1))
    size = sum(os.path.getsize(os.path.join(args.out, n)) for n in os.listdir(args.out) if n.startswith("chairs_"))
    print(f"done: {writer.total} samples in {writer.shard} shards, {size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
