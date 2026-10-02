#!/usr/bin/env python3
"""Build a compact FlyingThings3D (subset) training set for a small fixed-resolution model.

The official optical-flow archive is 80 GB of bzip2 (about 390 GB of .flo files for
both cameras and both time directions); only the left camera, forward direction is
needed, and only at the working scale of the model. This script streams the archive,
keeps that quarter, scales frames and flow to height 256, and writes a few large files:

    things_NN_flow.npy        float16 [n, H, W, 2], flow in pixels of the stored size
    things_NN_img.bin         JPEG bytes of both frames of every sample, back to back
    things_NN_index.npy       int64 [n, 2, 2]: (offset, length) of frame 0 and frame 1
    things_NN_ids.npy         int32 [n]: original frame number of frame 0
    meta.json

Runs on Kaggle (CPU session, internet on) with the image dataset
arjun12367/sceneflow-flyingthings-images attached; the result is the notebook output.

    python kaggle/build_things.py --out /kaggle/working
    python kaggle/build_things.py --flow-archive some.tar.bz2 --images DIR --out DIR   # local test
"""
import argparse
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time

import cv2
import numpy as np

FLOW_URL = "https://lmb.informatik.uni-freiburg.de/data/FlyingThings3D_subset/FlyingThings3D_subset_flow.tar.bz2"
WANTED = re.compile(r"train/flow/left/into_future/(\d+)\.(flo|pfm)$")
OUT_H, OUT_W = 256, 456          # 540x960 scaled so that the height equals the model height
SHARD = 2000
JPEG_QUALITY = 95


def parse_flo(buf: bytes) -> np.ndarray:
    magic = np.frombuffer(buf, np.float32, 1)[0]
    if magic != 202021.25:
        raise ValueError("bad .flo magic")
    w, h = np.frombuffer(buf, np.int32, 2, offset=4)
    return np.frombuffer(buf, np.float32, int(h) * int(w) * 2, offset=12).reshape(int(h), int(w), 2)


def parse_pfm(buf: bytes) -> np.ndarray:
    f = io.BytesIO(buf)
    header = f.readline().rstrip()
    w, h = map(int, f.readline().split())
    scale = float(f.readline().rstrip())
    channels = 3 if header == b"PF" else 1
    data = np.frombuffer(f.read(), "<f4" if scale < 0 else ">f4").reshape(h, w, channels)
    return np.ascontiguousarray(data[::-1, :, :2])      # PFM rows are bottom-up


def find_images() -> str:
    """Folder with the left clean frames of the training split among the Kaggle inputs."""
    level = ["/kaggle/input"]
    for _ in range(8):
        nxt = []
        for d in level:
            names = sorted(os.listdir(d))
            if "0000000.png" in names and d.rstrip("/").endswith("train/image_clean/left"):
                return d
            if len(names) > 64:
                continue
            nxt += [os.path.join(d, n) for n in names if os.path.isdir(os.path.join(d, n))]
        level = nxt
    raise FileNotFoundError("train/image_clean/left with 0000000.png not found under /kaggle/input")


def open_stream(source: str):
    """Decompressed tar stream of the archive, from a URL or a local file."""
    unpack = next((t for t in ("lbzip2", "pbzip2", "bzip2") if shutil.which(t)), None)
    if unpack is None:
        raise RuntimeError("no bzip2 tool found")
    if source.startswith("http"):
        cmd = f"curl -sSL --retry 5 --retry-delay 10 '{source}' | {unpack} -dc"
    else:
        cmd = f"{unpack} -dc '{source}'"
    print("stream:", cmd, flush=True)
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, bufsize=1 << 24)
    return proc, tarfile.open(fileobj=proc.stdout, mode="r|")


class Writer:
    def __init__(self, out_dir):
        self.out = out_dir
        self.shard = 0
        self.total = 0
        self._reset()

    def _reset(self):
        self.flows, self.index, self.ids = [], [], []
        self.bin = open(os.path.join(self.out, f"things_{self.shard:02d}_img.bin"), "wb")
        self.pos = 0

    def add(self, frame_id, jpg0, jpg1, flow16):
        entry = []
        for jpg in (jpg0, jpg1):
            self.bin.write(jpg)
            entry.append((self.pos, len(jpg)))
            self.pos += len(jpg)
        self.index.append(entry)
        self.flows.append(flow16)
        self.ids.append(frame_id)
        self.total += 1
        if len(self.flows) == SHARD:
            self.flush()

    def flush(self, final=False):
        self.bin.close()
        base = os.path.join(self.out, f"things_{self.shard:02d}")
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
    ap.add_argument("--flow-archive", default=FLOW_URL)
    ap.add_argument("--images", default="", help="folder with train/image_clean/left frames (found automatically on Kaggle)")
    ap.add_argument("--out", default="/kaggle/working")
    ap.add_argument("--max-hours", type=float, default=11.0, help="stop and keep what is done after this long")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many samples (testing)")
    args = ap.parse_args()

    if shutil.which("lbzip2") is None and os.path.isdir("/kaggle"):
        subprocess.run("apt-get install -y -q lbzip2 >/dev/null 2>&1", shell=True)   # parallel bzip2, 4x faster
    images = args.images or find_images()
    os.makedirs(args.out, exist_ok=True)
    print("images:", images, "| out:", args.out, flush=True)

    proc, tar = open_stream(args.flow_archive)
    writer = Writer(args.out)
    started = time.time()
    seen = skipped_missing = outside = 0
    dirs_seen = []
    last_img = (None, None)
    stop_reason = "archive ended"
    for member in tar:
        seen += 1
        d = os.path.dirname(member.name)
        if not dirs_seen or dirs_seen[-1] != d:
            dirs_seen.append(d)
            print(f"[{(time.time() - started) / 60:6.1f} min] entering {d} (member {seen}, samples {writer.total})", flush=True)
        m = WANTED.search(member.name)
        if not m:
            outside += 1
            # the wanted folder is contiguous in the archive: once past it, the rest is not needed
            if writer.total and outside > 200:
                stop_reason = "left the wanted folder"
                break
            continue
        outside = 0
        frame = int(m.group(1))
        buf = tar.extractfile(member).read()
        flow = parse_flo(buf) if m.group(2) == "flo" else parse_pfm(buf)
        p0 = os.path.join(images, f"{frame:07d}.png")
        p1 = os.path.join(images, f"{frame + 1:07d}.png")
        if not (os.path.exists(p0) and os.path.exists(p1)):
            skipped_missing += 1
            continue
        img0 = last_img[1] if last_img[0] == p0 else cv2.imread(p0)
        img1 = cv2.imread(p1)
        last_img = (p1, img1)
        h, w = flow.shape[:2]
        if img0 is None or img1 is None or img0.shape[:2] != (h, w):
            skipped_missing += 1
            continue
        small = cv2.resize(flow, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA) * np.array([OUT_W / w, OUT_H / h], np.float32)
        jpgs = [cv2.imencode(".jpg", cv2.resize(im, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA),
                             [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])[1].tobytes() for im in (img0, img1)]
        writer.add(frame, jpgs[0], jpgs[1], small.astype(np.float16))
        if writer.total % 500 == 0:
            el = time.time() - started
            print(f"[{el / 60:6.1f} min] {writer.total} samples, member {seen}, frame {frame}, "
                  f"{writer.total / el:.1f} samples/s", flush=True)
        if args.limit and writer.total >= args.limit:
            stop_reason = "sample limit"
            break
        if (time.time() - started) / 3600 > args.max_hours:
            stop_reason = "time limit"
            break

    writer.flush(final=True)
    proc.kill()
    meta = {"source": args.flow_archive, "split": "train", "camera": "left", "direction": "into_future",
            "stored_hw": [OUT_H, OUT_W], "original_hw": [540, 960], "samples": writer.total, "shards": writer.shard,
            "shard_size": SHARD, "jpeg_quality": JPEG_QUALITY, "flow_dtype": "float16, pixels of the stored size",
            "image_order": "BGR as decoded by cv2.imdecode", "skipped_missing_frames": skipped_missing,
            "archive_members_read": seen, "stop_reason": stop_reason, "folders_seen": dirs_seen[:40],
            "hours": round((time.time() - started) / 3600, 2)}
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1))
    size = sum(os.path.getsize(os.path.join(args.out, n)) for n in os.listdir(args.out) if n.startswith("things_"))
    print(f"done: {writer.total} samples in {writer.shard} shards, {size / 1e9:.1f} GB, stop reason: {stop_reason}")


if __name__ == "__main__":
    main()
