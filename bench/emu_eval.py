#!/usr/bin/env python3
"""Run a fuse_preprocess cvimodel on the x86 TPU emulator over an image-pair set.

Runs INSIDE the TPU-MLIR Docker image:
    python3 bench/emu_eval.py MODEL.cvimodel SET.npz OUT.npy
SET.npz holds img0, img1 as uint8 [N,3,H,W] (RGB). OUT.npy gets float32 flow [N,2,H,W].
"""
import sys
import time

import numpy as np
import pyruntime_cvi
from tools.model_runner import bf16_to_fp32


def main():
    model_path, set_path, out_path = sys.argv[1:4]
    data = np.load(set_path)
    img0, img1 = data["img0"], data["img1"]
    model = pyruntime_cvi.Model(model_path, output_all_tensors=False)
    names = [i.name for i in model.inputs]
    assert len(names) == 2, names
    flows = []
    t0 = time.time()
    for n in range(len(img0)):
        for tensor, img in zip(model.inputs, (img0[n], img1[n])):
            tensor.data.reshape(-1)[:] = img.astype(tensor.data.dtype).flatten()
        model.forward()
        out = model.outputs[0]
        if out.dtype == "bf16":
            flow = bf16_to_fp32(out.data)
        elif out.data.dtype in (np.int8, np.uint8) and out.qscale != 0:
            flow = (out.data.astype(np.float32) - out.qzero_point) * np.float32(out.qscale)
        else:
            flow = np.array(out.data, dtype=np.float32)
        flows.append(flow.reshape(2, img0.shape[2], img0.shape[3]).copy())
        if n == 0 or (n + 1) % 25 == 0:
            print(f"  {n + 1}/{len(img0)}  {time.time() - t0:.0f}s", flush=True)
    np.save(out_path, np.stack(flows))
    print(f"inputs {names} out dtype {model.outputs[0].dtype} qscale {model.outputs[0].qscale}")


if __name__ == "__main__":
    main()
