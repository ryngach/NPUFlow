#!/usr/bin/env python3
"""Which calibration recipe gives the best INT8 EPE? No training, no camera.

For every recipe: run_calibration.py on the trained December model with
different threshold methods / corrections, deploy an all-INT8 cvimodel and
measure EPE on the x86 TPU emulator (90-pair subset of the validation set).

Recipes run in parallel, each in its own work directory. Finished recipes are
appended to results/calib_sweep.jsonl at once and skipped on a rerun.

    .venv/bin/python bench/calib_sweep.py
"""
import argparse
import datetime
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))

import bf16_blocks  # noqa: E402
import eqexp  # noqa: E402
import speed  # noqa: E402

BASE = eqexp.WORK / "base"          # holds m.mlir of the trained checkpoint
TRANSFORM = speed.DOCKER_IMAGE != "edgeflownet-tpumlir:1.25"   # BASE/m.mlir was made by 1.25
WORK = ROOT / "bench" / "work" / ("_calib_" + speed.DOCKER_IMAGE.split(":")[-1] if TRANSFORM else "_calib")
RESULTS = ROOT / "results" / "calib_sweep.jsonl"
M = speed.MODEL_NAME

# name -> extra run_calibration.py arguments
RECIPES = {
    "p9999": "--cali_method use_percentile9999",          # what all earlier experiments used
    "kl": "--cali_method use_kl",
    "mse": "--cali_method use_mse",
    "max": "--cali_method use_max",
    "aciq_laplace": "--cali_method aciq_laplace",
    "aciq_gauss": "--cali_method aciq_gauss",
    "default": "",                                        # tool default (KL with threshold tuning)
    "mse_tuned": "--cali_method mse",
    "p9999_we": "--cali_method use_percentile9999 --we",  # + weight equalization
    "p9999_bc": "--cali_method use_percentile9999 --bc --bc_inference_num 30",   # + bias correction
    "p9999_we_bc": "--cali_method use_percentile9999 --we --bc --bc_inference_num 30",
    "search_threshold": "--search search_threshold --inference_num 10",
    "kl_bc": "--cali_method use_kl --bc --bc_inference_num 30",         # best method + bias correction
}


def done_recipes(tag: str) -> set:
    if not RESULTS.exists():
        return set()
    return {json.loads(line)["recipe"] for line in RESULTS.read_text().splitlines()
            if json.loads(line).get("tag") == tag and "epe" in json.loads(line)}


def run_recipe(name: str, sub_path: Path, sub, fp32, num_pairs: int) -> dict:
    workdir = WORK / name
    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True)
    rel_calib = eqexp.CALIB.relative_to(ROOT)
    if TRANSFORM:
        # Another TPU-MLIR version: rebuild the top MLIR from ONNX with that version.
        shutil.copy(BASE / f"{M}.onnx", workdir / f"{M}.onnx")
        h, w = eqexp.SIZE
        scale = 1.0 / 255.0
        transform = f"""
model_transform.py --model_name {M} --model_def {M}.onnx \\
  --input_shapes [[1,3,{h},{w}],[1,3,{h},{w}]] --channel_format nchw \\
  --pixel_format rgb --keep_aspect_ratio --mean 0,0,0 --scale {scale},{scale},{scale} \\
  --mlir {M}.mlir"""
    else:
        for f in (f"{M}.mlir", f"{M}_top_f32_all_weight.npz"):
            shutil.copy(BASE / f, workdir / f)
        transform = ""
    started = time.time()
    speed.docker_run(workdir, f"""{transform}
run_calibration.py {M}.mlir --data_list /workspace/{rel_calib}/pairs.txt --input_num {num_pairs} \\
  --chip cv181x {RECIPES[name]} -o {M}_cali_table
model_deploy.py --mlir {M}.mlir --quantize INT8 --processor cv181x \\
  --calibration_table {M}_cali_table --fuse_preprocess --quant_input --matmul_perchannel \\
  --model {M}_int8.cvimodel
mv *_final.mlir final_int8.mlir
""", "convert.log")
    convert_s = time.time() - started
    out = workdir / "flows.npy"
    eqexp.emulate(workdir / f"{M}_int8.cvimodel", sub_path, out)
    static = speed.mlir_metrics(workdir / "final_int8.mlir")
    return {"epe": eqexp.epe_table(np.load(out), sub, ref=fp32), "convert_seconds": round(convert_s),
            "bf16_ops": static["bf16_ops"], "cast_ops": static["cast_ops"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recipes", default="all")
    ap.add_argument("--checkpoint", default="train_out_v1/Phase2_step30000_EMA_best.pth")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tag", default="cal1")
    args = ap.parse_args()

    full = np.load(eqexp.EVALSET)
    idx = bf16_blocks.subset(full)
    sub = {k: full[k][idx] for k in full.files}
    sub_path = eqexp.WORK / "evalset_sub.npz"
    np.savez(sub_path, **sub)
    fp32 = eqexp.torch_flows(eqexp.load_model(args.checkpoint), sub)
    num_pairs = len((eqexp.CALIB / "pairs.txt").read_text().splitlines())
    commit, dirty = speed.git_state()

    names = list(RECIPES) if args.recipes == "all" else args.recipes.split(",")
    skip = done_recipes(args.tag)
    todo = [n for n in names if n not in skip]
    print(f"{len(idx)} eval pairs, {num_pairs} calibration pairs; to run: {todo}; already done: {sorted(skip)}")

    def record(name, payload):
        row = {"time": datetime.datetime.now().isoformat(timespec="seconds"), "tag": args.tag,
               "recipe": name, "args": RECIPES[name], "checkpoint": args.checkpoint,
               "git_commit": commit, "git_dirty": dirty, "docker_image": speed.DOCKER_IMAGE,
               "eval_pairs": int(len(idx)), "calib_pairs": num_pairs, **payload}
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_recipe, n, sub_path, sub, fp32, num_pairs): n for n in todo}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                payload = fut.result()
                record(name, payload)
                e = payload["epe"]
                print(f"{name:18s} " + " ".join(f"{e[d]['epe_noc']:.3f}" for d in e)
                      + f"  ({payload['convert_seconds']} s to convert)", flush=True)
            except Exception as ex:
                record(name, {"error": str(ex)[-1500:]})
                print(f"{name:18s} FAILED: {str(ex)[-300:]}", flush=True)

    rows = {}
    for line in RESULTS.read_text().splitlines():
        r = json.loads(line)
        if r.get("tag") == args.tag and "epe" in r:
            rows[r["recipe"]] = r
    fp = eqexp.epe_table(fp32, sub)
    datasets = list(fp)
    print(f"\n{'recipe':18s} " + " ".join(f"{d:>13s}" for d in datasets) + "  bf16 cast")
    print(f"{'fp32':18s} " + " ".join(f"{fp[d]['epe_noc']:13.3f}" for d in datasets))
    for name, r in sorted(rows.items(), key=lambda kv: kv[1]["epe"][datasets[0]]["epe_noc"]):
        print(f"{name:18s} " + " ".join(f"{r['epe'][d]['epe_noc']:13.3f}" for d in datasets)
              + f"  {r['bf16_ops']:4d} {r['cast_ops']:4d}")


if __name__ == "__main__":
    main()
