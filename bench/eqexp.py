#!/usr/bin/env python3
"""Does concat range equalization recover INT8 accuracy? No training involved.

Takes a trained checkpoint and compares, on the December validation split:
  fp32        PyTorch reference
  int8        everything INT8 (no qtable)
  mixed       full-resolution refine kept in BF16 (the December deployment)
  eq_int8     concat inputs equalized at export, everything INT8

INT8 models run on the x86 TPU emulator inside the TPU-MLIR Docker image.

    .venv/bin/python bench/eqexp.py --checkpoint train_out_v1/Phase2_step30000_EMA_best.pth
"""
import argparse
import copy
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))

import speed  # noqa: E402  (docker_run, deploy, mlir_metrics, git_state)
from model import EdgeFlowNet, ModelConfig  # noqa: E402
from utils import read_flo, resize_flow, resize_img, resize_mask_nearest  # noqa: E402

# Threshold method for run_calibration.py. The first experiments (eq1, bis1) used
# use_percentile9999; set EFN_CALI_METHOD=use_kl for the method chosen by the sweep.
CALI_METHOD = os.environ.get("EFN_CALI_METHOD", "use_percentile9999")
WORK = ROOT / "bench" / "work" / ("_eqexp" if CALI_METHOD == "use_percentile9999"
                                  else "_eqexp_" + CALI_METHOD.replace("use_", ""))
EVALSET = ROOT / "bench" / "work" / "_eqexp" / "evalset.npz"   # shared by all calibration methods
CALIB = ROOT / "bench" / "work" / "_ranges" / "calib"   # 92 adjacent Sintel pairs
SINTEL = ROOT / "datasets" / "MPI-Sintel-complete" / "training"
RESULTS = ROOT / "results" / "eqexp.jsonl"
SIZE = (256, 320)
SEED = 1337   # the split seed used by v1.py


# ------------------------------------------------------------------ eval set

def val_indices(n: int) -> np.ndarray:
    np.random.seed(SEED)
    perm = np.random.permutation(n)
    return perm[int(0.9 * n):]


def sintel_val(pass_name: str):
    import imageio.v3 as iio
    samples = []
    for scene in sorted(os.listdir(SINTEL / pass_name)):
        frames = sorted(f for f in os.listdir(SINTEL / pass_name / scene) if f.endswith(".png"))
        for a, b in zip(frames, frames[1:]):
            samples.append((scene, a, b))
    out = []
    for i in val_indices(len(samples)):
        scene, a, b = samples[i]
        base = a.replace(".png", "")
        img0 = iio.imread(SINTEL / pass_name / scene / a)
        img1 = iio.imread(SINTEL / pass_name / scene / b)
        flow = read_flo(str(SINTEL / "flow" / scene / (base + ".flo")))
        inv = iio.imread(SINTEL / "invalid" / scene / (base + ".png"))
        occ = iio.imread(SINTEL / "occlusions" / scene / (base + ".png"))
        valid_noc = ~(inv.astype(bool) | occ.astype(bool))
        valid_all = ~inv.astype(bool)
        out.append((img0, img1, flow, valid_noc, valid_all))
    return out


def kitti_val():
    # Same order and split as torchvision.datasets.KittiFlow used by v1.py.
    import cv2
    import imageio.v3 as iio
    kitti = ROOT / "datasets" / "data_scene_flow" / "training"
    first = sorted((kitti / "image_2").glob("*_10.png"))
    out = []
    for i in val_indices(len(first)):
        img0_path = first[int(i)]
        img1_path = img0_path.with_name(img0_path.name.replace("_10.png", "_11.png"))
        # KITTI PNG is R=u, G=v, B=valid; cv2 returns BGR.
        # (utils.read_kitti_png_flow indexes the BGR array as if it were RGB, so it is not used here.)
        png = cv2.imread(str(kitti / "flow_occ" / img0_path.name), cv2.IMREAD_UNCHANGED)[:, :, ::-1]
        valid = png[:, :, 2] > 0
        flow = (png[:, :, :2].astype(np.float32) - 32768.0) / 64.0
        out.append((iio.imread(img0_path), iio.imread(img1_path), flow, valid, valid))
    return out


def build_evalset(path: Path, with_kitti: bool):
    if path.exists():
        return
    groups = [("sintel-clean", sintel_val("clean")), ("sintel-final", sintel_val("final"))]
    if with_kitti:
        groups.append(("kitti15", kitti_val()))
    img0s, img1s, flows, noc, allv, names = [], [], [], [], [], []
    for name, items in groups:
        for img0, img1, flow, valid_noc, valid_all in items:
            img0s.append(resize_img(img0, SIZE).transpose(2, 0, 1))
            img1s.append(resize_img(img1, SIZE).transpose(2, 0, 1))
            f = resize_flow(flow, SIZE)
            flows.append(f if f.shape[0] == 2 else f.transpose(2, 0, 1))
            noc.append(resize_mask_nearest(valid_noc, SIZE))
            allv.append(resize_mask_nearest(valid_all, SIZE))
            names.append(name)
    np.savez(path, img0=np.stack(img0s).astype(np.uint8), img1=np.stack(img1s).astype(np.uint8),
             flow=np.stack(flows).astype(np.float32), valid_noc=np.stack(noc), valid_all=np.stack(allv),
             dataset=np.array(names))


# --------------------------------------------------------------------- model

def load_model(checkpoint: str) -> EdgeFlowNet:
    model = EdgeFlowNet(ModelConfig()).eval()
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()}, strict=True)
    return model


def calib_tensors():
    import cv2
    for line in (CALIB / "pairs.txt").read_text().splitlines():
        pair = []
        for name in line.split(","):
            img = cv2.cvtColor(cv2.imread(str(CALIB / Path(name.strip()).name)), cv2.COLOR_BGR2RGB)
            pair.append(torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0)
        yield pair


@torch.no_grad()
def concat_ranges(model: EdgeFlowNet) -> dict:
    """99.99th percentile of |x| for every group of every concat, over the calibration pairs."""
    torch.manual_seed(0)
    model._cat_stats = {}
    for img0, img1 in calib_tensors():
        model(img0, img1)
    stats, model._cat_stats = model._cat_stats, None
    return stats


def group_channels(model: EdgeFlowNet) -> dict:
    """Channels per concat group, read off a dry run."""
    sizes = {}
    orig = model._cat

    def spy(name, groups):
        sizes[name] = [g.shape[1] for g in groups]
        return orig(name, groups)

    model._cat = spy
    with torch.no_grad():
        model(torch.zeros(1, 3, *SIZE), torch.zeros(1, 3, *SIZE))
    del model._cat
    return sizes


@torch.no_grad()
def torch_flows(model: EdgeFlowNet, evalset) -> np.ndarray:
    out = []
    for i in range(0, len(evalset["img0"]), 8):
        a = torch.from_numpy(evalset["img0"][i:i + 8]).float() / 255.0
        b = torch.from_numpy(evalset["img1"][i:i + 8]).float() / 255.0
        out.append(model(a, b).numpy())
    return np.concatenate(out)


def export_onnx(model: EdgeFlowNet, path: Path):
    d0, d1 = torch.rand(1, 3, *SIZE), torch.rand(1, 3, *SIZE)
    torch.onnx.export(model, (d0, d1), str(path), input_names=["img0", "img1"],
                      output_names=["flow_full"], opset_version=speed.ONNX_OPSET, dynamic_axes=None,
                      do_constant_folding=True, dynamo=False, export_params=True)


# ----------------------------------------------------------------- pipeline

def convert(workdir: Path, num_pairs: int):
    rel_calib = CALIB.relative_to(ROOT)
    scale = 1.0 / 255.0
    speed.docker_run(workdir, f"""
model_transform.py --model_name {speed.MODEL_NAME} --model_def {speed.MODEL_NAME}.onnx \\
  --input_shapes [[1,3,{SIZE[0]},{SIZE[1]}],[1,3,{SIZE[0]},{SIZE[1]}]] --channel_format nchw \\
  --pixel_format rgb --keep_aspect_ratio --mean 0,0,0 --scale {scale},{scale},{scale} \\
  --mlir {speed.MODEL_NAME}.mlir
run_calibration.py {speed.MODEL_NAME}.mlir --data_list /workspace/{rel_calib}/pairs.txt \\
  --input_num {num_pairs} --cali_method {CALI_METHOD} -o {speed.MODEL_NAME}_cali_table
""", "transform.log")


def emulate(cvimodel: Path, evalset_path: Path, out_path: Path):
    rel = lambda p: Path(p).resolve().relative_to(ROOT)   # noqa: E731
    cmd = ["docker", "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp",
           "-v", f"{ROOT}:/workspace", "-w", "/workspace", speed.DOCKER_IMAGE, "bash", "-c",
           "P=/usr/local/lib/python3.10/dist-packages/tpu_mlir; "
           "export PYTHONPATH=$P/python LD_LIBRARY_PATH=$P/lib; "
           f"python3 bench/emu_eval.py {rel(cvimodel)} {rel(evalset_path)} {rel(out_path)}"]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError("emulator failed:\n" + (proc.stdout + proc.stderr)[-1500:])


def epe_table(pred: np.ndarray, evalset, ref: np.ndarray = None) -> dict:
    err = np.sqrt(((pred - evalset["flow"]) ** 2).sum(1))
    out = {}
    for name in dict.fromkeys(evalset["dataset"].tolist()):
        sel = evalset["dataset"] == name
        noc, allv = evalset["valid_noc"][sel], evalset["valid_all"][sel]
        row = {"epe_noc": float(err[sel][noc].mean()), "epe_all": float(err[sel][allv].mean())}
        if ref is not None:
            row["epe_vs_fp32"] = float(np.sqrt(((pred[sel] - ref[sel]) ** 2).sum(1)).mean())
        out[name] = row
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="train_out_v1/Phase2_step30000_EMA_best.pth")
    ap.add_argument("--no-kitti", action="store_true")
    ap.add_argument("--no-clearml", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    WORK.mkdir(parents=True, exist_ok=True)
    evalset_path = EVALSET
    evalset_path.parent.mkdir(parents=True, exist_ok=True)
    build_evalset(evalset_path, with_kitti=not args.no_kitti)
    evalset = np.load(evalset_path)
    num_calib = len((CALIB / "pairs.txt").read_text().splitlines())
    print(f"eval pairs: {len(evalset['img0'])}, calibration pairs: {num_calib}")

    base = load_model(args.checkpoint)
    ranges = concat_ranges(base)
    channels = group_channels(base)
    scales = {name: [r if r > 1e-6 else 1.0 for r in rs] for name, rs in ranges.items()}
    eq = copy.deepcopy(base)
    eq.fold_cat_scales(scales, channels)

    fp32 = torch_flows(base, evalset)
    fp32_eq = torch_flows(eq, evalset)
    results = {"fp32": epe_table(fp32, evalset)}
    max_diff = float(np.abs(fp32 - fp32_eq).max())
    print(f"equalized vs original in FP32: max abs diff {max_diff:.2e}")

    static, jobs = {}, []
    for tag, model, quants in (("base", base, ("int8", "mixed")), ("eq", eq, ("int8",))):
        workdir = WORK / tag
        workdir.mkdir(exist_ok=True)
        export_onnx(model, workdir / f"{speed.MODEL_NAME}.onnx")
        convert(workdir, num_calib)
        for quant in quants:
            name = quant if tag == "base" else f"eq_{quant}"
            cvimodel = speed.deploy(workdir, quant)
            static[name] = speed.mlir_metrics(workdir / f"final_{quant}.mlir")
            jobs.append((name, cvimodel, workdir / f"flows_{quant}.npy"))

    def run(job):
        name, cvimodel, out = job
        emulate(cvimodel, evalset_path, out)
        return name, epe_table(np.load(out), evalset, ref=fp32)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=3) as pool:
        for name, table in pool.map(run, jobs):
            results[name] = table
            print(name, json.dumps(table), flush=True)

    commit, dirty = speed.git_state()
    record = {
        "time": datetime.datetime.now().isoformat(timespec="seconds"), "tag": args.tag,
        "checkpoint": args.checkpoint, "git_commit": commit, "git_dirty": dirty,
        "docker_image": speed.DOCKER_IMAGE, "cali_method": CALI_METHOD, "eval_pairs": int(len(evalset["img0"])),
        "calib_pairs": num_calib, "concat_ranges": ranges, "concat_channels": channels,
        "fp32_equalized_max_abs_diff": max_diff, "results": results,
        "static": {k: {m: v[m] for m in ("cast_ops", "bf16_ops", "total_ops", "generic_cpu_ops")}
                   for k, v in static.items()},
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    if not args.no_clearml:
        try:
            from clearml import Task
            task = Task.init(project_name="EdgeFlowNet/quant", task_name="concat range equalization",
                             task_type=Task.TaskTypes.testing, tags=[t for t in (args.tag,) if t],
                             reuse_last_task_id=False, auto_connect_frameworks=False)
            task.connect({"checkpoint": args.checkpoint, "git_commit": commit, "git_dirty": dirty,
                          "docker_image": speed.DOCKER_IMAGE}, name="run")
            logger = task.get_logger()
            for model_name, table in results.items():
                for dataset, row in table.items():
                    for metric, value in row.items():
                        logger.report_single_value(f"{model_name}/{dataset}/{metric}", value)
            task.upload_artifact("record", artifact_object=record)
            task.close()
        except Exception as e:
            print(f"ClearML logging failed (result is in {RESULTS.name}): {e}")

    print(f"\n{'model':9s} " + " ".join(f"{d:>26s}" for d in results["fp32"]))
    print(f"{'':9s} " + " ".join(f"{'noc':>8s} {'all':>8s} {'vs fp32':>8s}" for _ in results["fp32"]))
    for model_name, table in results.items():
        cells = []
        for row in table.values():
            vs = row.get("epe_vs_fp32")
            cells.append(f"{row['epe_noc']:8.3f} {row['epe_all']:8.3f} {('-' if vs is None else f'{vs:.3f}'):>8s}")
        print(f"{model_name:9s} " + " ".join(cells))


if __name__ == "__main__":
    main()
