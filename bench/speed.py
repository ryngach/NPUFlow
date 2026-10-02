#!/usr/bin/env python3
"""On-device speed screening of architecture variants (no training).

For every variant in bench/variants.yaml:
  1. build the model with random weights and export ONNX
  2. convert to an INT8 cvimodel inside the TPU-MLIR Docker image
  3. read static metrics from the compiled final.mlir
  4. copy the cvimodel to the MaixCAM and time it with model_runner
  5. append one line to results/speed.jsonl and log a ClearML task

Run from the project root with the project venv:
    .venv/bin/python bench/speed.py --variants B0,S4_relu
"""
import argparse
import datetime
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BENCH = ROOT / "bench"
WORK = BENCH / "work"
CALIB = BENCH / "calib"
RESULTS = ROOT / "results" / "speed.jsonl"

# Pinned toolchain; set EFN_DOCKER_IMAGE to try another TPU-MLIR version.
DOCKER_IMAGE = os.environ.get("EFN_DOCKER_IMAGE", "edgeflownet-tpumlir:1.25")
DEVICE = os.environ.get("NPUFLOW_DEVICE", "root@192.168.88.18")   # MaixCAM over ssh
DEVICE_DIR = "/root/efn_bench"
SINTEL_CLEAN = ROOT / "datasets" / "MPI-Sintel-complete" / "training" / "clean"
ONNX_OPSET = 13
MODEL_NAME = "m"

# Ops that only move data in/out of local memory or group layers; not real work.
BOOKKEEPING_OPS = {"tpu.Load", "tpu.Store", "tpu.Yield", "tpu.Group", "top.Weight",
                   "top.Input", "top.None"}


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, text=True, capture_output=True, **kw)


def git_state():
    commit = run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT).stdout.strip()
    dirty = bool(run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT).stdout.strip())
    return commit, dirty


# ---------------------------------------------------------------- calibration

def ensure_calib(num_pairs: int) -> Path:
    """Write num_pairs adjacent-frame Sintel pairs (256x320) and a pair list."""
    pair_list = CALIB / "pairs.txt"
    if pair_list.exists() and len(pair_list.read_text().splitlines()) == num_pairs:
        return pair_list

    import cv2
    from utils import resize_img
    from model import ModelConfig

    size = ModelConfig().image_size
    scenes = sorted(d for d in SINTEL_CLEAN.iterdir() if d.is_dir())
    if not scenes:
        raise RuntimeError(f"No Sintel scenes in {SINTEL_CLEAN}")
    shutil.rmtree(CALIB, ignore_errors=True)
    CALIB.mkdir(parents=True)

    lines = []
    for i in range(num_pairs):
        scene = scenes[i * len(scenes) // num_pairs]
        frames = sorted(scene.glob("frame_*.png"))
        start = len(frames) // 2
        names = []
        for suffix, src in (("a", frames[start]), ("b", frames[start + 1])):
            img = cv2.cvtColor(cv2.imread(str(src)), cv2.COLOR_BGR2RGB)
            img = cv2.cvtColor(resize_img(img, size), cv2.COLOR_RGB2BGR)
            name = f"cal_{i:03d}_{suffix}.jpg"
            cv2.imwrite(str(CALIB / name), img)
            names.append(f"/workspace/bench/calib/{name}")
        lines.append(", ".join(names))
    pair_list.write_text("\n".join(lines) + "\n")
    return pair_list


# --------------------------------------------------------------------- export

def export_onnx(cfg_overrides: dict, out_path: Path) -> dict:
    import torch
    from model import EdgeFlowNet, ModelConfig

    torch.manual_seed(0)
    cfg = ModelConfig(**cfg_overrides)
    model = EdgeFlowNet(cfg).eval()
    d0 = torch.rand(1, 3, *cfg.image_size)
    d1 = torch.rand(1, 3, *cfg.image_size)
    torch.onnx.export(
        model, (d0, d1), str(out_path),
        input_names=["img0", "img1"], output_names=["flow_full"],
        opset_version=ONNX_OPSET, dynamic_axes=None,
        do_constant_folding=True, dynamo=False, export_params=True,
    )
    return {"config": cfg.to_dict(), "params": model.count_params()}


# ----------------------------------------------------------------- conversion

def docker_run(workdir: Path, script: str, log_name: str):
    rel = workdir.relative_to(ROOT)
    cmd = ["docker", "run", "--rm",
           "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp",
           "-v", f"{ROOT}:/workspace", "-w", f"/workspace/{rel}",
           DOCKER_IMAGE, "bash", "-c", "set -e\n" + script]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    (workdir / log_name).write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-25:])
        raise RuntimeError(f"docker step failed, see {workdir / log_name}\n{tail}")


def transform_and_calibrate(workdir: Path, cfg: dict, num_pairs: int):
    h, w = cfg["image_size"]
    scale = 1.0 / 255.0
    docker_run(workdir, f"""
model_transform.py --model_name {MODEL_NAME} --model_def {MODEL_NAME}.onnx \\
  --input_shapes [[1,3,{h},{w}],[1,3,{h},{w}]] --channel_format nchw \\
  --pixel_format rgb --keep_aspect_ratio \\
  --mean 0,0,0 --scale {scale},{scale},{scale} \\
  --mlir {MODEL_NAME}.mlir
run_calibration.py {MODEL_NAME}.mlir --data_list /workspace/bench/calib/pairs.txt \\
  --input_num {num_pairs} -o {MODEL_NAME}_cali_table
""", "transform.log")


def full_res_qtable(workdir: Path) -> list:
    """Names of the full-resolution refine ops, to be kept in BF16 ("mixed" policy)."""
    text = (workdir / f"{MODEL_NAME}.mlir").read_text()
    names = re.findall(r'#loc\d+ = loc\("([^"]+)"\)', text)
    return [n for n in names if n.startswith("/full_refine/") or n.startswith("flow_full")]


def deploy(workdir: Path, quant: str) -> Path:
    qtable_arg = ""
    if quant == "mixed":
        ops = full_res_qtable(workdir)
        (workdir / "qtable_mixed.txt").write_text(
            "# full-resolution refine kept in BF16\n" + "".join(f"{op} BF16\n" for op in ops))
        qtable_arg = "--quantize_table qtable_mixed.txt"
    out = f"{MODEL_NAME}_{quant}.cvimodel"
    docker_run(workdir, f"""
rm -f *_final.mlir
model_deploy.py --mlir {MODEL_NAME}.mlir --quantize INT8 --processor cv181x \\
  --calibration_table {MODEL_NAME}_cali_table {qtable_arg} \\
  --fuse_preprocess --quant_input --matmul_perchannel \\
  --model {out}
mv *_final.mlir final_{quant}.mlir
""", f"deploy_{quant}.log")
    return workdir / out


# ------------------------------------------------------------- static metrics

def mlir_metrics(final_mlir: Path) -> dict:
    text = final_mlir.read_text()
    ops = Counter()
    bf16_ops = 0
    for line in text.splitlines():
        m = re.search(r'= "((?:tpu|top)\.[A-Za-z0-9_]+)"', line)
        if not m:
            continue
        op = m.group(1)
        ops[op] += 1
        if op in BOOKKEEPING_OPS:
            continue
        out_type = line.rsplit("->", 1)[-1]
        if "xbf16" in out_type or "bf16<" in out_type:
            bf16_ops += 1
    work_ops = {k: v for k, v in ops.items() if k not in BOOKKEEPING_OPS}
    flops = re.search(r"module\.FLOPs = (\d+)", text)
    return {
        "generic_cpu_ops": ops.get("tpu.GenericCpu", 0),
        "subnets": len(re.findall(r"func\.func @subfunc", text)),
        "cast_ops": ops.get("tpu.Cast", 0),
        "bf16_ops": bf16_ops,
        "total_ops": sum(work_ops.values()),
        "flops": int(flops.group(1)) if flops else None,
        "op_histogram": dict(sorted(work_ops.items(), key=lambda kv: -kv[1])),
    }


# --------------------------------------------------------------------- device

SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]


def ssh(command: str, attempts: int = 3, timeout: int = 60) -> str:
    """Run a command on the camera; its Wi-Fi often drops the first connection."""
    last = None
    for i in range(attempts):
        try:
            proc = subprocess.run(["ssh", *SSH_OPTS, DEVICE, command], text=True,
                                  capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            last = f"timed out after {timeout}s"
            continue
        if proc.returncode == 0:
            return proc.stdout   # local ssh warnings go to stderr and are dropped
        last = proc.stderr.strip()
        if proc.returncode != 255:   # 255 = ssh transport error; anything else is the command itself
            break
        time.sleep(3)
    raise RuntimeError(f"ssh failed: {command!r}: {last}")


def device_uptime() -> float:
    return float(ssh("cat /proc/uptime").split()[0])


def wait_device(max_wait: int = 300) -> bool:
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            ssh("true", attempts=1, timeout=12)
            return True
        except RuntimeError:
            time.sleep(6)
    return False


def recover_device() -> str:
    """Reboot the camera after a failed run so the next variant starts from a clean state."""
    try:
        ssh("sync; reboot", attempts=1, timeout=20)
    except RuntimeError:
        pass   # already rebooting or hung; either way wait for it to come back
    time.sleep(25)
    if not wait_device():
        return "unreachable"
    try:
        ssh(f"rm -f {DEVICE_DIR}/*.cvimodel")
    except RuntimeError:
        pass
    return "rebooted"


def device_temp() -> float:
    return int(ssh("cat /sys/class/thermal/thermal_zone0/temp").strip().splitlines()[-1]) / 1000.0


def device_bench(cvimodel: Path, runs: int, count: int) -> dict:
    remote = f"{DEVICE_DIR}/{cvimodel.parent.name}_{cvimodel.name}"
    ssh(f"mkdir -p {DEVICE_DIR}")
    err = None
    for i in range(3):
        try:
            proc = subprocess.run(["scp", "-q", *SSH_OPTS, str(cvimodel), f"{DEVICE}:{remote}"],
                                  text=True, capture_output=True, timeout=90)
            if proc.returncode == 0:
                break
            err = proc.stderr.strip()
        except subprocess.TimeoutExpired:
            err = "timed out after 90s"
        time.sleep(3)
    else:
        raise RuntimeError(f"scp failed: {err}")

    times, shared_mem = [], None
    uptime_before = device_uptime()
    try:
        temp_before = device_temp()
        for _ in range(runs):
            out = ssh(f"model_runner --model {remote} --count {count} --enable-timer 2>&1", timeout=180)
            m = re.search(r"each run takes ([\d.]+) ms", out)
            if not m:
                raise RuntimeError(f"no timing in model_runner output:\n{out[-800:]}")
            times.append(float(m.group(1)))
            sm = re.search(r"Max SharedMem size:(\d+)", out)
            shared_mem = int(sm.group(1)) if sm else shared_mem
    except RuntimeError as e:
        # Tell a device reset apart from a plain network drop.
        rebooted = None
        if wait_device(180):
            try:
                rebooted = device_uptime() < uptime_before
            except RuntimeError:
                pass
        raise RuntimeError(f"{e} | device_rebooted_during_run={rebooted} | completed_runs={times}")
    finally:
        try:
            ssh(f"rm -f {remote}", attempts=1, timeout=20)
        except RuntimeError:
            pass
    return {
        "inference_ms_median": statistics.median(times),
        "inference_ms_min": min(times),
        "inference_ms_max": max(times),
        "inference_ms_runs": times,
        "runs": runs,
        "count_per_run": count,
        "shared_mem_bytes": shared_mem,
        "temp_c_before": temp_before,
        "temp_c_after": device_temp(),
    }


# -------------------------------------------------------------------- logging

def log_clearml(record: dict, artifacts: dict, tag: str):
    from clearml import Task

    task = Task.init(project_name="EdgeFlowNet/speed",
                     task_name=f"{record['variant']}/{record['quant']}",
                     task_type=Task.TaskTypes.testing,
                     tags=[t for t in (tag, record["quant"]) if t],
                     reuse_last_task_id=False,
                     auto_connect_frameworks=False)
    try:
        task.connect(record["config"], name="model")
        task.connect({k: record[k] for k in ("variant", "quant", "overrides", "git_commit", "git_dirty",
                                             "docker_image", "onnx_opset")}, name="run")
        logger = task.get_logger()
        for group in ("static", "device"):
            for key, value in (record.get(group) or {}).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    logger.report_single_value(f"{group}/{key}", value)
        logger.report_single_value("params", record["params"])
        logger.report_single_value("cvimodel_bytes", record["cvimodel_bytes"])
        task.upload_artifact("record", artifact_object=record)
        for name, path in artifacts.items():
            if Path(path).exists():
                task.upload_artifact(name, artifact_object=str(path))
        return task.id
    finally:
        task.close()


def append_result(record: dict):
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------------- main

def main():
    import yaml

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", default="all", help="comma-separated names from variants.yaml, or 'all'")
    ap.add_argument("--quant", default="int8,mixed",
                    help="int8: everything INT8; mixed: full-resolution refine kept in BF16")
    ap.add_argument("--runs", type=int, default=5, help="model_runner invocations per model")
    ap.add_argument("--count", type=int, default=30, help="inferences per model_runner invocation")
    ap.add_argument("--calib-pairs", type=int, default=10)
    ap.add_argument("--tag", default="", help="session label stored with every record")
    ap.add_argument("--no-device", action="store_true", help="convert only, skip the camera")
    ap.add_argument("--no-clearml", action="store_true")
    args = ap.parse_args()

    all_variants = yaml.safe_load((BENCH / "variants.yaml").read_text())
    names = list(all_variants) if args.variants == "all" else args.variants.split(",")
    unknown = [n for n in names if n not in all_variants]
    if unknown:
        sys.exit(f"Unknown variants: {unknown}. Known: {list(all_variants)}")
    quants = args.quant.split(",")

    commit, dirty = git_state()
    ensure_calib(args.calib_pairs)
    summary = []

    for name in names:
        overrides = all_variants[name] or {}
        workdir = WORK / name
        shutil.rmtree(workdir, ignore_errors=True)
        workdir.mkdir(parents=True)
        print(f"\n=== {name} {overrides}")
        try:
            info = export_onnx(overrides, workdir / f"{MODEL_NAME}.onnx")
            transform_and_calibrate(workdir, info["config"], args.calib_pairs)
        except Exception as e:
            print(f"  FAILED before deploy: {e}")
            append_result({"time": datetime.datetime.now().isoformat(timespec="seconds"),
                           "variant": name, "overrides": overrides, "tag": args.tag,
                           "git_commit": commit, "git_dirty": dirty, "error": str(e)[-2000:]})
            continue

        for quant in quants:
            if quant == "mixed" and not full_res_qtable(workdir):
                print("  mixed: no full-resolution refine ops in this variant, skipped")
                continue
            record = {
                "time": datetime.datetime.now().isoformat(timespec="seconds"),
                "variant": name, "quant": quant, "overrides": overrides, "tag": args.tag,
                "config": info["config"], "params": info["params"],
                "git_commit": commit, "git_dirty": dirty,
                "docker_image": DOCKER_IMAGE, "onnx_opset": ONNX_OPSET,
            }
            try:
                cvimodel = deploy(workdir, quant)
                record["cvimodel_bytes"] = cvimodel.stat().st_size
                record["static"] = mlir_metrics(workdir / f"final_{quant}.mlir")
                if not args.no_device:
                    record["device"] = device_bench(cvimodel, args.runs, args.count)
            except Exception as e:
                record["error"] = str(e)[-2000:]
                print(f"  {quant}: FAILED: {e}")
                if "static" in record and not args.no_device:
                    record["device_recovery"] = recover_device()
                    print(f"  {quant}: device {record['device_recovery']}")
                append_result(record)
                if record.get("device_recovery") == "unreachable":
                    sys.exit("Camera did not come back after reboot; power-cycle it and rerun.")
                continue

            append_result(record)   # local history first, so a ClearML problem cannot lose a result
            if not args.no_clearml:
                try:
                    task_id = log_clearml(record, {"cvimodel": cvimodel,
                                                   "final_mlir": workdir / f"final_{quant}.mlir",
                                                   "onnx": workdir / f"{MODEL_NAME}.onnx"}, args.tag)
                    print(f"  {quant}: ClearML task {task_id}")
                except Exception as e:
                    print(f"  {quant}: ClearML logging failed (result is in {RESULTS.name}): {e}")

            st, dev = record["static"], record.get("device", {})
            ms = dev.get("inference_ms_median")
            summary.append((name, quant, ms, st["generic_cpu_ops"], st["cast_ops"], st["bf16_ops"], st["total_ops"]))
            print(f"  {quant}: {ms if ms is None else f'{ms:.2f} ms'} | cpu_ops {st['generic_cpu_ops']} "
                  f"| casts {st['cast_ops']} | bf16 {st['bf16_ops']} | ops {st['total_ops']}")

    print(f"\n{'variant':22s} {'quant':6s} {'ms':>8s} {'cpu':>4s} {'cast':>5s} {'bf16':>5s} {'ops':>5s}")
    for name, quant, ms, cpu, cast, bf16, total in summary:
        print(f"{name:22s} {quant:6s} {('-' if ms is None else f'{ms:.2f}'):>8s} {cpu:4d} {cast:5d} {bf16:5d} {total:5d}")


if __name__ == "__main__":
    main()
