#!/usr/bin/env python3
"""Per-block choice of calibration method, judged by INT8 EPE on the emulator.

Start from the KL calibration table. For every block of the network, replace
that block's thresholds with the ones from another method (tables produced by
bench/calib_sweep.py), deploy, and measure EPE on a SELECTION subset. Then build
one hybrid table from the per-block winners and evaluate it on the HELD-OUT
pairs, next to plain KL on the same pairs.

    .venv/bin/python bench/hybrid_calib.py --stage select
    .venv/bin/python bench/hybrid_calib.py --stage final
"""
import argparse
import datetime
import json
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))

import bf16_blocks  # noqa: E402
import eqexp  # noqa: E402
import speed  # noqa: E402

TABLES = ROOT / "bench" / "work" / "_calib"          # <method>/m_cali_table from calib_sweep.py
SRC = ROOT / "bench" / "work" / "_eqexp" / "base"    # m.mlir + weights of the trained checkpoint
WORK = ROOT / "bench" / "work" / "_hybrid"
RESULTS = ROOT / "results" / "hybrid_calib.jsonl"
M = speed.MODEL_NAME
BASE_METHOD = "kl"
ALT_METHODS = ("aciq_laplace", "mse")
BLOCKS = ("encoder", "corr32_update32", "corr16", "update16", "refine_s16", "refine_s8",
          "refine_s4", "full_refine", "flow_path")
MIN_GAIN = 0.01   # EPE; smaller differences on the selection set are treated as noise


def read_table(method: str):
    """Return (header lines, {tensor name: full line}) in file order."""
    header, rows = [], {}
    for line in (TABLES / method / f"{M}_cali_table").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            header.append(line)
        else:
            rows[line.split()[0]] = line
    return header, rows


def hybrid_table(choice: dict) -> str:
    """choice: block -> method. Tensors outside the listed blocks keep the base method."""
    header, rows = read_table(BASE_METHOD)
    replaced = {}
    for block, method in choice.items():
        if method == BASE_METHOD:
            continue
        _, alt = read_table(method)
        pattern = bf16_blocks.BLOCKS[block]
        n = 0
        for name in rows:
            if re.search(pattern, name) and name in alt:
                rows[name] = alt[name]
                n += 1
        replaced[block] = n
    return "\n".join(header + list(rows.values())) + "\n", replaced


def build_and_run(tag: str, choice: dict, set_path: Path) -> np.ndarray:
    workdir = WORK / tag
    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True)
    for f in (f"{M}.mlir", f"{M}_top_f32_all_weight.npz"):
        shutil.copy(SRC / f, workdir / f)
    table, replaced = hybrid_table(choice)
    (workdir / f"{M}_cali_table").write_text(table)
    speed.docker_run(workdir, f"""
model_deploy.py --mlir {M}.mlir --quantize INT8 --processor cv181x \\
  --calibration_table {M}_cali_table --fuse_preprocess --quant_input --matmul_perchannel \\
  --model {M}_int8.cvimodel
""", "deploy.log")
    out = workdir / "flows.npy"
    eqexp.emulate(workdir / f"{M}_int8.cvimodel", set_path, out)
    return np.load(out), replaced


def save_set(full, idx, path: Path):
    data = {k: full[k][idx] for k in full.files}
    np.savez(path, **data)
    return data


def record(row: dict):
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def mean_epe(table: dict) -> float:
    return float(np.mean([r["epe_noc"] for r in table.values()]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("select", "final"), required=True)
    ap.add_argument("--checkpoint", default="train_out_v1/Phase2_step30000_EMA_best.pth")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--tag", default="hyb1")
    args = ap.parse_args()

    WORK.mkdir(parents=True, exist_ok=True)
    full = np.load(eqexp.EVALSET)
    sel_idx = bf16_blocks.subset(full)                      # 90 pairs: selection
    test_idx = np.setdiff1d(np.arange(len(full["img0"])), sel_idx)   # 140 pairs: held out
    model = eqexp.load_model(args.checkpoint)
    commit, dirty = speed.git_state()
    meta = {"tag": args.tag, "checkpoint": args.checkpoint, "git_commit": commit, "git_dirty": dirty,
            "docker_image": speed.DOCKER_IMAGE, "base_method": BASE_METHOD}

    if args.stage == "select":
        sel = save_set(full, sel_idx, WORK / "set_select.npz")
        fp32 = eqexp.torch_flows(model, sel)
        jobs = {"base": {}}
        for block in BLOCKS:
            for method in ALT_METHODS:
                jobs[f"{block}__{method}"] = {block: method}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(build_and_run, name, choice, WORK / "set_select.npz"): (name, choice)
                       for name, choice in jobs.items()}
            for fut in as_completed(futures):
                name, choice = futures[fut]
                try:
                    flows, replaced = fut.result()
                    table = eqexp.epe_table(flows, sel, ref=fp32)
                    record({"time": datetime.datetime.now().isoformat(timespec="seconds"), "stage": "select",
                            "variant": name, "choice": choice, "replaced_tensors": replaced,
                            "eval_pairs": int(len(sel_idx)), "epe": table, **meta})
                    print(f"{name:32s} " + " ".join(f"{r['epe_noc']:.3f}" for r in table.values())
                          + f"  mean {mean_epe(table):.3f}  replaced {replaced}", flush=True)
                except Exception as e:
                    record({"stage": "select", "variant": name, "error": str(e)[-1000:], **meta})
                    print(f"{name:32s} FAILED: {str(e)[-200:]}", flush=True)
        return

    # ---- final: pick winners from the recorded selection rows, test on held-out pairs
    rows = [json.loads(line) for line in RESULTS.read_text().splitlines()]
    rows = {r["variant"]: r for r in rows if r.get("tag") == args.tag and r.get("stage") == "select" and "epe" in r}
    base = mean_epe(rows["base"]["epe"])
    choice = {}
    for block in BLOCKS:
        best_method, best = BASE_METHOD, base
        for method in ALT_METHODS:
            r = rows.get(f"{block}__{method}")
            if r and mean_epe(r["epe"]) < best - MIN_GAIN:
                best_method, best = method, mean_epe(r["epe"])
        choice[block] = best_method
        print(f"{block:18s} -> {best_method:13s} (selection mean EPE {best:.3f}, base {base:.3f})")

    test = save_set(full, test_idx, WORK / "set_test.npz")
    fp32 = eqexp.torch_flows(model, test)
    results = {"fp32": eqexp.epe_table(fp32, test)}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futs = {pool.submit(build_and_run, "final_base", {}, WORK / "set_test.npz"): "kl",
                pool.submit(build_and_run, "final_hybrid", choice, WORK / "set_test.npz"): "hybrid"}
        for fut in as_completed(futs):
            flows, replaced = fut.result()
            results[futs[fut]] = eqexp.epe_table(flows, test, ref=fp32)
    record({"time": datetime.datetime.now().isoformat(timespec="seconds"), "stage": "final", "choice": choice,
            "eval_pairs": int(len(test_idx)), "results": results, **meta})
    print(f"\nheld-out {len(test_idx)} pairs      " + " ".join(f"{d:>13s}" for d in results["fp32"]))
    for name in ("fp32", "kl", "hybrid"):
        print(f"{name:26s} " + " ".join(f"{r['epe_noc']:13.3f}" for r in results[name].values()))


if __name__ == "__main__":
    main()
