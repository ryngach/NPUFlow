#!/usr/bin/env python3
"""Where does INT8 lose accuracy? Keep one block at a time in BF16 and measure EPE.

Reuses the converted baseline from bench/eqexp.py (bench/work/_eqexp/base:
m.mlir + calibration table of the trained December checkpoint). For every
block: deploy an INT8 cvimodel with that block forced to BF16 via a qtable,
run it on the x86 TPU emulator over a subset of the validation pairs, and
(optionally) time it on the camera.

    .venv/bin/python bench/bf16_blocks.py
"""
import argparse
import datetime
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))

import eqexp  # noqa: E402
import speed  # noqa: E402

BASE = eqexp.WORK / "base"
RESULTS = ROOT / "results" / "bf16_blocks.jsonl"
M = speed.MODEL_NAME

# Block name -> regex over op names in the top MLIR. None = special handling.
BLOCKS = {
    "bf16_all": None,                                   # upper bound: nothing in INT8
    "encoder": r"^/enc(_1)?/",
    "stem": r"^/enc(_1)?/s8_block\.[01]/",     # first two stride-2 blocks
    "corr32_update32": r"^/(corr32|update32|ctx32)/",
    "corr16": r"^/corr16/",
    "update16": r"^/(update16|ctx16)/",
    "refine_s16": r"^/refine_s16/",
    "refine_s8": r"^/(refine_s8|ctx8)/",
    "refine_s4": r"^/(refine_s4|ctx4)/",
    "full_refine": r"^(/full_refine/|flow_full)",
    "flow_path": r"^/[A-Za-z]+(_\d+)?_output_0_[A-Za-z]+$",  # top-level Add/Mul/Resize/Concat on flow
    # flow state end to end: the top-level flow ops plus the convs that emit flow increments
    "flow_heads": r"^/[A-Za-z]+(_\d+)?_output_0_[A-Za-z]+$|^/(update32|update16|refine_s16|refine_s8|refine_s4)/head(_\d+)?/",
    "flow_heads_full": r"^/[A-Za-z]+(_\d+)?_output_0_[A-Za-z]+$|^/(update32|update16|refine_s16|refine_s8|refine_s4)/head(_\d+)?/"
                       r"|^/full_refine/|^flow_full",
}


def op_names():
    text = (BASE / f"{M}.mlir").read_text()
    names = re.findall(r'#loc\d+ = loc\("([^"]+)"\)', text)
    return [n for n in names if n.startswith("/") or n.startswith("flow_full")]


def deploy(block: str) -> Path:
    out = f"{M}_bis_{block}.cvimodel"
    if block == "bf16_all":
        args = "--quantize BF16"
        selected = []
    else:
        selected = [n for n in op_names() if re.search(BLOCKS[block], n)]
        if not selected:
            raise RuntimeError(f"{block}: no ops matched")
        (BASE / f"qtable_{block}.txt").write_text("".join(f"{n} BF16\n" for n in selected))
        args = f"--quantize INT8 --calibration_table {M}_cali_table --quantize_table qtable_{block}.txt"
    speed.docker_run(BASE, f"""
rm -f *_final.mlir
model_deploy.py --mlir {M}.mlir {args} --processor cv181x \\
  --fuse_preprocess --quant_input --matmul_perchannel --model {out}
mv *_final.mlir final_bis_{block}.mlir
""", f"deploy_bis_{block}.log")
    return BASE / out, len(selected)


def subset(evalset):
    idx = []
    for name in dict.fromkeys(evalset["dataset"].tolist()):
        where = np.flatnonzero(evalset["dataset"] == name)
        idx.extend(where if name == "kitti15" else where[::3])
    return np.array(sorted(idx))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blocks", default="all")
    ap.add_argument("--checkpoint", default="train_out_v1/Phase2_step30000_EMA_best.pth")
    ap.add_argument("--no-device", action="store_true")
    ap.add_argument("--workers", type=int, default=4, help="parallel emulator runs")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    full = np.load(eqexp.EVALSET)
    idx = subset(full)
    sub = {k: full[k][idx] for k in full.files}
    sub_path = eqexp.WORK / "evalset_sub.npz"
    np.savez(sub_path, **sub)
    print(f"subset: {len(idx)} of {len(full['img0'])} pairs")

    fp32 = eqexp.torch_flows(eqexp.load_model(args.checkpoint), sub)
    rows = {"fp32": {"epe": eqexp.epe_table(fp32, sub)}}
    for name in ("int8", "mixed"):   # already emulated on the full set by eqexp.py
        flows = np.load(BASE / f"flows_{name}.npy")[idx]
        key = "int8" if name == "int8" else "full_refine"
        rows[key] = {"epe": eqexp.epe_table(flows, sub, ref=fp32)}
        static = speed.mlir_metrics(BASE / f"final_{name}.mlir")
        rows[key]["bf16_ops"], rows[key]["cast_ops"] = static["bf16_ops"], static["cast_ops"]

    blocks = [b for b in BLOCKS if b != "full_refine"] if args.blocks == "all" else args.blocks.split(",")
    commit, dirty = speed.git_state()

    def evaluate(block, cvimodel):
        out = BASE / f"flows_bis_{block}.npy"
        eqexp.emulate(cvimodel, sub_path, out)
        return eqexp.epe_table(np.load(out), sub, ref=fp32)

    futures = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for block in blocks:
            try:
                cvimodel, n_ops = deploy(block)
            except Exception as e:
                print(f"{block}: deploy FAILED: {str(e)[-400:]}")
                rows[block] = {"error": str(e)[-1000:]}
                continue
            static = speed.mlir_metrics(BASE / f"final_bis_{block}.mlir")
            rows[block] = {"qtable_ops": n_ops, "bf16_ops": static["bf16_ops"], "cast_ops": static["cast_ops"],
                           "cvimodel": cvimodel.name}
            print(f"{block}: deployed, {n_ops} ops in qtable, {static['bf16_ops']} BF16 ops in final model")
            futures[block] = pool.submit(evaluate, block, cvimodel)
        for block, fut in futures.items():
            try:
                rows[block]["epe"] = fut.result()
            except Exception as e:
                rows[block]["error"] = str(e)[-1000:]
                print(f"{block}: emulator FAILED: {str(e)[-300:]}")

    if not args.no_device:
        for key, model in [("int8", BASE / f"{M}_int8.cvimodel"), ("full_refine", BASE / f"{M}_mixed.cvimodel")] + \
                          [(b, BASE / rows[b]["cvimodel"]) for b in blocks if "cvimodel" in rows[b]]:
            try:
                rows[key]["inference_ms"] = speed.device_bench(model, runs=3, count=30)["inference_ms_median"]
            except Exception as e:
                rows[key]["device_error"] = str(e)[-300:]
                print(f"{key}: camera timing failed: {str(e)[-200:]}")

    record = {"time": datetime.datetime.now().isoformat(timespec="seconds"), "tag": args.tag,
              "checkpoint": args.checkpoint, "git_commit": commit, "git_dirty": dirty,
              "docker_image": speed.DOCKER_IMAGE, "cali_method": eqexp.CALI_METHOD,
              "eval_pairs": int(len(idx)), "rows": rows}
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    datasets = list(rows["fp32"]["epe"])
    base_gap = {d: rows["int8"]["epe"][d]["epe_noc"] - rows["fp32"]["epe"][d]["epe_noc"] for d in datasets}
    print(f"\n{'BF16 block':18s} " + " ".join(f"{d:>13s}" for d in datasets) + "   recovered(clean)  bf16  cast      ms")
    for key, row in rows.items():
        if "epe" not in row:
            print(f"{key:18s} FAILED")
            continue
        cells = " ".join(f"{row['epe'][d]['epe_noc']:13.3f}" for d in datasets)
        d0 = datasets[0]
        rec = "" if key in ("fp32", "int8") else \
            f"{100 * (rows['int8']['epe'][d0]['epe_noc'] - row['epe'][d0]['epe_noc']) / base_gap[d0]:5.0f}%"
        ms = row.get("inference_ms")
        print(f"{key:18s} {cells}   {rec:>16s}  {row.get('bf16_ops', ''):>4}  {row.get('cast_ops', ''):>4}  "
              f"{('' if ms is None else f'{ms:.2f}'):>6s}")


if __name__ == "__main__":
    main()
