#!/usr/bin/env python3
"""How many INT8 levels each input of every Concat gets.

An INT8 Concat has one scale for all inputs, set by the Concat OUTPUT threshold
from the calibration table. An input with threshold t then spans
127 * min(t, t_out) / t_out of the 127 levels. Note that t_out is the
calibrated threshold of the concat output itself (a percentile over all its
channels), which is usually lower than the largest input threshold, so
127 * t_i / max_j(t_j) gives smaller numbers than this script.

    .venv/bin/python bench/concat_levels.py MODEL.onnx CALI_TABLE
    .venv/bin/python bench/concat_levels.py nanoflow_v1_step30000.onnx bench/work/_ranges/m_cali_table
"""
import sys

import onnx

# ONNX Concat output -> meaning, for the December export of v1.py
LABELS = {
    "/Concat": "update32, iteration 1", "/Concat_1": "update32, iteration 2",
    "/Concat_4": "corr16 + upsampled corr32",
    "/Concat_5": "update16, iteration 1", "/Concat_6": "update16, iteration 2",
    "/Concat_7": "refine_s16", "/Concat_9": "refine_s8", "/Concat_11": "refine_s4",
    "/Concat_13": "full_refine",
}


def role(tensor: str) -> str:
    if "corr16/ReduceSum" in tensor:
        return "corr16"
    if "corr32" in tensor or "Resize_1" in tensor:
        return "corr32"
    if "ctx" in tensor:
        return "context"
    if "pos_proj" in tensor:
        return "features f0/f1"
    if tensor == "img0":
        return "image"
    if "Concat_4" in tensor:
        return "corr16+corr32"
    return "flow"


def load_table(path):
    table = {}
    for line in open(path):
        parts = line.split()
        if line.startswith("#") or len(parts) < 4:
            continue
        try:
            table[parts[0]] = float(parts[1])
        except ValueError:
            pass
    return table


def main():
    onnx_path, table_path = sys.argv[1:3]
    table = load_table(table_path)
    graph = onnx.load(onnx_path).graph
    producer = {o: n for n in graph.node for o in n.output}
    constants = {i.name for i in graph.initializer}

    def threshold(tensor):
        if tensor in table:
            return table[tensor]
        node = producer.get(tensor)
        return table.get(f"{tensor}_{node.op_type}") if node is not None else None

    print(f"{'concat':28s} {'t_out':>8s}  group: threshold -> levels of 127")
    for node in graph.node:
        name = node.output[0].replace("_output_0", "")
        if node.op_type != "Concat" or name not in LABELS:
            continue
        t_out = threshold(node.output[0])
        groups = {}
        for inp in node.input:
            t = None if inp in constants else threshold(inp)
            if t is not None:
                groups[role(inp)] = max(groups.get(role(inp), 0.0), t)
        cells = "; ".join(f"{g}: {t:.3f} -> {127 * min(t, t_out) / t_out:.1f}" for g, t in groups.items())
        print(f"{LABELS[name]:28s} {t_out:8.3f}  {cells}")


if __name__ == "__main__":
    main()
