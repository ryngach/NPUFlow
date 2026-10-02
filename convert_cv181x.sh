#!/bin/bash
set -xEeuo pipefail
trap 'st=$?; echo "Error at line $LINENO: \"$BASH_COMMAND\" (exit $st)" >&2; exit $st' ERR

# Usage: SINTEL_PATH=/path/to/sintel ./convert_cv181x.sh
# Default Sintel path: datasets/MPI-Sintel-complete

MODEL_NAME="nanoflow_v1_step30000"
INPUT_HEIGHT=256
INPUT_WIDTH=320
SINTEL_PATH="${SINTEL_PATH:-datasets/MPI-Sintel-complete}"

echo "=== Check ONNX ==="
[ -f "${MODEL_NAME}.onnx" ] || { echo "${MODEL_NAME}.onnx not found"; exit 1; }
echo "OK: ${MODEL_NAME}.onnx"

echo "=== Make test images ==="
python3 - <<PY
import numpy as np, cv2
h,w=${INPUT_HEIGHT},${INPUT_WIDTH}
np.random.seed(42)
base = np.zeros((h,w,3),np.uint8)
for x in range(0,w,20): base[:,x:x+2] = (255,0,0)
for y in range(0,h,20): base[y:y+2,:] = (0,255,0)
noise = np.random.randint(0,50,(h,w,3),dtype=np.uint8)
img0 = cv2.add(base, noise)
M = np.float32([[1,0,2],[0,1,1]])
img1 = cv2.warpAffine(img0, M, (w,h), borderValue=(0,0,0))
cv2.imwrite("dog0.jpg", img0)
cv2.imwrite("dog1.jpg", img1)
print("Created dog0.jpg & dog1.jpg")
PY

SINTEL_PATH="${SINTEL_PATH:-datasets/MPI-Sintel-complete}"

echo "=== Build calibration pairs from Sintel dataset ==="
echo "Using Sintel dataset: ${SINTEL_PATH}"
mkdir -p images
# Remove calibration frames left over from previous runs
rm -f images/cal_*.jpg

PAIR_LIST="calibration_pairs.txt"
NUM_CALIB_PAIRS=100

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 <<PY
import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, "${SCRIPT_DIR}")
from utils import resize_img

sintel_root = Path("${SINTEL_PATH}")
target_size = (${INPUT_HEIGHT}, ${INPUT_WIDTH})
count = 0

# Sintel structure: training/clean/<scene>/frame_*.png
clean_dir = sintel_root / 'training' / 'clean'

if not clean_dir.exists():
    print(f"Warning: {clean_dir} not found")
    raise RuntimeError(f"Sintel clean directory not found: {clean_dir}")

# Each calibration pair is two ADJACENT frames (i, i+1) of the same scene,
# with pair start positions spread evenly across every scene.
num_pairs = ${NUM_CALIB_PAIRS}
scene_dirs = sorted([d for d in clean_dir.iterdir() if d.is_dir()])
pairs_per_scene = -(-num_pairs // len(scene_dirs)) if scene_dirs else 0

pairs = []
for scene_dir in scene_dirs:
    scene_images = sorted(scene_dir.glob('frame_*.png'))
    n_starts = len(scene_images) - 1
    if n_starts < 1:
        continue
    k = min(pairs_per_scene, n_starts)
    for j in range(k):
        i = j * n_starts // k
        pairs.append((scene_images[i], scene_images[i + 1]))

# Drop evenly across scenes if there are more pairs than requested
if len(pairs) > num_pairs:
    pairs = [pairs[i * len(pairs) // num_pairs] for i in range(num_pairs)]

print(f"Selected {len(pairs)} adjacent-frame pairs from {len(scene_dirs)} Sintel scenes")

def save_frame(src, dst):
    img = cv2.imread(str(src))
    if img is None:
        return False
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # resize_img handles resizing and center cropping (default x_offset=None)
    img_resized = resize_img(img_rgb, target_size)
    cv2.imwrite(dst, cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))
    return True

with open("${PAIR_LIST}", "w") as pair_list:
    for path_a, path_b in pairs:
        name_a = f"images/cal_{count:03d}_a.jpg"
        name_b = f"images/cal_{count:03d}_b.jpg"
        if save_frame(path_a, name_a) and save_frame(path_b, name_b):
            pair_list.write(f"{name_a}, {name_b}\n")
            count += 1

print(f"{count} calibration pairs extracted and resized from Sintel dataset")

if count == 0:
    raise RuntimeError("No calibration pairs were created! Check Sintel dataset path.")
PY

# Verify pairs were created
if [ ! -s "${PAIR_LIST}" ]; then
    echo "ERROR: Calibration pairs were not created!"
    echo "Please check:"
    echo "  1. Sintel dataset path: ${SINTEL_PATH}"
    echo "  2. utils.py exists in: ${SCRIPT_DIR}"
    exit 1
fi

NUM_CALIB_PAIRS=$(wc -l < "${PAIR_LIST}")
echo "Pairs list: ${PAIR_LIST} (lines: ${NUM_CALIB_PAIRS})"

echo "=== Discover ONNX output names ==="
OUTPUT_NAMES=$(python3 - <<PY
import onnx
m=onnx.load("${MODEL_NAME}.onnx")
print(",".join([o.name for o in m.graph.output]))
PY
)
echo "Outputs: ${OUTPUT_NAMES}"

echo "=== Step 1: ONNX -> MLIR (FP32) ==="
model_transform.py \
  --model_name ${MODEL_NAME} \
  --model_def ${MODEL_NAME}.onnx \
  --input_shapes [[1,3,${INPUT_HEIGHT},${INPUT_WIDTH}],[1,3,${INPUT_HEIGHT},${INPUT_WIDTH}]] \
  --channel_format nchw \
  --output_names "${OUTPUT_NAMES}" \
  --test_input "dog0.jpg,dog1.jpg" \
  --test_result ${MODEL_NAME}_top_outputs.npz \
  --mlir ${MODEL_NAME}.mlir \
  --pixel_format rgb \
  --keep_aspect_ratio \
  --mean "0,0,0" \
  --scale "0.00392156862745098,0.00392156862745098,0.00392156862745098" \
  --tolerance '0.9984,0.9984' 
echo "OK: MLIR ${MODEL_NAME}.mlir"

echo "=== Step 2: BF16 Deploy (CV181x) ==="
model_deploy.py \
  --mlir ${MODEL_NAME}.mlir \
  --quantize BF16 \
  --processor cv181x \
  --tolerance '0.9984,0.9984' \
  --model ${MODEL_NAME}_bf16.cvimodel 
echo "OK: BF16 ${MODEL_NAME}_bf16.cvimodel"

echo "=== Step 2.1: BF16 Deploy with Fuse Preprocess ==="
model_deploy.py \
  --mlir ${MODEL_NAME}.mlir \
  --quantize BF16 \
  --processor cv181x \
  --tolerance '0.9984,0.9984' \
  --fuse_preprocess \
  --model ${MODEL_NAME}_bf16_fuse.cvimodel 
echo "OK: BF16 Fuse ${MODEL_NAME}_bf16_fuse.cvimodel"

echo "=== Step 3: INT8 Calibration & Mixed Precision Search ==="
export OMP_NUM_THREADS=$(($(nproc)/2))
export MKL_NUM_THREADS=$(($(nproc)/2))
export OPENBLAS_NUM_THREADS=$(($(nproc)/2))
# Threshold method chosen by INT8 EPE on the emulator (bench/calib_sweep.py,
# results/calib_sweep.jsonl, 2026-10-01): use_kl closes ~54% of the INT8 loss
# on Sintel compared with use_percentile9999; aciq_laplace is on par.
# Do NOT add --we: it looks fine by tensor cosine but breaks the model by EPE
# (Sintel clean 2.47 -> 9.14). --bc has no effect and fails in tpu_mlir 1.25.
run_calibration.py \
  ${MODEL_NAME}.mlir \
  --data_list ${PAIR_LIST} \
  --input_num ${NUM_CALIB_PAIRS} \
  --inference_num 30 \
  --cali_method use_kl \
  --chip cv181x \
  --fp_type BF16 \
  --expected_cos 0.9984 \
  --quantize_table ${MODEL_NAME}_qtable.txt \
  -o "${MODEL_NAME}_cali_table"
echo "OK: calibration table ${MODEL_NAME}_cali_table & auto qtable"

# echo "=== Analyze calibration table ==="
# python3 <<PY
# import re
# with open("${MODEL_NAME}_cali_table", 'r') as f:
#     lines = f.readlines()
    
# print(f"Total calibrated tensors: {len([l for l in lines if l.strip() and not l.startswith('#')])}")

# # Find tensors with extreme or unusual ranges
# suspicious_count = 0
# zero_threshold_layers = []
# for line in lines:
#     if line.strip() and not line.startswith('#'):
#         match = re.search(r'(\S+)\s+([-\d.e+]+)\s+([-\d.e+]+)', line)
#         if match:
#             name, min_val, max_val = match.groups()
#             min_v, max_v = float(min_val), float(max_val)
#             range_val = max_v - min_v
#             if range_val < 0.01 or abs(min_v) > 100 or abs(max_v) > 100:
#                 print(f"  Suspicious: {name} [{min_v:.6f}, {max_v:.6f}]")
#                 suspicious_count += 1
#             if abs(max_v) < 1e-6:  # Zero threshold
#                 zero_threshold_layers.append(name)

# if zero_threshold_layers:
#     print(f"\n⚠️  Found {len(zero_threshold_layers)} layers with zero threshold:")
#     for layer in zero_threshold_layers[:5]:
#         print(f"    • {layer}")
#     if len(zero_threshold_layers) > 5:
#         print(f"    ... and {len(zero_threshold_layers) - 5} more")
#     print("\n  → These layers will be automatically handled by the compiler")
# else:
#     print("\n✓ No zero-threshold layers found")

# print(f"\nTotal suspicious tensors: {suspicious_count}")
# PY

# echo "=== Create quantize table for mixed precision ==="
# # Extract operation names from final full-resolution layers
# # Strategy: Keep full-resolution refiner (256x320) in BF16 for quality
# python3 <<PY
# import re

# # Read MLIR
# with open('${MODEL_NAME}.mlir', 'r') as f:
#     lines = f.readlines()

# # 1. Build loc_id -> op_name map
# loc_map = {}
# for line in lines:
#     # Match: #loc123 = loc("op_name")
#     match = re.search(r'#(loc\d+)\s*=\s*loc\("([^"]+)"\)', line)
#     if match:
#         loc_id = match.group(1)
#         op_name = match.group(2)
#         loc_map[loc_id] = op_name

# print(f"Parsed {len(loc_map)} location mappings")

# # 2. Find operations with 256x320 resolution in the last part
# qtable_ops = []
# # Start scanning from line 990 (approximate start of full res block)
# for i, line in enumerate(lines[990:], start=990):
#     # Match operation definitions with 256x320 resolution
#     if '256x320' in line and ('top.Conv' in line or 'top.Add' in line or 'top.LeakyRelu' in line):
#         # Extract the location ID
#         match = re.search(r'loc\(#(loc\d+)\)', line)
#         if match:
#             loc_id = match.group(1)
#             if loc_id in loc_map:
#                 op_name = loc_map[loc_id]
#                 qtable_ops.append(op_name)
#             else:
#                 print(f"Warning: Location {loc_id} not found in map")

# print(f"Found {len(qtable_ops)} full-resolution operations to keep in BF16")

# # Write qtable
# with open('${MODEL_NAME}_qtable.txt', 'w') as f:
#     f.write("# Full resolution refiner - keep in BF16 for quality\n")
#     f.write("# All operations processing at 256x320 resolution\n")
#     for op in qtable_ops:
#         f.write(f"{op} BF16\n")

# print(f"Created qtable with {len(qtable_ops)} BF16 operations")
# with open('${MODEL_NAME}_qtable.txt', 'r') as f:
#     print(f.read())
# PY
# echo "Created quantize table: ${MODEL_NAME}_qtable.txt"

# echo "=== Create Optimized QTable from Auto Search ==="
# # Strategy: Keep top 10% most critical layers + mandatory SE layers
# if [ -f "Search_Qtable" ]; then
#     python3 << 'PYFILTER'
# import re

# results = []
# with open('Search_Qtable', 'r') as f:
#     for line in f:
#         if 'best_cos_loss' in line:
#             layer_match = re.search(r'layer (/[^,]+)', line)
#             loss_match = re.search(r'best_cos_loss = ([\d.]+)', line)
#             if layer_match and loss_match:
#                 layer = layer_match.group(1)
#                 layer = re.sub(r'_[A-Z][a-zA-Z]+$', '', layer)
#                 loss = float(loss_match.group(1))
#                 results.append((loss, layer))

# results.sort(reverse=True)

# # Keep top 10% most critical (minimum BF16 for speed)
# threshold_idx = int(len(results) * 0.10)

# # Get top layers
# top_layers = set(layer for loss, layer in results[:threshold_idx])

# # Add mandatory SE layers (have zero threshold issues)
# mandatory_se = {
#     "/enc/se_s16/act/Relu_output_0",
#     "/enc/se_s16/act_1/Relu_output_0"
# }

# final_layers = top_layers | mandatory_se

# with open('${MODEL_NAME}_qtable_filtered.txt', 'w') as f:
#     for loss, layer in results[:threshold_idx]:
#         f.write(f"{layer} BF16\n")
#     # Add SE layers if not already included
#     for se_layer in mandatory_se:
#         if se_layer not in top_layers:
#             f.write(f"{se_layer} BF16\n")

# threshold_loss = results[threshold_idx-1][0] if threshold_idx > 0 else 0
# threshold_sim = 1.0 - threshold_loss
# print(f"Optimized qtable: {len(final_layers)}/{len(results)} layers in BF16 (top 10% + SE)")
# print(f"Similarity threshold: {threshold_sim:.6f} ({threshold_sim*100:.4f}%)")
# PYFILTER
# else
#     echo "Warning: Search_Qtable not found, skipping filtering"
# fi

# echo "=== Merge QTables ==="
# # Use filtered auto qtable instead of full auto qtable
# if [ -f "${MODEL_NAME}_qtable_filtered.txt" ]; then
#     echo "Using filtered auto qtable (top 15% most critical layers)..."
#     cp ${MODEL_NAME}_qtable_filtered.txt ${MODEL_NAME}_qtable_final.txt
# elif [ -f "${MODEL_NAME}_qtable_auto.txt" ]; then
#     echo "Filtered qtable not found, using full auto qtable..."
#     cp ${MODEL_NAME}_qtable_auto.txt ${MODEL_NAME}_qtable_final.txt
# else
#     echo "Auto qtable not found, using manual only."
#     cp ${MODEL_NAME}_qtable.txt ${MODEL_NAME}_qtable_final.txt
# fi
# echo "Final QTable: $(wc -l < ${MODEL_NAME}_qtable_final.txt) layers in BF16"

echo "=== Step 4: INT8 Deploy with Fuse Preprocess (Mixed Precision) ==="
model_deploy.py \
  --mlir ${MODEL_NAME}.mlir \
  --quantize INT8 \
  --processor cv181x \
  --tolerance '0.9984,0.9984' \
  --calibration_table ${MODEL_NAME}_cali_table \
  --model ${MODEL_NAME}_int8_fuse.cvimodel \
  --fuse_preprocess \
  --quant_input \
  --matmul_perchannel \
  --quantize_table ${MODEL_NAME}_qtable.txt
echo "OK: INT8 Fuse ${MODEL_NAME}_int8_fuse.cvimodel (mixed BF16/INT8)"

echo "=== Step 4.1: INT8 Deploy (Mixed Precision) ==="
model_deploy.py \
  --mlir ${MODEL_NAME}.mlir \
  --quantize INT8 \
  --quant_input \
  --processor cv181x \
  --tolerance '0.9984,0.9984' \
  --calibration_table ${MODEL_NAME}_cali_table \
  --model ${MODEL_NAME}_int8.cvimodel \
  --matmul_perchannel \
  --quantize_table ${MODEL_NAME}_qtable.txt
echo "OK: INT8 ${MODEL_NAME}_int8.cvimodel (mixed BF16/INT8)"


echo "=== Summary ==="
ls -l ${MODEL_NAME}.mlir ${MODEL_NAME}_*.cvimodel ${MODEL_NAME}_cali_table || true
echo "Done."