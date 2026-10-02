# NPUFlow

Lightweight optical flow network for low-cost NPUs (developed on MaixCAM, Sophgo CV181x TPU, INT8).

## Layout

| Path | What |
|---|---|
| `model.py` | network with architecture switches (`ModelConfig`) |
| `data.py` | datasets (FlyingChairs, Sintel, KITTI-2015), augmentation |
| `losses.py` | multi-scale training loss |
| `train.py` | staged training with resume, honest validation, optional ClearML logging |
| `configs/` | training protocols (`proxy.yaml`, `smoke.yaml`) |
| `bench/variants.yaml` | architecture variants |
| `bench/speed.py` | on-device speed of variants, no training needed |
| `bench/eqexp.py`, `emu_eval.py` | INT8 accuracy on the x86 TPU emulator |
| `bench/calib_sweep.py`, `hybrid_calib.py` | calibration method studies |
| `bench/bf16_blocks.py`, `stage_diag.py`, `concat_levels.py` | where accuracy is lost, what each stage does |
| `docker/tpumlir.Dockerfile` | pinned TPU-MLIR toolchain |
| `results/` | append-only experiment records |

## Training

```bash
python train.py --config configs/proxy.yaml --variant C3_1d_ps_nopos_relu --name C3-proxy --clearml --max-hours 11
```

Dataset locations come from `NPUFLOW_CHAIRS`, `NPUFLOW_THINGS`, `NPUFLOW_SINTEL`, `NPUFLOW_KITTI`.
A stopped run continues when started again with the same `--name`.

Validation: original frames, sliding window at the training scale, EPE in original
pixels over all valid pixels.

## Conversion for CV181x

`convert_cv181x.sh` (TPU-MLIR, KL calibration). Judge quantization by EPE on the
emulator, not by tensor similarity.
