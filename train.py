#!/usr/bin/env python3
"""Train NPUFlow.

    python train.py --config configs/proxy.yaml --variant C3 --name C3-proxy
    python train.py --config configs/smoke.yaml --name smoke          # CPU check

A run is a list of stages (dataset mix, steps, learning rate). The full state
(model, optimizer, EMA, stage, step, RNG) is saved to runs/<name>/state.pt at
every validation, so a run interrupted at any point (Kaggle 12 h limit) simply
continues when started again with the same name. With --clearml the state and
the best weights are also stored as task artifacts and pulled back on resume.

Validation is the honest protocol: original frames, sliding window at the
training scale, EPE in original pixels over all valid pixels.
"""
import argparse
import math
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import ConcatDataset, DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data import color_jitter, make_dataset  # noqa: E402
from losses import multiscale_loss_masked  # noqa: E402
from model import EdgeFlowNet, ModelConfig  # noqa: E402

DEFAULTS = {
    "batch_size": 64, "num_workers": 8, "amp": True, "ema_decay": 0.9995, "warmup_steps": 125,
    "grad_clip": 1.0, "weight_decay": 1e-4, "log_interval": 100, "val_interval": 1000, "seed": 1337,
    "aug": "raft",           # "raft" | "december" (the recipe of the December training)
    "val_limit": 0,          # 0 = whole validation sets; N = first N pairs of each (smoke tests)
    "model": {},             # overrides of ModelConfig
    "stages": [],            # each: name, train [datasets], val [[dataset, split], ...], steps, lr_max, lr_min
}


# ----------------------------------------------------------------- utilities

class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {n: t.detach().clone() for n, t in self._tensors(model)}

    @staticmethod
    def _tensors(model):
        return [(n, t) for n, t in list(model.named_parameters()) + list(model.named_buffers())
                if t.dtype.is_floating_point]

    @torch.no_grad()
    def update(self, model):
        for n, t in self._tensors(model):
            self.shadow[n].mul_(self.decay).add_(t, alpha=1 - self.decay)

    @torch.no_grad()
    def copy_to(self, model):
        for n, t in self._tensors(model):
            t.copy_(self.shadow[n])


def plain(obj):
    """ClearML returns proxy containers; turn them back into ordinary dicts and lists."""
    if isinstance(obj, dict):
        return {k: plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    return obj


def lr_at(step, total, lr_max, lr_min, warmup):
    if step <= warmup:
        return lr_max * step / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + math.cos(math.pi * t))


def unwrap(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def rng_state():
    return {"py": random.getstate(), "np": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_state(state):
    random.setstate(state["py"])
    np.random.set_state(state["np"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        try:
            torch.cuda.set_rng_state_all(state["cuda"])
        except Exception:
            pass   # different number of GPUs than when the state was saved


# ---------------------------------------------------------------- evaluation

@torch.no_grad()
def predict_full(model, img0, img1, size_hw):
    """Flow for one original-resolution pair [3,H,W] uint8 -> [2,H,W] in original pixels.

    The frame is scaled by height to the model height (as in training) and covered
    with overlapping horizontal windows of the model width; predictions are blended.
    """
    h_m, w_m = size_hw
    device = next(model.parameters()).device
    _, h0, w0 = img0.shape
    scale = h_m / h0
    w_s = max(w_m, int(round(w0 * scale)))
    pair = torch.stack([img0, img1]).float().to(device) / 255.0
    pair = F.interpolate(pair, size=(h_m, w_s), mode="bilinear", align_corners=False)

    n = max(1, math.ceil((w_s - w_m) / (w_m * 0.5)) + 1) if w_s > w_m else 1
    offsets = [round(i * (w_s - w_m) / max(1, n - 1)) for i in range(n)]
    crops0 = torch.stack([pair[0, :, :, o:o + w_m] for o in offsets])
    crops1 = torch.stack([pair[1, :, :, o:o + w_m] for o in offsets])
    flows = model(crops0, crops1)

    # triangular blending weights so that window borders do not leave seams
    ramp = torch.minimum(torch.arange(1, w_m + 1, device=device), torch.arange(w_m, 0, -1, device=device)).float()
    acc = torch.zeros(2, h_m, w_s, device=device)
    wsum = torch.zeros(1, 1, w_s, device=device)
    for flow, o in zip(flows, offsets):
        acc[:, :, o:o + w_m] += flow * ramp
        wsum[:, :, o:o + w_m] += ramp
    flow_s = acc / wsum
    flow = F.interpolate(flow_s[None], size=(h0, w0), mode="bilinear", align_corners=False)[0]
    flow[0] *= w0 / w_s
    flow[1] *= h0 / h_m
    return flow


@torch.no_grad()
def evaluate(model, dataset, size_hw, limit=0):
    model.eval()
    device = next(model.parameters()).device
    sums = {"all": [0.0, 0], "noc": [0.0, 0], "out": [0.0, 0]}
    n = len(dataset) if not limit else min(limit, len(dataset))
    for i in range(n):
        img0, img1, gt, valid_noc, valid_all = dataset[i]
        flow = predict_full(model, img0, img1, size_hw)
        gt, valid_noc, valid_all = gt.to(device), valid_noc.to(device), valid_all.to(device)
        err = torch.sqrt(((flow - gt) ** 2).sum(0))
        mag = torch.sqrt((gt ** 2).sum(0))
        sums["all"][0] += float(err[valid_all].sum())
        sums["all"][1] += int(valid_all.sum())
        sums["noc"][0] += float(err[valid_noc].sum())
        sums["noc"][1] += int(valid_noc.sum())
        outlier = (err > 3.0) & (err > 0.05 * mag)             # KITTI Fl definition
        sums["out"][0] += float(outlier[valid_all].sum())
        sums["out"][1] += int(valid_all.sum())
    return {"epe_all": sums["all"][0] / max(1, sums["all"][1]),
            "epe_noc": sums["noc"][0] / max(1, sums["noc"][1]),
            "fl_all": 100.0 * sums["out"][0] / max(1, sums["out"][1]), "pairs": n}


# ------------------------------------------------------------------ training

def load_config(args):
    cfg = dict(DEFAULTS)
    cfg.update(yaml.safe_load(Path(args.config).read_text()) or {})
    if args.variant:
        variants = yaml.safe_load((ROOT / "bench" / "variants.yaml").read_text())
        cfg["model"] = {**(variants[args.variant] or {}), **cfg.get("model", {})}
        cfg["variant"] = args.variant
    for item in args.set or []:
        key, value = item.split("=", 1)
        cfg[key] = yaml.safe_load(value)
    cfg["name"] = args.name
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", required=True, help="run name; runs/<name>/ holds state and weights")
    ap.add_argument("--variant", default="", help="architecture variant from bench/variants.yaml")
    ap.add_argument("--set", action="append", help="override a top-level config key: key=value (YAML)")
    ap.add_argument("--max-hours", type=float, default=0, help="stop cleanly after this many hours (0 = no limit)")
    ap.add_argument("--clearml", action="store_true", help="log to ClearML and keep state as task artifacts")
    ap.add_argument("--project", default="NPUFlow/train")
    args = ap.parse_args()

    cfg = load_config(args)
    task = None
    if args.clearml:
        from clearml import Task
        task = Task.init(project_name=args.project, task_name=args.name, continue_last_task=True,
                         auto_connect_frameworks=False)
        cfg = plain(task.connect_configuration(cfg, name="train"))   # an agent run takes the stored config
    logger = task.get_logger() if task else None

    out_dir = ROOT / "runs" / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "state.pt"
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg["amp"]) and device.type == "cuda"

    mcfg = ModelConfig(**cfg["model"])
    size_hw = mcfg.image_size
    model = EdgeFlowNet(mcfg).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    eval_model = EdgeFlowNet(mcfg).to(device)
    print(f"run {cfg['name']} | device {device} x{max(1, torch.cuda.device_count())} | "
          f"params {unwrap(model).count_params() / 1e6:.3f}M | model {cfg['model']}")

    # ---- resume
    state = None
    if not state_path.exists() and task is not None and "train_state" in task.artifacts:
        local = task.artifacts["train_state"].get_local_copy()
        if local:
            state_path.write_bytes(Path(local).read_bytes())
            print("[resume] state restored from the ClearML task")
    if state_path.exists():
        state = torch.load(state_path, map_location=device, weights_only=False)
        print(f"[resume] stage {state['stage']} step {state['step']}")

    started = time.time()
    stopped_early = False
    history = state["history"] if state else []

    for stage_idx, stage in enumerate(cfg["stages"]):
        if state and stage_idx < state["stage"]:
            continue
        steps, lr_max, lr_min = int(stage["steps"]), float(stage["lr_max"]), float(stage["lr_min"])
        enc_factor = float(stage.get("encoder_lr_factor", 1.0))
        groups = [{"params": [p for n, p in unwrap(model).named_parameters() if n.startswith("enc.")], "scale": enc_factor},
                  {"params": [p for n, p in unwrap(model).named_parameters() if not n.startswith("enc.")], "scale": 1.0}]
        optimizer = torch.optim.AdamW(groups, lr=lr_max, weight_decay=cfg["weight_decay"])
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        step, best = 0, float("inf")

        if state and stage_idx == state["stage"]:
            unwrap(model).load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            if state.get("scaler") and use_amp:
                scaler.load_state_dict(state["scaler"])
            ema = EMA(unwrap(model), cfg["ema_decay"])
            ema.shadow = state["ema"]
            step, best = state["step"], state["best"]
            set_rng_state(state["rng"])
            state = None
        else:
            ema = EMA(unwrap(model), cfg["ema_decay"])    # each stage starts its own average
            state = None
        if step >= steps:
            continue

        train_sets = [make_dataset(n, "all" if n in ("chairs", "things") else "train", size_hw, augment=True,
                                   aug_recipe=cfg["aug"]) for n in stage["train"]]
        train_ds = train_sets[0] if len(train_sets) == 1 else ConcatDataset(train_sets)
        loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
                            num_workers=cfg["num_workers"], pin_memory=device.type == "cuda",
                            persistent_workers=cfg["num_workers"] > 0)
        val_sets = [(f"{n}/{split}", make_dataset(n, split, size_hw, augment=False)) for n, split in stage["val"]]
        print(f"[{stage['name']}] {len(train_ds)} training pairs, {len(loader)} it/epoch, "
              f"val {[(n, len(d)) for n, d in val_sets]}, steps {step}/{steps}")

        def save_state():
            torch.save({"stage": stage_idx, "step": step, "best": best, "model": unwrap(model).state_dict(),
                        "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict() if use_amp else None,
                        "ema": ema.shadow, "rng": rng_state(), "history": history, "config": cfg}, state_path)
            if task is not None:
                task.upload_artifact("train_state", artifact_object=str(state_path))

        def validate():
            nonlocal best
            eval_model.load_state_dict(unwrap(model).state_dict())
            ema.copy_to(eval_model)
            row = {"stage": stage["name"], "step": step}
            for name, ds in val_sets:
                res = evaluate(eval_model, ds, size_hw, cfg["val_limit"])
                row[name] = res
                if logger:
                    gstep = sum(int(s["steps"]) for s in cfg["stages"][:stage_idx]) + step
                    for k in ("epe_all", "epe_noc", "fl_all"):
                        logger.report_scalar(f"val {k}", name, res[k], gstep)
            score = float(np.mean([row[n]["epe_all"] for n, _ in val_sets]))
            row["score"] = score
            history.append(row)
            print(f"[VAL {stage['name']}] step {step}: mean EPE {score:.3f} | "
                  + " | ".join(f"{n} all {row[n]['epe_all']:.3f} noc {row[n]['epe_noc']:.3f} Fl {row[n]['fl_all']:.1f}%"
                               for n, _ in val_sets), flush=True)
            torch.save(eval_model.state_dict(), out_dir / f"{stage['name']}_ema_last.pth")
            if score < best:
                best = score
                torch.save(eval_model.state_dict(), out_dir / f"{stage['name']}_ema_best.pth")
                if task is not None:
                    task.upload_artifact(f"{stage['name']}_ema_best",
                                         artifact_object=str(out_dir / f"{stage['name']}_ema_best.pth"))
            model.train()

        model.train()
        t_log, n_log = time.time(), 0
        while step < steps and not stopped_early:
            for img0, img1, flow_gt, valid in loader:
                img0, img1 = img0.to(device, non_blocking=True), img1.to(device, non_blocking=True)
                flow_gt, valid = flow_gt.to(device, non_blocking=True), valid.to(device, non_blocking=True)
                if cfg["aug"] == "raft":
                    img0, img1 = color_jitter(img0, img1)
                lr = lr_at(step, steps, lr_max, lr_min, cfg["warmup_steps"])
                for g in optimizer.param_groups:
                    g["lr"] = lr * g["scale"]
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    preds, _ = model(img0, img1)
                    loss, stats = multiscale_loss_masked(preds, flow_gt, img0, valid_mask=valid)
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if cfg["grad_clip"]:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                ema.update(unwrap(model))
                step += 1
                n_log += 1

                if step % cfg["log_interval"] == 0:
                    rate = n_log * cfg["batch_size"] / (time.time() - t_log)
                    t_log, n_log = time.time(), 0
                    print(f"[{stage['name']}] step {step}/{steps} | lr {lr:.2e} | loss {stats['loss']:.4f} | "
                          f"epe {stats['epe']:.3f} | {rate:.1f} pairs/s | {(time.time() - started) / 3600:.2f} h",
                          flush=True)
                    if logger:
                        gstep = sum(int(s["steps"]) for s in cfg["stages"][:stage_idx]) + step
                        logger.report_scalar("train", "loss", stats["loss"], gstep)
                        logger.report_scalar("train", "epe", stats["epe"], gstep)
                        logger.report_scalar("train", "lr", lr, gstep)
                        logger.report_scalar("speed", "pairs_per_s", rate, gstep)
                if step % cfg["val_interval"] == 0 or step == steps:
                    validate()
                    save_state()
                    t_log, n_log = time.time(), 0
                if args.max_hours and (time.time() - started) / 3600 >= args.max_hours and step < steps:
                    save_state()
                    stopped_early = True
                if step >= steps or stopped_early:
                    break
        if stopped_early:
            break
        torch.save(unwrap(model).state_dict(), out_dir / f"{stage['name']}_final.pth")
        # hand over to the next stage: mark this one finished in the state file
        step = steps
        torch.save({"stage": stage_idx + 1, "step": 0, "best": float("inf"), "model": unwrap(model).state_dict(),
                    "optimizer": optimizer.state_dict(), "scaler": None, "ema": ema.shadow, "rng": rng_state(),
                    "history": history, "config": cfg}, state_path)
        if task is not None:
            task.upload_artifact("train_state", artifact_object=str(state_path))

    (out_dir / "history.yaml").write_text(yaml.safe_dump(history, sort_keys=False))
    if stopped_early:
        print(f"time budget of {args.max_hours} h reached; state saved, start again with the same --name to continue")
        if task is not None:
            task.mark_stopped(status_message="time budget reached; re-enqueue to continue")
        sys.exit(3)
    print("training finished")


if __name__ == "__main__":
    main()
