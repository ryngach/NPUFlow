"""Datasets and augmentations.

Training samples: RAFT-style augmentation (FlowAugmentor) around the "fit"
scale, at which the frame height equals the model height; the December recipe
is kept as an option for reproducing the old baseline.

Evaluation samples keep the ORIGINAL frames and ground truth; the model is run
by `predict_full` in train.py as a sliding window at the training scale, and
EPE is measured in original pixels over all valid pixels. That is the number
comparable with the literature.

Dataset roots come from the DATA dict (environment variables NPUFLOW_CHAIRS,
NPUFLOW_THINGS, NPUFLOW_SINTEL, NPUFLOW_KITTI, with local defaults).
"""
import math
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

from utils import read_flo, read_kitti_png_flow, resize_flow, resize_img, resize_mask_nearest

ROOT = Path(__file__).resolve().parent


def _find(env_name: str, marker: str, local: Path) -> str:
    """Dataset directory: the environment variable, else the local copy, else a Kaggle input.

    `marker` is a file path relative to the wanted directory. On Kaggle the datasets are mounted
    under /kaggle/input/<name>/..., a few levels deep depending on how the archive was packed;
    only fixed depths are tried, because a recursive search over the mounts takes minutes.
    """
    if os.environ.get(env_name):
        return os.environ[env_name]
    if (local / marker).exists() or not os.path.isdir("/kaggle/input"):
        return str(local)
    level = ["/kaggle/input"]
    for _ in range(6):
        nxt = []
        for d in level:
            names = sorted(os.listdir(d))
            if len(names) > 64:          # a folder of data files, not a place to look for dataset roots
                continue
            nxt += [os.path.join(d, n) for n in names if os.path.isdir(os.path.join(d, n))]
        level = nxt
        for d in level:
            if os.path.exists(os.path.join(d, marker)):
                return d
    return str(local)


DATA = {
    "chairs": _find("NPUFLOW_CHAIRS", "00001_img1.ppm", ROOT / "datasets" / "FlyingChairs_release" / "data"),
    "things": os.environ.get("NPUFLOW_THINGS", str(ROOT / "datasets" / "FlyingThings3D")),
    "sintel": _find("NPUFLOW_SINTEL", "training/clean/alley_1/frame_0001.png", ROOT / "datasets" / "MPI-Sintel-complete"),
    "kitti": _find("NPUFLOW_KITTI", "training/flow_occ/000000_10.png", ROOT / "datasets" / "data_scene_flow"),
}

# Sintel validation scenes for fine-tuning: whole scenes, so no frame of a validation
# scene is ever seen in training. Chosen to span small and large motion.
SINTEL_VAL_SCENES = ("ambush_2", "bamboo_1", "market_6", "temple_2")
SPLIT_SEED = 1337


def _apply_color_aug(t: torch.Tensor) -> torch.Tensor:
    b = random.uniform(0.9, 1.1)
    c = random.uniform(0.9, 1.1)
    s = random.uniform(0.9, 1.1)
    h = random.uniform(-0.05, 0.05)
    t = TF.adjust_brightness(t, b)
    t = TF.adjust_contrast(t, c)
    t = TF.adjust_saturation(t, s)
    t = TF.adjust_hue(t, h)
    return t


def _apply_motion_blur(t: torch.Tensor) -> torch.Tensor:
    """
    Motion blur optimized for 320×256 resolution
    10% probability, kernel 3-5 pixels
    """
    if random.random() > 0.9:
        # Optimal: 3 or 5 pixels for 320×256
        kernel_size = random.choice([3, 5])
        angle = random.uniform(0, 360)
        
        kernel = torch.zeros(kernel_size, kernel_size)
        center = kernel_size // 2
        angle_rad = math.radians(angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        for i in range(kernel_size):
            offset = i - center
            x = int(round(center + offset * cos_a))
            y = int(round(center + offset * sin_a))
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[y, x] = 1.0
        
        kernel = kernel / (kernel.sum() + 1e-6)
        kernel_3ch = kernel.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
        kernel_3ch = kernel_3ch.to(t.device)
        
        t_4d = t.unsqueeze(0)
        padding = kernel_size // 2
        t_blur = F.conv2d(t_4d, kernel_3ch, padding=padding, groups=3)
        
        return t_blur.squeeze(0).clamp(0, 1)
    return t


def _apply_random_occlusion(t: torch.Tensor) -> torch.Tensor:
    """50% probability"""
    if random.random() > 0.5:
        C, H, W = t.shape
        n_occ = random.randint(1, 2)
        for _ in range(n_occ):
            occ_h = random.randint(int(H * 0.05), int(H * 0.15))
            occ_w = random.randint(int(W * 0.05), int(W * 0.15))
            y = random.randint(0, H - occ_h)
            x = random.randint(0, W - occ_w)
            fill_val = random.uniform(0.0, 0.5)
            t[:, y:y + occ_h, x:x + occ_w] = fill_val
    return t


def _get_inverse_affine_matrix(
    center: List[float],
    angle: float,
    translate: List[float],
    scale: float,
    shear: List[float],
) -> List[float]:

    import math
    
    # Convert degrees to radians
    angle = math.radians(angle)
    shear_x = math.radians(shear[0])
    shear_y = math.radians(shear[1]) if len(shear) > 1 else 0.0

    # Calculate components
    cx, cy = center
    tx, ty = translate

    # RSS
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    cos_shear_y = math.cos(shear_y)
    sin_shear_y = math.sin(shear_y)
    tan_shear_x = math.tan(shear_x)

    a = scale * (cos_angle - sin_shear_y * sin_angle) / cos_shear_y
    b = scale * (-cos_angle * tan_shear_x / cos_shear_y - sin_angle)
    c = scale * (sin_angle + sin_shear_y * cos_angle) / cos_shear_y
    d = scale * (-sin_angle * tan_shear_x / cos_shear_y + cos_angle)

    matrix = [
        a, b, -a * cx - b * cy + cx + tx,
        c, d, -c * cx - d * cy + cy + ty
    ]
    
    return matrix

def _apply_geo_aug(t0: torch.Tensor, t1: torch.Tensor, tf: torch.Tensor,
                   mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

    device = t0.device
    dtype = t0.dtype
    _, H, W = t0.shape

    if random.random() < 0.5:
        return t0, t1, tf, mask

    angle = random.uniform(-5, 5)
    scale = random.uniform(0.95, 1.05)
    shear = [random.uniform(-5.0, 5.0), 0.0]
    translate = (random.uniform(-0.02, 0.02) * W, random.uniform(-0.02, 0.02) * H)

    params_dict = {'angle': angle, 'scale': scale, 'shear': shear, 'translate': translate}

    t0_aug = TF.affine(t0, **params_dict, interpolation=TF.InterpolationMode.BILINEAR, fill=0).clamp(0, 1)
    t1_aug = TF.affine(t1, **params_dict, interpolation=TF.InterpolationMode.BILINEAR, fill=0).clamp(0, 1)

    if mask is not None:
        mask_aug = TF.affine(mask, **params_dict, interpolation=TF.InterpolationMode.NEAREST, fill=0)
    else:
        mask_aug = None

    center = (W * 0.5, H * 0.5)
    matrix = _get_inverse_affine_matrix(center, angle, translate, scale, shear)
    matrix = torch.tensor(matrix, dtype=dtype, device=device).reshape(2, 3)

    # 1. Interpolate flow to new coordinates
    tf_interp = TF.affine(tf, **params_dict, interpolation=TF.InterpolationMode.BILINEAR, fill=0)

    # 2. Apply rotation + scale to flow vectors
    rot_scale_matrix = matrix[:, :2].float()

    # Transform flow [2, H, W] -> [H*W, 2]
    tf_flat = tf_interp.permute(1, 2, 0).reshape(-1, 2).float()

    # Apply rot+scale to flow vectors
    tf_transformed = torch.matmul(tf_flat, rot_scale_matrix.T)

    # Return to [2, H, W]
    tf_aug = tf_transformed.reshape(H, W, 2).permute(2, 0, 1).to(dtype=dtype)
    tf_aug = tf_aug.clamp(-1000, 1000)

    return t0_aug, t1_aug, tf_aug, mask_aug



def _random_x_offset(h0: int, w0: int, size_hw: Tuple[int, int]) -> Optional[int]:
    """Random horizontal crop offset for resize_img/resize_flow/resize_mask_nearest.

    Returns None (center crop / pad) when the scaled image is not wider than the target.
    """
    target_h, target_w = size_hw
    new_w = int(round(w0 * target_h / h0))
    if new_w <= target_w:
        return None
    return random.randint(0, new_w - target_w)


class FlowAugmentor:
    """RAFT-style augmentation adapted to a small fixed input.

    Geometry: the frame is scaled so that its height fits the crop ("fit" scale,
    the same scale evaluation uses), then with probability `spatial_prob` scaled
    up further by 2^U(0, max_scale_log2) with an optional anisotropic stretch;
    a random crop of the model size is taken in both directions. Scaling below
    the fit scale is impossible without padding, so it is not done.
    Photometry: random rectangles erased in the SECOND frame only, to imitate
    occlusions. Colour jitter is applied later on the GPU (color_jitter below),
    because on the CPU it costs more than everything else together.
    Sparse ground truth (KITTI) is resized by moving valid points, not by
    interpolation, and gets no stretch.
    """

    def __init__(self, size_hw, max_scale_log2=0.5, spatial_prob=0.5, stretch_prob=0.5, max_stretch=0.15,
                 hflip_prob=0.5, vflip_prob=0.1, eraser_prob=0.5, sparse=False):
        self.size_hw = tuple(size_hw)
        self.max_scale_log2, self.spatial_prob = max_scale_log2, spatial_prob
        self.stretch_prob, self.max_stretch = (0.0, 0.0) if sparse else (stretch_prob, max_stretch)
        self.hflip_prob, self.vflip_prob = hflip_prob, (0.0 if sparse else vflip_prob)
        self.eraser_prob = eraser_prob
        self.sparse = sparse

    def _eraser(self, img1):
        if random.random() < self.eraser_prob:
            h, w = img1.shape[:2]
            mean = img1.reshape(-1, 3).mean(0)
            for _ in range(random.randint(1, 2)):
                dx, dy = random.randint(w // 16, w // 4), random.randint(h // 16, h // 4)
                x0, y0 = random.randint(0, w - dx), random.randint(0, h - dy)
                img1[y0:y0 + dy, x0:x0 + dx] = mean
        return img1

    @staticmethod
    def _resize_sparse(flow, valid, sx, sy, out_hw):
        """Move every valid ground-truth point to its new position (no interpolation)."""
        h, w = flow.shape[:2]
        oh, ow = out_hw
        ys, xs = np.nonzero(valid)
        nx = np.round(xs * sx).astype(np.int64)
        ny = np.round(ys * sy).astype(np.int64)
        keep = (nx >= 0) & (nx < ow) & (ny >= 0) & (ny < oh)
        out_flow = np.zeros((oh, ow, 2), np.float32)
        out_valid = np.zeros((oh, ow), bool)
        out_flow[ny[keep], nx[keep]] = flow[ys[keep], xs[keep]] * np.array([sx, sy], np.float32)
        out_valid[ny[keep], nx[keep]] = True
        return out_flow, out_valid

    def __call__(self, img0, img1, flow, valid):
        """img*: HWC uint8; flow: [H,W,2] float32; valid: [H,W] bool. Returns the same, cropped to size_hw."""
        ch, cw = self.size_hw
        h, w = img0.shape[:2]
        fit = max(ch / h, cw / w)
        sx = sy = fit
        if random.random() < self.spatial_prob:
            sx = sy = fit * 2 ** random.uniform(0.0, self.max_scale_log2)
            if random.random() < self.stretch_prob:
                sx *= 2 ** random.uniform(-self.max_stretch, self.max_stretch)
                sy *= 2 ** random.uniform(-self.max_stretch, self.max_stretch)
        sx, sy = max(sx, cw / w), max(sy, ch / h)
        nh, nw = max(ch, int(round(h * sy))), max(cw, int(round(w * sx)))
        sx, sy = nw / w, nh / h

        img0 = cv2.resize(img0, (nw, nh), interpolation=cv2.INTER_LINEAR)
        img1 = cv2.resize(img1, (nw, nh), interpolation=cv2.INTER_LINEAR)
        if self.sparse:
            flow, valid = self._resize_sparse(flow, valid, sx, sy, (nh, nw))
        else:
            flow = cv2.resize(flow, (nw, nh), interpolation=cv2.INTER_LINEAR) * np.array([sx, sy], np.float32)
            valid = cv2.resize(valid.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST).astype(bool)

        if random.random() < self.hflip_prob:
            img0, img1, valid = img0[:, ::-1], img1[:, ::-1], valid[:, ::-1]
            flow = flow[:, ::-1] * np.array([-1.0, 1.0], np.float32)
        if random.random() < self.vflip_prob:
            img0, img1, valid = img0[::-1], img1[::-1], valid[::-1]
            flow = flow[::-1] * np.array([1.0, -1.0], np.float32)

        y0, x0 = random.randint(0, nh - ch), random.randint(0, nw - cw)
        crop = (slice(y0, y0 + ch), slice(x0, x0 + cw))
        img1 = self._eraser(np.ascontiguousarray(img1[crop]))
        return (np.ascontiguousarray(img0[crop]), img1,
                np.ascontiguousarray(flow[crop]), np.ascontiguousarray(valid[crop]))


@torch.no_grad()
def color_jitter(img0, img1, asym_prob=0.2, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.16):
    """Batched colour jitter for float images [B,3,H,W] in 0..1, on whatever device they are.

    Each pair gets one random brightness / contrast / saturation / hue change; with
    probability asym_prob the second frame gets its own. Hue is a rotation in YIQ space.
    """
    b = img0.shape[0]
    dev = img0.device

    def params():
        return {"b": 1 + (torch.rand(b, 1, 1, 1, device=dev) * 2 - 1) * brightness,
                "c": 1 + (torch.rand(b, 1, 1, 1, device=dev) * 2 - 1) * contrast,
                "s": 1 + (torch.rand(b, 1, 1, 1, device=dev) * 2 - 1) * saturation,
                "h": (torch.rand(b, device=dev) * 2 - 1) * hue * 2 * math.pi}

    to_yiq = torch.tensor([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]], device=dev)
    to_rgb = torch.linalg.inv(to_yiq)

    def apply(img, p):
        img = img * p["b"]
        mean = img.mean(dim=(1, 2, 3), keepdim=True)
        img = (img - mean) * p["c"] + mean
        gray = img.mean(dim=1, keepdim=True)
        img = (img - gray) * p["s"] + gray
        cos, sin = torch.cos(p["h"]), torch.sin(p["h"])
        rot = torch.zeros(b, 3, 3, device=dev)
        rot[:, 0, 0] = 1
        rot[:, 1, 1], rot[:, 1, 2], rot[:, 2, 1], rot[:, 2, 2] = cos, -sin, sin, cos
        m = to_rgb @ rot @ to_yiq                                   # [B,3,3]
        img = torch.einsum("bij,bjhw->bihw", m, img)
        return img.clamp(0, 1)

    p0 = params()
    p1 = params()
    same = (torch.rand(b, device=dev) >= asym_prob)
    for k in p1:
        mask = same.view(-1, *([1] * (p1[k].dim() - 1)))
        p1[k] = torch.where(mask, p0[k], p1[k])
    return apply(img0, p0), apply(img1, p1)


def _split_indices(n: int, split: str, val_fraction: float = 0.1) -> np.ndarray:
    """Deterministic split for datasets whose samples are independent (Chairs, KITTI)."""
    if split == "all":
        return np.arange(n)
    perm = np.random.RandomState(SPLIT_SEED).permutation(n)
    n_train = int((1.0 - val_fraction) * n)
    return perm[:n_train] if split == "train" else perm[n_train:]


class FlowPairs(Dataset):
    """Base class: subclasses fill self.samples and implement load(index) ->
    (img0 HWC uint8, img1 HWC uint8, flow [2,H,W] float32, valid_noc [H,W] bool, valid_all [H,W] bool)."""

    sparse = False         # sparse ground truth (KITTI): resized by moving points
    strong_aug = False     # "december" recipe only: motion blur and synthetic occlusions (FlyingChairs)
    aug_recipe = "raft"    # "raft" (FlowAugmentor) | "december" (the recipe of the December training)

    def __init__(self, size_hw: Tuple[int, int], augment: bool):
        self.size_hw = tuple(size_hw)
        self.augment = augment
        self.samples: list = []
        self.augmentor = FlowAugmentor(self.size_hw, sparse=self.sparse)

    def __len__(self):
        return len(self.samples)

    def load(self, index):
        raise NotImplementedError

    def __getitem__(self, index):
        img0, img1, flow, valid_noc, valid_all = self.load(index)
        if not self.augment:
            # evaluation: original frames, the caller handles resolution
            return (torch.from_numpy(np.ascontiguousarray(img0)).permute(2, 0, 1),
                    torch.from_numpy(np.ascontiguousarray(img1)).permute(2, 0, 1),
                    torch.from_numpy(flow), torch.from_numpy(valid_noc), torch.from_numpy(valid_all))

        if self.aug_recipe == "raft":
            # trained on all valid pixels, occluded ones included, to match the evaluation
            a0, a1, aflow, avalid = self.augmentor(np.ascontiguousarray(img0[..., :3]),
                                                   np.ascontiguousarray(img1[..., :3]),
                                                   np.ascontiguousarray(flow.transpose(1, 2, 0)), valid_all)
            return (torch.from_numpy(a0).permute(2, 0, 1).float() / 255.0,
                    torch.from_numpy(a1).permute(2, 0, 1).float() / 255.0,
                    torch.from_numpy(aflow).permute(2, 0, 1).float(),
                    torch.from_numpy(avalid.astype(np.float32))[None])

        # --- "december": scale by height, random horizontal crop, mild colour and affine jitter,
        #     loss on non-occluded pixels only
        x_off = _random_x_offset(*img0.shape[:2], self.size_hw)
        t0 = torch.from_numpy(resize_img(img0, self.size_hw, x_off)).permute(2, 0, 1).float() / 255.0
        t1 = torch.from_numpy(resize_img(img1, self.size_hw, x_off)).permute(2, 0, 1).float() / 255.0
        tf = torch.from_numpy(resize_flow(flow, self.size_hw, x_off)).float()
        mv = torch.from_numpy(resize_mask_nearest(valid_noc, self.size_hw, x_off).astype(np.float32))[None]

        t0 = _apply_color_aug(t0)
        t1 = _apply_color_aug(t1)
        if self.strong_aug:
            t0 = _apply_motion_blur(t0)
            t1 = _apply_motion_blur(t1)
            t0 = _apply_random_occlusion(t0)
            t1 = _apply_random_occlusion(t1)
        t0, t1, tf, mv = _apply_geo_aug(t0, t1, tf, mask=mv)
        if random.random() > 0.5:
            t0, t1 = TF.hflip(t0), TF.hflip(t1)
            tf[0] = -tf[0]
            tf = tf.flip(2)
            mv = mv.flip(2)
        return t0.clamp(0, 1), t1.clamp(0, 1), tf, mv


class FlyingChairs(FlowPairs):
    strong_aug = True

    def __init__(self, split, size_hw, augment):
        super().__init__(size_hw, augment)
        root = Path(DATA["chairs"])
        if not root.is_dir():
            raise FileNotFoundError(f"FlyingChairs not found: {root} (set NPUFLOW_CHAIRS)")
        bases = sorted(p.name[:-len("_img1.ppm")] for p in root.glob("*_img1.ppm"))
        self.samples = [(root / f"{b}_img1.ppm", root / f"{b}_img2.ppm", root / f"{b}_flow.flo")
                        for b in (bases[i] for i in _split_indices(len(bases), split))]

    def load(self, index):
        f0, f1, fflow = self.samples[index]
        flow = read_flo(str(fflow))
        valid = np.ones(flow.shape[1:], dtype=bool)
        return iio.imread(f0), iio.imread(f1), flow, valid, valid


class Sintel(FlowPairs):
    """split: "train" (all scenes except SINTEL_VAL_SCENES), "val" (those scenes), "all"."""

    def __init__(self, pass_name, split, size_hw, augment):
        super().__init__(size_hw, augment)
        root = Path(DATA["sintel"]) / "training"
        if not (root / pass_name).is_dir():
            raise FileNotFoundError(f"Sintel not found: {root / pass_name} (set NPUFLOW_SINTEL)")
        self.root = root
        for scene in sorted(os.listdir(root / pass_name)):
            is_val = scene in SINTEL_VAL_SCENES
            if (split == "train" and is_val) or (split == "val" and not is_val):
                continue
            frames = sorted(f for f in os.listdir(root / pass_name / scene) if f.endswith(".png"))
            for a, b in zip(frames, frames[1:]):
                self.samples.append((pass_name, scene, a, b))

    def load(self, index):
        pass_name, scene, a, b = self.samples[index]
        base = a[:-4]
        flow = read_flo(str(self.root / "flow" / scene / f"{base}.flo"))
        inv = iio.imread(self.root / "invalid" / scene / f"{base}.png").astype(bool)
        occ = iio.imread(self.root / "occlusions" / scene / f"{base}.png").astype(bool)
        return (iio.imread(self.root / pass_name / scene / a), iio.imread(self.root / pass_name / scene / b),
                flow, ~(inv | occ), ~inv)


class Kitti2015(FlowPairs):
    sparse = True

    def __init__(self, split, size_hw, augment):
        super().__init__(size_hw, augment)
        root = Path(DATA["kitti"]) / "training"
        if not (root / "image_2").is_dir():
            raise FileNotFoundError(f"KITTI 2015 not found: {root} (set NPUFLOW_KITTI)")
        first = sorted((root / "image_2").glob("*_10.png"))
        self.samples = [(first[i], first[i].with_name(first[i].name.replace("_10.png", "_11.png")),
                         root / "flow_occ" / first[i].name) for i in _split_indices(len(first), split)]

    def load(self, index):
        f0, f1, fflow = self.samples[index]
        flow, valid = read_kitti_png_flow(str(fflow))          # [H,W,2], [H,W]
        return iio.imread(f0), iio.imread(f1), np.ascontiguousarray(flow.transpose(2, 0, 1)), valid, valid


class FlyingThings3D(FlowPairs):
    def __init__(self, split, size_hw, augment):
        super().__init__(size_hw, augment)
        raise NotImplementedError(
            "FlyingThings3D loader is not written yet: the layout of the Kaggle copy has to be checked first.")


def make_dataset(name: str, split: str, size_hw, augment: bool, aug_recipe: str = "raft") -> FlowPairs:
    ds = _make_dataset(name, split, size_hw, augment)
    ds.aug_recipe = aug_recipe
    return ds


def _make_dataset(name: str, split: str, size_hw, augment: bool) -> FlowPairs:
    if name == "chairs":
        return FlyingChairs(split, size_hw, augment)
    if name == "things":
        return FlyingThings3D(split, size_hw, augment)
    if name in ("sintel-clean", "sintel-final"):
        return Sintel(name.split("-")[1], split, size_hw, augment)
    if name == "kitti15":
        return Kitti2015(split, size_hw, augment)
    raise ValueError(f"unknown dataset {name!r}")
