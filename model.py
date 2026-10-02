"""EdgeFlowNet model with architecture switches.

The default ModelConfig reproduces the network in v1.py exactly (same module
names, so v1 checkpoints load with strict=True). Every other setting is one
architecture variant for speed/accuracy experiments on the CV181x TPU.
"""
from dataclasses import dataclass, asdict
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    image_size: Tuple[int, int] = (256, 320)
    base_channels: int = 48
    r16: int = 6
    iters32: int = 2
    iters16: int = 2
    act: str = "leaky"          # "leaky" | "relu"
    use_se: bool = True
    use_pos_enc: bool = True
    use_l2norm: bool = True
    corr16: str = "2d"          # "2d": (2r+1)^2 shifts | "1d": horizontal + vertical shifts
    full_refine: str = "conv"   # "conv": full-res refine block | "pixelshuffle" | "none"
    # --- matching at s8 (PLAN.md section 9, replacements for warping / lookup) ---
    match8: str = "none"        # "none" | "global1d" (B1: dual-1D global corr + soft-argmax)
                                # | "local1d" (dual-1D local corr, radius r8)
                                # | "dcv" (B3: dilated 2D local corr, radius r8, dilations dcv_dilations)
    r8: int = 4                 # radius for local1d / dcv at s8
    dcv_dilations: Tuple[int, ...] = (1, 2, 4)
    match8_softargmax: bool = True   # global1d only: regress an initial s8 flow by soft-argmax
    match8_local_r: int = 0     # global1d only: extra 2D local corr of this radius (0 = off)
    warp8: str = "none"         # "none" | "kpn" (B2: separable kernel-prediction warp of f1 at s8)
    kpn_r8: int = 4             # kernel radius for warp8 = kpn
    refine8_f1: bool = True     # feed f1 (raw or kpn-warped) into refine_s8

    def __post_init__(self):
        self.image_size = tuple(self.image_size)
        self.dcv_dilations = tuple(int(d) for d in self.dcv_dilations)
        assert self.act in ("leaky", "relu"), self.act
        assert self.corr16 in ("2d", "1d"), self.corr16
        assert self.full_refine in ("conv", "pixelshuffle", "none"), self.full_refine
        assert self.iters32 >= 1 and self.iters16 >= 1
        assert self.match8 in ("none", "global1d", "local1d", "dcv"), self.match8
        assert self.warp8 in ("none", "kpn"), self.warp8
        assert self.r8 >= 1 and self.kpn_r8 >= 1 and self.match8_local_r >= 0

    def to_dict(self):
        d = asdict(self)
        d["image_size"] = list(self.image_size)
        d["dcv_dilations"] = list(self.dcv_dilations)
        return d


def make_act(kind: str) -> nn.Module:
    if kind == "relu":
        return nn.ReLU(inplace=True)
    return nn.LeakyReLU(0.1, inplace=True)


def l2norm(x, eps=1e-6):
    return x / (x.pow(2).sum(1, keepdim=True).add(eps).sqrt())


class DWConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, act="leaky"):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = make_act(act)

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class GlobalCorrelationDual1D(nn.Module):
    """Dual global correlation (horizontal + vertical).

    Output channels = W + H of the feature map (18 for 256x320 at 1/32).
    """

    def __init__(self, channels: int, normalize: bool = True):
        super().__init__()
        self.normalize = normalize
        divisor = torch.sqrt(torch.tensor(float(channels)))
        self.register_buffer('divisor', divisor)

    def forward(self, f0, f1):
        if self.normalize:
            f0, f1 = l2norm(f0), l2norm(f1)

        # Horizontal: [B,H,W,C] @ [B,H,C,W] -> [B,H,W,W] -> [B,W,H,W]
        q_h = f0.permute(0, 2, 3, 1).contiguous()
        k_h = f1.permute(0, 2, 1, 3).contiguous()
        corr_h = torch.matmul(q_h, k_h) / self.divisor
        corr_h = corr_h.permute(0, 3, 1, 2).contiguous()

        # Vertical: [B,W,H,C] @ [B,W,C,H] -> [B,W,H,H] -> [B,H,H,W]
        q_v = f0.permute(0, 3, 2, 1).contiguous()
        k_v = f1.permute(0, 3, 1, 2).contiguous()
        corr_v = torch.matmul(q_v, k_v) / self.divisor
        corr_v = corr_v.permute(0, 3, 2, 1).contiguous()

        return torch.cat([corr_h, corr_v], dim=1)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block; the squeeze is a fixed full-map conv (TPU friendly)."""

    def __init__(self, ch, r=16, spatial_size=None):
        super().__init__()
        if spatial_size is None:
            raise ValueError("spatial_size is required for CV181x TPU compatibility")
        H, W = spatial_size
        squeeze_weights = torch.full((ch, 1, H, W), 1.0 / (H * W), dtype=torch.float32)
        self.register_buffer('squeeze_weights', squeeze_weights)
        self.groups = ch
        self.fc1 = nn.Conv2d(ch, ch // r, 1)
        self.fc2 = nn.Conv2d(ch // r, ch, 1)
        self.act = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        s = F.conv2d(x, self.squeeze_weights, bias=None, stride=1, padding=0, groups=self.groups)
        s = self.fc1(s)
        s = self.act(s)
        s = self.fc2(s)
        s = self.sigmoid(s)
        return x * s


class StaticPositionalEncoder(nn.Module):
    def __init__(self, h, w):
        super().__init__()
        ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        grid = torch.stack([ys.float() / (h - 1 + 1e-6), xs.float() / (w - 1 + 1e-6)], 0)
        self.register_buffer('grid', grid.unsqueeze(0))

    def forward(self, feat):
        B = feat.size(0)
        return torch.cat([feat, self.grid.expand(B, -1, -1, -1)], dim=1)


class EfficientEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        C = cfg.base_channels
        H, W = cfg.image_size
        a = cfg.act
        self.use_se = cfg.use_se
        self.use_pos_enc = cfg.use_pos_enc

        self.s8_block = nn.Sequential(
            DWConvBlock(3, C, 2, a),  # /2
            DWConvBlock(C, C, 2, a),  # /4
            DWConvBlock(C, C * 2, 2, a),  # /8
            DWConvBlock(C * 2, C * 2, 1, a),
            DWConvBlock(C * 2, C * 2, 1, a),
        )
        self.s16_block = nn.Sequential(DWConvBlock(C * 2, C * 2, 2, a))  # /16
        self.s32_block = nn.Sequential(DWConvBlock(C * 2, C * 2, 2, a))  # /32

        self.downsample_s8_to_s16 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.downsample_s16_to_s32 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.s4_block = nn.Sequential(DWConvBlock(C, C, 1, a))

        self.out_s4_ch = C
        self.out_s8_ch = C * 2
        self.out_s16_ch = C * 2
        self.out_s32_ch = C * 2

        assert H % 32 == 0 and W % 32 == 0, "image_size must be divisible by 32"
        if self.use_pos_enc:
            self.pos_enc4 = StaticPositionalEncoder(H // 4, W // 4)
            self.pos_enc8 = StaticPositionalEncoder(H // 8, W // 8)
            self.pos_enc16 = StaticPositionalEncoder(H // 16, W // 16)
            self.pos_enc32 = StaticPositionalEncoder(H // 32, W // 32)

            self.pos_proj_s4 = nn.Conv2d(self.out_s4_ch + 2, self.out_s4_ch, 1, bias=False)
            self.pos_proj_s8 = nn.Conv2d(self.out_s8_ch + 2, self.out_s8_ch, 1, bias=False)
            self.pos_proj_s16 = nn.Conv2d(self.out_s16_ch + 2, self.out_s16_ch, 1, bias=False)
            self.pos_proj_s32 = nn.Conv2d(self.out_s32_ch + 2, self.out_s32_ch, 1, bias=False)

        if self.use_se:
            self.se_s8 = SEBlock(self.out_s8_ch, r=16, spatial_size=(H // 8, W // 8))
            self.se_s16 = SEBlock(self.out_s16_ch, r=16, spatial_size=(H // 16, W // 16))
            self.se_s32 = SEBlock(self.out_s32_ch, r=16, spatial_size=(H // 32, W // 32))

        self.alpha_s16 = nn.Parameter(torch.tensor(0.5))
        self.alpha_s32 = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        # s4
        s4 = self.s8_block[0](x)
        s4 = self.s8_block[1](s4)
        s4_feat = self.s4_block(s4)

        # s8
        s8 = self.s8_block[2](s4_feat)
        s8 = self.s8_block[3](s8)
        s8 = self.s8_block[4](s8)
        if self.use_se:
            s8 = self.se_s8(s8)

        # s16 + skip
        s16 = self.s16_block(s8)
        s16_down = self.downsample_s8_to_s16(s8)
        s16 = self.alpha_s16 * s16 + (1 - self.alpha_s16) * s16_down
        if self.use_se:
            s16 = self.se_s16(s16)

        # s32 + skip
        s32 = self.s32_block(s16)
        s32_down = self.downsample_s16_to_s32(s16)
        s32 = self.alpha_s32 * s32 + (1 - self.alpha_s32) * s32_down
        if self.use_se:
            s32 = self.se_s32(s32)

        if not self.use_pos_enc:
            return s4_feat, s8, s16, s32

        s4_pos = self.pos_proj_s4(self.pos_enc4(s4_feat))
        s8_pos = self.pos_proj_s8(self.pos_enc8(s8))
        s16_pos = self.pos_proj_s16(self.pos_enc16(s16))
        s32_pos = self.pos_proj_s32(self.pos_enc32(s32))
        return s4_pos, s8_pos, s16_pos, s32_pos


class LocalCostVolumePad(nn.Module):
    """Local cost volume by shifting a replicate-padded f1.

    mode "2d": all (2r+1)^2 shifts. mode "1d": horizontal and vertical shifts
    only, 2*(2r+1) channels.
    """

    def __init__(self, radius: int, mode: str = "2d", normalize: bool = True,
                 dilations=(1,)):
        super().__init__()
        self.r = int(radius)
        self.normalize = normalize
        r = self.r
        shifts = []
        for d in dilations:
            if mode == "2d":
                shifts += [(dy * d, dx * d) for dy in range(-r, r + 1) for dx in range(-r, r + 1)]
            else:
                shifts += ([(0, dx * d) for dx in range(-r, r + 1)]
                           + [(dy * d, 0) for dy in range(-r, r + 1)])
        self.shifts = shifts
        self.pad = r * max(dilations)
        self.out_channels = len(self.shifts)
        # Unused in forward; kept so v1 checkpoints load with strict=True
        self.register_buffer('offsets', torch.tensor(self.shifts))

    def forward(self, f0, f1):
        if not torch.onnx.is_in_onnx_export():
            assert f0.shape == f1.shape, f"CostVolume: f0 {f0.shape} vs f1 {f1.shape}"

        if self.normalize:
            f0, f1 = l2norm(f0), l2norm(f1)
        B, C, H, W = f0.shape
        r = self.pad

        f1_pad = F.pad(f1, (r, r, r, r), mode='replicate')

        costs = []
        for dy, dx in self.shifts:
            shifted = f1_pad[:, :, r + dy:r + dy + H, r + dx:r + dx + W]
            costs.append((f0 * shifted).sum(1, keepdim=True))
        return torch.cat(costs, 1)


class SoftArgmax1D(nn.Module):
    """Flow from a dual-1D global correlation by softmax-weighted coordinates (GMFlow-style).

    Input: [B, W+H, H, W] from GlobalCorrelationDual1D (first W channels: candidate x',
    next H channels: candidate y'). Output: [B, 2, H, W] flow in feature-map pixels.
    Only Slice / Mul / Softmax / ReduceSum / Sub-with-constant: compiles on CV181x.
    """

    def __init__(self, h: int, w: int, init_scale: float = 10.0):
        super().__init__()
        self.h, self.w = h, w
        self.register_buffer('cand_x', torch.arange(w).float().view(1, w, 1, 1))
        self.register_buffer('cand_y', torch.arange(h).float().view(1, h, 1, 1))
        self.register_buffer('grid_x', torch.arange(w).float().view(1, 1, 1, w))
        self.register_buffer('grid_y', torch.arange(h).float().view(1, 1, h, 1))
        # correlation values are ~[-0.1, 0.1] after l2norm / sqrt(C): a learnable
        # logit scale keeps the softmax from being near-uniform
        self.logit_scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, corr):
        corr_h = corr[:, :self.w]
        corr_v = corr[:, self.w:self.w + self.h]
        p_h = F.softmax(corr_h * self.logit_scale, dim=1)
        p_v = F.softmax(corr_v * self.logit_scale, dim=1)
        u = (p_h * self.cand_x).sum(1, keepdim=True) - self.grid_x
        v = (p_v * self.cand_y).sum(1, keepdim=True) - self.grid_y
        return torch.cat([u, v], 1)


class SeparableKPNWarp(nn.Module):
    """Warp f1 without GridSample: a predicted separable kernel over static shifts.

    f1_warped = sum_k w_v[k] * shift_y(k, sum_j w_h[j] * shift_x(j, f1)); the weights
    w_h, w_v (softmax over 2r+1 taps each) come from a small conv on [flow, context].
    SepConv / AdaCoF / KPN idea, restricted to static shifts (Slice + Mul + Add).
    """

    def __init__(self, guide_ch: int, radius: int, hidden: int = 32, act: str = "leaky"):
        super().__init__()
        self.r = int(radius)
        k = 2 * self.r + 1
        self.k = k
        self.net = nn.Sequential(
            nn.Conv2d(guide_ch, hidden, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden),
            make_act(act),
            nn.Conv2d(hidden, 2 * k, 1, 1, 0, bias=True),
        )

    def forward(self, f1, guide):
        B, C, H, W = f1.shape
        r, k = self.r, self.k
        w = self.net(guide)
        w_h = F.softmax(w[:, :k], dim=1)
        w_v = F.softmax(w[:, k:2 * k], dim=1)

        f1_pad = F.pad(f1, (r, r, 0, 0), mode='replicate')
        out = None
        for j in range(k):
            term = w_h[:, j:j + 1] * f1_pad[:, :, :, j:j + W]
            out = term if out is None else out + term

        out_pad = F.pad(out, (0, 0, r, r), mode='replicate')
        res = None
        for i in range(k):
            term = w_v[:, i:i + 1] * out_pad[:, :, i:i + H, :]
            res = term if res is None else res + term
        return res


def _dw_pw_stack(hidden: int, act: str, n: int = 3) -> nn.Sequential:
    layers = []
    for _ in range(n):
        layers += [
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            make_act(act),
            nn.Conv2d(hidden, hidden, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden),
            make_act(act),
        ]
    return nn.Sequential(*layers)


class CoarseUpdateLite(nn.Module):
    def __init__(self, in_ch, hidden=128, act="leaky"):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            make_act(act),
        )
        self.body = _dw_pw_stack(hidden, act)
        self.head = nn.Conv2d(hidden, 2, 3, 1, 1)

    def forward(self, x):
        return self.head(self.body(self.reduce(x)))


class RefineLite(nn.Module):
    """Refine block shared by s16/s8/s4; only in_ch/hidden differ."""

    def __init__(self, in_ch: int, hidden: int = 96, act="leaky"):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            make_act(act),
        )
        self.body = _dw_pw_stack(hidden, act)
        self.head = nn.Conv2d(hidden, 2, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(self.reduce(x)))


class EdgeFlowNet(nn.Module):
    def __init__(self, cfg: ModelConfig = None):
        super().__init__()
        cfg = cfg or ModelConfig()
        self.cfg = cfg
        a = cfg.act
        self.enc = EfficientEncoder(cfg)

        s4_ch = self.enc.out_s4_ch
        s8_ch = self.enc.out_s8_ch
        s16_ch = self.enc.out_s16_ch
        s32_ch = self.enc.out_s32_ch

        # 1. Global correlation (s32)
        self.corr32 = GlobalCorrelationDual1D(channels=s32_ch, normalize=cfg.use_l2norm)
        cv32_ch = cfg.image_size[0] // 32 + cfg.image_size[1] // 32

        # 2. Local correlation (s16)
        self.corr16 = LocalCostVolumePad(cfg.r16, mode=cfg.corr16, normalize=cfg.use_l2norm)
        cv16_ch = self.corr16.out_channels

        # 3. Context convolutions
        self.ctx32 = nn.Conv2d(s32_ch, s16_ch, 1, bias=False)
        self.ctx16 = nn.Conv2d(s16_ch, s16_ch, 1, bias=False)
        self.ctx8 = nn.Conv2d(s8_ch, s16_ch, 1, bias=False)
        self.ctx4 = nn.Conv2d(s4_ch, s16_ch, 1, bias=False)

        # 4. Coarse updates
        self.update32 = CoarseUpdateLite(cv32_ch + 2 + s16_ch, hidden=128, act=a)
        self.update16 = CoarseUpdateLite(cv16_ch + cv32_ch + 2 + s16_ch, hidden=128, act=a)
        self.iters32, self.iters16 = cfg.iters32, cfg.iters16

        # 4b. Matching / warping at s8 (PLAN.md section 9)
        H8, W8 = cfg.image_size[0] // 8, cfg.image_size[1] // 8
        cv8_ch = 0
        if cfg.match8 == "global1d":
            self.corr8 = GlobalCorrelationDual1D(channels=s8_ch, normalize=cfg.use_l2norm)
            cv8_ch += H8 + W8
            if cfg.match8_softargmax:
                self.softargmax8 = SoftArgmax1D(H8, W8)
                cv8_ch += 2
            if cfg.match8_local_r > 0:
                self.corr8_local = LocalCostVolumePad(cfg.match8_local_r, mode="2d",
                                                      normalize=cfg.use_l2norm)
                cv8_ch += self.corr8_local.out_channels
        elif cfg.match8 == "local1d":
            self.corr8 = LocalCostVolumePad(cfg.r8, mode="1d", normalize=cfg.use_l2norm)
            cv8_ch += self.corr8.out_channels
        elif cfg.match8 == "dcv":
            self.corr8 = LocalCostVolumePad(cfg.r8, mode="2d", normalize=cfg.use_l2norm,
                                            dilations=cfg.dcv_dilations)
            cv8_ch += self.corr8.out_channels
        if cfg.warp8 == "kpn":
            self.warp8 = SeparableKPNWarp(guide_ch=2 + s16_ch, radius=cfg.kpn_r8, act=a)
        f1_8_ch = s8_ch if cfg.refine8_f1 else 0

        # 5. Refine blocks
        self.refine_s16 = RefineLite(s16_ch + s16_ch + 2 + s16_ch, hidden=128, act=a)
        self.refine_s8 = RefineLite(s8_ch + f1_8_ch + cv8_ch + 2 + s16_ch, hidden=128, act=a)
        # s4 has the largest spatial size, so its refine is lighter
        self.refine_s4 = RefineLite(s4_ch + s4_ch + 2 + s16_ch, hidden=64, act=a)

        self.alpha_s16 = nn.Parameter(torch.tensor(0.5))
        self.alpha_s8 = nn.Parameter(torch.tensor(0.5))
        self.alpha_s4 = nn.Parameter(torch.tensor(0.5))

        # 6. Full-resolution head
        if cfg.full_refine == "conv":
            full_hidden = 32
            self.full_refine = nn.Sequential(
                DWConvBlock(5, full_hidden, stride=1, act=a),   # [flow(2) + img0(3)]
                # dilation=2 for a larger receptive field
                nn.Conv2d(full_hidden, full_hidden, 3, 1, 2, dilation=2,
                          groups=full_hidden, bias=False),
                nn.BatchNorm2d(full_hidden),
                make_act(a),
                nn.Conv2d(full_hidden, full_hidden, 1, 1, 0, bias=False),
                nn.BatchNorm2d(full_hidden),
                make_act(a),
                nn.Conv2d(full_hidden, 2, 1, 1, 0, bias=True),
            )
            self.alpha_full = nn.Parameter(torch.tensor(0.5))
        elif cfg.full_refine == "pixelshuffle":
            # Residual predicted at 1/4 scale and unfolded to full resolution,
            # so no convolution runs at 256x320.
            self.up_head = nn.Sequential(
                DWConvBlock(s4_ch + 2, 64, stride=1, act=a),
                nn.Conv2d(64, 2 * 16, 1, 1, 0, bias=True),
                nn.PixelShuffle(4),
            )
            self.alpha_full = nn.Parameter(torch.tensor(0.5))

        self._frozen_encoder = False

        # Deploy-time concat range equalization (see fold_cat_scales). None = plain concat.
        self.cat_scales = None
        self._cat_stats = None

    def _cat(self, name: str, groups):
        """Concatenate tensor groups along channels.

        With cat_scales set, every group is divided by its scale first, so all
        inputs of an INT8 Concat share a similar range; fold_cat_scales has
        already multiplied the next conv's weights by the same factors.
        """
        if self._cat_stats is not None:
            stats = self._cat_stats.setdefault(name, [0.0] * len(groups))
            for i, g in enumerate(groups):
                flat = g.detach().abs().flatten().float()
                if flat.numel() > 200_000:
                    flat = flat[torch.randperm(flat.numel())[:200_000]]
                stats[i] = max(stats[i], float(torch.quantile(flat, 0.9999)))
        if self.cat_scales is not None and name in self.cat_scales:
            groups = [g if k == 1.0 else g * (1.0 / k)
                      for g, k in zip(groups, self.cat_scales[name])]
        return torch.cat(groups, 1)

    def _cat_targets(self):
        """Concat name -> conv that consumes it (its weights absorb the scales)."""
        targets = {
            "update32": self.update32.reduce[0],
            "update16": self.update16.reduce[0],
            "refine_s16": self.refine_s16.reduce[0],
            "refine_s8": self.refine_s8.reduce[0],
            "refine_s4": self.refine_s4.reduce[0],
        }
        if self.cfg.full_refine == "conv":
            targets["full"] = self.full_refine[0].dw
        elif self.cfg.full_refine == "pixelshuffle":
            targets["full"] = self.up_head[0].dw
        return targets

    @torch.no_grad()
    def fold_cat_scales(self, scales: dict, group_channels: dict):
        """Divide concat inputs by per-group scales and compensate in the next conv.

        Exact in FP32: the consumer is a 1x1 conv (weight columns scaled) or a
        depthwise conv (one filter per input channel).
        scales[name] = [k per group], group_channels[name] = [channels per group].
        """
        targets = self._cat_targets()
        for name, ks in scales.items():
            conv = targets[name]
            per_channel = torch.cat([torch.full((c,), float(k)) for k, c in
                                     zip(ks, group_channels[name])]).to(conv.weight)
            if conv.groups == 1:
                assert conv.weight.shape[1] == per_channel.numel(), name
                conv.weight.mul_(per_channel.view(1, -1, 1, 1))
            else:   # depthwise: weight is [C, 1, kh, kw]
                assert conv.groups == conv.weight.shape[0] == per_channel.numel(), name
                conv.weight.mul_(per_channel.view(-1, 1, 1, 1))
        self.cat_scales = {k: [float(x) for x in v] for k, v in scales.items()}

    def set_encoder_frozen(self, frozen: bool):
        """Freeze encoder with correct BatchNorm handling"""
        self._frozen_encoder = frozen
        for name, module in self.enc.named_modules():
            for p in module.parameters():
                p.requires_grad = not frozen
            if not isinstance(module, nn.BatchNorm2d):
                if frozen:
                    module.eval()
                else:
                    module.train()

    def forward(self, img0, img1):
        Hf, Wf = img0.shape[-2:]

        f0_s4, f0_s8, f0_s16, f0_s32 = self.enc(img0)
        f1_s4, f1_s8, f1_s16, f1_s32 = self.enc(img1)

        ctx32 = self.ctx32(f0_s32)
        ctx16 = self.ctx16(f0_s16)
        ctx8 = self.ctx8(f0_s8)
        ctx4 = self.ctx4(f0_s4)

        # --- s32 ---
        B, _, H32, W32 = f0_s32.shape
        flow_s32 = torch.zeros(B, 2, H32, W32, device=img0.device, dtype=img0.dtype)
        cost32 = self.corr32(f0_s32, f1_s32)
        for _ in range(self.iters32):
            delta = self.update32(self._cat("update32", [cost32, flow_s32, ctx32]))
            flow_s32 = flow_s32 + delta

        # --- s16 ---
        _, _, H16, W16 = f0_s16.shape
        flow_s16 = F.interpolate(flow_s32, size=(H16, W16), mode='bilinear', align_corners=False) * 2.0
        cost16 = self.corr16(f0_s16, f1_s16)
        cost32_up = F.interpolate(cost32, size=(H16, W16), mode='bilinear', align_corners=False)
        for _ in range(self.iters16):
            delta = self.update16(self._cat("update16", [cost16, cost32_up, flow_s16, ctx16]))
            flow_s16 = flow_s16 + delta

        # --- s16 refine ---
        delta_s16 = self.refine_s16(self._cat("refine_s16", [f0_s16, f1_s16, flow_s16, ctx16]))
        flow_s16_ref = flow_s16 + self.alpha_s16 * delta_s16

        # --- s8 refine ---
        _, _, H8, W8 = f0_s8.shape
        flow_s8 = F.interpolate(flow_s16_ref, size=(H8, W8), mode='bilinear', align_corners=False) * 2.0
        groups8 = [f0_s8]
        if self.cfg.refine8_f1:
            if self.cfg.warp8 == "kpn":
                groups8.append(self.warp8(f1_s8, torch.cat([flow_s8, ctx8], 1)))
            else:
                groups8.append(f1_s8)
        if self.cfg.match8 == "global1d":
            cost8 = self.corr8(f0_s8, f1_s8)
            groups8.append(cost8)
            if self.cfg.match8_softargmax:
                groups8.append(self.softargmax8(cost8))
            if self.cfg.match8_local_r > 0:
                groups8.append(self.corr8_local(f0_s8, f1_s8))
        elif self.cfg.match8 in ("local1d", "dcv"):
            groups8.append(self.corr8(f0_s8, f1_s8))
        groups8 += [flow_s8, ctx8]
        delta_s8 = self.refine_s8(self._cat("refine_s8", groups8))
        flow_s8_ref = flow_s8 + self.alpha_s8 * delta_s8

        # --- s4 refine ---
        _, _, H4, W4 = f0_s4.shape
        flow_s4 = F.interpolate(flow_s8_ref, size=(H4, W4), mode='bilinear', align_corners=False) * 2.0
        delta_s4 = self.refine_s4(self._cat("refine_s4", [f0_s4, f1_s4, flow_s4, ctx4]))
        flow_s4_ref = flow_s4 + self.alpha_s4 * delta_s4

        # --- full resolution ---
        flow_full = F.interpolate(flow_s4_ref, size=(Hf, Wf), mode='bilinear',
                                  align_corners=False) * 4.0
        if self.cfg.full_refine == "conv":
            delta_full = self.full_refine(self._cat("full", [flow_full, img0]))
            flow_full_ref = flow_full + self.alpha_full * delta_full
        elif self.cfg.full_refine == "pixelshuffle":
            delta_full = self.up_head(self._cat("full", [f0_s4, flow_s4_ref]))
            flow_full_ref = flow_full + self.alpha_full * delta_full
        else:
            flow_full_ref = flow_full

        if self.training:
            return [flow_s32, flow_s16_ref, flow_s8_ref, flow_s4_ref, flow_full_ref], None
        return flow_full_ref

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
