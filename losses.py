"""Training loss: multi-scale Charbonnier on the five stage outputs (as in the December training)."""
import torch
import torch.nn.functional as F


def charbonnier(x, eps=1e-3):
    return torch.sqrt(x * x + eps * eps)


def endpoint_error(pred, gt):
    return torch.sqrt(((pred - gt) ** 2).sum(1) + 1e-6)


def gradient(data):
    """Compute image gradients (spatial derivatives)."""
    D_dy = data[:, :, 1:, :] - data[:, :, :-1, :]
    D_dx = data[:, :, :, 1:] - data[:, :, :, :-1]
    return D_dx, D_dy


def edge_aware_smoothness_loss(flow, img):
    """Edge-aware smoothness loss.

    Penalizes flow gradients in smooth image regions and preserves discontinuities at edges.
    """
    flow_dx, flow_dy = gradient(flow)
    img_dx, img_dy = gradient(img)

    weights_x = torch.exp(-torch.mean(torch.abs(img_dx), dim=1, keepdim=True))
    weights_y = torch.exp(-torch.mean(torch.abs(img_dy), dim=1, keepdim=True))

    loss = (torch.abs(flow_dx) * weights_x).mean() + \
           (torch.abs(flow_dy) * weights_y).mean()
    return loss




def multiscale_loss_masked(preds, gt, img_in, valid_mask=None,
                            w32=0.25, w16=0.5, w8=1.0, w4=1.0,
                            w_full=3.0,
                            w_edge_smooth=0.003,
                            smooth_lambda=0.0):
    """Multiscale loss without confidence head, aligned with current model.

    Expects 5 predictions from the model (s32, s16, s8, s4, full_ref), where full_ref
    is already the full-res output of the full_refine block (as in inference).

    Args:
        preds: [flow_s32, flow_s16_ref, flow_s8_ref, flow_s4_ref, flow_full]
        gt: [B, 2, Hf, Wf]
        img_in: [B, C, Hf, Wf]
        valid_mask: [B, 1, Hf, Wf]
    """

    # Without confidence, just a list of predictions (4 scales + full)
    assert isinstance(preds, (list, tuple)) and len(preds) == 5, f"Expected 5 predictions, got {len(preds)}"
    assert gt.dim() == 4 and gt.size(1) == 2
    assert img_in.dim() == 4 and img_in.size(1) in (1, 3)

    B, _, Hf, Wf = gt.shape

    H32, W32 = Hf // 32, Wf // 32
    H16, W16 = Hf // 16, Wf // 16
    H8, W8 = Hf // 8, Wf // 8
    H4, W4 = Hf // 4, Wf // 4

    # GT flow at different resolutions with correct scaling
    gt_s32 = F.interpolate(gt, size=(H32, W32), mode='area') / 32.0
    gt_s16 = F.interpolate(gt, size=(H16, W16), mode='area') / 16.0
    gt_s8 = F.interpolate(gt, size=(H8, W8), mode='area') / 8.0
    gt_s4 = F.interpolate(gt, size=(H4, W4), mode='area') / 4.0

    # Unpack 5 predictions from the model (multi-scale + full)
    p_s32, p_s16, p_s8, p_s4, p_full = preds

    # Mask for Loss and EPE
    vm_full = torch.ones(B, 1, Hf, Wf, device=gt.device, dtype=gt.dtype)
    if valid_mask is not None:
        vm_full = vm_full * valid_mask.float()

    # Mask scaling
    vm32 = F.interpolate(vm_full, size=(H32, W32), mode='nearest')
    vm16 = F.interpolate(vm_full, size=(H16, W16), mode='nearest')
    vm8 = F.interpolate(vm_full, size=(H8, W8), mode='nearest')
    vm4 = F.interpolate(vm_full, size=(H4, W4), mode='nearest')

    def masked_mean_charbonnier(x_epe, m):
        m_sq = m.squeeze(1)
        loss_val = charbonnier(x_epe)
        return (loss_val * m_sq).sum() / (m_sq.sum() + 1e-6)

    def masked_mean_epe(x_epe, m):
        m_sq = m.squeeze(1)
        return (x_epe * m_sq).sum() / (m_sq.sum() + 1e-6)

    # Loss at each scale
    loss_s32 = masked_mean_charbonnier(endpoint_error(p_s32, gt_s32), vm32)
    loss_s16 = masked_mean_charbonnier(endpoint_error(p_s16, gt_s16), vm16)
    loss_s8 = masked_mean_charbonnier(endpoint_error(p_s8, gt_s8), vm8)
    loss_s4 = masked_mean_charbonnier(endpoint_error(p_s4, gt_s4), vm4)

    # EPE at full-res
    epe_map = endpoint_error(p_full, gt)
    epe = masked_mean_epe(epe_map, vm_full)

    # Loss at full-res: directly on full_ref output of the model, with reweighting by error_map
    # (higher weight for pixels with larger error, no gradients through weights)
    epe_norm = epe_map / (epe_map.mean() + 1e-6)
    epe_w = torch.clamp(epe_norm.detach(), 0.5, 2.0)
    loss_full = masked_mean_charbonnier(epe_map * epe_w, vm_full)

    # Edge-Aware Smoothness Loss (enhanced method using gradient-based weights)
    edge_smooth = edge_aware_smoothness_loss(p_full, img_in)

    # FT-all metric (Sintel-style bad-pixel rate on full-res)
    gt_mag = torch.sqrt((gt ** 2).sum(1))  # [B,H,W]
    thr_abs = 3.0
    thr_rel = 0.05
    bad = (epe_map > thr_abs) & ((epe_map / (gt_mag + 1e-6)) > thr_rel)
    ft_all = (bad.float() * vm_full.squeeze(1)).sum() / (vm_full.sum() + 1e-6)

    # Final loss
    # Smoothness sm is no longer used in total (removed from loss).
    # Keep only edge-aware smoothness, controlled by w_edge_smooth.
    total = (w32 * loss_s32 + w16 * loss_s16 + w8 * loss_s8 + w4 * loss_s4
             + w_full * loss_full + w_edge_smooth * edge_smooth)

    # Detailed logging (multi-scale EPE + loss components)
    logs = {
        'loss': float(total.item()),
    'epe': float(epe.item()),
    'ft_all': float(ft_all.item()),
        'epe_s32': float(masked_mean_epe(endpoint_error(p_s32, gt_s32), vm32).item()),
        'epe_s16': float(masked_mean_epe(endpoint_error(p_s16, gt_s16), vm16).item()),
        'epe_s8': float(masked_mean_epe(endpoint_error(p_s8, gt_s8), vm8).item()),
        'epe_s4': float(masked_mean_epe(endpoint_error(p_s4, gt_s4), vm4).item()),
        'loss_s32': float(loss_s32.item()),
        'loss_s16': float(loss_s16.item()),
        'loss_s8': float(loss_s8.item()),
        'loss_s4': float(loss_s4.item()),
        'loss_full': float(loss_full.item()),
        'loss_smooth': 0.0,
        'loss_edge_smooth': float(edge_smooth.item()),
    }

    return total, logs
