"""
Common utilities for optical flow processing.

This module provides shared functions for image/flow preprocessing and dataset handling,
used across training (v1.py), testing (test_accurancy.py), and conversion (convert_cv181x.sh).

Functions:
    - resize_img: Aspect-preserving image resize (scale by height, pad/crop width)
    - resize_mask_nearest: Aspect-preserving mask resize with nearest interpolation
    - resize_flow: Aspect-preserving flow resize with proper vector scaling
    - read_flo: Read Middlebury .flo optical flow files
    - read_kitti_png_flow: Read KITTI 16-bit PNG flow files

Author: Ruslan Ryngach
2025
"""

import struct
import numpy as np
import cv2
from typing import Tuple, Optional


def resize_img(img: np.ndarray, size_hw: Tuple[int, int], x_offset: Optional[int] = None) -> np.ndarray:
    target_h, target_w = size_hw
    H0, W0 = img.shape[:2]
    scale = target_h / H0
    new_w = int(round(W0 * scale))

    resized = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_LINEAR)

    if new_w < target_w:
        pad_left = (target_w - new_w) // 2
        pad_right = target_w - new_w - pad_left
        if img.ndim == 3:
            resized = np.pad(resized, ((0, 0), (pad_left, pad_right), (0, 0)), mode='edge')
        else:
            resized = np.pad(resized, ((0, 0), (pad_left, pad_right)), mode='edge')
        return resized
    if x_offset is None:
        x_offset = (new_w - target_w) // 2
    else:
        x_offset = min(max(0, x_offset), new_w - target_w)
    return resized[:, x_offset:x_offset + target_w]


def resize_mask_nearest(mask_bool: np.ndarray, size_hw: Tuple[int, int], x_offset: Optional[int] = None) -> np.ndarray:
    if mask_bool.ndim == 3:
        mask_bool = mask_bool[..., 0]
    target_h, target_w = size_hw
    H0, W0 = mask_bool.shape[:2]
    scale = target_h / H0
    new_w = int(round(W0 * scale))
    mask_u8 = mask_bool.astype(np.uint8)
    resized = cv2.resize(mask_u8, (new_w, target_h), interpolation=cv2.INTER_NEAREST)
    if new_w < target_w:
        pad_left = (target_w - new_w) // 2
        pad_right = target_w - new_w - pad_left
        resized = np.pad(resized, ((0, 0), (pad_left, pad_right)), mode='edge')
        return resized.astype(bool)
    if x_offset is None:
        x_offset = (new_w - target_w) // 2
    else:
        x_offset = min(max(0, x_offset), new_w - target_w)
    return resized[:, x_offset:x_offset + target_w].astype(bool)


def resize_flow(flow: np.ndarray, size_hw: Tuple[int, int], x_offset: Optional[int] = None) -> np.ndarray:
    """Resize flow supporting both [H,W,2] and [2,H,W] layouts, returns same layout"""
    assert flow.ndim == 3, f"flow must be 3D, got {flow.shape}"
    
    # Detect layout
    if flow.shape[-1] == 2:
        # [H,W,2] layout
        is_hwc = True
        H0, W0 = flow.shape[0], flow.shape[1]
    elif flow.shape[0] == 2:
        # [2,H,W] layout
        is_hwc = False
        H0, W0 = flow.shape[1], flow.shape[2]
    else:
        raise ValueError(f"flow must be [H,W,2] or [2,H,W], got {flow.shape}")
    
    target_h, target_w = size_hw
    scale_y = target_h / H0
    new_w = int(round(W0 * scale_y))
    scale_x = new_w / W0
    
    # Convert to [H,W,2] for cv2.resize
    if not is_hwc:
        flow = np.moveaxis(flow, 0, -1)  # [2,H,W] -> [H,W,2]
    
    # Resize and scale flow components
    flow_resized = cv2.resize(flow, (new_w, target_h), interpolation=cv2.INTER_LINEAR)
    flow_resized[:, :, 0] *= scale_x  # scale u by width scale
    flow_resized[:, :, 1] *= scale_y  # scale v by height scale
    
    # Handle width padding or cropping
    if new_w < target_w:
        pad_left = (target_w - new_w) // 2
        pad_right = target_w - new_w - pad_left
        flow_resized = np.pad(flow_resized, ((0, 0), (pad_left, pad_right), (0, 0)), mode='edge')
    else:
        if x_offset is None:
            x_offset = (new_w - target_w) // 2
        else:
            x_offset = min(max(0, x_offset), new_w - target_w)
        flow_resized = flow_resized[:, x_offset:x_offset + target_w, :]
    
    # Convert back to original layout
    if not is_hwc:
        flow_resized = np.moveaxis(flow_resized, -1, 0)  # [H,W,2] -> [2,H,W]
    
    return flow_resized.astype(np.float32)


def resize_flow_direct(flow: np.ndarray, size_hw: Tuple[int, int], interpolation=cv2.INTER_LINEAR) -> np.ndarray:
    """
    Resize flow directly to target size WITHOUT preserving aspect ratio.
    Properly scales flow vectors by the resize ratios.
    
    Args:
        flow: Flow in [H,W,2] or [2,H,W] format
        size_hw: Target (height, width)
        interpolation: OpenCV interpolation method (default: cv2.INTER_LINEAR)
        
    Returns:
        Resized flow in same format as input
    """
    assert flow.ndim == 3, f"flow must be 3D, got {flow.shape}"
    
    # Detect layout
    if flow.shape[-1] == 2:
        # [H,W,2] layout
        is_hwc = True
        H0, W0 = flow.shape[0], flow.shape[1]
        flow_hwc = flow
    elif flow.shape[0] == 2:
        # [2,H,W] layout
        is_hwc = False
        H0, W0 = flow.shape[1], flow.shape[2]
        flow_hwc = np.moveaxis(flow, 0, -1)  # [2,H,W] -> [H,W,2]
    else:
        raise ValueError(f"flow must be [H,W,2] or [2,H,W], got {flow.shape}")
    
    target_h, target_w = size_hw
    
    # Calculate scale factors
    scale_x = target_w / W0
    scale_y = target_h / H0
    
    # Resize flow (cv2.resize expects (width, height))
    flow_resized = cv2.resize(flow_hwc, (target_w, target_h), interpolation=interpolation)
    
    # Scale flow vectors
    flow_resized[:, :, 0] *= scale_x  # scale u by width scale
    flow_resized[:, :, 1] *= scale_y  # scale v by height scale
    
    # Convert back to original layout
    if not is_hwc:
        flow_resized = np.moveaxis(flow_resized, -1, 0)  # [H,W,2] -> [2,H,W]
    
    return flow_resized.astype(np.float32)


def read_flo(path: str) -> np.ndarray:
    """Read .flo optical flow file.
    
    Args:
        path: Path to .flo file
        
    Returns:
        Flow array in [2, H, W] format
    """
    with open(path, 'rb') as f:
        magic = struct.unpack('f', f.read(4))[0]
        if magic != 202021.25:
            raise ValueError(f'Invalid .flo magic in {path}')
        w = struct.unpack('i', f.read(4))[0]
        h = struct.unpack('i', f.read(4))[0]
        data = np.frombuffer(f.read(h * w * 8), dtype=np.float32)
        flow = data.reshape(h, w, 2).transpose(2, 0, 1).astype(np.float32)
        return flow


def read_kitti_png_flow(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read KITTI optical flow from 16-bit PNG file.
    
    Args:
        path: Path to KITTI flow PNG file
        
    Returns:
        Tuple of (flow [H, W, 2] float32, valid_mask [H, W] boolean)
    """
    flow_img = cv2.imread(path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    
    if flow_img is None:
        raise FileNotFoundError(f"Cannot read KITTI flow file: {path}")
    
    assert flow_img.dtype == np.uint16, f"Expected uint16, got {flow_img.dtype}"
    
    # cv2 returns BGR; KITTI stores R = u, G = v, B = valid
    valid = (flow_img[:, :, 0] > 0).astype(bool)
    
    u = (flow_img[:, :, 2].astype(np.float32) - 32768.0) / 64.0
    v = (flow_img[:, :, 1].astype(np.float32) - 32768.0) / 64.0
    
    flow = np.stack([u, v], axis=-1).astype(np.float32)
    
    return flow, valid
