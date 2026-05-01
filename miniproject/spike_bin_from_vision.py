"""Spike binary masks from fly raw vision (same ``spikes_bin`` as obstacle debug)."""

from __future__ import annotations

from typing import Any, Literal, Sequence, TypedDict, Union
import cv2
import numpy as np
import matplotlib.pyplot as plt
from submission.controller import spikes_bin_raw_from_single_eye_rgb

RawVisionInput = Union[Sequence[np.ndarray], np.ndarray]


class SpikeBinVisionDict(TypedDict):
    """``left`` / ``right``: ``uint8`` (H, W) masks with values 0 or 255."""

    left: np.ndarray
    right: np.ndarray

def _eye_rgb_to_uint8(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.max() <= 1.0:
        img = (img * 255.0).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img
def _spike_forward_x_bounds(W: int, *, eye_is_left: bool, forward_frac: float) -> tuple[int, int]:
    """Inclusive horizontal slice kept for spike detection (forward / binocular overlap direction)."""
    fw = int(round(float(forward_frac) * W))
    fw = max(1, min(W, fw))
    if eye_is_left:
        xl = max(0, W - fw)
        xr = W - 1
    else:
        xl = 0
        xr = min(W - 1, fw - 1)
    return xl, max(xl, xr)
def _build_spike_ignore_mask(
    H: int,
    W: int,
    *,
    eye_is_left: bool,
    forward_frac: float,
    ignore_top_frac: float,
    ignore_bottom_band_frac: float,
) -> np.ndarray:
    """Ignore non-forward regions per eye (left eye → keep right side; right eye → keep left)."""
    ignore = np.zeros((H, W), np.uint8)
    if ignore_top_frac > 0:
        yt = max(0, int(ignore_top_frac * H))
        ignore[:yt, :] = 255
    if ignore_bottom_band_frac > 0:
        y_band = H - int(round(ignore_bottom_band_frac * H))
        y_band = max(0, min(H, y_band))
        ignore[y_band:, :] = 255

    xl, xr = _spike_forward_x_bounds(W, eye_is_left=eye_is_left, forward_frac=forward_frac)
    if eye_is_left:
        ignore[:, :xl] = 255
    elif xr + 1 < W:
        ignore[:, xr + 1 :] = 255
    return ignore

def _ground_mask_horizontal_plane(mask_green: np.ndarray, min_side: int, *, ground_open_frac_of_min_side: float, ground_open_min: int, ground_open_max: int) -> np.ndarray:
    """Estimate floor green as wide horizontal runs (not tall blobs).

    A circular opening treats a wide cone base like “ground” and subtracts it; a flat
    horizontal structuring element keeps vertically extended cones while still removing
    distant thin spikes vs. the planar floor band.
    """
    open_sz = int(round(ground_open_frac_of_min_side * min_side))
    open_sz = max(ground_open_min, min(ground_open_max, open_sz))
    # Wide horizontal opening: remove thick cone bases from the ground estimate (they are
    # narrower than the floor band in px) while keeping near-full-width floor runs.
    open_w = max(int(round(min_side * 0.30)), int(round(open_sz * 2.2)), ground_open_min + 4)
    open_w = min(open_w, min_side - 1)
    if open_w % 2 == 0:
        open_w -= 1
    open_w = max(3, open_w)
    open_h = max(3, min(13, max(3, open_w // 10)))
    if open_h % 2 == 0:
        open_h += 1
    k_ground = cv2.getStructuringElement(cv2.MORPH_RECT, (open_w, open_h))
    ground_mask = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, k_ground, iterations=1)
    ground_mask = cv2.morphologyEx(ground_mask, cv2.MORPH_CLOSE, k_ground, iterations=1)
    return ground_mask

def _compute_spikes_bin_binary_single_eye_rgb(
    img_rgb: np.ndarray,
    *,
    eye_is_left: bool = True,
    forward_frac: float = 0.58,
    ignore_top_frac: float = 0.0,
    ignore_bottom_band_frac: float = 0.22,
    hsv_green_low: tuple[int, ...] = (33, 18, 12),
    hsv_green_high: tuple[int, ...] = (92, 255, 255),
    leg_r_minus_g_min: int = 14,
    leg_blue_dom_margin: int = 22,
    ground_open_frac_of_min_side: float = 0.055,
    ground_open_min: int = 19,
    ground_open_max: int = 51,
) -> np.ndarray:
    """Same spike segmentation as ``detect_spike_centers_*`` before bottom-strip / CC peaks.

    Returns uint8 ``HxW`` with values ``0`` or ``1`` (before optional ``*255`` for visualization)."""
    rgb = _eye_rgb_to_uint8(img_rgb)
    H, W = rgb.shape[:2]
    min_side = min(H, W)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    low = np.array(hsv_green_low, dtype=np.uint8)
    high = np.array(hsv_green_high, dtype=np.uint8)
    mask_green = cv2.inRange(hsv, low, high)

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, k3, iterations=1)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, k3, iterations=1)

    ignore = _build_spike_ignore_mask(
        H,
        W,
        eye_is_left=eye_is_left,
        forward_frac=forward_frac,
        ignore_top_frac=ignore_top_frac,
        ignore_bottom_band_frac=ignore_bottom_band_frac,
    )
    mask_green = cv2.bitwise_and(mask_green, cv2.bitwise_not(ignore))

    rch = rgb[:, :, 0].astype(np.int16)
    gch = rgb[:, :, 1].astype(np.int16)
    bch = rgb[:, :, 2].astype(np.int16)
    legs_or_cyan = (rch > gch + int(leg_r_minus_g_min)) | (bch > gch + int(leg_blue_dom_margin))
    mask_green = cv2.bitwise_and(mask_green, np.where(legs_or_cyan, 0, 255).astype(np.uint8))

    ground_mask = _ground_mask_horizontal_plane(
        mask_green,
        min_side,
        ground_open_frac_of_min_side=ground_open_frac_of_min_side,
        ground_open_min=ground_open_min,
        ground_open_max=ground_open_max,
    )

    spikes_mask = cv2.subtract(mask_green, ground_mask)
    spikes_bin = (spikes_mask > 0).astype(np.uint8)
    spikes_bin = cv2.dilate(spikes_bin, k3, iterations=1)
    oz = int(round(ground_open_frac_of_min_side * min_side))
    oz = max(ground_open_min, min(ground_open_max, oz))
    v_close_h = max(7, min(21, oz + 4))
    if v_close_h % 2 == 0:
        v_close_h += 1
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (3, v_close_h))
    spikes_bin = cv2.morphologyEx(spikes_bin, cv2.MORPH_CLOSE, kv, iterations=1)
    return spikes_bin
def spikes_bin_raw_from_single_eye_rgb(
    img_rgb: np.ndarray,
    *,
    eye_is_left: bool = True,
    forward_frac: float = 0.58,
    ignore_top_frac: float = 0.0,
    ignore_bottom_band_frac: float = 0.22,
    hsv_green_low: tuple[int, ...] = (33, 18, 12),
    hsv_green_high: tuple[int, ...] = (92, 255, 255),
    leg_r_minus_g_min: int = 14,
    leg_blue_dom_margin: int = 22,
    ground_open_frac_of_min_side: float = 0.055,
    ground_open_min: int = 19,
    ground_open_max: int = 51,
) -> np.ndarray:
    """Binary spike mask as ``uint8`` ``0`` / ``255`` (same convention as debug ``spikes_bin``)."""
    b = _compute_spikes_bin_binary_single_eye_rgb(
        img_rgb,
        eye_is_left=eye_is_left,
        forward_frac=forward_frac,
        ignore_top_frac=ignore_top_frac,
        ignore_bottom_band_frac=ignore_bottom_band_frac,
        hsv_green_low=hsv_green_low,
        hsv_green_high=hsv_green_high,
        leg_r_minus_g_min=leg_r_minus_g_min,
        leg_blue_dom_margin=leg_blue_dom_margin,
        ground_open_frac_of_min_side=ground_open_frac_of_min_side,
        ground_open_min=ground_open_min,
        ground_open_max=ground_open_max,
    )
    return (b * 255).astype(np.uint8)
def split_get_raw_vision_into_left_right_images(
    raw_vision: RawVisionInput,
    *,
    layout: Literal["two_frames", "concat_along_width"] = "two_frames",
) -> tuple[np.ndarray, np.ndarray]:
    """Turn ``sim.get_raw_vision(fly_name)`` (or a stitched panorama) into two RGB images.

    Flygym returns **two separate** renders: ``[left_eye, right_eye]``. Some code stacks them
    with ``np.concatenate(..., axis=-2)`` → one wide image (left | right); use
    ``layout="concat_along_width"`` for that.

    Parameters
    ----------
    raw_vision :
        Either a sequence of two ``(H, W, 3)`` RGB arrays (normal ``get_raw_vision`` output),
        or a single ``(H, W_total, 3)`` array when ``layout="concat_along_width"``.
    layout :
        ``"two_frames"`` — indices ``0`` = left eye, ``1`` = right eye (simulator order).
        ``"concat_along_width"`` — split width in half: left half = left eye, right half = right eye.

    Returns
    -------
    left_eye_rgb, right_eye_rgb
        ``float`` or ``uint8`` ``(H, W, 3)``, same dtype/shape category as the inputs.
    """
    if layout == "concat_along_width":
        pan = np.asarray(raw_vision)
        if pan.ndim != 3 or pan.shape[-1] != 3:
            raise ValueError(
                "concat_along_width expects ndarray shape (H, W, 3); got "
                f"{getattr(pan, 'shape', None)}"
            )
        mid = pan.shape[1] // 2
        if mid < 1:
            raise ValueError("Panorama width too small to split into two eyes")
        left_eye_rgb = pan[:, :mid].copy()
        right_eye_rgb = pan[:, mid:].copy()
        return left_eye_rgb, right_eye_rgb

    seq = list(raw_vision)
    if len(seq) < 2:
        raise ValueError(
            "Expected sim.get_raw_vision(fly_name): a sequence of at least two RGB frames "
            f"[left_eye, right_eye]; got length {len(seq)}"
        )
    left_eye_rgb = np.asarray(seq[0])
    right_eye_rgb = np.asarray(seq[1])
    for name, im in (("left", left_eye_rgb), ("right", right_eye_rgb)):
        if im.ndim != 3 or im.shape[-1] != 3:
            raise ValueError(
                f"{name} eye image must have shape (H, W, 3); got {im.shape}"
            )
    return left_eye_rgb, right_eye_rgb


def spike_bin_masks_from_raw_vision(
    raw_vision: RawVisionInput,
    *,
    layout: Literal["two_frames", "concat_along_width"] = "two_frames",
    **kwargs: Any,
) -> SpikeBinVisionDict:
    """Compute spike masks for both eyes (same pipeline as debug ``spikes_bin``).

    Typical usage::

        masks = spike_bin_masks_from_raw_vision(sim.get_raw_vision(sim.fly.name))
        left_mask = masks["left"]
        right_mask = masks["right"]

    Parameters
    ----------
    raw_vision :
        Return value of ``sim.get_raw_vision(fly_name)``: ``list`` / tuple of two RGB images,
        **or** a single wide RGB image if you used ``layout="concat_along_width"``.
    layout :
        See :func:`split_get_raw_vision_into_left_right_images`.
    **kwargs :
        Passed to :func:`submission.controller.spikes_bin_raw_from_single_eye_rgb`
        (HSV bounds, ``forward_frac``, ``ignore_bottom_band_frac``, etc.).

    Returns
    -------
    dict
        ``{"left": spikes_bin_left, "right": spikes_bin_right}`` — ``uint8`` ``0``/``255``,
        full resolution per eye, **before** steering-only bottom crop used for peak detection.
    """
    left_eye_rgb, right_eye_rgb = split_get_raw_vision_into_left_right_images(
        raw_vision, layout=layout
    )
    return {
        "left": spikes_bin_raw_from_single_eye_rgb(
            left_eye_rgb, eye_is_left=True, **kwargs
        ),
        "right": spikes_bin_raw_from_single_eye_rgb(
            right_eye_rgb, eye_is_left=False, **kwargs
        ),
    }
def find_largest_white_stack(img, direction='left-to-right'):
    """
    Scans an image from left to right (or right to left) to find the column 
    with the largest vertical stack of white pixels.
    
    Args:
        img: Input image (numpy array). White pixels are assumed to be (255, 255, 255) or >200 in grayscale
        direction: 'left-to-right' or 'right-to-left'
    
    Returns:
        dict with keys:
            'column': x-coordinate of the largest white stack
            'max_stack_height': height of the largest consecutive white pixels in that column
            'start_row': starting row index of the largest stack
            'end_row': ending row index of the largest stack
    """
    # Convert to grayscale if it's a color image
    if len(img.shape) == 3:
        gray = np.mean(img, axis=2)
    else:
        gray = img
    
    # Create binary mask: white pixels (threshold > 200)
    binary = gray > 200
    
    height, width = binary.shape
    
    # Determine scan direction
    columns_to_scan = range(width-1, -1, -1) if direction == 'right-to-left' else range(width)
    
    max_height = 0
    best_column = 0
    best_start_row = 0
    best_end_row = 0
    
    # Scan each column
    for col in columns_to_scan:
        white_pixels = binary[:, col]
        
        # Find consecutive white pixels
        current_height = 0
        current_start = 0
        
        for row in range(height):
            if white_pixels[row]:
                if current_height == 0:
                    current_start = row
                current_height += 1
            else:
                if current_height > max_height:
                    max_height = current_height
                    best_column = col
                    best_start_row = current_start
                    best_end_row = row - 1
                current_height = 0
        
        # Check final segment
        if current_height > max_height:
            max_height = current_height
            best_column = col
            best_start_row = current_start
            best_end_row = height - 1
    
    return {
        'column': best_column,
        'max_stack_height': max_height,
        'start_row': best_start_row,
        'end_row': best_end_row
    }
def find_largest_stack_from_base_img(img, direction='left-to-right', layout="concat_along_width",print_debug=False):
    spike_bins = spike_bin_masks_from_raw_vision(raw_vision=img,layout=layout)
    result = {}
    result['left'] = find_largest_white_stack(spike_bins["left"], direction=direction)
    result['right'] = find_largest_white_stack(spike_bins["right"], direction=direction)
    if print_debug:
        fig,ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))
        ax[0].imshow(spike_bins["left"], cmap='gray')
        ax[0].axvline(x=result['left']['column'], color='red', linewidth=2, label=f"Largest stack at column {result['left']['column']}")
        ax[0].axhline(y=result['left']['start_row'], color='yellow', linewidth=1, linestyle='--', alpha=0.7)
        ax[0].axhline(y=result['left']['end_row'], color='yellow', linewidth=1, linestyle='--', alpha=0.7)
        ax[0].set_title(f"Largest Vertical White Stack: {result['left']['max_stack_height']} pixels")
        ax[1].imshow(spike_bins["right"], cmap='gray')
        ax[1].axvline(x=result['right']['column'], color='red', linewidth=2, label=f"Largest stack at column {result['right']['column']}")
        ax[1].axhline(y=result['right']['start_row'], color='yellow', linewidth=1, linestyle='--', alpha=0.7)
        ax[1].axhline(y=result['right']['end_row'], color='yellow', linewidth=1, linestyle='--', alpha=0.7)
        ax[1].set_title(f"Largest Vertical White Stack: {result['right']['max_stack_height']} pixels")
    return result