import numpy as np
import cv2
from miniproject.simulation import MiniprojectSimulation
from enum import Enum, auto
import matplotlib.pyplot as plt

SHOW_PRINTS = True
print_frequency = 5000
GO_STRAIGHT_THRESHOLD = 2 / 100


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


def _spike_roi_rect_xyxy(
    H: int,
    W: int,
    *,
    eye_is_left: bool,
    forward_frac: float,
    ignore_top_frac: float,
    ignore_bottom_band_frac: float,
) -> tuple[int, int, int, int]:
    """Inclusive (x0, y0, x1, y1): forward strip for this eye + optional top/bottom bands."""
    yt = max(0, int(ignore_top_frac * H))
    if ignore_bottom_band_frac > 0:
        yb = H - int(round(ignore_bottom_band_frac * H)) - 1
    else:
        yb = H - 1
    yb = max(yt, min(H - 1, yb))

    xl, xr = _spike_forward_x_bounds(W, eye_is_left=eye_is_left, forward_frac=forward_frac)
    return (xl, yt, xr, yb)


def _draw_spike_roi_rectangle(img: np.ndarray, roi_xyxy: tuple[int, int, int, int]) -> np.ndarray:
    """BGR/RGB color rectangle drawn in red (RGB order for matplotlib imshow)."""
    x0, y0, x1, y1 = roi_xyxy
    out = img.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
    cv2.rectangle(out, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), 2)
    return out


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


def _strip_bottom_rows_for_peak_detection(
    spikes_bin: np.ndarray,
    H: int,
    steering_bottom_ignore_frac: float,
) -> np.ndarray:
    """Clear bottom rows only for peak finding / steering; keep raw ``spikes_bin`` for debug masks."""
    if steering_bottom_ignore_frac <= 0:
        return spikes_bin
    out = spikes_bin.copy()
    y0 = H - int(round(float(steering_bottom_ignore_frac) * H))
    y0 = max(0, min(H, y0))
    out[y0:, :] = 0
    return out


def _spike_centers_from_tallest_components(
    spikes_for_peaks: np.ndarray,
    *,
    min_area: int,
    min_height: int,
    max_width_over_height: float,
    max_spikes: int,
    landmark_y_frac: float,
) -> list[tuple[int, int]]:
    """One landmark per 8-connected blob, ranked by **pixel** vertical span ``ymax - ymin + 1``.

    ``landmark_y_frac`` interpolates between the top and bottom **white pixels** of the blob
    (0 ≈ top row of the component, 1 ≈ bottom row). We avoid ``CC_STAT_TOP`` alone: it sits at
    the bbox edge and jumps into sky when noise merges or the bbox is loose."""
    if spikes_for_peaks.max() == 0:
        return []
    bin_u8 = (spikes_for_peaks > 0).astype(np.uint8)
    n_lbl, labels, stats, _cents = cv2.connectedComponentsWithStats(bin_u8, connectivity=8)
    ranked: list[tuple[int, int, int, int]] = []
    frac = float(np.clip(landmark_y_frac, 0.0, 1.0))
    for i in range(1, n_lbl):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        ys = np.where(labels == i)[0]
        xs = np.where(labels == i)[1]
        ymin = int(ys.min())
        ymax = int(ys.max())
        xmin = int(xs.min())
        xmax = int(xs.max())
        span_y = ymax - ymin + 1
        span_x = xmax - xmin + 1
        if span_y < min_height:
            continue
        if float(span_x) / float(span_y) > max_width_over_height:
            continue
        ly = ymin + int(round(frac * float(ymax - ymin)))
        band = (ys >= ly - 1) & (ys <= ly + 1)
        if np.any(band):
            apex_x = int(np.median(xs[band]))
        else:
            apex_x = int(np.median(xs))
        apex_y = ly
        ranked.append((span_y, area, apex_x, apex_y))
    ranked.sort(key=lambda t: (-t[0], -t[1], -t[3]))
    return [(cx, cy) for _, _, cx, cy in ranked[:max_spikes]]


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


def detect_spike_centers_single_eye_rgb(
    img_rgb: np.ndarray,
    *,
    eye_is_left: bool = True,
    forward_frac=0.58,
    hsv_green_low=(33, 18, 12),
    hsv_green_high=(92, 255, 255),
    ignore_top_frac=0.0,
    ignore_bottom_band_frac=0.22,
    steering_bottom_ignore_frac=0.08,
    leg_r_minus_g_min=14,
    leg_blue_dom_margin=22,
    ground_open_frac_of_min_side=0.055,
    ground_open_min=19,
    ground_open_max=51,
    spike_cc_min_area=28,
    spike_cc_min_height=5,
    spike_cc_max_width_over_height=14.0,
    spike_cc_max_instances=10,
    spike_cc_landmark_y_frac=0.22,
) -> list[tuple[int, int]]:
    """Spikes (brins) dans un seul œil — image RGB (sortie simulateur).

    Centres = un point par composante connexe; tri par étendue verticale réelle des pixels;
    la position suit ``landmark_y_frac`` entre le haut et le bas du blob."""
    spikes_bin = _compute_spikes_bin_binary_single_eye_rgb(
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
    H, W = spikes_bin.shape[:2]

    if spikes_bin.max() == 0:
        return []

    spikes_for_peaks = _strip_bottom_rows_for_peak_detection(
        spikes_bin, H, steering_bottom_ignore_frac
    )
    if spikes_for_peaks.max() == 0:
        return []

    return _spike_centers_from_tallest_components(
        spikes_for_peaks,
        min_area=int(spike_cc_min_area),
        min_height=int(spike_cc_min_height),
        max_width_over_height=float(spike_cc_max_width_over_height),
        max_spikes=int(spike_cc_max_instances),
        landmark_y_frac=float(spike_cc_landmark_y_frac),
    )


def detect_spike_debug_single_eye_rgb(
    img_rgb: np.ndarray,
    *,
    eye_is_left: bool = True,
    forward_frac=0.58,
    hsv_green_low=(33, 18, 12),
    hsv_green_high=(92, 255, 255),
    ignore_top_frac=0.0,
    ignore_bottom_band_frac=0.22,
    leg_r_minus_g_min=14,
    leg_blue_dom_margin=22,
    ground_open_frac_of_min_side=0.055,
    ground_open_min=19,
    ground_open_max=51,
    steering_bottom_ignore_frac=0.08,
    spike_cc_min_area=28,
    spike_cc_min_height=5,
    spike_cc_max_width_over_height=14.0,
    spike_cc_max_instances=10,
    spike_cc_landmark_y_frac=0.22,
):
    """Version debug: retourne (centers, debug_dict) pour visualiser les masques."""
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
    mask_green2 = cv2.bitwise_and(mask_green, cv2.bitwise_not(ignore))

    rch = rgb[:, :, 0].astype(np.int16)
    gch = rgb[:, :, 1].astype(np.int16)
    bch = rgb[:, :, 2].astype(np.int16)
    legs_or_cyan = (rch > gch + int(leg_r_minus_g_min)) | (bch > gch + int(leg_blue_dom_margin))
    mask_green2 = cv2.bitwise_and(mask_green2, np.where(legs_or_cyan, 0, 255).astype(np.uint8))

    roi_xyxy = _spike_roi_rect_xyxy(
        H,
        W,
        eye_is_left=eye_is_left,
        forward_frac=forward_frac,
        ignore_top_frac=ignore_top_frac,
        ignore_bottom_band_frac=ignore_bottom_band_frac,
    )

    ground_mask = _ground_mask_horizontal_plane(
        mask_green2,
        min_side,
        ground_open_frac_of_min_side=ground_open_frac_of_min_side,
        ground_open_min=ground_open_min,
        ground_open_max=ground_open_max,
    )

    spikes_mask = cv2.subtract(mask_green2, ground_mask)
    spikes_bin = (spikes_mask > 0).astype(np.uint8)
    spikes_bin = cv2.dilate(spikes_bin, k3, iterations=1)
    oz = int(round(ground_open_frac_of_min_side * min_side))
    oz = max(ground_open_min, min(ground_open_max, oz))
    v_close_h = max(7, min(21, oz + 4))
    if v_close_h % 2 == 0:
        v_close_h += 1
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (3, v_close_h))
    spikes_bin = cv2.morphologyEx(spikes_bin, cv2.MORPH_CLOSE, kv, iterations=1)

    spikes_for_peaks = _strip_bottom_rows_for_peak_detection(
        spikes_bin, H, steering_bottom_ignore_frac
    )

    dist = np.zeros((H, W), dtype=np.float32)
    peaks_img = np.zeros((H, W), dtype=np.uint8)
    centers: list[tuple[int, int]] = []
    if spikes_for_peaks.max() != 0:
        dist = cv2.distanceTransform(spikes_for_peaks, cv2.DIST_L2, 5)
        centers = _spike_centers_from_tallest_components(
            spikes_for_peaks,
            min_area=int(spike_cc_min_area),
            min_height=int(spike_cc_min_height),
            max_width_over_height=float(spike_cc_max_width_over_height),
            max_spikes=int(spike_cc_max_instances),
            landmark_y_frac=float(spike_cc_landmark_y_frac),
        )
        for x, y in centers:
            x0, x1 = max(0, x - 1), min(W, x + 2)
            y0, y1 = max(0, y - 1), min(H, y + 2)
            peaks_img[y0:y1, x0:x1] = 255

    dbg = {
        "rgb": rgb,
        "mask_green_raw": mask_green,
        "mask_green": mask_green2,
        "ground_mask": ground_mask,
        "spikes_mask": spikes_mask,
        "spikes_bin": (spikes_bin * 255).astype(np.uint8),
        "spikes_bin_for_peaks": (spikes_for_peaks * 255).astype(np.uint8),
        "dist": dist,
        "peaks": peaks_img,
        "roi_rect_xyxy": roi_xyxy,
        "peak_use_y0": (
            H - int(round(float(steering_bottom_ignore_frac) * H))
            if steering_bottom_ignore_frac > 0
            else H
        ),
    }
    return centers, dbg


def lateral_spike_threat(
    centers: list[tuple[int, int]],
    W: int,
    H: int,
    *,
    y_horizon_frac=0.10,
    x_forward_bounds: tuple[int, int] | None = None,
) -> tuple[float, float]:
    """Retourne (menace_gauche, menace_droite) dans l'image (somme pondérée).

    ``y_horizon_frac`` petit = on garde les spikes haut dans l'image (souvent le cas
    en montée ou quand les brins se découpent sur le ciel). Une valeur trop grande
    les excluait et la mouche « ne voyait » plus l'obstacle.

    Si ``x_forward_bounds`` (xl, xr) est fourni (ROI avant par œil), ``u`` est
    normalisé dans cette bande pour garder un gauche/droit significatif."""
    threat_left = 0.0
    threat_right = 0.0
    y_cut = int(y_horizon_frac * H)
    if x_forward_bounds is not None:
        xa, xb = int(x_forward_bounds[0]), int(x_forward_bounds[1])
        span = float(max(1, xb - xa))
    else:
        xa, xb, span = 0, W - 1, float(max(1, W - 1))
    for x, y in centers:
        if y < y_cut:
            continue
        if x_forward_bounds is not None:
            u = ((float(x) - xa) / span) - 0.5
        else:
            u = (x / float(W)) - 0.5
        closeness = (y / float(H)) ** 2
        center_weight = 1.0 / (abs(u) + 0.15)
        w = closeness * center_weight
        if u < 0:
            threat_left += w
        else:
            threat_right += w
    return threat_left, threat_right


def lateral_spike_dominant_peaks_xy(
    centers: list[tuple[int, int]],
    W: int,
    H: int,
    *,
    y_horizon_frac: float = 0.10,
    x_forward_bounds: tuple[int, int] | None = None,
) -> tuple[tuple[int, int] | None, tuple[int, int] | None, float, float]:
    """Pixels with largest threat weight on each lateral half (same metric as lateral_spike_threat).

    Weight favors spikes lower in the image (``closeness``) and nearer strip centre."""
    max_left_w = 0.0
    max_right_w = 0.0
    left_xy: tuple[int, int] | None = None
    right_xy: tuple[int, int] | None = None
    y_cut = int(y_horizon_frac * H)
    if x_forward_bounds is not None:
        xa, xb = int(x_forward_bounds[0]), int(x_forward_bounds[1])
        span = float(max(1, xb - xa))
    else:
        xa, xb, span = 0, W - 1, float(max(1, W - 1))
    for x, y in centers:
        if y < y_cut:
            continue
        if x_forward_bounds is not None:
            u = ((float(x) - xa) / span) - 0.5
        else:
            u = (x / float(W)) - 0.5
        closeness = (y / float(H)) ** 2
        center_weight = 1.0 / (abs(u) + 0.15)
        w = closeness * center_weight
        xi, yi = int(x), int(y)
        if u < 0:
            if w > max_left_w:
                max_left_w = w
                left_xy = (xi, yi)
        else:
            if w > max_right_w:
                max_right_w = w
                right_xy = (xi, yi)
    return left_xy, right_xy, max_left_w, max_right_w


def lateral_spike_threat_max(
    centers: list[tuple[int, int]],
    W: int,
    H: int,
    *,
    y_horizon_frac: float = 0.10,
    x_forward_bounds: tuple[int, int] | None = None,
) -> tuple[float, float]:
    """Strongest single spike on each lateral half (same weighting as sums in lateral_spike_threat)."""
    _, _, ml, mr = lateral_spike_dominant_peaks_xy(
        centers,
        W,
        H,
        y_horizon_frac=y_horizon_frac,
        x_forward_bounds=x_forward_bounds,
    )
    return ml, mr


class State(Enum):
    FOLLOW_SCENT = auto()
    AVOID_OBSTACLE = auto()
    DODGE_DRAGON = auto()


class Controller:
    def __init__(self, sim: MiniprojectSimulation):
        from flygym.examples.locomotion import TurningController

        self.turning_controller = TurningController(sim.timestep)
        self.state = State.FOLLOW_SCENT
        self.show_prints = False

        self.odor_smooth = None
        self.alpha = 0.0005

        self.obstacle_threshold = 0.015
        self.sky_region_ratio = 0.55        #allow to modify the % of height seen from the sky (the smaller the more high we see)
        self.min_green_height_ratio = 0.10

        # Évitement basé sur les spikes (HSV + distance transform) — seuil sur la menace par œil
        self.use_spike_obstacles = True
        self.spike_obstacle_threshold = 2.5
        self._spike_threat_left_img = 0.0
        self._spike_threat_right_img = 0.0
        self._spike_max_threat_left_img = 0.0
        self._spike_max_threat_right_img = 0.0

        # Préférer le pic dominant par moitié (max) avant la somme pour choisir le virage.
        self.spike_use_max_for_steering = True
        self.spike_max_steering_floor = 0.035
        self.spike_max_rel_margin = 0.10

        # Vision mode: "spikes_raw" (OpenCV + get_raw_vision) ou "ommatidia" (plus rapide).
        # "ommatidia" garde la logique AVOID/FOLLOW mais remplace la détection obstacle.
        self.vision_mode = "spikes_raw"
        #self.vision_mode = "ommatidia"

        # Bande du haut ignorée pour la menace spike seulement (voir lateral_spike_threat).
        self.spike_y_horizon_frac = 0.10

        # Détection spikes: champ avant par œil (œil gauche → partie droite de l’image, œil droit → gauche).
        self.spike_forward_frac = 0.58
        self.spike_ignore_top_frac = 0.0
        # Bande du bas (fraction de la hauteur), ignorée sur toute la largeur — pattes / sol.
        self.spike_ignore_bottom_band_frac = 0.22

        # Après le masque complet: bande du bas retirée seulement pour pics / steering (spikes_bin debug inchangé).
        self.spike_steering_bottom_ignore_frac = 0.08

        # Un centre par blob connexe, tri par hauteur de bbox (blobs « plus hauts » ≈ plus proches).
        self.spike_cc_min_area = 28
        self.spike_cc_min_height = 5
        self.spike_cc_max_width_over_height = 14.0
        self.spike_cc_max_instances = 10
        # 0 = haut du blob (ymin), 1 = bas (ymax); éviter 0 si la fusion tire le haut vers le ciel.
        self.spike_cc_landmark_y_frac = 0.22

        # Si les spikes ne ressortent pas (pente, fusion avec le sol), garder l’heuristique ciel.
        self.spike_use_legacy_fallback = True
        self._legacy_left_score = 0.0
        self._legacy_right_score = 0.0

        # Ommatidia-based obstacle cues (fast path)
        self._omma_ready = False
        self._omma_cols = None  # list[np.ndarray] per column of ommatidia ids (0..N-1)
        self._omma_rows = None  # list[np.ndarray] per row of ommatidia ids (0..N-1)
        self.omma_row_top_frac = 0.55  # use only top rows (sky band) for obstacle-on-sky cue
        self.omma_green_dom_threshold = 0.10  # threshold on (g-b)/(g+b+eps)
        self.omma_tall_col_ratio_threshold = 0.18  # per-column fraction to mark column as obstacle
        self.omma_obstacle_threshold = 0.12  # threshold on fraction of "tall" columns
        self._omma_threat_left_img = 0.0
        self._omma_threat_right_img = 0.0

        # Perf: get_raw_vision + spike CV sont coûteux. Mettre > 1 accélère les longues runs.
        self.vision_update_every_n_steps = 20
        # Pendant AVOID: recalculer la vision plus souvent (état précédent).
        self.vision_every_step_while_avoiding = False
        self._cached_left_score = 0.0
        self._cached_right_score = 0.0
        self._cached_spike_threat_left_img = 0.0
        self._cached_spike_threat_right_img = 0.0
        self._cached_spike_max_threat_left_img = 0.0
        self._cached_spike_max_threat_right_img = 0.0
        self._cached_legacy_left = 0.0
        self._cached_legacy_right = 0.0

        self._prev_state = State.FOLLOW_SCENT
        self._avoid_streak = 0
        self.avoid_stuck_blend_after_steps = 420
        self.avoid_stuck_blend_ramp_steps = 280

        # Adhésion: forcer "collé" quand on évite (réduit les retournements sur obstacles).
        self.force_full_adhesion_when_avoiding = False
        self.full_adhesion_cooldown_steps = 80
        self._full_adhesion_timer = 0

        # Prepare ommatidia lookup tables once (cheap; uses retina ommatidia_id_map).
        self._init_ommatidia_tables(sim)

        # Downhill / runaway speed: scale CPG drives using thorax velocity (physics gains speed on slopes).
        self.downhill_brake_enable = True
        self.downhill_vz_thresh = -0.018
        self.downhill_vz_brake_gain = 12.0
        self.downhill_speed_xy_thresh = 0.035
        self.downhill_drive_min_scale = 0.42
        self.downhill_vel_smooth_beta = 0.82
        self.downhill_urgent_vision_scale_threshold = 0.92

        self._cached_thorax_idx: int | None = None
        self._prev_thorax_p: np.ndarray | None = None
        self._thorax_vel_ema: np.ndarray | None = None

        #UNCOMMENT TO ACTIVATE TIME FOR DETECTION
        #self.avoid_duration = 300
        #self.avoid_timer = 0

    def _thorax_body_index(self, sim: MiniprojectSimulation) -> int:
        if self._cached_thorax_idx is None:
            try:
                segs = sim.fly.get_bodysegs_order()
                self._cached_thorax_idx = next(
                    (i for i, s in enumerate(segs) if s.name == "c_thorax"),
                    0,
                )
            except Exception:
                self._cached_thorax_idx = 0
        return self._cached_thorax_idx

    def _downhill_drive_scale_and_urgent(self, sim: MiniprojectSimulation) -> tuple[float, bool]:
        """Reduce stepping amplitude when descending / moving fast horizontally (terrain slopes)."""
        if not self.downhill_brake_enable:
            return 1.0, False

        idx = self._thorax_body_index(sim)
        p = np.asarray(sim.get_body_positions(sim.fly.name)[idx], dtype=np.float64)
        dt = float(sim.timestep)

        if self._prev_thorax_p is None:
            self._prev_thorax_p = p.copy()
            self._thorax_vel_ema = np.zeros(3, dtype=np.float64)
            return 1.0, False

        v_inst = (p - self._prev_thorax_p) / max(dt, 1e-12)
        self._prev_thorax_p = p.copy()
        beta = float(self.downhill_vel_smooth_beta)
        vel_ema = self._thorax_vel_ema
        if vel_ema is None:
            vel_ema = np.zeros(3, dtype=np.float64)
        self._thorax_vel_ema = beta * vel_ema + (1.0 - beta) * v_inst

        vx, vy, vz = self._thorax_vel_ema
        speed_xy = float(np.hypot(vx, vy))

        scale = 1.0
        vz_th = float(self.downhill_vz_thresh)
        descending = vz < vz_th
        if descending:
            dz = vz_th - float(vz)
            scale = max(
                float(self.downhill_drive_min_scale),
                1.0 - float(self.downhill_vz_brake_gain) * dz,
            )
        if descending and speed_xy > float(self.downhill_speed_xy_thresh):
            s_xy = max(
                float(self.downhill_drive_min_scale),
                float(self.downhill_speed_xy_thresh) / max(speed_xy, 1e-9),
            )
            scale = min(scale, s_xy)

        scale = float(np.clip(scale, float(self.downhill_drive_min_scale), 1.0))
        urgent = descending or scale <= float(self.downhill_urgent_vision_scale_threshold)
        return scale, urgent

    def _init_ommatidia_tables(self, sim: MiniprojectSimulation) -> None:
        """Precompute ommatidia grouping by row/col for fast obstacle cues."""
        try:
            retina = sim.world.fly_lookup[sim.fly.name].retina
            id_map = retina.ommatidia_id_map  # (nrows, ncols), values 0 or 1..N
        except Exception:
            return

        nrows, ncols = id_map.shape
        self._omma_nrows = nrows
        self._omma_ncols = ncols
        self._omma_col_xnorm = (np.arange(ncols, dtype=np.float32) / max(1, ncols - 1)) - 0.5
        cols: list[np.ndarray] = []
        for c in range(ncols):
            ids = np.unique(id_map[:, c])
            ids = ids[ids > 0] - 1
            cols.append(ids.astype(np.int32))

        rows: list[np.ndarray] = []
        for r in range(nrows):
            ids = np.unique(id_map[r, :])
            ids = ids[ids > 0] - 1
            rows.append(ids.astype(np.int32))

        self._omma_cols = cols
        self._omma_rows = rows
        self._omma_ready = True

    def _detect_obstacle_from_ommatidia(self, sim: MiniprojectSimulation) -> tuple[float, float]:
        """Fast obstacle cue using ommatidia readouts (no OpenCV)."""
        if not self._omma_ready:
            return 0.0, 0.0

        omma = sim.get_ommatidia_readouts(sim.fly.name)  # (2, N, 2)
        if omma.ndim != 3 or omma.shape[0] < 2:
            return 0.0, 0.0

        eps = 1e-8
        g = omma[:, :, 0]
        b = omma[:, :, 1]
        dom = (g - b) / (g + b + eps)  # green-dominance proxy in [-1,1]

        # Use only top rows to catch "spikes against sky"
        top_rows = int(self.omma_row_top_frac * self._omma_nrows)
        top_rows = max(1, min(self._omma_nrows, top_rows))
        top_ids = np.unique(np.concatenate(self._omma_rows[:top_rows]))

        def per_eye_score(eye_idx: int) -> tuple[float, float, float]:
            de = dom[eye_idx]
            tall_cols = 0
            threat_left = 0.0
            threat_right = 0.0
            for c, ids in enumerate(self._omma_cols):
                if ids.size == 0:
                    continue
                ids2 = ids[np.isin(ids, top_ids)]
                if ids2.size == 0:
                    continue
                ratio = float(np.mean(de[ids2] > self.omma_green_dom_threshold))
                if ratio > self.omma_tall_col_ratio_threshold:
                    tall_cols += 1
                # lateral threat: weight by ratio and closeness to center
                x = float(self._omma_col_xnorm[c])
                w = ratio / (abs(x) + 0.20)
                if x < 0:
                    threat_left += w
                else:
                    threat_right += w
            score = tall_cols / float(max(1, self._omma_ncols))
            return score, threat_left, threat_right

        s0, tl0, tr0 = per_eye_score(0)
        s1, tl1, tr1 = per_eye_score(1)
        self._omma_threat_left_img = tl0 + tl1
        self._omma_threat_right_img = tr0 + tr1
        return float(s0), float(s1)

    def step(self, sim: MiniprojectSimulation, step):
        self.show_prints = SHOW_PRINTS and step % print_frequency == 0

        raw_olfaction = sim.get_olfaction(sim.fly.name)

        if self.odor_smooth is None:
            self.odor_smooth = raw_olfaction.copy()
        else:
            self.odor_smooth = (
                (1 - self.alpha) * self.odor_smooth
                + self.alpha * raw_olfaction
            )

        drive_scale, downhill_urgent = self._downhill_drive_scale_and_urgent(sim)
        urgent_vision = (
            self.vision_every_step_while_avoiding
            and self._prev_state == State.AVOID_OBSTACLE
        ) or downhill_urgent
        if (
            self.vision_update_every_n_steps <= 1
            or step % self.vision_update_every_n_steps == 0
            or urgent_vision
        ):
            left_score, right_score = self.detect_green_obstacle(sim)
            self._cached_left_score = left_score
            self._cached_right_score = right_score
            self._cached_spike_threat_left_img = self._spike_threat_left_img
            self._cached_spike_threat_right_img = self._spike_threat_right_img
            self._cached_spike_max_threat_left_img = self._spike_max_threat_left_img
            self._cached_spike_max_threat_right_img = self._spike_max_threat_right_img
            self._cached_legacy_left = self._legacy_left_score
            self._cached_legacy_right = self._legacy_right_score
        else:
            left_score = self._cached_left_score
            right_score = self._cached_right_score
            self._spike_threat_left_img = self._cached_spike_threat_left_img
            self._spike_threat_right_img = self._cached_spike_threat_right_img
            self._spike_max_threat_left_img = self._cached_spike_max_threat_left_img
            self._spike_max_threat_right_img = self._cached_spike_max_threat_right_img
            self._legacy_left_score = self._cached_legacy_left
            self._legacy_right_score = self._cached_legacy_right

        if self.show_prints:
            print(f"Vision obstacle scores: L={left_score:.4f}, R={right_score:.4f}")
            if self.use_spike_obstacles:
                print(
                    f"Spike lateral threat (sum both eyes): "
                    f"gauche_image={self._spike_threat_left_img:.4f}, "
                    f"droite_image={self._spike_threat_right_img:.4f}"
                )

        if self.vision_mode == "ommatidia":
            avoid = (
                left_score > self.omma_obstacle_threshold
                or right_score > self.omma_obstacle_threshold
            )
        elif self.use_spike_obstacles:
            spike_avoid = (
                left_score > self.spike_obstacle_threshold
                or right_score > self.spike_obstacle_threshold
            )
            legacy_avoid = (
                self._legacy_left_score > self.obstacle_threshold
                or self._legacy_right_score > self.obstacle_threshold
            )
            avoid = spike_avoid or (
                self.spike_use_legacy_fallback and legacy_avoid
            )
        else:
            avoid = (
                left_score > self.obstacle_threshold
                or right_score > self.obstacle_threshold
            )

        if avoid:
            self.state = State.AVOID_OBSTACLE
            self._avoid_streak += 1
        else:
            self.state = State.FOLLOW_SCENT
            self._avoid_streak = 0

        if self.state == State.FOLLOW_SCENT:
            drives = self.follow_scent()
        elif self.state == State.AVOID_OBSTACLE:
            drives_avoid = self.avoid_obstacle(left_score, right_score)
            if self._avoid_streak > self.avoid_stuck_blend_after_steps:
                t = (
                    self._avoid_streak - self.avoid_stuck_blend_after_steps
                ) / float(max(1, self.avoid_stuck_blend_ramp_steps))
                t = float(np.clip(t, 0.0, 1.0))
                drives_follow = self.follow_scent()
                drives = (1.0 - t) * drives_avoid + t * drives_follow
            else:
                drives = drives_avoid
        else:
            drives = np.array([0.0, 0.0])

        drives = np.asarray(drives, dtype=np.float64) * drive_scale
        joint_angles, adhesion = self.turning_controller.step(drives)

        if self.force_full_adhesion_when_avoiding:
            if self.state == State.AVOID_OBSTACLE:
                self._full_adhesion_timer = self.full_adhesion_cooldown_steps
            elif self._full_adhesion_timer > 0:
                self._full_adhesion_timer -= 1

            if self._full_adhesion_timer > 0:
                adhesion = np.ones_like(adhesion)

        self._prev_state = self.state

        if step > 0 and self.show_prints:
            self._plot_obstacle_debug(sim, step)
        #    plt.figure(figsize=(10, 4))
        #    plt.imshow(fly_vision)
        #   plt.title(f"Vision step {step} — {self.state.name}")
        #    plt.axis("off")
        #    plt.show()


        return joint_angles, adhesion

    def _plot_obstacle_debug(self, sim: MiniprojectSimulation, step: int) -> None:
        """Plots debug overlays for obstacle detection when show_prints is True."""
        if self.vision_mode == "ommatidia":
            omma = sim.get_ommatidia_readouts(sim.fly.name)  # (2, N, 2)
            if omma is None:
                return
            g = omma[:, :, 0]
            b = omma[:, :, 1]
            eps = 1e-8
            dom = (g - b) / (g + b + eps)

            retina = sim.world.fly_lookup[sim.fly.name].retina
            left_img = retina.hex_pxls_to_human_readable(dom[0], color_8bit=True)
            right_img = retina.hex_pxls_to_human_readable(dom[1], color_8bit=True)

            plt.figure(figsize=(10, 4))
            plt.suptitle(f"Ommatidia obstacle debug — step {step}")
            plt.subplot(1, 2, 1)
            plt.title("Left eye: green dominance")
            plt.imshow(left_img, cmap="coolwarm", vmin=0, vmax=255)
            plt.axis("off")
            plt.subplot(1, 2, 2)
            plt.title("Right eye: green dominance")
            plt.imshow(right_img, cmap="coolwarm", vmin=0, vmax=255)
            plt.axis("off")
            plt.show()
            return

        # spikes_raw debug
        eye_imgs = sim.get_raw_vision(sim.fly.name)
        debugs = []
        centers_all = []
        # Ordre flygym habituel: eye_imgs[0] = gauche, [1] = droite.
        for eye_i, img in enumerate(eye_imgs):
            centers, dbg = detect_spike_debug_single_eye_rgb(
                img,
                eye_is_left=(eye_i == 0),
                forward_frac=self.spike_forward_frac,
                ignore_top_frac=self.spike_ignore_top_frac,
                ignore_bottom_band_frac=self.spike_ignore_bottom_band_frac,
                steering_bottom_ignore_frac=self.spike_steering_bottom_ignore_frac,
                spike_cc_min_area=self.spike_cc_min_area,
                spike_cc_min_height=self.spike_cc_min_height,
                spike_cc_max_width_over_height=self.spike_cc_max_width_over_height,
                spike_cc_max_instances=self.spike_cc_max_instances,
                spike_cc_landmark_y_frac=self.spike_cc_landmark_y_frac,
            )
            debugs.append(dbg)
            centers_all.append(centers)

        yh = self.spike_y_horizon_frac
        per_dom: list[
            tuple[tuple[int, int] | None, tuple[int, int] | None, float, float]
        ] = []
        global_best_left: tuple[float, int, int, int] | None = None
        global_best_right: tuple[float, int, int, int] | None = None

        for eye_i, centers in enumerate(centers_all):
            dbg = debugs[eye_i]
            h_i, w_i = dbg["rgb"].shape[:2]
            xb = _spike_forward_x_bounds(
                w_i,
                eye_is_left=(eye_i == 0),
                forward_frac=self.spike_forward_frac,
            )
            l_xy, r_xy, ml_i, mr_i = lateral_spike_dominant_peaks_xy(
                centers,
                w_i,
                h_i,
                y_horizon_frac=yh,
                x_forward_bounds=xb,
            )
            per_dom.append((l_xy, r_xy, ml_i, mr_i))
            if l_xy is not None and (
                global_best_left is None or ml_i > global_best_left[0]
            ):
                global_best_left = (ml_i, eye_i, l_xy[0], l_xy[1])
            if r_xy is not None and (
                global_best_right is None or mr_i > global_best_right[0]
            ):
                global_best_right = (mr_i, eye_i, r_xy[0], r_xy[1])

        mtl_plot = max((d[2] for d in per_dom), default=0.0)
        mtr_plot = max((d[3] for d in per_dom), default=0.0)
        peak_m = max(mtl_plot, mtr_plot)
        floor_m = float(self.spike_max_steering_floor)
        rel_m = float(self.spike_max_rel_margin)
        min_sep = rel_m * max(peak_m, 1e-9)

        priority_eye: int | None = None
        priority_xy: tuple[int, int] | None = None
        if self.spike_use_max_for_steering and peak_m >= floor_m:
            if mtl_plot > mtr_plot + min_sep and global_best_left is not None:
                priority_eye = global_best_left[1]
                priority_xy = (global_best_left[2], global_best_left[3])
            elif mtr_plot > mtl_plot + min_sep and global_best_right is not None:
                priority_eye = global_best_right[1]
                priority_xy = (global_best_right[2], global_best_right[3])

        # overlay centers on the eye images
        overlays = []
        for eye_i, (dbg, centers) in enumerate(zip(debugs, centers_all)):
            rgb = dbg["rgb"].copy()
            for x, y in centers:
                cv2.circle(rgb, (int(x), int(y)), 5, (255, 0, 0), -1)  # red in RGB
                cv2.circle(rgb, (int(x), int(y)), 9, (255, 255, 255), 2)

            l_xy, r_xy, _, _ = per_dom[eye_i]
            if l_xy is not None:
                cv2.circle(rgb, (l_xy[0], l_xy[1]), 12, (0, 220, 255), 3)
            if r_xy is not None:
                cv2.circle(rgb, (r_xy[0], r_xy[1]), 12, (255, 140, 0), 3)

            if priority_eye == eye_i and priority_xy is not None:
                px, py = priority_xy
                cv2.circle(rgb, (px, py), 22, (255, 0, 200), 4)

            sb = float(self.spike_steering_bottom_ignore_frac)
            if sb > 0:
                hh, ww = rgb.shape[:2]
                y_lim = int(dbg.get("peak_use_y0", hh - int(round(sb * hh))))
                y_lim = max(0, min(hh - 1, y_lim))
                cv2.line(rgb, (0, y_lim), (ww - 1, y_lim), (80, 230, 90), 2)

            roi = dbg["roi_rect_xyxy"]
            rgb = _draw_spike_roi_rectangle(rgb, roi)
            overlays.append(rgb)

        plt.figure(figsize=(12, 8))
        roi_note = (
            "ROI red | green line = steering-only bottom crop | red dots = one apex per white blob (tallest blobs first) "
            "| cyan/orange = max-weight lateral half | magenta = max-based steer pick"
        )
        plt.suptitle(f"Spike obstacle debug — step {step}\n{roi_note}", fontsize=9)

        # Top row: overlay per eye
        plt.subplot(2, 3, 1)
        plt.title(f"Left overlay (n={len(centers_all[0])})")
        plt.imshow(overlays[0])
        plt.axis("off")
        plt.subplot(2, 3, 2)
        plt.title("Left spikes_bin (raw mask)")
        plt.imshow(_draw_spike_roi_rectangle(debugs[0]["spikes_bin"], debugs[0]["roi_rect_xyxy"]))
        plt.axis("off")
        plt.subplot(2, 3, 3)
        plt.title("Left peaks (1 apex / blob, tallest first)")
        plt.imshow(_draw_spike_roi_rectangle(debugs[0]["peaks"], debugs[0]["roi_rect_xyxy"]))
        plt.axis("off")

        plt.subplot(2, 3, 4)
        plt.title(f"Right overlay (n={len(centers_all[1])})")
        plt.imshow(overlays[1])
        plt.axis("off")
        plt.subplot(2, 3, 5)
        plt.title("Right spikes_bin (raw mask)")
        plt.imshow(_draw_spike_roi_rectangle(debugs[1]["spikes_bin"], debugs[1]["roi_rect_xyxy"]))
        plt.axis("off")
        plt.subplot(2, 3, 6)
        plt.title("Right peaks (1 apex / blob, tallest first)")
        plt.imshow(_draw_spike_roi_rectangle(debugs[1]["peaks"], debugs[1]["roi_rect_xyxy"]))
        plt.axis("off")

        plt.show()

    def detect_green_obstacle(self, sim):
        """
        Retourne (score_œil_gauche, score_œil_droit).

        Si ``use_spike_obstacles``: chaque score est la somme des menaces des spikes
        dans cet œil (pondération position / proximité apparente). Les attributs
        ``_spike_threat_left_img`` / ``_spike_threat_right_img`` agrègent la menace
        latérale sur les deux yeux (pour choisir la direction de virage).

        Sinon: ancienne heuristique vert dans la région ciel (colonnes).
        """
        if self.vision_mode == "ommatidia":
            # Fast path: no OpenCV, no raw image processing.
            left_score, right_score = self._detect_obstacle_from_ommatidia(sim)
            # Reuse these for steering in avoid_obstacle()
            self._spike_threat_left_img = self._omma_threat_left_img
            self._spike_threat_right_img = self._omma_threat_right_img
            # Legacy scores not used in this mode.
            self._legacy_left_score = 0.0
            self._legacy_right_score = 0.0
            return left_score, right_score

        eye_imgs = sim.get_raw_vision(sim.fly.name)

        if self.use_spike_obstacles:
            leg_l, leg_r = self._legacy_sky_green_scores(eye_imgs)
            self._legacy_left_score = leg_l
            self._legacy_right_score = leg_r

            per_eye_totals = []
            agg_tl = 0.0
            agg_tr = 0.0
            agg_max_tl = 0.0
            agg_max_tr = 0.0
            for eye_i, img in enumerate(eye_imgs):
                img = np.asarray(img)
                h, w = img.shape[:2]
                centers = detect_spike_centers_single_eye_rgb(
                    img,
                    eye_is_left=(eye_i == 0),
                    forward_frac=self.spike_forward_frac,
                    ignore_top_frac=self.spike_ignore_top_frac,
                    ignore_bottom_band_frac=self.spike_ignore_bottom_band_frac,
                    steering_bottom_ignore_frac=self.spike_steering_bottom_ignore_frac,
                    spike_cc_min_area=self.spike_cc_min_area,
                    spike_cc_min_height=self.spike_cc_min_height,
                    spike_cc_max_width_over_height=self.spike_cc_max_width_over_height,
                    spike_cc_max_instances=self.spike_cc_max_instances,
                    spike_cc_landmark_y_frac=self.spike_cc_landmark_y_frac,
                )
                x_bounds = _spike_forward_x_bounds(
                    w,
                    eye_is_left=(eye_i == 0),
                    forward_frac=self.spike_forward_frac,
                )
                tl, tr = lateral_spike_threat(
                    centers,
                    w,
                    h,
                    y_horizon_frac=self.spike_y_horizon_frac,
                    x_forward_bounds=x_bounds,
                )
                mtl, mtr = lateral_spike_threat_max(
                    centers,
                    w,
                    h,
                    y_horizon_frac=self.spike_y_horizon_frac,
                    x_forward_bounds=x_bounds,
                )
                agg_tl += tl
                agg_tr += tr
                agg_max_tl = max(agg_max_tl, mtl)
                agg_max_tr = max(agg_max_tr, mtr)
                per_eye_totals.append(tl + tr)
            self._spike_threat_left_img = agg_tl
            self._spike_threat_right_img = agg_tr
            self._spike_max_threat_left_img = agg_max_tl
            self._spike_max_threat_right_img = agg_max_tr
            return per_eye_totals[0], per_eye_totals[1]

        scores = []
        for img in eye_imgs:
            img = np.asarray(img)

            if img.max() <= 1.0:
                img = img * 255.0

            h, w, c = img.shape

            sky_region = img[: int(h * self.sky_region_ratio), :, :]

            r = sky_region[:, :, 0].astype(float)
            g = sky_region[:, :, 1].astype(float)
            b = sky_region[:, :, 2].astype(float)

            green_mask = (
                (g > 90)
                & (g > r * 1.25)
                & (g > b * 1.05)
            )

            column_green_ratio = green_mask.mean(axis=0)
            tall_green_columns = column_green_ratio > self.min_green_height_ratio

            score = tall_green_columns.mean()
            scores.append(score)

        return scores[0], scores[1]

    def _legacy_sky_green_scores(self, eye_imgs) -> tuple[float, float]:
        """Fraction de colonnes avec vert « haut » dans la bande ciel — même logique que l’ancien détecteur."""
        scores = []
        for img in eye_imgs:
            img = np.asarray(img)
            if img.max() <= 1.0:
                img = img * 255.0
            h, w, _ = img.shape
            sky_region = img[: int(h * self.sky_region_ratio), :, :]
            r = sky_region[:, :, 0].astype(float)
            g = sky_region[:, :, 1].astype(float)
            b = sky_region[:, :, 2].astype(float)
            green_mask = (g > 90) & (g > r * 1.25) & (g > b * 1.05)
            column_green_ratio = green_mask.mean(axis=0)
            tall_green_columns = column_green_ratio > self.min_green_height_ratio
            scores.append(float(tall_green_columns.mean()))
        return scores[0], scores[1]

    def avoid_obstacle(self, left_score, right_score):
        if self.show_prints:
            print(f"AVOID_OBSTACLE: L={left_score:.4f}, R={right_score:.4f}")

        ll = self._legacy_left_score
        rr = self._legacy_right_score

        if self.use_spike_obstacles:
            tl = self._spike_threat_left_img
            tr = self._spike_threat_right_img
            mtl = self._spike_max_threat_left_img
            mtr = self._spike_max_threat_right_img
            spike_strong = max(left_score, right_score) > self.spike_obstacle_threshold
            spike_skewed = abs(tl - tr) > 0.08

            if spike_strong or spike_skewed:
                peak_m = max(mtl, mtr)
                floor_m = float(self.spike_max_steering_floor)
                rel_m = float(self.spike_max_rel_margin)
                min_sep = rel_m * max(peak_m, 1e-9)
                use_max = self.spike_use_max_for_steering and peak_m >= floor_m
                if use_max:
                    if mtl > mtr + min_sep:
                        if self.show_prints:
                            print(
                                "Pic spike dominant à gauche (max menace) → tourner à droite"
                            )
                        return np.array([1.2, 0.2])
                    if mtr > mtl + min_sep:
                        if self.show_prints:
                            print(
                                "Pic spike dominant à droite (max menace) → tourner à gauche"
                            )
                        return np.array([0.2, 1.2])
                if tl > tr + 1e-6:
                    if self.show_prints:
                        print("Spikes plutôt à gauche du champ (somme) → tourner à droite")
                    return np.array([1.2, 0.2])
                if tr > tl + 1e-6:
                    if self.show_prints:
                        print("Spikes plutôt à droite du champ (somme) → tourner à gauche")
                    return np.array([0.2, 1.2])

            if ll > rr + 1e-7:
                if self.show_prints:
                    print("Fallback ciel: vert à gauche → tourner à droite")
                return np.array([1.2, 0.2])
            if rr > ll + 1e-7:
                if self.show_prints:
                    print("Fallback ciel: vert à droite → tourner à gauche")
                return np.array([0.2, 1.2])

        if left_score > right_score:
            if self.show_prints:
                print("Obstacle à gauche → tourner à droite")
            return np.array([1.2, 0.2])
        else:
            if self.show_prints:
                print("Obstacle à droite → tourner à gauche")
            return np.array([0.2, 1.2])

    def follow_scent(self):
        left_odor_a = self.odor_smooth[0, 0]
        left_odor_b = self.odor_smooth[2, 0]
        right_odor_a = self.odor_smooth[1, 0]
        right_odor_b = self.odor_smooth[3, 0]

        left_signal = left_odor_a + left_odor_b
        right_signal = right_odor_a + right_odor_b

        eps = 1e-8
        ratio = abs(left_signal) / (abs(right_signal) + eps)

        if self.show_prints:
            print(f"Olfaction: {self.odor_smooth}")
            print(f"Left signal: {left_signal}, Right signal: {right_signal}")
            print(f"ratio: {ratio}")

        if abs(ratio - 1) < GO_STRAIGHT_THRESHOLD:
            if self.show_prints:
                print("FOLLOW_SCENT: going straight")
            return np.array([2.0, 2.0])

        elif left_signal > right_signal:
            if self.show_prints:
                print("FOLLOW_SCENT: turning left")
            return np.array([0.2, 1.0])

        else:
            if self.show_prints:
                print("FOLLOW_SCENT: turning right")
            return np.array([1.0, 0.2])