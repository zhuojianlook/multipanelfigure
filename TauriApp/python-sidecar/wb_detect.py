"""Vendored western-blot lane/band detection — VERBATIM copy of the user's
detect_bands_equal_boxes.py (equal-size sample band boxes). Keep in sync with
/Users/zhuojian/Documents/New project/detect_bands_equal_boxes.py.

The geometric defaults (roi_width=74, roi_height=33, expected_lanes=8,
threshold_percentile=98.0, min_gap=45, group min_lanes=3, group tolerance=35,
channel="max") match the CLI exactly so the in-app result reproduces the user's
annotated reference. Thin wrappers expose two entry points the endpoint uses:

    detect_wb_bands(rgb, ...)           # auto-detect + refine + uniform boxes
    detect_wb_bands_in_lanes(rgb, ...)  # --lanes mode (skip auto-detect)

Both return (lanes, bands, (H, W)) where each band carries `roi_x1`,
`roi_y1`, `roi_x2_exclusive`, `roi_y2_exclusive` for the endpoint to project
into normalised picker coordinates.
"""
from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


# (lane_name, x1, x2, y1, y2)
Lane = tuple

# Defaults mirrored from detect_bands_equal_boxes.py argparse + main.
DEFAULT_SIGNAL_POLARITY = "bright"        # accepted for API back-compat; script is bright-only
DEFAULT_MARKER_LANES = {"L", "Ladder", "Marker"}
DEFAULT_EXPECTED_LANES = 8                # --expected-lanes default
DEFAULT_AUTO_LANE_THRESHOLD_PERCENTILE = 98.0
DEFAULT_AUTO_LANE_MIN_GAP = 45             # main() hard-codes this
DEFAULT_AUTO_LANE_MAX_BAND_WIDTH = 0       # kept for compatibility; unused
DEFAULT_AUTO_MIN_GROUP_LANES = 3
DEFAULT_GROUP_TOLERANCE = 35
DEFAULT_AUTO_CONSENSUS_TOLERANCE = 24      # kept for compatibility; not used by new script
DEFAULT_ROI_WIDTH = 74
DEFAULT_ROI_HEIGHT = 33
DEFAULT_MEASUREMENT_CHANNEL = "max"


def contrast_scale(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = np.percentile(rgb, [0.5, 99.95])
    scaled = np.clip((rgb - lo) / max(hi - lo, 1), 0, 1)
    display = np.clip(scaled ** 0.5 * 255, 0, 255).astype(np.uint8)
    analysis = np.clip(scaled * 255, 0, 255).astype(np.uint8)
    return display, analysis


def horizontal_signal(analysis_rgb: np.ndarray, signal_polarity: str = "bright") -> np.ndarray:
    """The script is bright-only (max-channel + bg-subtract). The polarity
    parameter is accepted for API back-compat and currently ignored."""
    _ = signal_polarity
    gray = np.max(analysis_rgb, axis=2)
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    background = cv2.GaussianBlur(blur, (0, 0), 28)
    signal = blur.astype(np.float32) - background.astype(np.float32)
    signal[signal < 0] = 0
    return cv2.GaussianBlur(signal, (0, 0), sigmaX=3, sigmaY=1)


def measurement_plane(rgb: np.ndarray, channel: str) -> np.ndarray:
    if channel == "max":
        return np.max(rgb, axis=2)
    if channel == "mean":
        return np.mean(rgb, axis=2)
    if channel == "red":
        return rgb[..., 0]
    if channel == "green":
        return rgb[..., 1]
    if channel == "blue":
        return rgb[..., 2]
    raise ValueError(channel)


def robust_background(values: np.ndarray) -> tuple[float, float, int]:
    flat = values.reshape(-1).astype(np.float64)
    if flat.size == 0:
        return 0.0, 0.0, 0
    median = float(np.median(flat))
    mad = float(np.median(np.abs(flat - median)))
    if mad > 0:
        sigma = 1.4826 * mad
        keep = np.abs(flat - median) <= 3.0 * sigma
        if int(np.sum(keep)) >= max(20, flat.size // 5):
            flat = flat[keep]
    sd = float(np.std(flat, ddof=1)) if flat.size > 1 else 0.0
    return float(np.mean(flat)), sd, int(flat.size)


def complete_lane_grid(
    candidates: list,
    expected_lanes,
    image_width: int,
    lane_width,
) -> list:
    candidates = sorted(candidates, key=lambda item: item[1])
    if expected_lanes is None or len(candidates) < 2:
        return candidates
    while len(candidates) < expected_lanes:
        centers = np.array([item[1] for item in candidates], dtype=np.float64)
        gaps = np.diff(centers)
        if gaps.size == 0:
            break
        gap_index = int(np.argmax(gaps))
        left = candidates[gap_index]
        right = candidates[gap_index + 1]
        center = int(round((left[1] + right[1]) / 2))
        width = int(round(np.median([item[3] - item[2] for item in candidates])))
        x1 = max(0, center - width // 2)
        x2 = min(image_width, x1 + width)
        candidates.append((0.0, center, x1, x2, left[4], left[5]))
        candidates.sort(key=lambda item: item[1])

    if len(candidates) == expected_lanes:
        centers = np.array([item[1] for item in candidates], dtype=np.float64)
        gaps = np.diff(centers)
        pitch = float(np.median(gaps[gaps <= np.percentile(gaps, 75)])) if gaps.size else 80
        observed = float(np.median([item[3] - item[2] for item in candidates]))
        width = int(lane_width) if lane_width else int(round(max(observed, pitch * 0.82)))
        width = max(40, min(image_width, width))
        normalized = []
        for score, center, _x1, _x2, y1, y2 in candidates:
            x1 = max(0, int(center) - width // 2)
            x2 = min(image_width, x1 + width)
            if x2 - x1 < width and x2 == image_width:
                x1 = max(0, x2 - width)
            normalized.append((score, int(center), x1, x2, y1, y2))
        candidates = normalized
    return candidates[:expected_lanes]


def auto_detect_lanes(
    signal: np.ndarray,
    expected_lanes,
    lane_width,
    threshold_percentile: float,
    min_gap: int,
) -> list:
    height, width = signal.shape
    threshold = max(float(np.percentile(signal, threshold_percentile)), 1.0)
    feature = cv2.morphologyEx(
        signal.astype(np.uint8),
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (11, 3)),
    )
    mask = (feature > threshold).astype(np.uint8)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3)),
    )
    num, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    components = []
    for label in range(1, num):
        x, y, w, h, area = stats[label]
        if w < 12 or w > max(50, int(width * 0.16)):
            continue
        if h < 3 or h > max(60, int(height * 0.08)):
            continue
        if area < max(30, w):
            continue
        if w / max(h, 1) < 2.0:
            continue
        cx, cy = cents[label]
        components.append(
            {
                "cx": float(cx),
                "x1": int(x),
                "x2": int(x + w),
                "y1": int(y),
                "y2": int(y + h),
                "score": float(area * signal[labels == label].mean()),
            }
        )
    if not components:
        raise ValueError("No lane seed components found")

    components.sort(key=lambda item: item["cx"])
    clusters = []
    for component in components:
        if not clusters or component["cx"] - clusters[-1][-1]["cx"] > min_gap:
            clusters.append([component])
        else:
            clusters[-1].append(component)

    global_y1 = max(0, min(item["y1"] for item in components) - 35)
    global_y2 = min(height, max(item["y2"] for item in components) + 35)
    candidates = []
    for cluster in clusters:
        centers = np.array([item["cx"] for item in cluster])
        scores = np.array([max(item["score"], 1.0) for item in cluster])
        center = int(round(float(np.average(centers, weights=scores))))
        score = float(np.sum(scores))
        width_guess = int(lane_width) if lane_width else max(
            40, int(np.percentile([item["x2"] - item["x1"] for item in cluster], 75)) + 24
        )
        x1 = max(0, center - width_guess // 2)
        x2 = min(width, x1 + width_guess)
        candidates.append((score, center, x1, x2, global_y1, global_y2))

    candidates.sort(key=lambda item: item[0], reverse=True)
    if expected_lanes is not None:
        candidates = candidates[:expected_lanes]
    else:
        strongest = candidates[0][0]
        candidates = [item for item in candidates if item[0] >= strongest * 0.015]
    candidates = complete_lane_grid(candidates, expected_lanes, width, lane_width)
    candidates.sort(key=lambda item: item[1])

    lanes = []
    for index, (_score, _center, x1, x2, y1, y2) in enumerate(candidates):
        name = "L" if index == 0 else f"S{index}"
        lanes.append((name, int(x1), int(x2), int(y1), int(y2)))
    return lanes


def refine_lane_edges(signal: np.ndarray, lanes: list) -> list:
    image_height, image_width = signal.shape
    refined = []
    for lane_name, x1, x2, y1, y2 in lanes:
        lane_width = x2 - x1
        sx1 = max(0, x1 - 24)
        sx2 = min(image_width, x2 + 24)
        roi = signal[max(0, y1):min(image_height, y2), sx1:sx2]
        if roi.size == 0:
            refined.append((lane_name, x1, x2, y1, y2))
            continue
        high = np.percentile(roi, 85, axis=0)
        low = np.percentile(roi, 40, axis=0)
        profile = gaussian_filter1d(high - low, 3)
        center = (x1 + x2) / 2 - sx1
        local_left = max(0, int(center - lane_width / 2))
        local_right = min(len(profile), int(center + lane_width / 2) + 1)
        peak_idx = local_left + int(np.argmax(profile[local_left:local_right]))
        baseline = float(np.percentile(profile, 20))
        peak = float(profile[peak_idx])
        if peak <= baseline:
            refined.append((lane_name, x1, x2, y1, y2))
            continue
        threshold = baseline + 0.2 * (peak - baseline)
        left = peak_idx
        right = peak_idx
        while left > 0 and profile[left] > threshold:
            left -= 1
        while right < len(profile) - 1 and profile[right] > threshold:
            right += 1
        rx1 = max(0, sx1 + left - 4)
        rx2 = min(image_width, sx1 + right + 4)
        if rx2 - rx1 < lane_width * 0.75 or rx2 - rx1 > lane_width * 1.35:
            refined.append((lane_name, x1, x2, y1, y2))
        else:
            refined.append((lane_name, int(rx1), int(rx2), y1, y2))
    return refined


def is_marker(lane_name: str) -> bool:
    return lane_name in {"L", "Ladder", "Marker"}


def is_marker_lane(lane_name: str, marker_lanes=None) -> bool:
    """API back-compat — accepts a marker set or uses the default."""
    if marker_lanes is None:
        return is_marker(lane_name)
    return lane_name in marker_lanes


def detect_lane_bands(signal: np.ndarray, lanes: list, marker_lanes=None) -> list:
    """Detect band peaks in each lane. `marker_lanes` is accepted for
    API back-compat; the script uses the hardcoded {L,Ladder,Marker} set."""
    _ = marker_lanes
    bands = []
    for lane_name, x1, x2, y1, y2 in lanes:
        lane = signal[y1:y2, x1:x2]
        if lane.size == 0:
            continue
        profile = np.percentile(lane, 65, axis=1)
        profile = gaussian_filter1d(profile, 2.2)
        baseline = gaussian_filter1d(profile, 18)
        peak_profile = profile - baseline
        peak_profile[peak_profile < 0] = 0
        min_prominence = max(np.percentile(peak_profile, 90) * 0.45, 1.5)
        min_distance = 18
        if is_marker(lane_name):
            min_prominence = max(np.percentile(peak_profile, 80) * 0.30, 2.0)
            min_distance = 20
        peaks, props = find_peaks(
            peak_profile,
            distance=min_distance,
            prominence=min_prominence,
            width=(3, 35),
        )
        for i, peak in enumerate(peaks):
            prominence = float(props["prominences"][i])
            width = float(props["widths"][i])
            if is_marker(lane_name) or prominence >= 10:
                confidence = "clear"
            elif prominence >= 4:
                confidence = "faint"
            else:
                continue
            bands.append(
                {
                    "lane": lane_name,
                    "x1": x1,
                    "x2": x2,
                    "y": int(peak + y1),
                    "prominence": round(prominence, 2),
                    "width": round(width, 1),
                    "confidence": confidence,
                }
            )
    return bands


def assign_groups(bands: list, tolerance: int) -> None:
    sample = [band for band in bands if not is_marker(band["lane"])]
    sample.sort(key=lambda band: band["y"])
    groups = []
    for band in sample:
        if not groups:
            groups.append([band])
            continue
        center = float(np.median([item["y"] for item in groups[-1]]))
        if abs(band["y"] - center) <= tolerance:
            groups[-1].append(band)
        else:
            groups.append([band])
    for group_index, group in enumerate(groups, start=1):
        for band in group:
            band["band_group"] = f"G{group_index}"
    for band in bands:
        band.setdefault("band_group", "")


def assign_band_groups(bands: list, tolerance: int, marker_lanes=None) -> None:
    """API back-compat — wraps assign_groups (ignores marker_lanes; the script
    uses the hardcoded {L,Ladder,Marker} set via is_marker)."""
    _ = marker_lanes
    assign_groups(bands, tolerance)


def filter_sample_groups(bands: list, min_lanes: int, tolerance: int) -> list:
    assign_groups(bands, tolerance)
    grouped = defaultdict(list)
    kept = [band for band in bands if is_marker(band["lane"])]
    for band in bands:
        if band["band_group"] and not is_marker(band["lane"]):
            grouped[band["band_group"]].append(band)
    for group_bands in grouped.values():
        lanes = {band["lane"] for band in group_bands}
        if len(lanes) < min_lanes:
            continue
        median_y = float(np.median([band["y"] for band in group_bands]))
        by_lane = defaultdict(list)
        for band in group_bands:
            if abs(band["y"] - median_y) <= tolerance:
                by_lane[band["lane"]].append(band)
        for lane_bands in by_lane.values():
            kept.append(max(lane_bands, key=lambda band: band["prominence"]))
    kept.sort(key=lambda band: (band["x1"], band["y"]))
    for band in kept:
        band.pop("band_group", None)
    assign_groups(kept, tolerance)
    return kept


def filter_sample_band_groups(bands, group_tolerance, marker_lanes, min_lanes, consensus_tolerance):
    """API back-compat alias for filter_sample_groups."""
    _ = marker_lanes
    _ = consensus_tolerance
    return filter_sample_groups(bands, min_lanes=min_lanes, tolerance=group_tolerance)


def centered_bounds(center: int, size: int, lower: int, upper: int) -> tuple:
    size = max(1, min(int(size), upper - lower))
    start = int(round(center - (size - 1) / 2))
    end = start + size
    if start < lower:
        start = lower
        end = start + size
    if end > upper:
        end = upper
        start = end - size
    return int(start), int(end)


def estimate_lane_centers(plane: np.ndarray, lanes: list, bands: list) -> dict:
    centers = {}
    for lane_name, x1, x2, y1, y2 in lanes:
        centers[lane_name] = int(round((x1 + x2 - 1) / 2))
        if is_marker(lane_name):
            continue
        estimates = []
        for band in bands:
            if band["lane"] != lane_name or band["confidence"] != "clear":
                continue
            sy1 = max(y1, band["y"] - 14)
            sy2 = min(y2, band["y"] + 15)
            sx1 = max(0, x1 - 10)
            sx2 = min(plane.shape[1], x2 + 10)
            roi = plane[sy1:sy2, sx1:sx2].astype(np.float64)
            bg = robust_background(roi)[0]
            residual = roi - bg
            residual[residual < 0] = 0
            if not np.any(residual):
                continue
            profile = gaussian_filter1d(np.percentile(residual, 70, axis=0), 1.2)
            baseline = float(np.percentile(profile, 20))
            peak = float(np.max(profile))
            threshold = baseline + 0.22 * (peak - baseline)
            cols = np.where(profile > threshold)[0]
            if cols.size:
                weights = np.maximum(profile[cols], 1e-6)
                estimates.append(int(round(float(np.average(cols + sx1, weights=weights)))))
        if estimates:
            centers[lane_name] = int(round(float(np.median(estimates))))
    return centers


def quantify(
    rgb: np.ndarray,
    lanes: list,
    bands: list,
    roi_width: int,
    roi_height: int,
    channel: str,
) -> list:
    plane = measurement_plane(rgb, channel)
    lane_centers = estimate_lane_centers(plane, lanes, bands)
    lane_lookup = {lane[0]: lane for lane in lanes}
    rows = []
    for index, band in enumerate(bands, start=1):
        if band["lane"] not in lane_lookup:
            continue
        lane_name, lane_x1, lane_x2, lane_y1, lane_y2 = lane_lookup[band["lane"]]
        if is_marker(band["lane"]):
            x1, x2 = lane_x1 + 4, lane_x2 - 4
            y1, y2 = centered_bounds(band["y"], roi_height, lane_y1, lane_y2)
        else:
            x1, x2 = centered_bounds(lane_centers[band["lane"]], roi_width, 0, plane.shape[1])
            y1, y2 = centered_bounds(band["y"], roi_height, lane_y1, lane_y2)
        roi = plane[y1:y2, x1:x2].astype(np.float64)
        area = int(roi.size)
        gap = 6
        above_y2 = max(lane_y1, y1 - gap)
        above_y1 = max(lane_y1, above_y2 - (y2 - y1))
        below_y1 = min(lane_y2, y2 + gap)
        below_y2 = min(lane_y2, below_y1 + (y2 - y1))
        parts = []
        if above_y2 > above_y1:
            parts.append(plane[above_y1:above_y2, x1:x2].reshape(-1))
        if below_y2 > below_y1:
            parts.append(plane[below_y1:below_y2, x1:x2].reshape(-1))
        bg_pixels = np.concatenate(parts) if parts else np.array([])
        bg_mean, bg_sd, bg_area = robust_background(bg_pixels)
        raw_sum = float(np.sum(roi))
        corrected = raw_sum - bg_mean * area
        row = dict(band)
        row.update(
            {
                "band_index": index,
                "roi_x1": int(x1),
                "roi_x2_exclusive": int(x2),
                "roi_y1": int(y1),
                "roi_y2_exclusive": int(y2),
                "roi_area": area,
                "background_area": bg_area,
                "raw_integrated_density": round(raw_sum, 3),
                "background_mean": round(bg_mean, 3),
                "background_sd": round(bg_sd, 3),
                "background_corrected_integrated_density": round(corrected, 3),
                "background_corrected_mean": round(corrected / area, 3) if area else None,
                "qc_flags": "ladder_reference" if is_marker(band["lane"]) else "",
            }
        )
        rows.append(row)
    return rows


# ───────────────── API wrappers used by the wb-detect-bands endpoint ──────


def _coerce_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError("expected an HxWx3 RGB image")
    return rgb[:, :, :3].astype(np.float32)


def detect_wb_bands(
    rgb: np.ndarray,
    signal_polarity: str = DEFAULT_SIGNAL_POLARITY,
    marker_lanes=None,
    min_group_lanes: int = DEFAULT_AUTO_MIN_GROUP_LANES,
    group_tolerance: int = DEFAULT_GROUP_TOLERANCE,
    consensus_tolerance: int = DEFAULT_AUTO_CONSENSUS_TOLERANCE,
    min_gap: int = DEFAULT_AUTO_LANE_MIN_GAP,
    threshold_percentile: float = DEFAULT_AUTO_LANE_THRESHOLD_PERCENTILE,
    max_band_width: int = DEFAULT_AUTO_LANE_MAX_BAND_WIDTH,
    first_lane_marker: bool = True,
    detect_ladder: bool = False,
    expected_lanes=DEFAULT_EXPECTED_LANES,
    lane_width=None,
    roi_width: int = DEFAULT_ROI_WIDTH,
    roi_height: int = DEFAULT_ROI_HEIGHT,
    channel: str = DEFAULT_MEASUREMENT_CHANNEL,
):
    """Run detect_bands_equal_boxes.py's full pipeline on an HxWx3 image. The
    optional kwargs are accepted for API back-compat with the previous module
    (signal_polarity / consensus_tolerance / max_band_width / first_lane_marker
    / detect_ladder / marker_lanes are not used by the new script). Returns
    (lanes, quantified_bands, signal.shape)."""
    _ = signal_polarity, marker_lanes, consensus_tolerance, max_band_width, first_lane_marker, detect_ladder

    rgb_f = _coerce_rgb(rgb)
    _display, analysis = contrast_scale(rgb_f)
    signal = horizontal_signal(analysis)

    lanes = auto_detect_lanes(
        signal,
        expected_lanes=expected_lanes,
        lane_width=lane_width,
        threshold_percentile=threshold_percentile,
        min_gap=min_gap,
    )
    lanes = refine_lane_edges(signal, lanes)
    bands = detect_lane_bands(signal, lanes)
    if min_group_lanes > 1:
        bands = filter_sample_groups(bands, min_lanes=min_group_lanes, tolerance=group_tolerance)
    else:
        assign_groups(bands, group_tolerance)
    quantified = quantify(rgb_f, lanes, bands, roi_width, roi_height, channel)
    return lanes, quantified, signal.shape


def detect_wb_bands_in_lanes(
    rgb: np.ndarray,
    lanes,
    signal_polarity: str = DEFAULT_SIGNAL_POLARITY,
    marker_lanes=None,
    roi_width: int = DEFAULT_ROI_WIDTH,
    roi_height: int = DEFAULT_ROI_HEIGHT,
    channel: str = DEFAULT_MEASUREMENT_CHANNEL,
    group_tolerance: int = DEFAULT_GROUP_TOLERANCE,
):
    """--lanes mode: detect bands WITHIN user-defined lane boxes (skips auto-
    detect + consensus filter). `lanes` is a list of (name, x1, x2, y1, y2)
    in analysed-image pixels. Returns (clamped_lanes, quantified_bands, shape)."""
    _ = signal_polarity, marker_lanes

    rgb_f = _coerce_rgb(rgb)
    _display, analysis = contrast_scale(rgb_f)
    signal = horizontal_signal(analysis)
    height, width = signal.shape

    clamped = []
    for (name, x1, x2, y1, y2) in lanes:
        x1c = max(0, min(int(round(x1)), width - 1))
        x2c = max(x1c + 1, min(int(round(x2)), width))
        y1c = max(0, min(int(round(y1)), height - 1))
        y2c = max(y1c + 1, min(int(round(y2)), height))
        clamped.append((str(name) or "Lane", x1c, x2c, y1c, y2c))

    bands = detect_lane_bands(signal, clamped)
    assign_groups(bands, group_tolerance)
    quantified = quantify(rgb_f, clamped, bands, roi_width, roi_height, channel)
    return clamped, quantified, signal.shape


def detect_ladder_lane(*args, **kwargs):
    """Back-compat shim — the new script handles the ladder via the leftmost
    auto-detected lane being named "L" (no separate ladder heuristic)."""
    _ = args, kwargs
    return None
