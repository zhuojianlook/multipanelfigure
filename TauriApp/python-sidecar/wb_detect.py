"""Vendored western-blot lane/band detection.

These functions are a faithful, verbatim copy of the user's validated
``detect_bands.py`` (the lane-detection + band-detection half). Only the CLI,
file I/O, and densitometry/quantification have been omitted — the geometry
detection is identical so the in-app "Auto-detect bands" button reproduces the
script's results exactly.

Default pipeline (matches ``detect_bands.py main()`` with default args):

    display_rgb, analysis_rgb = contrast_scale(rgb)
    signal = horizontal_signal(analysis_rgb, signal_polarity)
    lanes  = auto_detect_lanes(signal, ...defaults...)
    lanes  = name_auto_lanes(lanes, None, first_lane_marker=True)
    bands  = detect_lane_bands(signal, lanes, marker_lanes)
    bands  = filter_sample_band_groups(bands, 35, marker_lanes, 3, 24)

``detect_wb_bands(rgb)`` runs that whole default path and returns
``(lanes, bands, signal_shape)``.

Keep this file in sync with /Users/zhuojian/Documents/New project/detect_bands.py.
"""
from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# (lane_name, x1, x2, y1, y2)
Lane = tuple

# Defaults mirrored from detect_bands.py argparse.
DEFAULT_SIGNAL_POLARITY = "bright"
DEFAULT_MARKER_LANES = {"L", "Ladder", "Marker"}
DEFAULT_AUTO_LANE_MIN_GAP = 45
# detect_bands.py --auto-lane-threshold-percentile default is 98.0 (NOT 99.0).
# 99.0 rejects too many band components and collapses adjacent lanes (e.g.
# 6 lanes → 4 on f35t_nrf2-5000ms-m2.tiff), so this must stay 98.0 for parity.
DEFAULT_AUTO_LANE_THRESHOLD_PERCENTILE = 98.0
DEFAULT_AUTO_LANE_MAX_BAND_WIDTH = 0
DEFAULT_AUTO_MIN_GROUP_LANES = 3
DEFAULT_AUTO_CONSENSUS_TOLERANCE = 24
DEFAULT_GROUP_TOLERANCE = 35


def contrast_scale(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert high-bit-depth data to display/analysis images.

    The TIFF is 16-bit and very dark. Percentile scaling avoids letting
    a few saturated pixels dominate the contrast.
    """
    lo, hi = np.percentile(rgb, [0.5, 99.95])
    denom = float(hi - lo)
    if denom <= 1e-6:
        denom = 1e-6
    scaled = np.clip((rgb - lo) / denom, 0, 1)
    display_rgb = np.clip(scaled ** 0.5 * 255, 0, 255).astype(np.uint8)
    analysis_rgb = np.clip(scaled * 255, 0, 255).astype(np.uint8)
    return display_rgb, analysis_rgb


def horizontal_signal(
    analysis_rgb: np.ndarray,
    signal_polarity: str,
) -> np.ndarray:
    """
    Build a band-enhanced image.

    1. Use max RGB channel so colored ladder bands and white bands both count.
    2. Blur slightly to reduce pixel noise.
    3. Estimate broad background with a large Gaussian blur.
    4. Subtract background and clamp negatives to zero.
    5. Smooth more along x than y, favoring horizontal band-like structures.
    """
    gray = np.max(analysis_rgb, axis=2)
    if signal_polarity == "dark":
        gray = 255 - gray
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    background = cv2.GaussianBlur(blur, (0, 0), 28)
    signal = blur.astype(np.float32) - background.astype(np.float32)
    signal[signal < 0] = 0
    return cv2.GaussianBlur(signal, (0, 0), sigmaX=3, sigmaY=1)


def parse_name_set(text: str) -> set:
    return {item.strip() for item in text.split(",") if item.strip()}


def is_marker_lane(lane_name: str, marker_lanes: set) -> bool:
    return lane_name in marker_lanes


def complete_lane_grid(
    lane_candidates: list,
    expected_lanes,
    image_width: int,
    lane_width=None,
) -> list:
    """
    Fill missing lane boxes when strong band anchors skip a faint lane.

    Automatic lane detection starts from band components. That is reliable for
    visible lanes, but faint lanes can be absent from the component set. When
    the expected lane count is known, large center-to-center gaps are split to
    preserve the lane grid before edge refinement tightens each lane.
    """
    completed = sorted(lane_candidates, key=lambda item: item[1])
    if expected_lanes is None:
        return completed
    if len(completed) < 2:
        return completed

    while len(completed) < expected_lanes and len(completed) >= 2:
        centers = np.array([item[1] for item in completed], dtype=np.float64)
        gaps = np.diff(centers)
        if gaps.size == 0:
            break
        gap_index = int(np.argmax(gaps))
        left = completed[gap_index]
        right = completed[gap_index + 1]
        center = int(round((left[1] + right[1]) / 2))
        width = int(round(np.median([max(1, item[3] - item[2]) for item in completed])))
        x1 = max(0, center - width // 2)
        x2 = min(image_width, x1 + width)
        if x2 - x1 < width and x2 == image_width:
            x1 = max(0, x2 - width)
        y1 = int(round((left[4] + right[4]) / 2))
        y2 = int(round((left[5] + right[5]) / 2))
        completed.append((0.0, center, int(x1), int(x2), y1, y2))
        completed.sort(key=lambda item: item[1])

    if len(completed) == expected_lanes:
        centers = np.array([item[1] for item in completed], dtype=np.float64)
        gaps = np.diff(centers)
        small_gaps = gaps[gaps <= np.percentile(gaps, 75)] if gaps.size else gaps
        pitch = float(np.median(small_gaps if small_gaps.size else gaps))
        observed_width = float(np.median([max(1, item[3] - item[2]) for item in completed]))
        target_width = int(lane_width) if lane_width is not None else int(
            round(max(observed_width, pitch * 0.82))
        )
        target_width = max(40, min(image_width, target_width))
        normalized = []
        for score, center, _x1, _x2, y1, y2 in completed:
            x1 = max(0, int(center) - target_width // 2)
            x2 = min(image_width, x1 + target_width)
            if x2 - x1 < target_width and x2 == image_width:
                x1 = max(0, x2 - target_width)
            normalized.append((score, int(center), int(x1), int(x2), int(y1), int(y2)))
        completed = normalized

    return completed


def name_auto_lanes(
    lanes: list,
    lane_names,
    first_lane_marker: bool,
) -> list:
    """Apply stable names to auto-detected lanes."""
    parsed_names = [item.strip() for item in (lane_names or "").split(",") if item.strip()]
    if parsed_names:
        if len(parsed_names) != len(lanes):
            raise ValueError(
                f"--auto-lane-names provided {len(parsed_names)} names for "
                f"{len(lanes)} detected lanes"
            )
        return [
            (parsed_names[index], x1, x2, y1, y2)
            for index, (_name, x1, x2, y1, y2) in enumerate(lanes)
        ]

    named = []
    for index, (_name, x1, x2, y1, y2) in enumerate(lanes):
        if index == 0 and first_lane_marker:
            lane_name = "L"
        elif first_lane_marker:
            lane_name = f"S{index}"
        else:
            lane_name = f"Lane{index + 1}"
        named.append((lane_name, x1, x2, y1, y2))
    return named


def auto_detect_lanes(
    signal: np.ndarray,
    expected_lanes,
    lane_width,
    lane_y1,
    lane_y2,
    min_gap: int,
    threshold_percentile: float,
    max_band_width: int,
) -> list:
    """
    Infer lane boxes from horizontally extended band components.

    This is intentionally conservative and parameterized. It is not a
    substitute for reviewing lane geometry on publication-quality analyses.
    """
    height, width = signal.shape
    threshold = float(np.percentile(signal, threshold_percentile))
    threshold = max(threshold, 1.0)
    feature = cv2.morphologyEx(
        signal.astype(np.uint8),
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (11, 3)),
    )
    mask = (feature > threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3)),
    )

    num, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    components = []
    min_band_width = max(12, int(width * 0.01))
    if max_band_width <= 0:
        max_band_width = max(50, int(width * 0.16))

    for label in range(1, num):
        x, y, comp_width, comp_height, area = stats[label]
        cx, cy = cents[label]
        if comp_width < min_band_width or comp_width > max_band_width:
            continue
        if comp_height < 3 or comp_height > max(60, int(height * 0.08)):
            continue
        if area < max(30, min_band_width * 2):
            continue
        if comp_width / max(comp_height, 1) < 2.0:
            continue
        mean_signal = float(signal[labels == label].mean())
        components.append(
            {
                "cx": float(cx),
                "cy": float(cy),
                "x1": int(x),
                "x2": int(x + comp_width),
                "y1": int(y),
                "y2": int(y + comp_height),
                "score": float(area * mean_signal),
            }
        )

    if not components:
        raise ValueError(
            "Auto-lane detection found no band-like components. Provide --lanes."
        )

    components.sort(key=lambda item: item["cx"])
    clusters = []
    for component in components:
        if not clusters or component["cx"] - clusters[-1][-1]["cx"] > min_gap:
            clusters.append([component])
        else:
            clusters[-1].append(component)

    lane_candidates = []
    global_y1 = (
        int(lane_y1)
        if lane_y1 is not None
        else max(0, min(component["y1"] for component in components) - 35)
    )
    global_y2 = (
        int(lane_y2)
        if lane_y2 is not None
        else min(height, max(component["y2"] for component in components) + 35)
    )

    for cluster in clusters:
        centers = np.array([item["cx"] for item in cluster])
        scores = np.array([max(item["score"], 1.0) for item in cluster])
        center = int(round(float(np.average(centers, weights=scores))))
        score = float(np.sum(scores))
        candidate_width = (
            int(lane_width)
            if lane_width is not None
            else max(
                40,
                int(np.percentile([item["x2"] - item["x1"] for item in cluster], 75))
                + 24,
            )
        )
        x1 = max(0, center - candidate_width // 2)
        x2 = min(width, x1 + candidate_width)
        if x2 - x1 < candidate_width and x2 == width:
            x1 = max(0, x2 - candidate_width)
        lane_candidates.append((score, center, x1, x2, global_y1, global_y2))

    lane_candidates.sort(key=lambda item: item[0], reverse=True)
    if expected_lanes is not None:
        lane_candidates = lane_candidates[:expected_lanes]
    else:
        strongest = lane_candidates[0][0]
        lane_candidates = [
            item for item in lane_candidates if item[0] >= strongest * 0.015
        ]

    lane_candidates = complete_lane_grid(
        lane_candidates,
        expected_lanes,
        width,
        lane_width=lane_width,
    )
    lane_candidates.sort(key=lambda item: item[1])
    return [
        (f"Lane{index}", int(x1), int(x2), int(y1), int(y2))
        for index, (_, _center, x1, x2, y1, y2) in enumerate(lane_candidates, start=1)
    ]


def detect_lane_bands(
    signal: np.ndarray,
    lanes: list,
    marker_lanes: set,
) -> list:
    """
    Detect y-position peaks inside each lane.

    For each lane, the horizontal signal is collapsed into a vertical
    profile using the 65th percentile across lane width. This keeps bands
    that span much of the lane while reducing single-pixel speckles.
    """
    bands = []

    for lane_name, x1, x2, y1, y2 in lanes:
        lane = signal[y1:y2, x1:x2]
        if lane.size == 0:
            continue

        vertical_profile = np.percentile(lane, 65, axis=1)
        vertical_profile = gaussian_filter1d(vertical_profile, 2.2)

        baseline = gaussian_filter1d(vertical_profile, 18)
        peak_profile = vertical_profile - baseline
        peak_profile[peak_profile < 0] = 0

        min_prominence = max(np.percentile(peak_profile, 90) * 0.45, 1.5)
        min_distance = 18

        # The colored ladder has many strong bands packed more tightly and
        # should use a slightly more permissive lane-specific threshold.
        if is_marker_lane(lane_name, marker_lanes):
            min_prominence = max(np.percentile(peak_profile, 80) * 0.30, 2.0)
            min_distance = 20

        peaks, properties = find_peaks(
            peak_profile,
            distance=min_distance,
            prominence=min_prominence,
            width=(3, 35),
        )

        for index, peak in enumerate(peaks):
            prominence = float(properties["prominences"][index])
            width = float(properties["widths"][index])

            if is_marker_lane(lane_name, marker_lanes) or prominence >= 10:
                confidence = "clear"
            elif prominence >= 4:
                confidence = "faint"
            else:
                # These were shown as tentative during tuning, but excluded
                # from the final probable-band count.
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


def assign_band_groups(
    bands: list,
    tolerance: int,
    marker_lanes: set,
) -> None:
    """
    Group sample bands by vertical position.

    The group labels are useful for comparing the same apparent molecular
    weight across lanes. The ladder is left ungrouped because it is a marker.
    """
    sample_bands = [
        band for band in bands if not is_marker_lane(band["lane"], marker_lanes)
    ]
    sample_bands.sort(key=lambda band: band["y"])
    groups = []

    for band in sample_bands:
        if not groups:
            groups.append([band])
            continue

        current_center = float(np.median([item["y"] for item in groups[-1]]))
        if abs(band["y"] - current_center) <= tolerance:
            groups[-1].append(band)
        else:
            groups.append([band])

    for group_index, group in enumerate(groups, start=1):
        group_name = f"G{group_index}"
        for band in group:
            band["band_group"] = group_name

    for band in bands:
        band.setdefault("band_group", "")


def filter_sample_band_groups(
    bands: list,
    group_tolerance: int,
    marker_lanes: set,
    min_lanes: int,
    consensus_tolerance: int,
) -> list:
    """
    Remove weak automatic detections that do not form a cross-lane band row.

    A quantitative blot normally compares the same molecular-weight region
    across lanes. In fully automatic mode this consensus filter suppresses
    isolated dust/hot spots while keeping marker bands untouched.
    """
    if min_lanes <= 1:
        return bands

    assign_band_groups(bands, group_tolerance, marker_lanes)
    grouped = defaultdict(list)
    kept_marker = [band for band in bands if is_marker_lane(band["lane"], marker_lanes)]
    for band in bands:
        if band.get("band_group") and not is_marker_lane(band["lane"], marker_lanes):
            grouped[band["band_group"]].append(band)

    kept_sample = []
    for group_bands in grouped.values():
        median_y = float(np.median([band["y"] for band in group_bands]))
        close_bands = [
            band for band in group_bands if abs(band["y"] - median_y) <= consensus_tolerance
        ]
        close_lanes = {band["lane"] for band in close_bands}
        if len(close_lanes) < min_lanes:
            continue

        by_lane = defaultdict(list)
        for band in group_bands:
            if abs(band["y"] - median_y) <= group_tolerance:
                by_lane[band["lane"]].append(band)

        for lane_bands in by_lane.values():
            kept_sample.append(
                max(
                    lane_bands,
                    key=lambda band: (
                        -abs(band["y"] - median_y),
                        float(band.get("prominence", 0.0)),
                    ),
                )
            )

    kept = kept_marker + kept_sample
    kept.sort(key=lambda band: (band["x1"], band["y"]))
    for band in kept:
        band.pop("band_group", None)
    return kept


def detect_ladder_lane(signal: np.ndarray, sample_lanes: list, min_bands: int = 5):
    """Locate the MOLECULAR-WEIGHT LADDER column, which auto_detect_lanes can't
    find (the ladder is a tall vertical marker strip whose bands merge into
    vertical blobs that fail the horizontal band-shape filter).

    The ladder is conventionally the LEFTMOST lane, so we search only the region
    LEFT of the leftmost detected sample lane — this both (a) avoids the
    right-edge membrane noise that can fake a dense column, and (b) means a blot
    whose ladder was already detected as the leftmost lane (nothing to its left)
    correctly adds nothing. We pick the window with the most band-like peaks,
    scored by total peak strength so a faint noisy patch can't win.

    Returns (x1, x2, y1, y2) for the ladder lane box, or None."""
    height, width = signal.shape
    if not sample_lanes:
        return None

    def _count_bands(x0, x1, y0, y1):
        sub = signal[y0:y1, x0:x1]
        if sub.size == 0:
            return 0
        vp = gaussian_filter1d(np.percentile(sub, 65, axis=1), 2.2)
        base = gaussian_filter1d(vp, 18)
        pp = vp - base
        pp[pp < 0] = 0
        mp = max(np.percentile(pp, 80) * 0.30, 2.0)
        pk, _p = find_peaks(pp, distance=20, prominence=mp, width=(3, 35))
        return int(len(pk))

    # If the leftmost detected lane is ALREADY ladder-like (many bands), the
    # ladder was captured by auto_detect_lanes — don't add a duplicate.
    left_lane = min(sample_lanes, key=lambda l: l[1])
    if _count_bands(left_lane[1], left_lane[2], left_lane[3], left_lane[4]) >= min_bands:
        return None

    sx1 = min(l[1] for l in sample_lanes)
    if sx1 < 30:
        return None  # no room left of the leftmost lane → ladder already leftmost
    gy1 = min(l[3] for l in sample_lanes)
    gy2 = max(l[4] for l in sample_lanes)

    best = None  # (score, x0, x1, ytop, ybot)
    for win in (32, 40, 48):
        step = max(4, win // 6)
        x = 0
        while x + win <= sx1:
            col = signal[:, x:x + win]
            vp = gaussian_filter1d(np.percentile(col, 65, axis=1), 2.2)
            base = gaussian_filter1d(vp, 18)
            pp = vp - base
            pp[pp < 0] = 0
            min_prom = max(np.percentile(pp, 80) * 0.30, 2.0)
            peaks, props = find_peaks(pp, distance=20, prominence=min_prom, width=(3, 35))
            if len(peaks) >= min_bands:
                score = float(np.sum(props["prominences"]))  # total band strength
                if best is None or score > best[0]:
                    ytop = int(max(0, int(peaks.min()) - 25))
                    ybot = int(min(height, int(peaks.max()) + 25))
                    best = (score, x, x + win, ytop, ybot)
            x += step

    if best is None:
        return None
    _, x0, x1, ytop, ybot = best
    # Expand the lane box to cover both the sample y-range and the ladder's own
    # (taller) extent so detect_lane_bands sees every marker band.
    return (int(x0), int(x1), int(min(gy1, ytop)), int(max(gy2, ybot)))


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
    detect_ladder: bool = False,  # NOT in detect_bands.py auto mode; off for parity
):
    """Run the script's default auto-detect path and return (lanes, bands, (H, W)).

    ``rgb`` is an ``HxWx3`` array (any numeric dtype). Returns the named lanes
    (list of ``(name, x1, x2, y1, y2)``) and the consensus-filtered bands
    (list of dicts with lane/x1/x2/y/width/confidence/prominence), plus the
    signal shape so callers can normalise coordinates.
    """
    if marker_lanes is None:
        marker_lanes = set(DEFAULT_MARKER_LANES)

    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError("expected an HxWx3 RGB image")
    rgb = rgb[:, :, :3].astype(np.float32)

    _display, analysis = contrast_scale(rgb)
    signal = horizontal_signal(analysis, signal_polarity)

    lanes = auto_detect_lanes(
        signal=signal,
        expected_lanes=None,
        lane_width=None,
        lane_y1=None,
        lane_y2=None,
        min_gap=min_gap,
        threshold_percentile=threshold_percentile,
        max_band_width=max_band_width,
    )
    # Add the MW ladder column (auto_detect_lanes only finds horizontal sample
    # bands, never the vertical marker strip). Inserted leftmost so the
    # first-lane-marker naming labels it "L" and its bands are quantified.
    if detect_ladder:
        ladder = detect_ladder_lane(signal, lanes)
        if ladder is not None:
            lanes = list(lanes) + [("Ladder", ladder[0], ladder[1], ladder[2], ladder[3])]
            lanes.sort(key=lambda lane: lane[1])
    lanes = name_auto_lanes(lanes, lane_names=None, first_lane_marker=first_lane_marker)

    bands = detect_lane_bands(signal, lanes, marker_lanes)
    if min_group_lanes > 1:
        bands = filter_sample_band_groups(
            bands,
            group_tolerance=group_tolerance,
            marker_lanes=marker_lanes,
            min_lanes=min_group_lanes,
            consensus_tolerance=consensus_tolerance,
        )

    return lanes, bands, signal.shape
