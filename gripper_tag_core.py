"""Shared gripper-tag detection and series-recovery core.

Used by gripper_estimation_april_tag.py (config-driven CLI) and
gripper_estimation_april_tag.py (flag-driven CLI), so both
entry points share one implementation of:

  * per-frame AprilTag detection with a preprocessing fallback chain
    (raw -> CLAHE -> global histogram equalization). Strong-light episodes
    lose one tag under plain grayscale but recover with contrast
    normalization; good frames stop at the first attempt, so the chain only
    costs time on frames that need it.
  * the symmetric-rig constraint: the color camera looks straight at the
    jaw pivot, so left/right tag centers mirror around a fixed image
    center. Measured on normal episodes the center is stable to <0.5 px.
    - both tags detected  -> distance = |x_R - x_L|, cross-validated by
      |x_L + x_R - 2*cx| <= tol (violations are treated as no-detection;
      they are exactly the single-frame misfits that used to become spikes)
    - one tag detected    -> distance = 2*|x_tag - cx| (recovers frames
      where the other tag is occluded or unreadable)
  * flat-context spike removal (RETIRED 2026-10-23 — misdetections are now
    structurally excluded upstream; see experiments/spike_origin/): a run of at
    most 2 consecutive valid samples is a spike only when the surrounding
    valid samples on BOTH sides are each internally consistent, agree with
    each other (flat), and the run deviates from both by more than the
    threshold. Real fast motion fails the flat-context test and survives.
  * decoding quality gate (v3): window-strip candidates must decode with
    zero error correction (``hamming <= DECODING_HAMMING_MAX``). Glare and
    jaw-highlight misdetections are error-corrected guesses; full-corpus
    validation and the spike post-mortem live in experiments/spike_origin/.
  * gap filling by linear interpolation (interior) with nearest-value
    edges, replacing the old forward-fill that produced staircases during
    motion-blur dropouts.

cx is estimated per session as the median midpoint of both-tag frames
(falling back to the per-episode median and finally the image center),
and the per-episode stats record the center actually used.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

import cv2 as cv
import numpy as np

# Preprocessing variants tried in order until both tags are valid.
# equalize_hist recovered ~100% of the strong-light failures in testing;
# clahe is kept as an intermediate, gentler step.
DEFAULT_PREPROCESS_CHAIN = ("clahe", "equalize_hist")

DEFAULT_SYMMETRY_TOL_PX = 6.0  # legacy, superseded by the prior screen
DEFAULT_PRIOR_CENTER_PX = 320.0   # theoretical assembly center: image center
DEFAULT_PRIOR_CENTER_TOL_PX = 12.0  # loose prior screen; tolerates rig offsets
DEFAULT_TAG_DISTANCE_BOUNDS = (78.0, 267.0)  # calibrated stroke ±5 px (rig family)
DEFAULT_CALIB_MIN_PX = 83.0   # fixed normalization anchors from the 36-trial
DEFAULT_CALIB_MAX_PX = 262.0  # calibration extremes (捏死/全开), the safety pair
DEFAULT_ANCHOR_SANITY_PX = 5.0  # session stats beyond this from the pair -> fallback
DEFAULT_SPIKE_THRESHOLD = 0.15  # of the (max-min) gripper range, user-tuned
DEFAULT_SPIKE_MAX_LEN = 2       # spikes longer than this are kept (real data)
DEFAULT_SPIKE_CONTEXT = 3       # valid samples inspected on each side
DECODING_HAMMING_MAX = 0        # v3 decoding gate: reject error-corrected decodes

# Per-episode detection stats (gripper_detection_stats.json), v3 schema.
STATS_FIELD_DOCS = {
    "n_frames": "总帧数（等于 color 帧数）",
    "symmetry_center_px": "对称补全所用中心：session 级池化中位数；整 session 无可用中点时才回退本集估计",
    "n_valid_distances": "进入序列的有效开度样本数（补全判定后、插值前）",
    "coverage": "n_valid_distances / n_frames",
    "n_both_tags": "双 tag 均检出且过判据的帧数",
    "n_sym_filled": "仅单侧检出、经对称约束补全的帧数",
    "n_frames_preprocessed": "需 CLAHE/直方图均衡才检出的帧数",
    "gap_lengths_frames": "各缺检洞长度（帧）；>10 帧的洞为 NaN 交 QC",
}


def build_preprocessors():
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return {
        "clahe": clahe.apply,
        "equalize_hist": cv.equalizeHist,
    }


def _diag(tag) -> float:
    return float(np.linalg.norm(tag.corners[0] - tag.corners[2]))


def _diag2(tag) -> float:
    return float(np.linalg.norm(tag.corners[1] - tag.corners[3]))


def check_tag_valid(tag, nominal_diag_distance=50, threshold=10):
    """Check whether a detected tag's diagonal distances are within expected range."""
    if abs(_diag(tag) - nominal_diag_distance) > threshold or \
       abs(_diag2(tag) - nominal_diag_distance) > threshold:
        return False
    return True


def detect_frame_tags(gray, at_detector, left_id, right_id,
                      nominal_diag_distance, threshold,
                      preprocess_chain=DEFAULT_PREPROCESS_CHAIN,
                      preprocessors=None):
    """Detect the left/right gripper tags in one frame.

    Legacy full-band path kept for faithful pre-v3 comparisons: NO decoding
    quality gate here (error-corrected decodes are accepted on purpose).
    Tries plain grayscale first, then each preprocessing variant until both
    tags are valid, accumulating: a tag found by a later attempt fills the
    slot still missing from earlier ones. Returns a dict
    {"left": tag|None, "right": tag|None, "attempts": [variant names]}.
    """
    if preprocessors is None:
        preprocessors = build_preprocessors()

    def run_detect(img):
        found = {}
        for res in at_detector.detect(img):
            if res.tag_id in (left_id, right_id) and res.tag_id not in found:
                if check_tag_valid(res, nominal_diag_distance=nominal_diag_distance,
                                   threshold=threshold):
                    found[res.tag_id] = res
        return found

    found = run_detect(gray)
    attempts = ["raw"]
    for name in preprocess_chain:
        if left_id in found and right_id in found:
            break
        attempts.append(name)
        later = run_detect(preprocessors[name](gray))
        for tag_id, tag in later.items():
            found.setdefault(tag_id, tag)
    return {"left": found.get(left_id), "right": found.get(right_id),
            "attempts": attempts}


def detect_window_strip(strip, at_detector, tag_id, nominal_diag_distance,
                        threshold, preprocess_chain=DEFAULT_PREPROCESS_CHAIN,
                        preprocessors=None):
    """Detect ONE designated tag inside ONE cropped window strip.

    The window crop makes the ID<->side binding structural: the left strip
    only ever accepts the left tag's ID, so a tag detected on the wrong side
    cannot enter the series at all. Runs the raw strip first (zero overhead
    on good frames), then the preprocessing chain until the tag is found.
    Candidates must additionally decode with zero error correction
    (``hamming <= DECODING_HAMMING_MAX``, v3 decoding quality gate): glare
    and jaw-highlight misdetections are almost always error-corrected
    guesses (see experiments/spike_origin/).
    Returns (x_in_strip_or_None, used_preprocess).
    """
    if preprocessors is None:
        preprocessors = build_preprocessors()

    def run(img):
        for res in at_detector.detect(img):
            if res.tag_id == tag_id and check_tag_valid(
                    res, nominal_diag_distance=nominal_diag_distance,
                    threshold=threshold) and res.hamming <= DECODING_HAMMING_MAX:
                return res
        return None

    res = run(strip)
    if res is not None:
        return res.center[0], False
    for name in preprocess_chain:
        res = run(preprocessors[name](strip))
        if res is not None:
            return res.center[0], True
    return None, False


@dataclass
class EpisodeDetection:
    """Raw per-frame tag positions for one episode (pre-series)."""
    folder: str
    frame_count: int
    left_xs: list   # per-frame left tag center x (None when undetected)
    right_xs: list  # per-frame right tag center x (None when undetected)
    preprocess_frames: int = 0
    center_px: Optional[float] = None  # episode symmetry center, set by series()
    stats: dict = field(default_factory=dict)

    def series(self, center, tol=DEFAULT_SYMMETRY_TOL_PX,
               prior_center=DEFAULT_PRIOR_CENTER_PX,
               prior_tol=DEFAULT_PRIOR_CENTER_TOL_PX,
               distance_bounds=DEFAULT_TAG_DISTANCE_BOUNDS):
        """Valid distance samples under the rig's hardware constraints.

        Prior screen (both tags): the pair's midpoint must sit within
        `prior_tol` of the THEORETICAL assembly center `prior_center`
        (image center 320 by default). This is a loose existence screen
        against noise pairs; surviving midpoints cluster tightly and their
        median is the rig's measured center (`center`, computed by the
        caller from prior-passing frames only).

        both tags  -> d = |x_R - x_L| after the prior screen
        one tag    -> d = 2*|x - center| using the MEASURED rig center
        both       -> rejected when d falls outside `distance_bounds`
                      (the rig family's calibrated stroke in pixels).
        center=None -> no anchor: keep both-tag frames only (no mirror
        estimates).
        Returns (indices, values, flags).
        """
        d_lo, d_hi = distance_bounds

        indices, values, flags = [], [], []
        for i, (l, r) in enumerate(zip(self.left_xs, self.right_xs)):
            if l is not None and r is not None:
                if abs((l + r) / 2 - prior_center) > prior_tol:
                    continue  # prior screen failure -> treat as missing
                d = abs(r - l)
                if not (d_lo <= d <= d_hi):
                    continue  # outside the calibrated stroke
                indices.append(i); values.append(d); flags.append("both")
            elif l is not None or r is not None:
                if center is None:
                    continue
                x = l if l is not None else r
                d = 2.0 * abs(x - center)
                if not (d_lo <= d <= d_hi):
                    continue  # implausible mirror distance
                indices.append(i); values.append(d)
                flags.append("left_sym" if l is not None else "right_sym")
        return indices, values, flags


def detect_episode(image_paths, at_detector, left_id, right_id,
                   nominal_diag_distance, threshold, gripper_roi,
                   gripper_roi_bottom=None,
                   left_window=None, right_window=None,
                   preprocess_chain=DEFAULT_PREPROCESS_CHAIN,
                   preprocessors=None, verbose=False):
    """Detect tags for every frame of one episode (pass A).

    When left_window/right_window ((x0, x1) pixel columns) are given, each
    frame is cropped to the two window strips inside the row band and each
    strip only accepts its designated tag ID (structural ID<->side binding;
    detected x is offset back to full-image coordinates). Without windows,
    falls back to the legacy full-band masking with both IDs accepted
    anywhere in the band.
    """
    if preprocessors is None:
        preprocessors = build_preprocessors()

    left_xs, right_xs = [], []
    n_pre = 0
    for path in image_paths:
        image = cv.imread(str(path))
        if image is None:
            if verbose:
                print(f"failed to read {path}")
            left_xs.append(None); right_xs.append(None)
            continue
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        if left_window is not None and right_window is not None:
            y0 = gripper_roi
            y1 = gripper_roi_bottom if gripper_roi_bottom is not None else gray.shape[0]
            l0, l1 = left_window
            r0, r1 = right_window
            lx, pre_l = detect_window_strip(
                gray[y0:y1, l0:l1], at_detector, left_id,
                nominal_diag_distance, threshold,
                preprocess_chain=preprocess_chain, preprocessors=preprocessors)
            rx, pre_r = detect_window_strip(
                gray[y0:y1, r0:r1], at_detector, right_id,
                nominal_diag_distance, threshold,
                preprocess_chain=preprocess_chain, preprocessors=preprocessors)
            left_xs.append(lx + l0 if lx is not None else None)
            right_xs.append(rx + r0 if rx is not None else None)
            if pre_l or pre_r:
                n_pre += 1
            if verbose:
                print(path.name, "L=", left_xs[-1], "R=", right_xs[-1])
            continue
        gray[0:gripper_roi, :] = 0
        if gripper_roi_bottom is not None:
            gray[gripper_roi_bottom:, :] = 0
        hit = detect_frame_tags(gray, at_detector, left_id, right_id,
                                nominal_diag_distance, threshold,
                                preprocess_chain=preprocess_chain,
                                preprocessors=preprocessors)
        if len(hit["attempts"]) > 1:
            n_pre += 1
        left_xs.append(hit["left"].center[0] if hit["left"] else None)
        right_xs.append(hit["right"].center[0] if hit["right"] else None)
        if verbose:
            print(path.name, "L=", hit["left"], "R=", hit["right"])

    det = EpisodeDetection(
        folder=str(image_paths[0].parent) if image_paths else "",
        frame_count=len(image_paths),
        left_xs=left_xs, right_xs=right_xs,
        preprocess_frames=n_pre,
    )
    # Rig center estimate: median of midpoints that PASS the prior screen
    # (|mid - 320| <= 12 px), so noise pairs cannot skew the measured center.
    mids = [(l + r) / 2 for l, r in zip(left_xs, right_xs)
            if l is not None and r is not None
            and abs((l + r) / 2 - DEFAULT_PRIOR_CENTER_PX) <= DEFAULT_PRIOR_CENTER_TOL_PX]
    det.center_px = float(np.median(mids)) if mids else None
    return det


def run_center_fallback(detections):
    """Session-level symmetry center, used uniformly for every episode.

    A session is always recorded on one rig, so all episodes' prior-passing
    midpoints belong to the same distribution -- pooling them (instead of
    per-episode estimates) maximizes the sample count and immunizes episodes
    with few both-tag frames. Median of all prior-screened midpoints; None
    if the whole session produced none."""
    mids = []
    for d in detections:
        for l, r in zip(d.left_xs, d.right_xs):
            if l is not None and r is not None:
                mid = (l + r) / 2
                if abs(mid - DEFAULT_PRIOR_CENTER_PX) <= DEFAULT_PRIOR_CENTER_TOL_PX:
                    mids.append(mid)
    if mids:
        return float(np.median(mids))
    return None


def remove_spikes(indices, values, range_px,
                  threshold=DEFAULT_SPIKE_THRESHOLD,
                  max_len=DEFAULT_SPIKE_MAX_LEN,
                  context=DEFAULT_SPIKE_CONTEXT):
    """Remove isolated 1-2 sample outliers that stick out of BOTH-side envelopes.

    A spike is a run of up to `max_len` valid samples where every value sits
    beyond max(context)+thr (sticking up) or below min(context)-thr (sticking
    down), with `context` valid samples available on EACH side. The two sides
    do NOT need to agree with each other, so spikes riding on a fast ramp
    (left cluster high, right cluster low) are still caught. Real fast motion
    survives: its samples move continuously and never stick out of both
    envelopes at once. One side missing -> keep (conservative).
    Returns the positions (indices into `indices`) to drop.
    """
    thr = threshold * range_px
    n = len(indices)
    drop = []
    k = 0
    while k < n:
        removed_here = False
        for run in range(1, max_len + 1):
            if k + run > n:
                break
            pre = values[max(0, k - context):k]
            post = values[k + run: k + run + context]
            if not pre or not post:
                break  # no context on one side -> keep (conservative)
            env_hi = max(pre + post) + thr
            env_lo = min(pre + post) - thr
            seg_vals = values[k:k + run]
            if all(v > env_hi for v in seg_vals) or all(v < env_lo for v in seg_vals):
                drop.extend(range(k, k + run))
                k += run
                removed_here = True
                break
        if not removed_here:
            k += 1
    return drop


def _gap_lengths(frame_count, indices):
    """Lengths of all invalid stretches (interior + edges) of the valid set."""
    gaps = []
    for j in range(len(indices) - 1):
        g = indices[j + 1] - indices[j] - 1
        if g > 0:
            gaps.append(g)
    if indices:
        if indices[0] > 0:
            gaps.append(indices[0])
        tail = frame_count - 1 - indices[-1]
        if tail > 0:
            gaps.append(tail)
    return gaps


def interpolate_gaps(frame_count, indices, values, mode="linear",
                     max_interp_gap=None):
    """Full-length series from valid samples, with a gap-length policy.

    linear + max_interp_gap=None : plain np.interp (linear inside, nearest
    at the edges) -- the original behavior.
    linear + max_interp_gap=N    : interior gaps of length <= N are linearly
    interpolated; longer gaps stay NaN (classified as sustained occlusion:
    the jaw position is unknowable and downstream QC should cull or handle
    the episode). Leading/trailing gaps follow the same rule (short -> pad
    with the nearest valid value, long -> NaN).
    forward_fill: legacy behavior (first-valid seed, then previous value),
    gaps are never NaN.
    """
    if not values:
        return None
    if mode == "forward_fill":
        out = np.empty(frame_count)
        prev = values[0]
        vi = 0
        for i in range(frame_count):
            if vi < len(indices) and indices[vi] == i:
                prev = values[vi]; vi += 1
            out[i] = prev
        return out
    out = np.interp(np.arange(frame_count), indices, values)
    if max_interp_gap is not None:
        n = len(indices)
        # interior gaps
        for j in range(n - 1):
            gap = indices[j + 1] - indices[j] - 1
            if gap > max_interp_gap:
                out[indices[j] + 1: indices[j + 1]] = np.nan
        # edge gaps
        if indices[0] > max_interp_gap:
            out[: indices[0]] = np.nan
        if frame_count - 1 - indices[-1] > max_interp_gap:
            out[indices[-1] + 1:] = np.nan
    return out


def finalize_series(detection, min_distance, max_distance, *,
                    center=None, symmetry_tol=DEFAULT_SYMMETRY_TOL_PX,
                    prior_center=DEFAULT_PRIOR_CENTER_PX,
                    prior_tol=DEFAULT_PRIOR_CENTER_TOL_PX,
                    distance_bounds=DEFAULT_TAG_DISTANCE_BOUNDS,
                    interp_mode="linear",
                    max_interp_gap=None,
                    spike_threshold=DEFAULT_SPIKE_THRESHOLD,
                    spike_max_len=DEFAULT_SPIKE_MAX_LEN,
                    spike_context=DEFAULT_SPIKE_CONTEXT,
                    median_filter_fn=None):
    """Apply symmetry series, range-dependent cleanup and gap filling.

    `center` should be the episode center when measurable, else the run
    center (may stay None: both-tag frames only, no mirror estimates).
    Returns (raw_filled np.array or None, stats dict).
    """
    if center is None:
        center = detection.center_px
    indices, values, flags = detection.series(
        center, prior_center=prior_center, prior_tol=prior_tol,
        distance_bounds=distance_bounds)

    # 尖峰剔除已退役（2026-10-23，v3 schema 起一并移除 stats 键）：误检在检测
    # 层被窗条+尺寸闸+解码质量闸（hamming==0）+先验筛+行程界结构性排除后，
    # 序列层去尖峰在全量重跑中零触发。成因分析、旧版复现与全集验证见
    # experiments/spike_origin/。remove_spikes 函数定义保留作历史参考，不再
    # 调用；stats 不再输出 spikes_removed 键（v3 schema，无下游依赖）。
    # range_px = max_distance - min_distance
    # if range_px > 0 and len(indices) > 2:
    #     drop = remove_spikes(indices, values, range_px,
    #                          threshold=spike_threshold,
    #                          max_len=spike_max_len,
    #                          context=spike_context)
    #     if drop:
    #         spikes = [{"frame": indices[d], "value": values[d]} for d in drop]
    #         keep = set(range(len(indices))) - set(drop)
    #         indices = [indices[j] for j in sorted(keep)]
    #         values = [values[j] for j in sorted(keep)]
    #         flags = [flags[j] for j in sorted(keep)]
    raw = interpolate_gaps(detection.frame_count, indices, values,
                            mode=interp_mode, max_interp_gap=max_interp_gap)
    if raw is not None and median_filter_fn is not None:
        raw = median_filter_fn(raw)
    stats = {
        "n_frames": detection.frame_count,
        "symmetry_center_px": center,
        "n_valid_distances": len(indices),
        "coverage": len(indices) / detection.frame_count if detection.frame_count else 0.0,
        "n_both_tags": flags.count("both"),
        "n_sym_filled": flags.count("left_sym") + flags.count("right_sym"),
        "n_frames_preprocessed": detection.preprocess_frames,
        "gap_lengths_frames": _gap_lengths(detection.frame_count, indices),
    }
    return raw, {"field_docs": STATS_FIELD_DOCS, **stats}


def find_distances_max_min(distances, threshold=0.05, threshold_num=10):
    """Find robust max/min distances by requiring multiple measurements near the extremum."""
    distances_copy = copy.deepcopy(distances)
    ret_max_distance, ret_min_distance = None, None
    for _ in range(len(distances)):
        max_dist_id = np.argmax(distances)
        max_distance = distances[max_dist_id]
        data_in_range = distances[
            (distances > max_distance * (1 - threshold)) &
            (distances < max_distance * (1 + threshold))
        ]
        if len(data_in_range) >= threshold_num:
            ret_max_distance = max_distance
            break
        distances[max_dist_id] = -1
    if ret_max_distance is None:
        ret_max_distance = np.max(distances_copy)

    distances = distances_copy
    for _ in range(len(distances)):
        min_dist_id = np.argmin(distances)
        min_distance = distances[min_dist_id]
        data_in_range = distances[
            (distances > min_distance * (1 - threshold)) &
            (distances < min_distance * (1 + threshold))
        ]
        if len(data_in_range) >= threshold_num:
            ret_min_distance = min_distance
            break
        distances[min_dist_id] = np.inf
    if ret_min_distance is None:
        ret_min_distance = np.min(distances_copy)

    return ret_max_distance, ret_min_distance
