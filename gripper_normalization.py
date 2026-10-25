"""Shared provenance helpers for normalized gripper-distance outputs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


MANIFEST_NAME = "gripper_normalization.json"
ROBUST_EXTREMUM_THRESHOLD = 0.05
ROBUST_EXTREMUM_MIN_SAMPLES = 10


def build_manifest(
    *,
    min_distance: float,
    max_distance: float,
    source_type: str,
    config_path: str | Path | None,
    config: dict[str, Any],
    median_filter_size: int,
    episode_count: int,
    frame_count: int,
    interp_mode: str = "linear",
    preprocess_chain: list[str] | None = None,
    symmetry_enabled: bool = True,
    symmetry_tol_px: float | None = None,
    hamming_max: int = 0,
) -> dict[str, Any]:
    """Build the normalization provenance shared by a processing run."""
    min_distance = float(min_distance)
    max_distance = float(max_distance)
    if not max_distance > min_distance:
        raise ValueError(
            f"gripper max must be greater than min (min={min_distance}, max={max_distance})"
        )

    source: dict[str, Any] = {"type": source_type}
    if config_path is not None:
        source["config_path"] = str(Path(config_path).expanduser().resolve())
    reference = config.get("gripper_range_source")
    if reference is not None:
        source["reference"] = reference
    if source_type == "pooled_robust_extrema":
        source["algorithm"] = {
            "name": "clustered_extremum",
            "relative_threshold": ROBUST_EXTREMUM_THRESHOLD,
            "minimum_nearby_samples": ROBUST_EXTREMUM_MIN_SAMPLES,
        }

    fill_description = (
        "linear_interpolation_interior_nearest_edges"
        if interp_mode == "linear"
        else "leading_first_valid_then_forward_fill"
    )
    processing: dict[str, Any] = {
        "missing_detection_fill": fill_description,
        "median_filter_size": int(median_filter_size),
        "preprocess_chain": list(preprocess_chain) if preprocess_chain else [],
        "symmetry_constraint": {
            "enabled": bool(symmetry_enabled),
            **({"tolerance_px": float(symmetry_tol_px)}
               if symmetry_tol_px is not None else {}),
        },
    }
    # v3 decoding quality gate (spike-removal retired 2026-10-24, see
    # experiments/spike_origin/ in the repo): error-corrected decodes are
    # rejected at detection time.
    processing["decoding_quality_gate"] = {"hamming_max": int(hamming_max)}

    return {
        "schema_version": 2,
        "output_file": "gripper_distances.txt",
        "value_convention": {
            "closed": 0.0,
            "open": 1.0,
        },
        "normalization": {
            "formula": "clip((raw_distance_px - min_distance_px) / "
                       "(max_distance_px - min_distance_px), 0, 1)",
            "min_distance_px": min_distance,
            "max_distance_px": max_distance,
        },
        "range_source": source,
        "detection": {
            "tag_family": config.get("tag_family"),
            "left_gripper_tag_id": config.get("left_gripper_tag_id"),
            "right_gripper_tag_id": config.get("right_gripper_tag_id"),
            "gripper_roi_top": config.get("gripper_roi"),
            "gripper_roi_bottom": config.get("gripper_roi_bottom"),
            "nominal_diag_distance_px": config.get(
                "nominal_diag_distance", config.get("norminal_diag_distance")
            ),
            "tag_size_threshold_px": config.get("threshold"),
        },
        "processing": processing,
        "run_scope": {
            "episode_count": int(episode_count),
            "frame_count": int(frame_count),
        },
    }


def write_manifest(directory: str | Path, manifest: dict[str, Any]) -> Path:
    """Atomically write a normalization manifest into *directory*."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / MANIFEST_NAME
    temporary = directory / f".{MANIFEST_NAME}.tmp"
    with temporary.open("w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    os.replace(temporary, destination)
    return destination
