#!/usr/bin/env python3
"""Create a masked stereo ORB-SLAM input directory from one episode.

The source episode is never modified. Mask geometry and expected image dimensions come
from a JSON config. Output stereo frames are always PNG so stereo_kitti can load them.

When the actual frames differ from the config's image_size, the pixel geometry
(polygons and black_below_row) is linearly rescaled onto the frames — e.g. a
640x480 D405 config on 848x480 D435i IR frames — and a note is printed, so one
mask definition can serve several resolutions.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _validate_mask_config(config: dict, config_path: Path) -> dict:
    if config.get("schema_version") != 1:
        raise ValueError(f"{config_path}: schema_version must be 1")

    size = config.get("image_size")
    if not isinstance(size, dict):
        raise ValueError(f"{config_path}: image_size must be an object")
    width = size.get("width")
    height = size.get("height")
    if not isinstance(width, int) or width <= 0 or not isinstance(height, int) or height <= 0:
        raise ValueError(f"{config_path}: image_size width/height must be positive integers")

    streams = config.get("streams")
    if not isinstance(streams, dict):
        raise ValueError(f"{config_path}: streams must be an object")

    normalized = {"image_size": (width, height), "streams": {}}
    for stream in ("left", "right"):
        spec = streams.get(stream)
        if not isinstance(spec, dict):
            raise ValueError(f"{config_path}: streams.{stream} must be an object")
        polygons = spec.get("polygons", [])
        if not isinstance(polygons, list):
            raise ValueError(f"{config_path}: streams.{stream}.polygons must be a list")

        normalized_polygons = []
        for polygon_index, polygon in enumerate(polygons):
            if not isinstance(polygon, list) or len(polygon) < 3:
                raise ValueError(
                    f"{config_path}: streams.{stream}.polygons[{polygon_index}] "
                    "must contain at least three points"
                )
            points = np.asarray(polygon, dtype=np.float64)
            if points.shape != (len(polygon), 2) or not np.isfinite(points).all():
                raise ValueError(
                    f"{config_path}: streams.{stream}.polygons[{polygon_index}] "
                    "must contain finite [x, y] points"
                )
            if not np.equal(points, np.floor(points)).all():
                raise ValueError(f"{config_path}: mask polygon coordinates must be integers")
            points = points.astype(np.int32)
            if (
                (points[:, 0] < 0).any()
                or (points[:, 0] >= width).any()
                or (points[:, 1] < 0).any()
                or (points[:, 1] >= height).any()
            ):
                raise ValueError(
                    f"{config_path}: streams.{stream}.polygons[{polygon_index}] "
                    f"falls outside {width}x{height}"
                )
            normalized_polygons.append(points)

        black_below = spec.get("black_below_row")
        if black_below is not None and (
            not isinstance(black_below, int) or not 0 <= black_below <= height
        ):
            raise ValueError(
                f"{config_path}: streams.{stream}.black_below_row must be null "
                f"or an integer in [0, {height}]"
            )
        if not normalized_polygons and black_below is None:
            raise ValueError(f"{config_path}: streams.{stream} defines no mask")

        normalized["streams"][stream] = {
            "polygons": normalized_polygons,
            "black_below_row": black_below,
        }
    return normalized


def load_mask_config(config_path: Path | str) -> dict:
    config_path = Path(config_path).resolve()
    try:
        with config_path.open() as config_file:
            config = json.load(config_file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"failed to load mask config {config_path}: {error}") from error
    if not isinstance(config, dict):
        raise ValueError(f"{config_path}: top-level JSON value must be an object")
    return _validate_mask_config(config, config_path)


def _rescale_stream_config(stream_config: dict, from_size: tuple[int, int],
                           to_size: tuple[int, int]) -> dict:
    """Linearly rescale pixel mask geometry from one resolution to another."""
    scale = (to_size[0] / from_size[0], to_size[1] / from_size[1])
    return {
        "polygons": [
            np.round(polygon * np.asarray(scale, dtype=np.float64)).astype(np.int32)
            for polygon in stream_config["polygons"]
        ],
        "black_below_row": (
            None if stream_config["black_below_row"] is None
            else int(round(stream_config["black_below_row"] * scale[1]))
        ),
    }


def apply_mask(image: np.ndarray, stream_config: dict, image_size: tuple[int, int],
               source_path: Path | None = None) -> np.ndarray:
    if image is None:
        label = f": {source_path}" if source_path else ""
        raise ValueError(f"cannot mask an unreadable image{label}")
    width, height = image_size
    if image.shape[:2] != (height, width):
        label = f" ({source_path})" if source_path else ""
        raise ValueError(
            f"expected {width}x{height}, got {image.shape[1]}x{image.shape[0]}{label}"
        )

    masked = image.copy()
    for polygon in stream_config["polygons"]:
        cv2.fillPoly(masked, [polygon], 0)
    black_below = stream_config["black_below_row"]
    if black_below is not None:
        masked[black_below:] = 0
    return masked


def _find_stream_images(episode_dir: Path, stream: str) -> list[Path]:
    images = []
    for suffix in ("png", "jpg", "jpeg"):
        images.extend(episode_dir.glob(f"{stream}_*.{suffix}"))
    images = sorted(images, key=lambda path: path.stem)
    stems = [path.stem for path in images]
    if len(stems) != len(set(stems)):
        raise ValueError(f"{episode_dir}: duplicate {stream} frame stems across image formats")
    return images


def _link_metadata(source_dir: Path, output_dir: Path) -> None:
    metadata = [source_dir / "times.txt", *sorted(source_dir.glob("*.yaml"))]
    for source in metadata:
        if not source.exists():
            continue
        destination = output_dir / source.name
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        destination.symlink_to(source.resolve())


def prepare_masked_episode(
    source_dir: Path | str,
    output_dir: Path | str,
    config: dict,
) -> tuple[int, int]:
    source_dir = Path(source_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if not source_dir.is_dir():
        raise ValueError(f"episode directory does not exist: {source_dir}")
    if source_dir == output_dir:
        raise ValueError("masked output directory must differ from the source episode")

    output_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("left_*.png", "right_*.png"):
        for stale in output_dir.glob(pattern):
            stale.unlink()
    for stale_name in ("CameraTrajectory.txt", "CameraTrajectoryTransformed.txt"):
        stale = output_dir / stale_name
        if stale.exists():
            stale.unlink()
    _link_metadata(source_dir, output_dir)

    counts = {}
    for stream in ("left", "right"):
        images = _find_stream_images(source_dir, stream)
        if not images:
            raise ValueError(f"{source_dir}: no {stream}_* stereo images found")
        stream_config = config["streams"][stream]
        expected_size = config["image_size"]
        first = cv2.imread(str(images[0]), cv2.IMREAD_UNCHANGED)
        if first is None:
            raise ValueError(f"{source_dir}: cannot read {images[0]}")
        if (first.shape[1], first.shape[0]) != expected_size:
            expected_size = (first.shape[1], first.shape[0])
            stream_config = _rescale_stream_config(
                stream_config, config["image_size"], expected_size)
            print(
                f"mask geometry rescaled {config['image_size'][0]}x{config['image_size'][1]} "
                f"-> {expected_size[0]}x{expected_size[1]} for '{stream}' frames "
                f"({source_dir.name})"
            )
        for frame_index, source in enumerate(images):
            expected_stem = f"{stream}_{frame_index:06d}"
            if source.stem != expected_stem:
                raise ValueError(
                    f"{source_dir}: expected {expected_stem}, found {source.stem}"
                )
            image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
            masked = apply_mask(
                image,
                stream_config,
                expected_size,
                source,
            )
            destination = output_dir / f"{source.stem}.png"
            if not cv2.imwrite(str(destination), masked):
                raise OSError(f"failed to write masked image: {destination}")
        counts[stream] = len(images)

    if counts["left"] != counts["right"]:
        raise ValueError(
            f"{source_dir}: stereo frame count mismatch "
            f"({counts['left']} left, {counts['right']} right)"
        )

    times_path = source_dir / "times.txt"
    if not times_path.is_file():
        raise ValueError(f"{source_dir}: times.txt is required by stereo_kitti")
    timestamp_count = sum(1 for line in times_path.read_text().splitlines() if line.strip())
    if timestamp_count != counts["left"]:
        raise ValueError(
            f"{source_dir}: times.txt has {timestamp_count} timestamps for "
            f"{counts['left']} stereo frames"
        )
    return counts["left"], counts["right"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--config", required=True, type=Path, help="Gripper mask JSON config")
    args = parser.parse_args()

    try:
        config = load_mask_config(args.config)
        left_count, right_count = prepare_masked_episode(
            args.episode_dir,
            args.output_dir,
            config,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(
        f"Prepared masked ORB input: {left_count} left / {right_count} right frames "
        f"at {args.output_dir.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
