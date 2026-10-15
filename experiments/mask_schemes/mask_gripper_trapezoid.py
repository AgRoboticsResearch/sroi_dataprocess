#!/usr/bin/env python3
"""Apply a config-driven gripper mask to a schemes tree, session, or episode."""

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from apply_gripper_mask import load_mask_config, prepare_masked_episode

DEFAULT_MASK_CONFIG = REPO_ROOT / "configs" / "gripper_mask_sroi_v2_d405.json"


def find_episodes(path: Path) -> list[Path]:
    if path.name.startswith("episode_") and path.is_dir():
        return [path]
    return sorted(
        child
        for child in path.iterdir()
        if child.is_dir() and child.name.startswith("episode_")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Schemes/session directory or one episode")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_MASK_CONFIG,
        help=f"Gripper mask JSON config (default: {DEFAULT_MASK_CONFIG})",
    )
    args = parser.parse_args()
    input_path = args.path.resolve()
    if not input_path.is_dir():
        parser.error(f"directory not found: {input_path}")

    episodes = find_episodes(input_path)
    if not episodes:
        parser.error(f"no episode_* directories found under: {input_path}")

    try:
        config = load_mask_config(args.config)
        for episode in episodes:
            source = episode / "raw" if (episode / "raw").is_dir() else episode
            output = episode / "maskgripper"
            left_count, right_count = prepare_masked_episode(source, output, config)
            print(f"  {episode.name}: {left_count}L {right_count}R", flush=True)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print(f"done -> {input_path} ({len(episodes)} episodes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
