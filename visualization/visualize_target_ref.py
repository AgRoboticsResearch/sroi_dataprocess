#!/usr/bin/env python3
"""QC contact sheet for target conditioning (target_ref).

Randomly samples episodes, computes the hindsight target point for each
(target_ref.py), and renders the initial image with the target-strawberry
marker into a multi-page PDF so labels can be checked quickly. Rejected
episodes are shown with their failure reason.

The future trajectory is never drawn — only the marker that the policy sees.

Usage (from repo root):
    python visualization/visualize_target_ref.py /path/to/session-png
    python visualization/visualize_target_ref.py /path/to/data --recursive --num-samples 24
    python visualization/visualize_target_ref.py /path/to/data --recursive -o /tmp/target_qc.pdf
"""

import argparse
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import target_ref as tr

EPISODES_PER_PAGE = 6  # 2 cols x 3 rows


def collect_episodes(path: Path, recursive: bool) -> list[Path]:
    if (path / "CameraTrajectoryTransformed.txt").exists() or (path / "CameraTrajectory.txt").exists():
        return [path]
    if recursive:
        return sorted(d for d in path.rglob("episode_*") if d.is_dir())
    return sorted(d for d in path.glob("episode_*") if d.is_dir())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", type=Path, help="Episode dir, session dir, or parent tree with --recursive")
    p.add_argument("--recursive", action="store_true", help="Find episode_* dirs recursively")
    p.add_argument("--num-samples", type=int, default=24, help="Episodes to sample (default 24; 0 = all)")
    p.add_argument("--seed", type=int, default=0, help="Sampling seed (default 0)")
    p.add_argument("-o", "--output", type=Path, default=Path("target_ref_qc.pdf"))
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--closed-threshold", type=float, default=0.5)
    p.add_argument("--grasp-hold-seconds", type=float, default=0.5)
    p.add_argument("--marker-style", choices=["circle", "bbox", "point"], default="circle")
    p.add_argument("--marker-radius-px", type=int, default=None)
    p.add_argument("--marker-offset-px", type=int, default=None)
    p.add_argument("--extrinsics-config", type=Path, default=None,
                   help=f"Camera-to-gripper-tip JSON (default: {tr.DEFAULT_EXTRINSICS_CONFIG})")
    args = p.parse_args(argv)

    if not args.path.exists():
        print(f"not found: {args.path}", file=sys.stderr)
        return 1

    episodes = collect_episodes(args.path, args.recursive)
    if not episodes:
        print(f"no episode_* dirs under {args.path}", file=sys.stderr)
        return 1
    if args.num_samples and len(episodes) > args.num_samples:
        episodes = sorted(random.Random(args.seed).sample(episodes, args.num_samples))
    print(f"{len(episodes)} episode(s) to render")

    tip_kin = tr.load_tip_kin(args.extrinsics_config or tr.DEFAULT_EXTRINSICS_CONFIG)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    with PdfPages(args.output) as pdf:
        for page_start in range(0, len(episodes), EPISODES_PER_PAGE):
            batch = episodes[page_start:page_start + EPISODES_PER_PAGE]
            fig, axes = plt.subplots(3, 2, figsize=(11.0, 9.5))
            for ax, ep in zip(axes.flat, batch):
                result = tr.compute_episode_target(
                    ep, tip_kin, fps=args.fps,
                    closed_threshold=args.closed_threshold, hold_s=args.grasp_hold_seconds,
                )
                ok = result["status"] == tr.STATUS_OK
                n_ok += ok
                if result["first_image"] is not None:
                    img = np.array(Image.open(result["first_image"]))
                    if ok:
                        img = tr.draw_target_marker(
                            img, result["u"], result["v"],
                            radius_px=args.marker_radius_px, offset_px=args.marker_offset_px,
                            style=args.marker_style,
                        )
                    ax.imshow(img)
                else:
                    ax.text(0.5, 0.5, "no frame", ha="center", va="center",
                            transform=ax.transAxes, fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
                color = "#2e7d32" if ok else "#c62828"
                label = f"{ep.parent.name}/{ep.name}"
                detail = (f"t_g={result['t_g']}  ({result['u']:.0f}, {result['v']:.0f})"
                          if ok else result["status"])
                ax.set_title(f"{label}\n{detail}", fontsize=9, color=color)
            for ax in axes.flat[len(batch):]:
                ax.axis("off")
            fig.suptitle(f"target_ref QC — {args.path.name}", fontsize=12)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            pdf.savefig(fig)
            plt.close(fig)

    print(f"{n_ok}/{len(episodes)} episodes with valid target")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
