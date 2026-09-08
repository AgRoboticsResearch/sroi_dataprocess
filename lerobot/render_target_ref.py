#!/usr/bin/env python3
"""Render a deployment-time target_ref image: initial frame + target marker.

At deployment the user clicks the desired strawberry on the initial image; this
script draws the SAME marker used during training (target_ref.draw_target_marker)
so train/deploy style, resolution, and coordinate conventions match exactly.
Freeze the output image as observation.images.target_ref for the whole episode.

Usage:
    python lerobot/render_target_ref.py --image init.png --u 320 --v 240 -o target_ref.png
    # larger marker / different style:
    python lerobot/render_target_ref.py --image init.png --u 320 --v 240 --radius-px 40 --style bbox
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import target_ref as tr


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", type=Path, required=True, help="Initial RGB frame")
    p.add_argument("--u", type=float, required=True, help="Clicked pixel x (column)")
    p.add_argument("--v", type=float, required=True, help="Clicked pixel y (row)")
    p.add_argument("--radius-px", type=int, default=None,
                   help="Marker radius (default: 30 at 480p, scaled by resolution)")
    p.add_argument("--offset-px", type=int, default=0,
                   help="Vertical offset of the marker center below (u, v). Default 0: the user "
                        "clicks the fruit directly, so the circle is centered on the click.")
    p.add_argument("--style", choices=["circle", "bbox", "point"], default="circle")
    p.add_argument("-o", "--output", type=Path, required=True)
    args = p.parse_args(argv)

    if not args.image.exists():
        print(f"not found: {args.image}", file=sys.stderr)
        return 1
    image = np.array(Image.open(args.image).convert("RGB"))
    h, w = image.shape[:2]
    if not (0 <= args.u < w and 0 <= args.v < h):
        print(f"error: ({args.u}, {args.v}) outside image {w}x{h}", file=sys.stderr)
        return 1

    marked = tr.draw_target_marker(
        image, args.u, args.v,
        radius_px=args.radius_px, offset_px=args.offset_px, style=args.style,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(marked).save(args.output)
    print(f"wrote {args.output} (marker at ({args.u:.0f}, {args.v:.0f}), style={args.style})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
