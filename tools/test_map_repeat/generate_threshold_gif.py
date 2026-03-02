#!/usr/bin/env python3
"""Generate a back-and-forth looping GIF that thresholds ranked_output.png at all levels."""

import argparse
import numpy as np
from PIL import Image
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Generate threshold sweep GIF from ranked dither array")
    parser.add_argument("-i", "--input", default=None, help="Input ranked image (default: ranked_output.png in same dir)")
    parser.add_argument("-o", "--output", default=None, help="Output GIF path (default: threshold_sweep.gif in same dir)")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")
    parser.add_argument("--skip", type=int, default=1, help="Step size between threshold levels (default: 1)")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    input_path = Path(args.input) if args.input else script_dir / "ranked_output.png"
    output_path = Path(args.output) if args.output else script_dir / "threshold_sweep.gif"

    img = Image.open(input_path)
    arr = np.array(img)
    print(f"Input: {input_path} ({img.size[0]}x{img.size[1]}, {img.mode})")
    print(f"Value range: {arr.min()}-{arr.max()}, unique: {len(np.unique(arr))}")

    # Generate threshold levels: 0, skip, 2*skip, ..., 255, then back down
    levels_forward = list(range(0, 256, args.skip))
    # Ensure 255 is included
    if levels_forward[-1] != 255:
        levels_forward.append(255)
    # Back: reverse excluding endpoints (no duplicate of first/last frame)
    levels_backward = levels_forward[-2:0:-1]
    levels = levels_forward + levels_backward

    print(f"Threshold levels: {len(levels_forward)} forward + {len(levels_backward)} backward = {len(levels)} frames")
    print(f"FPS: {args.fps}, duration: {len(levels) / args.fps:.1f}s")

    # Generate frames
    frames = []
    for level in levels:
        # Pixels with value < level become white (on), rest black (off)
        binary = ((arr < level) * 255).astype(np.uint8)
        frames.append(Image.fromarray(binary, mode='L'))

    # Save as GIF
    frame_duration = int(1000 / args.fps)  # ms per frame
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration,
        loop=0,  # infinite loop
    )

    print(f"Output: {output_path} ({len(frames)} frames, {frame_duration}ms/frame)")


if __name__ == "__main__":
    main()
