"""'Scoop' the pixels underneath a projected mesh surface and set them to a fixed fill value.

This script modifies a volume using a projected mesh surface and 'scoops' all pixels under the projected line, setting
them to a specified fill value (with some configurable jitter). The purpose is to remove a surface from a volume to
reveal the next layer, so it can be better segmented (likely with Canny edge detection) and further processed.
"""

import argparse
import multiprocessing
from pathlib import Path

from multiprocessing import Pool
import numpy as np
from PIL import Image
from tqdm import tqdm


def scoop(input_tuple: tuple) -> None:
    filename, input_volume_dir, output_volume_dir, fill, jitter, flip_vertical = input_tuple
    projection_img = Image.open(filename).convert("RGB")
    output_img = Image.open(input_volume_dir / f"{filename.stem}.tif").copy()
    assert output_img.mode == "I;16"

    if flip_vertical:
        projection_img = projection_img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        output_img = output_img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    # Column by column
    for x in range(projection_img.width):
        # Within a column look for where the line intersects. Assume line has original width 1 but anti aliasing
        # makes it width 3. So we find the first instance of color and then go one pixel further for line center.
        for y in range(projection_img.height - 1, 0, -1):
            rgb = projection_img.getpixel((x, y))
            if (
                len(set(rgb)) > 1
            ):  # Check if there is more than 1 unique value within RGB values, indicating color
                for inner_y in range(y - 1, projection_img.height):
                    output_img.putpixel(
                        (x, inner_y), fill + np.random.randint(-jitter, jitter)
                    )
                break  # On to the next column
    output_img.save(output_volume_dir / f"{filename.stem}.tif")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input-volume-dir",
        required=True,
        help="Input volume slices to be modified (scooped)",
    )
    parser.add_argument(
        "-p",
        "--projections-dir",
        required=True,
        help="Slices with a projected mesh, below which will be scooped out",
    )
    parser.add_argument(
        "-o",
        "--output-volume-dir",
        required=True,
        help="Output directory for scooped slices",
    )
    parser.add_argument(
        "-f",
        "--fill",
        type=int,
        default=32768,
        help="Fill scooped region with this intensity (16-bit range)",
    )
    parser.add_argument(
        "-j",
        "--jitter",
        type=int,
        default=0,
        help="Randomly vary intensity of scooped pixels from -j to +j",
    )
    parser.add_argument(
        "--flip-vertical", action="store_true", help="Flip the loaded images vertically before scooping"
    )
    args = parser.parse_args()

    input_volume_dir = Path(args.input_volume_dir)
    projections_dir = Path(args.projections_dir)
    output_volume_dir = Path(args.output_volume_dir)
    output_volume_dir.mkdir(exist_ok=True)

    projection_filenames = sorted(list(projections_dir.glob("*.png")))
    with Pool(processes=multiprocessing.cpu_count() * 3 // 4) as p:
        pool_args = [
            (f, input_volume_dir, output_volume_dir, args.fill, args.jitter, args.flip_vertical)
            for f in projection_filenames
        ]
        for _ in tqdm(p.imap(scoop, pool_args), total=len(projection_filenames)):
            pass


if __name__ == "__main__":
    main()
