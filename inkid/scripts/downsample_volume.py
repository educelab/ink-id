import argparse
from datetime import datetime
import json
import math
from pathlib import Path

from tqdm import tqdm
from wand.image import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("scale", type=int)
    parser.add_argument("output")
    args = parser.parse_args()

    if not math.log2(args.scale).is_integer():
        print(f"error: Scale factor {args.scale} is not a power of 2.")
        return

    new_volume_dirname = datetime.now().strftime("%Y%m%d%H%M%S")
    out_dir = Path(args.output) / new_volume_dirname
    out_dir.mkdir()

    # Remove slices to downsample on z axis
    image_filenames = sorted(Path(args.input).glob("*.tif"))
    image_filenames = image_filenames[:: args.scale]

    # Precalculate some values
    num_digits_in_filenames = math.ceil(math.log10(len(image_filenames)))
    with Image(filename=image_filenames[0]) as img:
        new_width = img.width // args.scale
        new_height = img.height // args.scale
        new_slices = len(image_filenames)

    # Use imagemagick to downsample remaining images in xy
    for i, image_filename in tqdm(list(enumerate(image_filenames))):
        with Image(filename=image_filename) as img:
            with img.clone() as img_clone:
                img_clone.resize(width=new_width, height=new_height)
                img_clone.save(filename=out_dir / f"{i:0{num_digits_in_filenames}}.tif")

    # New meta.json file
    with open(Path(args.input) / "meta.json") as f:
        metadata = json.load(f)
        metadata["width"] = new_width
        metadata["height"] = new_height
        metadata["slices"] = new_slices
        metadata["uuid"] = new_volume_dirname
        metadata["name"] = f'{metadata["name"]}, downscaled {args.scale}x'
        metadata["voxelsize"] = metadata["voxelsize"] * args.scale

    with open(out_dir / "meta.json", "w") as f:
        json.dump(metadata, f)


if __name__ == "__main__":
    main()
