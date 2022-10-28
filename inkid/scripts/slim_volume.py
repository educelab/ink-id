import argparse
import json
from pathlib import Path

import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm


matplotlib.use("TkAgg")


def get_slices(dirname):
    slice_files = []
    for child in dirname.iterdir():
        if not child.is_file():
            continue
        if child.name[0] != "." and child.suffix in [".tif", ".png"]:
            slice_files.append(str(child))
    slice_files.sort()

    return slice_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-slices", required=True)
    parser.add_argument("--projection-slices", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--voxels-above", type=int, default=32)
    parser.add_argument("--voxels-below", type=int, default=32)
    parser.add_argument("--vertical-flip", action="store_true")
    args = parser.parse_args()

    input_vol_path = Path(args.input_slices)
    projection_slices_path = Path(args.projection_slices)
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)

    # Load metadata
    metadata_filename = input_vol_path / "meta.json"
    if not metadata_filename.exists():
        raise FileNotFoundError(f"No volume meta.json file found in {input_vol_path}")
    else:
        with open(metadata_filename) as f:
            metadata = json.loads(f.read())
    shape = (metadata["slices"], metadata["height"], metadata["width"])

    slice_files = get_slices(input_vol_path)
    assert len(slice_files) == shape[0]

    projection_slice_files = get_slices(projection_slices_path)
    assert len(projection_slice_files) == shape[0]

    slim_vol = np.empty(
        (args.voxels_below + args.voxels_above, shape[0], shape[2]), dtype=np.uint16
    )
    for slice_i, (slice_file, projection_slice_file) in tqdm(
        list(enumerate(zip(slice_files, projection_slice_files)))
    ):
        projection_img = Image.open(projection_slice_file)

        # Find the green in each column.
        # The intersection line is rastered such that there are not (0, 255, 0) pixels but rather some are just more
        # green than others. So we first convert the image to a difference of green - (red + blue) and then find the
        # highest one of these (the greenest pixel in the column).
        projection_img = np.array(projection_img).astype(int)
        projection_img = (
            projection_img[:, :, 1] - projection_img[:, :, 0] - projection_img[:, :, 2]
        )
        projection_img = np.clip(projection_img, 0, None)
        greenest = np.argmax(projection_img, axis=0)

        slice_img = Image.open(slice_file)
        slice_img = np.array(slice_img, dtype=np.uint16)
        slimmed_slice_img = np.zeros(
            (args.voxels_below + args.voxels_above, shape[2]), dtype=np.uint16
        )

        # Fetch around that
        for x in range(shape[2]):
            y = greenest[x]
            if y != 0:
                slimmed_slice_img[:, x] = slice_img[
                    y - args.voxels_above : y + args.voxels_below, x
                ]

        if args.vertical_flip:
            slimmed_slice_img = np.flipud(slimmed_slice_img)

        slim_vol[:, slice_i, :] = slimmed_slice_img

    num_digits = len(str(slim_vol.shape[0]))
    for z in range(slim_vol.shape[0]):
        new_slice = Image.fromarray(slim_vol[z])
        new_slice_path = output_path / f"{z:0{num_digits}d}.tif"
        new_slice.save(new_slice_path)


if __name__ == "__main__":
    main()
