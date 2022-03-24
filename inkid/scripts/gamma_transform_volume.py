import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def uint16_img_to_float_0_1_array(input_img):
    assert input_img.mode == "I;16"
    return np.array(input_img.convert("F")) / np.iinfo(np.uint16).max


def float_0_1_array_to_uint16_img(input_array):
    input_array *= np.iinfo(np.uint16).max
    output_array = input_array.astype(np.uint16)
    return Image.fromarray(output_array, "I;16")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-slices-dir",
        "-i",
        required=True,
        help="Input slice directory (16-bit .tif images)",
    )
    parser.add_argument(
        "--output-slices-dir",
        "-o",
        required=True,
        help="Output slice directory (16-bit .tif images)",
    )
    parser.add_argument(
        "--output-mean",
        "-m",
        type=int,
        default=np.iinfo(np.uint16).max // 2,
        help="Apply a gamma transformation to the input slices such that their transformed mean takes on this value. "
        "If a region of interest is specified, the mean inside that region will take on this value. "
        "Value assumes 16-bit unsigned image.",
    )
    parser.add_argument(
        "--region-of-interest",
        "-r",
        type=int,
        nargs=4,
        default=None,
        help="Crop slices to this region (left, top, right, bottom) before performing intensity calculations. "
        "Crop is not applied to the output slices, only used for calculations.",
    )
    args = parser.parse_args()

    input_slices_dir = Path(args.input_slices_dir)
    input_slices = sorted(input_slices_dir.glob("*.tif"))

    output_slices_dir = Path(args.output_slices_dir)
    output_slices_dir.mkdir(exist_ok=True)

    for input_slice_path in tqdm(input_slices):
        input_slice_img = Image.open(input_slice_path)
        input_slice = uint16_img_to_float_0_1_array(input_slice_img)

        roi = np.copy(input_slice)
        if args.region_of_interest is not None:
            left, right, top, bottom = tuple(args.region_of_interest)
            roi = roi[top : bottom + 1, left : right + 1]

        input_mean = roi.mean()
        output_mean = args.output_mean / np.iinfo(np.uint16).max
        gamma = math.log(output_mean) / math.log(input_mean)

        output_slice = np.power(input_slice, gamma)

        output_slice_img = float_0_1_array_to_uint16_img(output_slice)
        output_slice_img.save(output_slices_dir / input_slice_path.name)


if __name__ == "__main__":
    main()
