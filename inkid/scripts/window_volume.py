import argparse
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm


def determine_image_min_max(im_path):
    im = iio.imread(im_path)
    filtered_im = gaussian_filter(im, 3)

    return np.amin(filtered_im), np.amax(filtered_im)


def rescale_and_write_to_file(input_img_path, minimum, window_width, output_dir):
    img = iio.imread(input_img_path).astype(float)

    rescaled_img = ((img - minimum) / window_width) * np.iinfo(np.uint16).max
    clipped_img = np.clip(
        rescaled_img,
        0,
        np.iinfo(np.uint16).max,
    )
    img = clipped_img.astype(np.uint16)

    output_path = output_dir / input_img_path.name
    iio.imwrite(output_path, img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-slices-dir", "-i", required=True, help="Input slice directory"
    )
    parser.add_argument(
        "--output-slices-dir", "-o", required=True, help="Output slice directory"
    )
    parser.add_argument(
        "--slice-skip",
        type=int,
        default=10,
        help="Number of slices to skip when determining global min/max",
    )
    args = parser.parse_args()

    input_slices_dir = Path(args.input_slices_dir)
    input_slices = sorted(input_slices_dir.glob("*.tif"))
    slices_for_finding_min_max = input_slices[:: args.slice_skip]

    output_slices_dir = Path(args.output_slices_dir)
    output_slices_dir.mkdir(exist_ok=True, parents=True)

    slice_mins = []
    slice_maxs = []

    print("Determining global min/max (after Gaussian blur)...")
    for test_slice_path in tqdm(slices_for_finding_min_max):
        slice_min, slice_max = determine_image_min_max(test_slice_path)
        slice_mins.append(slice_min)
        slice_maxs.append(slice_max)

    global_min = min(slice_mins)
    global_max = max(slice_maxs)
    window_width = global_max - global_min

    print(f"Global minimum: {global_min}")
    print(f"Global maximum: {global_max}")

    print("Windowing and writing slices...")
    for input_slice_path in tqdm(input_slices):
        rescale_and_write_to_file(
            input_slice_path, global_min, window_width, output_slices_dir
        )


if __name__ == "__main__":
    main()
