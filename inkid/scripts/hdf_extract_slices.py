import argparse
from pathlib import Path
import timeit

import h5py
import imageio.v3 as iio
import numpy as np
from tqdm import tqdm

"""
This script extracts .tif slices from .hdf input files, optionally windowing and/or cropping them
"""


def window_img(img, input_min, input_max, output_min, output_max):
    input_window_width = input_max - input_min
    output_window_width = output_max - output_min
    img = img.astype(float)
    img -= input_min  # Shift to 0
    img /= input_window_width  # Scale to 0-1
    img *= output_window_width  # Scale to output window
    img += output_min  # Shift to output window
    img = np.clip(
        img,
        output_min,
        output_max,
    )
    return img


def main():
    parser = argparse.ArgumentParser(
        description="Extract .tif slices from .hdf input files"
    )
    parser.add_argument(
        "--input-files",
        "-i",
        required=True,
        help="Input HDF file(s). Can be specified multiple times",
        nargs="+",
    )
    parser.add_argument("--output-dir", "-o", required=True, help="Output directory")
    parser.add_argument(
        "--combine-output-in-single-dir",
        action="store_true",
        help="Combine all slices into a single output directory "
        "(default is to create a separate subdirectory for each input .hdf file)",
    )
    parser.add_argument(
        "--slice-skip", type=int, default=1, help="Only sample every Nth slice"
    )
    parser.add_argument(
        "--dataset-name",
        default="entry/data/data",
        help="Dataset to fetch within the HDF file(s)",
    )
    # Cropping
    parser.add_argument(
        "--crop-min-x",
        type=int,
        default=0,
        help="Crop the resulting slice images to start at a minimum x value.",
    )
    parser.add_argument(
        "--crop-width",
        type=int,
        default=None,
        help="Crop the resulting slice images to have a specified width.",
    )
    parser.add_argument(
        "--crop-min-y",
        type=int,
        default=0,
        help="Crop the resulting slice images to start at a minimum y value.",
    )
    parser.add_argument(
        "--crop-height",
        type=int,
        default=None,
        help="Crop the resulting slice images to have a specified height.",
    )
    # Windowing
    parser.add_argument(
        "--input-window-min", type=float, default=-1.0, help="Input window min"
    )
    parser.add_argument(
        "--input-window-max", type=float, default=1.0, help="Input window max"
    )
    parser.add_argument(
        "--output-window-min", type=float, default=0.0, help="Output window min"
    )
    parser.add_argument(
        "--output-window-max",
        type=float,
        default=np.iinfo(np.uint16).max,
        help="Output window max",
    )
    # Automatic percentile windowing
    parser.add_argument(
        "--auto-percentile-windowing",
        action="store_true",
        help="Apply automatic percentile windowing",
    )
    parser.add_argument(
        "--percentile-min", type=float, default=0.01, help="Percentile min"
    )
    parser.add_argument(
        "--percentile-max", type=float, default=99.99, help="Percentile max"
    )
    parser.add_argument(
        "--percentile-slice-samples",
        type=int,
        default=100,
        help="Number of slices to sample for percentile calculation",
    )
    args = parser.parse_args()

    start = timeit.default_timer()

    # Print files and their datasets
    total_depth = 0
    for file in args.input_files:
        if not file.endswith(".hdf"):
            raise ValueError(f"Error, file {file} is not of valid extension .hdf")
        input_file = h5py.File(file, "r")
        print(f"File: {file} contains datasets:")
        input_file.visit(lambda x: print(f"\t{x}"))  # Print all datasets in file
        in_data = input_file[args.dataset_name]
        assert isinstance(
            in_data, h5py.Dataset
        ), "Error, data at this path is not of type Dataset"
        (depth, height, width) = in_data.shape
        total_depth += depth

    input_window_min = args.input_window_min
    input_window_max = args.input_window_max
    output_window_min = args.output_window_min
    output_window_max = args.output_window_max

    crop_max_x = None
    if args.crop_width is not None:
        crop_max_x = args.crop_min_x + args.crop_width
    crop_max_y = None
    if args.crop_height is not None:
        crop_max_y = args.crop_min_y + args.crop_height

    if args.auto_percentile_windowing:
        print("Calculating input window based on percentiles...")

        input_window_min_samples = []
        input_window_max_samples = []

        for _ in tqdm(range(args.percentile_slice_samples)):
            file = np.random.choice(args.input_files)
            input_file = h5py.File(file, "r")
            in_data = input_file[args.dataset_name]

            assert isinstance(
                in_data, h5py.Dataset
            ), "Error, data at this path is not of type Dataset"
            (depth, height, width) = in_data.shape
            z = np.random.randint(depth)

            img = in_data[z, args.crop_min_y : crop_max_y, args.crop_min_x : crop_max_x]
            input_window_min_samples.append(np.percentile(img, args.percentile_min))
            input_window_max_samples.append(np.percentile(img, args.percentile_max))

        input_window_min = np.mean(input_window_min_samples)
        input_window_max = np.mean(input_window_max_samples)

        print(
            f"Input window min: {input_window_min}, input window max: {input_window_max}"
        )

    slice_counter = -1

    for file_name in args.input_files:
        print(f"Processing {file_name}...")

        with h5py.File(file_name, "r") as input_file:
            dataset = input_file[args.dataset_name]

            assert isinstance(
                dataset, h5py.Dataset
            ), "Error, data at this path is not of type Dataset"
            (depth, height, width) = dataset.shape

            if args.combine_output_in_single_dir:
                this_output_dir = Path(args.output_dir)
            else:
                this_output_dir = Path(args.output_dir) / Path(file_name).stem
            this_output_dir.mkdir(parents=True, exist_ok=True)

            for z in tqdm(range(depth), desc="Extract .tifs"):
                slice_counter += 1

                if z % args.slice_skip != 0:
                    continue

                img = dataset[
                    z, args.crop_min_y : crop_max_y, args.crop_min_x : crop_max_x
                ]

                img = window_img(
                    img,
                    input_window_min,
                    input_window_max,
                    output_window_min,
                    output_window_max,
                )

                img = img.astype("uint16")

                if args.combine_output_in_single_dir:
                    slice_name = (
                        str(slice_counter).zfill(len(str(total_depth))) + ".tif"
                    )
                else:
                    slice_name = str(z).zfill(len(str(depth))) + ".tif"

                slice_path = this_output_dir / slice_name
                iio.imwrite(slice_path, img)

    print(f"Total time: {timeit.default_timer() - start}")


if __name__ == "__main__":
    main()
