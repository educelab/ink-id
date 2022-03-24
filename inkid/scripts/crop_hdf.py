import argparse
import os
import timeit

import h5py
from PIL import Image
from tqdm import tqdm

"""
This script internally crops hdf files or converts them to tiff format
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract HDF to HDF or TIFF")
    parser.add_argument(
        "--input_files", "-i", required=True, help="Input NXS/HDF file", nargs="+"
    )
    parser.add_argument("--output-dir", "-o", required=True, help="Output directory")
    parser.add_argument("--dataset", default="entry/data/data", help="HDF dataset path")
    parser.add_argument(
        "--min-x",
        type=int,
        default=0,
        help="Crop the resulting slice images to have a minimum x value. Only used if slice axis is X.",
    )
    parser.add_argument(
        "--max-x",
        type=int,
        default=None,
        help="Crop the resulting slice images to have a maximum x value. Only used if slice axis is X.",
    )
    parser.add_argument(
        "--min-y",
        type=int,
        default=0,
        help="Crop the resulting slice images to have a minimum y value. Only used if slice axis is X.",
    )
    parser.add_argument(
        "--max-y",
        type=int,
        default=None,
        help="Crop the resulting slice images to have a maximum y value. Only used if slice axis is X.",
    )
    parser.add_argument(
        "--output-type",
        "-t",
        choices=["tiff", "hdf"],
        default="tiff",
        help="output file type, default to tiff",
    )
    args = parser.parse_args()

    time_start = timeit.default_timer()

    for index, file in enumerate(args.input_files):
        if not file.endswith(".hdf"):
            raise ValueError(f"Error, file {file} is not of valid extension .hdf")

        input_file = h5py.File(file, "r")
        in_data = input_file[args.dataset]
        assert isinstance(
            in_data, h5py.Dataset
        ), "Error, data at this path is not of type Dataset"
        (depth, height, width) = in_data.shape

        if args.output_type == "tiff":
            for i in tqdm(range(depth), desc="Extract TIFFs"):
                mat1 = in_data[i, args.min_y : args.max_y, args.min_x : args.max_x]
                im = Image.fromarray(mat1)
                output_path = os.path.join(
                    args.output_dir, os.path.basename(file).rstrip(".hdf")
                )
                os.makedirs(output_path, exist_ok=True)
                filename = os.path.join(
                    args.output_dir,
                    output_path,
                    "slice_" + str(i).zfill(len(str(depth))) + ".tiff",
                )
                im.save(filename)

        else:
            if args.max_y is None:
                y_size = height - args.min_y
            else:
                y_size = args.max_y - args.min_y

            if args.max_x is None:
                x_size = width - args.min_x
            else:
                x_size = args.max_x - args.min_x

            output_path = file.replace(".hdf", "_cropped.hdf")
            with h5py.File(os.path.join(args.output_dir, output_path), "w") as out:
                dset = out.create_dataset(
                    "cropped_slab", (depth, y_size, x_size), dtype=in_data.dtype
                )
                for i in range(depth):
                    dset[i] = in_data[i, args.min_y : args.max_y, args.max_x]

        input_file.close()

    time_stop = timeit.default_timer()
    print(f"Time to run {timeit.default_timer() - time_start}", flush=True)
