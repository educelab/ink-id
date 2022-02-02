import argparse
import os

import numpy as np
from PIL import Image

import inkid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ppm")
    parser.add_argument("buffer", type=int)
    args = parser.parse_args()

    root = os.path.splitext(args.ppm)[0]
    mask_file = root + "_mask.png"
    mask = Image.open(mask_file).convert("1")
    mask_arr = np.array(mask).flatten()
    ppm = inkid.data.PPM(args.ppm)

    vol_xs = ppm._data[:, :, 0].flatten()[mask_arr]
    vol_ys = ppm._data[:, :, 1].flatten()[mask_arr]
    vol_zs = ppm._data[:, :, 2].flatten()[mask_arr]

    b = args.buffer
    xmin, xmax = int(np.amin(vol_xs)) - b, int(np.amax(vol_xs)) + b
    ymin, ymax = int(np.amin(vol_ys)) - b, int(np.amax(vol_ys)) + b
    zmin, zmax = int(np.amin(vol_zs)) - b, int(np.amax(vol_zs)) + b

    crop_command = "find <slice_dir>/ -type f -name '*.tif'"
    crop_command += (
        f" | parallel --bar convert -crop '{xmax - xmin}x{ymax - ymin}+{xmin}+{ymin}'"
    )
    crop_command += " {} <cropped_dir>/{/}"

    print("Crop:")
    print(f"{xmax - xmin}x{ymax - ymin}x{zmax - zmin}+{xmin}+{ymin}+{zmin}")

    print("\nCommands to remove slices:")
    print("rm -f <slices_dir>/{0000.." + str(zmin - 1) + "}.tif")
    print("rm -f <slices_dir>/{" + str(zmax) + "..<max>}.tif")

    print("\nCommand to crop slices:")
    print(crop_command)


if __name__ == "__main__":
    main()
