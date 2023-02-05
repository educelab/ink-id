import argparse
from pathlib import Path

import imageio.v3 as iio
from tqdm import tqdm
import numpy as np

from inkid.data import PPM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-tif-directory", required=True)
    parser.add_argument("--output-ppm", required=True)
    args = parser.parse_args()

    input_tif_dir = Path(args.input_tif_directory)
    num_slices = len(list(input_tif_dir.glob("*.tif")))

    # Get dimensions of sample tif
    sample_tif_path = input_tif_dir.glob("*.tif").__next__()
    sample_tif_img = iio.imread(sample_tif_path)
    height, width = sample_tif_img.shape

    print("Generating PPM data...")
    dim = 6
    data = np.zeros((height, width, dim), dtype=float)
    z = (num_slices - 1) / 2.0
    for y in tqdm(range(height)):
        for x in range(width):
            data[y, x, :] = [x, y, z, 0.0, 0.0, 1.0]

    print("Writing output PPM...")
    PPM.write_ppm_from_data(
        args.output_ppm,
        data,
        width,
        height,
        dim
    )


if __name__ == "__main__":
    main()
