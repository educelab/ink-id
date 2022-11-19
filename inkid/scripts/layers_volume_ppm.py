import argparse

import imageio.v3 as iio
from tqdm import tqdm

from inkid.data import PPM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-tif-stack", required=True)
    parser.add_argument("--input-ppm", required=True)
    parser.add_argument("--output-ppm", required=True)
    args = parser.parse_args()

    tif_stack = iio.imread(args.input_tif_stack)

    print("Reading input PPM...")
    ppm = PPM.from_path(args.input_ppm)

    print("Modifying PPM...")
    z = (tif_stack.shape[0] - 1) / 2.0
    for y in tqdm(range(ppm.height)):
        for x in range(ppm.width):
            ppm.data[y, x, :] = [x, y, z, 0.0, 0.0, 1.0]

    print("Writing output PPM...")
    ppm.write(args.output_ppm)


if __name__ == "__main__":
    main()
