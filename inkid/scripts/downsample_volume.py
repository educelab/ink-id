import argparse
from glob import glob
import math

import inkid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('scale', type=int)
    parser.add_argument('output')
    args = parser.parse_args()

    if not math.log2(args.scale).is_integer():
        print(f'error: Scale factor {args.scale} is not a power of 2.')
        return

    # Remove slices to downsample on z axis
    image_filenames = glob(f'{args.input}/*.tif')
    print(image_filenames)

    # Use imagemagick to downsample remaining images in xy



if __name__ == '__main__':
    main()