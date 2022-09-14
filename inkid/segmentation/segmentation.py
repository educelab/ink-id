import argparse
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image


def load_volume(slices_dir):
    slices_dir = Path(slices_dir)
    slices = list(sorted(slices_dir.glob("*.tif")))
    slice0 = Image.open(slices[0])
    width, height = slice0.width, slice0.height
    vol = np.zeros((len(slices), height, width), dtype=np.uint16)
    for z, slice_path in enumerate(slices):
        vol[z] = np.array(Image.open(slice_path))
    return vol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-volume", required=True)
    args = parser.parse_args()

    vol = load_volume(args.input_volume)
    slice0 = vol[100]
    plt.ion()
    plt.imshow(slice0)
    input("Press any key to close...")


if __name__ == "__main__":
    main()
