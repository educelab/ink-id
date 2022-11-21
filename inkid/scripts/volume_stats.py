import argparse
from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-slices-dir", "-i", required=True, help="Input slice directory"
    )
    parser.add_argument(
        "--slice-skip",
        type=int,
        default=10,
        help="Number of slices to skip when calculating stats",
    )
    args = parser.parse_args()

    input_slices_dir = Path(args.input_slices_dir)
    slice_paths = sorted(input_slices_dir.glob("*.tif"))
    slice_paths = slice_paths[:: args.slice_skip]

    slice_indices = []
    mins = []
    maxs = []
    means = []

    for slice_path in tqdm(slice_paths):
        img = iio.imread(slice_path)
        slice_indices.append(int(slice_path.stem))
        mins.append(np.amin(img))
        maxs.append(np.amax(img))
        means.append(np.mean(img))

    fig, ax = plt.subplots()
    ax.plot(slice_indices, mins, label="min")
    ax.plot(slice_indices, maxs, label="max")
    ax.plot(slice_indices, means, label="mean")

    ax.legend()
    ax.set_title(f"Stats for volume: {input_slices_dir.stem}")

    plt.show()


if __name__ == "__main__":
    main()
