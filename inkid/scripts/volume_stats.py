import argparse
from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
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
    blurred_mins = []
    blurred_maxs = []
    blurred_means = []

    for slice_path in tqdm(slice_paths):
        slice_indices.append(int(slice_path.stem))

        img = iio.imread(slice_path)
        mins.append(np.amin(img))
        maxs.append(np.amax(img))
        means.append(np.mean(img))

        filtered_img = gaussian_filter(img, 3)
        blurred_mins.append(np.amin(filtered_img))
        blurred_maxs.append(np.amax(filtered_img))
        blurred_means.append(np.mean(filtered_img))

    print(f"Global raw min: {np.amin(mins)}")
    print(f"Global raw max: {np.amax(maxs)}")
    print(f"Global raw mean: {np.mean(means)}")
    print(f"Global blurred min: {np.amin(blurred_mins)}")
    print(f"Global blurred max: {np.amax(blurred_maxs)}")
    print(f"Global blurred mean: {np.mean(blurred_means)}")

    fig, ax = plt.subplots()
    ax.plot(slice_indices, mins, label="min", color="blue")
    ax.plot(slice_indices, maxs, label="max", color="red")
    ax.plot(slice_indices, means, label="mean", color="green")
    ax.plot(slice_indices, blurred_mins, label="blurred min", color="lightskyblue")
    ax.plot(slice_indices, blurred_maxs, label="blurred max", color="lightcoral")
    ax.plot(slice_indices, blurred_means, label="blurred mean", color="darkseagreen")

    ax.legend()
    ax.set_title(f"Stats for volume: {input_slices_dir.stem}")

    plt.show()


if __name__ == "__main__":
    main()
