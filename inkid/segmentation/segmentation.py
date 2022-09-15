import argparse
import math
from pathlib import Path

import imageio.v3 as iio
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
import numpy as np


def load_volume(slices_dir):
    slices_dir = Path(slices_dir)
    slices = list(sorted(slices_dir.glob("*.tif")))
    img = iio.imread(slices[0])
    z_size = len(slices)
    y_size = img.shape[0]
    x_size = img.shape[1]
    vol = np.zeros((z_size, y_size, x_size), dtype=np.uint8)
    for z, slice_path in enumerate(slices):
        # Divide to [0, 255] values by bit shifting (same as /= 256 but faster), then convert to 8-bit
        vol[z] = (iio.imread(slice_path) >> 8).astype(np.uint8)
    return vol


def display_volume(vol, initial_slice=0):
    fig, ax = plt.subplots()
    ax.imshow(vol[initial_slice])
    # Adjust the main plot to make room for the slider
    plt.subplots_adjust(bottom=0.25)
    # Make a horizontal slider to control the slice
    ax_slice = plt.axes([0.25, 0.1, 0.65, 0.03])
    slice_slider = Slider(
        ax=ax_slice,
        label="Slice [idx]",
        valmin=0,
        valmax=vol.shape[0] - 1,
        valinit=0,
        valstep=1,
    )

    # The function to be called anytime the slider's value changes
    def update(val):
        ax.imshow(vol[int(val)])
        fig.canvas.draw_idle()

    slice_slider.on_changed(update)
    plt.show()


def select_seed_points():
    return [(256, 145, 428)]


def determine_orientation(vol, point):
    point = np.array(point)
    fig, ((ax_xy, ax_yz), (ax_xz, ax_slice)) = plt.subplots(2, 2)
    x, y, z = point
    ax_xy.imshow(vol[z, :, :])
    ax_yz.imshow(vol[:, :, x])
    ax_xz.imshow(vol[:, y, :])

    # Get the slice image
    r = 50
    slice_img = np.zeros((r * 2 + 1, r * 2 + 1), dtype=np.uint8)
    basis_x = np.array([1, 0, 0], dtype=float)
    basis_y = np.array([0, 1, 0], dtype=float)

    for slice_y in range(slice_img.shape[1]):
        for slice_x in range(slice_img.shape[0]):
            slice_x_about_point = slice_x - r
            slice_y_about_point = slice_y - r
            if math.sqrt(slice_x_about_point**2 + slice_y_about_point**2) > r:
                continue
            new_x, new_y, new_z = (point + slice_x_about_point * basis_x + slice_y_about_point * basis_y).astype(int)
            if 0 <= new_z < vol.shape[0] and 0 <= new_y < vol.shape[1] and 0 <= new_x < vol.shape[2]:
                slice_img[slice_y, slice_x] = vol[new_z, new_y, new_x]

    ax_slice.imshow(slice_img)

    plt.show()
    return [math.pi, math.pi]


def get_next_points(current_point, orientation):
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-volume", required=True)
    args = parser.parse_args()

    vol = load_volume(args.input_volume)

    points_queue = select_seed_points()
    while points_queue:
        current_point = points_queue.pop(0)
        orientation = determine_orientation(vol, current_point)
    #     next_points = get_next_points(current_point, orientation)
    #     points_queue += next_points


if __name__ == "__main__":
    main()
