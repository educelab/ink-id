import argparse
import math
from pathlib import Path

import imageio.v3 as iio
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
from scipy.spatial.transform import Rotation


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


def get_slice(vol, vol_point, rotation_angles, radius):
    slice_img = np.zeros((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8)
    for slice_y, slice_x in np.ndindex(*slice_img.shape):
        new_point = np.array([slice_x, slice_y, 0])
        # Translate to make the origin the center of the slice image
        new_point -= np.array([radius, radius, 0])
        # Check if new point is within slice image circle and continue if not
        if math.sqrt(new_point[0]**2 + new_point[1]**2) > radius:
            continue
        # Rotate
        r = Rotation.from_euler("xyz", rotation_angles, degrees=False).as_matrix()
        new_point = np.matmul(r, new_point)
        # Translate
        new_point += vol_point
        # Assign the slice pixel
        new_point = new_point.astype(int)
        x, y, z = new_point
        if (
            0 <= z < vol.shape[0]
            and 0 <= y < vol.shape[1]
            and 0 <= x < vol.shape[2]
        ):
            slice_img[slice_y, slice_x] = vol[z, y, x]
    return slice_img


def determine_orientation(vol, point):
    point = np.array(point)
    fig, ((ax_xy, ax_yz, ax_radius_slider), (ax_xz, ax_slice, _)) = plt.subplots(2, 3)
    x, y, z = point
    ax_xy.imshow(vol[z, :, :])
    ax_yz.imshow(vol[:, :, x])
    ax_xz.imshow(vol[:, y, :])

    # Get the slice image
    radius = 50
    rotation_angles = [0, 0, 0]
    slice_img = get_slice(vol, point, rotation_angles, radius)
    ax_slice.imshow(slice_img)

    radius_slider = Slider(
        ax=ax_radius_slider,
        label="Radius [voxels]",
        valmin=8,
        valmax=256,
        valinit=radius,
        valstep=1,
    )

    def update(val):
        new_slice_img = get_slice(vol, point, rotation_angles, val)
        ax_slice.imshow(new_slice_img)
        fig.canvas.draw_idle()

    radius_slider.on_changed(update)

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
