import argparse
import json
import math
from pathlib import Path

import imageio.v3 as iio
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import pyrender
from scipy.spatial.transform import Rotation
import trimesh


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

    metadata_path = slices_dir / "meta.json"
    with metadata_path.open() as f:
        metadata = json.load(f)
        voxelsize_microns = metadata["voxelsize"]

    return vol, voxelsize_microns


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


def get_slice(
    vol, vol_point, rotation_angles, voxelsize_microns, radius_microns, resolution
):
    radius_voxels_if_full_resolution = radius_microns // voxelsize_microns
    radius_pixels = int(radius_voxels_if_full_resolution * resolution)
    slice_img_shape = (radius_pixels * 2 + 1, radius_pixels * 2 + 1)
    # List of all the pixel indices of the slice image we want to generate
    points = np.array(list(np.ndindex(*slice_img_shape)))
    # Add a third value to each pixel index making it 3D
    points = np.hstack((points, np.zeros((points.shape[0], 1))))
    # Translate to make the origin the center of the slice image
    points -= np.array([radius_pixels, radius_pixels, 0])
    # Rotate (have to add a dimension and then remove it to get the vectorized matmul to cooperate)
    r = Rotation.from_euler("xyz", rotation_angles, degrees=False).as_matrix()
    points = np.expand_dims(points, axis=2)
    points = np.matmul(r, points)
    points = np.squeeze(points)
    # Scale
    points /= resolution
    # Translate
    points += vol_point
    # Make image
    points = points.astype(int)
    xs = np.clip(points[:, 0], 0, vol.shape[2] - 1)
    ys = np.clip(points[:, 1], 0, vol.shape[1] - 1)
    zs = np.clip(points[:, 2], 0, vol.shape[0] - 1)
    slice_img = vol[zs, ys, xs].reshape(slice_img_shape).transpose()
    return slice_img


def determine_orientation(vol, voxelsize_um, point):
    # TODO draw intersection on orthogonal views
    fig, axes = plt.subplots(3, 3)
    plt.get_current_fig_manager().set_window_title("Papyrus Fiber Explorer 3000")
    ax_xy = axes[0, 0]
    ax_yz = axes[0, 1]
    ax_resolution_slider = axes[0, 2]
    ax_xz = axes[1, 0]
    ax_slice = axes[1, 1]
    ax_radius_slider = axes[1, 2]
    ax_alpha_slider = axes[2, 0]
    ax_beta_slider = axes[2, 1]
    ax_gamma_slider = axes[2, 2]

    point = np.array(point)
    x, y, z = point
    ax_xy.imshow(vol[z, :, :])
    ax_yz.imshow(vol[:, :, x])
    ax_xz.imshow(vol[:, y, :])

    # Get the slice image
    radius_um = 800
    resolution = 0.25
    rotation_angles = [0, 0, 0]
    slice_img = get_slice(
        vol, point, rotation_angles, voxelsize_um, radius_um, resolution
    )
    ax_slice.imshow(slice_img)

    radius_slider = Slider(
        ax=ax_radius_slider,
        label="Radius [um]",
        valmin=100,
        valmax=4000,
        valinit=radius_um,
        valstep=10,
    )
    resolution_slider = Slider(
        ax=ax_resolution_slider,
        label="Resolution",
        valmin=0.1,
        valmax=1,
        valinit=resolution,
    )
    alpha_slider = Slider(
        ax=ax_alpha_slider,
        label="Alpha [rad]",
        valmin=0,
        valmax=2 * math.pi,
        valinit=0,
    )
    beta_slider = Slider(
        ax=ax_beta_slider,
        label="Beta [rad]",
        valmin=0,
        valmax=2 * math.pi,
        valinit=0,
    )
    gamma_slider = Slider(
        ax=ax_gamma_slider,
        label="Gamma [rad]",
        valmin=0,
        valmax=2 * math.pi,
        valinit=0,
    )

    def update_radius(val):
        nonlocal radius_um
        radius_um = val
        new_slice_img = get_slice(
            vol, point, rotation_angles, voxelsize_um, radius_um, resolution
        )
        ax_slice.imshow(new_slice_img)
        fig.canvas.draw_idle()

    def update_resolution(val):
        nonlocal resolution
        resolution = val
        new_slice_img = get_slice(
            vol, point, rotation_angles, voxelsize_um, radius_um, resolution
        )
        ax_slice.imshow(new_slice_img)
        fig.canvas.draw_idle()

    def update_alpha(val):
        nonlocal rotation_angles
        rotation_angles[0] = val
        new_slice_img = get_slice(
            vol, point, rotation_angles, voxelsize_um, radius_um, resolution
        )
        ax_slice.imshow(new_slice_img)
        fig.canvas.draw_idle()

    def update_beta(val):
        nonlocal rotation_angles
        rotation_angles[1] = val
        new_slice_img = get_slice(
            vol, point, rotation_angles, voxelsize_um, radius_um, resolution
        )
        ax_slice.imshow(new_slice_img)
        fig.canvas.draw_idle()

    def update_gamma(val):
        nonlocal rotation_angles
        rotation_angles[2] = val
        new_slice_img = get_slice(
            vol, point, rotation_angles, voxelsize_um, radius_um, resolution
        )
        ax_slice.imshow(new_slice_img)
        fig.canvas.draw_idle()

    radius_slider.on_changed(update_radius)
    resolution_slider.on_changed(update_resolution)
    alpha_slider.on_changed(update_alpha)
    beta_slider.on_changed(update_beta)
    gamma_slider.on_changed(update_gamma)

    plt.show()

    return rotation_angles, radius_um


def get_next_points(vol, voxelsize_um, current_point, rotation_angles, radius_um):
    points = np.array(
        [
            [1, 0, 0],
            [0, -1, 0],
            [-1, 0, 0],
            [0, 1, 0],
        ]
    )
    # Rotate (have to add a dimension and then remove it to get the vectorized matmul to cooperate)
    r = Rotation.from_euler("xyz", rotation_angles, degrees=False).as_matrix()
    points = np.expand_dims(points, axis=2)
    points = np.matmul(r, points)
    points = np.squeeze(points)
    # Scale
    points *= radius_um
    points /= voxelsize_um
    # Translate
    points += current_point
    points = points.astype(int)
    # Remove if out of bounds
    in_bounds_points = []
    vz, vy, vx = vol.shape
    for point in points:
        x, y, z = point
        if 0 <= x < vx and 0 <= y < vy and 0 <= z < vz:
            in_bounds_points.append(list(point))
    return in_bounds_points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-volume", required=True)
    args = parser.parse_args()

    vol, voxelsize_um = load_volume(args.input_volume)

    mesh_vertices = []
    mesh_faces = []

    remaining_points_queue = select_seed_points()
    while remaining_points_queue:
        current_point = remaining_points_queue.pop(0)
        rotation_angles, radius_um = determine_orientation(
            vol, voxelsize_um, current_point
        )
        next_points = get_next_points(
            vol, voxelsize_um, current_point, rotation_angles, radius_um
        )
        remaining_points_queue += next_points

        # TODO This could work for some cases even if some next_points were out of bounds
        if len(next_points) == 4:
            mesh_vertices.append(current_point)
            mesh_vertices += next_points
            triangles = np.array(
                [
                    [0, 4, 1],
                    [0, 1, 2],
                    [0, 2, 3],
                    [0, 3, 4],
                ],
                dtype=int,
            )
            triangles += len(mesh_vertices) - 5
            mesh_faces += list(triangles)

        tri_mesh = trimesh.Trimesh(vertices=mesh_vertices, faces=mesh_faces)
        mesh = pyrender.Mesh.from_trimesh(tri_mesh)
        scene = pyrender.Scene()
        scene.add(mesh)
        viewer = pyrender.Viewer(scene, render_flags={"cull_faces": False, "all_wireframe": True})


if __name__ == "__main__":
    main()
