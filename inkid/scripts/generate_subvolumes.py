import argparse
import math
import numpy as np
import os

import io
import matplotlib.pyplot as plt
from PIL import Image
import plotly.graph_objects as go
import torch

import inkid


def render_slices(subvol, direction, color="jet", imgs_in_row=6):
    """for 2D slices along axes"""
    size_x, size_y, size_z = np.shape(subvol)

    # Make the range consistent for all slices
    max_v = np.amax(subvol)
    min_v = np.amin(subvol)

    if direction == "x":
        img_count, img_width, img_height = size_x, size_y, size_z
    elif direction == "y":
        img_count, img_width, img_height = size_y, size_z, size_x
    elif direction == "z":
        img_count, img_width, img_height = size_z, size_y, size_x
    else:
        raise ValueError(f"Unknown direction {direction}")

    row_count = math.ceil(img_count / imgs_in_row)
    # divisors below are just for a friendly size
    figsize = (imgs_in_row * img_width / 15, row_count * img_height / 12)

    # set up the graph
    f, ax_arr = plt.subplots(row_count, imgs_in_row, figsize=figsize)

    for j, row in enumerate(ax_arr):
        for i, ax in enumerate(row):
            if j * imgs_in_row + i < img_count:
                if direction == "x":
                    ax.imshow(
                        subvol[j * imgs_in_row + i, :, :],
                        cmap=color,
                        vmax=max_v,
                        vmin=min_v,
                    )
                    ax.set_title(f"x-slice {j*imgs_in_row+i}")
                elif direction == "y":
                    ax.imshow(
                        subvol[:, j * imgs_in_row + i, :],
                        cmap=color,
                        vmax=max_v,
                        vmin=min_v,
                    )
                    ax.set_title(f"y-slice {j*imgs_in_row+i}")
                else:  # z
                    ax.imshow(
                        subvol[:, :, j * imgs_in_row + i],
                        cmap=color,
                        vmax=max_v,
                        vmin=min_v,
                    )
                    ax.set_title(f"z-slice {j*imgs_in_row+i}")

    title = f"{direction}-slices"
    f.suptitle(title, fontsize=12)

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    return buf


def render_3d_volume_plotly(subvol, direction, color="jet"):
    size_x, size_y, size_z = np.shape(subvol)

    x, y, z = np.mgrid[0:size_x, 0:size_y, 0:size_z]

    vol = go.Volume(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=subvol.flatten(),
        opacity=0.3,
        opacityscale=0.3,
        surface_count=10,
        colorscale=color,
        isomax=40000,
        # isomin=20000,
    )
    fig = go.Figure(data=vol)

    def generate_ticks(axis, interval, size_um=None):
        vals = []
        ticks = []

        if not size_um:
            size_um = 1
        for i in range(0, axis, interval):
            vals.append(i)
            ticks.append(i * size_um)

        return vals, [str(tick) for tick in ticks]

    x_vals, x_ticks = generate_ticks(size_x, 8)
    y_vals, y_ticks = generate_ticks(size_y, 8)
    z_vals, z_ticks = generate_ticks(size_z, 8)

    if direction == "x":
        up_direction = dict(x=1, y=0, z=0)
        eye = dict(x=1.7, y=1.7, z=1.7)
    elif direction == "y":
        up_direction = dict(x=0, y=1, z=0)
        eye = dict(x=1.7, y=1.7, z=1.7)
    else:  # direction == 'z'
        up_direction = dict(x=0, y=0, z=1)
        eye = dict(x=-1.7, y=1.7, z=1.7)

    fig.update_layout(
        scene=dict(
            xaxis=dict(ticktext=x_ticks, tickvals=x_vals),
            yaxis=dict(ticktext=y_ticks, tickvals=y_vals),
            zaxis=dict(ticktext=z_ticks, tickvals=z_vals),
        ),
        scene_aspectmode="data",
        scene_camera=dict(up=up_direction, eye=eye),
    )

    plotly_bytes = fig.to_image(format="png")

    return io.BytesIO(plotly_bytes)


def visualize(subvol):
    # Render 2D Images
    images = []
    for direction in ["x", "y", "z"]:
        slice_img = render_slices(subvol, direction)
        images.append(slice_img)
    # Render 3D Images
    for direction in ["x", "y", "z"]:
        plotly_img = render_3d_volume_plotly(subvol, direction)
        images.append(plotly_img)

    # Convert the byte arrays to viewable images and concatenate
    graphs = [Image.open(img) for img in images]

    # Calculate the final rendering image file size
    img_size = [0, 0]  # (width, height)
    for graph in graphs[0:4]:
        if graph.width > img_size[0]:
            img_size[0] = graph.width
        img_size[1] = img_size[1] + graph.height

    summary_img = Image.new("RGB", (img_size[0], img_size[1]))

    #  Stitch all graphs together
    current_pos = [0, 0]
    # Matplotlib (2D) images
    for graph in graphs[0:3]:
        summary_img.paste(graph, current_pos)
        # bring the current position down by the pasted image's height
        current_pos = (0, current_pos[1] + graph.height)
    # Plotly (3D) images
    for graph in graphs[3:6]:
        summary_img.paste(graph, current_pos)
        # shift it to the side
        current_pos = (current_pos[0] + graph.width, current_pos[1])

    for img in images:
        img.close()

    return summary_img


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-set", metavar="path", nargs="*", help="input dataset(s)", default=[]
    )
    parser.add_argument(
        "--output", help="directory to hold output subvolumes", required=True
    )
    parser.add_argument(
        "--number",
        "-n",
        metavar="N",
        default=4,
        type=int,
        help="number of subvolumes to keep",
    )
    parser.add_argument(
        "--ink", action="store_true", help="restrict to points on ink areas"
    )
    parser.add_argument(
        "--no-ink", action="store_true", help="restrict to points not on ink areas"
    )
    parser.add_argument(
        "--concat-subvolumes",
        action="store_true",
        help="create one set of slices containing all subvolumes",
    )
    parser.add_argument(
        "--random-seed", type=int, default=42, help="seed for random number generators"
    )
    inkid.data.add_subvolume_arguments(parser)

    # Image rendering option
    parser.add_argument(
        "--visualize", action="store_true", help="generate and save 2D/3D renderings"
    )

    args = parser.parse_args(argv)

    # Make sure some sort of input is provided, else there is nothing to do
    if len(args.input_set) == 0:
        raise ValueError("Some --input-set must be specified.")

    os.makedirs(args.output, exist_ok=True)

    subvolume_args = dict(
        shape_voxels=args.subvolume_shape_voxels,
        shape_microns=args.subvolume_shape_microns,
        move_along_normal=args.move_along_normal,
        method=args.subvolume_method,
        normalize=args.normalize_subvolumes,
        augment_subvolume=args.augmentation,
        jitter_max=args.jitter_max,
    )

    input_ds = inkid.data.Dataset(args.input_set)
    specify_inkness = None
    if args.ink:
        specify_inkness = True
    elif args.no_ink:
        specify_inkness = False
    sampler = inkid.data.RegionPointSampler(specify_inkness=specify_inkness)

    for region in input_ds.regions():
        region.feature_args = subvolume_args
        region.sampler = sampler

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    input_dl = None
    if len(input_ds) > 0:
        input_dl = torch.utils.data.DataLoader(input_ds, shuffle=True)

    square_side_length = math.ceil(math.sqrt(args.number))
    pad = 20
    padded_shape_voxels = [i + pad * 2 for i in args.subvolume_shape_voxels]
    concatenated_shape = [
        padded_shape_voxels[0],
        padded_shape_voxels[1] * square_side_length,
        padded_shape_voxels[2] * square_side_length,
    ]
    concatenated_subvolumes = np.zeros(concatenated_shape)

    counter = 0
    for batch in input_dl:
        if counter >= args.number:
            break
        subvolume = batch["feature"]
        subvolume = subvolume.numpy()[0][0]

        if args.concat_subvolumes:
            concat_x = (counter // square_side_length) * padded_shape_voxels[2]
            concat_y = (counter % square_side_length) * padded_shape_voxels[1]
            subvolume = np.pad(subvolume, pad)
            concatenated_subvolumes[
                0 : padded_shape_voxels[0],
                concat_y : concat_y + padded_shape_voxels[1],
                concat_x : concat_x + padded_shape_voxels[2],
            ] = subvolume
        else:
            inkid.util.save_volume_to_image_stack(
                subvolume, os.path.join(args.output, str(counter))
            )

            if args.visualize:
                rendered_img = visualize(subvolume)
                rendered_img.save(os.path.join(args.output, f"{str(counter)}.png"))

        counter += 1

    if args.concat_subvolumes:
        inkid.util.save_volume_to_image_stack(concatenated_subvolumes, args.output)


if __name__ == "__main__":
    main()
