import argparse
from itertools import product
from multiprocessing import Pool
import os
from pathlib import Path

import imageio
from scipy.stats import pearsonr
from scipy.ndimage import median_filter
import numpy as np


def mutual_information(im_1, im_2, num_bins=20):
    hist_2d, x_edges, y_edges = np.histogram2d(
        im_1.ravel(), im_2.ravel(), bins=num_bins
    )
    # Convert bins counts to probability values

    pxy = hist_2d / float(np.sum(hist_2d))

    px = np.sum(pxy, axis=1)  # marginal for x over y
    py = np.sum(pxy, axis=0)  # marginal for y over x

    px_py = px[:, None] * py[None, :]  # Broadcast to multiply marginals

    nzs = pxy > 0  # Only non-zero pxy values contribute to the sum

    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))


def pearson_correlation(x, y, filter_sigma=3):
    x_filtered = median_filter(x, filter_sigma)
    y_filtered = median_filter(y, filter_sigma)
    z = pearsonr(x_filtered.flatten(), y_filtered.flatten())

    return z


def image_comparison_worker(moving_paths, fixed_array, comp_function):
    x_im = np.hstack([imageio.imread(moving_path) for moving_path in moving_paths])
    if comp_function == "pearson":
        return pearson_correlation(x_im, fixed_array), moving_paths
    else:
        return mutual_information(x_im, fixed_array), moving_paths


if __name__ == "__main__":
    """This script finds the slice indices that are most likely to be the starting
    point of overlap for a pair of overlapping slabs"""

    parser = argparse.ArgumentParser(
        description="Find candidate overlap indices for pairs of slabs"
    )
    parser.add_argument(
        "--fixed-slab",
        "-f",
        required=True,
        help="Slices directory for fixed slab",
    )
    parser.add_argument(
        "--moving-slab", "-m", required=True, help="Slices directory for moving slab"
    )
    parser.add_argument(
        "--num-candidates",
        "-n",
        type=int,
        default=5,
        help="Number of candidate slices to list",
    )
    # parser.add_argument(
    #     "--validation_dir", "-v", default=".", help="directory for validation images"
    # )
    parser.add_argument(
        "--min-index", type=int, default=0, help="minimum slab index to check against"
    )
    parser.add_argument(
        "--max-index",
        type=int,
        default=None,
        help="maximum slab index to check against",
    )
    parser.add_argument(
        "--num-references",
        "-r",
        type=int,
        default=1,
        help="Number of slices used in comparison",
    )

    parser.add_argument(
        "--comparison_function",
        "-c",
        default="mutual_information",
        choices=["mutual_information", "pearson"],
    )

    args = parser.parse_args()

    print(
        f"Finding overlap between {Path(args.moving_slab).stem} and {Path(args.fixed_slab).stem}",
        flush=True,
    )

    fixed_slab_slices = sorted(
        [
            os.path.join(args.fixed_slab, slice_filename)
            for slice_filename in os.listdir(args.fixed_slab)
        ]
    )
    fixed_slab_slices = list(filter(lambda x: ".tif" in x, fixed_slab_slices))
    fixed_slab_slices = fixed_slab_slices[: args.num_references]
    moving_slab_slices = sorted(
        [
            os.path.join(args.moving_slab, slice_filename)
            for slice_filename in os.listdir(args.moving_slab)
        ]
    )
    moving_slab_slices = list(filter(lambda x: ".tif" in x, moving_slab_slices))
    moving_slab_slices = moving_slab_slices[args.min_index : args.max_index]

    reference_image = np.hstack(
        [imageio.imread(fixed_slab_slice) for fixed_slab_slice in fixed_slab_slices]
    )

    moving_slab_pairs = [
        moving_slab_slices[i : i + args.num_references]
        for i in range(len(moving_slab_slices) - args.num_references)
    ]

    process_parameters = product(
        moving_slab_pairs, [reference_image], [args.comparison_function]
    )

    with Pool() as worker_pool:
        correlations = worker_pool.starmap(image_comparison_worker, process_parameters)

    correlations = sorted(correlations)
    reference_stack = np.vstack([imageio.imread(path) for path in fixed_slab_slices])
    reference_min_slice_name = os.path.basename(fixed_slab_slices[-1]).split(".")[0]

    for i, (cor, paths) in enumerate(correlations):
        if i >= args.num_candidates:
            break
        else:
            print(
                f"Slice sequence {[os.path.basename(path) for path in paths]} has correlation {cor} "
                f"with slice sequence {[os.path.basename(path) for path in fixed_slab_slices]}",
                flush=True,
            )

            # moving_min_slice_name = os.path.basename(paths[-1]).split(".")[0]
            # moving_overlap = np.vstack([imageio.imread(path) for path in paths])
            #
            # image_comparison = np.hstack([moving_overlap, reference_stack])
            #
            # rescaled_image_comparison = np.clip(
            #     (
            #         (image_comparison - np.min(image_comparison))
            #         / (np.max(image_comparison) - np.min(image_comparison))
            #     )
            #     * (2**16),
            #     0,
            #     2**16,
            #     dtype=np.int16,
            # )
            #
            # output_path = os.path.join(
            #     os.path.abspath(args.validation_dir),
            #     f"moving_min_slice_{moving_min_slice_name}_ref_min_slice_{reference_min_slice_name}.tif",
            # )
            # imageio.imwrite(output_path, rescaled_image_comparison)
