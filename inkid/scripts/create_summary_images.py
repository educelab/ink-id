import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import re
from typing import List
import warnings

from humanize import naturalsize
import imageio
from matplotlib import colormaps
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import pygifsicle
import torch
import torch.nn as nn
from torchmetrics import Dice, StatScores
from tqdm import tqdm

import inkid


# Ignore PIL warnings about large images
Image.MAX_IMAGE_PIXELS = 10000000000

WHITE = (255, 255, 255)
LIGHT_GRAY = (104, 104, 104)
DARK_GRAY = (64, 64, 64)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
region_set_type_to_color = {"training": RED, "prediction": YELLOW, "validation": BLUE}


def get_preds_and_labels(pred_img, label_img, mask_img=None, bounding_box=None):
    # Convert images to the desired formats
    pred_img = pred_img.convert("L")
    pred_img = np.array(pred_img)

    label_img = label_img.convert("L")
    label_img = np.array(label_img)

    assert pred_img.shape == label_img.shape

    # Mask list of points to sample based on PPM mask
    if mask_img is not None:
        mask_img = mask_img.convert("L")
        mask_img = np.array(mask_img)
        assert mask_img.shape == pred_img.shape
        points = np.argwhere(mask_img)
    else:
        points = np.array(list(np.ndindex(*pred_img.shape)))

    # Mask list of points to sample based on validation region bounding box
    if bounding_box is not None:
        left, top, right, bottom = bounding_box
        top_left = np.array([top, left])
        bottom_right = np.array([bottom, right])
        # https://stackoverflow.com/a/33051576
        idxs_in_box = np.all(
            np.logical_and(top_left <= points, points <= bottom_right), axis=1
        )
        points = points[idxs_in_box]

    # Sample the chosen points from the prediction and label images
    # https://stackoverflow.com/a/16396203
    preds = pred_img[points[:, 0], points[:, 1]]
    labels = label_img[points[:, 0], points[:, 1]]

    # Convert prediction image pixel values to (not ink, ink) class scores
    preds = preds.astype(float)
    preds /= 255
    preds = np.vstack((1 - preds, preds)).transpose()

    # Convert label image pixel values to class indices (0: not ink, 1: ink)
    labels = np.where(labels > 0, 1, 0)

    return preds, labels


def compute_ink_classes_metrics(preds, labels):
    # Compute cross entropy loss
    preds = torch.tensor(preds)
    labels = torch.tensor(labels)
    loss = nn.CrossEntropyLoss()
    loss_val = float(loss(preds, labels))

    # Compute dice score
    preds = torch.argmax(preds, dim=1)
    dice = Dice()
    dice_val = float(dice(preds, labels))

    # Compute stat scores
    stats = StatScores(task="binary")
    stats_val = stats(preds, labels)
    tp, fp, tn, fn, support = list(stats_val)
    false_positive_rate = float(fp / (fp + tn))

    return {
        "crossEntropyLoss": loss_val,
        "dice": dice_val,
        "falsePositiveRate": false_positive_rate,
    }


def is_job_dir(dirname: str) -> bool:
    """
    Return whether this directory matches the expected structure for a job dir.

    For example, we often want a list of the k-fold subdirs in a directory but do not want e.g. previous summary
    subdirs, or others.
    """
    # k-fold job
    if is_k_fold_dir(dirname):
        return True
    # Standard job
    dirname = os.path.basename(dirname)
    if re.match(r".*\d\d\d\d-\d\d-\d\d_\d\d\.\d\d\.\d\d$", dirname):
        return True
    # Must be something else
    return False


def is_k_fold_dir(dirname: str) -> bool:
    """Return whether this directory is a k-fold job directory."""
    dirname = os.path.basename(dirname)
    return re.match(r".*\d\d\d\d-\d\d-\d\d_\d\d\.\d\d\.\d\d_(\d+)$", dirname)


def n_from_dir(dirname: str) -> int:
    """
    Return which k-fold job this directory corresponds to (if any, -1 otherwise).

    For the purpose of sorting a list of dirs based on job #.
    """
    # Only has a valid k-fold # if it matches this particular format
    if is_k_fold_dir(dirname):
        # Return the last number in the dirname, which is the k-fold #
        return int(re.findall(r"(\d+)", dirname)[-1])
    else:
        # Otherwise probably was standalone job (not part of k-fold)
        return -1


def iteration_str_sort_key(iteration: str) -> int:
    if iteration == "final":
        epoch, batch = 10e9, 10e9
    else:
        epoch, batch = int(iteration.split("_")[0]), int(iteration.split("_")[1])
    return epoch * 10e9 + batch


def iteration_series_sort_key(iterations_series):
    return np.array(list(map(iteration_str_sort_key, list(iterations_series))))


def label_key_from_prediction_type(prediction_type: str) -> str:
    if prediction_type == "rgb_values":
        label_key = "rgb_label"
    elif prediction_type == "ink_classes":
        label_key = "ink_label"
    elif prediction_type == "volcart_texture":
        label_key = "volcart_texture_label"
    else:
        raise ValueError(f"Unknown prediction type {prediction_type}")
    return label_key


class JobSummarizer:
    def __init__(self, directory):
        # Get list of directories (not files) in given parent dir
        possible_dirs = [
            os.path.join(directory, name) for name in os.listdir(directory)
        ]
        job_dirs = list(filter(is_job_dir, possible_dirs))
        job_dirs = sorted(job_dirs, key=n_from_dir)
        print("Found job directories:")
        for d in job_dirs:
            print(f"\t{d}")

        # We might issue this warning but only need to do it once
        self.already_warned_about_missing_label_images = False

        self.date = None
        self.name = Path(directory).name
        self.job_metadatas = dict()
        self.regions_df = pd.DataFrame(
            {
                "region_name": pd.Series(dtype="str"),
                "ppm_path": pd.Series(dtype="str"),
                "ppm_width": pd.Series(dtype="int"),
                "ppm_height": pd.Series(dtype="int"),
                "invert_normals": pd.Series(dtype="bool"),
                "bounding_box": pd.Series(dtype="int"),
            }
        )
        self.prediction_images_df = pd.DataFrame(
            {
                "path": pd.Series(dtype="str"),
                "region_name": pd.Series(dtype="str"),
                "iteration_str": pd.Series(dtype="str"),
                "prediction_type": pd.Series(dtype="str"),
                "job_dir": pd.Series(dtype="str"),
                "n_from_k_fold": pd.Series(dtype="int"),
                "training": pd.Series(dtype="bool"),
                "prediction": pd.Series(dtype="bool"),
                "validation": pd.Series(dtype="bool"),
            }
        )

        # Iterate through job dirs
        for job_dir in job_dirs:
            # Read the job metadata file (we use the region info)
            metadata_filename = os.path.join(job_dir, "metadata.json")
            n_from_k_fold = n_from_dir(job_dir)
            if not os.path.exists(metadata_filename):
                print(f"No job metadata file found in {job_dir}")
                return
            with open(metadata_filename) as f:
                metadata = json.loads(f.read())
                self.job_metadatas[job_dir] = metadata
                if self.date is None:
                    self.date = metadata["Date"]

            # Look in predictions directory to process prediction images
            pred_dir = os.path.join(job_dir, "predictions")
            if os.path.isdir(pred_dir):
                # Get all filenames in that directory
                names = os.listdir(pred_dir)
                # Filter out those that do not match the desired name format
                names = list(
                    filter(lambda name: re.match(".*_prediction_", name), names)
                )
                for name in names:
                    image_path = os.path.join(pred_dir, name)
                    # Get region name from image filename
                    region_name = re.search("(.*)_prediction_", name).group(1)
                    set_types, region_full_name, region_info = [], None, None
                    # Find which of the possible datasets this region is in
                    for set_type in ["training", "prediction", "validation"]:
                        regions = [
                            r
                            for r in metadata["Data"][set_type].keys()
                            if region_name in r
                        ]
                        if regions:
                            set_types.append(set_type)
                            # Also get its full name and its info from the metadata
                            region_full_name = regions[0]
                            region_info = metadata["Data"][set_type][region_full_name]
                    # Make sure it actually appeared somewhere
                    assert (
                        len(set_types) > 0
                        and region_full_name is not None
                        and region_info is not None
                    )
                    if region_name not in self.regions_df.region_name.values:
                        ppm_size = Image.open(image_path).size
                        bounding_box = region_info.get("bounding_box")
                        if bounding_box is not None:
                            bounding_box = tuple(bounding_box)
                        self.regions_df = pd.concat(
                            [
                                self.regions_df,
                                pd.DataFrame.from_dict(
                                    {
                                        "region_name": [region_name],
                                        "ppm_path": [region_info["ppm"]],
                                        "ppm_width": [ppm_size[0]],
                                        "ppm_height": [ppm_size[1]],
                                        "invert_normals": [
                                            region_info["invert_normals"]
                                        ],
                                        "bounding_box": [bounding_box],
                                    },
                                ),
                            ],
                            ignore_index=True,
                        )
                    if re.match(r".*_prediction_\d+_\d+", name):
                        iteration_str = re.search(
                            r".*_prediction_(\d+_\d+)_.*\.png", name
                        ).group(1)
                        prediction_type = re.search(
                            r".*_prediction_\d+_\d+_(.*)\.png", name
                        ).group(1)
                    elif re.match(".*_prediction_final", name):
                        iteration_str = "final"
                        prediction_type = re.search(
                            r".*_prediction_final_(.*)\.png", name
                        ).group(1)
                    else:
                        raise ValueError(
                            f"Image filename {name} does not match expected format"
                        )
                    self.prediction_images_df = pd.concat(
                        [
                            self.prediction_images_df,
                            pd.DataFrame.from_dict(
                                {
                                    "path": [image_path],
                                    "region_name": [region_name],
                                    "iteration_str": [iteration_str],
                                    "prediction_type": [prediction_type],
                                    "job_dir": [job_dir],
                                    "n_from_k_fold": [n_from_k_fold],
                                    "training": ["training" in set_types],
                                    "prediction": ["prediction" in set_types],
                                    "validation": ["validation" in set_types],
                                },
                            ),
                        ],
                        ignore_index=True,
                    )

    def any_prediction_images_found(self) -> bool:
        return len(self.prediction_images_df) > 0

    def prediction_types(self) -> List[str]:
        return self.prediction_images_df.prediction_type.unique()

    def iterations_encountered(self, prediction_type: str) -> List[str]:
        filtered_images_df = self.prediction_images_df[
            self.prediction_images_df["prediction_type"] == prediction_type
        ]
        unique_iterations = list(set(filtered_images_df.iteration_str.values))
        return sorted(unique_iterations, key=iteration_str_sort_key)

    def last_iteration_seen(self, prediction_type: str) -> str:
        return self.iterations_encountered(prediction_type)[-1]

    def job_dirs(self) -> List[str]:
        return sorted(self.prediction_images_df.job_dir.unique(), key=n_from_dir)

    def max_ppm_width(self):
        return self.prediction_images_df.merge(self.regions_df).ppm_width.max()

    def faces(self):
        """Where 'face' is a (ppm, invert_normals) unique pair"""
        merged = self.prediction_images_df.merge(self.regions_df)
        unique_faces = merged.groupby(
            ["ppm_path", "invert_normals"], as_index=False
        ).first()
        return unique_faces[["ppm_path", "invert_normals"]]

    def face_heights(self):
        region_per_face = self.regions_df.drop_duplicates(
            subset=["ppm_path", "invert_normals"]
        )
        return list(self.faces().merge(region_per_face, how="left").ppm_height)

    def faces_list(self):
        return list(self.faces().to_records(index=False))

    def get_face_prediction_image(
        self,
        job_dir,
        ppm_path,
        invert_normals,
        iteration,
        prediction_type,
        region_sets_to_include,
        region_sets_to_label,
        rectangle_line_width,
        cmap_name,
        return_latest_if_not_found=True,
    ):
        """
        Return a prediction image of the specified iteration, job directory, and PPM.

        If such an image does not exist return None.
        """
        df = self.prediction_images_df.merge(self.regions_df)
        # Filter to only those from this job, on this face and correct prediction type
        df = df.loc[
            (df["ppm_path"] == ppm_path)
            & (df["invert_normals"] == invert_normals)
            & (df["prediction_type"] == prediction_type)
        ]
        if job_dir is not None:
            df = df.loc[df["job_dir"] == job_dir]
        # Filter out those which aren't from the requested region types (training, prediction, validation)
        if region_sets_to_include is not None:
            filtered_dfs = []
            for region_set_type in region_sets_to_include:
                filtered_dfs.append(df[df[region_set_type]])
            df = pd.concat(filtered_dfs, ignore_index=True).drop_duplicates(
                ignore_index=True
            )
        if return_latest_if_not_found:
            # Sort by iteration
            df = df.sort_values("iteration_str", key=iteration_series_sort_key)
            # Filter out those past the iteration we are looking for
            df = df.loc[
                (
                    iteration_series_sort_key(df["iteration_str"])
                    <= iteration_str_sort_key(iteration)
                )
            ]
            # Get the last one from each region
            df = df.groupby(["region_name"], as_index=False).last()
        else:
            df = df.loc[(df["iteration_str"] == iteration)]
        image_paths = list(df.path)
        image_bounding_boxes = list(df.bounding_box)
        image_label_as = zip(
            list(df.training), list(df.prediction), list(df.validation)
        )
        return merge_imgs(
            image_paths,
            image_bounding_boxes,
            image_label_as,
            region_sets_to_label,
            rectangle_line_width,
            cmap_name,
        )

    def get_label_image_path(self, ppm_path, invert_normals, label_key):
        for metadata in self.job_metadatas.values():
            for dataset in metadata["Data"].values():
                for source in dataset.values():
                    if (
                        source.get("ppm") == ppm_path
                        and source.get("invert_normals") == invert_normals
                        and source.get(label_key) is not None
                    ):
                        return source.get(label_key)
        return None

    def get_mask_image_path(self, ppm_path, invert_normals):
        for metadata in self.job_metadatas.values():
            for dataset in metadata["Data"].values():
                for source in dataset.values():
                    if (
                        source.get("ppm") == ppm_path
                        and source.get("invert_normals") == invert_normals
                        and source.get("mask") is not None
                    ):
                        return source.get("mask")
        return None

    def any_label_images_found(self) -> bool:
        for prediction_type in self.prediction_types():
            for face_i, (ppm_path, invert_normals) in enumerate(self.faces_list()):
                label_key = label_key_from_prediction_type(prediction_type)
                label_img_path = self.get_label_image_path(
                    ppm_path, invert_normals, label_key
                )
                if label_img_path is not None:
                    # Try getting label image file from recorded location (may not exist on this machine)
                    label_img = try_get_img_from_data_files(label_img_path)
                    if label_img is not None:
                        return True
        return False

    def build_frame(
        self,
        iteration,
        prediction_type,
        max_size=None,
        region_sets_to_include=None,
        region_sets_to_label=None,
        cmap_name=None,
        superimpose_all_jobs=False,
        label_column=True,
    ):
        if region_sets_to_include is None:
            region_sets_to_include = ["training", "prediction", "validation"]
        if region_sets_to_label is None:
            region_sets_to_label = []

        job_dirs = self.job_dirs()
        col_width = self.max_ppm_width()
        row_heights = self.face_heights()
        buffer_size = int(col_width / 10)
        if superimpose_all_jobs:
            width = col_width + buffer_size * 2  # Only need space for one result column
        else:
            width = col_width * len(job_dirs) + buffer_size * (len(job_dirs) + 1)
        if label_column:
            width += col_width + buffer_size
        height = sum(row_heights) + buffer_size * (len(row_heights) + 2)
        # Prevent weird aspect ratios
        width_pad_offset = 0
        if width < height * 0.7:
            old_width = width
            width = int(height * 0.7)
            width_pad_offset = int((width - old_width) / 2)
        # Add space for footer
        footer_height = int(width / 16)
        rectangle_line_width = int(footer_height / 40)
        height = height + footer_height
        # Create empty frame
        frame = Image.new("RGB", (width, height))

        # Add prediction images
        # Make a row for each face
        for face_i, (ppm_path, invert_normals) in enumerate(self.faces_list()):
            ppm_mask_img_path = self.get_mask_image_path(ppm_path, invert_normals)
            ppm_mask_img = try_get_img_from_data_files(ppm_mask_img_path)
            imgs = []
            offsets = []
            if superimpose_all_jobs:
                offsets.append(
                    (
                        width_pad_offset + buffer_size,
                        sum(row_heights[:face_i]) + (face_i + 2) * buffer_size,
                    )
                )
                imgs.append(
                    self.get_face_prediction_image(
                        None,
                        ppm_path,
                        invert_normals,
                        iteration,
                        prediction_type,
                        region_sets_to_include,
                        region_sets_to_label,
                        rectangle_line_width,
                        cmap_name,
                    )
                )
            else:
                for job_i, job_dir in enumerate(job_dirs):
                    offsets.append(
                        (
                            width_pad_offset
                            + job_i * col_width
                            + (job_i + 1) * buffer_size,
                            sum(row_heights[:face_i]) + (face_i + 2) * buffer_size,
                        )
                    )
                    imgs.append(
                        self.get_face_prediction_image(
                            job_dir,
                            ppm_path,
                            invert_normals,
                            iteration,
                            prediction_type,
                            region_sets_to_include,
                            region_sets_to_label,
                            rectangle_line_width,
                            cmap_name,
                        )
                    )
            for img, offset in zip(imgs, offsets):
                if img is None:
                    continue
                # Only paste the parts of the image that actually contain something
                mask_img = img.convert("L")
                mask_img = mask_img.point(lambda x: x > 0, mode="1")
                # Also mask by the PPM mask
                if ppm_mask_img is not None:
                    ppm_mask_img = ppm_mask_img.convert("L")
                    ppm_mask_img = ppm_mask_img.point(lambda x: x > 0, mode="1")
                    mask_img = Image.composite(
                        mask_img, ppm_mask_img, mask=ppm_mask_img
                    )
                frame.paste(img, offset, mask=mask_img)

            # Add label column
            if label_column:
                label_key = label_key_from_prediction_type(prediction_type)
                label_img_path = self.get_label_image_path(
                    ppm_path, invert_normals, label_key
                )
                if label_img_path is not None:
                    # Try getting label image file from recorded location (may not exist on this machine)
                    label_img = try_get_img_from_data_files(label_img_path)
                    if label_img is not None:
                        if cmap_name is not None:
                            color_map = colormaps[cmap_name]
                            label_img = label_img.convert("L")
                            img_data = np.array(label_img)
                            label_img = Image.fromarray(
                                np.uint8(color_map(img_data) * 255)
                            )
                        if superimpose_all_jobs:
                            offset = (
                                width_pad_offset + col_width + buffer_size * 2,
                                sum(row_heights[:face_i]) + (face_i + 2) * buffer_size,
                            )
                        else:
                            offset = (
                                width_pad_offset
                                + len(job_dirs) * col_width
                                + (len(job_dirs) + 1) * buffer_size,
                                sum(row_heights[:face_i]) + (face_i + 2) * buffer_size,
                            )
                        if ppm_mask_img is not None:
                            ppm_mask_img = ppm_mask_img.convert("L")
                            ppm_mask_img = ppm_mask_img.point(lambda x: x > 0, mode="1")
                            frame.paste(label_img, offset, mask=ppm_mask_img)
                        else:
                            frame.paste(label_img, offset)
                    elif not self.already_warned_about_missing_label_images:
                        warnings.warn(
                            "At least one label image not found, check if dataset locally available",
                            RuntimeWarning,
                        )
                        self.already_warned_about_missing_label_images = True

        # Make column headers
        if superimpose_all_jobs:
            draw = ImageDraw.Draw(frame)
            font_path = os.path.join(
                os.path.dirname(inkid.__file__), "assets", "fonts", "Roboto-Regular.ttf"
            )
            fontsize = 1
            font_regular = ImageFont.truetype(font_path, fontsize)
            txt = f"Generated image"
            allowed_width = col_width
            while (
                int(font_regular.getlength(txt)) < allowed_width
                and int(font_regular.getbbox(txt)[3]) < buffer_size
            ):
                fontsize += 1
                font_regular = ImageFont.truetype(font_path, fontsize)
            fontsize -= 1
            offset_for_centering = int(
                (col_width - int(font_regular.getlength(txt))) / 2
            )
            offset = (
                width_pad_offset + buffer_size + offset_for_centering,
                int(buffer_size * 0.5),
            )
            draw.text(offset, txt, WHITE, font=font_regular)
        else:
            for job_i, job_dir in enumerate(job_dirs):
                draw = ImageDraw.Draw(frame)
                font_path = os.path.join(
                    os.path.dirname(inkid.__file__),
                    "assets",
                    "fonts",
                    "Roboto-Regular.ttf",
                )
                fontsize = 1
                font_regular = ImageFont.truetype(font_path, fontsize)
                txt = f"Job {job_i}"
                allowed_width = col_width
                while (
                    int(font_regular.getlength(txt)) < allowed_width
                    and int(font_regular.getbbox(txt)[3]) < buffer_size
                ):
                    fontsize += 1
                    font_regular = ImageFont.truetype(font_path, fontsize)
                fontsize -= 1
                offset_for_centering = int(
                    (col_width - font_regular.getlength(txt)) / 2
                )
                offset = (
                    width_pad_offset
                    + job_i * col_width
                    + (job_i + 1) * buffer_size
                    + offset_for_centering,
                    int(buffer_size * 0.5),
                )
                draw.text(offset, txt, WHITE, font=font_regular)
        if label_column:
            draw = ImageDraw.Draw(frame)
            font_path = os.path.join(
                os.path.dirname(inkid.__file__), "assets", "fonts", "Roboto-Regular.ttf"
            )
            fontsize = 1
            font_regular = ImageFont.truetype(font_path, fontsize)
            txt = "Label imaee"  # Hack because I don't want it to care about the part of the 'g' that sticks down
            allowed_width = col_width
            while (
                int(font_regular.getlength(txt)) < allowed_width
                and int(font_regular.getbbox(txt)[3]) < buffer_size
            ):
                fontsize += 1
                font_regular = ImageFont.truetype(font_path, fontsize)
            fontsize -= 1
            txt = "Label image"
            offset_for_centering = int((col_width - font_regular.getlength(txt)) / 2)
            if superimpose_all_jobs:
                offset = (
                    width_pad_offset
                    + col_width
                    + buffer_size * 2
                    + offset_for_centering,
                    int(buffer_size * 0.5),
                )
            else:
                offset = (
                    width_pad_offset
                    + len(job_dirs) * col_width
                    + (len(job_dirs) + 1) * buffer_size
                    + offset_for_centering,
                    int(buffer_size * 0.5),
                )
            draw.text(offset, txt, WHITE, font=font_regular)

        # Add footer
        footer = build_footer_img(
            width,
            footer_height,
            iteration,
            prediction_type,
            region_sets_to_include,
            region_sets_to_label,
            cmap_name,
        )
        frame.paste(footer, (0, height - footer_height))

        # Downsize image while keeping aspect ratio
        if max_size is not None:
            frame.thumbnail(max_size, Image.Resampling.BICUBIC)

        return frame

    def create_animation(
        self, filename, fps, iterations, write_sequence, *args, **kwargs
    ):
        animation = []
        for iteration in tqdm(iterations):
            frame = self.build_frame(iteration, *args, **kwargs)
            animation.append(frame)
        write_gif(animation, filename + ".gif", fps=fps)
        if write_sequence:
            write_img_sequence(animation, filename)

    def compute_metrics(self, validation_ds_path):
        def get_path_name(path: str) -> str:
            return Path(path).name

        validation_ds = inkid.data.Dataset([validation_ds_path], lazy_load=True)
        iteration_to_preds = {}
        iteration_to_labels = {}
        for region in validation_ds.regions():
            # We know the PPM and invert_normals for this region, so get them
            normalized_ppm_path = get_path_name(region.source_json["ppm"])
            invert_normals = region.source_json["invert_normals"]
            # Filter to the prediction images matching that (ppm, invert_normal) pair
            pred_imgs_df = self.prediction_images_df.merge(self.regions_df)
            pred_imgs_df["normalized_ppm_path"] = pred_imgs_df["ppm_path"].apply(
                get_path_name
            )
            pred_imgs_df = pred_imgs_df[
                (pred_imgs_df["normalized_ppm_path"] == normalized_ppm_path)
                & (pred_imgs_df["invert_normals"] == invert_normals)
            ]
            # For each (iteration, prediction_type) in this (ppm, invert_normal) pair compose the prediction image
            pred_imgs_groups = pred_imgs_df.groupby(
                ["iteration_str", "prediction_type"],
                as_index=False,
            )
            for (iteration_str, prediction_type), group_df in pred_imgs_groups:
                # Compose the prediction image
                ppm_path = group_df["ppm_path"].iloc[0]
                bounding_box = region.bounding_box
                img = self.get_face_prediction_image(
                    job_dir=None,
                    ppm_path=ppm_path,
                    invert_normals=invert_normals,
                    iteration=iteration_str,
                    prediction_type=prediction_type,
                    region_sets_to_include=["prediction"],
                    region_sets_to_label=[],
                    rectangle_line_width=0,
                    cmap_name=None,
                )

                # Get the label image for comparison
                label_key = label_key_from_prediction_type(prediction_type)
                label_img_path = self.get_label_image_path(
                    ppm_path, invert_normals, label_key
                )
                assert label_img_path is not None
                label_img = try_get_img_from_data_files(label_img_path)
                assert label_img is not None
                ppm_mask_img_path = self.get_mask_image_path(ppm_path, invert_normals)
                ppm_mask_img = try_get_img_from_data_files(ppm_mask_img_path)

                print(
                    f"Getting {prediction_type} preds for PPM: "
                    f"{Path(ppm_path).name}, "
                    f"from iteration: {iteration_str} "
                    f"and labels from image: {Path(label_img_path).name}"
                )

                preds, labels = get_preds_and_labels(img, label_img, ppm_mask_img, bounding_box)
                if iteration_str not in iteration_to_preds:
                    iteration_to_preds[iteration_str] = np.array(preds)
                    iteration_to_labels[iteration_str] = np.array(labels)
                else:
                    iteration_to_preds[iteration_str] = np.concatenate(
                        (iteration_to_preds[iteration_str], preds)
                    )
                    iteration_to_labels[iteration_str] = np.concatenate(
                        (iteration_to_labels[iteration_str], labels)
                    )

                break

        metrics_results = {}
        # Compute the metric
        if prediction_type == "ink_classes":
            metrics_results["iterations"] = []
            metrics_results["crossEntropyLoss"] = []
            metrics_results["dice"] = []
            metrics_results["falsePositiveRate"] = []
            all_iterations = sorted(iteration_to_preds.keys())
            for iteration_str in all_iterations:
                print(f"Computing metrics for iteration: {iteration_str}")
                preds = iteration_to_preds[iteration_str]
                labels = iteration_to_labels[iteration_str]
                result = compute_ink_classes_metrics(preds, labels)
                metrics_results["iterations"].append(iteration_str)
                metrics_results["crossEntropyLoss"].append(result["crossEntropyLoss"])
                metrics_results["dice"].append(result["dice"])
                metrics_results["falsePositiveRate"].append(result["falsePositiveRate"])

        return metrics_results


def merge_imgs(
    paths,
    bounding_boxes,
    image_label_as,
    sets_to_label,
    rectangle_line_width,
    cmap_name,
):
    merged_img = None

    new_bounding_boxes = []
    for path, bounding_box in zip(paths, bounding_boxes):
        img = Image.open(path)
        if bounding_box is None:
            bounding_box = (0, 0, img.width, img.height)
        new_bounding_boxes.append(bounding_box)
        # For some images (grayscale PNGs in my experience so far), Pillow opens them as 32-bit integer images and
        # then clips them to 8-bit in later stages, creating washed out images. If it has opened an image as 32-bit
        # we detect and convert it here, so it is not clipped later.
        if img.mode == "I":
            array = np.uint8(np.array(img) / 256)
            img = Image.fromarray(array)
        # Convert all to RGB since we might draw on them with color
        img = img.convert("RGB")
        # Paste onto merged image
        if merged_img is None:
            merged_img = img
        else:
            assert img.size == merged_img.size
            img = img.crop(bounding_box)
            merged_img.paste(img, (bounding_box[0], bounding_box[1]))

    # Apply color map
    if cmap_name is not None and merged_img is not None:
        color_map = colormaps[cmap_name]
        merged_img = merged_img.convert("L")
        merged_img_data = np.array(merged_img)
        merged_img = Image.fromarray(np.uint8(color_map(merged_img_data) * 255))

    # Draw bounding boxes
    for bounding_box, (training, prediction, validation) in zip(
        new_bounding_boxes, image_label_as
    ):
        to_draw = {
            "training": training,
            "prediction": prediction,
            "validation": validation,
        }
        for region_set_type, is_this_type in to_draw.items():
            if is_this_type and region_set_type in sets_to_label:
                d = ImageDraw.Draw(merged_img)
                color = region_set_type_to_color[region_set_type]
                d.rectangle(
                    bounding_box, outline=color, fill=None, width=rectangle_line_width
                )

    return merged_img


def build_footer_img(
    width,
    height,
    iteration,
    label_type,
    regions_shown,
    regions_to_label,
    cmap_name=None,
):
    footer = Image.new("RGB", (width, height))
    horizontal_offset = 0
    divider_bar_size = max(1, int(width / 500))
    buffer_size = int(height / 8)
    # Fill with dark gray
    footer.paste(DARK_GRAY, (0, 0, width, height))
    # Add logos
    for logo_filename in ["EduceLabBW.png", "UK logo-white.png"]:
        logo_path = os.path.join(
            os.path.dirname(inkid.__file__), "assets", logo_filename
        )
        logo = Image.open(logo_path)
        logo.thumbnail((100000, height - 2 * buffer_size), Image.Resampling.BICUBIC)
        logo_offset = (horizontal_offset + buffer_size, buffer_size)
        # Third argument used as transparency mask. Convert to RGBA to force presence of alpha channel
        footer.paste(logo, logo_offset, logo.convert("RGBA"))
        horizontal_offset += logo.size[0] + 2 * buffer_size
        # Add divider bar
        footer.paste(
            LIGHT_GRAY,
            (horizontal_offset, 0, horizontal_offset + divider_bar_size, height),
        )
        horizontal_offset += divider_bar_size
    # Add epoch
    if iteration != "final":
        draw = ImageDraw.Draw(footer)
        font_path = os.path.join(
            os.path.dirname(inkid.__file__), "assets", "fonts", "Roboto-Regular.ttf"
        )
        fontsize = 1
        font_regular = ImageFont.truetype(font_path, fontsize)
        txt = "eooch"  # Hack as I don't want it to care about 'p' sticking down
        allowed_font_height = int((height - buffer_size * 3) / 2)
        while int(font_regular.getbbox(txt)[3]) < allowed_font_height:
            fontsize += 1
            font_regular = ImageFont.truetype(font_path, fontsize)
        fontsize -= 1
        txt = "epoch"
        draw.text(
            (horizontal_offset + buffer_size, buffer_size),
            txt,
            WHITE,
            font=font_regular,
        )
        font_w = int(font_regular.getlength(txt) + 2 * buffer_size)
        font_path_black = os.path.join(
            os.path.dirname(inkid.__file__), "assets", "fonts", "Roboto-Black.ttf"
        )
        font_black = ImageFont.truetype(font_path_black, fontsize)
        epoch = (
            "" if iteration == "final" else re.search(r"(\d+)_\d+", iteration).group(1)
        )
        draw.text(
            (horizontal_offset + buffer_size, allowed_font_height + 2 * buffer_size),
            epoch,
            WHITE,
            font=font_black,
        )
        horizontal_offset += font_w
        # Add divider bar
        footer.paste(
            LIGHT_GRAY,
            (horizontal_offset, 0, horizontal_offset + divider_bar_size, height),
        )
        horizontal_offset += divider_bar_size
    # Add batch
    draw = ImageDraw.Draw(footer)
    font_path = os.path.join(
        os.path.dirname(inkid.__file__), "assets", "fonts", "Roboto-Regular.ttf"
    )
    fontsize = 1
    font_regular = ImageFont.truetype(font_path, fontsize)
    txt = "batch"
    allowed_font_height = int((height - buffer_size * 3) / 2)
    while int(font_regular.getbbox(txt)[3]) < allowed_font_height:
        fontsize += 1
        font_regular = ImageFont.truetype(font_path, fontsize)
    fontsize -= 1
    draw.text(
        (horizontal_offset + buffer_size, buffer_size), txt, WHITE, font=font_regular
    )
    font_w = int(font_regular.getlength(txt) + 2 * buffer_size)
    font_path_black = os.path.join(
        os.path.dirname(inkid.__file__), "assets", "fonts", "Roboto-Black.ttf"
    )
    font_black = ImageFont.truetype(font_path_black, fontsize)
    batch = (
        "final" if iteration == "final" else re.search(r"\d+_(\d+)", iteration).group(1)
    )
    draw.text(
        (horizontal_offset + buffer_size, allowed_font_height + 2 * buffer_size),
        batch,
        WHITE,
        font=font_black,
    )
    batch_font_w = int(font_black.getlength(batch) + 2 * buffer_size)
    horizontal_offset += max(font_w, batch_font_w)
    # Add divider bar
    footer.paste(
        LIGHT_GRAY, (horizontal_offset, 0, horizontal_offset + divider_bar_size, height)
    )
    horizontal_offset += divider_bar_size
    # Add color map name
    if cmap_name is not None:
        cmap_title = "color map"
        draw.text(
            (horizontal_offset + buffer_size, buffer_size),
            cmap_title,
            WHITE,
            font=font_regular,
        )
        draw.text(
            (horizontal_offset + buffer_size, allowed_font_height + 2 * buffer_size),
            cmap_name,
            WHITE,
            font=font_black,
        )
        font_w = int(
            max(font_regular.getlength(cmap_title), font_black.getlength(cmap_name))
        )
        horizontal_offset += font_w + 2 * buffer_size
        # Add divider bar
        footer.paste(
            LIGHT_GRAY,
            (horizontal_offset, 0, horizontal_offset + divider_bar_size, height),
        )
        horizontal_offset += divider_bar_size
    # Add color map swatch
    if label_type == "ink_classes":
        swatch_title = "no ink            ink"
        swatch_title_w = int(font_regular.getlength(swatch_title))
        swatch_title_h = int(font_regular.getbbox(swatch_title)[3])
        swatch = inkid.util.create_colormap_swatch(
            cmap_name, swatch_title_w, swatch_title_h
        )
        draw.text(
            (horizontal_offset + buffer_size, buffer_size),
            swatch_title,
            WHITE,
            font=font_regular,
        )
        footer.paste(
            swatch,
            (horizontal_offset + buffer_size, allowed_font_height + 2 * buffer_size),
        )
        horizontal_offset += swatch.width + 2 * buffer_size
        # Add divider bar
        footer.paste(
            LIGHT_GRAY,
            (horizontal_offset, 0, horizontal_offset + divider_bar_size, height),
        )
        horizontal_offset += divider_bar_size
    regions_title = "regions shown"
    draw.text(
        (horizontal_offset + buffer_size, buffer_size),
        regions_title,
        WHITE,
        font=font_regular,
    )
    regions_txt = ""
    for i, region_type in enumerate(regions_shown):
        regions_txt += region_type
        # Leave space for legend rectangle to surround word
        if region_type in regions_to_label:
            regions_txt += " "
        # Add commas between words
        if i != len(regions_shown) - 1:
            regions_txt += ", "
        if len(regions_shown) == 1:
            regions_txt += " only"
    draw.text(
        (horizontal_offset + buffer_size, allowed_font_height + 2 * buffer_size),
        regions_txt,
        WHITE,
        font=font_black,
    )
    font_w = int(
        max(font_regular.getlength(regions_title), font_black.getlength(regions_txt))
    )
    for region_type in regions_to_label:
        if region_type in regions_txt:
            preceding_txt = regions_txt.partition(region_type)[0]
            x0 = (
                horizontal_offset
                + buffer_size
                + int(font_black.getlength(preceding_txt))
            )
            label_w = int(font_black.getlength(region_type))
            x1 = x0 + label_w
            y0 = allowed_font_height + 2 * buffer_size
            label_h = int(font_black.getbbox(regions_txt)[3])
            y1 = y0 + label_h
            # Pad these a bit to not be right on the text
            x0 -= int(buffer_size / 2)
            y0 -= int(buffer_size / 2)
            x1 += int(buffer_size / 2)
            y1 += int(buffer_size / 2)
            color = region_set_type_to_color[region_type]
            draw = ImageDraw.Draw(footer)
            draw.rectangle(
                (x0, y0, x1, y1), outline=color, fill=None, width=int(height / 40)
            )
    horizontal_offset += font_w + 2 * buffer_size
    # Add divider bar
    footer.paste(
        LIGHT_GRAY, (horizontal_offset, 0, horizontal_offset + divider_bar_size, height)
    )
    horizontal_offset += divider_bar_size
    return footer


def try_get_img_from_data_files(img_path):
    img = None
    if os.path.isfile(img_path):
        img = Image.open(img_path)
    # If not there, maybe it is on the local machine under ~/data.
    else:
        for possible_source_prefix in ["/pscratch/seales_uksr/", "/scratch/srpa226/"]:
            if possible_source_prefix in img_path:
                simplified_img_path = img_path.replace(possible_source_prefix, "")
                for possible_target_prefix in ["data", "bigdata", "temp"]:
                    possible_target_path = os.path.join(
                        Path.home(), possible_target_prefix, simplified_img_path
                    )
                    if os.path.isfile(possible_target_path):
                        img = Image.open(possible_target_path)
                        break
    return img


def write_img_sequence(animation, outdir):
    if len(animation) == 0:
        return
    print("\nWriting image sequence to", outdir)
    prefix = os.path.join(outdir, "sequence_")
    for i, img in enumerate(animation):
        outfile = prefix + str(i) + ".png"
        img.save(outfile)


def write_gif(animation, outfile, fps=10):
    if len(animation) == 0:
        return
    print("\nWriting gif to", outfile)
    durations = [1 / fps] * len(animation)
    # Make the last frame hold for longer.
    durations[-1] = 5
    with imageio.get_writer(outfile, mode="I", duration=durations) as writer:
        for img in tqdm(animation):
            writer.append_data(np.array(img))

    # Optimize .gif file size
    prev_size = os.path.getsize(outfile)
    print("\nOptimizing .gif file", outfile)
    pygifsicle.optimize(outfile, options=["-w"])
    new_size = os.path.getsize(outfile)
    reduction = (prev_size - new_size) / prev_size * 100
    print(
        f"Size reduced {reduction:.2f}% from {naturalsize(prev_size)} to {naturalsize(new_size)}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # Input directory with job output
    parser.add_argument("dir", metavar="path", help="input directory")
    # Image generation options
    parser.add_argument(
        "--img-seq",
        action="store_true",
        help="Save an image sequence for any animations made",
    )
    parser.add_argument("--gif-fps", default=10, type=int, help="GIF frames per second")
    parser.add_argument("--gif-max-size", type=int, nargs=2, default=[1920, 1080])
    parser.add_argument("--static-max-size", type=int, nargs=2, default=[3840, 2160])
    parser.add_argument("--no-label-column", action="store_true")
    parser.add_argument("--validation-dataset-for-metrics")
    parser.add_argument("--metrics-only", action="store_true")
    args = parser.parse_args()

    out_dir = f'{datetime.datetime.today().strftime("%Y-%m-%d_%H.%M.%S")}_summary'
    out_dir = os.path.join(args.dir, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    job_summarizer = JobSummarizer(args.dir)

    if args.validation_dataset_for_metrics:
        print("\nComputing metrics over validation dataset...")
        metrics_results = job_summarizer.compute_metrics(args.validation_dataset_for_metrics)
        metrics_dir = Path(out_dir) / "metrics"
        metrics_dir.mkdir()

        csv_rows = []
        job_date = job_summarizer.date
        job_date = datetime.datetime.strptime(job_date, "%Y-%m-%d-%H:%M:%S").date()
        job_name = job_summarizer.name

        csv_rows.append(["iterations", "crossEntropyLoss", "dice"])
        for i, iteration in enumerate(metrics_results["iterations"]):
            cross_entropy_loss = metrics_results["crossEntropyLoss"][i]
            dice = metrics_results["dice"][i]
            csv_rows.append([iteration, cross_entropy_loss, dice])

        csv_path = metrics_dir / "metrics.csv"
        with open(csv_path, "w", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(csv_rows)

        print("done.")
        if args.metrics_only:
            return

    label_column = not args.no_label_column
    if not job_summarizer.any_label_images_found():
        print(
            "Label column requested but no label images found. Not adding label column to summary images."
        )
        label_column = False

    if not job_summarizer.any_prediction_images_found():
        print(
            "No iterations encountered in prediction folders. No images will be produced."
        )
        return

    for prediction_type in job_summarizer.prediction_types():
        last_iteration_seen = job_summarizer.last_iteration_seen(prediction_type)

        # Static images
        print(f"\nCreating final static {prediction_type} image with all regions...")
        final_static_all_img = job_summarizer.build_frame(
            last_iteration_seen,
            prediction_type,
            args.static_max_size,
            region_sets_to_label=["training"],
            label_column=label_column,
        )
        final_static_all_img.save(
            os.path.join(out_dir, f"{prediction_type}_{last_iteration_seen}_all.png")
        )
        print("done.")

        print(
            f"\nCreating final static {prediction_type} image with only prediction regions..."
        )
        final_static_prediction_img = job_summarizer.build_frame(
            last_iteration_seen,
            prediction_type,
            args.static_max_size,
            region_sets_to_include=["prediction"],
            superimpose_all_jobs=True,
            label_column=label_column,
        )
        final_static_prediction_img.save(
            os.path.join(
                out_dir, f"{prediction_type}_{last_iteration_seen}_prediction.png"
            )
        )
        print("done.")

        # Color map images
        color_maps_dir = os.path.join(out_dir, "colormaps")
        os.makedirs(color_maps_dir, exist_ok=True)
        for cmap in [
            "plasma",
            "viridis",
            "hot",
            "inferno",
            "seismic",
            "Spectral",
            "coolwarm",
            "bwr",
        ]:
            print(
                f"\nCreating final static {prediction_type} image with all regions and color map: {cmap}..."
            )
            final_static_all_cmap_img = job_summarizer.build_frame(
                last_iteration_seen,
                prediction_type,
                args.static_max_size,
                cmap_name=cmap,
                region_sets_to_label=["training"],
                label_column=label_column,
            )
            final_static_all_cmap_img.save(
                os.path.join(
                    color_maps_dir,
                    f"{prediction_type}_{last_iteration_seen}_all_{cmap}.png",
                )
            )
            print("done.")

            print(
                f"\nCreating final static {prediction_type} image with prediction regions and color map: {cmap}..."
            )
            final_static_prediction_cmap_img = job_summarizer.build_frame(
                last_iteration_seen,
                prediction_type,
                args.static_max_size,
                region_sets_to_include=["prediction"],
                superimpose_all_jobs=True,
                label_column=label_column,
                cmap_name=cmap,
            )
            final_static_prediction_cmap_img.save(
                os.path.join(
                    color_maps_dir,
                    f"{prediction_type}_{last_iteration_seen}_prediction_{cmap}.png",
                )
            )
            print("done.")

        # Gifs
        print(f"\nCreating {prediction_type} animation with all regions:")
        job_summarizer.create_animation(
            os.path.join(out_dir, f"{prediction_type}_animation_all"),
            args.gif_fps,
            job_summarizer.iterations_encountered(prediction_type),
            args.img_seq,
            prediction_type,
            args.gif_max_size,
            region_sets_to_label=["training"],
            label_column=label_column,
        )

        print("\nCreating animation with only prediction regions:")
        job_summarizer.create_animation(
            os.path.join(out_dir, f"{prediction_type}_animation_prediction"),
            args.gif_fps,
            job_summarizer.iterations_encountered(prediction_type),
            args.img_seq,
            prediction_type,
            args.gif_max_size,
            region_sets_to_include=["prediction"],
            superimpose_all_jobs=True,
            label_column=label_column,
        )


if __name__ == "__main__":
    main()
