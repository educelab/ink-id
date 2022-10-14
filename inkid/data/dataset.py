# For DataSource.from_path() https://stackoverflow.com/a/33533514
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass
import json
import math
import os
import random
from typing import Dict, List, Optional, Tuple

import jsonschema
import numpy as np
from PIL import Image, ImageFilter
import torch

import inkid


# Which subvolumes are retrieved
@dataclass
class RegionPointSampler:
    grid_spacing: int = 1
    specify_inkness: Optional[bool] = None
    undersampling_ink_ratio: Optional[float] = None
    oversampling_ink_ratio: Optional[float] = None
    ambiguous_ink_labels_filter_radius: Optional[float] = None
    ambiguous_labels_mask = None


# How subvolumes are retrieved
@dataclass
class SubvolumeGeneratorInfo:
    shape_microns: Optional[tuple[float, float, float]] = None
    shape_voxels: Optional[tuple[int, int, int]] = (48, 48, 48)
    move_along_normal: float = 0
    normalize_subvolume: bool = False
    jitter_max: int = 4
    augment_subvolume: bool = True


# How subvolumes are retrieved
def add_subvolume_arguments(parser):
    default = SubvolumeGeneratorInfo()
    parser.add_argument(
        "--subvolume-shape-microns",
        metavar="um",
        nargs=3,
        type=float,
        default=default.shape_microns,
        help="subvolume shape (microns) in (z, y, x)",
    )
    parser.add_argument(
        "--subvolume-shape-voxels",
        metavar="n",
        nargs=3,
        type=int,
        help="subvolume shape (voxels) in (z, y, x)",
        default=default.shape_voxels,
    )
    parser.add_argument(
        "--move-along-normal",
        metavar="n",
        type=float,
        default=default.move_along_normal,
        help="number of voxels to move along normal vector before sampling a subvolume",
    )
    parser.add_argument(
        "--normalize-subvolumes",
        action="store_true",
        help="normalize each subvolume to zero mean and unit variance",
    )
    parser.add_argument(
        "--jitter-max", metavar="n", type=int, default=default.jitter_max
    )
    parser.add_argument("--no-augmentation", action="store_false", dest="augmentation")


# Tuple (not dataclass) I believe because needs to be passed through PyTorch and needs to be basic structure
FeatureMetadata = namedtuple(
    "FeatureMetadata",
    ("path", "surface_x", "surface_y", "x", "y", "z", "n_x", "n_y", "n_z"),
    defaults=None,
)


class DataSource(ABC):
    """Can be either a region or a volume. Produces inputs (e.g. subvolumes) and possibly labels."""

    def __init__(self, path: str) -> None:
        self.path = path
        source_file_contents, relative_url = inkid.util.get_raw_data_from_file_or_url(
            path, return_relative_url=True
        )
        self.unmodified_source_json: Dict = json.load(source_file_contents)
        self.source_json = self.unmodified_source_json.copy()
        # Validate JSON fits schema
        jsonschema.validate(self.source_json, inkid.util.json_schema("dataSource0.1"))
        # Normalize paths in JSON
        for key in [
            "volume",
            "ppm",
            "mask",
            "ink_label",
            "rgb_label",
            "volcart_texture_label",
        ]:
            if key in self.source_json:
                self.source_json[key] = inkid.util.normalize_path(
                    self.source_json[key], relative_url
                )

        self.feature_args: dict = dict()
        self.label_types: list[str] = []
        self.label_args: dict = dict()

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError

    @staticmethod
    def from_path(path: str, lazy_load: bool = False) -> DataSource:
        """Check first whether this is a region or volume data source, then instantiate accordingly.

        Also checks to make sure the old region set file format was not provided. If it was,
        a message is issued telling the user how to update the file format.

        """
        with open(path, "r") as f:
            source_json = json.load(f)
        if "ppms" in source_json and "regions" in source_json:
            raise ValueError(
                f"Source file {path} uses the deprecated region set file format. "
                f"Please convert it to the updated data source format. "
                f"The following script can be used: \n\n"
                f"\tpython inkid/scripts/update_data_file.py {path}"
            )
        if source_json.get("type") == "region":
            return RegionSource(path, lazy_load=lazy_load)
        elif source_json.get("type") == "volume":
            return VolumeSource(path)
        else:
            raise ValueError(
                f'Source file {path} does not specify valid "type" of "region" or "volume"'
            )


class RegionSource(DataSource):
    """A region source generates subvolumes and labels from a region (bounding box) on a PPM.

    (x, y) are always in the PPM space, not this region's bounding box space.

    """

    def __init__(self, path: str, lazy_load: bool = False) -> None:
        super().__init__(path)

        # Initialize region's PPM, volume, etc
        self._ppm: inkid.data.PPM = inkid.data.PPM.from_path(self.source_json["ppm"], lazy_load=lazy_load)
        self.bounding_box: Tuple[int, int, int, int] = (
            self.source_json["bounding_box"] or self.get_default_bounds()
        )
        self._invert_normals: bool = self.source_json["invert_normals"]
        if lazy_load:
            self.volume = None
        else:
            self.volume: inkid.data.Volume = inkid.data.Volume.from_path(
                self.source_json["volume"]
            )

        # Mask and label images
        self._mask, self._ink_label, self._rgb_label, self._volcart_texture_label = (
            None,
            None,
            None,
            None,
        )
        if self.source_json["mask"] is not None:
            self._mask = np.array(Image.open(self.source_json["mask"]))
        if self.source_json["ink_label"] is not None:
            im = Image.open(self.source_json["ink_label"]).convert(
                "L"
            )  # Allow RGB mode images
            self._ink_label = np.array(im)
        if self.source_json["rgb_label"] is not None:
            im = Image.open(self.source_json["rgb_label"]).convert("RGB")
            self._rgb_label = np.array(im).astype(np.float32)
            # Make sure RGB image loaded wasn't more than 8 bit
            assert np.amax(self._rgb_label) <= np.iinfo(np.uint8).max
            # Map from [0, 255] to [0, 1]
            self._rgb_label /= np.iinfo(np.uint8).max
        if self.source_json["volcart_texture_label"] is not None:
            im = Image.open(self.source_json["volcart_texture_label"])
            # Assuming image is uint16 data but Pillow loads it as uint32 ('I')
            assert im.mode == "I"
            self._volcart_texture_label = np.array(im).astype(np.float32)
            # Make sure data seems to be 16 bit (not a perfect check)
            assert (
                np.iinfo(np.uint8).max
                <= np.amax(self._volcart_texture_label)
                <= np.iinfo(np.uint16).max
            )
            # Normalize to [0.0, 1.0]
            self._volcart_texture_label /= np.iinfo(np.uint16).max

        # This region generates points, here we create the empty list
        self._points = list()
        # Mark that this list needs updating so that it will be filled before being accessed
        self._points_list_needs_update: bool = True

        self.sampler = RegionPointSampler()

        # Prediction images
        self._ink_classes_prediction_image = np.zeros(
            (self._ppm.height, self._ppm.width), np.uint16
        )
        self._ink_classes_prediction_image_written_to = False
        self._rgb_values_prediction_image = np.zeros(
            (self._ppm.height, self._ppm.width, 3), np.uint8
        )
        self._rgb_values_prediction_image_written_to = False
        self._volcart_texture_prediction_image = np.zeros(
            (self._ppm.height, self._ppm.width), np.uint16
        )
        self._volcart_texture_prediction_image_written_to = False

    @property
    def name(self) -> str:
        return os.path.splitext(os.path.basename(self.path))[0]

    def __len__(self) -> int:
        if self._points_list_needs_update:
            self.update_points_list()
        return len(self._points)

    def __getitem__(self, item):
        if self._points_list_needs_update:
            self.update_points_list()
        # Get the point (x, y) from list of points
        surface_x, surface_y = self._points[item]
        # Read that value from PPM
        x, y, z, n_x, n_y, n_z = self._ppm.get_point_with_normal(surface_x, surface_y)
        # Invert normal if needed
        if self._invert_normals:
            n_x, n_y, n_z = -n_x, -n_y, -n_z
        # Get the feature metadata (useful for e.g. knowing where this feature came from on the surface)
        feature_metadata = FeatureMetadata(
            self.path, surface_x, surface_y, x, y, z, n_x, n_y, n_z
        )
        # Get the feature
        feature = self.volume.get_subvolume(
            center=(x, y, z), normal=(n_x, n_y, n_z), **self.feature_args
        )
        item = {
            "feature_metadata": feature_metadata,
            "feature": feature,
        }
        # Get the label
        if "ink_classes" in self.label_types:
            item["ink_classes"] = self.point_to_ink_classes_label(
                (surface_x, surface_y), **self.label_args["ink_classes"]
            )
        if "rgb_values" in self.label_types:
            item["rgb_values"] = self.point_to_rgb_values_label(
                (surface_x, surface_y), **self.label_args["rgb_values"]
            )
        if "volcart_texture" in self.label_types:
            item["volcart_texture"] = self.point_to_volcart_texture_label(
                (surface_x, surface_y), **self.label_args["volcart_texture"]
            )
        return item

    def update_points_list(self) -> None:
        """Update the list of points after changes to the bounding box, grid spacing, or some other options."""
        # TODO move this inside sampler
        if (
            self._ink_label is not None
            and self.sampler.ambiguous_ink_labels_filter_radius is not None
        ):
            ambiguous_labels_mask = Image.fromarray(self._ink_label)
            # 3x3 Laplacian kernel for edge detection
            ambiguous_labels_mask = ambiguous_labels_mask.filter(ImageFilter.FIND_EDGES)
            # Max filter for dilation
            dilation_kernel_width = int(
                2 * self.sampler.ambiguous_ink_labels_filter_radius + 1
            )
            ambiguous_labels_mask = ambiguous_labels_mask.filter(
                ImageFilter.MaxFilter(dilation_kernel_width)
            )
            self.sampler.ambiguous_labels_mask = np.array(ambiguous_labels_mask)
        positive_points = list()
        negative_points = list()
        unlabeled_points = list()
        x0, y0, x1, y1 = self.bounding_box
        for y in range(y0, y1, self.sampler.grid_spacing):
            for x in range(x0, x1, self.sampler.grid_spacing):
                if not self.is_on_surface(x, y):
                    continue
                if self.sampler.specify_inkness is not None:
                    if self.sampler.specify_inkness and not self.is_ink(x, y):
                        continue
                    elif not self.sampler.specify_inkness and self.is_ink(x, y):
                        continue
                # Filter out points with ambiguous ink labels
                if self.sampler.ambiguous_labels_mask is not None:
                    if self.sampler.ambiguous_labels_mask[y, x] != 0:
                        continue
                if (
                    self.sampler.oversampling_ink_ratio is not None
                    or self.sampler.undersampling_ink_ratio is not None
                ):
                    if self.is_ink(x, y):
                        positive_points.append((x, y))
                    else:
                        negative_points.append((x, y))
                else:
                    unlabeled_points.append((x, y))

        """
        For the given ink ratio,
        Undersampling: reduces the number of negative points for the positive
            points available.
        Oversampling: repeats the positive points in order to achieve the
            desired ratio for the given number of negative points.
        If neither undersampling nor oversampling is chosen, all the
            available points are used without balancing classes.

        undersampling_ink_ratio: float (default=None)
        oversampling_ink_ratio: float  (default=None)
        """
        if self.sampler.undersampling_ink_ratio is not None:
            negative_ratio = 1.0 - self.sampler.undersampling_ink_ratio
            negatives_needed = int(
                len(positive_points)
                * negative_ratio
                / self.sampler.undersampling_ink_ratio
            )
            self._points = positive_points + random.choices(
                negative_points, k=negatives_needed
            )
        elif self.sampler.oversampling_ink_ratio is not None:
            negative_ratio = 1.0 - self.sampler.oversampling_ink_ratio
            positives_needed = int(
                len(negative_points)
                * self.sampler.oversampling_ink_ratio
                / negative_ratio
            )
            positive_reps = (
                1
                if positives_needed < len(positive_points)
                else int(positives_needed / len(positive_points))
            )
            extended_positive_points = list(
                np.repeat(np.array(positive_points), positive_reps, axis=0)
            )
            self._points = extended_positive_points + negative_points
        else:
            self._points = unlabeled_points

        self._points_list_needs_update = False

    @property
    def sampler(self):
        return self._sampler

    @sampler.setter
    def sampler(self, sampler: RegionPointSampler):
        self._sampler = sampler
        self._points_list_needs_update = True

    def is_ink(self, x: int, y: int) -> bool:
        assert self._ink_label is not None
        return self._ink_label[y, x] != 0

    def is_on_surface(self, x: int, y: int, r: int = 1) -> bool:
        """Return whether a point is on the surface mask.

        Check a point and a square of radius r around it, and return
        False if any of those points are not on the surface
        mask. Return True otherwise.

        """
        assert self._mask is not None
        square = self._mask[y - r : y + r + 1, x - r : x + r + 1]
        return np.size(square) > 0 and np.min(square) != 0

    def get_default_bounds(self) -> Tuple[int, int, int, int]:
        """Return the full bounds of the PPM in (x0, y0, x1, y1) format."""
        return 0, 0, self._ppm.width, self._ppm.height

    def point_to_ink_classes_label(self, point, shape):
        # 0 = no ink, 1 = ink
        assert self._ink_label is not None
        x, y = point
        label = np.zeros(shape).astype(np.float32)  # Create array of no-ink labels
        y_d, x_d = (
            np.array(shape) // 2
        )  # Calculate distance from center to edges of square we are sampling
        # Iterate over label indices
        for idx, _ in np.ndenumerate(label):
            y_idx, x_idx = idx
            y_s = (
                y - y_d + y_idx
            )  # Sample point is center minus distance (half edge length) plus label index
            x_s = x - x_d + x_idx
            # Bounds check to make sure inside PPM
            if (
                0 <= y_s < self._ink_label.shape[0]
                and 0 <= x_s < self._ink_label.shape[1]
            ):
                if self._ink_label[y_s, x_s] != 0:
                    label[y_idx, x_idx] = 1.0  # Mark this "ink"
        return torch.Tensor(label).long()

    def point_to_rgb_values_label(self, point, shape):
        assert self._rgb_label is not None
        x, y = point
        label = np.zeros((3,) + shape).astype(np.float32)
        y_d, x_d = (
            np.array(shape) // 2
        )  # Calculate distance from center to edges of square we are sampling
        # Iterate over label indices
        for idx, _ in np.ndenumerate(label):
            _, y_idx, x_idx = idx
            y_s = (
                y - y_d + y_idx
            )  # Sample point is center minus distance (half edge length) plus label index
            x_s = x - x_d + x_idx
            # Bounds check to make sure inside PPM
            if (
                0 <= y_s < self._rgb_label.shape[0]
                and 0 <= x_s < self._rgb_label.shape[1]
            ):
                label[:, y_idx, x_idx] = self._rgb_label[y_s, x_s]
        return label

    def point_to_volcart_texture_label(self, point, shape):
        assert self._volcart_texture_label is not None
        x, y = point
        label = np.zeros((1,) + shape).astype(np.float32)
        y_d, x_d = (
            np.array(shape) // 2
        )  # Calculate distance from center to edges of square we are sampling
        # Iterate over label indices
        for idx, _ in np.ndenumerate(label):
            _, y_idx, x_idx = idx
            y_s = (
                y - y_d + y_idx
            )  # Sample point is center minus distance (half edge length) plus label index
            x_s = x - x_d + x_idx
            # Bounds check to make sure inside PPM
            if (
                0 <= y_s < self._volcart_texture_label.shape[0]
                and 0 <= x_s < self._volcart_texture_label.shape[1]
            ):
                label[0, y_idx, x_idx] = self._volcart_texture_label[y_s, x_s]
        return label

    def store_prediction(self, x, y, prediction, label_type):
        """Store an incoming prediction in the corresponding prediction image buffer.

        The incoming predictions have shape [d, h, w]. When h and w are both 1, a prediction has been provided for
        a single point rather than a rectangular region.

        """
        # Repeat prediction to fill grid square so prediction image is not single pixels in sea of blackness
        if prediction.shape[1] == 1 and prediction.shape[2] == 1:
            prediction = np.repeat(
                prediction, repeats=self.sampler.grid_spacing, axis=1
            )
            prediction = np.repeat(
                prediction, repeats=self.sampler.grid_spacing, axis=2
            )
        # Calculate distance from center to edges of square we are writing
        y_d, x_d = np.array(prediction.shape)[1:] // 2
        # Iterate over label indices
        for idx in np.ndindex(prediction.shape[1:]):
            y_idx, x_idx = idx
            value = prediction[:, y_idx, x_idx]
            # Sample point in PPM space is center minus distance (half edge length) plus label index
            y_s = y - y_d + y_idx
            x_s = x - x_d + x_idx
            # Bounds check to make sure inside PPM
            if 0 <= x_s < self._ppm.width and 0 <= y_s < self._ppm.height:
                if label_type == "ink_classes":
                    # Convert ink class probability to image intensity
                    v = value[1] * np.iinfo(np.uint16).max
                    self._ink_classes_prediction_image[y_s, x_s] = v
                    self._ink_classes_prediction_image_written_to = True
                elif label_type == "rgb_values":
                    # Rescale from [0, 1] to [0, 255]
                    v = value * np.iinfo(np.uint8).max
                    # Restrict value to uint8 range
                    v = np.clip(v, 0, np.iinfo(np.uint8).max)
                    self._rgb_values_prediction_image[y_s, x_s] = v
                    self._rgb_values_prediction_image_written_to = True
                elif label_type == "volcart_texture":
                    # Rescale from [0, 1]
                    v = value[0] * np.iinfo(np.uint16).max
                    # Restrict value to uint16 range
                    v = np.clip(v, 0, np.iinfo(np.uint16).max)
                    self._volcart_texture_prediction_image[y_s, x_s] = v
                    self._volcart_texture_prediction_image_written_to = True
                else:
                    raise ValueError(
                        f"Unknown label_type: {label_type} used for prediction"
                    )

    def write_predictions(self, directory, suffix):
        """Write the buffered prediction images to disk."""
        if not os.path.exists(directory):
            os.makedirs(directory)

        filename_base = os.path.join(
            directory, "{}_prediction_{}_".format(self.name, suffix)
        )
        if self._ink_classes_prediction_image_written_to:
            im = Image.fromarray(self._ink_classes_prediction_image)
            im.save(filename_base + "ink_classes.png")
        if self._rgb_values_prediction_image_written_to:
            im = Image.fromarray(self._rgb_values_prediction_image)
            im.save(filename_base + "rgb_values.png")
        if self._volcart_texture_prediction_image_written_to:
            im = Image.fromarray(self._volcart_texture_prediction_image)
            im.save(filename_base + "volcart_texture.png")

    def reset_predictions(self):
        """Reset the prediction image buffers."""
        self._ink_classes_prediction_image = np.zeros(
            (self._ppm.height, self._ppm.width), np.uint16
        )
        self._ink_classes_prediction_image_written_to = False
        self._rgb_values_prediction_image = np.zeros(
            (self._ppm.height, self._ppm.width, 3), np.uint8
        )
        self._rgb_values_prediction_image_written_to = False
        self._volcart_texture_prediction_image = np.zeros(
            (self._ppm.height, self._ppm.width), np.uint16
        )
        self._volcart_texture_prediction_image_written_to = False

    def write_ambiguous_labels_diagnostic_mask(self, directory):
        if self._points_list_needs_update:
            self.update_points_list()
        # Overlay on ink label image for diagnostic purposes
        overlay = Image.fromarray(self.sampler.ambiguous_labels_mask).convert("RGBA")
        # Set alpha to 0 everywhere so the black regions don't mask the ink labels later
        overlay.putalpha(0)
        overlay_data = np.array(overlay)
        red, green, blue, _ = overlay_data.T
        white_areas = (red == 255) & (blue == 255) & (green == 255)
        overlay_data[...][white_areas.T] = (255, 0, 0, 128)  # Transpose back needed
        overlay = Image.fromarray(overlay_data)
        ink_label_diagnostic_img = (
            Image.fromarray(self._ink_label).copy().convert("RGBA")
        )
        ink_label_diagnostic_img.alpha_composite(overlay)
        ink_label_diagnostic_img.save(
            os.path.join(directory, f"{self.name}_ink_label_diagnostic.png")
        )


class VolumeSource(DataSource):
    """A volume data source generates subvolumes and possibly labels from anywhere in a volume.

    The points are not restricted to a particular surface, PPM, or segmentation.

    """

    def __init__(self, path: str) -> None:
        super().__init__(path)

        self.volume: inkid.data.Volume = inkid.data.Volume.from_path(
            self.source_json["volume"]
        )

        # This volume generates points, here we create the empty list
        self._points = list()

    def __len__(self):
        # Could be anything, we can keep sampling all day. PyTorch requires a finite value, so here's one.
        # This could either be changed or one could combine the settings for multiple epochs and limited
        # training samples.
        return 10000000

    def __getitem__(self, _):
        # Random 3d position
        shape = self.volume.shape()
        x = random.random() * shape[2]
        y = random.random() * shape[1]
        z = random.random() * shape[0]

        # Random 3d direction https://math.stackexchange.com/a/44691
        theta = random.random() * 2 * math.pi
        zed = random.random() * 2 - 1
        n_x = math.sqrt(1 - zed**2) * math.cos(theta)
        n_y = math.sqrt(1 - zed**2) * math.sin(theta)
        n_z = zed

        # Get the feature metadata (useful for e.g. knowing where this feature came from on the surface)
        feature_metadata = FeatureMetadata(self.path, -1, -1, x, y, z, n_x, n_y, n_z)
        # Get the feature
        feature = self.volume.get_subvolume(
            center=(x, y, z), normal=(n_x, n_y, n_z), **self.feature_args
        )
        item = {
            "feature_metadata": feature_metadata,
            "feature": feature,
        }
        return item


def flatten_data_sources_list(source_paths: List[str]) -> List[str]:
    """Expand list of .txt and .json filenames into flattened list of .json filenames.

    The file paths in the input can point to either .txt or .json files. The .txt
    files are themselves lists of other files, which can further be .txt or .json.
    This function goes through this list, and for any .txt file it reads the list
    that file contains and recursively processes it. The result is a list of only
    .json data source file paths.

    Args:
        source_paths (List[str]): A list of .txt dataset or .json data source file paths.

    Returns:
        List[str]: A list of absolute paths to the .json data source files representing this dataset.

    """
    expanded_paths: List[str] = list()
    for source_path in source_paths:
        file_extension = os.path.splitext(source_path)[1]
        if file_extension == ".json":
            expanded_paths.append(source_path)
        elif file_extension == ".txt":
            with open(source_path, "r") as f:
                sources_in_file = f.read().splitlines()
            sources_in_file = [
                os.path.join(os.path.dirname(source_path), s) for s in sources_in_file
            ]
            expanded_paths += flatten_data_sources_list(sources_in_file)
        else:
            raise ValueError(
                f"Data source {source_path} is not a permitted file type (.txt or .json)"
            )
    return list(
        dict.fromkeys(expanded_paths)
    )  # Remove duplicates, keep order https://stackoverflow.com/a/17016257


class Dataset(torch.utils.data.Dataset):
    """A PyTorch Dataset to serve inkid features and labels.

    An inkid dataset is a PyTorch Dataset which maintains a set of inkid data
    sources. Each source generates features (e.g. subvolumes) and possibly labels
    (e.g. ink presence). A PyTorch Dataloader can be created from this Dataset,
    which allows direct input to a model.

    """

    def __init__(self, source_paths: List[str], lazy_load: bool = False) -> None:
        """Initialize the dataset given .json data source and/or .txt dataset paths.

        This recursively expands any provided .txt dataset files until there is just a
        list of absolute paths to .json data source files. InkidDataSource objects are
        then instantiated.

        Args:
            source_paths: A list of .txt dataset or .json data source file paths.

        """
        source_paths = flatten_data_sources_list(source_paths)
        self.sources: List[DataSource] = list()
        for source_path in source_paths:
            self.sources.append(DataSource.from_path(source_path, lazy_load=lazy_load))

    def __len__(self) -> int:
        return sum([len(source) for source in self.sources])

    def __getitem__(self, idx: int):
        for source in self.sources:
            source_len = len(source)
            if idx < source_len:
                return source[idx]
            idx -= source_len
        raise IndexError

    def regions(self) -> list[RegionSource]:
        return [source for source in self.sources if isinstance(source, RegionSource)]

    def volumes(self) -> list[VolumeSource]:
        return [source for source in self.sources if isinstance(source, VolumeSource)]

    def pop_nth_region(self, n: int) -> RegionSource:
        region = self.regions()[n]
        for i, source in enumerate(self.sources):
            if source.path == region.path:
                return self.sources.pop(i)
        raise ValueError("No source found with same path as desired region.")

    def source(self, source_path: str) -> Optional[DataSource]:
        for s in self.sources:
            if s.path == source_path:
                return s
        return None

    def source_json(self) -> dict:
        return {source.path: source.source_json for source in self.sources}
