# For DataSource.from_path() https://stackoverflow.com/a/33533514
from __future__ import annotations

from abc import ABC, abstractmethod
import json
import os
from typing import Dict, List, Optional, Tuple

import jsonschema
import numpy as np
from PIL import Image
import torch

import inkid


class DataSource(ABC):
    """Can be either a region or a volume. Produces inputs (e.g. subvolumes) and possibly labels."""

    def __init__(self, path: str) -> None:
        self.path = path
        source_file_contents, relative_url = inkid.ops.get_raw_data_from_file_or_url(path, return_relative_url=True)
        source_json = json.load(source_file_contents)
        # Validate JSON fits schema
        jsonschema.validate(source_json, inkid.ops.json_schema('dataSource0.1'))
        # Normalize paths in JSON
        for key in ['volume', 'ppm', 'mask', 'ink_label', 'rgb_label', 'volcart_texture_label']:
            if key in source_json:
                source_json[key] = inkid.ops.normalize_path(source_json[key], relative_url)
        self._source_json = source_json

        self.feature_type: Optional[str] = None
        self.feature_args: Dict = dict()

        self.label_type: Optional[str] = None
        self.label_args: Dict = dict()

    def data_dict(self):
        return self._source_json

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, item):
        pass

    @staticmethod
    def from_path(path: str) -> DataSource:
        with open(path, 'r') as f:
            source_json = json.load(f)
        if source_json.get('type') == 'region':
            return RegionSource(path)
        elif source_json.get('type') == 'volume':
            return VolumeSource(path)
        else:
            raise ValueError(f'Source file {path} does not specify valid "type" of "region" or "volume"')


RegionPoint = Tuple[str, int, int]


class RegionSource(DataSource):
    def __init__(self, path: str) -> None:
        """TODO

        (x, y) are always in PPM space not this region space

        """
        super().__init__(path)

        self._ppm: inkid.data.PPM = inkid.data.PPM.from_path(self._source_json['ppm'])
        self._volume: inkid.data.Volume = inkid.data.Volume.from_path(self._source_json['volume'])
        self._bounding_box: Optional[Tuple[int, int, int, int]] = self._source_json['bounding_box']
        if self._bounding_box is None:
            self._bounding_box = self.get_default_bounds()
        self._invert_normals: bool = self._source_json['invert_normals']

        self._mask, self._ink_label, self._rgb_label, self._volcart_texture_label = None, None, None, None
        if self._source_json['mask'] is not None:
            self._mask = np.array(Image.open(self._source_json['mask']))
        if self._source_json['ink_label'] is not None:
            self._ink_label = np.array(Image.open(self._source_json['ink_label']))
        if self._source_json['rgb_label'] is not None:
            self._rgb_label = np.array(Image.open(self._source_json['rgb_label']))
        if self._source_json['volcart_texture_label'] is not None:
            self._volcart_texture_label = np.array(Image.open(self._source_json['volcart_texture_label']))

        self._points = list()
        self._points_list_needs_update: bool = True

        self.grid_spacing = 1
        self.specify_inkness = None

        self._ink_classes_prediction_image = np.zeros((self._ppm.height, self._ppm.width), np.uint16)
        self._ink_classes_prediction_image_written_to = False
        self._rgb_values_prediction_image = np.zeros((self._ppm.height, self._ppm.width, 3), np.uint8)
        self._rgb_values_prediction_image_written_to = False

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
        # Get the feature metadata (useful for e.g. knowing where this feature came from on the surface)
        feature_metadata: RegionPoint = (self.path, surface_x, surface_y)
        # Get the feature
        feature = None
        if self.feature_type == 'subvolume_3dcnn':
            feature = self._volume.get_subvolume(
                center=(x, y, z),
                normal=(n_x, n_y, n_z),
                **self.feature_args
            )
        elif self.feature_type == 'voxel_vector_1dcnn':
            feature = self._volume.get_voxel_vector(
                center=(x, y, z),
                normal=(n_x, n_y, n_z),
                **self.feature_args
            )
        elif self.feature_type == 'descriptive_statistics':
            subvolume = self._volume.get_subvolume(
                center=(x, y, z),
                normal=(n_x, n_y, n_z),
                **self.feature_args
            )
            feature = inkid.ops.get_descriptive_statistics(subvolume)
        elif self.feature_type is not None:
            raise ValueError(f'Unknown feature_type: {self.feature_type} set for InkidRegionSource'
                             f' {self.path}')
        # Get the label
        label = None
        if self.label_type == 'ink_classes':
            label = self.point_to_ink_classes_label(
                (surface_x, surface_y),
                **self.label_args
            )
        elif self.label_type == 'rgb_values':
            label = self.point_to_rgb_values_label(
                (surface_x, surface_y),
                **self.label_args
            )
        elif self.label_type is not None:
            raise ValueError(f'Unknown label_type: {self.label_type} set for InkidRegionSource'
                             f' {self.path}')
        if label is None:
            return feature_metadata, feature
        else:
            return feature_metadata, feature, label

    def update_points_list(self) -> None:
        self._points = list()
        x0, y0, x1, y1 = self._bounding_box
        for y in range(y0, y1, self.grid_spacing):
            for x in range(x0, x1, self.grid_spacing):
                if self.specify_inkness is not None:
                    if self.specify_inkness and not self.is_ink(x, y):
                        continue
                    elif not self.specify_inkness and self.is_ink(x, y):
                        continue
                if self.is_on_surface(x, y):
                    self._points.append((x, y))
        self._points_list_needs_update = False

    @property
    def grid_spacing(self):
        return self._grid_spacing

    @grid_spacing.setter
    def grid_spacing(self, spacing: int):
        self._grid_spacing = spacing
        self._points_list_needs_update = True

    @property
    def specify_inkness(self):
        return self._specify_inkness

    @specify_inkness.setter
    def specify_inkness(self, inkness: Optional[bool]):
        self._specify_inkness = inkness
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
        square = self._mask[y - r:y + r + 1, x - r:x + r + 1]
        return np.size(square) > 0 and np.min(square) != 0

    def get_default_bounds(self) -> Tuple[int, int, int, int]:
        """Return the full bounds of the PPM in (x0, y0, x1, y1) format."""
        return 0, 0, self._ppm.width, self._ppm.height

    def point_to_ink_classes_label(self, point, shape):
        assert self._ink_label is not None
        x, y = point
        label = np.stack((np.ones(shape), np.zeros(shape))).astype(np.float32)  # Create array of no-ink labels
        y_d, x_d = np.array(shape) // 2  # Calculate distance from center to edges of square we are sampling
        # Iterate over label indices
        for idx, _ in np.ndenumerate(label):
            _, y_idx, x_idx = idx
            y_s = y - y_d + y_idx  # Sample point is center minus distance (half edge length) plus label index
            x_s = x - x_d + x_idx
            # Bounds check to make sure inside PPM
            if 0 <= y_s < self._ink_label.shape[0] and 0 <= x_s < self._ink_label.shape[1]:
                if self._ink_label[y_s, x_s] != 0:
                    label[:, y_idx, x_idx] = [0.0, 1.0]  # Mark this "ink"
        return label

    def point_to_rgb_values_label(self, point, shape):
        assert self._rgb_label is not None
        x, y = point
        label = np.zeros((3,) + shape).astype(np.float32)
        y_d, x_d = np.array(shape) // 2  # Calculate distance from center to edges of square we are sampling
        # Iterate over label indices
        for idx, _ in np.ndenumerate(label):
            _, y_idx, x_idx = idx
            y_s = y - y_d + y_idx  # Sample point is center minus distance (half edge length) plus label index
            x_s = x - x_d + x_idx
            # Bounds check to make sure inside PPM
            if 0 <= y_s < self._rgb_label.shape[0] and 0 <= x_s < self._rgb_label.shape[1]:
                label[:, y_idx, x_idx] = self._rgb_label[y_s, x_s]
        return label

    def store_prediction(self, x, y, prediction, label_type):
        """TODO prediction shape [v, y, x]"""
        # Repeat prediction to fill grid square so prediction image is not single pixels in sea of blackness
        if prediction.shape[1] == 1 and prediction.shape[2] == 1:
            prediction = np.repeat(prediction, repeats=self.grid_spacing, axis=1)
            prediction = np.repeat(prediction, repeats=self.grid_spacing, axis=2)
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
                if label_type == 'ink_classes':
                    # Convert ink class probability to image intensity
                    v = value[1] * np.iinfo(np.uint16).max
                    self._ink_classes_prediction_image[y_s, x_s] = v
                    self._ink_classes_prediction_image_written_to = True
                elif label_type == 'rgb_values':
                    # Restrict value to uint8 range
                    v = np.clip(value, 0, np.iinfo(np.uint8).max)
                    self._rgb_values_prediction_image[y_s, x_s] = v
                    self._rgb_values_prediction_image_written_to = True
                else:
                    raise ValueError(f'Unknown label_type: {label_type} used for prediction')

    def save_predictions(self, directory, suffix):
        if not os.path.exists(directory):
            os.makedirs(directory)

        im = None
        if self._ink_classes_prediction_image_written_to:
            im = Image.fromarray(self._ink_classes_prediction_image)
        elif self._rgb_values_prediction_image_written_to:
            im = Image.fromarray(self._rgb_values_prediction_image)
        if im is not None:
            im.save(
                os.path.join(
                    directory,
                    '{}_prediction_{}.png'.format(
                        self.name,
                        suffix,
                    ),
                ),
            )

    def reset_predictions(self):
        self._ink_classes_prediction_image = np.zeros((self._ppm.height, self._ppm.width), np.uint16)
        self._ink_classes_prediction_image_written_to = False
        self._rgb_values_prediction_image = np.zeros((self._ppm.height, self._ppm.width, 3), np.uint8)
        self._rgb_values_prediction_image_written_to = False


class VolumeSource(DataSource):
    def __init__(self, path: str) -> None:
        super().__init__(path)

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass


class Dataset(torch.utils.data.Dataset):
    """A PyTorch Dataset to serve inkid features and labels.

        An inkid dataset is a PyTorch Dataset which maintains a set of inkid data
        sources. Each source generates features (e.g. subvolumes) and possibly labels
        (e.g. ink presence). A PyTorch Dataloader can be created from this Dataset,
        which allows direct input to a model.

    """
    def __init__(self, source_paths: List[str]) -> None:
        """Initialize the dataset given .json data source and/or .txt dataset paths.

        This recursively expands any provided .txt dataset files until there is just a
        list of absolute paths to .json data source files. InkidDataSource objects are
        then instantiated.

        Args:
            source_paths: A list of .txt dataset or .json data source file paths.
        """
        # Convert the list of paths to a list of InkidDataSources
        source_paths = self.expand_data_sources(source_paths)
        self.sources: List[DataSource] = list()
        for source_path in source_paths:
            self.sources.append(DataSource.from_path(source_path))

    def expand_data_sources(self, source_paths: List[str]) -> List[str]:
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
            if file_extension == '.json':
                expanded_paths.append(source_path)
            elif file_extension == '.txt':
                with open(source_path, 'r') as f:
                    sources_in_file = f.read().splitlines()
                sources_in_file = [os.path.join(os.path.dirname(source_path), s) for s in sources_in_file]
                expanded_paths += self.expand_data_sources(sources_in_file)
            else:
                raise ValueError(f'Data source {source_path} is not a permitted file type (.txt or .json)')
        return list(dict.fromkeys(expanded_paths))  # Remove duplicates, keep order https://stackoverflow.com/a/17016257

    def set_regions_grid_spacing(self, spacing: int):
        for source in self.sources:
            if isinstance(source, RegionSource):
                source.grid_spacing = spacing

    def __len__(self) -> int:
        return sum([len(source) for source in self.sources])

    def __getitem__(self, item: int):
        for source in self.sources:
            source_len = len(source)
            if item < source_len:
                return source[item]
            item -= source_len
        raise IndexError()

    def regions(self) -> List[RegionSource]:
        return [source for source in self.sources if isinstance(source, RegionSource)]

    def get_source(self, source_path: str) -> Optional[DataSource]:
        for source in self.sources:
            if source.path == source_path:
                return source
        return None

    def remove_source(self, source_path: str) -> None:
        source_idx_to_remove: int = self.source_paths().index(source_path)
        self.sources.pop(source_idx_to_remove)

    def source_paths(self) -> List[str]:
        return [source.path for source in self.sources]

    def data_dict(self):
        return {source.path: source.data_dict() for source in self.sources}

    def set_for_all_sources(self, attribute: str, value):
        for source in self.sources:
            setattr(source, attribute, value)

    def save_predictions(self, directory, suffix):
        for region in self.regions():
            region.save_predictions(directory, suffix)

    def reset_predictions(self):
        for region in self.regions():
            region.reset_predictions()
