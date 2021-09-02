# For DataSource.from_path() https://stackoverflow.com/a/33533514
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import namedtuple
import json
import os
from typing import Dict, List, Optional, Tuple

import jsonschema
import numpy as np
from PIL import Image
import torch

import inkid

FeatureMetadata = namedtuple(
    'FeatureMetadata',
    ('path', 'surface_x', 'surface_y', 'x', 'y', 'z', 'n_x', 'n_y', 'n_z'),
    defaults=None,
)


class DataSource(ABC):
    """Can be either a region or a volume. Produces inputs (e.g. subvolumes) and possibly labels."""

    def __init__(self, path: str) -> None:
        self.path = path
        source_file_contents, relative_url = inkid.ops.get_raw_data_from_file_or_url(path, return_relative_url=True)
        source_json: Dict = json.load(source_file_contents)
        # Validate JSON fits schema
        jsonschema.validate(source_json, inkid.ops.json_schema('dataSource0.1'))
        # Save original source
        self.source_json: Dict = source_json.copy()
        # Normalize paths in JSON
        for key in ['volume', 'ppm', 'mask', 'ink_label', 'rgb_label', 'volcart_texture_label']:
            if key in source_json:
                source_json[key] = inkid.ops.normalize_path(source_json[key], relative_url)
        self.data_dict: Dict = source_json

        self.feature_args: Dict = dict()

        self.label_types: List[str] = []
        self.label_args: Dict = dict()

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError

    @staticmethod
    def from_path(path: str) -> DataSource:
        """Check first whether this is a region or volume data source, then instantiate accordingly.

        Also checks to make sure the old region set file format was not provided. If it was,
        a message is issued telling the user how to update the file format.

        """
        with open(path, 'r') as f:
            source_json = json.load(f)
        if 'ppms' in source_json and 'regions' in source_json:
            raise ValueError(f'Source file {path} uses the deprecated region set file format. '
                             f'Please convert it to the updated data source format. '
                             f'The following script can be used: \n\n'
                             f'\tpython inkid/scripts/update_data_file.py {path}')
        if source_json.get('type') == 'region':
            return RegionSource(path)
        elif source_json.get('type') == 'volume':
            return VolumeSource(path)
        else:
            raise ValueError(f'Source file {path} does not specify valid "type" of "region" or "volume"')


class RegionSource(DataSource):
    """A region source generates subvolumes and labels from a region (bounding box) on a PPM.

    (x, y) are always in the PPM space, not this region's bounding box space.

    """

    def __init__(self, path: str) -> None:
        super().__init__(path)

        # Initialize region's PPM, volume, etc
        self._ppm: inkid.data.PPM = inkid.data.PPM.from_path(self.data_dict['ppm'])
        self.volume: inkid.data.Volume = inkid.data.Volume.from_path(self.data_dict['volume'])
        self.bounding_box: Tuple[int, int, int, int] = self.data_dict['bounding_box'] or self.get_default_bounds()
        self._invert_normals: bool = self.data_dict['invert_normals']

        # Mask and label images
        self._mask, self._ink_label, self._rgb_label, self._volcart_texture_label = None, None, None, None
        if self.data_dict['mask'] is not None:
            self._mask = np.array(Image.open(self.data_dict['mask']))
        if self.data_dict['ink_label'] is not None:
            self._ink_label = np.array(Image.open(self.data_dict['ink_label']))
        if self.data_dict['rgb_label'] is not None:
            self._rgb_label = np.array(Image.open(self.data_dict['rgb_label']))
        if self.data_dict['volcart_texture_label'] is not None:
            im = Image.open(self.data_dict['volcart_texture_label'])
            self._volcart_texture_label = np.array(im).astype(np.float32)
            # Assuming image is uint16 data but Pillow loads it as uint32 ('I')
            assert im.mode == 'I'
            # Make sure the data isn't actually greater than uint16
            assert np.amax(self._volcart_texture_label) <= np.iinfo(np.uint16).max
            # Normalize to [0.0, 1.0]
            self._volcart_texture_label /= np.iinfo(np.uint16).max

        # This region generates points, here we create the empty list
        self._points = list()
        # Mark that this list needs updating so that it will be filled before being accessed
        self._points_list_needs_update: bool = True

        self.grid_spacing = 1
        self.specify_inkness = None

        # Prediction images
        self._ink_classes_prediction_image = np.zeros((self._ppm.height, self._ppm.width), np.uint16)
        self._ink_classes_prediction_image_written_to = False
        self._rgb_values_prediction_image = np.zeros((self._ppm.height, self._ppm.width, 3), np.uint8)
        self._rgb_values_prediction_image_written_to = False
        self._volcart_texture_prediction_image = np.zeros((self._ppm.height, self._ppm.width), np.uint16)
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
        feature_metadata = FeatureMetadata(self.path, surface_x, surface_y, x, y, z, n_x, n_y, n_z)
        # Get the feature
        feature = self.volume.get_subvolume(
            center=(x, y, z),
            normal=(n_x, n_y, n_z),
            **self.feature_args
        )
        item = {
            'feature_metadata': feature_metadata,
            'feature': feature,
        }
        # Get the label
        if 'ink_classes' in self.label_types:
            item['ink_classes'] = self.point_to_ink_classes_label(
                (surface_x, surface_y),
                **self.label_args['ink_classes']
            )
        if 'rgb_values' in self.label_types:
            item['rgb_values'] = self.point_to_rgb_values_label(
                (surface_x, surface_y),
                **self.label_args['rgb_values']
            )
        if 'volcart_texture' in self.label_types:
            item['volcart_texture'] = self.point_to_volcart_texture_label(
                (surface_x, surface_y),
                **self.label_args['volcart_texture']
            )
        return item

    def update_points_list(self) -> None:
        """Update the list of points after changes to the bounding box, grid spacing, or some other options."""
        self._points = list()
        x0, y0, x1, y1 = self.bounding_box
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
        return torch.Tensor(label)

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

    def point_to_volcart_texture_label(self, point, shape):
        assert self._volcart_texture_label is not None
        x, y = point
        label = np.zeros((1,) + shape).astype(np.float32)
        y_d, x_d = np.array(shape) // 2  # Calculate distance from center to edges of square we are sampling
        # Iterate over label indices
        for idx, _ in np.ndenumerate(label):
            _, y_idx, x_idx = idx
            y_s = y - y_d + y_idx  # Sample point is center minus distance (half edge length) plus label index
            x_s = x - x_d + x_idx
            # Bounds check to make sure inside PPM
            if 0 <= y_s < self._volcart_texture_label.shape[0] and 0 <= x_s < self._volcart_texture_label.shape[1]:
                label[0, y_idx, x_idx] = self._volcart_texture_label[y_s, x_s]
        return label

    def store_prediction(self, x, y, prediction, label_type):
        """Store an incoming prediction in the corresponding prediction image buffer.

        The incoming predictions have shape [d, h, w]. When h and w are both 1, a prediction has been provided for
        a single point rather than a rectangular region.

        """
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
                elif label_type == 'volcart_texture':
                    v = value[0] * np.iinfo(np.uint16).max
                    self._volcart_texture_prediction_image[y_s, x_s] = v
                    self._volcart_texture_prediction_image_written_to = True
                else:
                    raise ValueError(f'Unknown label_type: {label_type} used for prediction')

    def save_predictions(self, directory, suffix):
        """Write the buffered prediction images to disk."""
        if not os.path.exists(directory):
            os.makedirs(directory)

        filename_base = os.path.join(directory, '{}_prediction_{}_'.format(self.name, suffix))
        if self._ink_classes_prediction_image_written_to:
            im = Image.fromarray(self._ink_classes_prediction_image)
            im.save(filename_base + 'ink_classes.png')
        if self._rgb_values_prediction_image_written_to:
            im = Image.fromarray(self._rgb_values_prediction_image)
            im.save(filename_base + 'rgb_values.png')
        if self._volcart_texture_prediction_image_written_to:
            im = Image.fromarray(self._volcart_texture_prediction_image)
            im.save(filename_base + 'volcart_texture.png')

    def reset_predictions(self):
        """Reset the prediction image buffers."""
        self._ink_classes_prediction_image = np.zeros((self._ppm.height, self._ppm.width), np.uint16)
        self._ink_classes_prediction_image_written_to = False
        self._rgb_values_prediction_image = np.zeros((self._ppm.height, self._ppm.width, 3), np.uint8)
        self._rgb_values_prediction_image_written_to = False
        self._volcart_texture_prediction_image = np.zeros((self._ppm.height, self._ppm.width), np.uint16)
        self._volcart_texture_prediction_image_written_to = False


class VolumeSource(DataSource):
    """A volume data source generates subvolumes and possibly labels from anywhere in a volume.

    The points are not restricted to a particular surface, PPM, or segmentation.

    """

    def __init__(self, path: str) -> None:
        super().__init__(path)

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError


class Dataset(torch.utils.data.Dataset):
    """A PyTorch Dataset to serve inkid features and labels.

    An inkid dataset is a PyTorch Dataset which maintains a set of inkid data
    sources. Each source generates features (e.g. subvolumes) and possibly labels
    (e.g. ink presence). A PyTorch Dataloader can be created from this Dataset,
    which allows direct input to a model.

    """

    def __init__(self, source_paths: List[str], feature_args: Dict = None, label_types: Dict = None,
                 label_args: Dict = None) -> None:
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

        self._set_for_all_sources('feature_args', feature_args)
        if label_types is not None:
            self._set_for_all_sources('label_types', label_types)
        if label_args is not None:
            self._set_for_all_sources('label_args', label_args)

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
        raise IndexError

    def regions(self) -> List[RegionSource]:
        return [source for source in self.sources if isinstance(source, RegionSource)]

    def get_source(self, source_path: str) -> Optional[DataSource]:
        for source in self.sources:
            if source.path == source_path:
                return source
        return None

    def pop_source(self, source_path: str) -> DataSource:
        source_idx_to_remove: int = self.source_paths().index(source_path)
        return self.sources.pop(source_idx_to_remove)

    def source_paths(self) -> List[str]:
        return [source.path for source in self.sources]

    def data_dict(self):
        return {source.path: source.data_dict for source in self.sources}

    def _set_for_all_sources(self, attribute: str, value):
        for source in self.sources:
            setattr(source, attribute, value)

    def save_predictions(self, directory, suffix):
        for region in self.regions():
            region.save_predictions(directory, suffix)

    def reset_predictions(self):
        for region in self.regions():
            region.reset_predictions()
