"""Train and predict using subvolumes.

This script will read a RegionSet JSON file and create a RegionSet for
training, validation, and prediction. It will then run the training
process, validating and predicting along the way as defined by the
RegionSet and the parameters file.

The optional value k can be passed in order to use this script for
k-fold cross validation (and prediction in this case). To do that,
create a RegionSet of entirely training regions, and then pass an
index k to this script via the command line argument. It will take the
kth training region, remove it from the training set, and add it to
the prediction and validation sets for that run. TODO update

"""
# For InkidDataSource.from_path() https://stackoverflow.com/a/33533514
from __future__ import annotations

from abc import ABC, abstractmethod
import argparse
import datetime
import itertools
import json
import logging
import multiprocessing
import os
import sys
import time
import timeit
from typing import Dict, List, Optional, Tuple

import git
import kornia
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchsummary
from tqdm import tqdm

import inkid


class InkidDataSource(ABC):
    """Can be either a region or a volume. Produces inputs (e.g. subvolumes) and possibly labels."""
    def __init__(self, path: str) -> None:
        self.path = path
        with open(path, 'r') as f:
            source_json = json.load(f)
        self._source_json = source_json

        self.feature_type: Optional[str] = None
        self.feature_args: Dict = dict()

        self.label_type: Optional[str] = None
        self.label_args: Dict = dict()

    def data_dict(self):
        return self._source_json

    def make_path_absolute(self, path: str) -> str:
        return os.path.abspath(os.path.join(os.path.dirname(self.path), path))

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, item):
        pass

    @staticmethod
    def from_path(path: str) -> InkidDataSource:
        with open(path, 'r') as f:
            source_json = json.load(f)
        if source_json.get('type') == 'region':
            return InkidRegionSource(path)
        elif source_json.get('type') == 'volume':
            return InkidVolumeSource(path)
        else:
            raise ValueError(f'Source file {path} does not specify valid "type" of "region" or "volume"')


RegionPoint = Tuple[str, int, int]


class InkidRegionSource(InkidDataSource):
    def __init__(self, path: str) -> None:
        """TODO

        (x, y) are always in PPM space not this region space

        """
        super().__init__(path)

        for key in ['ppm', 'volume', 'mask', 'ink-label', 'rgb-label', 'volcart-texture-label']:
            if self._source_json[key] is not None:
                self._source_json[key] = self.make_path_absolute(self._source_json[key])

        self._ppm: inkid.data.PPM = inkid.data.PPM.from_path(self._source_json['ppm'])
        self._volume: inkid.data.Volume = inkid.data.Volume.from_path(self._source_json['volume'])
        self._bounding_box: Optional[Tuple[int, int, int, int]] = self._source_json['bounding-box']
        if self._bounding_box is None:
            self._bounding_box = self.get_default_bounds()
        self._invert_normal: bool = self._source_json['invert-normal']

        self._mask, self._ink_label, self._rgb_label, self._volcart_texture_label = None, None, None, None
        if self._source_json['mask'] is not None:
            self._mask = np.array(Image.open(self._source_json['mask']))
        if self._source_json['ink-label'] is not None:
            self._ink_label = np.array(Image.open(self._source_json['ink-label']))
        if self._source_json['rgb-label'] is not None:
            self._rgb_label = np.array(Image.open(self._source_json['rgb-label']))
        if self._source_json['volcart-texture-label'] is not None:
            self._volcart_texture_label = np.array(Image.open(self._source_json['volcart-texture-label']))

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
        square = self._mask[y-r:y+r+1, x-r:x+r+1]
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


class InkidVolumeSource(InkidDataSource):
    def __init__(self, path: str) -> None:
        super().__init__(path)

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass


class InkidDataset(torch.utils.data.Dataset):
    """A PyTorch Dataset to serve inkid features and labels.

        An inkid dataset is a PyTorch Dataset which maintains a set of inkid data
        sources. Each source generates features (e.g. subvolumes) and possibly labels
        (e.g. ink presence). A PyTorch Dataloader can be created from this Dataset,
        which allows direct input to a model.

    """
    def __init__(self, source_paths: List[str], data_root: str) -> None:
        """Initialize the dataset given .json data source and/or .txt dataset paths.

        This recursively expands any provided .txt dataset files until there is just a
        list of absolute paths to .json data source files. InkidDataSource objects are
        then instantiated.

        Args:
            source_paths: A list of .txt dataset or .json data source file paths.
            data_root: The Invisible Library root data directory.

        """
        self.data_root = data_root

        # Convert the list of paths to a list of InkidDataSources
        source_paths = self.expand_data_sources(source_paths)
        self.sources: List[InkidDataSource] = list()
        for source_path in source_paths:
            self.sources.append(InkidDataSource.from_path(source_path))

    def expand_data_sources(self, source_paths: List[str], recursing: bool = False) -> List[str]:
        """Expand list of .txt and .json filenames into flattened list of .json filenames.

        The file paths in the input can point to either .txt or .json files. The .txt
        files are themselves lists of other files, which can further be .txt or .json.
        This function goes through this list, and for any .txt file it reads the list
        that file contains and recursively processes it. The result is a list of only
        .json data source file paths.

        Args:
            source_paths (List[str]): A list of .txt dataset or .json data source file paths.
            recursing: (bool): Whether or not this is a recursive call. This affects whether the paths
                in source_paths are expected to be absolute (when not recursive, since they were passed
                from command line) or relative to self.data_root (when recursive, since they were read
                from a .txt dataset file).

        Returns:
            List[str]: A list of absolute paths to the .json data source files representing this dataset.

        """
        expanded_paths: List[str] = list()
        for source_path in source_paths:
            if recursing:
                source_path = os.path.join(self.data_root, source_path)
            file_extension = os.path.splitext(source_path)[1]
            if file_extension == '.json':
                expanded_paths.append(source_path)
            elif file_extension == '.txt':
                with open(source_path, 'r') as f:
                    sources_in_file = f.read().splitlines()
                expanded_paths += self.expand_data_sources(sources_in_file, recursing=True)
            else:
                raise ValueError(f'Data source {source_path} is not a permitted file type (.txt or .json)')
        return list(set(expanded_paths))  # Remove duplicates

    def set_regions_grid_spacing(self, spacing: int):
        for source in self.sources:
            if isinstance(source, InkidRegionSource):
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

    def regions(self) -> List[InkidRegionSource]:
        return [source for source in self.sources if isinstance(source, InkidRegionSource)]

    def get_source(self, source_path: str) -> Optional[InkidDataSource]:
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


def perform_validation(model, dataloader, metrics, device, label_type):
    """Run the validation process using a model and dataloader, and return the results of all metrics."""
    model.eval()  # Turn off training mode for batch norm and dropout purposes
    with torch.no_grad():
        metric_results = {metric: [] for metric in metrics}
        for _, xb, yb in tqdm(dataloader):
            pred = model(xb.to(device))
            yb = yb.to(device)
            if label_type == 'ink_classes':
                _, yb = yb.max(1)  # Argmax
            for metric, fn in metrics.items():
                metric_results[metric].append(fn(pred, yb))
    model.train()
    return metric_results


def generate_prediction_image(dataloader, model, output_size, label_type, device, predictions_dir, suffix,
                              prediction_averaging):
    """Helper function to generate a prediction image given a model and dataloader, and save it to a file."""
    model.eval()  # Turn off training mode for batch norm and dropout purposes
    with torch.no_grad():
        for batch_metadata, batch_features in tqdm(dataloader):
            # Smooth predictions via augmentation. Augment each subvolume 8-fold via rotations and flips
            if prediction_averaging:
                rotations = range(4)
                flips = [False, True]
            else:
                rotations = [0]
                flips = [False]
            batch_preds = None
            for rotation, flip in itertools.product(rotations, flips):
                # Example batch_features.shape = [64, 1, 48, 48, 48] (BxCxDxHxW)
                # Augment via rotation and flip
                aug_pxb = batch_features.rot90(rotation, [3, 4])
                if flip:
                    aug_pxb = aug_pxb.flip(4)
                pred = model(aug_pxb.to(device))
                if label_type == 'ink_classes':
                    pred = F.softmax(pred, dim=1)
                pred = pred.cpu()
                # Example pred.shape = [64, 2, 48, 48] (BxCxHxW)
                # Undo flip and rotation
                if flip:
                    pred = pred.flip(3)
                pred = pred.rot90(-rotation, [2, 3])
                pred = np.expand_dims(pred.numpy(), axis=0)
                # Example pred.shape = [1, 64, 2, 48, 48] (BxCxHxW)
                # Save this augmentation to the batch totals
                if batch_preds is None:
                    batch_preds = np.zeros((0, batch_features.shape[0], output_size, pred.shape[3], pred.shape[4]))
                batch_preds = np.append(batch_preds, pred, axis=0)
            # Average over batch of predictions after augmentation
            batch_pred = batch_preds.mean(0)
            # Separate these three lists
            source_paths, xs, ys = batch_metadata
            for prediction, source_path, x, y in zip(batch_pred, source_paths, xs, ys):
                dataloader.dataset.get_source(source_path).store_prediction(
                    int(x),
                    int(y),
                    prediction,
                    label_type
                )
    dataloader.dataset.save_predictions(predictions_dir, suffix)
    dataloader.dataset.reset_predictions()
    model.train()


def main():
    """Run the training and prediction process."""
    start = timeit.default_timer()

    parser = argparse.ArgumentParser(description=__doc__)

    # Needed files
    parser.add_argument('output', metavar='path', help='output directory')
    parser.add_argument('--training-set', metavar='path', nargs='*', help='training dataset(s)', default=[])
    parser.add_argument('--validation-set', metavar='path', nargs='*', help='validation dataset(s)', default=[])
    parser.add_argument('--prediction-set', metavar='path', nargs='*', help='prediction dataset(s)', default=[])
    parser.add_argument('--data-root', metavar='path', default=None,
                        help='path to the root directory that contains the .volpkgs, etc.')

    # Dataset modifications
    parser.add_argument('--cross-validate-on', metavar='n', default=None, type=int,
                        help='remove the nth source from the flattened set of all training data sources, and '
                             'add this set to the validation and prediction sets')

    # Method
    parser.add_argument('--feature-type', default='subvolume_3dcnn', help='type of input features',
                        choices=['subvolume_3dcnn', 'voxel_vector_1dcnn', 'descriptive_statistics'])
    parser.add_argument('--label-type', default='ink_classes', help='type of labels',
                        choices=['ink_classes', 'rgb_values'])
    parser.add_argument('--model-3d-to-2d', action='store_true',
                        help='Use 2d labels per subvolume rather than single value')
    parser.add_argument('--loss', default='cross_entropy', choices=['cross_entropy', 'dice', 'tversky', 'focal'])
    parser.add_argument('--tversky-loss-alpha', metavar='n', type=float, default=0.5)
    parser.add_argument('--focal-loss-alpha', metavar='n', type=float, default=0.5)

    # Subvolumes
    inkid.ops.add_subvolume_args(parser)

    # Voxel vectors
    parser.add_argument('--length-in-each-direction', metavar='n', type=int, default=8,
                        help='length of voxel vector in each direction along normal')

    # Data organization/augmentation
    parser.add_argument('--jitter-max', metavar='n', type=int, default=4)
    parser.add_argument('--no-augmentation', action='store_true')

    # Network architecture
    parser.add_argument('--model', default='original', help='model to run against',
                        choices=['original', '3dunet_full', '3dunet_half'])
    parser.add_argument('--learning-rate', metavar='n', type=float, default=0.001)
    parser.add_argument('--drop-rate', metavar='n', type=float, default=0.5)
    parser.add_argument('--batch-norm-momentum', metavar='n', type=float, default=0.9)
    parser.add_argument('--no-batch-norm', action='store_true')
    parser.add_argument('--filters', metavar='n', nargs='*', type=int, default=[32, 16, 8, 4],
                        help='number of filters for each convolution layer')
    parser.add_argument('--unet-starting-channels', metavar='n', type=int, default=32,
                        help='number of channels to start with in 3D-UNet')
    parser.add_argument('--load-weights-from', metavar='path', default=None,
                        help='pretrained model checkpoint to initialize network')

    # Run configuration
    parser.add_argument('--batch-size', metavar='n', type=int, default=32)
    parser.add_argument('--training-max-samples', metavar='n', type=int, default=None)
    parser.add_argument('--training-epochs', metavar='n', type=int, default=1)
    parser.add_argument('--prediction-grid-spacing', metavar='n', type=int, default=4,
                        help='prediction points will be taken from an NxN grid')
    parser.add_argument('--prediction-averaging', action='store_true',
                        help='average multiple predictions based on rotated and flipped input subvolumes')
    parser.add_argument('--validation-max-samples', metavar='n', type=int, default=5000)
    parser.add_argument('--summary-every-n-batches', metavar='n', type=int, default=10)
    parser.add_argument('--checkpoint-every-n-batches', metavar='n', type=int, default=5000)
    parser.add_argument('--final-prediction-on-all', action='store_true')
    parser.add_argument('--skip-training', action='store_true')
    parser.add_argument('--dataloaders-num-workers', metavar='n', type=int, default=None)

    # Rclone
    parser.add_argument('--rclone-transfer-remote', metavar='remote', default=None,
                        help='if specified, and if matches the name of one of the directories in '
                             'the output path, transfer the results to that rclone remote into the '
                             'subpath following the remote name')

    args = parser.parse_args()

    # Argument makes more sense as a negative, variable makes more sense as a positive
    args.augmentation = not args.no_augmentation

    # Make sure some sort of input is provided, else there is nothing to do
    if len(args.training_set) == 0 and len(args.prediction_set) == 0 and len(args.validation_set) == 0:
        raise ValueError('At least one of --training-set, --prediction-set, or --validation-set '
                         'must be specified.')

    # If this is part of a cross-validation job, append n (--cross-validate-on) to the output path
    if args.cross_validate_on is None:
        dir_name = datetime.datetime.today().strftime('%Y-%m-%d_%H.%M.%S')
    else:
        dir_name = datetime.datetime.today().strftime('%Y-%m-%d_%H.%M.%S') + '_' + str(args.cross_validate_on)
    output_path = os.path.join(args.output, dir_name)

    # If this is not a cross-validation job, then the defined output dir should be empty.
    # If this is a cross-validation job, that directory is allowed to have output from other
    # cross-validation splits, but not this one
    if os.path.isdir(args.output):
        if args.cross_validate_on is None:
            if len(os.listdir(args.output)) > 0:
                logging.error(f'Provided output directory must be empty: {args.output}')
                return
        else:
            dirs_in_output = [os.path.join(args.output, f) for f in os.listdir(args.output)]
            dirs_in_output = list(filter(os.path.isdir, dirs_in_output))
            for job_dir in dirs_in_output:
                if job_dir.endswith(f'_{args.cross_validate_on}'):
                    logging.error(f'Cross-validation directory for same hold-out set already exists '
                                  f'in output directory: {job_dir}')
                    return

    os.makedirs(output_path)

    # Create TensorBoard writer
    writer = SummaryWriter(os.path.join(output_path, 'tensorboard'))

    # Configure logging
    # noinspection PyArgumentList
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_path, f'{dir_name}.log')),
            logging.StreamHandler()
        ]
    )

    # Automatically increase prediction grid spacing if using 2D labels, and turn off augmentation
    if args.model_3d_to_2d:
        args.prediction_grid_spacing = args.subvolume_shape_voxels[-1]
        args.augmentation = False

    # Define directories for prediction images and checkpoints
    predictions_dir = os.path.join(output_path, 'predictions')
    os.makedirs(predictions_dir)
    checkpoints_dir = os.path.join(output_path, 'checkpoints')
    os.makedirs(checkpoints_dir)

    # Look for the Invisible Library data root directory, if it is not specified
    if args.data_root is None:
        args.data_root = inkid.ops.try_find_data_root()
        if args.data_root is None:
            raise FileNotFoundError('Unable to find Invisible Library data root directory by guessing.'
                                    ' Please specify --data-root.')

    train_ds = InkidDataset(args.training_set, args.data_root)
    val_ds = InkidDataset(args.validation_set, args.data_root)
    pred_ds = InkidDataset(args.prediction_set, args.data_root)

    # If k-fold job, remove nth region from training and put in prediction/validation sets
    if args.cross_validate_on is not None:
        nth_region_path: str = train_ds.regions()[args.cross_validate_on].path
        train_ds.remove_source(nth_region_path)
        val_ds.sources.append(InkidRegionSource(nth_region_path))
        pred_ds.sources.append(InkidRegionSource(nth_region_path))

    pred_ds.set_regions_grid_spacing(args.prediction_grid_spacing)

    # Create metadata dict
    metadata = {
        'Arguments': vars(args),
        'Data': {
            'training': train_ds.data_dict(),
            'validation': val_ds.data_dict(),
            'prediction': pred_ds.data_dict(),
        },
        'Command': ' '.join(sys.argv)
    }

    # Add git hash to metadata if inside a git repository
    try:
        repo = git.Repo(os.path.join(os.path.dirname(inkid.__file__), '..'))
        sha = repo.head.object.hexsha
        metadata['Git hash'] = repo.git.rev_parse(sha, short=6)
    except git.exc.InvalidGitRepositoryError:
        metadata['Git hash'] = 'No git hash available (unable to find valid repository).'

    # Add SLURM info if it exists
    if 'SLURM_JOB_ID' in os.environ:
        metadata['SLURM Job ID'] = os.getenv('SLURM_JOB_ID')
    if 'SLURM_JOB_NAME' in os.environ:
        metadata['SLURM Job Name'] = os.getenv('SLURM_JOB_NAME')

    # Print metadata for logging and diagnostics
    logging.info('\n' + json.dumps(metadata, indent=4, sort_keys=False))

    # Write preliminary metadata to file (will be updated when job completes)
    with open(os.path.join(output_path, 'metadata.json'), 'w') as metadata_file:
        metadata_file.write(json.dumps(metadata, indent=4, sort_keys=False))

    # Define the feature inputs to the network
    if args.feature_type == 'subvolume_3dcnn':
        subvolume_args = dict(
            shape_voxels=args.subvolume_shape_voxels,
            shape_microns=args.subvolume_shape_microns,
            out_of_bounds='all_zeros',
            move_along_normal=args.move_along_normal,
            method=args.subvolume_method,
            normalize=args.normalize_subvolumes,
        )
        train_feature_args = subvolume_args.copy()
        train_feature_args.update(
            augment_subvolume=args.augmentation,
            jitter_max=args.jitter_max,
        )
        val_feature_args = subvolume_args.copy()
        val_feature_args.update(
            augment_subvolume=False,
            jitter_max=0,
        )
        pred_feature_args = val_feature_args.copy()
    elif args.feature_type == 'voxel_vector_1dcnn':
        train_feature_args = dict(
            length_in_each_direction=args.length_in_each_direction,
            out_of_bounds='all_zeros',
        )
        val_feature_args = train_feature_args.copy()
        pred_feature_args = train_feature_args.copy()
    elif args.feature_type == 'descriptive_statistics':
        train_feature_args = dict(
            subvolume_shape_voxels=args.subvolume_shape_voxels,
            subvolume_shape_microns=args.subvolume_shape_microns
        )
        val_feature_args = train_feature_args.copy()
        pred_feature_args = train_feature_args.copy()
    else:
        logging.error('Feature type not recognized: {}'.format(args.feature_type))
        return

    train_ds.set_for_all_sources('feature_type', args.feature_type)
    train_ds.set_for_all_sources('feature_args', train_feature_args)
    val_ds.set_for_all_sources('feature_type', args.feature_type)
    val_ds.set_for_all_sources('feature_args', val_feature_args)
    pred_ds.set_for_all_sources('feature_type', args.feature_type)
    pred_ds.set_for_all_sources('feature_args', pred_feature_args)

    # Define the labels
    if args.model_3d_to_2d:
        label_shape = (args.subvolume_shape_voxels[1], args.subvolume_shape_voxels[2])
    else:
        label_shape = (1, 1)
    if args.label_type == 'ink_classes':
        label_args = dict(
            shape=label_shape
        )
        output_size = 2
        metrics = {
            'loss': {
                'cross_entropy': nn.CrossEntropyLoss(),
                'dice': kornia.losses.DiceLoss(),
                'tversky': kornia.losses.TverskyLoss(alpha=args.tversky_loss_alpha, beta=1 - args.tversky_loss_alpha),
                'focal': kornia.losses.FocalLoss(alpha=args.focal_loss_alpha),
            }[args.loss],
            'accuracy': inkid.metrics.accuracy,
            'precision': inkid.metrics.precision,
            'recall': inkid.metrics.recall,
            'fbeta': inkid.metrics.fbeta,
            'auc': inkid.metrics.auc
        }
    elif args.label_type == 'rgb_values':
        label_args = dict(
            shape=label_shape
        )
        output_size = 3
        metrics = {
            'loss': nn.SmoothL1Loss(reduction='mean')
        }
    else:
        logging.error('Label type not recognized: {}'.format(args.label_type))
        return

    train_ds.set_for_all_sources('label_type', args.label_type)  # TODO should all this stuff just be in init
    train_ds.set_for_all_sources('label_args', label_args)
    val_ds.set_for_all_sources('label_type', args.label_type)
    val_ds.set_for_all_sources('label_args', label_args)
    # pred_ds does not generate labels, left as None

    if args.training_max_samples is not None:
        train_ds = inkid.ops.take_from_dataset(train_ds, args.training_max_samples)
    # Only take n samples for validation, not the entire region
    if args.validation_max_samples is not None:
        val_ds = inkid.ops.take_from_dataset(val_ds, args.validation_max_samples)

    if args.dataloaders_num_workers is None:
        args.dataloaders_num_workers = multiprocessing.cpu_count()

    # Define the dataloaders which implement batching, shuffling, etc.
    train_dl, val_dl, pred_dl = None, None, None
    if len(train_ds) > 0:
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.dataloaders_num_workers)
    if len(val_ds) > 0:
        val_dl = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=True,
                            num_workers=args.dataloaders_num_workers)
    if len(pred_ds) > 0:
        pred_dl = DataLoader(pred_ds, batch_size=args.batch_size * 2, shuffle=False,
                             num_workers=args.dataloaders_num_workers)

    # Specify the compute device for PyTorch purposes
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f'PyTorch device: {device}')
    if device.type == 'cuda':
        logging.info('    ' + torch.cuda.get_device_name(0))
        logging.info(f'    Memory Allocated: {round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1)} GB')
        logging.info(f'    Memory Cached:    {round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1)} GB')

    # Create the model for training
    if args.feature_type == 'subvolume_3dcnn':
        in_channels = 1
        if args.model == 'original':
            encoder = inkid.model.Subvolume3DcnnEncoder(args.subvolume_shape_voxels,
                                                        args.batch_norm_momentum,
                                                        args.no_batch_norm,
                                                        args.filters,
                                                        in_channels)
        elif args.model in ('3dunet_full', '3dunet_half'):
            encoder = inkid.model.Subvolume3DUNet(args.subvolume_shape_voxels,
                                                  args.batch_norm_momentum,
                                                  args.unet_starting_channels,
                                                  in_channels,
                                                  decode=(args.model == '3dunet_full'))
        else:
            logging.error(f'Model {args.model} is invalid for feature type {args.feature_type}.')
            return
        if args.model_3d_to_2d:
            decoder = inkid.model.ConvolutionalInkDecoder(args.filters, output_size)
        else:
            decoder = inkid.model.LinearInkDecoder(args.drop_rate, encoder.output_shape, output_size)
        model = torch.nn.Sequential(encoder, decoder)
    else:
        logging.error('Feature type: {} does not have a model implementation.'.format(args.feature_type))
        return

    # Load pretrained weights if specified
    if args.load_weights_from is not None:
        checkpoint = torch.load(args.load_weights_from)
        model.load_state_dict(checkpoint['model_state_dict'])

    # Move model to device (possibly GPU)
    model = model.to(device)

    # Show model in TensorBoard
    try:
        if train_dl is not None:
            _, features, _ = next(iter(train_dl))
            writer.add_graph(model, features)
            writer.flush()
    except RuntimeError:
        logging.warning('Unable to add model graph to TensorBoard, skipping this step')

    # Print summary of model
    shape = (in_channels,) + tuple(args.subvolume_shape_voxels)
    summary = torchsummary.summary(model, shape, device=device, verbose=0, branching=False)
    logging.info('Model summary (sizes represent single batch):\n' + str(summary))

    # Define optimizer
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    metric_results = {metric: [] for metric in metrics}
    last_summary = time.time()

    # Run training loop
    if train_dl is not None and not args.skip_training:
        try:
            for epoch in range(args.training_epochs):
                model.train()  # Turn on training mode
                total_batches = len(train_dl)
                for batch_num, (_, xb, yb) in enumerate(train_dl):
                    xb = xb.to(device)
                    yb = yb.to(device)
                    pred = model(xb)
                    if args.label_type == 'ink_classes':
                        _, yb = yb.max(1)  # Argmax
                    for metric, fn in metrics.items():
                        metric_results[metric].append(fn(pred, yb))

                    metric_results['loss'][-1].backward()
                    opt.step()
                    opt.zero_grad()

                    if batch_num % args.summary_every_n_batches == 0:
                        logging.info(f'Batch: {batch_num:>5d}/{total_batches:<5d} '
                                     f'{inkid.metrics.metrics_str(metric_results)} '
                                     f'Seconds: {time.time() - last_summary:5.3g}')
                        for metric, result in inkid.metrics.metrics_dict(metric_results).items():
                            writer.add_scalar('train_' + metric, result, epoch * len(train_dl) + batch_num)
                            writer.flush()
                        for result in metric_results.values():
                            result.clear()
                        last_summary = time.time()

                    if batch_num % args.checkpoint_every_n_batches == 0:
                        # Save model checkpoint
                        torch.save({
                            'epoch': epoch,
                            'batch': batch_num,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': opt.state_dict()
                        }, os.path.join(checkpoints_dir, f'checkpoint_{epoch}_{batch_num}.pt'))

                        # Periodic evaluation and prediction
                        if val_dl is not None:
                            logging.info('Evaluating on validation set... ')
                            val_results = perform_validation(model, val_dl, metrics, device, args.label_type)
                            for metric, result in inkid.metrics.metrics_dict(val_results).items():
                                writer.add_scalar('val_' + metric, result, epoch * len(train_dl) + batch_num)
                                writer.flush()
                            logging.info(f'done ({inkid.metrics.metrics_str(val_results)})')
                        else:
                            logging.info('Empty validation set, skipping validation.')

                        # Prediction image
                        if pred_dl is not None:
                            logging.info('Generating prediction image... ')
                            generate_prediction_image(pred_dl, model, output_size, args.label_type, device,
                                                      predictions_dir, f'{epoch}_{batch_num}',
                                                      args.prediction_averaging)
                            logging.info('done')
                        else:
                            logging.info('Empty prediction set, skipping prediction image generation.')
        except KeyboardInterrupt:
            pass

    # Run a final prediction on all regions
    if args.final_prediction_on_all:
        try:
            all_sources = list(set(args.training_set + args.validation_set + args.prediction_set))
            final_pred_ds = InkidDataset(all_sources, args.data_root)
            final_pred_ds.set_for_all_sources('feature_type', args.feature_type)
            final_pred_ds.set_for_all_sources('feature_args', pred_feature_args)
            final_pred_ds.set_regions_grid_spacing(args.prediction_grid_spacing)
            if len(final_pred_ds) > 0:
                final_pred_dl = DataLoader(final_pred_ds, batch_size=args.batch_size * 2, shuffle=False,
                                           num_workers=args.dataloaders_num_workers)
                generate_prediction_image(final_pred_dl, model, output_size, args.label_type, device,
                                          predictions_dir, 'final', args.prediction_averaging)
        # Perform finishing touches even if cut short
        except KeyboardInterrupt:
            pass

    # Add final validation metrics to metadata
    try:
        if val_dl is not None:
            logging.info('Performing final evaluation on validation set... ')
            val_results = perform_validation(model, val_dl, metrics, device, args.label_type)
            metadata['Final validation metrics'] = inkid.metrics.metrics_dict(val_results)
            logging.info(f'done ({inkid.metrics.metrics_str(val_results)})')
        else:
            logging.info('Empty validation set, skipping final validation.')
    except KeyboardInterrupt:
        pass

    # Add some post-run info to metadata file
    stop = timeit.default_timer()
    metadata['Runtime'] = stop - start
    metadata['Finished at'] = time.strftime('%Y-%m-%d %H:%M:%S')

    # Update metadata file on disk with results after run
    with open(os.path.join(output_path, 'metadata.json'), 'w') as f:
        f.write(json.dumps(metadata, indent=4, sort_keys=False))

    # Transfer results via rclone if requested
    if args.rclone_transfer_remote is not None:
        inkid.ops.rclone_transfer_to_remote(args.rclone_transfer_remote, output_path)

    writer.close()


if __name__ == '__main__':
    main()
