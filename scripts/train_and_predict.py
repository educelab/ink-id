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
import functools
import itertools
import json
import logging
import multiprocessing
import os
import sys
import time
import timeit
from typing import List, Optional, Tuple

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

RegionPoint: Tuple[str, int, int]


class InkidDataSource(ABC):
    """Can be either a region or a volume. Produces inputs (e.g. subvolumes) and possibly labels."""
    def __init__(self, path: str) -> None:
        self.path = path
        with open(path, 'r') as f:
            source_json = json.load(f)
        self._source_json = source_json

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

        self._points: List[RegionPoint] = list()
        self._points_list_needs_update: bool = True

        self.grid_spacing = 1
        self.specify_inkness = None
        
    def __len__(self) -> int:
        if self._points_list_needs_update:
            self.update_points_list()
        return len(self._points)

    def __getitem__(self, item):
        if self._points_list_needs_update:
            self.update_points_list()
        point: RegionPoint = self._points[item]
        # TODO get that point (x, y) from list of points
        # TODO read that value from PPM
        # TODO get the feature using feature_fn
        # TODO get the label using label_fn
        # TODO return them
        pass

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
                    self._points.append((self.path, x, y))
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


class InkidVolumeSource(InkidDataSource):
    def __init__(self, path: str) -> None:
        super().__init__(path)

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass


class InkidDataset(torch.utils.data.Dataset):
    def __init__(self, source_paths: List[str], data_root: str) -> None:
        """TODO

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

    def remove_source(self, source_path: str) -> None:
        source_idx_to_remove: int = self.source_paths().index(source_path)
        self.sources.pop(source_idx_to_remove)

    def source_paths(self) -> List[str]:
        return [source.path for source in self.sources]

    def data_dict(self):
        return {source.path: source.data_dict() for source in self.sources}


def perform_validation(model, dataloader, metrics, device, label_type):
    """Run the validation process using a model and dataloader, and return the results of all metrics."""
    model.eval()  # Turn off training mode for batch norm and dropout purposes
    with torch.no_grad():
        metric_results = {metric: [] for metric in metrics}
        for xb, yb in tqdm(dataloader):
            pred = model(xb.to(device))
            yb = yb.to(device)
            if label_type == 'ink_classes':
                _, yb = yb.max(1)  # Argmax
            for metric, fn in metrics.items():
                metric_results[metric].append(fn(pred, yb))
    model.train()
    return metric_results


def generate_prediction_image(dataloader, model, output_size, label_type, device, predictions_dir, suffix,
                              reconstruct_fn, region_set, label_shape, prediction_averaging, grid_spacing):
    """Helper function to generate a prediction image given a model and dataloader, and save it to a file."""
    if label_shape == (1, 1):
        pred_shape = (grid_spacing, grid_spacing)
    else:
        pred_shape = label_shape
    predictions = np.empty(shape=(0, output_size, pred_shape[0], pred_shape[1]))
    points = np.empty(shape=(0, 3))
    model.eval()  # Turn off training mode for batch norm and dropout purposes
    with torch.no_grad():
        for pxb, pts in tqdm(dataloader):
            # Smooth predictions via augmentation. Augment each subvolume 8-fold via rotations and flips
            if prediction_averaging:
                rotations = range(4)
                flips = [False, True]
            else:
                rotations = [0]
                flips = [False]
            batch_preds = np.zeros((0, pxb.shape[0], output_size, pred_shape[0], pred_shape[1]))
            for rotation, flip in itertools.product(rotations, flips):
                # Example pxb.shape = [64, 1, 48, 48, 48] (BxCxDxHxW)
                # Augment via rotation and flip
                aug_pxb = pxb.rot90(rotation, [3, 4])
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
                # Repeat prediction to fill grid square so prediction image is not single pixels in sea of blackness
                if label_shape == (1, 1):
                    pred = np.repeat(pred, repeats=grid_spacing, axis=3)
                    pred = np.repeat(pred, repeats=grid_spacing, axis=4)
                # Save this augmentation to the batch totals
                batch_preds = np.append(batch_preds, pred, axis=0)
            # Average over batch of predictions after augmentation
            batch_pred = batch_preds.mean(0)
            pts = pts.numpy()
            # Add batch of predictions to list
            predictions = np.append(predictions, batch_pred, axis=0)
            points = np.append(points, pts, axis=0)
    model.train()
    for prediction, point in zip(predictions, points):
        region_id, x, y = point
        reconstruct_fn([int(region_id)], [prediction], [[int(x), int(y)]])
    region_set.save_predictions(predictions_dir, suffix)
    region_set.reset_predictions()


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
    pred_ds.region_grid_spacing = args.prediction_grid_spacing

    # If k-fold job, remove nth region from training and put in prediction/validation sets
    if args.cross_validate_on is not None:
        nth_region_path: str = train_ds.regions()[args.cross_validate_on].path
        train_ds.remove_source(nth_region_path)
        val_ds.sources.append(InkidRegionSource(nth_region_path))
        pred_ds.sources.append(InkidRegionSource(nth_region_path))

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

    # Diagnostic printing
    logging.info('\n' + json.dumps(metadata, indent=4, sort_keys=False))

    # Write preliminary metadata to file (will be updated when job completes)
    with open(os.path.join(output_path, 'metadata.json'), 'w') as metadata_file:
        metadata_file.write(json.dumps(metadata, indent=4, sort_keys=False))

    # Define the feature inputs to the network
    if args.feature_type == 'subvolume_3dcnn':
        point_to_subvolume_input = functools.partial(
            regions.point_to_subvolume_input,  # TODO region set
            subvolume_shape_voxels=args.subvolume_shape_voxels,
            subvolume_shape_microns=args.subvolume_shape_microns,
            out_of_bounds='all_zeros',
            move_along_normal=args.move_along_normal,
            method=args.subvolume_method,
            normalize=args.normalize_subvolumes,
            model_3d_to_2d=args.model_3d_to_2d,
        )
        training_features_fn = functools.partial(
            point_to_subvolume_input,
            augment_subvolume=args.augmentation,
            jitter_max=args.jitter_max,
        )
        validation_features_fn = functools.partial(
            point_to_subvolume_input,
            augment_subvolume=False,
            jitter_max=0,
        )
        prediction_features_fn = validation_features_fn
    elif args.feature_type == 'voxel_vector_1dcnn':
        training_features_fn = functools.partial(
            regions.point_to_voxel_vector_input,
            length_in_each_direction=args.length_in_each_direction,
            out_of_bounds='all_zeros',
        )
        validation_features_fn = training_features_fn
        prediction_features_fn = training_features_fn
    elif args.feature_type == 'descriptive_statistics':
        training_features_fn = functools.partial(
            regions.point_to_descriptive_statistics,
            subvolume_shape_voxels=args.subvolume_shape_voxels,
            subvolume_shape_microns=args.subvolume_shape_microns
        )
        validation_features_fn = training_features_fn
        prediction_features_fn = training_features_fn
    else:
        logging.error('Feature type not recognized: {}'.format(args.feature_type))
        return

    # Define the labels
    if args.model_3d_to_2d:
        label_shape = (args.subvolume_shape_voxels[1], args.subvolume_shape_voxels[2])
    else:
        label_shape = (1, 1)
    if args.label_type == 'ink_classes':
        label_fn = functools.partial(regions.point_to_ink_classes_label, shape=label_shape)  # TODO region set
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
        reconstruct_fn = regions.reconstruct_predicted_ink_classes
    elif args.label_type == 'rgb_values':
        label_fn = functools.partial(regions.point_to_rgb_values_label, shape=label_shape)  # TODO region set
        output_size = 3
        metrics = {
            'loss': nn.SmoothL1Loss(reduction='mean')
        }
        reconstruct_fn = regions.reconstruct_predicted_rgb  # TODO region set
    else:
        logging.error('Label type not recognized: {}'.format(args.label_type))
        return

    # Define the datasets TODO region set
    # train_ds = inkid.data.PointsDataset(regions, ['training'], training_features_fn, label_fn)
    # if args.training_max_samples is not None:
    #     train_ds = inkid.ops.take_from_dataset(train_ds, args.training_max_samples)
    # val_ds = inkid.data.PointsDataset(regions, ['validation'], validation_features_fn, label_fn)
    # # Only take n samples for validation, not the entire region
    # if args.validation_max_samples is not None:
    #     val_ds = inkid.ops.take_from_dataset(val_ds, args.validation_max_samples)
    # pred_ds = inkid.data.PointsDataset(regions, ['prediction'], prediction_features_fn, lambda p: p,
    #                                    grid_spacing=args.prediction_grid_spacing)

    if args.dataloaders_num_workers is None:
        args.dataloaders_num_workers = multiprocessing.cpu_count()

    # Define the dataloaders which implement batching, shuffling, etc. TODO region set
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
        logging.info(f'    Memory Cached:    {round(torch.cuda.memory_cached(0) / 1024 ** 3, 1)} GB')

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
            inputs, _ = next(iter(train_dl))
            writer.add_graph(model, inputs)
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
                for batch_num, (xb, yb) in enumerate(train_dl):
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
                        logging.info('Batch: {:>5d}/{:<5d} {} Seconds: {:5.3g}'.format(
                            batch_num, total_batches,
                            inkid.metrics.metrics_str(metric_results),
                            time.time() - last_summary))
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
                                                      predictions_dir, f'{epoch}_{batch_num}', reconstruct_fn, regions,
                                                      label_shape, args.prediction_averaging,
                                                      args.prediction_grid_spacing)
                            logging.info('done')
                        else:
                            logging.info('Empty prediction set, skipping prediction image generation.')
        except KeyboardInterrupt:
            pass

    # Run a final prediction on all regions
    if args.final_prediction_on_all:
        try:
            final_pred_ds = inkid.data.PointsDataset(regions, ['prediction', 'training', 'validation'],
                                                     prediction_features_fn, lambda p: p,
                                                     grid_spacing=args.prediction_grid_spacing)
            if len(final_pred_ds) > 0:
                final_pred_dl = DataLoader(final_pred_ds, batch_size=args.batch_size * 2, shuffle=False,
                                           num_workers=args.dataloaders_num_workers)
                generate_prediction_image(final_pred_dl, model, output_size, args.label_type, device,
                                          predictions_dir, 'final', reconstruct_fn, regions, label_shape,
                                          args.prediction_averaging, args.prediction_grid_spacing)
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
