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

import argparse
import contextlib
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
from typing import List

import git
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchsummary
from tqdm import tqdm

import inkid


class InkidDataSource(object):
    """Can be either a region or a volume. Produces subvolumes and possibly labels."""

    def __init__(self, path: str) -> None:
        pass


class InkidDataset(torch.utils.data.Dataset):
    def __init__(self, data_source_paths: List[str], data_root: str) -> None:
        """TODO

        Args:
            data_source_paths: a list of dataset or data source file paths
        """
        # Convert the list of paths to a list of InkidDataSources
        data_source_paths = self.expand_data_sources(data_source_paths)
        self.sources: List[InkidDataSource] = list()
        for data_source_path in data_source_paths:
            self.sources.append(InkidDataSource(data_source_path))
        print(self.sources)

    def __len__(self) -> int:
        pass

    def __getitem__(self, idx: int):
        pass

    def expand_data_sources(self, data_source_paths: List[str], are_we_recursing: bool = False) -> List[str]:
        """Expand .txt and .json contents into flattened list of .json files.

        The file paths in the input can point to either .txt or .json files. The .txt
        files are themselves lists of other files, which can further be .txt or .json.
        This function goes through this list, and for any .txt file it reads the list
        that file contains and recursively processes it. The result is a list of only
        .json data source file paths.

        The original input file paths are absolute since they are passed from the command line.
        The rest of the paths will be relative to the data root directory. This changes
        all paths to be absolute.

        """
        expanded_paths: List[str] = list()
        for source_path in data_source_paths:
            file_extension = os.path.splitext(source_path)[1]
            if file_extension == '.json':
                self.sources.append(InkidDataSource(source_path))
            elif file_extension == '.txt':
                source_json_paths = self.expand_data_sources([source_path], are_we_recursing=True)
                for source_json_path in source_json_paths:
                    self.sources.append(InkidDataSource(source_json_path))
            else:
                raise ValueError(f'Data source {source_path} is not a permitted file type (.txt or .json)')
        with open(dataset_txt_file_path) as f:
            return f.readlines()


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
    parser.add_argument('--data-root', matavar='path', default=None,
                        help='path to the data root that contains the ')

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
                        help='Use semi-fully convolutional model (which removes a dimension) with 2d labels per '
                             'subvolume')
    parser.add_argument('--loss', choices=['cross_entropy'], default='cross_entropy')

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

    # Profiling
    parser.add_argument('--no-profiling', action='store_false', dest='profiling',
                        help='Disable PyTorch profiling during training')

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
    tensorboard_path = os.path.join(output_path, 'tensorboard')
    writer = SummaryWriter(tensorboard_path)

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

    train_ds = InkidDataset(args.training_set)

    # Transform the input file into region set, can handle JSON or PPM
    region_data = inkid.data.RegionSet.get_data_from_file_or_url(args.data)  # TODO region set LEFT OFF

    # If k-fold job, remove nth region from training and put in prediction/validation sets TODO region set
    if args.cross_validate_on is not None:
        n_region = region_data['regions']['training'].pop(int(args.cross_validate_on))
        region_data['regions']['prediction'].append(n_region)
        region_data['regions']['validation'].append(n_region)

    # Now that we have made all these changes to the region data, create a region set from this data
    regions = inkid.data.RegionSet(region_data)  # TODO region set

    # Create metadata dict TODO region set
    metadata = {'Arguments': vars(args), 'Region set': region_data, 'Command': ' '.join(sys.argv)}

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
    train_ds = inkid.data.PointsDataset(regions, ['training'], training_features_fn, label_fn)
    if args.training_max_samples is not None:
        train_ds = inkid.ops.take_from_dataset(train_ds, args.training_max_samples)
    val_ds = inkid.data.PointsDataset(regions, ['validation'], validation_features_fn, label_fn)
    # Only take n samples for validation, not the entire region
    if args.validation_max_samples is not None:
        val_ds = inkid.ops.take_from_dataset(val_ds, args.validation_max_samples)
    pred_ds = inkid.data.PointsDataset(regions, ['prediction'], prediction_features_fn, lambda p: p,
                                       grid_spacing=args.prediction_grid_spacing)

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

    # Set up profiling
    if args.profiling:
        context_manager = torch.profiler.profile(
            record_shapes=True, with_stack=True,
            schedule=torch.profiler.schedule(
                wait=10,  # Wait ten batches before doing anything
                warmup=10,  # Profile ten batches, but don't actually record those results
                active=10,  # Profile ten batches, and record those
                repeat=1  # Only do this process once
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(tensorboard_path)
        )
    else:
        context_manager = contextlib.nullcontext()

    # Run training loop
    if train_dl is not None and not args.skip_training:
        try:
            with context_manager:
                for epoch in range(args.training_epochs):
                    model.train()  # Turn on training mode
                    total_batches = len(train_dl)
                    for batch_num, (xb, yb) in enumerate(train_dl):
                        with torch.profiler.record_function(f'train_batch_{batch_num}'):
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
                                                              predictions_dir, f'{epoch}_{batch_num}', reconstruct_fn,
                                                              regions,
                                                              label_shape, args.prediction_averaging,
                                                              args.prediction_grid_spacing)
                                    logging.info('done')
                                else:
                                    logging.info('Empty prediction set, skipping prediction image generation.')
                            # Only advance profiler step if profiling is enabled (the context manager is not null)
                            if isinstance(context_manager, torch.profiler.profile):
                                context_manager.step()
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
