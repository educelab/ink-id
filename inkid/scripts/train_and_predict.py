"""Train and predict using subvolumes.

This script reads data source files and then runs a training process, with intermediate validation and predictions
made as defined by the data and provided arguments.

The optional value --cross-validate-on <n> can be passed to use this script for k-fold cross validation (and
prediction in this case). The nth data source from the training dataset will be removed from the training dataset and
added to those for validation and prediction.

"""

import argparse
import contextlib
import datetime
import json
import logging
import multiprocessing
import os
import sys
import time
import timeit

import git
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchsummary

import inkid


def main():
    """Run the training and prediction process."""
    start = timeit.default_timer()

    parser = argparse.ArgumentParser(description=__doc__)

    # Needed files
    parser.add_argument('output', metavar='output', help='output directory')
    parser.add_argument('--training-set', metavar='path', nargs='*', help='training dataset(s)', default=[])
    parser.add_argument('--validation-set', metavar='path', nargs='*', help='validation dataset(s)', default=[])
    parser.add_argument('--prediction-set', metavar='path', nargs='*', help='prediction dataset(s)', default=[])

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
                        choices=['original', '3dunet_full', '3dunet_half', 'autoencoder'])
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

    train_ds = inkid.data.Dataset(args.training_set)
    val_ds = inkid.data.Dataset(args.validation_set)
    pred_ds = inkid.data.Dataset(args.prediction_set)

    # If k-fold job, remove nth region from training and put in prediction/validation sets
    if args.cross_validate_on is not None:
        nth_region_path: str = train_ds.regions()[args.cross_validate_on].path
        train_ds.remove_source(nth_region_path)
        val_ds.sources.append(inkid.data.RegionSource(nth_region_path))
        pred_ds.sources.append(inkid.data.RegionSource(nth_region_path))

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

    train_ds.set_for_all_sources('label_type', args.label_type)  # TODO should all this stuff just be in init?
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
        if args.model in ['original', 'autoencoder']:
            encoder = inkid.model.Subvolume3DcnnEncoder(args.subvolume_shape_voxels,
                                                        args.batch_norm_momentum,
                                                        args.no_batch_norm,
                                                        args.filters,
                                                        in_channels)
        elif args.model in ['3dunet_full', '3dunet_half']:
            encoder = inkid.model.Subvolume3DUNet(args.subvolume_shape_voxels,
                                                  args.batch_norm_momentum,
                                                  args.unet_starting_channels,
                                                  in_channels,
                                                  decode=(args.model == '3dunet_full'))
        else:
            logging.error(f'Model {args.model} is invalid for feature type {args.feature_type}.')
            return
        if args.model == 'autoencoder':
            decoder = inkid.model.Subvolume3DcnnDecoder(args.batch_norm_momentum,
                                                        args.no_batch_norm,
                                                        args.filters,
                                                        in_channels)
        elif args.model_3d_to_2d:
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
                    for batch_num, (_, xb, yb) in enumerate(train_dl):
                        with torch.profiler.record_function(f'train_batch_{batch_num}'):
                            xb = xb.to(device)
                            if args.model == 'autoencoder':
                                yb = xb.clone()
                            else:
                                yb = yb.to(device)
                            pred = model(xb)
                            if args.label_type == 'ink_classes' and args.model != 'autoencoder':
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
                                    val_results = inkid.ops.perform_validation(model, val_dl, metrics, device,
                                                                               args.label_type)
                                    for metric, result in inkid.metrics.metrics_dict(val_results).items():
                                        writer.add_scalar('val_' + metric, result, epoch * len(train_dl) + batch_num)
                                        writer.flush()
                                    logging.info(f'done ({inkid.metrics.metrics_str(val_results)})')
                                else:
                                    logging.info('Empty validation set, skipping validation.')

                                # Prediction image
                                if pred_dl is not None:
                                    logging.info('Generating prediction image... ')
                                    inkid.ops.generate_prediction_images(pred_dl, model, output_size, args.label_type,
                                                                         device,
                                                                         predictions_dir, f'{epoch}_{batch_num}',
                                                                         args.prediction_averaging)
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
            all_sources = list(set(args.training_set + args.validation_set + args.prediction_set))
            final_pred_ds = inkid.data.Dataset(all_sources)
            final_pred_ds.set_for_all_sources('feature_type', args.feature_type)
            final_pred_ds.set_for_all_sources('feature_args', pred_feature_args)
            final_pred_ds.set_regions_grid_spacing(args.prediction_grid_spacing)
            if len(final_pred_ds) > 0:
                final_pred_dl = DataLoader(final_pred_ds, batch_size=args.batch_size * 2, shuffle=False,
                                           num_workers=args.dataloaders_num_workers)
                inkid.ops.generate_prediction_images(final_pred_dl, model, output_size, args.label_type, device,
                                                     predictions_dir, 'final', args.prediction_averaging)
        # Perform finishing touches even if cut short
        except KeyboardInterrupt:
            pass

    # Add final validation metrics to metadata
    try:
        if val_dl is not None:
            logging.info('Performing final evaluation on validation set... ')
            val_results = inkid.ops.perform_validation(model, val_dl, metrics, device, args.label_type)
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
