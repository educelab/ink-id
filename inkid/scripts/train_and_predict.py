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


class NoTrainingLossError(RuntimeError):
    pass


def main():
    """Run the training and prediction process."""
    start = timeit.default_timer()

    parser = argparse.ArgumentParser(description=__doc__)

    # Needed files
    parser.add_argument('--output', metavar='output', help='output directory', required=True)
    parser.add_argument('--training-set', metavar='path', nargs='*', help='training dataset(s)', default=[])
    parser.add_argument('--validation-set', metavar='path', nargs='*', help='validation dataset(s)', default=[])
    parser.add_argument('--prediction-set', metavar='path', nargs='*', help='prediction dataset(s)', default=[])

    # Dataset modifications
    parser.add_argument('--cross-validate-on', metavar='n', default=None, type=int,
                        help='remove the nth source from the flattened set of all training data sources, and '
                             'add this set to the validation and prediction sets')

    # Method
    parser.add_argument('--model-3d-to-2d', action='store_true',
                        help='Use semi-fully convolutional model (which removes a dimension) with 2d labels per '
                             'subvolume')

    # Subvolumes
    inkid.ops.add_subvolume_args(parser)

    # Data organization/augmentation
    parser.add_argument('--jitter-max', metavar='n', type=int, default=4)
    parser.add_argument('--no-augmentation', action='store_true')

    # Network architecture
    parser.add_argument('--model', default='InkClassifier3DCNN', help='model to run against',
                        choices=inkid.ops.model_choices())
    parser.add_argument('--volcart-texture-loss-coefficient', default=100, type=float,
                        help='Multiplicative weight of the volcart texture loss in any loss sums')
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

    # If this is a cross-validation job, that directory is allowed to have output from other
    # cross-validation splits, but not this one
    if os.path.isdir(args.output):
        if args.cross_validate_on is not None:
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

    if args.dataloaders_num_workers is None:
        args.dataloaders_num_workers = multiprocessing.cpu_count()

    # Create metadata dict
    metadata = {
        'Arguments': vars(args),
        'Command': ' '.join(sys.argv),
        'Date': datetime.datetime.today().strftime('%Y-%m-%d-%H:%M:%S'),
    }

    # Add git hash to metadata if inside a git repository
    try:
        repo = git.Repo(os.path.join(os.path.dirname(inkid.__file__), '..'))
        sha = repo.head.object.hexsha
        metadata['Git hash'] = repo.git.rev_parse(sha, short=6)
    except git.exc.InvalidGitRepositoryError:
        metadata['Git hash'] = 'No git hash available (unable to find valid repository).'

    # Add SLURM info if it exists
    for slurm_var in ['SLURM_JOB_ID', 'SLURM_JOB_NAME', 'SLURMD_NODENAME', 'SLURM_JOB_NODELIST']:
        if slurm_var in os.environ:
            metadata[slurm_var] = os.getenv(slurm_var)

    # Define the feature inputs to the network
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

    # Create the model for training
    in_channels = 1
    model = {
        'Autoencoder': inkid.model.Autoencoder(
            args.subvolume_shape_voxels, args.batch_norm_momentum, args.no_batch_norm, args.filters),
        'AutoencoderAndInkClassifier': inkid.model.AutoencoderAndInkClassifier(
            args.subvolume_shape_voxels, args.batch_norm_momentum, args.no_batch_norm, args.filters, args.drop_rate),
        'InkClassifier3DCNN': inkid.model.InkClassifier3DCNN(
            args.subvolume_shape_voxels, args.batch_norm_momentum, args.no_batch_norm, args.filters, args.drop_rate),
        'InkClassifier3DUNet': inkid.model.InkClassifier3DUNet(
            args.subvolume_shape_voxels, args.batch_norm_momentum, args.unet_starting_channels, in_channels,
            args.drop_rate),
        'InkClassifier3DUNetHalf': inkid.model.InkClassifier3DUNetHalf(
            args.subvolume_shape_voxels, args.batch_norm_momentum, args.unet_starting_channels, in_channels,
            args.drop_rate),
        'InkClassifierCrossTaskVCTexture': inkid.model.InkClassifierCrossTaskVCTexture(
            args.subvolume_shape_voxels, args.batch_norm_momentum, args.no_batch_norm, args.filters, args.drop_rate),
        'RGB3DCNN': inkid.model.RGB3DCNN(
            args.subvolume_shape_voxels, args.batch_norm_momentum, args.no_batch_norm, args.filters, args.drop_rate),
    }[args.model]

    # Define the labels and metrics
    if args.model_3d_to_2d:
        label_shape = (args.subvolume_shape_voxels[1], args.subvolume_shape_voxels[2])
    else:
        label_shape = (1, 1)
    metrics = {}
    label_args = {}
    if 'ink_classes' in model.labels:
        label_args['ink_classes'] = dict(
            shape=label_shape
        )
        metrics['ink_classes'] = {
            'loss': nn.CrossEntropyLoss(),
            'accuracy': inkid.metrics.accuracy,
            'precision': inkid.metrics.precision,
            'recall': inkid.metrics.recall,
            'fbeta': inkid.metrics.fbeta,
            'auc': inkid.metrics.auc,
        }
    if 'rgb_values' in model.labels:
        label_args['rgb_values'] = dict(
            shape=label_shape
        )
        metrics['rgb_values'] = {
            'loss': nn.SmoothL1Loss(reduction='mean'),
        }
    if 'autoencoded' in model.labels:
        metrics['autoencoded'] = {
            'loss': nn.MSELoss(),
        }
    if 'volcart_texture' in model.labels:
        label_args['volcart_texture'] = dict(
            shape=label_shape
        )
        metrics['volcart_texture'] = {
            'loss': inkid.metrics.weight_loss(args.volcart_texture_loss_coefficient, nn.SmoothL1Loss()),
        }
    metric_results = {label_type: {metric: [] for metric in metrics[label_type]} for label_type in metrics}

    train_ds = inkid.data.Dataset(args.training_set, train_feature_args, model.labels, label_args)
    val_ds = inkid.data.Dataset(args.validation_set, val_feature_args, model.labels, label_args)
    pred_ds = inkid.data.Dataset(args.prediction_set, pred_feature_args)  # pred_ds has no labels

    # If k-fold job, remove nth region from training and put in prediction/validation sets
    if args.cross_validate_on is not None:
        nth_region_path: str = train_ds.regions()[args.cross_validate_on].path
        nth_region_source = train_ds.pop_source(nth_region_path)
        val_ds.sources.append(nth_region_source)
        pred_ds.sources.append(nth_region_source)

    pred_ds.set_regions_grid_spacing(args.prediction_grid_spacing)

    metadata['Data'] = {
        'training': train_ds.data_dict(),
        'validation': val_ds.data_dict(),
        'prediction': pred_ds.data_dict(),
    }

    # Print metadata for logging and diagnostics
    logging.info('\n' + json.dumps(metadata, indent=4, sort_keys=False))

    # Write preliminary metadata to file (will be updated when job completes)
    with open(os.path.join(output_path, 'metadata.json'), 'w') as metadata_file:
        metadata_file.write(json.dumps(metadata, indent=4, sort_keys=False))

    if args.training_max_samples is not None:
        logging.info(f'Trimming training dataset to {args.training_max_samples} samples...')
        train_ds = inkid.ops.take_from_dataset(train_ds, args.training_max_samples)
        logging.info('done')
    # Only take n samples for validation, not the entire region
    if args.validation_max_samples is not None:
        logging.info(f'Trimming validation dataset to {args.validation_max_samples}...')
        val_ds = inkid.ops.take_from_dataset(val_ds, args.validation_max_samples)
        logging.info('done')

    # Define the dataloaders which implement batching, shuffling, etc.
    logging.info('Creating dataloaders...')
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
    logging.info('done')

    # Specify the compute device for PyTorch purposes
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f'PyTorch device: {device}')
    if device.type == 'cuda':
        logging.info(f'    {torch.cuda.get_device_name(0)}')
        logging.info(f'    Memory Allocated: {round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1)} GB')
        logging.info(f'    Memory Cached:    {round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1)} GB')

    # Load pretrained weights if specified
    if args.load_weights_from is not None:
        logging.info('Loading pretrained weights...')
        model_dict = model.state_dict()
        checkpoint = torch.load(args.load_weights_from)
        pretrained_dict = checkpoint['model_state_dict']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        logging.info('done')

    # Show model in TensorBoard and save sample subvolumes
    sample_dl = train_dl or val_dl or pred_dl
    if sample_dl is not None:
        features = next(iter(sample_dl))['feature']
        writer.add_graph(inkid.ops.ImmutableOutputModelWrapper(model), features)
        writer.flush()
        inkid.ops.save_subvolume_batch_to_img(model, device, sample_dl, os.path.join(output_path, 'subvolumes'))

    # Move model to device (possibly GPU)
    model = model.to(device)

    # Print summary of model
    shape = (in_channels,) + tuple(args.subvolume_shape_voxels)
    summary = torchsummary.summary(model, shape, device=device, verbose=0, branching=False)
    logging.info('Model summary (sizes represent single batch):\n' + str(summary))

    # Define optimizer
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

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
                    for batch_num, batch in enumerate(train_dl):
                        xb = batch['feature']
                        with torch.profiler.record_function(f'train_batch_{batch_num}'):
                            xb = xb.to(device)
                            preds = model(xb)
                            total_loss = None
                            for label_type in model.labels:
                                yb = xb.clone() if label_type == 'autoencoded' else batch[label_type].to(device)
                                if label_type == 'ink_classes':
                                    _, yb = yb.max(1)  # Argmax
                                pred = preds[label_type]
                                for metric, fn in metrics[label_type].items():
                                    metric_result = fn(pred, yb)
                                    metric_results[label_type][metric].append(metric_result)
                                    if metric == 'loss':
                                        if total_loss is None:
                                            total_loss = metric_result
                                        else:
                                            total_loss = total_loss + metric_result
                            if total_loss is not None:
                                total_loss.backward()
                                if 'total' not in metric_results:
                                    metric_results['total'] = {'loss': []}
                                metric_results['total']['loss'].append(total_loss)
                            else:
                                raise NoTrainingLossError('Training is running, but no loss function encountered')
                            opt.step()
                            opt.zero_grad()

                            if batch_num % args.summary_every_n_batches == 0:
                                logging.info(f'Batch: {batch_num:>5d}/{total_batches:<5d} '
                                             f'{inkid.metrics.metrics_str(metric_results)} '
                                             f'Seconds: {time.time() - last_summary:5.3g}')
                                for metric, result in inkid.metrics.metrics_dict(metric_results).items():
                                    writer.add_scalar('train_' + metric, result, epoch * len(train_dl) + batch_num)
                                    writer.flush()
                                for metrics_for_label in metric_results.values():
                                    for single_metric_results in metrics_for_label.values():
                                        single_metric_results.clear()
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
                                    val_results = inkid.ops.perform_validation(model, val_dl, metrics, device)
                                    for metric, result in inkid.metrics.metrics_dict(val_results).items():
                                        writer.add_scalar('val_' + metric, result, epoch * len(train_dl) + batch_num)
                                        writer.flush()
                                    logging.info(f'done ({inkid.metrics.metrics_str(val_results)})')
                                else:
                                    logging.info('Empty validation set, skipping validation.')

                                # Prediction image
                                if pred_dl is not None:
                                    logging.info('Generating prediction image... ')
                                    inkid.ops.generate_prediction_images(
                                        pred_dl, model, device, predictions_dir, f'{epoch}_{batch_num}',
                                        args.prediction_averaging)
                                    logging.info('done')
                                else:
                                    logging.info('Empty prediction set, skipping prediction image generation.')

                                # Visualize autoencoder outputs
                                if args.model in ['autoencoder', 'AutoencoderAndInkClassifier']:
                                    logging.info('Visualizing autoencoder outputs...')
                                    inkid.ops.save_subvolume_batch_to_img(
                                        model, device, train_dl, os.path.join(output_path, 'subvolumes'),
                                        include_autoencoded=True, iteration=batch_num, include_vol_slices=False
                                    )
                                    logging.info('done')
                            # Only advance profiler step if profiling is enabled (the context manager is not null)
                            if isinstance(context_manager, torch.profiler.profile):
                                context_manager.step()
        except KeyboardInterrupt:
            pass

    # Run a final prediction on all regions
    if args.final_prediction_on_all:
        try:
            logging.info('Generating final prediction images... ')
            all_sources = list(set(args.training_set + args.validation_set + args.prediction_set))
            final_pred_ds = inkid.data.Dataset(all_sources, pred_feature_args)
            final_pred_ds.set_regions_grid_spacing(args.prediction_grid_spacing)
            if len(final_pred_ds) > 0:
                final_pred_dl = DataLoader(final_pred_ds, batch_size=args.batch_size * 2, shuffle=False,
                                           num_workers=args.dataloaders_num_workers)
                inkid.ops.generate_prediction_images(final_pred_dl, model, device,
                                                     predictions_dir, 'final', args.prediction_averaging)
            logging.info('done')
        # Perform finishing touches even if cut short
        except KeyboardInterrupt:
            pass

    # Add final validation metrics to metadata
    try:
        if val_dl is not None:
            logging.info('Performing final evaluation on validation set... ')
            val_results = inkid.ops.perform_validation(model, val_dl, metrics, device)
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
