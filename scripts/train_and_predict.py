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
the prediction and validation sets for that run.

"""

import datetime
import functools
import inspect
import itertools
import json
import logging
import multiprocessing
import os
import sys
import time
import timeit

import configargparse
import git
import kornia
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torchsummary

import inkid


def take_from_dataset(dataset, n_samples):
    """Take the first n samples from a dataset to reduce the size."""
    if n_samples < len(dataset):
        dataset = random_split(dataset, [n_samples, len(dataset) - n_samples])[0]
    return dataset


def perform_validation(model, dataloader, metrics, device, label_type):
    """Run the validation process using a model and dataloader, and return the results of all metrics."""
    model.eval()  # Turn off training mode for batch norm and dropout purposes
    with torch.no_grad():
        metric_results = {metric: [] for metric in metrics}
        for xb, yb in dataloader:
            pred = model(xb.to(device))
            yb = yb.to(device)
            if label_type == 'ink_classes':
                _, yb = yb.max(1)  # Argmax
            for metric, fn in metrics.items():
                metric_results[metric].append(fn(pred, yb))
    model.train()
    return metric_results


def generate_prediction_image(dataloader, model, output_size, label_type, device, predictions_dir, filename,
                              reconstruct_fn, region_set, subvolume_shape):
    """Helper function to generate a prediction image given a model and dataloader, and save it to a file."""
    predictions = np.empty(shape=(0, output_size, subvolume_shape[2], subvolume_shape[1]))
    points = np.empty(shape=(0, 3))
    model.eval()  # Turn off training mode for batch norm and dropout purposes
    with torch.no_grad():
        for pxb, pts in dataloader:
            # Smooth predictions via augmentation. Augment each subvolume 8-fold via rotations and flips
            rotations = range(4)
            flips = [False, True]
            batch_preds = np.zeros((0, pxb.shape[0], output_size, subvolume_shape[2], subvolume_shape[1]))
            for rotation, flip in itertools.product(rotations, flips):
                # Example pxb.shape = [64, 1, 48, 48, 48] (BxCxDxHxW)
                # Augment via rotation and flip
                pxb = pxb.rot90(rotation, [3, 4])
                if flip:
                    pxb = pxb.flip(4)
                pred = model(pxb.to(device))
                if label_type == 'ink_classes':
                    pred = F.softmax(pred, dim=1)
                pred = pred.cpu()
                # Example pred.shape = [64, 2, 48, 48] (BxCxHxW)
                # Undo flip and rotation
                if flip:
                    pred = pred.flip(3)
                pred = pred.rot90(-rotation, [2, 3])
                pred = np.expand_dims(pred.numpy(), axis=0)
                batch_preds = np.append(batch_preds, pred, axis=0)
                # TODO visualize to double check augmentations
            # Average predictions after augmentation
            batch_pred = batch_preds.mean(0)
            pts = pts.numpy()
            # Add averaged predictions to list
            predictions = np.append(predictions, batch_pred, axis=0)
            points = np.append(points, pts, axis=0)
    model.train()
    for prediction, point in zip(predictions, points):
        region_id, x, y = point
        reconstruct_fn([int(region_id)], [prediction], [[int(x), int(y)]])
    region_set.save_predictions(predictions_dir, filename)
    region_set.reset_predictions()


def main():
    """Run the training and prediction process."""
    start = timeit.default_timer()

    parser = configargparse.ArgumentParser(
        description=__doc__,
        default_config_files=[inkid.ops.default_arguments_file()],
    )
    # Needed files
    parser.add_argument('data', metavar='infile', help='input data file (JSON or PPM)', nargs='?')
    parser.add_argument('output', metavar='outfile', help='output directory', nargs='?')

    # Config file so the user can have various configs saved for e.g. different scans that are often processed
    parser.add_argument('-c', '--config-file', metavar='path', is_config_file=True,
                        help='file of pre-specified arguments (in addition to pre-loaded defaults)')

    # Region set modifications
    parser.add_argument('-k', metavar='num', default=None, type=int,
                        help='index of region to use for prediction and validation')
    parser.add_argument('--override-volume-slices-dir', metavar='path', default=None,
                        help='override directory for all volume slices (only works if there is '
                             'only one volume in the region set file)')

    # Pre-trained model
    parser.add_argument('--model', metavar='path', default=None,
                        help='existing model directory to load checkpoints from')

    # Method
    parser.add_argument('--feature-type', metavar='name', default='subvolume_3dcnn',
                        help='type of feature model is built on',
                        choices=[
                            'subvolume_3dcnn',
                            'voxel_vector_1dcnn',
                            'descriptive_statistics',
                        ])
    parser.add_argument('--label-type', metavar='name', default='ink_classes',
                        help='type of label to train',
                        choices=[
                            'ink_classes',
                            'rgb_values',
                        ])
    parser.add_argument('--model-3d-to-2d', action='store_true',
                        help='Use semi-fully convolutional model (which removes a dimension) with 2d labels per '
                             'subvolume')
    parser.add_argument('--loss', choices=['cross_entropy', 'dice', 'tversky', 'focal'], default='cross_entropy')
    parser.add_argument('--tversky-loss-alpha', type=float, default=0.5)
    parser.add_argument('--focal-loss-alpha', type=float, default=0.5)

    # Subvolumes
    parser.add_argument('--subvolume-method', metavar='name', default='nearest_neighbor',
                        help='method for getting subvolumes',
                        choices=[
                            'nearest_neighbor',
                            'interpolated',
                        ])
    parser.add_argument('--subvolume-shape', metavar='n', nargs=3, type=int,
                        help='subvolume shape in z y x')
    parser.add_argument('--pad-to-shape', metavar='n', nargs=3, type=int, default=None,
                        help='pad subvolume with zeros to be of given shape (default no padding)')
    parser.add_argument('--move-along-normal', metavar='n', type=float,
                        help='number of voxels to move along normal before getting a subvolume')
    parser.add_argument('--normalize-subvolumes', action='store_true',
                        help='normalize each subvolume to zero mean and unit variance on the fly')
    parser.add_argument('--fft', action='store_true', help='Apply FFT to subvolumes')
    parser.add_argument('--dwt', metavar='name', default=None, help='Apply specified DWT to subvolumes')
    parser.add_argument('--dwt-channel-subbands', action='store_true',
                        help='Combine DWT subbands into multiple channels of smaller subvolume')

    # Voxel vectors
    parser.add_argument('--length-in-each-direction', metavar='n', type=int,
                        help='length of voxel vector in each direction along normal')

    # Data organization/augmentation
    parser.add_argument('--jitter-max', metavar='n', type=int)
    parser.add_argument('--augmentation', action='store_true', dest='augmentation')
    parser.add_argument('--no-augmentation', action='store_false', dest='augmentation')

    # Network architecture
    parser.add_argument('--learning-rate', metavar='n', type=float)
    parser.add_argument('--drop-rate', metavar='n', type=float)
    parser.add_argument('--batch-norm-momentum', metavar='n', type=float)
    parser.add_argument('--no-batch-norm', action='store_true')
    parser.add_argument('--filters', metavar='n', nargs='*', type=int,
                        help='number of filters for each convolution layer')

    # Run configuration
    parser.add_argument('--batch-size', metavar='n', type=int)
    parser.add_argument('--training-max-samples', metavar='n', type=int, default=None)
    parser.add_argument('--training-epochs', metavar='n', type=int, default=None)
    parser.add_argument('--prediction-grid-spacing', metavar='n', type=int,
                        help='prediction points will be taken from an NxN grid')
    parser.add_argument('--validation-max-samples', metavar='n', type=int)
    parser.add_argument('--summary-every-n-batches', metavar='n', type=int)
    parser.add_argument('--checkpoint-every-n-batches', metavar='n', type=int)
    parser.add_argument('--final-prediction-on-all', action='store_true')
    parser.add_argument('--skip-training', action='store_true')

    # Rclone
    parser.add_argument('--rclone-transfer-remote', metavar='remote', default=None,
                        help='if specified, and if matches the name of one of the directories in '
                             'the output path, transfer the results to that rclone remote into the '
                             'subpath following the remote name')

    args = parser.parse_args()

    # Make sure both input and output are provided
    if args.data is None and args.output is None:
        parser.print_help()
        return

    # If this is one of a k-fold cross-validation job, then append k to the output path
    # Whether or not that is the case, go ahead and create the output directory
    if args.k is None:
        dir_name = datetime.datetime.today().strftime('%Y-%m-%d_%H.%M.%S')
    else:
        dir_name = datetime.datetime.today().strftime('%Y-%m-%d_%H.%M.%S') + '_' + str(args.k)
    output_path = os.path.join(args.output, dir_name)
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

    # Point to preexisting model path if there is one
    if args.model is not None:
        model_path = args.model
    else:
        model_path = output_path

    # Automatically increase prediction grid spacing if using 2D labels, and turn off augmentation
    if args.model_3d_to_2d:
        args.prediction_grid_spacing = args.subvolume_shape[-1]
        args.augmentation = False

    # Define directories for prediction images and checkpoints
    predictions_dir = os.path.join(output_path, 'predictions')
    os.makedirs(predictions_dir)
    checkpoints_dir = os.path.join(output_path, 'checkpoints')
    os.makedirs(checkpoints_dir)

    # If input file is a PPM, treat this as a texturing module
    # Skip training, run a prediction on all regions, require trained model, require slices dir
    _, file_extension = os.path.splitext(args.data)
    file_extension = file_extension.lower()
    if file_extension == '.ppm':
        if args.model is None:
            logging.error("Pre-trained model (--model) required when texturing a .ppm file.")
            return
        if args.override_volume_slices_dir is None:
            logging.error("Volume (--override-volume-slices-dir) required when texturing a .ppm file.")
            return
        logging.info("PPM input file provided. Skipping training and running final prediction on all.")
        args.skip_training = True
        args.final_prediction_on_all = True

    # Transform the input file into region set, can handle JSON or PPM
    region_data = inkid.data.RegionSet.get_data_from_file(args.data)

    # Override volume slices directory (iff only one volume specified in the region set)
    if args.override_volume_slices_dir is not None:
        volume_dirs_seen = set()
        for ppm in region_data['ppms']:
            volume_dirs_seen.add(region_data['ppms'][ppm]['volume'])
            if len(volume_dirs_seen) > 1:
                raise ValueError('--override-volume-slices-dir only '
                                 'permitted if there is one volume in the region set')
        for ppm in region_data['ppms']:
            region_data['ppms'][ppm]['volume'] = args.override_volume_slices_dir

    # If k-fold job, remove kth region from training and put in prediction/validation sets
    if args.k is not None:
        k_region = region_data['regions']['training'].pop(int(args.k))
        region_data['regions']['prediction'].append(k_region)
        region_data['regions']['validation'].append(k_region)

    # Now that we have made all these changes to the region data, create a region set from this data
    regions = inkid.data.RegionSet(region_data)

    # Create metadata dict
    metadata = {'Arguments': vars(args), 'Region set': region_data, 'Command': ' '.join(sys.argv)}

    # Add git hash to metadata if inside a git repository
    try:
        repo = git.Repo(os.path.join(os.path.dirname(inspect.getfile(inkid)), '..'))
        sha = repo.head.object.hexsha
        metadata['Git hash'] = repo.git.rev_parse(sha, short=6)
    except git.exc.InvalidGitRepositoryError:
        metadata['Git hash'] = 'No git hash available (unable to find valid repository).'

    # Diagnostic printing
    logging.info('\n' + json.dumps(metadata, indent=4, sort_keys=False))

    # Write preliminary metadata to file
    with open(os.path.join(output_path, 'metadata.json'), 'w') as metadata_file:
        metadata_file.write(json.dumps(metadata, indent=4, sort_keys=False))

    # Define the feature inputs to the network
    if args.feature_type == 'subvolume_3dcnn':
        point_to_subvolume_input = functools.partial(
            regions.point_to_subvolume_input,
            subvolume_shape=args.subvolume_shape,
            out_of_bounds='all_zeros',
            move_along_normal=args.move_along_normal,
            method=args.subvolume_method,
            normalize=args.normalize_subvolumes,
            pad_to_shape=args.pad_to_shape,
            model_3d_to_2d=args.model_3d_to_2d,
            fft=args.fft,
            dwt=args.dwt,
            dwt_channel_subbands=args.dwt_channel_subbands,
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
            subvolume_shape=args.subvolume_shape,
        )
        validation_features_fn = training_features_fn
        prediction_features_fn = training_features_fn
    else:
        logging.error('Feature type not recognized: {}'.format(args.feature_type))
        return

    # Define the labels
    if args.model_3d_to_2d:
        label_shape = (args.subvolume_shape[1], args.subvolume_shape[2])
    else:
        label_shape = (1, 1)
    if args.label_type == 'ink_classes':
        label_fn = functools.partial(regions.point_to_ink_classes_label, shape=label_shape)
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
        label_fn = functools.partial(regions.point_to_rgb_values_label, shape=label_shape)
        output_size = 3
        metrics = {
            'loss': nn.SmoothL1Loss(reduction='mean')
        }
        reconstruct_fn = regions.reconstruct_predicted_rgb
    else:
        logging.error('Label type not recognized: {}'.format(args.label_type))
        return

    # Define the datasets
    train_ds = inkid.data.PointsDataset(regions, ['training'], training_features_fn, label_fn)
    if args.training_max_samples is not None:
        train_ds = take_from_dataset(train_ds, args.training_max_samples)
    val_ds = inkid.data.PointsDataset(regions, ['validation'], validation_features_fn, label_fn)
    # Only take n samples for validation, not the entire region
    if args.validation_max_samples is not None:
        val_ds = take_from_dataset(val_ds, args.validation_max_samples)
    pred_ds = inkid.data.PointsDataset(regions, ['prediction'], prediction_features_fn, lambda p: p,
                                       grid_spacing=args.prediction_grid_spacing)

    # Define the dataloaders which implement batching, shuffling, etc.
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=multiprocessing.cpu_count())
    val_dl = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=True,
                        num_workers=multiprocessing.cpu_count())
    pred_dl = DataLoader(pred_ds, batch_size=args.batch_size * 2, shuffle=False,
                         num_workers=multiprocessing.cpu_count())

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
        if args.dwt_channel_subbands:
            in_channels = 8
            args.subvolume_shape = [i // 2 for i in args.subvolume_shape]
            args.pad_to_shape = None
        encoder = inkid.model.Subvolume3DcnnEncoder(args.subvolume_shape, args.pad_to_shape, args.batch_norm_momentum,
                                                    args.no_batch_norm, args.filters, in_channels)
        if args.model_3d_to_2d:
            decoder = inkid.model.ConvolutionalInkDecoder(args.filters, output_size)
        else:
            decoder = inkid.model.LinearInkDecoder(args.drop_rate, encoder.output_shape, output_size)
        model = torch.nn.Sequential(encoder, decoder)
    else:
        logging.error('Feature type: {} does not have a model implementation.'.format(args.feature_type))
        return
    model = model.to(device)
    # Show model in TensorBoard
    try:
        inputs, _ = iter(train_dl).next()
        writer.add_graph(model, inputs)
        writer.flush()
    except RuntimeError:
        logging.warning('Unable to add model graph to TensorBoard, skipping this step')
    # Print summary of model
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    shape = (in_channels,) + tuple(args.pad_to_shape or args.subvolume_shape)
    summary, _ = torchsummary.summary_string(model, input_size=shape, batch_size=args.batch_size, device=device_str)
    logging.info('\n' + summary)
    # Define optimizer
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    metric_results = {metric: [] for metric in metrics}
    last_summary = time.time()

    # Run training loop
    if not args.skip_training:
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
                        logging.info('Evaluating on validation set... ')
                        val_results = perform_validation(model, val_dl, metrics, device, args.label_type)
                        for metric, result in inkid.metrics.metrics_dict(val_results).items():
                            writer.add_scalar('val_' + metric, result, epoch * len(train_dl) + batch_num)
                            writer.flush()
                        logging.info(f'done ({inkid.metrics.metrics_str(val_results)})')

                        # Prediction image
                        logging.info('Generating prediction image... ')
                        generate_prediction_image(pred_dl, model, output_size, args.label_type, device,
                                                  predictions_dir, f'{epoch}_{batch_num}', reconstruct_fn, regions,
                                                  args.subvolume_shape)
                        logging.info('done')
        except KeyboardInterrupt:
            pass

    # Run a final prediction on all regions
    if args.final_prediction_on_all:
        try:
            final_pred_ds = inkid.data.PointsDataset(regions, ['prediction', 'training', 'validation'],
                                                     prediction_features_fn, lambda p: p,
                                                     grid_spacing=args.prediction_grid_spacing)
            final_pred_dl = DataLoader(final_pred_ds, batch_size=args.batch_size * 2, shuffle=False,
                                       num_workers=multiprocessing.cpu_count())
            generate_prediction_image(final_pred_dl, model, output_size, args.label_type, device,
                                      predictions_dir, 'final', reconstruct_fn, regions, args.subvolume_shape)
        # Perform finishing touches even if cut short
        except KeyboardInterrupt:
            pass

    # Add final validation metrics to metadata
    try:
        logging.info('Performing final evaluation on validation set... ')
        val_results = perform_validation(model, val_dl, metrics, device, args.label_type)
        metadata['Final validation metrics'] = inkid.metrics.metrics_dict(val_results)
        logging.info(f'done ({inkid.metrics.metrics_str(val_results)})')
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
    inkid.ops.rclone_transfer_to_remote(args.rclone_transfer_remote, output_path)

    writer.close()


if __name__ == '__main__':
    main()
