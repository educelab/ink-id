"""Train and predict using subvolumes.

This script will read a RegionSet JSON file and create a RegionSet for
training, validation, and prediction. It will then run the training
process, validating and predicting along the way as defined by the
RegionSet and the parameters file.

It is possible to pass a model directory to the script, in which case
it will use that model as a starting point rather than random
initialization.

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
import json
import multiprocessing
import os
import sys
import time
import timeit

import configargparse
import git
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary

import inkid


def perform_validation(model, dataloader, metrics, device, label_type):
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
                              reconstruct_fn, region_set):
    predictions = np.empty(shape=(0, output_size))
    points = np.empty(shape=(0, 3))
    model.eval()  # Turn off training mode for batch norm and dropout purposes
    with torch.no_grad():
        for pxb, pts in dataloader:
            pred = model(pxb.to(device))
            if label_type == 'ink_classes':
                pred = F.softmax(pred, dim=1)
            pred = pred.cpu().numpy()
            pts = pts.numpy()
            predictions = np.append(predictions, pred, axis=0)
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

    # Config file
    parser.add_argument('-c', '--config-file', metavar='path', is_config_file=True,
                        help='file of pre-specified arguments (in addition to pre-loaded defaults)')

    # Region set modifications
    parser.add_argument('-k', metavar='num', default=None, type=int,
                        help='index of region to use for prediction and validation')
    parser.add_argument('--override-volume-slices-dir', metavar='path', default=None,
                        help='override directory for all volume slices (only works if there is '
                             'only one volume in the region set file)')

    # Pre-trained model
    # TODO(pytorch) maybe change vocabulary/structure of this dir
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

    # Subvolumes
    parser.add_argument('--subvolume-method', metavar='name', default='nearest_neighbor',
                        help='method for getting subvolumes',
                        choices=[
                            'nearest_neighbor',
                            'interpolated',
                            'snap_to_axis_aligned',
                        ])
    parser.add_argument('--subvolume-shape', metavar='n', nargs=3, type=int,
                        help='subvolume shape in z y x')
    parser.add_argument('--pad-to-shape', metavar='n', nargs=3, type=int, default=None,
                        help='pad subvolume with zeros to be of given shape (default no padding)')
    parser.add_argument('--move-along-normal', metavar='n', type=float,
                        help='number of voxels to move along normal before getting a subvolume')
    parser.add_argument('--normalize-subvolumes', action='store_true',
                        help='normalize each subvolume to zero mean and unit variance on the fly')

    # Frequency domain transforms for subvolume
    parser.add_argument('--fft', action='store_true', help='Apply FFT to subvolumes')
    parser.add_argument('--dwt', metavar='name', default=None, help='Apply specified DWT to subvolumes')

    # Voxel vectors
    parser.add_argument('--length-in-each-direction', metavar='n', type=int,
                        help='length of voxel vector in each direction along normal')

    # Data organization/augmentation
    parser.add_argument('--jitter-max', metavar='n', type=int)
    parser.add_argument('--augmentation', action='store_true', dest='augmentation')
    parser.add_argument('--no-augmentation', action='store_false', dest='augmentation')

    # Network architecture
    # TODO(pytorch) make sure these accounted for/changed if needed
    parser.add_argument('--learning-rate', metavar='n', type=float)
    parser.add_argument('--drop-rate', metavar='n', type=float)
    parser.add_argument('--batch-norm-momentum', metavar='n', type=float)
    parser.add_argument('--no-batch-norm', action='store_true')
    parser.add_argument('--fbeta-weight', metavar='n', type=float)
    parser.add_argument('--filter-size', metavar='n', nargs=3, type=int,
                        help='3D convolution filter size')
    parser.add_argument('--filters', metavar='n', nargs='*', type=int,
                        help='number of filters for each convolution layer')
    parser.add_argument('--adagrad-optimizer', action='store_true')
    parser.add_argument('--decay-steps', metavar='n', type=int, default=None)
    parser.add_argument('--decay-rate', metavar='n', type=float, default=None)

    # Run configuration
    # TODO(pytorch) make sure these accounted for/changed if needed
    parser.add_argument('--batch-size', metavar='n', type=int)
    parser.add_argument('--training-max-batches', metavar='n', type=int, default=None)
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
        output_path = os.path.join(
            args.output,
            datetime.datetime.today().strftime('%Y-%m-%d_%H.%M.%S')
        )
    else:
        output_path = os.path.join(
            args.output,
            datetime.datetime.today().strftime('%Y-%m-%d_%H.%M.%S') + '_' + str(args.k)
        )
    os.makedirs(output_path)

    # Point to preexisting model path if there is one
    if args.model is not None:
        model_path = args.model
    else:
        model_path = output_path

    # Define directory for prediction images
    predictions_dir = os.path.join(output_path, 'predictions')

    # If input file is a PPM, treat this as a texturing module
    # Skip training, run a prediction on all regions, require trained model, require slices dir
    _, file_extension = os.path.splitext(args.data)
    file_extension = file_extension.lower()
    if file_extension == '.ppm':
        if args.model is None:
            print("Pre-trained model (--model) required when texturing a .ppm file.")
            return
        if args.override_volume_slices_dir is None:
            print("Volume (--override-volume-slices-dir) required when texturing a .ppm file.")
            return
        print("PPM input file provided. Skipping training and running final prediction on all.")
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

    # Diagnostic printing
    print('Arguments:\n{}\n'.format(args))
    print('Region Set:\n{}\n'.format(json.dumps(region_data, indent=4, sort_keys=False)))

    # Create metadata dict
    metadata = {'Arguments': vars(args), 'Region set': region_data, 'Command': ' '.join(sys.argv)}

    # Add git hash to metadata if inside a git repository
    try:
        repo = git.Repo(os.path.join(os.path.dirname(inspect.getfile(inkid)), '..'))
        sha = repo.head.object.hexsha
        metadata['Git hash'] = repo.git.rev_parse(sha, short=6)
    except git.exc.InvalidGitRepositoryError:
        metadata['Git hash'] = 'No git hash available (unable to find valid repository).'

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
            fft=args.fft,
            dwt=args.dwt,
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
        print('Feature type not recognized: {}'.format(args.feature_type))
        return

    # Define the labels
    if args.label_type == 'ink_classes':
        label_fn = regions.point_to_ink_classes_label
        output_size = 2
        metrics = {
            'loss': nn.CrossEntropyLoss(reduction='mean'),
            'accuracy': inkid.metrics.accuracy,
            'precision': inkid.metrics.precision,
            'recall': inkid.metrics.recall,
            'fbeta': inkid.metrics.fbeta,
            'auc': inkid.metrics.auc
        }
        reconstruct_fn = regions.reconstruct_predicted_ink_classes
    elif args.label_type == 'rgb_values':
        label_fn = regions.point_to_rgb_values_label
        output_size = 3
        metrics = {
            'loss': nn.SmoothL1Loss(reduction='mean')
        }
        reconstruct_fn = regions.reconstruct_predicted_rgb
    else:
        print('Label type not recognized: {}'.format(args.label_type))
        return

    # Define the datasets
    train_ds = inkid.data.PointsDataset(regions, ['training'], training_features_fn, label_fn)
    val_ds = inkid.data.PointsDataset(regions, ['validation'], validation_features_fn, label_fn)
    # Only take n samples for validation, not the entire region
    if args.validation_max_samples < len(val_ds):
        val_ds = torch.utils.data.random_split(
            val_ds,
            [args.validation_max_samples, len(val_ds) - args.validation_max_samples]
            )[0]
    pred_ds = inkid.data.PointsDataset(regions, ['prediction'], prediction_features_fn, lambda p: p,
                                       grid_spacing=args.prediction_grid_spacing)

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                           num_workers=multiprocessing.cpu_count())
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=True,
                                         num_workers=multiprocessing.cpu_count())
    pred_dl = torch.utils.data.DataLoader(pred_ds, batch_size=args.batch_size * 2, shuffle=False,
                                          num_workers=multiprocessing.cpu_count())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create the model for training
    if args.feature_type == 'subvolume_3dcnn':
        model = inkid.model.Subvolume3DcnnModel(
            args.drop_rate, args.subvolume_shape, args.pad_to_shape,
            args.batch_norm_momentum, args.no_batch_norm, args.filters, output_size)
    else:
        print('Feature type: {} does not yet have a PyTorch model.'.format(args.feature_type))
        return
    model = model.to(device)
    # Print summary of model
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    shape = (1,) + tuple(args.pad_to_shape or args.subvolume_shape)
    torchsummary.summary(model, input_size=shape, batch_size=args.batch_size, device=device_str)
    # Define optimizer
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    metric_results = {metric: [] for metric in metrics}
    last_summary = time.time()

    # Run training loop
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
                    print('Batch: {:>5d}/{:<5d} {} Seconds: {:5.3g}'.format(
                        batch_num, total_batches,
                        inkid.metrics.metrics_str(metric_results),
                        time.time() - last_summary))
                    for result in metric_results.values():
                        result.clear()
                    last_summary = time.time()

                if batch_num % args.checkpoint_every_n_batches == 0:
                    # Periodic evaluation and prediction
                    print('Evaluating on validation set... ', end='')
                    val_results = perform_validation(model, val_dl, metrics, device, args.label_type)
                    print(f'done ({inkid.metrics.metrics_str(val_results)})')

                    # Prediction image
                    print('Generating prediction image... ', end='')
                    generate_prediction_image(pred_dl, model, output_size, args.label_type, device,
                                              predictions_dir, f'{epoch}_{batch_num}', reconstruct_fn, regions)
                    print('done')
    except KeyboardInterrupt:
        pass

    # Run a final prediction on all regions
    if args.final_prediction_on_all:
        try:
            final_pred_ds = inkid.data.PointsDataset(regions, ['prediction', 'training', 'validation'],
                                                     prediction_features_fn, lambda p: p,
                                                     grid_spacing=args.prediction_grid_spacing)
            final_pred_dl = torch.utils.data.DataLoader(final_pred_ds, batch_size=args.batch_size * 2, shuffle=False,
                                                        num_workers=multiprocessing.cpu_count())
            generate_prediction_image(final_pred_dl, model, output_size, args.label_type, device,
                                      predictions_dir, 'final', reconstruct_fn, regions)
        # Perform finishing touches even if cut short
        except KeyboardInterrupt:
            pass

    # Add final validation metrics to metadata
    try:
        print('Performing final evaluation on validation set... ', end='')
        val_results = perform_validation(model, val_dl, metrics, device, args.label_type)
        metadata['Final validation metrics'] = inkid.metrics.metrics_dict(val_results)
        print(f'done ({inkid.metrics.metrics_str(val_results)})')
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


if __name__ == '__main__':
    main()
