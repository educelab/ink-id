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

from contextlib import ExitStack
import datetime
import functools
import inspect
import json
import os
import random
import time
import timeit

import configargparse
import git
import numpy as np

# TODO(PyTorch) remove
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import tensorflow as tf

import inkid


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
                        help='file with pre-specified arguments (in addition to pre-loaded defaults)')

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
    parser.add_argument('--training-batch-size', metavar='n', type=int)
    parser.add_argument('--training-max-batches', metavar='n', type=int, default=None)
    parser.add_argument('--training-epochs', metavar='n', type=int, default=None)
    parser.add_argument('--prediction-batch-size', metavar='n', type=int)
    parser.add_argument('--prediction-grid-spacing', metavar='n', type=int,
                        help='prediction points will be taken from an NxN grid')
    parser.add_argument('--validation-batch-size', metavar='n', type=int)
    parser.add_argument('--validation-max-samples', metavar='n', type=int)
    parser.add_argument('--summary-every-n-steps', metavar='n', type=int)
    parser.add_argument('--save-checkpoint-every-n-steps', metavar='n', type=int)
    parser.add_argument('--validate-every-n-checkpoints', metavar='n', type=int)
    parser.add_argument('--predict-every-n-checkpoints', metavar='n', type=int)
    parser.add_argument('--final-prediction-on-all', action='store_true')
    parser.add_argument('--skip-training', action='store_true')
    parser.add_argument('--skip-batches', metavar='n', type=int, default=0)
    parser.add_argument('--training-shuffle-seed', metavar='n', type=int, default=random.randint(0, 10000))

    # Logging/metadata
    # TODO(pytorch) make sure these accounted for/changed if needed
    parser.add_argument('--val-metrics-to-write', metavar='metric', nargs='*',
                        default=[
                            'area_under_roc_curve',
                            'loss'
                        ],
                        help='will try the final value for each of these val metrics and '
                             'add it to metadata file.')

    # Rclone
    parser.add_argument('--rclone-transfer-remote', metavar='remote', default=None,
                        help='if specified, and if matches the name of one of the directories in '
                             'the output path, transfer the results to that rclone remote into the '
                             'subpath following the remote name')

    args = parser.parse_args()

    if args.data is None and args.output is None:
        parser.print_help()
        return

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

    if args.model is not None:
        model_path = args.model
    else:
        model_path = output_path

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
        print("PPM input file provided. Automatically skipping training and running a final prediction on all.")
        args.skip_training = True
        args.final_prediction_on_all = True

    # Transform the input file into region set, can handle JSON or PPM
    region_data = inkid.data.RegionSet.get_data_from_file(args.data)

    if args.override_volume_slices_dir is not None:
        volume_dirs_seen = set()
        for ppm in region_data['ppms']:
            volume_dirs_seen.add(region_data['ppms'][ppm]['volume'])
            if len(volume_dirs_seen) > 1:
                raise ValueError('--override-volume-slices-dir only '
                                 'permitted if there is one volume in the region set')
        for ppm in region_data['ppms']:
            region_data['ppms'][ppm]['volume'] = args.override_volume_slices_dir

    if args.k is not None:
        k_region = region_data['regions']['training'].pop(int(args.k))
        region_data['regions']['prediction'].append(k_region)
        region_data['regions']['validation'].append(k_region)

    regions = inkid.data.RegionSet(region_data)

    print('Arguments:\n{}\n'.format(args))
    print('Region Set:\n{}\n'.format(json.dumps(region_data, indent=4, sort_keys=False)))

    # Write metadata to file
    metadata = {'Arguments': vars(args), 'Region set': region_data}

    # Add the git hash if there is a repository
    try:
        repo = git.Repo(os.path.join(os.path.dirname(inspect.getfile(inkid)), '..'))
        sha = repo.head.object.hexsha
        metadata['Git hash'] = repo.git.rev_parse(sha, short=6)
    except git.exc.InvalidGitRepositoryError:
        metadata['Git hash'] = 'No git hash available (unable to find valid repository).'

    with open(os.path.join(output_path, 'metadata.json'), 'w') as f:
        f.write(json.dumps(metadata, indent=4, sort_keys=False))

    # Save checkpoints every n steps. EvalCheckpointSaverListener
    # (below) runs a validation each time this happens.
    run_config = tf.estimator.RunConfig(
        save_checkpoints_steps=args.save_checkpoint_every_n_steps,
        keep_checkpoint_max=None,  # save all checkpoints
    )

    # Create an Estimator with the run configuration, hyperparameters,
    # and model directory specified.
    estimator = tf.estimator.Estimator(
        model_fn={
            'ink_classes': inkid.model.ink_classes_model_fn,
            'rgb_values': inkid.model.rgb_values_model_fn,
        }[args.label_type],
        model_dir=model_path,
        config=run_config,
        params={
            'drop_rate': args.drop_rate,
            'subvolume_shape': args.subvolume_shape,
            'pad_to_shape': args.pad_to_shape,
            'length_in_each_direction': args.length_in_each_direction,
            'batch_norm_momentum': args.batch_norm_momentum,
            'no_batch_norm': args.no_batch_norm,
            'filters': args.filters,
            'learning_rate': args.learning_rate,
            'fbeta_weight': args.fbeta_weight,
            'feature_type': args.feature_type,
            'label_type': args.label_type,
            'adagrad_optimizer': args.adagrad_optimizer,
            'decay_steps': args.decay_steps,
            'decay_rate': args.decay_rate,
        },
    )

    # Define tensors to be shown in a "summary" step.
    if args.label_type == 'ink_classes':
        tensors_to_log = {
            'train_accuracy': 'train_accuracy',
            'train_precision': 'train_precision',
            'train_recall': 'train_recall',
            'train_fbeta_score': 'train_fbeta_score',
            'train_positives': 'train_positives',
            'train_negatives': 'train_negatives',
        }
    else:
        tensors_to_log = {}
    logging_hook = tf.estimator.LoggingTensorHook(
        tensors=tensors_to_log,
        every_n_iter=args.summary_every_n_steps,
    )
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

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
    elif args.label_type == 'rgb_values':
        label_fn = regions.point_to_rgb_values_label
    else:
        print('Label type not recognized: {}'.format(args.label_type))
        return

    # Define the datasets
    training_input_fn = regions.create_tf_input_fn(
        region_groups=['training'],
        batch_size=args.training_batch_size,
        features_fn=training_features_fn,
        label_fn=label_fn,
        perform_shuffle=True,
        shuffle_seed=args.training_shuffle_seed,
        restrict_to_surface=True,
        epochs=args.training_epochs,
        skip_batches=args.skip_batches,
    )
    validation_input_fn = regions.create_tf_input_fn(
        region_groups=['validation'],
        batch_size=args.validation_batch_size,
        features_fn=validation_features_fn,
        label_fn=label_fn,
        max_samples=args.validation_max_samples,
        perform_shuffle=True,
        shuffle_seed=0,  # We want the val set to be the same each time
        restrict_to_surface=True,
    )
    prediction_input_fn = regions.create_tf_input_fn(
        region_groups=['prediction'],
        batch_size=args.prediction_batch_size,
        features_fn=prediction_features_fn,
        label_fn=None,
        perform_shuffle=False,
        restrict_to_surface=True,
        grid_spacing=args.prediction_grid_spacing,
    )

    # Run the training process. Predictions are run during training
    # and also after training.
    try:
        with ExitStack():
            # Only train if the training region set group is not empty
            if len(regions.region_groups['training']) > 0 and not args.skip_training:
                estimator.train(
                    input_fn=training_input_fn,
                    steps=args.training_max_batches,
                    hooks=[logging_hook],
                    saving_listeners=[
                        inkid.model.EvalCheckpointSaverListener(
                            estimator=estimator,
                            val_input_fn=validation_input_fn,
                            predict_input_fn=prediction_input_fn,
                            validate_every_n_checkpoints=args.validate_every_n_checkpoints,
                            predict_every_n_checkpoints=args.predict_every_n_checkpoints,
                            region_set=regions,
                            predictions_dir=os.path.join(output_path, 'predictions'),
                            label_type=args.label_type,
                        ),
                    ],
                )

    # Still attempt final prediction
    except KeyboardInterrupt:
        pass

    try:
        if args.final_prediction_on_all:
            print('Running a final prediction on all regions...')
            final_prediction_input_fn = regions.create_tf_input_fn(
                region_groups=['prediction', 'training', 'validation'],
                batch_size=args.prediction_batch_size,
                features_fn=prediction_features_fn,
                label_fn=None,
                perform_shuffle=False,
                restrict_to_surface=True,
                grid_spacing=args.prediction_grid_spacing,
            )

            if args.label_type == 'ink_classes':
                predictions = estimator.predict(
                    final_prediction_input_fn,
                    predict_keys=[
                        'region_id',
                        'ppm_xy',
                        'probabilities',
                    ],
                )

                for prediction in predictions:
                    regions.reconstruct_predicted_ink_classes(
                        np.array([prediction['region_id']]),
                        np.array([prediction['probabilities']]),
                        np.array([prediction['ppm_xy']]),
                    )

                regions.save_predictions(os.path.join(output_path, 'predictions'), 'final')
                regions.reset_predictions()

            elif args.label_type == 'rgb_values':
                predictions = estimator.predict(
                    final_prediction_input_fn,
                    predict_keys=[
                        'region_id',
                        'ppm_xy',
                        'rgb',
                    ],
                )

                for prediction in predictions:
                    regions.reconstruct_predicted_rgb(
                        np.array([prediction['region_id']]),
                        np.array([prediction['rgb']]),
                        np.array([prediction['ppm_xy']]),
                    )

                regions.save_predictions(os.path.join(output_path, 'predictions'), 'final')
                regions.reset_predictions()

    # Perform finishing touches even if cut short
    except KeyboardInterrupt:
        pass

    # Add some post-run info to metadata file
    stop = timeit.default_timer()
    metadata['Runtime'] = stop - start
    metadata['Finished at'] = time.strftime('%Y-%m-%d %H:%M:%S')

    # Add some final metrics
    metrics = {}
    try:
        val_event_acc = EventAccumulator(os.path.join(output_path, 'val'))
        val_event_acc.Reload()
        for metric in args.val_metrics_to_write:
            if metric in val_event_acc.Tags()['scalars']:
                metrics[metric] = val_event_acc.Scalars(metric)[-1].value
    except KeyboardInterrupt:
        pass
    metadata['Final validation metrics'] = metrics

    with open(os.path.join(output_path, 'metadata.json'), 'w') as f:
        f.write(json.dumps(metadata, indent=4, sort_keys=False))

    # Transfer via rclone if requested
    inkid.ops.rclone_transfer_to_remote(args.rclone_transfer_remote, output_path)


if __name__ == '__main__':
    main()
