"""Train and predict using subvolumes.

This script will read a RegionSet JSON file and create a RegionSet for
training, evaluation, and prediction. It will then run the training
process, evaluating and predicting along the way as defined by the
RegionSet and the parameters file.

It is possible to pass a model directory to the script, in which case
it will use that model as a starting point rather than random
initialization.

The optional value k can be passed in order to use this script for
k-fold cross validation (and prediction in this case). To do that,
create a RegionSet of entirely training regions, and then pass an
index k to this script via the command line argument. It will take the
kth training region, remove it from the training set, and add it to
the prediction and evaluation sets for that run.

"""

from contextlib import ExitStack
import datetime
import functools
import inspect
import json
import os
import random
import re
import subprocess
import time
import timeit

import configargparse
import git
import numpy as np
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
    parser.add('data', metavar='infile', help='input data file (JSON or PPM)', nargs='?')
    parser.add('output', metavar='outfile', help='output directory', nargs='?')

    # Config file
    parser.add('-c', '--config-file', metavar='path', is_config_file=True,
               help='file with pre-specified arguments (in addition to pre-loaded defaults)')

    # Region set modifications
    parser.add('-k', metavar='num', default=None, type=int,
               help='index of region to use for prediction and evaluation')
    parser.add('--override-volume-slices-dir', metavar='path', default=None,
               help='override directory for all volume slices (only works if there is '
               'only one volume in the region set file)')

    # Pretrained model
    parser.add('--model', metavar='path', default=None,
               help='existing model directory to load checkpoints from')

    # Method
    parser.add('--feature-type', metavar='name', default='subvolume_3dcnn',
               help='type of feature model is built on',
               choices=[
                   'subvolume_3dcnn',
                   'voxel_vector_1dcnn',
                   'descriptive_statistics',
               ])
    parser.add('--label-type', metavar='name', default='ink_classes',
               help='type of label to train',
               choices=[
                   'ink_classes',
                   'rgb_values',
               ])

    # Volume
    parser.add('--normalize-volumes', action='store_true',
               help='normalize volumes to zero mean and unit variance before training')

    # Subvolumes
    parser.add('--subvolume-method', metavar='name', default='nearest_neighbor',
               help='method for getting subvolumes',
               choices=[
                   'nearest_neighbor',
                   'interpolated',
                   'snap_to_axis_aligned',
               ])
    parser.add('--subvolume-shape', metavar='n', nargs=3, type=int,
               help='subvolume shape in z y x')
    parser.add('--pad-to-shape', metavar='n', nargs=3, type=int, default=None,
               help='pad subvolume with zeros to be of given shape (default no padding)')
    parser.add('--move-along-normal', metavar='n', type=float,
               help='number of voxels to move along normal before getting a subvolume')
    parser.add('--normalize-subvolumes', action='store_true',
               help='normalize each subvolume to zero mean and unit variance on the fly')

    # Voxel vectors
    parser.add('--length-in-each-direction', metavar='n', type=int,
               help='length of voxel vector in each direction along normal')

    # Data organization/augmentation
    parser.add('--jitter-max', metavar='n', type=int)
    parser.add('--augmentation', action='store_true', dest='augmentation')
    parser.add('--no-augmentation', action='store_false', dest='augmentation')

    # Network architecture
    parser.add('--learning-rate', metavar='n', type=float)
    parser.add('--drop-rate', metavar='n', type=float)
    parser.add('--batch-norm-momentum', metavar='n', type=float)
    parser.add('--no-batch-norm', action='store_true')
    parser.add('--fbeta-weight', metavar='n', type=float)
    parser.add('--filter-size', metavar='n', nargs=3, type=int,
               help='3D convolution filter size')
    parser.add('--filters', metavar='n', nargs='*', type=int,
               help='number of filters for each convolution layer')
    parser.add('--adagrad-optimizer', action='store_true')

    # Run configuration
    parser.add('--training-batch-size', metavar='n', type=int)
    parser.add('--training-max-batches', metavar='n', type=int, default=None)
    parser.add('--training-epochs', metavar='n', type=int, default=None)
    parser.add('--prediction-batch-size', metavar='n', type=int)
    parser.add('--prediction-grid-spacing', metavar='n', type=int,
               help='prediction points will be taken from an NxN grid')
    parser.add('--evaluation-batch-size', metavar='n', type=int)
    parser.add('--evaluation-max-samples', metavar='n', type=int)
    parser.add('--summary-every-n-steps', metavar='n', type=int)
    parser.add('--save-checkpoint-every-n-steps', metavar='n', type=int)
    parser.add('--evaluate-every-n-checkpoints', metavar='n', type=int)
    parser.add('--predict-every-n-checkpoints', metavar='n', type=int)
    parser.add('--final-prediction-on-all', action='store_true')
    parser.add('--skip-training', action='store_true')
    parser.add('--continue-training-from-checkpoint', metavar='path', default=None)
    parser.add('--skip-batches', metavar='n', type=int, default=0)
    parser.add('--training-shuffle-seed', metavar='n', type=int, default=random.randint(0,10000))

    # Profiling
    parser.add('--profile-dir-name', metavar='path', default=None,
               help='dirname to dump TensorFlow profile '
               '(no profile produced if not defined)')
    parser.add('--profile-start-and-end-steps', metavar='n', nargs=2, default=[10, 90],
               help='start and end steps (and dump step) for profiling')

    # Logging/metadata
    parser.add('--eval-metrics-to-write', metavar='metric', nargs='*',
               default=[
                   'area_under_roc_curve',
                   'loss'
               ],
               help='will try the final value for each of these eval metrics and '
               'add it to metadata file.')

    # Rclone
    parser.add('--rclone-transfer-remote', metavar='remote', default=None,
               help='if specified, and if matches the name of one of the directories in '
               'the output path, transfer the results to that rclone remote into the '
               'subpath following the remote name')

    args = parser.parse_args()

    if not args.data and not args.output and not args.continue_training_from_checkpoint:
        parser.print_help()
        return

    if args.continue_training_from_checkpoint is not None:
        # Get previous metadata file
        prev_dir = args.continue_training_from_checkpoint
        prev_metadata_file = os.path.join(args.continue_training_from_checkpoint, 'metadata.json')
        with open(prev_metadata_file) as f:
            prev_metadata = json.load(f)

        # Set all the args to be what they were last time
        prev_args = prev_metadata['Arguments']
        d_args = vars(args)
        for prev_arg in prev_args:
            d_args[prev_arg] = prev_args[prev_arg]

        # Calculate number of batches to drop
        files = os.listdir(prev_dir)
        checkpoint_files = list(filter(lambda name: re.search('model\.ckpt-(\d+)\.index', name) is not None, files))
        iterations = [int(re.findall('model\.ckpt-(\d+)\.index', name)[0]) for name in checkpoint_files]
        max_iteration = max(iterations)
        d_args['skip_batches'] = max_iteration
        print('Skipping {} batches.'.format(max_iteration))

        # Set this again since it got wiped above
        d_args['continue_training_from_checkpoint'] = prev_dir

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

    if args.continue_training_from_checkpoint is not None:
        model_path = prev_dir
    elif args.model is not None:
        model_path = args.model
    else:
        model_path = output_path

    # If input file is a PPM, treat this as a texturing module
    # Skip training, run a prediction on all regions, require trained model, require slices dir
    _, file_extension = os.path.splitext(args.data)
    file_extension = file_extension.lower()
    if file_extension == '.ppm':
        if args.model is None:
            print("Pretrained model (--model) required when texturing a .ppm file.")
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
        region_data['regions']['evaluation'].append(k_region)

    regions = inkid.data.RegionSet(region_data)
    if args.normalize_volumes:
        print('Normalizing volumes...')
        regions.normalize_volumes()
        print('done')

    print('Arguments:\n{}\n'.format(args))
    print('Region Set:\n{}\n'.format(json.dumps(region_data, indent=4, sort_keys=False)))

    # Write metadata to file
    metadata = {}
    metadata['Arguments'] = vars(args)
    metadata['Region set'] = region_data

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
    # (below) runs an evaluation each time this happens.
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
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log,
        every_n_iter=args.summary_every_n_steps,
    )
    tf.logging.set_verbosity(tf.logging.INFO)

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
        evaluation_features_fn = functools.partial(
            point_to_subvolume_input,
            augment_subvolume=False,
            jitter_max=0,
        )
        prediction_features_fn = evaluation_features_fn
    elif args.feature_type == 'voxel_vector_1dcnn':
        training_features_fn = functools.partial(
            regions.point_to_voxel_vector_input,
            length_in_each_direction=args.length_in_each_direction,
            out_of_bounds='all_zeros',
        )
        evaluation_features_fn = training_features_fn
        prediction_features_fn = training_features_fn
    elif args.feature_type == 'descriptive_statistics':
        training_features_fn = functools.partial(
            regions.point_to_descriptive_statistics,
            subvolume_shape=args.subvolume_shape,
        )
        evaluation_features_fn = training_features_fn
        prediction_features_fn = training_features_fn

    # Define the labels
    if args.label_type == 'ink_classes':
        label_fn = regions.point_to_ink_classes_label
    elif args.label_type == 'rgb_values':
        label_fn = regions.point_to_rgb_values_label

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
    evaluation_input_fn = regions.create_tf_input_fn(
        region_groups=['evaluation'],
        batch_size=args.evaluation_batch_size,
        features_fn=evaluation_features_fn,
        label_fn=label_fn,
        max_samples=args.evaluation_max_samples,
        perform_shuffle=True,
        shuffle_seed=0,  # We want the eval set to be the same each time
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
        with ExitStack() as stack:
            # Only do profiling if user provided a profile file path
            # https://stackoverflow.com/questions/27803059/conditional-with-statement-in-python?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
            if args.profile_dir_name is not None:
                print('Enabling TensorFlow profiling...')
                pctx = stack.enter_context(
                    tf.contrib.tfprof.ProfileContext(
                        'tmp',
                        trace_steps=range(
                            args.profile_start_and_end_steps[0],
                            args.profile_start_and_end_steps[1]
                        ),
                        dump_steps=[args.profile_start_and_end_steps[1]]
                    )
                )

                opts = tf.profiler.ProfileOptionBuilder.time_and_memory()
                opts2 = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
                builder = tf.profiler.ProfileOptionBuilder
                opts3 = builder(builder.time_and_memory()).order_by('micros').build()
                pctx.add_auto_profiling(
                    'op',
                    opts,
                    [args.profile_start_and_end_steps[0], args.profile_start_and_end_steps[1]]
                )
                pctx.add_auto_profiling(
                    'scope',
                    opts2,
                    [args.profile_start_and_end_steps[0], args.profile_start_and_end_steps[1]]
                )
                pctx.add_auto_profiling(
                    'op',
                    opts3,
                    [args.profile_start_and_end_steps[0], args.profile_start_and_end_steps[1]]
                )

            # Only train if the training region set group is not empty
            if len(regions._region_groups['training']) > 0 and not args.skip_training:
                estimator.train(
                    input_fn=training_input_fn,
                    steps=args.training_max_batches,
                    hooks=[logging_hook],
                    saving_listeners=[
                        inkid.model.EvalCheckpointSaverListener(
                            estimator=estimator,
                            eval_input_fn=evaluation_input_fn,
                            predict_input_fn=prediction_input_fn,
                            evaluate_every_n_checkpoints=args.evaluate_every_n_checkpoints,
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
                region_groups=['prediction', 'training', 'evaluation'],
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
        eval_event_acc = EventAccumulator(os.path.join(output_path, 'eval'))
        eval_event_acc.Reload()
        for metric in args.eval_metrics_to_write:
            if metric in eval_event_acc.Tags()['scalars']:
                metrics[metric] = eval_event_acc.Scalars(metric)[-1].value
    except:  # NOQA
        pass
    metadata['Final evaluation metrics'] = metrics

    with open(os.path.join(output_path, 'metadata.json'), 'w') as f:
        f.write(json.dumps(metadata, indent=4, sort_keys=False))

    # Transfer via rclone if requested
    if args.rclone_transfer_remote is not None:
        folders = []
        path = os.path.abspath(output_path)
        while True:
            path, folder = os.path.split(path)
            if folder != "":
                folders.append(folder)
            else:
                if path != "":
                    folders.append(path)
                break
        folders.reverse()

        if args.rclone_transfer_remote not in folders:
            print('Provided rclone transfer remote was not a directory '
                  'name in the output path, so it is not clear where in the '
                  'remote to put the files. Transfer canceled.')
        else:
            while folders.pop(0) != args.rclone_transfer_remote:
                continue

            command = [
                'rclone',
                'move',
                '-v',
                '--delete-empty-src-dirs',
                output_path,
                args.rclone_transfer_remote + ':' + os.path.join(*folders)
            ]
            print(' '.join(command))
            subprocess.call(command)


if __name__ == '__main__':
    main()
