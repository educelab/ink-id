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

import argparse
from contextlib import ExitStack
import datetime
import functools
import inspect
import json
import os
import subprocess
import time
import timeit

import git
import numpy as np
import tensorflow as tf

import inkid


def main():
    """Run the training and prediction process."""
    start = timeit.default_timer()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', metavar='path', help='input data file (JSON)')
    parser.add_argument('output', metavar='path', help='output directory')
    parser.add_argument('-m', '--model', metavar='path', default=None,
                        help='existing model directory to start with')
    parser.add_argument('--model-type', metavar='name', default='subvolume_3dcnn',
                        help='type of model to train (subvolume_3dcnn or voxel_vector_1dcnn)')
    parser.add_argument('--subvolume-method', metavar='name', default='snap_to_axis_aligned',
                        help='method for getting subvolumes')
    parser.add_argument('-k', metavar='num', default=None,
                        help='index of region to use for prediction and evaluation')
    parser.add_argument('--final-prediction-on-all', action='store_true')
    parser.add_argument('--override-volume-slices-dir', metavar='path', default=None,
                        help='override directory for all volume slices (only works if there is '
                        'only one volume in the region set file)')

    parser.add_argument('--profile-dir-name', metavar='path', default=None,
                        help='dirname to dump TensorFlow profile '
                        '(no profile produced if not defined)')
    parser.add_argument('--profile-start-and-end-steps', metavar='num', nargs=2, default=[10, 90],
                        help='start and end steps (and dump step) for profiling')

    parser.add_argument('--rclone-transfer-remote', metavar='remote', default=None,
                        help='if specified, and if matches the name of one of the directories in '
                        'the output path, transfer the results to that rclone remote into the '
                        'subpath following the remote name')

    args = parser.parse_args()
    if args.k is None:
        output_path = os.path.join(
            args.output,
            datetime.datetime.today().strftime('%Y-%m-%d_%H.%M.%S')
        )
    else:
        output_path = os.path.join(
            args.output,
            datetime.datetime.today().strftime('%Y-%m-%d_%H.%M.%S') + '_' + args.k
        )
    os.makedirs(output_path)
    if args.model is not None:
        model_path = args.model
    else:
        model_path = output_path

    region_data = inkid.data.RegionSet.get_data_from_json(args.data)

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

    params = inkid.ops.load_default_parameters()

    print('Parameters:\n{}\n'.format(json.dumps(params, indent=4, sort_keys=True)))
    print('Region Set:\n{}\n'.format(json.dumps(region_data, indent=4, sort_keys=False)))

    # Save checkpoints every n steps. EvalCheckpointSaverListener
    # (below) runs an evaluation each time this happens.
    run_config = tf.estimator.RunConfig(
        save_checkpoints_steps=params['save_checkpoint_every_n_steps'],
        keep_checkpoint_max=None,  # save all checkpoints
    )

    # Create an Estimator with the run configuration, hyperparameters,
    # and model directory specified.
    estimator = tf.estimator.Estimator(
        model_fn=inkid.model.model_fn,
        model_dir=model_path,
        config=run_config,
        params={
            'drop_rate': params['drop_rate'],
            'subvolume_shape': params['subvolume_shape'],
            'length_in_each_direction': params['length_in_each_direction'],
            'batch_norm_momentum': params['batch_norm_momentum'],
            'filters': params['filters'],
            'learning_rate': params['learning_rate'],
            'fbeta_weight': params['fbeta_weight'],
            'model': args.model_type,
        },
    )

    # Define tensors to be shown in a "summary" step.
    tensors_to_log = {
        'train_accuracy': 'train_accuracy',
        'train_precision': 'train_precision',
        'train_recall': 'train_recall',
        'train_fbeta_score': 'train_fbeta_score',
        'train_positives': 'train_positives',
        'train_negatives': 'train_negatives',
    }
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log,
        every_n_iter=params['summary_every_n_steps'],
    )
    tf.logging.set_verbosity(tf.logging.INFO)

    if args.model_type == 'subvolume_3dcnn':
        point_to_subvolume_input = functools.partial(
            regions.point_to_subvolume_input,
            subvolume_shape=params['subvolume_shape'],
            out_of_bounds='all_zeros',
            move_along_normal=params['move_along_normal'],
            method=args.subvolume_method,
        )
        training_features_fn = functools.partial(
            point_to_subvolume_input,
            augment_subvolume=params['add_augmentation'],
            jitter_max=params['jitter_max'],
        )
        evaluation_features_fn = functools.partial(
            point_to_subvolume_input,
            augment_subvolume=False,
            jitter_max=0,
        )
        prediction_features_fn = evaluation_features_fn
    elif args.model_type == 'voxel_vector_1dcnn':
        training_features_fn = functools.partial(
            regions.point_to_voxel_vector_input,
            length_in_each_direction=params['length_in_each_direction'],
            out_of_bounds='all_zeros',
        )
        evaluation_features_fn = training_features_fn
        prediction_features_fn = training_features_fn
    elif args.model_type == 'descriptive_statistics':
        training_features_fn = functools.partial(
            regions.point_to_descriptive_statistics,
            subvolume_shape=params['subvolume_shape'],
        )
        evaluation_features_fn = training_features_fn
        prediction_features_fn = training_features_fn

    training_input_fn = regions.create_tf_input_fn(
        region_groups=['training'],
        batch_size=params['training_batch_size'],
        features_fn=training_features_fn,
        label_fn=regions.point_to_ink_classes_label,
        perform_shuffle=True,
        restrict_to_surface=True,
    )

    evaluation_input_fn = regions.create_tf_input_fn(
        region_groups=['evaluation'],
        batch_size=params['evaluation_batch_size'],
        features_fn=evaluation_features_fn,
        label_fn=regions.point_to_ink_classes_label,
        max_samples=params['evaluation_max_samples'],
        perform_shuffle=True,
        shuffle_seed=0,  # We want the eval set to be the same each time
        restrict_to_surface=True,
    )

    prediction_input_fn = regions.create_tf_input_fn(
        region_groups=['prediction'],
        batch_size=params['prediction_batch_size'],
        features_fn=prediction_features_fn,
        label_fn=None,
        perform_shuffle=False,
        restrict_to_surface=True,
        grid_spacing=params['prediction_grid_spacing'],
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
            if len(regions._region_groups['training']) > 0:
                estimator.train(
                    input_fn=training_input_fn,
                    steps=params.get('training_max_batches'),
                    hooks=[logging_hook],
                    saving_listeners=[
                        inkid.model.EvalCheckpointSaverListener(
                            estimator=estimator,
                            eval_input_fn=evaluation_input_fn,
                            predict_input_fn=prediction_input_fn,
                            evaluate_every_n_checkpoints=params['evaluate_every_n_checkpoints'],
                            predict_every_n_checkpoints=params['predict_every_n_checkpoints'],
                            region_set=regions,
                            predictions_dir=os.path.join(output_path, 'predictions'),
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
                batch_size=params['prediction_batch_size'],
                features_fn=prediction_features_fn,
                label_fn=None,
                perform_shuffle=False,
                restrict_to_surface=True,
                grid_spacing=params['prediction_grid_spacing'],
            )

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

    # Perform finishing touches even if cut short
    except KeyboardInterrupt:
        pass

    # Write metadata to file
    stop = timeit.default_timer()
    with open(os.path.join(output_path, 'metadata.txt'), 'w') as f:
        f.write('Command Line Arguments:\n{}\n\n'.format(args))
        f.write('Parameters:\n{}\n\n'.format(json.dumps(params, indent=4, sort_keys=True)))
        f.write('Region Set:\n{}\n\n'.format(json.dumps(region_data, indent=4, sort_keys=False)))
        f.write('Runtime:\n{}s\n\n'.format(stop - start))
        f.write('Finished at:\n{}\n\n'.format(time.strftime('%Y-%m-%d %H:%M:%S')))

        # Print out the git hash if there is a repository
        try:
            repo = git.Repo(os.path.join(os.path.dirname(inspect.getfile(inkid)), '..'))
            sha = repo.head.object.hexsha
            f.write('Git hash:\n{}\n\n'.format(repo.git.rev_parse(sha, short=6)))
        except git.exc.InvalidGitRepositoryError:
            f.write('No git hash available (unable to find valid repository).\n\n')

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
                output_path,
                args.rclone_transfer_remote + ':' + os.path.join(*folders)
            ]
            print(' '.join(command))
            subprocess.call(command)


if __name__ == '__main__':
    main()
