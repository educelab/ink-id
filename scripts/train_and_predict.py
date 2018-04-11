"""
Train an ink classifier and produce predicted output for a volume.
"""

import argparse
import datetime
import functools
import os
import time

import tensorflow as tf
import numpy as np

import inkid.model
import inkid.ops
import inkid.data


def main():
    """Run the training and prediction process."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--data', metavar='path', required=True,
                        help='input data file (JSON)')
    parser.add_argument('-o', '--output', metavar='path', default='out',
                        help='output directory')
    parser.add_argument('-m', '--model', metavar='path', default=None,
                        help='existing model directory to start with')

    args = parser.parse_args()
    output_path = os.path.join(
        args.output,
        datetime.datetime.today().strftime('%Y-%m-%d_%H.%M.%S')
    )
    if args.model is not None:
        model_path = args.model
    else:
        model_path = output_path

    params = inkid.ops.load_default_parameters()
    regions = inkid.data.RegionSet.from_json(args.data)
    
    # Save checkpoints every n steps. EvalCheckpointSaverListener
    # (below) runs an evaluation each time this happens.
    run_config = tf.estimator.RunConfig(
        save_checkpoints_steps=params['evaluate_every_n_steps'],
        keep_checkpoint_max=None, # save all checkpoints
    )

    # Create an Estimator with the run configuration, hyperparameters,
    # and model directory specified.
    estimator = tf.estimator.Estimator(
        model_fn=inkid.model.model_fn_3dcnn,
        model_dir=model_path,
        config=run_config,
        params={
            'drop_rate': params['drop_rate'],
            'subvolume_shape': params['subvolume_shape'],
            'batch_norm_momentum': params['batch_norm_momentum'],
            'filters': params['filters'],
            'learning_rate': params['learning_rate'],
            'fbeta_weight': params['fbeta_weight'],
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

    point_to_subvolume_input = functools.partial(
        regions.point_to_subvolume_input,
        subvolume_shape=params['subvolume_shape'],
    )

    training_input_fn = regions.create_tf_input_fn(
        region_groups=['training'],
        batch_size=params['training_batch_size'],
        features_fn=point_to_subvolume_input,
        label_fn=regions.point_to_ink_classes_label,
        perform_shuffle=True,
        restrict_to_surface=True,
    )

    evaluation_input_fn = regions.create_tf_input_fn(
        region_groups=['evaluation'],
        batch_size=params['evaluation_batch_size'],
        features_fn=point_to_subvolume_input,
        label_fn=regions.point_to_ink_classes_label,
        max_samples=params['evaluation_max_samples'],
        perform_shuffle=True,
        restrict_to_surface=True,
    )

    prediction_input_fn = regions.create_tf_input_fn(
        region_groups=['prediction'],
        batch_size=params['prediction_batch_size'],
        features_fn=point_to_subvolume_input,
        label_fn=None,
        perform_shuffle=False,
        restrict_to_surface=True,
        grid_spacing=params['prediction_grid_spacing'],
    )

    # Run the training process.
    estimator.train(
        input_fn=training_input_fn,
        steps=params.get('training_max_batches'),
        hooks=[logging_hook],
        saving_listeners=[
            inkid.model.EvalCheckpointSaverListener(
                estimator=estimator,
                eval_input_fn=evaluation_input_fn,
                predict_input_fn=prediction_input_fn,
                predict_every_n_steps=params['predict_every_n_steps'],
                region_set=regions,
            ),
        ],
    )

    # predictions = estimator.predict(
    #     input_fn=lambda: volumes.prediction_input_fn(params['prediction_batch_size'])
    # )

    # for prediction in predictions:
    #     volumes.reconstruct(params, np.array([prediction['probabilities']]), np.array([[prediction['XYZcoordinate'][0], prediction['XYZcoordinate'][1], 0]]))
    # volumes.saveAllPredictions(params, 0)

    # TODO write runtime to file
    
if __name__ == '__main__':
    main()
