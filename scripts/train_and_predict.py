"""
Train an ink classifier and produce predicted output for a volume.
"""

import argparse
import datetime
import os
import time

import tensorflow as tf
import numpy as np

from inkid.volumes import VolumeSet
import inkid.model
import inkid.ops
import inkid.data


def main():
    """Run the training and prediction process."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--data', metavar='path', required=True,
                        help='path to input data file (JSON)')
    parser.add_argument('-o', '--output', metavar='path', default='out',
                        help='path to output directory')

    args = parser.parse_args()

    # Load default parameters
    params = inkid.ops.load_default_parameters()

    # Adjust some parameters from supplied arguments
    params['output_path'] = os.path.join( # TODO make this an optional input argument so we can pick up from previous training
        args.output,
        '3dcnn-predictions',
        datetime.datetime.today().strftime('%Y-%m-%d_%H.%M.%S')
    )

    regions = inkid.data.RegionSet.from_json(args.data)

    # volumes = VolumeSet(params)

    # # Save checkpoints every n steps. EvalCheckpointSaverListener
    # # (below) runs an evaluation each time this happens.
    # run_config = tf.estimator.RunConfig(
    #     save_checkpoints_steps=params['evaluate_every_n_steps'],
    #     keep_checkpoint_max=None, # save all checkpoints
    # )

    # # Create an Estimator with the run configuration, hyperparameters,
    # # and model directory specified.
    # estimator = tf.estimator.Estimator(
    #     model_fn=inkid.model.model_fn_3dcnn,
    #     model_dir=params['output_path'],
    #     config=run_config,
    #     params={
    #         'drop_rate': params['drop_rate'],
    #         'subvolume_shape': params['subvolume_shape'],
    #         'batch_norm_momentum': params['batch_norm_momentum'],
    #         'filters': params['filters'],
    #         'learning_rate': params['learning_rate'],
    #         'fbeta_weight': params['fbeta_weight'],
    #     },
    # )

    # # Define tensors to be shown in a "display" step.
    # tensors_to_log = {
    #     'train_accuracy': 'train_accuracy',
    #     'train_precision': 'train_precision',
    #     'train_recall': 'train_recall',
    #     'train_fbeta_score': 'train_fbeta_score',
    #     'train_positives': 'train_positives',
    #     'train_negatives': 'train_negatives',
    # }
    # logging_hook = tf.train.LoggingTensorHook(
    #     tensors=tensors_to_log,
    #     every_n_iter=params['display_every_n_steps'],
    # )
    # tf.logging.set_verbosity(tf.logging.INFO)

    # # Run the training process.
    # estimator.train(
    #     input_fn=lambda: volumes.training_input_fn(
    #         params['training_batch_size'],
    #     ),
    #     # steps=params['training_steps'],  # If not defined, will run through entire training dataset.
    #     hooks=[logging_hook],
    #     saving_listeners=[
    #         inkid.model.EvalCheckpointSaverListener(
    #             estimator=estimator,
    #             eval_input_fn=lambda: volumes.evaluation_input_fn(
    #                 params['evaluation_batch_size'],
    #             ),
    #             predict_input_fn=lambda: volumes.prediction_input_fn(
    #                 params['prediction_batch_size'],
    #             ),
    #             predict_every_n_steps=params['predict_every_n_steps'],
    #             volume_set=volumes,
    #             args=params,
    #         ),
    #     ],
    # )

    # predictions = estimator.predict(
    #     input_fn=lambda: volumes.prediction_input_fn(params['prediction_batch_size'])
    # )

    # for prediction in predictions:
    #     volumes.reconstruct(params, np.array([prediction['probabilities']]), np.array([[prediction['XYZcoordinate'][0], prediction['XYZcoordinate'][1], 0]]))
    # volumes.saveAllPredictions(params, 0)

    # TODO write runtime to file
    
if __name__ == '__main__':
    main()
