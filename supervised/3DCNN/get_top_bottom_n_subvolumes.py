"""
Using a trained model and volume data, get the n subvolumes on that surface
that have the highest and lowest prediction score in the model.
"""

import argparse
import datetime
import os
import time

import numpy as np
import tensorflow as tf

from volumeset import VolumeSet
import ops
import model


def main():
    """Get a prediction value for each subvolume and output the n top and bottom subvolumes."""
    start_time = time.time()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', '-d', metavar='path', required=True,
                        help='path to volume data (slices directory)')
    parser.add_argument('--surfacemask', metavar='path', required=True,
                        help='path to surface mask image')
    parser.add_argument('--surfacedata', metavar='path', required=True,
                        help='path to surface data')
    parser.add_argument('--model', metavar='path', required=True,
                        help='path to trained model')
    parser.add_argument('--number', '-n', metavar='N', default=5, type=int,
                        help='number of subvolumes to keep')
    parser.add_argument('--outputdir', metavar='path', default='out',
                        help='path to output directory')

    args = parser.parse_args()

    # Load default parameters
    params = ops.load_parameters_from_json('default_parameters.json')

    # Adjust some parameters from supplied arguments
    params['volumes'][0]['data_path'] = args.data
    params['volumes'][0]['surface_mask'] = args.surfacemask
    params['volumes'][0]['surface_data'] = args.surfacedata
    params['output_path'] = os.path.join(args.outputdir, '3dcnn-predictions', datetime.datetime.today().strftime('%Y-%m-%d_%H.%M.%S'))

    x = tf.placeholder(tf.float32,
                       [None, params["x_dimension"], params["y_dimension"], params["z_dimension"]])
    y = tf.placeholder(tf.float32, [None, params["n_classes"]])
    drop_rate = tf.placeholder(tf.float32)
    training_flag = tf.placeholder(tf.bool)
    pred, loss = model.build_model(x, y, drop_rate, params, training_flag)
    
    volumes = VolumeSet(params)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, args.model)

        starting_coordinates = [0, 0, 0]
        prediction_samples, coordinates, next_coordinates = volumes.getPredictionBatch(
            params, starting_coordinates)

        top_n_predictions_and_volumes = []
        bottom_n_predictions_and_volumes = []

        try:
            while next_coordinates is not None:
                prediction_values = sess.run(pred, feed_dict={
                    x: prediction_samples,
                    drop_rate: 0.0,
                    training_flag: False
                })
            
                indices_and_ink_predictions = [(values[0], values[1][1]) for values in enumerate(prediction_values)]
                sort = sorted(indices_and_ink_predictions, key=lambda x: x[1])

                # current_top_n_predictions_and_volumes = [(pair[1], prediction_samples[pair[0]]) for pair in sort[(-1 * args.number):]]
                # current_bottom_n_predictions_and_volumes = [(pair[1], prediction_samples[pair[0]]) for pair in sort[:args.number]]
                current_top_prediction_and_volume = [(pair[1], prediction_samples[pair[0]]) for pair in sort[-1:]]
                current_bottom_prediction_and_volume = [(pair[1], prediction_samples[pair[0]]) for pair in sort[:1]]
                
                # top_n_predictions_and_volumes = sorted(top_n_predictions_and_volumes + current_top_n_predictions_and_volumes, key = lambda x: x[0])[(-1 * args.number):]
                # bottom_n_predictions_and_volumes = sorted(bottom_n_predictions_and_volumes + current_bottom_n_predictions_and_volumes, key = lambda x: x[0])[:args.number]
                top_n_predictions_and_volumes = sorted(top_n_predictions_and_volumes + current_top_prediction_and_volume, key = lambda x: x[0])[(-1 * args.number):]
                bottom_n_predictions_and_volumes = sorted(bottom_n_predictions_and_volumes + current_bottom_prediction_and_volume, key = lambda x: x[0])[:args.number]
                
                prediction_samples, coordinates, next_coordinates = volumes.getPredictionBatch(
                    params, next_coordinates)

        except KeyboardInterrupt:
            pass

        top_volume = np.pad(np.ones((96, 96, 48)) * 65535, 20, 'constant')
        bottom_volume = np.pad(np.zeros((96, 96, 48)), 20, 'constant')
        
        for i in range(len(top_n_predictions_and_volumes)):
            top_volume = np.append(top_volume, np.pad(top_n_predictions_and_volumes[i][1], 20, 'constant'), axis=0)
            # ops.save_subvolume_to_image_stack(top_n_predictions_and_volumes[i][1], os.path.join(params['output_path'], 'top-subvolumes', str(i)))
        # ops.save_subvolume_to_image_stack(top_volume, os.path.join(params['output_path'], 'top-subvolume'))

        for i in range(len(bottom_n_predictions_and_volumes)):
            bottom_volume = np.append(bottom_volume, np.pad(bottom_n_predictions_and_volumes[i][1], 20, 'constant'), axis=0)
            # ops.save_subvolume_to_image_stack(bottom_n_predictions_and_volumes[i][1], os.path.join(params['output_path'], 'bottom-subvolumes', str(i)))
        # ops.save_subvolume_to_image_stack(bottom_volume, os.path.join(params['output_path'], 'bottom-subvolume'))

        ops.save_subvolume_to_image_stack(np.append(top_volume, bottom_volume, axis=1), os.path.join(params['output_path'], 'both-subvolume-sets'))

if __name__ == '__main__':
    main()
