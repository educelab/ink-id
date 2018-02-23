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

from inkid.volumes import Volume
import inkid.ops
import inkid.model


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
    params = inkid.ops.load_default_parameters()

    # Adjust some parameters from supplied arguments
    params['volumes'][0]['data_path'] = args.data
    params['volumes'][0]['surface_mask'] = args.surfacemask
    params['volumes'][0]['surface_data'] = args.surfacedata
    params['output_path'] = os.path.join(args.outputdir, '3dcnn-predictions', datetime.datetime.today().strftime('%Y-%m-%d_%H.%M.%S'))

    
    volume = Volume(params, 0)
    
    points = volume.yield_coordinate_pool(10)
    points = volume.filter_on_surface(points)
    subvolumes = volume.coordinates_to_subvolumes(points)
    generator = inkid.ops.generator_from_iterator(subvolumes)

    dataset = tf.data.Dataset.from_generator(generator, (tf.int64, tf.float32),
                                             ([2], [params["subvolume_dimension_x"], params["subvolume_dimension_y"], params["subvolume_dimension_z"]]))
    # dataset = dataset.shuffle(10000)
    dataset = dataset.batch(params["prediction_batch_size"])
    next_batch = dataset.make_one_shot_iterator().get_next()

    y = tf.placeholder(tf.float32, [None, params["n_classes"]])
    drop_rate = tf.placeholder(tf.float32)
    training_flag = tf.placeholder(tf.bool)
    pred, _, inputs, coordinates = inkid.model.build_model(next_batch, y, drop_rate, params, training_flag)
    
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, args.model)

        top_n_samples = []
        bottom_n_samples = []

        try:
            while True:
                try:
                    print("Running a batch.")
                    prediction_values, subvolumes, coords = sess.run((pred, inputs, coordinates), feed_dict={
                        drop_rate: 0.0,
                        training_flag: False
                    })

                    samples = [{"prediction": item[0][1], "subvolume": item[1], "coordinates":item[2]} for item in zip(prediction_values, subvolumes, coords)]
                    
                    sorted_samples = sorted(samples, key=lambda x: x["prediction"])

                    current_top_sample = sorted_samples[-1:][0]
                    current_bottom_sample = sorted_samples[:1][0]

                    too_close = False
                    for sample in top_n_samples:
                        if inkid.ops.are_coordinates_within(sample["coordinates"], current_top_sample["coordinates"], params["subvolume_dimension_x"]):
                            too_close = True
                            break
                    if not too_close:
                        top_n_samples.append(current_top_sample)
                        top_n_samples = sorted(top_n_samples, key = lambda x: x["prediction"])[(-1 * args.number):]

                    too_close = False
                    for sample in bottom_n_samples:
                        if inkid.ops.are_coordinates_within(sample["coordinates"], current_bottom_sample["coordinates"], params["subvolume_dimension_x"]):
                            too_close = True
                            break
                    if not too_close:
                        bottom_n_samples.append(current_bottom_sample)
                        bottom_n_samples = sorted(bottom_n_samples, key = lambda x: x["prediction"])[:args.number]
                
                except tf.errors.OutOfRangeError:
                    break

        except KeyboardInterrupt:
            pass

        top_volume = np.pad(np.ones((96, 96, 48)) * 65535, 20, 'constant')
        bottom_volume = np.pad(np.zeros((96, 96, 48)), 20, 'constant')
        
        for i in range(min(len(top_n_samples), len(bottom_n_samples))):
            top_volume = np.append(top_volume, np.pad(top_n_samples[i]["subvolume"], 20, 'constant'), axis=0)

        for i in range(min(len(top_n_samples), len(bottom_n_samples))):
            bottom_volume = np.append(bottom_volume, np.pad(bottom_n_samples[i]["subvolume"], 20, 'constant'), axis=0)
            
        inkid.ops.save_volume_to_image_stack(np.append(top_volume, bottom_volume, axis=1), os.path.join(params['output_path'], 'both-subvolume-sets'))

if __name__ == '__main__':
    main()
