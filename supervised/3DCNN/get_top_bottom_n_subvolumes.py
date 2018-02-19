"""
Using a trained model and volume data, get the n subvolumes on that surface
that have the highest and lowest prediction score in the model.
"""

import argparse
import os
import time

import tensorflow as tf

from volumeset import VolumeSet


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
    parser.add_argument('--modeldir', metavar='path', required=True,
                        help='path to directory containing trained model')
    parser.add_argument('--number', '-n', metavar='N', default=5, type=int,
                        help='number of subvolumes to keep')
    parser.add_argument('--outputdir', metavar='path', default='out',
                        help='path to output directory')

    args = parser.parse_args()

    with tf.Session() as sess:
        tf.saved_model.loader.load(sess, ['SERVING'], args.modeldir)
        starting_coordinates = [0, 0, 0]
        # prediction_samples, coordinates, next_coordinates = volumes.getPredictionBatch(

if __name__ == '__main__':
    main()
