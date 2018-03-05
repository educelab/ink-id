"""
Train an ink classifier and produce predicted output for a volume.
"""

import argparse
import datetime
import os
import time

from sklearn.metrics import precision_score, fbeta_score
import tensorflow as tf
import numpy as np

from inkid.volumes import VolumeSet
import inkid.model
import inkid.ops


def main():
    """Run the training and prediction process."""
    start_time = time.time()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', metavar='path', required=True,
                        help='path to volume data (slices directory)')
    parser.add_argument('--groundtruth', metavar='path', required=True,
                        help='path to ground truth image')
    parser.add_argument('--surfacemask', metavar='path', required=True,
                        help='path to surface mask image')
    parser.add_argument('--surfacedata', metavar='path', required=True,
                        help='path to surface data')
    parser.add_argument('--gridtestsquare', metavar='num', default=0, type=int,
                        help='index of grid test square for this k-fold run')
    parser.add_argument('--outputdir', metavar='path', default='out',
                        help='path to output directory')

    args = parser.parse_args()

    # Load default parameters
    params = inkid.ops.load_default_parameters()

    # Adjust some parameters from supplied arguments
    params['volumes'][0]['data_path'] = args.data
    params['volumes'][0]['ground_truth'] = args.groundtruth
    params['volumes'][0]['surface_mask'] = args.surfacemask
    params['volumes'][0]['surface_data'] = args.surfacedata
    params['grid_test_square'] = args.gridtestsquare
    params['output_path'] = os.path.join(args.outputdir, '3dcnn-predictions/{}-{}-{}h'.format(
            datetime.datetime.today().timetuple()[1],
            datetime.datetime.today().timetuple()[2],
            datetime.datetime.today().timetuple()[3]))

    volumes = VolumeSet(params)

    next_batch, next_labels = volumes.tf_input_fn()

    with tf.Session() as sess:
        first_batch = sess.run((next_batch, next_labels))
    print(first_batch)
    

if __name__ == '__main__':
    main()
