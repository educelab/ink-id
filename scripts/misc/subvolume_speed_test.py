"""
Test performance of various subvolume methods.
"""

import argparse
import functools
import os
import time

import tensorflow as tf

import inkid


def println(method, times):
    print('{0: <24}'.format(method) + '\t'.join(times), end='\r')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', metavar='path', help='input data file (JSON)')
    parser.add_argument('--batch-size', metavar='num', default=32, type=int,
                        help='number of subvolumes per TensorFlow batch')
    parser.add_argument('--batches-per-method', metavar='num', default=1000, type=int,
                        help='number of batches to fetch per method')
    parser.add_argument('--subvolume-shape', metavar='n', nargs=3, type=int,
                        help='subvolume shape in z y x')

    num_threads_to_try = [1, 2, 4, 8, 16, 32, 64, 128]

    args = parser.parse_args()
    print(args)
    print()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    regions = inkid.data.RegionSet.from_json(args.data)

    methods = [
        'snap_to_axis_aligned',
        'nearest_neighbor',
        'interpolated'
    ]

    # Get list of points (to be used multiple times)
    points = [p for p in regions.get_points_generator(
        ['training', 'prediction', 'evaluation'],
        restrict_to_surface=True,
        perform_shuffle=True,
        shuffle_seed=37,
    )()]

    speed_test_fn = functools.partial(
        run_speed_test,
        regions=regions,
        batch_size=args.batch_size,
        subvolume_shape=args.subvolume_shape,
        points=points,
        batches_per_method=args.batches_per_method
    )

    print()
    print('\t\t\tOOB\t(s)\t' + '\t'.join([str(i) for i in num_threads_to_try]))
    for method in methods:
        times = []
        # See how many subvolumes using each method are out of bounds
        # Do this separately in case checking for zeros takes time
        start = time.time()
        total_zero = speed_test_fn(
            count_zero=True,
            method=method,
            threads=1,
        )
        end = time.time()
        times.append(str(total_zero))
        times.append('{0:.2f}'.format(end - start))
        println(method, times)

        for num_threads in num_threads_to_try:
            start = time.time()
            speed_test_fn(
                count_zero=False,
                method=method,
                threads=num_threads,
            )
            end = time.time()
            times.append('{0:.2f}'.format(end - start))
            println(method, times)

        print()


def run_speed_test(regions=None, count_zero=None, batch_size=None, method=None,
                   subvolume_shape=None, points=None, batches_per_method=None,
                   threads=None):
    point_to_subvolume_input = functools.partial(
        regions.point_to_subvolume_input,
        subvolume_shape=subvolume_shape,
        out_of_bounds='all_zeros',
        move_along_normal=0,
        augment_subvolume=True,
        jitter_max=0,
    )

    if count_zero:
        total_zero = 0

    input_fn = regions.create_tf_input_fn(
        ['training', 'prediction', 'evaluation'],
        batch_size,
        functools.partial(
            point_to_subvolume_input,
            method=method
        ),
        shuffle_seed=37,
        restrict_to_surface=True,
        epochs=-1,
        threads=threads,
        premade_points_generator=lambda: (p for p in points)
    )

    batch_features, _ = input_fn()

    with tf.Session() as sess:
        for i in range(batches_per_method):
            subvolumes = sess.run(batch_features)['Input']
            
            # for (idx, subvolume) in enumerate(subvolumes):
            #     inkid.ops.save_volume_to_image_stack(subvolume, method + '_' + str(idx))
                
            if count_zero:
                for subvolume in subvolumes:
                    if not subvolume.any():
                        total_zero += 1

    if count_zero:
        return total_zero


if __name__ == '__main__':
    main()
