import argparse
import datetime
import os
import sys

import numpy as np
import progressbar

import inkid.model
import inkid.ops
import inkid.data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--data', metavar='path', required=True,
                        help='input data file (JSON)')
    parser.add_argument('-o', '--output', metavar='path', default='out',
                        help='output directory')

    args = parser.parse_args()
    output_path = os.path.join(
        args.output,
        datetime.datetime.today().strftime('%Y-%m-%d_%H.%M.%S')
    )

    regions = inkid.data.RegionSet.from_json(args.data)

    params = inkid.ops.load_default_parameters()

    points = regions.get_points_generator(
        'training',
        restrict_to_surface=True,
        perform_shuffle=False,
    )

    print('Calculating summary statistics...')
    all_statistics = []
    all_points = []
    bar = progressbar.ProgressBar()
    for point in bar(points()):
        region_id, x, y = point
        subvolume = regions._regions[region_id].ppm.point_to_subvolume(
            (x, y),
            params['subvolume_shape'],
            augment_subvolume=False
        )
        statistics = inkid.ops.get_descriptive_statistics(subvolume)
        all_statistics.append(statistics)
        all_points.append(point)
    all_statistics = np.array(all_statistics)

    print('Calculating max and min for each statistic...', end='')
    sys.stdout.flush()
    maxs = np.amax(all_statistics, axis=0)
    mins = np.amin(all_statistics, axis=0)
    print('done')

    print('Saving prediction images...', end='')
    sys.stdout.flush()
    for stat_idx in range(len(all_statistics[0])):
        for i in range(len(all_statistics)):
            region_id, x, y = all_points[i]
            statistics = all_statistics[i]
            regions.reconstruct_prediction_values(
                np.array([region_id]),
                np.array(
                    [
                        int(inkid.ops.remap(
                            statistics[stat_idx],
                            mins[stat_idx],
                            maxs[stat_idx],
                            0,
                            np.iinfo(np.uint16).max
                        ))
                    ]
                ),
                np.array([[x, y]])
            )
        regions.save_predictions(
            os.path.join(output_path, 'predictions'),
            'stats_{}'.format(stat_idx)
        )
        regions.reset_predictions()
    print('done')


if __name__ == '__main__':
    main()
