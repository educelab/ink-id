"""Split a RegionSet with one training region into a grid.

This expects a RegionSet with one training region. The training region
will be split into multiple training regions per the grid
arguments. This can then be used with
k_fold_validation_and_prediction.py to isolate one of the regions at a
time to be the predict/evaluate region while the others are used for
training.

"""

import argparse
import json
import os

from jsmin import jsmin

import inkid.data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', metavar='infile', help='input file (JSON)')
    parser.add_argument('output', metavar='outfile', help='output file')
    parser.add_argument('--rows', metavar='num', default=3,
                        help='grid squares along x axis (number of rows)')
    parser.add_argument('--columns', metavar='num', default=3,
                        help='grid squares along y axis (number of columns)')

    args = parser.parse_args()

    with open(args.input, 'r') as f:
        minified = jsmin(str(f.read()))
        data = json.loads(minified)

    assert len(data['regions']['training']) == 1
    assert len(data['regions']['evaluation']) == 0
    assert len(data['regions']['prediction']) == 0

    bounds = data['regions']['training'][0].get('bounds')

    if bounds is None:
        ppm_name = data['regions']['training'][0]['ppm']
        ppm_path = data['ppms'][ppm_name]['path']
        ppm_path = os.path.normpath(
            os.path.join(
                os.path.dirname(args.input),
                ppm_path
            )
        )

        header = inkid.data.PPM.parse_PPM_header(ppm_path)
        bounds = [0, 0, header['width'], header['height']]

    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    rows = int(args.rows)
    columns = int(args.columns)
    grid_square_width = width // columns
    grid_square_height = height // rows

    training_regions = []

    for x in range(columns):
        for y in range(rows):
            region = {'ppm': ppm_name}
            region['bounds'] = [
                bounds[0] + x * grid_square_width,
                bounds[1] + y * grid_square_height,
                bounds[0] + (x + 1) * grid_square_width,
                bounds[1] + (y + 1) * grid_square_height,
            ]
            training_regions.append(region)

    data['regions']['training'] = training_regions

    with open(args.output, 'w') as outfile:
        json.dump(data, outfile, indent=4)


if __name__ == '__main__':
    main()
