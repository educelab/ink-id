"""Split a region data source into a grid dataset.

This expects a region data source. The region will be split into multiple regions per the grid arguments. This can then
be used with --cross-validate-on <n> in train_and_predict.py to perform k-fold validation/prediction over the grid.

"""

import argparse
import json
import os

import inkid.data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', metavar='infile', help='input region data source file (.json)')
    parser.add_argument('rows', metavar='rows', type=int, help='grid squares along x axis (number of rows)')
    parser.add_argument('columns', metavar='cols', type=int, help='grid squares along y axis (number of columns)')

    args = parser.parse_args()

    data_source = inkid.data.RegionSource(args.input)
    bounds = data_source.bounding_box

    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    rows = int(args.rows)
    columns = int(args.columns)
    grid_square_width = width // columns
    grid_square_height = height // rows

    data_source_paths = []

    for row in range(rows):
        for col in range(columns):
            grid_square_json = data_source.source_json.copy()
            grid_square_json['bounding_box'] = [
                bounds[0] + col * grid_square_width,
                bounds[1] + row * grid_square_height,
                bounds[0] + (col + 1) * grid_square_width,
                bounds[1] + (row + 1) * grid_square_height,
            ]
            grid_square_path = f'{os.path.splitext(args.input)[0]}_grid{rows}x{columns}_row{row}_col{col}.json'
            data_source_paths.append(grid_square_path)
            with open(grid_square_path, 'w') as f:
                json.dump(grid_square_json, f, indent=4)
            print(f'Wrote region data source file {grid_square_path}')

    dataset_filename = f'{os.path.splitext(args.input)[0]}_grid{rows}x{columns}.txt'
    with open(dataset_filename, 'w') as f:
        for data_source_path in data_source_paths:
            f.write(os.path.basename(data_source_path) + '\n')
    print(f'Wrote grid dataset file {dataset_filename}')


if __name__ == '__main__':
    main()
