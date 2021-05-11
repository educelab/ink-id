import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', metavar='path', help='input region set file to be converted')
    args = parser.parse_args()

    dataset_name = input('Enter a dataset name: ')

    is_grid = False
    if input('Is this region set a grid? [y/N]: ').lower().strip() == 'y':
        is_grid = True

    # Write region data source files
    region_set_path = args.input
    region_set_dir = os.path.dirname(region_set_path)
    with open(region_set_path) as f:
        regions = json.loads(f.read())
        region_files_written = list()
        for i, train_region in enumerate(regions['regions']['training']):
            new_region = dict()
            ppm = regions['ppms'][train_region['ppm']]
            new_region['type'] = 'region'
            new_region['version'] = 0.1
            new_region['ppm'] = ppm.get('path')
            new_region['mask'] = ppm.get('mask')
            new_region['bounding-box'] = train_region['bounds']
            new_region['invert-normal'] = bool(ppm.get('invert-normal'))
            new_region['volume'] = ppm.get('volume')
            new_region['ink-label'] = ppm.get('ink-label')
            new_region['rgb-label'] = ppm.get('rgb-label')
            new_region['volcart-texture-label'] = ppm.get('volcart-texture-label')
            if is_grid:
                if i == 0:
                    rows = int(input('Enter number of rows: '))
                    num_training_regions = len(regions['regions']['training'])
                    if num_training_regions % rows != 0:
                        raise ValueError(f'Number of rows provided ({rows}) does not cleanly divide '
                                         f'number of training regions ({num_training_regions})')
                    cols = num_training_regions // rows
                row = i // cols
                col = i % cols
                region_filename = f'{dataset_name}_Row{row}_Col{col}.json'
            else:
                region_filename = input(f'Enter a name for region {i}: ') + '.json'
            new_region_file_path = os.path.join(region_set_dir, region_filename)
            with open(new_region_file_path, 'w') as f2:
                f2.write(json.dumps(new_region, indent=4, sort_keys=False))
                print(f'Wrote {new_region_file_path}')
            region_files_written.append(region_filename)

    # Write dataset file
    dataset_file_name = dataset_name + '.txt'
    dataset_file_path = os.path.join(region_set_dir, dataset_file_name)
    with open(dataset_file_path, 'w') as f:
        for region_file_written in region_files_written:
            f.write(region_file_written + '\n')
    print(f'Wrote dataset file to {dataset_file_path}')


if __name__ == '__main__':
    main()
