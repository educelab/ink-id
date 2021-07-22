import argparse
import json
import os

import inkid


def update_old_region_set_file_format(filepath, json_data):
    # Write region data source files
    region_set_dir = os.path.dirname(filepath)
    original_filename = os.path.splitext(os.path.basename(filepath))[0]
    data_source_files_written = list()
    dataset_files_written = list()
    for region_set in json_data['regions'].keys():
        region_files_written = list()
        for region in json_data['regions'][region_set]:
            new_region = dict()
            ppm = json_data['ppms'][region['ppm']]
            new_region['schema_version'] = '0.1'
            new_region['type'] = 'region'
            new_region['volume'] = ppm.get('volume')
            new_region['ppm'] = ppm.get('path')
            new_region['mask'] = ppm.get('mask')
            new_region['invert_normals'] = bool(ppm.get('invert-normal'))
            new_region['bounding_box'] = region.get('bounds')
            new_region['ink_label'] = ppm.get('ink-label')
            new_region['rgb_label'] = ppm.get('rgb-label')
            new_region['volcart_texture_label'] = ppm.get('volcart-texture-label')
            print('\nFound region:')
            print(json.dumps(new_region, indent=4, sort_keys=False))
            region_filename = input(f'Enter a name for region shown above: ') + '.json'
            new_region_file_path = os.path.join(region_set_dir, region_filename)
            with open(new_region_file_path, 'w') as f:
                f.write(json.dumps(new_region, indent=4, sort_keys=False))
                data_source_files_written.append(new_region_file_path)
            region_files_written.append(region_filename)

        # Write dataset file
        if len(region_files_written) > 0:
            dataset_file_name = f'{original_filename}_{region_set}.txt'
            dataset_file_path = os.path.join(region_set_dir, dataset_file_name)
            with open(dataset_file_path, 'w') as f:
                for region_file_written in region_files_written:
                    f.write(region_file_written + '\n')
            dataset_files_written.append(dataset_file_path)

    if len(data_source_files_written) > 0:
        print('\nWrote the following data source files:')
        for filename in data_source_files_written:
            print(f'\t{filename}')
    if len(dataset_files_written) > 0:
        print('\nWrote the following dataset files:')
        for filename in dataset_files_written:
            print(f'\t{filename}')
    if len(data_source_files_written) > 0 or len(dataset_files_written) > 0:
        print('\nYou may wish to rename or relocate the written files.')
    else:
        print('\nNo regions found, no files written.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', metavar='path', help='input data file to be converted')
    args = parser.parse_args()

    file_extension = os.path.splitext(args.input)[1]

    if file_extension == '.json':
        raw_data = inkid.ops.get_raw_data_from_file_or_url(args.input)
        json_data = json.loads(raw_data.read())
        if 'ppms' in json_data and 'regions' in json_data:
            update_old_region_set_file_format(args.input, json_data)
    else:
        raise RuntimeError(f'Not sure how to convert {file_extension} files')


if __name__ == '__main__':
    main()
