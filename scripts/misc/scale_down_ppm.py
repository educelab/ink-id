import argparse
import re
import struct

import numpy as np
import progressbar

import inkid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('--size-scale-factor', type=int, default=2)
    parser.add_argument('--coordinate-scale-factor', type=int, default=1)
    args = parser.parse_args()

    header = parse_PPM_header(args.input)
    ppm_data = process_PPM_file(args.input, header)
    size_scale_factor = args.size_scale_factor
    coordinate_scale_factor = args.coordinate_scale_factor
    height, width = header['height'], header['width']
    new_height, new_width = height // size_scale_factor, width // size_scale_factor
    resized_data = np.empty((new_height, new_width, 6))

    print('Creating new PPM...')
    bar = progressbar.ProgressBar()
    for y in bar(range(new_height)):
        for x in range(new_width):
            resized_data[y, x] = ppm_data[y * size_scale_factor, x * size_scale_factor]
            resized_data[y, x, 0] /= coordinate_scale_factor
            resized_data[y, x, 1] /= coordinate_scale_factor
            resized_data[y, x, 2] /= coordinate_scale_factor

    print('Writing new PPM...')
    # Write the header
    with open(args.output, 'w') as f:
        f.write('width: {}\n'.format(new_width))
        f.write('height: {}\n'.format(new_height))
        f.write('dim: {}\n'.format(6))
        f.write('ordered: {}\n'.format('true'))
        f.write('type: {}\n'.format('double'))
        f.write('version: {}\n'.format(1))
        f.write('{}\n'.format('<>'))

    # Write the data
    bar = progressbar.ProgressBar()
    with open(args.output, 'ab') as f:
        for y in bar(range(new_height)):
            for x in range(new_width):
                pm = [float(i) for i in resized_data[y, x]]
                s = struct.pack('d'*6, *pm)
                f.write(s)



def parse_PPM_header(filename):
    comments_re = re.compile('^#')
    width_re = re.compile('^width')
    height_re = re.compile('^height')
    dim_re = re.compile('^dim')
    ordering_re = re.compile('^ordered')
    type_re = re.compile('^type')
    version_re = re.compile('^version')
    header_terminator_re = re.compile('^<>$')

    with open(filename, 'rb') as f:
        while True:
            line = f.readline().decode('utf-8')
            if comments_re.match(line):
                pass
            elif width_re.match(line):
                width = int(line.split(': ')[1])
            elif height_re.match(line):
                height = int(line.split(': ')[1])
            elif dim_re.match(line):
                dim = int(line.split(': ')[1])
            elif ordering_re.match(line):
                ordering = line.split(': ')[1].strip() == 'true'
            elif type_re.match(line):
                val_type = line.split(': ')[1].strip()
                assert val_type in ['double']
            elif version_re.match(line):
                version = line.split(': ')[1].strip()
            elif header_terminator_re.match(line):
                break
            else:
                print('Warning: PPM header contains unknown line: {}'.format(line.strip()))

    return {
        'width': width,
        'height': height,
        'dim': dim,
        'ordering': ordering,
        'type': val_type,
        'version': version
    }

def process_PPM_file(filename, header):
    width = header['width']
    height = header['height']
    dim = header['dim']
    ordering = header['ordering']
    type = header['type']
    version = header['version']

    print(
        'Processing PPM data for {} with width {}, height {}, dim {}... '.format(
            filename, width, height, dim
        )
    )

    data = np.empty((height, width, dim))

    with open(filename, 'rb') as f:
        header_terminator_re = re.compile('^<>$')
        while True:
            line = f.readline().decode('utf-8')
            if header_terminator_re.match(line):
                break

        bar = progressbar.ProgressBar()
        for y in bar(range(height)):
            for x in range(width):
                for idx in range(dim):
                    # This only works if we assume dimension 6
                    # PPMs with (x, y, z, n_x, n_y, n_z)
                    if dim == 6:
                        data[y, x, idx] = struct.unpack('d', f.read(8))[0]
    print()
    return data


if __name__ == '__main__':
    main()
