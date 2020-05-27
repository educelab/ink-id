import argparse
import struct

import numpy as np
from PIL import Image
import progressbar


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dummy_image')
    parser.add_argument('output_ppm')
    parser.add_argument('-z', default=0)
    args = parser.parse_args()

    img = Image.open(args.dummy_image).convert('L')
    version = 1
    dim = 6
    w, h = img.size

    ppm_data = np.zeros((h, w, dim))
    for y in range(h):
        for x in range(w):
            ppm_data[y, x] = [x, y, args.z, 0, 0, 1]

    with open(args.output_ppm, 'wb') as f:
        print('Writing PPM to file {}...'.format(args.output_ppm))
        f.write('width: {}\n'.format(w).encode('utf-8'))
        f.write('height: {}\n'.format(h).encode('utf-8'))
        f.write('dim: {}\n'.format(dim).encode('utf-8'))
        f.write('ordered: {}\n'.format('true').encode('utf-8'))
        f.write('type: double\n'.encode('utf-8'))
        f.write('version: {}\n'.format(version).encode('utf-8'))
        f.write('<>\n'.encode('utf-8'))
        bar = progressbar.ProgressBar()
        for y in bar(range(h)):
            for x in range(w):
                for idx in range(dim):
                    f.write(struct.pack('d', ppm_data[y, x, idx]))


if __name__ == '__main__':
    main()
