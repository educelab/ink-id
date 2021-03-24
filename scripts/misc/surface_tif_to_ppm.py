"""Convert a surface .tif to a .ppm.

This was written for the lunate sigma surface .tif and so makes
assumptions about axes that should be changed if used for other data.

"""
import argparse
import struct

import numpy as np
from PIL import Image
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tif', metavar='path', required=True,
                        help='path to input surface .tif')
    parser.add_argument('--ppm', metavar='path', required=True,
                        help='path to output .ppm')
    parser.add_argument('--stepsize', metavar='n', required=False, type=int, default=1,
                        help='step along input this many pixels at a time (will downscale output)')

    args = parser.parse_args()

    tif = np.array(Image.open(args.tif))

    # Write the header
    with open(args.ppm, 'w') as f:
        f.write('width: {}\n'.format(tif.shape[1] // args.stepsize))  # TODO hacky
        f.write('height: {}\n'.format(tif.shape[0] // args.stepsize))
        f.write('dim: {}\n'.format(6))
        f.write('ordered: {}\n'.format('true'))
        f.write('type: {}\n'.format('double'))
        f.write('version: {}\n'.format(1))
        f.write('{}\n'.format('<>'))

    # Write the data
    with open(args.ppm, 'ab') as f:
        for tif_y in tqdm(range(0, len(tif), args.stepsize)):
            for tif_x in range(0, len(tif[tif_y]), args.stepsize):
                tif_z = tif[tif_y][tif_x]
                ppm_coordinate = [
                    tif_z // args.stepsize,
                    tif_x // args.stepsize,
                    tif_y // args.stepsize,
                ]
                ppm_normal = [-1, 0, 0]
                ppm = ppm_coordinate + ppm_normal
                ppm = [float(i) for i in ppm]
                s = struct.pack('d'*len(ppm), *ppm)
                f.write(s)


if __name__ == '__main__':
    main()
