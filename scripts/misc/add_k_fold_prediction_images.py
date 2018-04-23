import argparse
import os
import re

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dir', metavar='path', required=True,
                        help='input directory')
    parser.add_argument('--outfile', metavar='path', required=True,
                        help='output image name')
    parser.add_argument('--best-f1', action='store_true')

    args = parser.parse_args()
    dirs = [os.path.join(args.dir, name) for name in os.listdir(args.dir)
            if os.path.isdir(os.path.join(args.dir, name))]

    print('Using images:')

    image = None
    for d in dirs:
        # Sort by the iteration number and pick the last one
        names = os.listdir(os.path.join(d, 'predictions'))
        if args.best_f1:
            names = list(filter(lambda s: '_best_f1' in s, names))
        image_name = sorted(
            names,
            key=lambda name: int(re.findall('\d+', name)[0])
        )[-1]
        image_name = os.path.join(d, 'predictions', image_name)
        print('\t{}'.format(image_name))
        if image is None:
            image = np.array(Image.open(image_name))
        else:
            image += np.array(Image.open(image_name))
    image = Image.fromarray(image)
    image.save(args.outfile)


if __name__ == '__main__':
    main()
