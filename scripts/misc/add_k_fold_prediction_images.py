import argparse
import os
import re

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dir', metavar='path', help='input directory')

    args = parser.parse_args()
    dirs = [os.path.join(args.dir, name) for name in os.listdir(args.dir)
            if os.path.isdir(os.path.join(args.dir, name))]

    print('Note: This script relies on the auto-generated iteration number'
          ' in the filename being the only number preceded by "_"'
          ' and followed by "_" or ".".')

    print('\nFor final predictions, using images:')
    get_and_merge_images(dirs, False, os.path.join(args.dir, 'final.tif'))
    print('\nFor best f1 predictions, using images:')
    get_and_merge_images(dirs, True, os.path.join(args.dir, 'best_auc.tif'))


def get_and_merge_images(dirs, best_auc, outfile):
    image = None
    for d in dirs:
        # Sort by the iteration number and pick the last one
        names = os.listdir(os.path.join(d, 'predictions'))
        names = list(filter(lambda name: re.search('_(\d+)[\._]', name) is not None, names))
        names = sorted(
            names,
            key=lambda name: int(re.findall('_(\d+)[\._]', name)[0])
        )
        # TODO add option to create gif here
        if best_auc:
            names = list(filter(lambda s: '_best_auc' in s, names))
        image_name = os.path.join(d, 'predictions', names[-1])
        print('\t{}'.format(image_name))
        if image is None:
            image = np.array(Image.open(image_name))
        else:
            image += np.array(Image.open(image_name))
    image = Image.fromarray(image)
    image.save(outfile)


if __name__ == '__main__':
    main()
