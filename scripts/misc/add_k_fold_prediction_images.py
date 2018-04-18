import argparse
import os

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dir', metavar='path', required=True,
                        help='input directory')
    parser.add_argument('--outfile', metavar='path', required=True,
                        help='output image name')

    args = parser.parse_args()
    dirs = [os.path.join(args.dir, name) for name in os.listdir(args.dir)
            if os.path.isdir(os.path.join(args.dir, name))]

    image = None
    for d in dirs:
        image_name = sorted(os.listdir(os.path.join(d, 'predictions')))[-1]
        image_name = os.path.join(d, 'predictions', image_name)
        if image is None:
            image = np.array(Image.open(image_name))
        else:
            image += np.array(Image.open(image_name))
    image = Image.fromarray(image)
    image.save(args.outfile)


if __name__ == '__main__':
    main()
