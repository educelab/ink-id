import argparse
import os
import re

import numpy as np
from PIL import Image
import wand.image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dir', metavar='path', help='input directory')
    parser.add_argument('--final', action='store_true')
    parser.add_argument('--best-auc', action='store_true')
    parser.add_argument('--gif', action='store_true')

    args = parser.parse_args()
    dirs = [os.path.join(args.dir, name) for name in os.listdir(args.dir)
            if os.path.isdir(os.path.join(args.dir, name))]

    print('Note: This script relies on the auto-generated iteration number'
          ' in the filename being the only number preceded by "_"'
          ' and followed by "_" or ".".')

    if args.final:
        print('\nFor final predictions, using images:')
        get_and_merge_images(dirs, False, os.path.join(args.dir, 'final.tif'))
    if args.best_auc:
        print('\nFor best auc predictions, using images:')
        get_and_merge_images(dirs, True, os.path.join(args.dir, 'best_auc.tif'))
    if args.gif:
        print('\nFor training .gif, using images:')
        create_gif(dirs, os.path.join(args.dir, 'training.gif'))


def create_gif(dirs, outfile):
    filenames_in_each_dir = []
    for d in dirs:
        names = os.listdir(os.path.join(d, 'predictions'))
        names = list(filter(lambda name: re.search('_(\d+)[\._]', name) is not None, names))
        names = sorted(
            names,
            key=lambda name: int(re.findall('_(\d+)[\._]', name)[0])
        )
        names = [os.path.join(d, 'predictions', name) for name in names]
        filenames_in_each_dir.append(names)
    max_number_of_images = max([len(i) for i in filenames_in_each_dir])

    num_frames = max_number_of_images

    gif = wand.image.Image()
    for i in range(num_frames):
        frame = None
        for d in filenames_in_each_dir:
            filename = d[i] if i < len(d) else d[-1]
            print(filename)
            try:
                partial_frame = wand.image.Image(filename=filename)
                partial_frame.transform(resize='20%')
            except wand.exceptions.CoderError:
                partial_frame = wand.image.Image(width=1, height=1)
            if frame is None:
                frame = partial_frame
            else:
                frame.composite_channel(
                    'all_channels',
                    partial_frame,
                    'add',
                    left=0,
                    top=0,
                )
        gif.sequence.append(frame)

    for i in range(num_frames):
        with gif.sequence[i] as frame:
            frame.delay = 10
    gif.type = 'optimize'
    gif.save(filename=outfile)


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
