import argparse
import os
import re

import numpy as np
from PIL import Image
import wand.image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dir', metavar='path', help='input directory')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--final', action='store_true')
    parser.add_argument('--best-auc', action='store_true')
    parser.add_argument('--seq-dir', type=String, "Generate an image sequence and save it to the provided directory")
    parser.add_argument('--gif', action='store_true', "Generate an image sequence and save it as an animated GIF")
    parser.add_argument('--gif-delay', type=int, "GIF frame delay in hundredths of a second", default=10)
    parser.add_argument('--no-caption-gif-with-iterations', action='store_false')
    parser.add_argument('--composite-at-iteration', type=int, default=None)

    args = parser.parse_args()
    dirs = [os.path.join(args.dir, name) for name in os.listdir(args.dir)
            if os.path.isdir(os.path.join(args.dir, name))]

    print('Note: This script relies on the auto-generated iteration number'
          ' in the filename being the only number preceded by "_"'
          ' and followed by "_" or ".".')

    if not (args.final or args.best_auc or args.gif or args.all or args.composite_at_iteration):
        parser.print_help()
    if args.final or args.all:
        print('\nFor final predictions, using images:')
        get_and_merge_images(dirs, False, os.path.join(args.dir, 'final.tif'))
    if args.best_auc or args.all:
        print('\nFor best auc predictions, using images:')
        get_and_merge_images(dirs, True, os.path.join(args.dir, 'best_auc.tif'))
    if args.composite_at_iteration is not None:
        print('\nFor composite at iteration {}, using images:'.format(args.composite_at_iteration))
        get_and_merge_images(
            dirs,
            False,
            os.path.join(args.dir, 'iteration_{}.tif'.format(args.composite_at_iteration)),
            iteration=args.composite_at_iteration
        )

    if args.seq_dir or args.gif or args.all:
        sequence = create_sequence(dirs, args.no_caption_gif_with_iterations)

    if args.seq_dir:
        write_img_sequence(sequence, args.seq_dir)

    if args.gif or args.all:
        print('\nFor training .gif, using images:')
        if args.no_caption_gif_with_iterations:
            filename = 'training_captioned.gif'
        else:
            filename = 'training.gif'
        write_gif(sequence, os.path.join(args.dir, filename), args.gif_delay)

def create_sequence(dirs, caption):
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
    frames = []
    for i in range(num_frames):
        frame = None
        iterations_getting_shown = []
        for d in filenames_in_each_dir:
            filename = d[i] if i < len(d) else d[-1]
            iterations_getting_shown.append(
                int(re.findall('_(\d+)[\._]', os.path.basename(filename))[0])
            )
            print('\t{}'.format(filename))
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
        if caption:
            iteration = max(iterations_getting_shown)
            frame.caption(
                '{:08d}'.format(iteration),
                font=wand.font.Font(
                    path='',
                    color=wand.color.Color('#0C0687')
                )
            )
        frames.append(frame)
    return frames

def write_img_sequence(sequence, outdir):
    return False

def write_gif(sequence, outfile, delay = 10):
    gif = wand.image.Image()
    gif.sequence = sequence
    for frame in gif.sequence:
        frame.delay = delay
    gif.type = 'optimize'
    gif.save(filename=outfile)

def get_and_merge_images(dirs, best_auc, outfile, iteration=None):
    image = None
    for d in dirs:
        # Sort by the iteration number and pick the last one
        names = os.listdir(os.path.join(d, 'predictions'))
        names = list(filter(lambda name: re.search('_(\d+)[\._]', name) is not None, names))
        if iteration is not None:
            names = list(filter(
                lambda name: re.findall('_(\d+)[\._]', name)[0] in [
                    str(iteration),
                    str(iteration+1)
                ],
                names
            ))
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
