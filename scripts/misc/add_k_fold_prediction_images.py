import argparse
import os
import re

import numpy as np
from PIL import Image
import wand.image

import inkid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # Input directory with job output
    parser.add_argument('dir', metavar='path', help='input directory')
    # Image generation options
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--final', action='store_true')
    parser.add_argument('--img-seq', help='Generate an image sequence and save it to the provided directory')
    parser.add_argument('--gif', action='store_true', help='Generate an image sequence and save it as an animated GIF')
    parser.add_argument('--gif-prefix', default='training', help='The prefix used for the output GIF filename')
    parser.add_argument('--gif-delay', default=10, type=int, help='GIF frame delay in hundredths of a second')
    parser.add_argument('--caption-gif-with-iterations', action='store_true')
    # Rclone upload options
    parser.add_argument('--rclone-transfer-remote', metavar='remote', default=None,
                        help='if specified, and if matches the name of one of the directories in '
                             'the output path, transfer the results to that rclone remote into the '
                             'subpath following the remote name')
    args = parser.parse_args()

    # Get list of directories (not files) in given parent dir
    dirs = [os.path.join(args.dir, name) for name in os.listdir(args.dir)
            if os.path.isdir(os.path.join(args.dir, name))]

    # Print help if no summary images requested
    if not (args.final or args.img_seq or args.gif or args.all):
        parser.print_help()
        # But don't return, in case still doing rclone upload later
        # Could change this if we put all image processes into function

    # Generate "final" image
    if args.final or args.all:
        print('\nFor final predictions, using images:')
        get_and_merge_images(dirs, os.path.join(args.dir, 'final.png'))

    # Generate training animation
    animation = None
    if args.img_seq or args.gif or args.all:
        print('\nCreating animation:')
        animation = create_animation(dirs, args.caption_gif_with_iterations)

    # Write to image sequence
    if args.img_seq:
        write_img_sequence(animation, args.img_seq)

    # Write to gif
    if args.gif or args.all:
        if args.caption_gif_with_iterations:
            filename = args.gif_prefix + '_captioned.gif'
        else:
            filename = args.gif_prefix + '.gif'
        write_gif(animation, os.path.join(args.dir, filename), args.gif_delay)

    # Transfer results via rclone if requested
    inkid.ops.rclone_transfer_to_remote(args.rclone_transfer_remote, args.dir)


def create_animation(dirs, caption):
    filenames_in_each_dir = []
    for d in dirs:
        names = os.listdir(os.path.join(d, 'predictions'))
        names = list(filter(lambda name: re.search('_(\d+)_(\d+)[\._]', name) is not None, names))
        names = sorted(
            names,
            key=lambda name: [int(v) for v in re.findall('_(\d+)_(\d+)[\._]', name)[0]]
        )
        names = [os.path.join(d, 'predictions', name) for name in names]
        filenames_in_each_dir.append(names)

    max_number_of_images = max([len(i) for i in filenames_in_each_dir])
    num_frames = max_number_of_images
    animation = wand.image.Image()
    for i in range(num_frames):
        frame = None
        iterations_getting_shown = []
        for d in filenames_in_each_dir:
            if len(d) == 0:
                continue
            filename = d[i] if i < len(d) else d[-1]
            iterations_getting_shown.append(
                [int(v) for v in re.findall('_(\d+)_(\d+)[\._]', os.path.basename(filename))[0]]
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
                    'modulus_add',
                    left=0,
                    top=0,
                )
        if frame is None:
            return None
        if caption:
            epoch, batch = iterations_getting_shown[-1]
            frame.caption(
                '{:02d} {:08d}'.format(epoch, batch),
                font=wand.font.Font(
                    path='',
                    color=wand.color.Color('#0C0687')
                )
            )
        animation.sequence.append(frame)
    return animation


def write_img_sequence(animation, outdir):
    if animation is None:
        return
    print('\nWriting training image sequence to', outdir)
    prefix = os.path.join(outdir, "sequence_")
    images = animation.sequence
    for i in range(len(images)):
        outfile = prefix + str(i) + ".png"
        wand.image.Image(images[i]).save(filename=outfile)


def write_gif(animation, outfile, delay=10):
    if animation is None:
        return
    print('\nWriting training gif to', outfile)
    gif = animation
    for frame in gif.sequence:
        frame.delay = delay
    gif.type = 'optimize'
    gif.save(filename=outfile)


def get_and_merge_images(dirs, outfile):
    image = None
    for d in dirs:
        # Sort by the iteration number and pick the last one
        names = os.listdir(os.path.join(d, 'predictions'))
        if len(names) == 0:
            continue
        names = list(filter(lambda name: re.search('_(\d+)_(\d+)[\._]', name) is not None, names))
        names = sorted(
            names,
            key=lambda name: [int(v) for v in re.findall('_(\d+)_(\d+)[\._]', name)[0]]
        )
        image_name = os.path.join(d, 'predictions', names[-1])
        print('\t{}'.format(image_name))
        if image is None:
            image = np.array(Image.open(image_name))
        else:
            image += np.array(Image.open(image_name))
    if image is not None:
        image = Image.fromarray(image)
        image.save(outfile)


if __name__ == '__main__':
    main()
