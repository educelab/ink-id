import argparse
import json
import os
from pathlib import Path
import re

from humanize import naturalsize
import imageio
import numpy as np
from PIL import Image
import pygifsicle
from tensorboard.backend.event_processing.event_multiplexer import EventMultiplexer
from tqdm import tqdm

import inkid


# TODO handle epochs in filenames
# In general, there is a lot of tedious string processing in this file. Parsing the
# filenames to get the PPM, iteration, and epoch is annoying and fragile. It would
# be better to use EXIF data or similar, perhaps integrated with Smeagol.


# Return which k-fold job this directory corresponds to.
# For the purpose of sorting a list of dirs based on job #.
def k_from_dir(dirname):
    # Only interested in the basename, other parts of the path do not matter
    dirname = os.path.basename(dirname)
    # Only has a valid k-fold # if it matches this particular format
    if re.match(r'.*\d\d\d\d-\d\d-\d\d_\d\d\.\d\d\.\d\d_(\d+)', dirname):
        # Return the last number in the dirname, which is the k-fold #
        return int(re.findall(r'(\d+)', dirname)[-1])
    else:
        # Otherwise probably was standalone job (not part of k-fold)
        return -1


# Return a set of all iterations encountered in a k-fold directory
# Optionally restrict to those of only a particular PPM
def iterations_in_k_fold_dir(k_fold_dir, ppm_name=None):
    iterations = set()
    # Look in predictions directory to get all iterations from prediction images
    pred_dir = os.path.join(k_fold_dir, 'predictions')
    if os.path.isdir(pred_dir):
        # Get all filenames in that directory
        names = os.listdir(pred_dir)
        # Filter out those that do not match the desired name format
        if ppm_name is not None:
            names = list(filter(lambda x: re.match(f'{ppm_name}_prediction_.*', x), names))
        else:
            names = list(filter(lambda x: re.match('.*_prediction_', x), names))
        for name in names:
            if re.match(r'.*_prediction_\d+_(\d+)', name):
                iterations.add(re.search(r'.*_prediction_\d+_(\d+)', name).group(1))
            elif re.match('.*_prediction_final', name):
                iterations.add('final')
    return sort_iterations(iterations)


# Return a prediction image of the specified iteration, k-fold directory,
# and PPM. If such an image does not exist return None
def get_prediction_image(iteration, k_fold_dir, ppm_name, return_latest_if_not_found=True):
    filename = f'{ppm_name}_prediction_{iteration}.png'
    full_filename = os.path.join(k_fold_dir, 'predictions', filename)
    if os.path.isfile(full_filename):
        return Image.open(full_filename)
    elif return_latest_if_not_found:
        ret_iteration = None
        # We did not find original file. Check to see if the requested iteration
        # is greater than any we have. If so, return the greatest one we have.
        iterations = iterations_in_k_fold_dir(k_fold_dir, ppm_name)
        if len(iterations) == 0:
            return None
        if iteration == 'final':
            ret_iteration = iterations[-1]
        else:
            # Remove 'final' if present
            numeric_iterations = [i for i in iterations if i != 'final']
            # Get int(yyy) from 'xxx_yyy'
            numeric_iterations = [int(i.split('_')[1]) for i in numeric_iterations]
            # Do the same with requested iteration
            int_iteration = int(iteration.split('_')[1])
            # If requested is beyond any we have, return the last one we have
            if len(numeric_iterations) > 0 and int_iteration > max(numeric_iterations):
                ret_iteration = iterations[-1]
        if ret_iteration is not None:
            filename = f'{ppm_name}_prediction_{ret_iteration}.png'
            full_filename = os.path.join(k_fold_dir, 'predictions', filename)
            if os.path.isfile(full_filename):
                return Image.open(full_filename)
    return None


def build_frame(iteration, k_fold_dirs, ppms, label_type, max_size=None):
    col_width = max([ppm['size'][0] for ppm in ppms.values()])
    row_heights = [ppm['size'][1] for ppm in ppms.values()]
    width = col_width * (len(k_fold_dirs) + 1)
    height = sum(row_heights)
    frame = Image.new('RGB', (width, height))
    # One column at a time
    for k, k_fold_dir in enumerate(k_fold_dirs):
        # Make each row of this column
        for ppm_i, ppm in enumerate(ppms.values()):
            ppm_path = os.path.splitext(os.path.basename(ppm['path']))[0]
            img = get_prediction_image(iteration, k_fold_dir, ppm_path)
            if img is not None:
                offset = (k * col_width, sum(row_heights[:ppm_i]))
                frame.paste(img, offset)
    for ppm_i, ppm in enumerate(ppms.values()):
        if label_type == 'rgb_values':
            label_key = 'rgb-label'
        elif label_type == 'ink_labels':
            label_key = 'ink-label'
        else:
            break
        label_img_path = ppm[label_key]
        # Try getting label image file from recorded location (may not exist on this machine)
        if os.path.isfile(label_img_path):
            img = Image.open(label_img_path)
            offset = (len(k_fold_dirs) * col_width, sum(row_heights[:ppm_i]))
            frame.paste(img, offset)
        # If not there, maybe it is on the local machine under ~/data. This check is not very robust.
        elif '/pscratch/seales_uksr/' in label_img_path:
            label_img_path = label_img_path.replace('/pscratch/seales_uksr/', '')
            label_img_path = os.path.join(Path.home(), 'data', label_img_path)
            if os.path.isfile(label_img_path):
                img = Image.open(label_img_path)
                offset = (len(k_fold_dirs) * col_width, sum(row_heights[:ppm_i]))
                frame.paste(img, offset)

    # Downsize image while keeping aspect ratio
    if max_size is not None:
        frame.thumbnail(max_size, Image.BICUBIC)

    return frame


def create_animation(k_fold_dirs, iterations, ppms, label_type, max_size):
    animation = []
    for iteration in tqdm(iterations):
        frame = build_frame(iteration, k_fold_dirs, ppms, label_type, max_size)
        animation.append(frame)
    return animation

    # filenames_in_each_dir = []
    # for d in k_fold_dirs:
    #     names = os.listdir(os.path.join(d, 'predictions'))
    #     names = list(filter(lambda name: re.search('_(\d+)_(\d+)[._]', name) is not None, names))
    #     names = sorted(
    #         names,
    #         key=lambda name: [int(v) for v in re.findall('_(\d+)_(\d+)[._]', name)[0]]
    #     )
    #     names = [os.path.join(d, 'predictions', name) for name in names]
    #     filenames_in_each_dir.append(names)
    #
    # max_number_of_images = max([len(i) for i in filenames_in_each_dir])
    # if max_number_of_images == 0:
    #     return None
    # num_frames = max_number_of_images
    # animation = wand.image.Image()
    # for i in range(num_frames):
    #     frame = None
    #     iterations_getting_shown = []
    #     for d in filenames_in_each_dir:
    #         if len(d) == 0:
    #             return None
    #         filename = d[i] if i < len(d) else d[-1]
    #         iterations_getting_shown.append(
    #             [int(v) for v in re.findall('_(\d+)_(\d+)[._]', os.path.basename(filename))[0]]
    #         )
    #         print('\t{}'.format(filename))
    #         try:
    #             partial_frame = wand.image.Image(filename=filename)
    #             partial_frame.transform(resize='20%')
    #         except wand.exceptions.CoderError:
    #             partial_frame = wand.image.Image(width=1, height=1)
    #         if frame is None:
    #             frame = partial_frame
    #         else:
    #             frame.composite_channel(
    #                 'all_channels',
    #                 partial_frame,
    #                 'modulus_add',
    #                 left=0,
    #                 top=0,
    #             )
    #     if caption:
    #         epoch, batch = iterations_getting_shown[-1]
    #         frame.caption(
    #             '{:02d} {:08d}'.format(epoch, batch),
    #             font=wand.font.Font(
    #                 path='',
    #                 color=wand.color.Color('#0C0687')
    #             )
    #         )
    #     animation.sequence.append(frame)
    # return animation


def write_img_sequence(animation, outdir):
    if len(animation) == 0:
        return
    print('\nWriting training image sequence to', outdir)
    prefix = os.path.join(outdir, "sequence_")
    for i, img in enumerate(animation):
        outfile = prefix + str(i) + ".png"
        img.save(outfile)


def write_gif(animation, outfile, fps=10):
    if len(animation) == 0:
        return
    print('\nWriting training gif to', outfile)
    durations = [1 / fps] * len(animation)
    # Make the last frame hold for longer.
    durations[-1] = 5
    with imageio.get_writer(outfile, mode='I', duration=durations) as writer:
        for img in tqdm(animation):
            writer.append_data(np.array(img))

    # Optimize .gif file size
    prev_size = os.path.getsize(outfile)
    print('\nOptimizing .gif file', outfile)
    pygifsicle.optimize(outfile)
    new_size = os.path.getsize(outfile)
    reduction = (prev_size - new_size) / prev_size * 100
    print(f'Size reduced {reduction:.2f}% from {naturalsize(prev_size)} to {naturalsize(new_size)}')

#
# def get_and_merge_images(k_fold_dirs, outfile):
#     image = None
#     for d in k_fold_dirs:
#         # Sort by the iteration number and pick the last one
#         names = os.listdir(os.path.join(d, 'predictions'))
#         if len(names) == 0:
#             continue
#         names = list(filter(lambda name: re.search('_(\d+)_(\d+)[._]', name) is not None, names))
#         names = sorted(
#             names,
#             key=lambda name: [int(v) for v in re.findall('_(\d+)_(\d+)[._]', name)[0]]
#         )
#         image_name = os.path.join(d, 'predictions', names[-1])
#         print('\t{}'.format(image_name))
#         if image is None:
#             image = np.array(Image.open(image_name))
#         else:
#             image += np.array(Image.open(image_name))
#     if image is not None:
#         image = Image.fromarray(image)
#         image.save(outfile)


def sort_iterations(iterations):
    has_final = 'final' in iterations
    if has_final:
        iterations.remove('final')
    iterations = sorted(list(map(int, iterations)))
    iterations = ['0_' + str(i) for i in iterations]
    if has_final:
        iterations.append('final')
    return iterations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # Input directory with job output
    parser.add_argument('dir', metavar='path', help='input directory')
    # Image generation options
    parser.add_argument('--img-seq', default=None,
                        help='Generate an image sequence and save it to the provided directory')
    parser.add_argument('--gif-delay', default=10, type=int, help='GIF frame delay in hundredths of a second')
    parser.add_argument('--caption-gif-with-iterations', action='store_true')
    parser.add_argument('--max-size', type=int, nargs=2, default=[1920, 1080])
    # Rclone upload options
    parser.add_argument('--rclone-transfer-remote', metavar='remote', default=None,
                        help = 'if specified, and if matches the name of one of the directories in '
                        'the output path, transfer the results to that rclone remote into the '
                        'sub-path following the remote name')
    args = parser.parse_args()

    # Get list of directories (not files) in given parent dir
    k_fold_dirs = [os.path.join(args.dir, name) for name in os.listdir(args.dir)
            if os.path.isdir(os.path.join(args.dir, name))]
    k_fold_dirs = sorted(k_fold_dirs, key=k_from_dir)
    print('Found job directories:')
    for d in k_fold_dirs:
        print(f'\t{d}')

    # Get PPM data, and list of all iterations encountered across jobs (some might have more than others)
    metadata_file_used = None
    metadata = None
    ppms_from_metadata = None
    # Iterate through k_fold dirs
    for k_fold_dir in k_fold_dirs:
        # If this is the first one, get metadata.json and read the PPM information
        if metadata_file_used is None:
            metadata_file_used = os.path.join(k_fold_dir, 'metadata.json')
            with open(metadata_file_used) as f:
                metadata = json.loads(f.read())
                ppms_from_metadata = metadata['Region set']['ppms']

        # Look in predictions directory to get all iterations from prediction images
        pred_dir = os.path.join(k_fold_dir, 'predictions')
        if os.path.isdir(pred_dir):
            # Get all filenames in that directory
            names = os.listdir(pred_dir)
            # Filter out those that do not match the desired name format
            names = list(filter(lambda name: re.match('.*_prediction_', name), names))
            for name in names:
                # Get PPM filename from image filename
                ppm_filename = re.search('(.*)_prediction_', name).group(1)
                # Note PPM size based on image size. Would get this from the PPM file
                # headers, but those are not necessarily on the machine this script is run on.
                for ppm in ppms_from_metadata.values():
                    # Locate correct PPM in the metadata
                    if os.path.splitext(os.path.basename(ppm['path']))[0] == ppm_filename:
                        # Only add size information if we haven't already
                        if 'size' not in ppm:
                            image_path = os.path.join(pred_dir, name)
                            ppm['size'] = Image.open(image_path).size
                        # Extract iteration name from each image
                        if 'iterations' not in ppm:
                            ppm['iterations'] = []
                        iteration = None
                        if re.match(r'.*_prediction_\d+_(\d+)', name):
                            iteration = re.search(r'.*_prediction_\d+_(\d+)', name).group(1)
                        elif re.match('.*_prediction_final', name):
                            iteration = 'final'
                        if iteration is not None and iteration not in ppm['iterations']:
                            ppm['iterations'].append(iteration)
    print(f'\nFound PPMs from {metadata_file_used}:')
    for ppm in ppms_from_metadata.keys():
        print(f'\t{ppm} {ppms_from_metadata[ppm]["size"]}')

    encountered_iterations = set()
    for ppm in ppms_from_metadata.values():
        for iteration in ppm['iterations']:
            encountered_iterations.add(iteration)
    encountered_iterations = sort_iterations(encountered_iterations)

    # Tensorboard
    multiplexer = EventMultiplexer().AddRunsFromDirectory(args.dir)
    multiplexer.Reload()
    for run in multiplexer.Runs():
        accumulator = multiplexer.GetAccumulator(run)
        print(accumulator.Scalars('train_loss'))
        # TODO LEFT OFF make plots or something with this info :)

    label_type = metadata.get('Arguments').get('label_type')

    # Generate training animation
    print('\nCreating animation:')
    animation = create_animation(k_fold_dirs, encountered_iterations,
                                 ppms_from_metadata, label_type, args.max_size)

    # TODO generate final frame and save to image at full/high res

    # Write to image sequence
    if args.img_seq is not None:
        write_img_sequence(animation, args.img_seq)

    # Write to gif
    write_gif(animation, os.path.join(args.dir, 'training.gif'), args.gif_delay)




    # Transfer results via rclone if requested
    inkid.ops.rclone_transfer_to_remote(args.rclone_transfer_remote, args.dir)


if __name__ == '__main__':
    main()
