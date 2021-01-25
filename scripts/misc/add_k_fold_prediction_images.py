import argparse
import datetime
import json
import os
from pathlib import Path, PurePath
import re
import warnings

from humanize import naturalsize
import imageio
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pygifsicle
from scipy.signal import savgol_filter
from tensorboard.backend.event_processing.event_multiplexer import EventMultiplexer
from tensorboard.backend.event_processing.event_accumulator import STORE_EVERYTHING_SIZE_GUIDANCE
from tqdm import tqdm

import inkid

# General note:
# There is a lot of tedious string processing in this file. Parsing the
# filenames to get the PPM, iteration, and epoch is annoying and fragile. It would
# be better to use EXIF data or similar, perhaps integrated with Smeagol.
# Or perhaps to develop the RegionSet class some more instead of relying so much
# on directly reading the metadata dictionary.

# We might issue this warning but only need to do it once
already_warned_about_missing_label_images = False

WHITE = (255, 255, 255)
LIGHT_GRAY = (104, 104, 104)
DARK_GRAY = (64, 64, 64)
RED = (255, 0, 0)


# Return whether the given directory matches the expected structure for a k-fold job dir.
# For example we often want a list of the k-fold subdirs in a directory but do not want
# e.g. previous summary subdirs, or others.
def is_k_fold_dir(dirname):
    dirname = os.path.basename(dirname)
    return re.match(r'.*\d\d\d\d-\d\d-\d\d_\d\d\.\d\d\.\d\d_(\d+)', dirname)


# Return which k-fold job this directory corresponds to.
# For the purpose of sorting a list of dirs based on job #.
def k_from_dir(dirname):
    # Only has a valid k-fold # if it matches this particular format
    if is_k_fold_dir(dirname):
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


# Reading Tensorboard files similar to this method https://stackoverflow.com/a/41083104
def create_tensorboard_plots(base_dir, out_dir):
    out_dir = os.path.join(out_dir, 'plots')
    os.makedirs(out_dir, exist_ok=True)
    guidance = STORE_EVERYTHING_SIZE_GUIDANCE
    multiplexer = EventMultiplexer(size_guidance=guidance).AddRunsFromDirectory(base_dir)
    multiplexer.Reload()
    scalars = []
    for run in multiplexer.Runs():
        scalars = multiplexer.GetAccumulator(run).Tags()['scalars']
        break
    # Disable warning when making more than 20 figures
    plt.rcParams.update({'figure.max_open_warning': 0})
    for scalar in scalars:
        for smooth in [True, False]:
            fig, ax = plt.subplots()
            for run in multiplexer.Runs():
                run_path = PurePath(run)
                label = k_from_dir(run_path.parts[0])
                accumulator = multiplexer.GetAccumulator(run)
                step = [x.step for x in accumulator.Scalars(scalar)]
                value = [x.value for x in accumulator.Scalars(scalar)]
                if smooth:
                    # Make smoothing window a fraction of number of values
                    smoothing_window = len(value) / 500
                    # Ensure odd number, required for savgol_filter
                    smoothing_window = int(smoothing_window / 2) * 2 + 1
                    # For small sets make sure we at least do some smoothing
                    smoothing_window = max(smoothing_window, 5)
                    value = savgol_filter(value, smoothing_window, 3)
                plt.plot(step, value, label=label)
            title = scalar
            if smooth:
                title += ' (smoothed)'
            ax.set(xlabel='step', ylabel=scalar, title=title)
            ax.grid()
            if len(multiplexer.Runs()) > 1:
                plt.legend(title='k-fold job')
            fig.savefig(os.path.join(out_dir, f'{title}.png'))


# Return a prediction image of the specified iteration, k-fold directory,
# and PPM. If such an image does not exist return None
def get_prediction_image(iteration, k_fold_dir, ppm_name, return_latest_if_not_found=True):
    filename = f'{ppm_name}_prediction_{iteration}.png'
    full_filename = os.path.join(k_fold_dir, 'predictions', filename)
    img = None
    if os.path.isfile(full_filename):
        img = Image.open(full_filename)
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
                img = Image.open(full_filename)
    if img is not None:
        # For some images (grayscale PNGs in my experience so far), Pillow opens them as
        # 32-bit integer images and then clips them to 8-bit in later stages, creating
        # washed out images. If it has opened an image as 32-bit we detect and convert it
        # here, so it is not clipped later.
        if img.mode == 'I':
            array = np.uint8(np.array(img) / 256)
            img = Image.fromarray(array)
    return img.convert('RGB')


def build_footer_img(width, height, iteration, label_type, cmap_name=None, label_training_regions=False):
    footer = Image.new('RGB', (width, height))
    horizontal_offset = 0
    divider_bar_size = max(1, int(width / 500))
    buffer_size = int(height / 8)
    # Fill with dark gray
    footer.paste(DARK_GRAY, (0, 0, width, height))
    # Add logos
    for logo_filename in ['EduceLabBW.png', 'UK logo-white.png']:
        logo_path = os.path.join(os.path.dirname(inkid.__file__), 'assets', logo_filename)
        logo = Image.open(logo_path)
        logo.thumbnail((100000, height - 2 * buffer_size), Image.BICUBIC)
        logo_offset = (horizontal_offset + buffer_size, buffer_size)
        # Third argument used as transparency mask. Convert to RGBA to force presence of alpha channel
        footer.paste(logo, logo_offset, logo.convert('RGBA'))
        horizontal_offset += logo.size[0] + 2 * buffer_size
        # Add divider bar
        footer.paste(
            LIGHT_GRAY,
            (horizontal_offset, 0, horizontal_offset + divider_bar_size, height)
        )
        horizontal_offset += divider_bar_size
    # Add iteration/batch #
    draw = ImageDraw.Draw(footer)
    font_path = os.path.join(os.path.dirname(inkid.__file__), 'assets', 'fonts', 'Roboto-Regular.ttf')
    fontsize = 1
    font_regular = ImageFont.truetype(font_path, fontsize)
    txt = 'training batch'
    allowed_font_height = int((height - buffer_size * 3) / 2)
    while font_regular.getsize(txt)[1] < allowed_font_height:
        fontsize += 1
        font_regular = ImageFont.truetype(font_path, fontsize)
    fontsize -= 1
    draw.text(
        (horizontal_offset + buffer_size, buffer_size),
        txt,
        WHITE,
        font=font_regular
    )
    font_w = font_regular.getsize(txt)[0] + 2 * buffer_size
    font_path_black = os.path.join(os.path.dirname(inkid.__file__), 'assets', 'fonts', 'Roboto-Black.ttf')
    font_black = ImageFont.truetype(font_path_black, fontsize)
    if re.match(r'\d+_\d+', iteration):
        iteration = re.search(r'\d+_(\d+)', iteration).group(1)
    draw.text(
        (horizontal_offset + buffer_size, allowed_font_height + 2 * buffer_size),
        iteration,
        WHITE,
        font=font_black
    )
    horizontal_offset += font_w
    # Add divider bar
    footer.paste(
        LIGHT_GRAY,
        (horizontal_offset, 0, horizontal_offset + divider_bar_size, height)
    )
    horizontal_offset += divider_bar_size
    # Add color map name
    if cmap_name is not None:
        cmap_title = 'color map'
        draw.text(
            (horizontal_offset + buffer_size, buffer_size),
            cmap_title,
            WHITE,
            font=font_regular
        )
        draw.text(
            (horizontal_offset + buffer_size, allowed_font_height + 2 * buffer_size),
            cmap_name,
            WHITE,
            font=font_black
        )
        font_w = max(font_regular.getsize(cmap_title)[0], font_black.getsize(cmap_name)[0])
        horizontal_offset += font_w + 2 * buffer_size
        # Add divider bar
        footer.paste(
            LIGHT_GRAY,
            (horizontal_offset, 0, horizontal_offset + divider_bar_size, height)
        )
        horizontal_offset += divider_bar_size
    # Add color map swatch
    if label_type == 'ink_classes':
        swatch_title = 'no ink            ink'
        swatch = Image.new('RGB', font_regular.getsize(swatch_title))
        for x in range(swatch.width):
            intensity = int((x / swatch.width) * 255)
            swatch.paste(
                (intensity, intensity, intensity),
                (x, 0, x + 1, swatch.height)
            )
        if cmap_name is not None:
            color_map = cm.get_cmap(cmap_name)
            swatch = swatch.convert('L')
            img_data = np.array(swatch)
            swatch = Image.fromarray(np.uint8(color_map(img_data) * 255))
        draw.text(
            (horizontal_offset + buffer_size, buffer_size),
            swatch_title,
            WHITE,
            font=font_regular
        )
        footer.paste(
            swatch,
            (horizontal_offset + buffer_size, allowed_font_height + 2 * buffer_size)
        )
        horizontal_offset += swatch.width + 2 * buffer_size
        # Add divider bar
        footer.paste(
            LIGHT_GRAY,
            (horizontal_offset, 0, horizontal_offset + divider_bar_size, height)
        )
        horizontal_offset += divider_bar_size
    if label_training_regions:
        training_regions_title = 'training regions'
        draw.text(
            (horizontal_offset + buffer_size, buffer_size),
            training_regions_title,
            WHITE,
            font=font_regular
        )
        draw = ImageDraw.Draw(footer)
        x0 = horizontal_offset + buffer_size
        y0 = allowed_font_height + 2 * buffer_size
        x1 = x0 + font_regular.getsize(training_regions_title)[0]
        y1 = y0 + font_regular.getsize(training_regions_title)[1]
        draw.rectangle((x0, y0, x1, y1), outline=RED, fill=None, width=int(height / 40))
        horizontal_offset += font_regular.getsize(training_regions_title)[0] + 2 * buffer_size
        # Add divider bar
        footer.paste(
            LIGHT_GRAY,
            (horizontal_offset, 0, horizontal_offset + divider_bar_size, height)
        )
        horizontal_offset += divider_bar_size
    return footer


def build_frame(iteration, k_fold_dir_to_metadata, ppms, label_type, max_size=None, cmap_name=None,
                label_training_regions=False):
    # TODO: regions_to_include=['training', 'prediction', 'validation'],
    #  merge_all_of_same_PPM=False
    global already_warned_about_missing_label_images

    k_fold_dirs = k_fold_dir_to_metadata.keys()
    col_width = max([ppm['size'][0] for ppm in ppms.values()])
    row_heights = [ppm['size'][1] for ppm in ppms.values()]
    buffer_size = int(col_width / 10)
    width = col_width * (len(k_fold_dirs) + 1) + buffer_size * (len(k_fold_dirs) + 2)
    height = sum(row_heights) + buffer_size * (len(row_heights) + 1)
    # Add space for footer
    footer_height = int(width / 16)
    line_width = int(footer_height / 40)
    height += footer_height
    # Create empty frame
    frame = Image.new('RGB', (width, height))
    # Add prediction images
    # One column at a time
    for k, k_fold_dir in enumerate(k_fold_dirs):
        # Get metadata for this job
        metadata = k_fold_dir_to_metadata[k_fold_dir]
        # Make each row of this column
        for ppm_i, (ppm_name, ppm) in enumerate(ppms.items()):
            ppm_path = os.path.splitext(os.path.basename(ppm['path']))[0]
            img = get_prediction_image(iteration, k_fold_dir, ppm_path)
            if img is not None:
                offset = (
                    k * col_width + (k + 1) * buffer_size,
                    sum(row_heights[:ppm_i]) + (ppm_i + 1) * buffer_size
                )
                if cmap_name is not None:
                    color_map = cm.get_cmap(cmap_name)
                    img = img.convert('L')
                    img_data = np.array(img)
                    img = Image.fromarray(np.uint8(color_map(img_data) * 255))
                if label_training_regions:
                    training_regions = metadata['Region set']['regions']['training']
                    for region in training_regions:
                        if region['ppm'] == ppm_name:
                            if 'bounds' not in region:
                                region['bounds'] = (0, 0, img.width, img.height)
                        draw = ImageDraw.Draw(img)
                        draw.rectangle(region['bounds'], outline=RED, fill=None, width=line_width)
                frame.paste(img, offset)
    # Add label column
    for ppm_i, ppm in enumerate(ppms.values()):
        if label_type == 'rgb_values':
            label_key = 'rgb-label'
        elif label_type == 'ink_classes':
            label_key = 'ink-label'
        else:
            break
        label_img_path = ppm[label_key]
        # Try getting label image file from recorded location (may not exist on this machine)
        label_img = None
        if os.path.isfile(label_img_path):
            label_img = Image.open(label_img_path)
        # If not there, maybe it is on the local machine under ~/data.
        elif '/pscratch/seales_uksr/' in label_img_path:
            label_img_path = label_img_path.replace('/pscratch/seales_uksr/', '')
            label_img_path = os.path.join(Path.home(), 'data', label_img_path)
            if os.path.isfile(label_img_path):
                label_img = Image.open(label_img_path)
        if label_img is not None:
            if cmap_name is not None:
                color_map = cm.get_cmap(cmap_name)
                label_img = label_img.convert('L')
                img_data = np.array(label_img)
                label_img = Image.fromarray(np.uint8(color_map(img_data) * 255))
            offset = (
                len(k_fold_dirs) * col_width + (len(k_fold_dirs) + 1) * buffer_size,
                sum(row_heights[:ppm_i]) + (ppm_i + 1) * buffer_size
            )
            frame.paste(label_img, offset)
        elif not already_warned_about_missing_label_images:
            warnings.warn(
                'At least one label image not found, check if dataset locally available',
                RuntimeWarning
            )
            already_warned_about_missing_label_images = True

    # Add footer
    footer = build_footer_img(width, footer_height, iteration, label_type, cmap_name,
                              label_training_regions)
    frame.paste(footer, (0, height - footer_height))

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
    pygifsicle.optimize(outfile, options=['-w'])
    new_size = os.path.getsize(outfile)
    reduction = (prev_size - new_size) / prev_size * 100
    print(f'Size reduced {reduction:.2f}% from {naturalsize(prev_size)} to {naturalsize(new_size)}')


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
    parser.add_argument('--gif-fps', default=10, type=int, help='GIF frames per second')
    parser.add_argument('--gif-max-size', type=int, nargs=2, default=[1920, 1080])
    parser.add_argument('--static-max-size', type=int, nargs=2, default=[3840, 2160])
    # Rclone upload options
    parser.add_argument('--rclone-transfer-remote', metavar='remote', default=None,
                        help='if specified, and if matches the name of one of the directories in '
                        'the output path, transfer the results to that rclone remote into the '
                        'sub-path following the remote name')
    args = parser.parse_args()

    out_dir = f'{datetime.datetime.today().strftime("%Y-%m-%d_%H.%M.%S")}_summary'
    out_dir = os.path.join(args.dir, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Get list of directories (not files) in given parent dir
    possible_dirs = [os.path.join(args.dir, name) for name in os.listdir(args.dir)]
    k_fold_dirs = list(filter(is_k_fold_dir, possible_dirs))
    k_fold_dirs = sorted(k_fold_dirs, key=k_from_dir)
    print('Found job directories:')
    for d in k_fold_dirs:
        print(f'\t{d}')

    # Get PPM data, and list of all iterations encountered across jobs (some might have more than others)
    # Start by storing a dict where we map the directory name to the metadata for that job
    k_fold_dir_to_metadata = dict()
    ppms_from_metadata = None
    # Iterate through k_fold dirs
    for k_fold_dir in k_fold_dirs:
        # Read the metadata file and store each of them (we need the region info
        # from all k_fold directories)
        metadata_file = os.path.join(k_fold_dir, 'metadata.json')
        with open(metadata_file) as f:
            metadata = json.loads(f.read())
            k_fold_dir_to_metadata[k_fold_dir] = metadata

        # Useful to separately store just the PPM info (should be same across all jobs)
        ppms_from_metadata = list(k_fold_dir_to_metadata.values())[0]['Region set']['ppms']

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
    print(f'\nFound PPMs:')
    for ppm in ppms_from_metadata.keys():
        print(f'\t{ppm}, size: {ppms_from_metadata[ppm]["size"]}')

    encountered_iterations = set()
    for ppm in ppms_from_metadata.values():
        for iteration in ppm['iterations']:
            encountered_iterations.add(iteration)
    encountered_iterations = sort_iterations(encountered_iterations)

    label_type = metadata.get('Arguments').get('label_type')

    # # Tensorboard
    # print('\nCreating Tensorboard plots...')
    # create_tensorboard_plots(args.dir, out_dir)
    # print('done.')
    #
    # # Generate training animation
    # print('\nCreating animation:')
    # animation = create_animation(k_fold_dirs, encountered_iterations,
    #                              ppms_from_metadata, label_type, args.gif_max_size)
    #
    # # Write to image sequence
    # if args.img_seq is not None:
    #     write_img_sequence(animation, args.img_seq)
    #
    # # Write to gif
    # write_gif(animation, os.path.join(out_dir, 'training.gif'), args.gif_fps)

    # Write final frame to static image
    print('\nCreating final static image...')
    final_frame = build_frame('final', k_fold_dir_to_metadata, ppms_from_metadata, label_type,
                              args.static_max_size, label_training_regions=True)
    final_frame.save(os.path.join(out_dir, 'final.png'))
    print('done.')
    #
    # color_maps_dir = os.path.join(out_dir, 'colormaps')
    # os.makedirs(color_maps_dir, exist_ok=True)
    # for cmap in ['plasma', 'viridis', 'hot', 'inferno', 'seismic', 'Spectral', 'coolwarm', 'bwr']:
    #     print(f'\nCreating final static image with color map: {cmap}...')
    #     final_frame = build_frame('final', k_fold_dirs, ppms_from_metadata, label_type, args.static_max_size,
    #                               cmap_name=cmap)
    #     final_frame.save(os.path.join(color_maps_dir, f'final_{cmap}.png'))
    #     print('done.')

    # Transfer results via rclone if requested
    inkid.ops.rclone_transfer_to_remote(args.rclone_transfer_remote, args.dir)


if __name__ == '__main__':
    main()
