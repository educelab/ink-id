import argparse
import os
import re

from tqdm import tqdm
import wand.image

import inkid

# TODO handle epochs in filenames


# Return which k-fold job this directory corresponds to.
# For the purpose of sorting a list of dirs based on job #.
def k_from_dir(dirname):
    # Only interested in the basename, other parts of the path do not matter
    dirname = os.path.basename(dirname)
    # Only has a valid k-fold # if it matches this particular format
    if re.match('.*\d\d\d\d-\d\d-\d\d_\d\d\.\d\d\.\d\d_(\d+)', dirname):
        # Return the last number in the dirname, which is the k-fold #
        return int(re.findall('(\d+)', dirname)[-1])
    else:
        # Otherwise probably was standalone job (not part of k-fold)
        return -1


# Return a prediction image of the specified iteration, k-fold directory,
# and PPM. If such an image does not exist return None
def get_prediction_image(iteration, k_fold_dir, ppm):
    filename = ppm + '_prediction_0_' + str(iteration) + '.png'
    full_filename = os.path.join(k_fold_dir, 'predictions', filename)
    if os.path.isfile(full_filename):
        return wand.image.Image(filename=full_filename)
    else:
        return None


def build_frame(iteration, k_fold_dirs, ppms, caption_with_iteration):
    img_cols = []
    for d in k_fold_dirs:
        new_col = []
        for ppm in ppms:
            img = get_prediction_image(iteration, d, ppm)
            new_col.append(img)
        img_cols.append(new_col)
    col_width = max([img.width for img in img_cols[0] if img is not None])
    print(col_width)
    # TODO blank frame
    # TODO fill in frame
    # TODO add label column
    # TODO return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # Input directory with job output
    parser.add_argument('dir', metavar='path', help='input directory')
    # Image generation options
    parser.add_argument('--img-seq', help='Generate an image sequence and save it to the provided directory')
    parser.add_argument('--gif-delay', default=10, type=int, help='GIF frame delay in hundredths of a second')
    parser.add_argument('--caption-gif-with-iterations', action='store_true')
    # Rclone upload options
    parser.add_argument('--rclone-transfer-remote', metavar='remote', default=None,
                        help='if specified, and if matches the name of one of the directories in '
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

    # Get list of all PPMs and iterations encountered
    encountered_ppms = set()
    encountered_iterations = set()
    for k_fold_dir in k_fold_dirs:
        pred_dir = os.path.join(k_fold_dir, 'predictions')
        if os.path.isdir(pred_dir):
            # Get all filenames in that directory
            names = os.listdir(pred_dir)
            # Filter out those that do not match the desired name format
            names = list(filter(lambda name: re.match('.*_prediction_', name), names))
            for name in names:
                # Extract the ppm name from each
                ppm = re.search('(.*)_prediction_', name).group(1)
                encountered_ppms.add(ppm)
                # Extract iteration name from each
                if re.match('.*_prediction_\d+_(\d+)', name):
                    iteration = re.search('.*_prediction_\d+_(\d+)', name).group(1)
                    encountered_iterations.add(int(iteration))
    print('\nFound PPMs:')
    for ppm in encountered_ppms:
        print(f'\t{ppm}')
    encountered_iterations = sorted(list(encountered_iterations))

    # Generate training animation
    print('\nCreating animation:')
    animation = create_animation(k_fold_dirs, encountered_iterations,
                                 encountered_ppms, args.caption_gif_with_iterations)

    # TODO generate final frame and save to image

    # Write to image sequence
    if args.img_seq:
        write_img_sequence(animation, args.img_seq)

    # Write to gif
    write_gif(animation, os.path.join(args.dir, 'training.gif'), args.gif_delay)

    # Transfer results via rclone if requested
    inkid.ops.rclone_transfer_to_remote(args.rclone_transfer_remote, args.dir)


def create_animation(k_fold_dirs, iterations, ppms, caption_with_iteration):
    animation = wand.image.Image()
    for iteration in tqdm(iterations):
        frame = build_frame(iteration, k_fold_dirs, ppms, caption_with_iteration)
        # animation.sequence.append(frame)
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
    if len(animation.sequence) == 0:
        return
    print('\nWriting training image sequence to', outdir)
    prefix = os.path.join(outdir, "sequence_")
    images = animation.sequence
    for i in range(len(images)):
        outfile = prefix + str(i) + ".png"
        wand.image.Image(images[i]).save(filename=outfile)


def write_gif(animation, outfile, delay=10):
    if len(animation.sequence) == 0:
        return
    print('\nWriting training gif to', outfile)
    gif = animation
    for frame in gif.sequence:
        frame.delay = delay
    gif.type = 'optimize'
    gif.save(filename=outfile)

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


if __name__ == '__main__':
    main()
