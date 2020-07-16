import argparse
import functools
import math
import numpy as np
import os

import torch

import inkid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input region JSON')
    parser.add_argument('output', help='Directory to hold output subvolumes')
    parser.add_argument('--number', '-n', metavar='N', default=4, type=int,
                        help='number of subvolumes to keep')
    parser.add_argument('--ink', action='store_true', help='Restrict to points on ink areas')
    parser.add_argument('--no-ink', action='store_true', help='Restrict to points not on ink areas')
    parser.add_argument('--ink-mask', help='Specify a different ink mask from the default in the region JSON',
                        default=None)
    parser.add_argument('--concat-subvolumes', action='store_true',
                        help='Create one set of slices containing all subvolumes')
    inkid.ops.add_subvolume_args(parser)

    # Data organization/augmentation
    parser.add_argument('--jitter-max', metavar='n', type=int)
    parser.add_argument('--augmentation', action='store_true', dest='augmentation')
    parser.add_argument('--no-augmentation', action='store_false', dest='augmentation')
    args = parser.parse_args()

    region_data = inkid.data.RegionSet.get_data_from_file(args.input)
    os.makedirs(args.output, exist_ok=True)

    if args.ink_mask is not None:
        for ppm in region_data['ppms']:
            region_data['ppms'][ppm]['ink-label'] = args.ink_mask

    region_set = inkid.data.RegionSet(region_data)

    point_to_subvolume_input = functools.partial(
        region_set.point_to_subvolume_input,
        subvolume_shape=args.subvolume_shape,
        out_of_bounds='all_zeros',
        move_along_normal=args.move_along_normal,
        method=args.subvolume_method,
        normalize=args.normalize_subvolumes,
        pad_to_shape=args.pad_to_shape,
        fft=args.fft,
        dwt=args.dwt,
        augment_subvolume=args.augmentation,
        jitter_max=args.jitter_max,
    )

    if args.ink:
        specify_inkness = True
    elif args.no_ink:
        specify_inkness = False
    else:
        specify_inkness = None

    points_ds = inkid.data.PointsDataset(region_set, ['training', 'validation', 'prediction'], point_to_subvolume_input,
                                         specify_inkness=specify_inkness)

    points_dl = torch.utils.data.DataLoader(points_ds, shuffle=True)

    square_side_length = math.ceil(math.sqrt(args.number))
    subvolume_shape = args.pad_to_shape or args.subvolume_shape
    pad = 20
    padded_shape = [i + pad * 2 for i in subvolume_shape]
    concatenated_shape = [padded_shape[0],
                          padded_shape[1] * square_side_length,
                          padded_shape[2] * square_side_length]
    concatenated_subvolumes = np.zeros(concatenated_shape)

    counter = 0
    for subvolume in points_dl:
        if counter >= args.number:
            break
        subvolume = subvolume.numpy()[0][0]
        if args.concat_subvolumes:
            concat_x = (counter // square_side_length) * padded_shape[2]
            concat_y = (counter % square_side_length) * padded_shape[1]
            subvolume = np.pad(subvolume, pad)
            concatenated_subvolumes[0:padded_shape[0],
                                    concat_y:concat_y + padded_shape[1],
                                    concat_x:concat_x + padded_shape[2]] = subvolume
        else:
            inkid.ops.save_volume_to_image_stack(subvolume, os.path.join(args.output, str(counter)))
        counter += 1

    if args.concat_subvolumes:
        inkid.ops.save_volume_to_image_stack(concatenated_subvolumes, args.output)


if __name__ == '__main__':
    main()
