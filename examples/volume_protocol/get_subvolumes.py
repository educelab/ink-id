"""Example of getting subvolumes from a Volume Server."""
from PIL import Image
from inkid.volumes.volume_protocol import RequestArgs, get_subvolumes
import sys

remote_host = sys.argv[1] if (len(sys.argv) > 1 and len(sys.argv[1] > 0)) else "127.0.0.1"
remote_port = int(sys.argv[2]) if (len(sys.argv) > 2 and len(sys.argv[2] > 0)) else 8087

request = RequestArgs(volpkg="PHercParis Objet 59", volume="20190910132130",
                      center_x=152.5, center_y=485.0, center_z=605.0,
                      sampling_r_x=152.5, sampling_r_y=485.0, sampling_r_z=1.0)
subvolumes = get_subvolumes([request], server=(remote_host, remote_port))
for args, data in subvolumes:
    print(f"Got back a subvolume: {args}")
    # The data is a list of 16-bit unsigned intensity values, but 8-bit values
    # are fine for displaying. Since the data is represented in little endian,
    # we just drop all the least-significant (even) bytes to get an 8-bit
    # downsampled version.
    downsampled = bytes([data[i] for i in range(len(data)) if i % 2 == 1])
    # Compute all of the available slices.
    for z in range(args.extent_z):
        slice_data = bytearray()
        for y in range(args.extent_y):
            for x in range(args.extent_x):
                true_index = (z * args.extent_y * args.extent_x) + \
                    (y * args.extent_x) + x
                slice_data.append(downsampled[true_index])
        image = Image.frombytes(
            "L", (args.extent_x, args.extent_y), bytes(slice_data))
        image_name = f"example-slice{z}.tiff"
        image.save(image_name)
        print(f"Saved slice {z} as: {image_name}")
