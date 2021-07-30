import argparse
from pathlib import Path
import struct

import numpy as np
from PIL import Image
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dummy_image')
    parser.add_argument('output_ppm')
    parser.add_argument('-z', default=0)
    args = parser.parse_args()

    img = Image.open(args.dummy_image).convert('L')
    version = 1
    dim = 6
    w, h = img.size

    ppm_data = np.zeros((h, w, dim))
    for y in range(h):
        for x in range(w):
            ppm_data[y, x] = [x, y, args.z, 0, 0, 1]

    mask_img = np.zeros_like(img)
    ink_label_img = np.zeros_like(img)
    rgb_label_img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            v = img.getpixel((x, y))
            if v == 0:
                rgb_label_img[y, x] = [0, 51, 160]  # Blue
            elif 0 < v < 255:
                mask_img[y, x] = 255
                rgb_label_img[y, x] = [210, 180, 140]  # Brown
            else:
                mask_img[y, x] = 255
                ink_label_img[y, x] = 255

    mask_img = Image.fromarray(mask_img, 'L')
    ink_label_img = Image.fromarray(ink_label_img, 'L')
    rgb_label_img = Image.fromarray(rgb_label_img, 'RGB')

    filename = Path(args.output_ppm).stem
    mask_img.save(filename + '_mask.png')
    ink_label_img.save(filename + '_ink-mask.png')
    rgb_label_img.save(filename + '_rgb-mask.png')

    with open(args.output_ppm, 'wb') as f:
        print('Writing PPM to file {}...'.format(args.output_ppm))
        f.write('width: {}\n'.format(w).encode('utf-8'))
        f.write('height: {}\n'.format(h).encode('utf-8'))
        f.write('dim: {}\n'.format(dim).encode('utf-8'))
        f.write('ordered: {}\n'.format('true').encode('utf-8'))
        f.write('type: double\n'.encode('utf-8'))
        f.write('version: {}\n'.format(version).encode('utf-8'))
        f.write('<>\n'.encode('utf-8'))
        for y in tqdm(range(h)):
            for x in range(w):
                for idx in range(dim):
                    f.write(struct.pack('d', ppm_data[y, x, idx]))


if __name__ == '__main__':
    main()
