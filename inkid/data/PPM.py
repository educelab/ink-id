import os
import re
import struct
import sys

import numpy as np

class PPM:
    def __init__(self, path, volume, mask_path, ground_truth_path):
        self._path = path
        self._volume = volume
        self._predicted_ink_classes = None

        ppm_path_stem, ext = os.path.splitext(self._path)

        if mask_path is not None:
            self._mask_path = mask_path
        else:
            default_mask_path = ppm_path_stem + '_mask.png'
            if os.path.isfile(default_mask_path):
                self._mask_path = default_mask_path
            else:
                self._mask_path = None

        if ground_truth_path is not None:
            self._ground_truth_path = ground_truth_path
        else:
            default_ground_truth_path = ppm_path_stem + '_ground-truth.png'
            if os.path.isfile(default_ground_truth_path):
                self._ground_truth_path = default_ground_truth_path
            else:
                self._ground_truth_path = None

        self.process_PPM_file(self._path)

    def process_PPM_file(self, filename):
        comments_re = re.compile('^#')
        width_re = re.compile('^width')
        height_re = re.compile('^height')
        dim_re = re.compile('^dim')
        ordering_re = re.compile('^ordered')
        type_re = re.compile('^type')
        version_re = re.compile('^version')
        header_terminator_re = re.compile('^<>$')

        with open(filename, 'rb') as f:
            while True:
                line = f.readline().decode('utf-8')
                if comments_re.match(line):
                    pass
                elif width_re.match(line):
                    self._width = int(line.split(': ')[1])
                elif height_re.match(line):
                    self._height = int(line.split(': ')[1])
                elif dim_re.match(line):
                    self._dim = int(line.split(': ')[1])
                elif ordering_re.match(line):
                    self._ordering = line.split(': ')[1].strip() == 'true'
                elif type_re.match(line):
                    self._type = line.split(': ')[1].strip()
                    # assert(self._type in ['double', 'int', 'float'])
                    assert(self._type in ['double'])  # TODO support other data formats
                elif version_re.match(line):
                    self._version = line.split(': ')[1].strip()
                elif header_terminator_re.match(line):
                    break
                else:
                    print('Warning: PPM header contains unknown line: {}'.format(line.strip()))
            print(
                'Processing PPM data for {} with width {}, height {}, dim {}... '.format(
                    self._path, self._width, self._height, self._dim
                ),
                end='',
            )
            sys.stdout.flush()

            self._data = np.empty((self._height, self._width, self._dim))

            for y in range(self._height):
                for x in range(self._width):
                    for idx in range(self._dim):
                        self._data[y, x, idx] = struct.unpack('d', f.read(8))[0]

            print('done.')

    def get_default_bounds(self):
        return 
