import os

class PPM:
    def __init__(self, path, volume, mask_path, ground_truth_path):
        self._path = path
        self._volume = volume

        ppm_path_without_file_extension, ext = os.path.splitext(self._path)

        if mask_path is not None:
            self._mask_path = mask_path
        else:
            default_mask_path = ppm_path_without_file_extension + '_mask.png'
            if os.path.isfile(default_mask_path):
                self._mask_path = default_mask_path

        if ground_truth_path is not None:
            self._ground_truth_path = ground_truth_path
        else:
            default_ground_truth_path = ppm_path_without_file_extension + '_ground-truth.png'
            if os.path.isfile(default_ground_truth_path):
                self._ground_truth_path = default_ground_truth_path
