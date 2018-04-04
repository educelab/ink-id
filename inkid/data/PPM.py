import os.path

class PPM:
    def __init__(self, path, volume, mask_path, ground_truth_path):
        self._path = path
        self._volume = volume

        if mask_path is not None:
            self._mask_path = mask_path
        else:
            
            
            
        self._ground_truth_path = ground_truth_path


    def default_bounds(self):
        return [0, 0, 1440, 2530] # TODO make this actually work
