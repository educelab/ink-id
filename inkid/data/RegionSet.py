import json

from jsmin import jsmin
import tensorflow as tf

import inkid

class RegionSet:
    def __init__(self, data):
        self._regions = []
        self._ppms = {}
        self._volumes = {}

        self._region_groups = {}

        region_counter = 0
        
        for region_group in data['regions']:
            self._region_groups[region_group] = []
            
            for region_data in data['regions'][region_group]:
                ppm_name = region_data['ppm']
                
                if ppm_name not in self._ppms:
                    ppm_data = data['ppms'][ppm_name]
                    ppm_path = ppm_data['path']
                    volume_path = ppm_data['volume']

                    mask_path = None
                    if 'mask' in ppm_data:
                        mask_path = ppm_data['mask']

                    ground_truth_path = None
                    if 'ground_truth' in ppm_data:
                        ground_truth_path = ppm_data['ground_truth']
                    
                    if volume_path not in self._volumes:
                        self._volumes[volume_path] = inkid.data.Volume(volume_path)

                    volume = self._volumes[volume_path]
                    self._ppms[ppm_name] = inkid.data.PPM(ppm_path, volume, mask_path, ground_truth_path)

                ppm = self._ppms[ppm_name]

                bounds = None
                if 'bounds' in region_data:
                    bounds = region_data['bounds']
                    
                region = inkid.data.Region(region_counter, ppm, bounds)
                    
                self._regions.append(region)
                self._region_groups[region_group].append(region_counter)
                
                region_counter += 1
                    

    @classmethod
    def from_json(cls, filename):
        """Initialize a RegionSet from a JSON filename."""
        with open(filename, 'r') as f:
            # minify to remove comments
            minified = jsmin(str(f.read()))
            data = json.loads(minified)
            return cls(data)


    def num_regions(self):
        return sum([len(self._region_groups[region_group]) for region_group in self._region_groups])


    def train_input_fn(self, batch_size):
        """TODO"""
        pass


    def evaluate_input_fn(self, batch_size):
        """TODO"""
        pass


    def predict_input_fn(self, batch_size):
        """TODO"""
        pass
    

    def points_with_normals_dataset(self,
                                    region_group,
                                    restrict_to_surface=False,
                                    perform_shuffle=False,
                                    grid_spacing=None,
                                    probability_of_selection=None):
        """TODO"""
        points_with_normals = np.array([])
        for region in self._regions[region_group]:
            points_with_normals = np.concatenate(
                points_with_normals,
                region.get_points_with_normals(
                    restrict_to_surface=restrict_to_surface,
                    grid_spacing=grid_spacing,
                    probability_of_selection=probability_of_selection
                )
            )
        if perform_shuffle:
            np.shuffle(points_with_normals)
        return points_with_normals


    def point_with_normal_to_subvolume(self,
                                       point_with_normal,
                                       radians_about_normal_axis,
                                       subvolume_shape,
                                       translation_along_normal):
        """TODO

        Return None if the subvolume is not bounded in the original
        volume.

        https://en.wikipedia.org/wiki/Euler_angles
        https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation

        """
        pass
