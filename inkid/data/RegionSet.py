import json
import os

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
                ppm = self.create_ppm_if_needed(region_data['ppm'], data['ppms'][region_data['ppm']])
                bounds = region_data.get('bounds')
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
            data = RegionSet.make_data_paths_absolute(data, os.path.dirname(filename))
            return cls(data)


    @staticmethod
    def make_data_paths_absolute(data, paths_were_relative_to=os.getcwd()):
        """Convert the input data paths to absolute paths.

        This was designed for the case where the user passes a JSON
        data input file to a script, containing paths that were
        relative to the location of the data file and not to the
        script. In this case you would set paths_were_relative_to to
        the dirname of the JSON file.

        """
        for ppm in data['ppms']:
            for key in data['ppms'][ppm]:
                data['ppms'][ppm][key] = os.path.normpath(
                    os.path.join(
                        paths_were_relative_to,
                        data['ppms'][ppm][key]
                    )
                )
        return data
        

    def create_ppm_if_needed(self, ppm_name, ppm_data):
        """Return the ppm from its name and data, creating first if needed.

        Check if the provided ppm_name already has a ppm stored in the
        RegionSet. If so, go ahead and return it. If not, create the
        ppm object first and then return it. When creating the ppm,
        perform the same check for that ppm's volume (create it if
        needed, and then get the object so the ppm object can hold a
        reference to it).

        """
        if ppm_name not in self._ppms:
            ppm_path = ppm_data['path']
            volume_path = ppm_data['volume']
            
            # .get() will return None if the key is not
            # defined (which is good here).
            mask_path = ppm_data.get('mask')
            ground_truth_path = ppm_data.get('ground_truth')

            # TODO handle paths relative to input file
            if volume_path not in self._volumes:
                self._volumes[volume_path] = inkid.data.Volume(volume_path)

            volume = self._volumes[volume_path]
            self._ppms[ppm_name] = inkid.data.PPM(ppm_path, volume, mask_path, ground_truth_path)

        return self._ppms[ppm_name]


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
        print('Fetching points/normals for region group: {}... '.format(region_group), end='')
        points_with_normals = []
        for region in self._regions[region_group]:
            points_with_normals.append(
                region.get_points_with_normals(
                    restrict_to_surface=restrict_to_surface,
                    grid_spacing=grid_spacing,
                    probability_of_selection=probability_of_selection
                )
            )
        points_with_normals = np.array(points_with_normals)
        print('done.')
        if perform_shuffle:
            print('Shuffling points/normals for region group: {}... '.format(region_group), end='')
            np.shuffle(points_with_normals)
            print('done.')
        return points_with_normals


    def point_with_normal_to_subvolume(self,
                                       point_with_normal,
                                       subvolume_shape,
                                       translation_along_normal):
        """TODO

        Return None if the subvolume is not bounded in the original
        volume.

        """
        assert(len(point_with_normal) == 6)
        assert(len(subvolume_shape) == 3)
