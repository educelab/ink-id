import json
import logging
import os
import sys

from jsmin import jsmin
import numpy as np
import torch

import inkid


class PointsDataset(torch.utils.data.Dataset):
    def __init__(self, region_set, region_groups, feature_transform, label_transform=None, grid_spacing=None,
                 specify_inkness=None):
        self._region_set = region_set
        self._points = self._region_set.get_points(region_groups, grid_spacing=grid_spacing,
                                                   specify_inkness=specify_inkness)
        self._feature_transform = feature_transform
        self._label_transform = label_transform

    def __len__(self):
        return len(self._points)

    def __getitem__(self, idx):
        point = self._points[idx]
        feature = torch.from_numpy(self._feature_transform(point))[None, :, :, :]
        if self._label_transform is not None:
            label = torch.from_numpy(self._label_transform(point))
            return feature, label
        else:
            return feature


class RegionSet:
    def __init__(self, data):
        self._regions = []
        self._ppms = {}
        self._volumes = {}

        self._region_groups = {}

        for region_group in data['regions']:
            self._region_groups[region_group] = []

            for region_data in data['regions'][region_group]:
                ppm = self.create_ppm_if_needed(
                    region_data['ppm'],
                    data['ppms'][region_data['ppm']]
                )
                bounds = region_data.get('bounds')
                region = inkid.data.Region(len(self._regions), ppm, bounds)

                self._region_groups[region_group].append(len(self._regions))
                self._regions.append(region)

        ppm_names = [os.path.basename(ppm['path']) for ppm in data['ppms'].values()]
        for p in ppm_names:
            if ppm_names.count(p) > 1:
                err = f'Multiple PPMs with filename {p}, please use unique filenames.'
                logging.error(err)
                raise ValueError(err)

    @classmethod
    def from_json(cls, filename):
        """Initialize a RegionSet from a JSON filename."""
        data = RegionSet.get_data_from_file(filename)
        return cls(data)

    @staticmethod
    def get_data_from_file(filename):
        with open(filename, 'r') as f:
            # minify to remove comments
            minified = jsmin(str(f.read()))
            data = json.loads(minified)
            data = RegionSet.make_data_paths_absolute(data, os.path.dirname(filename))
            return data

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
                if isinstance(data['ppms'][ppm][key], str):
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
            ink_label_path = ppm_data.get('ink-label')
            rgb_label_path = ppm_data.get('rgb-label')

            if volume_path not in self._volumes:
                self._volumes[volume_path] = inkid.data.Volume(volume_path)

            volume = self._volumes[volume_path]

            invert_normal = ppm_data.get('invert_normal')
            self._ppms[ppm_name] = inkid.data.PPM(
                ppm_path,
                volume,
                mask_path,
                ink_label_path,
                rgb_label_path,
                invert_normal,
                ppm_name,
            )

        return self._ppms[ppm_name]

    def get_points(self, region_groups,
                   perform_shuffle=False, shuffle_seed=None,
                   grid_spacing=None,
                   probability_of_selection=None,
                   specify_inkness=None):
        """Return a numpy array of region_ids and points.

        Used as the initial input to a Dataset, which will later map
        these points to network inputs and do other dataset processing
        such as batching.

        """
        if isinstance(region_groups, str):
            region_groups = [region_groups]
        logging.info('Fetching points for region groups: {}... '
                     .format(region_groups))
        sys.stdout.flush()
        points = []
        for region_group in region_groups:
            for region_id in self._region_groups[region_group]:
                points += self._regions[region_id].get_points(
                    grid_spacing=grid_spacing,
                    probability_of_selection=probability_of_selection,
                    specify_inkness=specify_inkness
                )
        points = np.array(points)
        logging.info('done ({} points)'.format(len(points)))
        if perform_shuffle:
            logging.info('Shuffling points for region groups: {}... '
                         .format(region_groups))
            sys.stdout.flush()
            if shuffle_seed is not None:
                np.random.seed(shuffle_seed)
            # Dataset objects also have a .shuffle() method
            # which would be a more natural fit, but it is much slower
            # in practice.
            np.random.shuffle(points)
            logging.info('done')
        return points

    def get_points_generator(self, **kwargs):
        points = self.get_points(**kwargs)

        def generator():
            for point in points:
                yield point

        return generator

    def point_to_descriptive_statistics(self, region_id_with_point,
                                        subvolume_shape_voxels,
                                        subvolume_shape_microns):
        region_id, x, y = region_id_with_point
        subvolume = self._regions[region_id].ppm.point_to_subvolume(
            (x, y),
            subvolume_shape_voxels,
            subvolume_shape_microns
        )
        return np.asarray(inkid.ops.get_descriptive_statistics(subvolume), np.float32)

    def point_to_voxel_vector_input(self, region_id_with_point,
                                    length_in_each_direction, out_of_bounds=None):
        region_id, x, y = region_id_with_point
        voxel_vector = self._regions[region_id].ppm.point_to_voxel_vector(
            (x, y),
            length_in_each_direction,
            out_of_bounds=out_of_bounds,
        )
        return np.asarray(voxel_vector, np.float32)

    def point_to_subvolume_input(self, region_id_with_point,
                                 subvolume_shape_voxels, subvolume_shape_microns,
                                 out_of_bounds=None,
                                 move_along_normal=None,
                                 jitter_max=None,
                                 augment_subvolume=None, method=None,
                                 normalize=None, model_3d_to_2d=None):
        """Take a region_id and (x, y) point, and return a subvolume.

        First use the PPM (x, y) to find the 3D position and normal
        orientation of this point in the Volume. Then get a subvolume
        from the Volume at that position.

        """
        region_id, x, y = region_id_with_point
        subvolume = self._regions[region_id].ppm.point_to_subvolume(
            (x, y),
            subvolume_shape_voxels,
            subvolume_shape_microns,
            out_of_bounds=out_of_bounds,
            move_along_normal=move_along_normal,
            jitter_max=jitter_max,
            augment_subvolume=augment_subvolume,
            method=method,
            normalize=normalize,
            model_3d_to_2d=model_3d_to_2d,
        )
        return np.asarray(subvolume, np.float32)

    def point_to_ink_classes_label(self, region_id_with_point, shape=(1, 1)):
        """Take a region_id and point, and return the ink classes label."""
        region_id, x, y = region_id_with_point
        return self._regions[region_id].ppm.point_to_ink_classes_label((x, y), shape=shape)

    def point_to_rgb_values_label(self, region_id_with_point, shape=(1, 1)):
        region_id, x, y = region_id_with_point
        return self._regions[region_id].ppm.point_to_rgb_values_label((x, y), shape=shape)

    def reconstruct_predicted_rgb(self, region_ids, rgbs, ppm_xy_coordinates):
        assert len(region_ids) == len(rgbs) == len(ppm_xy_coordinates)
        for region_id, rgb, ppm_xy in zip(
                region_ids,
                rgbs,
                ppm_xy_coordinates):
            self._regions[region_id].ppm.reconstruct_predicted_rgb(
                rgb,
                ppm_xy
            )

    def reconstruct_predicted_ink_classes(self, region_ids, probabilities, ppm_xy_coordinates):
        """Save a predicted ink classes label in the internal prediction image.

        Inputs are lists so that multiple can be processed at once.

        """
        assert len(region_ids) == len(probabilities) == len(ppm_xy_coordinates)
        for region_id, class_probabilities, ppm_xy in zip(
                region_ids,
                probabilities,
                ppm_xy_coordinates):
            self._regions[region_id].ppm.reconstruct_predicted_ink_classes(
                class_probabilities,
                ppm_xy
            )

    def save_predictions(self, directory, suffix):
        """Save all predictions to files, with suffix in the filename."""
        for region in self._regions:
            region.ppm.save_predictions(directory, suffix)

    def reset_predictions(self):
        """Reset all predictions."""
        for region in self._regions:
            region.ppm.reset_predictions()

    @property
    def region_groups(self):
        return self._region_groups
