import json
import multiprocessing
import os
import sys

from jsmin import jsmin
import numpy as np
import tensorflow as tf

import inkid


class RegionSet:
    def __init__(self, data):
        self._regions = []
        self._ppms = {}
        self._volumes = {}

        self._region_groups = {}

        for region_group in data['regions']:
            self._region_groups[region_group] = []
            
            for region_data in data['regions'][region_group]:
                ppm = self.create_ppm_if_needed(region_data['ppm'], data['ppms'][region_data['ppm']])
                bounds = region_data.get('bounds')
                region = inkid.data.Region(len(self._regions), ppm, bounds)
                    
                self._region_groups[region_group].append(len(self._regions))
                self._regions.append(region)
                    
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
            ink_label_path = ppm_data.get('ink_label')

            if volume_path not in self._volumes:
                self._volumes[volume_path] = inkid.data.Volume(volume_path)

            volume = self._volumes[volume_path]
            self._ppms[ppm_name] = inkid.data.PPM(ppm_path, volume, mask_path, ink_label_path)

        return self._ppms[ppm_name]

    def create_tf_input_fn(self, region_groups, batch_size, features_fn,
                    label_fn=None, epochs=1, max_samples=-1,
                    perform_shuffle=None, restrict_to_surface=None,
                    grid_spacing=None, probability_of_selection=None):
        """Generate Tensorflow input_fn function for the model/network.

        A Tensorflow Estimator requires an input_fn to be passed to
        any call such as .train(), .evaluate() or .predict(). The
        input_fn should return Tensorflow Dataset iterators over the
        batch features and labels.

        The user can define their own functions features_fn and
        label_fn, each of which takes as input a region_id and (x, y)
        point in that region - and then returns either the network
        input feature, or the expected label.

        This function then takes those two functions and some
        parameters for the Dataset, then builds and returns a function
        to be used as the input_fn for the Tensorflow Estimator.

        """
        def tf_input_fn():
            dataset = tf.data.Dataset.from_tensor_slices(
                self.get_points(
                    region_groups=region_groups,
                    restrict_to_surface=restrict_to_surface,
                    perform_shuffle=perform_shuffle,
                    grid_spacing=grid_spacing,
                    probability_of_selection=probability_of_selection,
                )
            )

            dataset = dataset.map(
                self.create_point_to_network_input_function(
                    features_fn=features_fn,
                    label_fn=label_fn,
                ),
                num_parallel_calls=multiprocessing.cpu_count()
            )

            dataset = dataset.repeat(epochs)
            dataset = dataset.take(max_samples)
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(1)

            if label_fn is None:
                batch_features = dataset.make_one_shot_iterator().get_next()
                return batch_features, None
            else:
                batch_features, batch_labels = dataset.make_one_shot_iterator().get_next()
                return batch_features, batch_labels
        
        return tf_input_fn
    
    def get_points(self, region_groups, restrict_to_surface=False,
                     perform_shuffle=False,
                     grid_spacing=None,
                     probability_of_selection=None):
        """Return a numpy array of region_ids and points.

        Used as the initial input to a Dataset, which will later map
        these points to network inputs and do other dataset processing
        such as batching.

        """
        print('Fetching points for region groups: {}... '.format(region_groups), end='')
        sys.stdout.flush()
        points = []
        for region_group in region_groups:
            for region_id in self._region_groups[region_group]:
                points += self._regions[region_id].get_points(
                    restrict_to_surface=restrict_to_surface,
                    grid_spacing=grid_spacing,
                    probability_of_selection=probability_of_selection
                )
        points = np.array(points)
        print('done ({} points)'.format(len(points)))
        if perform_shuffle:
            print('Shuffling points for region groups: {}... '.format(region_groups), end='')
            sys.stdout.flush()
            # Tensorflow Dataset objects also have a .shuffle() method
            # which would be a more natural fit, but it is much slower
            # in practice.
            np.random.shuffle(points)
            print('done')
        return points

    def point_to_subvolume_input(self, region_id_with_point, subvolume_shape):
        """Take a region_id and (x, y) point, and return a subvolume.

        First use the PPM (x, y) to find the 3D position and normal
        orientation of this point in the Volume. Then get a subvolume
        from the Volume at that position.

        Return None if the subvolume is not bounded in the original
        volume.

        """
        region_id, x, y = region_id_with_point
        subvolume = self._regions[region_id].point_to_subvolume((x, y), subvolume_shape)
        return np.asarray(subvolume, np.float32)
    
    def point_to_ink_classes_label(self, region_id_with_point):
        """Take a region_id and point, and return the ink classes label."""
        region_id, x, y = region_id_with_point
        return self._regions[region_id].point_to_ink_classes_label((x, y))

    def point_to_other_feature_tensors(self, region_id_with_point):
        """Take a region_id and point, and return some general network inputs.

        Sometimes it is useful to pass some information to the network
        that is not used as a feature in the actual feedforward
        processing. For example, when using subvolumes, we can pass
        the 3D position and orientation to the network, not so that
        they can be used as features, but so that we can request them
        back out along with the ink prediction. This is helpful to
        create predictions that we have other information about beyond
        just their expected and actual values.

        This function generates the region_id and PPM (x, y) position
        as values that can be passed to the model function. Elsewhere,
        the program will add to these 1) the actual feature input, 2)
        a label (if training or evaluating) before passing the full
        input into the network/model.

        This is done separately from the feature input and any labels
        because the other features and labels are variable depending
        on what inputs and outputs the user has configured the network
        for.

        """
        region_id, x, y = region_id_with_point
        return (region_id, np.asarray((x, y), np.int64))

    def create_point_to_network_input_function(self, features_fn, label_fn):
        """Build (point -> network input) mapping function.

        The user can define their own functions features_fn and
        label_fn, each of which takes as input a region_id and (x, y)
        point in that region - and then returns either the network
        input feature, or the expected label.

        This function then takes those two functions, then builds and
        returns a function based on them that will take a point as
        input and will return the network inputs and labels (if there
        are any).

        The returned function is used to map the Tensorflow Dataset
        from a set of points in regions to a set of full network
        inputs with features and labels.

        """
        def point_to_network_input(region_id_with_point):
            other_feature_names = ['RegionID', 'PPM_XY']
            other_feature_tensors = tf.py_func(self.point_to_other_feature_tensors,
                                               [region_id_with_point],
                                               [tf.int64, tf.int64])
            input_feature_name = 'Input'
            input_feature_tensor = tf.py_func(features_fn,
                                              [region_id_with_point],
                                              tf.float32)
            network_input = dict(zip(other_feature_names, other_feature_tensors))
            network_input.update({input_feature_name: input_feature_tensor})

            if label_fn is None:
                return network_input
            else:
                label = tf.py_func(label_fn,
                                   [region_id_with_point],
                                   tf.float32)
                return network_input, label
            
        return point_to_network_input

    def reconstruct_predicted_ink_classes(self, region_ids, probabilities, ppm_xy_coordinates):
        """TODO"""
        assert len(region_ids) == len(probabilities) == len(ppm_xy_coordinates)
        for region_id, class_probabilities, ppm_xy in zip(region_ids, probabilities, ppm_xy_coordinates):
            self._regions[region_id]._ppm.reconstruct_predicted_ink_classes(class_probabilities, ppm_xy)

    def save_predictions(self, directory, iteration):
        """TODO"""
        for region in self._regions:
            region._ppm.save_predictions(directory, iteration)  # TODO(srp) clean up use of ppm

    def reset_predictions(self):
        """TODO"""
        for region in self._regions:
            region._ppm.reset_predictions()  # TODO(srp) clean up use of ppm
