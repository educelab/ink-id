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
                ppm = self.create_ppm_if_needed(
                    region_data['ppm'],
                    data['ppms'][region_data['ppm']]
                )
                bounds = region_data.get('bounds')
                region = inkid.data.Region(len(self._regions), ppm, bounds)

                self._region_groups[region_group].append(len(self._regions))
                self._regions.append(region)

    @classmethod
    def from_json(cls, filename):
        """Initialize a RegionSet from a JSON filename."""
        data = RegionSet.get_data_from_file(filename)
        return cls(data)

    @staticmethod
    def get_data_from_file(filename):
        _, file_extension = os.path.splitext(filename)
        file_extension = file_extension.lower()
        # If it's a PPM, just make one region covering the entire image
        if file_extension == '.ppm':
            return {
                "ppms": {
                    "ppm": {
                        "path": filename,
                        "volume": "",
                    },
                },
                "regions": {
                    "training": [
                        { "ppm": "ppm" },
                    ],
                    "evaluation": [],
                    "prediction": []
                },
            }
        # If it's a JSON file then actually read the JSON region set
        elif file_extension == '.json':
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
                if type(data['ppms'][ppm][key]) == str:
                    data['ppms'][ppm][key] = os.path.normpath(
                        os.path.join(
                            paths_were_relative_to,
                            data['ppms'][ppm][key]
                        )
                    )
        return data

    def normalize_volumes(self):
        for volume_path in self._volumes:
            self._volumes[volume_path].normalize()

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
            )

        return self._ppms[ppm_name]

    def create_tf_input_fn(self, region_groups, batch_size,
                           features_fn, label_fn=None, epochs=None,
                           max_samples=-1, perform_shuffle=None,
                           shuffle_seed=None,
                           restrict_to_surface=None,
                           grid_spacing=None,
                           probability_of_selection=None,
                           premade_points_generator=None,
                           threads=multiprocessing.cpu_count(),
                           skip_batches=None):
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
            # It is also possible to create a Dataset using
            # .from_tensor_slices(), to which you can pass a
            # np.array() of the already calculated set of points. For
            # some reason this saves a lot of information in the
            # graph.pbtxt file and makes the graph file impossibly
            # large for Tensorboard to parse it and show you
            # anything. Using .from_generator() instead does not do
            # this. Practically they are the same.
            if premade_points_generator is None:
                dataset = tf.data.Dataset.from_generator(
                    self.get_points_generator(
                        region_groups=region_groups,
                        restrict_to_surface=restrict_to_surface,
                        perform_shuffle=perform_shuffle,
                        shuffle_seed=shuffle_seed,
                        grid_spacing=grid_spacing,
                        probability_of_selection=probability_of_selection,
                    ),
                    (tf.int64),
                )
            else:
                dataset = tf.data.Dataset.from_generator(
                    premade_points_generator,
                    (tf.int64)
                )

            dataset = dataset.map(
                self.create_point_to_network_input_function(
                    features_fn=features_fn,
                    label_fn=label_fn,
                ),
                num_parallel_calls=threads
            )

            # Filter out inputs that are all 0
            if label_fn is None:
                dataset = dataset.filter(
                    lambda tensors: tf.not_equal(tf.reduce_sum(tf.abs(tensors['Input'])), 0)
                )
            else:
                dataset = dataset.filter(
                    lambda tensors, _: tf.not_equal(tf.reduce_sum(tf.abs(tensors['Input'])), 0)
                )

            if epochs is not None:
                dataset = dataset.repeat(epochs)

            dataset = dataset.take(max_samples)
            dataset = dataset.batch(batch_size)

            if skip_batches is not None:
                dataset = dataset.skip(skip_batches)

            dataset = dataset.prefetch(1)

            if label_fn is None:
                batch_features = dataset.make_one_shot_iterator().get_next()
                return batch_features, None
            else:
                batch_features, batch_labels = dataset.make_one_shot_iterator().get_next()
                return batch_features, batch_labels

        return tf_input_fn

    def get_points_generator(self, region_groups,
                             restrict_to_surface=False,
                             perform_shuffle=False, shuffle_seed=None,
                             grid_spacing=None,
                             probability_of_selection=None):
        """Return a numpy array of region_ids and points.

        Used as the initial input to a Dataset, which will later map
        these points to network inputs and do other dataset processing
        such as batching.

        """
        if type(region_groups) == str:
            region_groups = [region_groups]
        print('Fetching points for region groups: {}... '
              .format(region_groups), end='')
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
            print('Shuffling points for region groups: {}... '
                  .format(region_groups), end='')
            sys.stdout.flush()
            if shuffle_seed is None:
                np.random.seed()
            else:
                np.random.seed(shuffle_seed)
            # Tensorflow Dataset objects also have a .shuffle() method
            # which would be a more natural fit, but it is much slower
            # in practice.
            np.random.shuffle(points)
            print('done')

        def generator():
            for point in points:
                yield point
        return generator

    def point_to_descriptive_statistics(self, region_id_with_point,
                                        subvolume_shape):
        region_id, x, y = region_id_with_point
        subvolume = self._regions[region_id].ppm.point_to_subvolume(
            (x, y),
            subvolume_shape,
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
                                 subvolume_shape, out_of_bounds=None,
                                 move_along_normal=None,
                                 jitter_max=None,
                                 augment_subvolume=None, method=None,
                                 normalize=None, pad_to_shape=None):
        """Take a region_id and (x, y) point, and return a subvolume.

        First use the PPM (x, y) to find the 3D position and normal
        orientation of this point in the Volume. Then get a subvolume
        from the Volume at that position.

        """
        region_id, x, y = region_id_with_point
        subvolume = self._regions[region_id].ppm.point_to_subvolume(
            (x, y),
            subvolume_shape,
            out_of_bounds=out_of_bounds,
            move_along_normal=move_along_normal,
            jitter_max=jitter_max,
            augment_subvolume=augment_subvolume,
            method=method,
            normalize=normalize,
            pad_to_shape=pad_to_shape,
        )
        return np.asarray(subvolume, np.float32)

    def point_to_ink_classes_label(self, region_id_with_point):
        """Take a region_id and point, and return the ink classes label."""
        region_id, x, y = region_id_with_point
        return self._regions[region_id].ppm.point_to_ink_classes_label((x, y))

    def point_to_rgb_values_label(self, region_id_with_point):
        region_id, x, y = region_id_with_point
        return self._regions[region_id].ppm.point_to_rgb_values_label((x, y))

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

    def reconstruct_prediction_values(self, region_ids, values, ppm_xy_coordinates):
        for region_id, value, ppm_xy in zip(
                region_ids,
                values,
                ppm_xy_coordinates):
            self._regions[region_id].ppm.reconstruct_prediction_value(
                value,
                ppm_xy
            )

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

    def save_predictions(self, directory, iteration):
        """Save all predictions to files, with iteration in the filename."""
        for region in self._regions:
            region.ppm.save_predictions(directory, iteration)

    def reset_predictions(self):
        """Reset all predictions."""
        for region in self._regions:
            region.ppm.reset_predictions()
