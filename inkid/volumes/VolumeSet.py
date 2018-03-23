import multiprocessing

import numpy as np
import tensorflow as tf

from inkid.volumes import Volume


class VolumeSet:
    def __init__(self, args):
        # instantiate all volumes
        self.n_total_volumes = len(args["volumes"])
        self.volume_set = []
        self.train_volume_indices = []
        self.test_volume_indices = []
        self.predict_volume_indices = []
        self.current_prediction_batch = 0
        self.current_prediction_total_batches = 0
        self.n_train_volumes = 0
        self.n_predict_volumes = 0
        self.n_test_volumes = 0

        for i in range(self.n_total_volumes):
            print("Initializing volume {}...".format(i))
            current_vol_args = args['volumes'][i]
            self.volume_set.append(Volume(args, volume_ID=i))
            if current_vol_args['use_in_training']:
                self.n_train_volumes += 1
                self.train_volume_indices.append(i)
            if current_vol_args['use_in_prediction']:
                self.n_predict_volumes += 1
                self.predict_volume_indices.append(i)
            if current_vol_args['use_in_evaluation']:
                self.n_test_volumes += 1
                self.test_volume_indices.append(i)

        self.current_prediction_volume = self.predict_volume_indices[0]

        print("Initialized {} total volumes, {} to be used for training...".format(self.n_total_volumes, self.n_train_volumes))


    def make_dataset(self,
                     return_labels,
                     perform_shuffle,
                     batch_size,
                     restrict_to_surface,
                     augment_samples,
                     grid_spacing,
                     max_samples):

        # TODO make this something cleaner (this might not even work, really)
        self._augment_samples = augment_samples
        
        dataset = tf.data.Dataset.from_generator(
            self.coordinate_pool_generator(grid_spacing, perform_shuffle),
            (tf.int64, tf.int64)
        )

        if restrict_to_surface:
            dataset = dataset.filter(self.tf_is_on_surface)

        if return_labels:
            dataset = dataset.map(
                self.tf_coordinate_to_labeled_input,
                num_parallel_calls=multiprocessing.cpu_count()
            )
        else:
            dataset = dataset.map(
                self.tf_coordinate_to_unlabeled_input,
                num_parallel_calls=multiprocessing.cpu_count()
            )

        dataset = dataset.take(max_samples)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(1)
        
        return dataset


    def tf_input_fn(self,
                    return_labels=True,
                    perform_shuffle=True,
                            batch_size=32,
                    restrict_to_surface=True,
                    augment_samples=False,
                    grid_spacing=1,
                    max_samples=-1):
        """Create a tensorflow input_fn."""
        dataset = self.make_dataset(
            return_labels=return_labels,
            perform_shuffle=perform_shuffle,
            batch_size=batch_size,
            restrict_to_surface=restrict_to_surface,
            augment_samples=augment_samples,
            grid_spacing=grid_spacing,
            max_samples=max_samples,
        )
        
        if return_labels:
            batch_features, batch_labels = dataset.make_one_shot_iterator().get_next()
            return batch_features, batch_labels
        else:
            batch_features = dataset.make_one_shot_iterator().get_next()
            return batch_features, None


    def training_input_fn(self, batch_size):
        return self.tf_input_fn(
            return_labels=True,
            perform_shuffle=True,
            batch_size=batch_size,
            restrict_to_surface=True,
            augment_samples=True,
            grid_spacing=1,
        )


    def prediction_input_fn(self, batch_size):
        return self.tf_input_fn(
            return_labels=False,
            perform_shuffle=False,
            batch_size=batch_size,
            restrict_to_surface=True,
            augment_samples=False,
            grid_spacing=60,
        )


    def evaluation_input_fn(self, batch_size):
        return self.tf_input_fn(
            return_labels=True,
            perform_shuffle=True,
            batch_size=batch_size,
            restrict_to_surface=True,
            augment_samples=False,
            grid_spacing=10,
            max_samples=5000,
        )


    def tf_coordinate_to_unlabeled_input(self, vol_id, xy_coordinate):
        tensors = tf.py_func(self.coordinate_to_input,
                             [vol_id, xy_coordinate, False, self._augment_samples],
                             [tf.int64, tf.int64, tf.float32])
        feature_names = ['VolumeID', 'XYZCoordinate', 'Subvolume']
        return dict(zip(feature_names, tensors))


    def tf_coordinate_to_labeled_input(self, vol_id, xy_coordinate):
        tensors = tf.py_func(self.coordinate_to_input,
                             [vol_id, xy_coordinate, True, self._augment_samples],
                             [tf.int64, tf.int64, tf.float32, tf.float32])
        feature_names = ['VolumeID', 'XYZCoordinate', 'Subvolume']
        return dict(zip(feature_names, tensors[:3])), tensors[3]


    def coordinate_to_input(self, vol_id, xy_coordinate, return_label, augment_samples):
        return self.volume_set[vol_id].coordinate_to_input(xy_coordinate,
                                                           return_label,
                                                           augment_samples)

    
    def coordinate_pool_generator(self, grid_spacing, perform_shuffle):
        def generator():
            for volume in self.volume_set:
                for coordinate in volume.yield_coordinates(grid_spacing, perform_shuffle):
                    yield coordinate
        return generator


    def is_on_surface(self, vol_id, xy_coordinate):
        return self.volume_set[vol_id].is_on_surface(xy_coordinate)
    

    def tf_is_on_surface(self, vol_id, xy_coordinate):
        return tf.py_func(self.is_on_surface, [vol_id, xy_coordinate], [tf.bool])

        
    def getTestBatch(self, args):
        """DEPRECATED"""
        # gather testing samples from other volumes
        trainingSamples = np.zeros(
            (args["evaluation_batch_size"],
             args["subvolume_shape"][0],
             args["subvolume_shape"][1],
             args["subvolume_shape"][2]
            ),
            dtype=np.float32
        )
        groundTruth = np.zeros(
            (args["evaluation_batch_size"], 2),
            dtype=np.float32
        )

        if self.n_test_volumes == 1:
            # all samples come from the same volume
            volume_samples, volume_truth = self.volume_set[self.test_volume_indices[0]].getTestBatch(args, args["evaluation_batch_size"])
            trainingSamples[:] = volume_samples
            groundTruth[:] = volume_truth

        else: #TODO validate this scheme
            samples_per_volume = int(args["evaluation_batch_size"] / self.n_test_volumes)
            for i in range(self.n_test_volumes):
                volume_index = self.test_volume_indices[i]

                start = i*samples_per_volume
                end = (i+1)*samples_per_volume
                if i == self.n_test_volumes-1: # make sure to 'fill up' all the slots
                    end = args["evaluation_batch_size"]

                volume_samples, volume_truth = self.volume_set[volume_index].getTestBatch(args, end-start)
                trainingSamples[start:end] = volume_samples
                groundTruth[start:end] = volume_truth

        return trainingSamples, groundTruth

    def getTrainingBatch(self, args):
        """DEPRECATED"""
        # gather training samples from other volumes
        # should be as simple as getting batches from each volume and combining them
        trainingSamples = np.zeros(
            (args["training_batch_size"],
             args["subvolume_shape"][0],
             args["subvolume_shape"][1],
             args["subvolume_shape"][2]
            ),
            dtype=np.float32)
        groundTruth = np.zeros(
            (args["training_batch_size"], 2),
            dtype=np.float32
        )
        samples_per_volume = int(args["training_batch_size"] / self.n_train_volumes)
        for i in range(self.n_train_volumes):
            volume_index = self.train_volume_indices[i]
            start = i*samples_per_volume
            end = (i+1)*samples_per_volume
            if i == self.n_train_volumes-1: # make sure to 'fill up' all the slots
                end = args["training_batch_size"]

            volume_samples, volume_truth, volume_epoch = self.volume_set[volume_index].getTrainingBatch(args, n_samples=(end-start))
            trainingSamples[start:end] = volume_samples
            groundTruth[start:end] = volume_truth

        #TODO return actual epoch instead of the last volume's epoch
        return trainingSamples, groundTruth, volume_epoch



    def getPredictionBatch(self, args, starting_coordinates):
        # predict on one volume at a time
        # batches should be from one volume at a time
        self.current_prediction_total_batches = int(self.volume_set[self.current_prediction_volume].totalPredictions(args, args["prediction_overlap_step"]) / args["prediction_batch_size"])

        if self.current_prediction_batch < self.current_prediction_total_batches:
            # case 1: stay in the current volume
            samples, coordinates, nextCoordinates = self.volume_set[self.current_prediction_volume].getPredictionSample3D(args, starting_coordinates, overlap_step=args['prediction_overlap_step'])
            self.current_prediction_batch += 1
            if self.current_prediction_batch % int(self.current_prediction_total_batches / 10) == 0:
                print("\tPredicting batch {}/{}...".format(self.current_prediction_batch, self.current_prediction_total_batches))

        elif self.current_prediction_volume is not self.predict_volume_indices[-1]:
            # case 2: finished one volume, and it was not the last volume to predict on
            print("Finished predictions on volume {}...".format(self.current_prediction_volume))
            self.current_prediction_volume += 1
            while not args["volumes"][self.current_prediction_volume]['make_prediction']:
                self.current_prediction_volume += 1
            v_olap = args["prediction_overlap_step"]
            self.current_prediction_batch = 0
            starting_coordinates = [0,0,0]
            samples, coordinates, nextCoordinates = self.volume_set[self.current_prediction_volume].getPredictionSample3D(args, starting_coordinates, overlap_step=v_olap)
            print("\nBeginning predictions on volume {}...".format(self.current_prediction_volume))

        else:
            # case 3: finished volume, no other volume to predict on
            samples, coordinates, nextCoordinates = None, None, None
            self.current_prediction_volume = self.predict_volume_indices[0]
            self.current_prediction_batch = 0

        return samples, coordinates, nextCoordinates


    def reconstruct(self, args, prediction_values, coordinates):
        # fill in the prediction images
        self.volume_set[self.current_prediction_volume].reconstruct3D(args, prediction_values, coordinates)



    def saveAllPredictions(self, args, iteration):
        # save the prediction images to a file
        for i in self.predict_volume_indices:
            self.volume_set[i].savePrediction3D(args, iteration)



    def saveAllPredictionMetrics(self, args, iteration, minutes):
        # save metrics on performance
        for i in self.predict_volume_indices:
            self.volume_set[i].savePredictionMetrics(args, iteration, minutes)



    def wobbleVolumes(self, args):
        for i in self.train_volume_indices:
            self.volume_set[i].wobbleVolume(args)
