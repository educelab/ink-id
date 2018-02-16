import numpy as np
import tensorflow as tf
import data


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
            self.volume_set.append(data.Volume(args, volume_number=i))
            if current_vol_args['use_in_training']:
                self.n_train_volumes += 1
                self.train_volume_indices.append(i)
            if current_vol_args['make_prediction']:
                self.n_predict_volumes += 1
                self.predict_volume_indices.append(i)
            if current_vol_args['use_in_test_set']:
                self.n_test_volumes += 1
                self.test_volume_indices.append(i)

        self.current_prediction_volume = self.predict_volume_indices[0]

        print("Initialized {} total volumes, {} to be used for training...".format(self.n_total_volumes, self.n_train_volumes))



    def getTrainingBatch(self, args):
        # gather training samples from other volumes
        # should be as simple as getting batches from each volume and combining them
        trainingSamples = np.zeros((args["batch_size"], args["x_dimension"], args["y_dimension"], args["z_dimension"]), dtype=np.float32)
        groundTruth = np.zeros((args["batch_size"], args["n_classes"]), dtype=np.float32)
        samples_per_volume = int(args["batch_size"] / self.n_train_volumes)
        for i in range(self.n_train_volumes):
            volume_index = self.train_volume_indices[i]
            start = i*samples_per_volume
            end = (i+1)*samples_per_volume
            if i == self.n_train_volumes-1: # make sure to 'fill up' all the slots
                end = args["batch_size"]

            volume_samples, volume_truth, volume_epoch = self.volume_set[volume_index].getTrainingBatch(args, n_samples=(end-start))
            trainingSamples[start:end] = volume_samples
            groundTruth[start:end] = volume_truth

        '''
        for j in range(args["batch_size"]):
            # randomly swap samples
            # pretty sure this is statistically invalid because certain items are more likely to get swapped
            index_a, index_b = np.random.choice(args["batch_size"], 2, replace=False)
            tmpSample = trainingSamples[index_a]
            tmpTruth = groundTruth[index_a]
            trainingSamples[index_a] = trainingSamples[index_b]
            trainingSamples[index_b] = tmpSample
            groundTruth[index_a] = groundTruth[index_b]
            groundTruth[index_b] = tmpTruth'''

        #TODO return actual epoch instead of the last volume's epoch
        return trainingSamples, groundTruth, volume_epoch



    def getTestBatch(self, args):
        # gather testing samples from other volumes
        trainingSamples = np.zeros((args["num_test_cubes"], args["x_dimension"], args["y_dimension"], args["z_dimension"]), dtype=np.float32)
        groundTruth = np.zeros((args["num_test_cubes"], args["n_classes"]), dtype=np.float32)

        if self.n_test_volumes == 1:
            # all samples come from the same volume
            volume_samples, volume_truth = self.volume_set[self.test_volume_indices[0]].getTestBatch(args, args["num_test_cubes"])
            trainingSamples[:] = volume_samples
            groundTruth[:] = volume_truth

        else: #TODO validate this scheme
            samples_per_volume = int(args["num_test_cubes"] / self.n_test_volumes)
            for i in range(self.n_test_volumes):
                volume_index = self.test_volume_indices[i]

                start = i*samples_per_volume
                end = (i+1)*samples_per_volume
                if i == self.n_test_volumes-1: # make sure to 'fill up' all the slots
                    end = args["num_test_cubes"]

                volume_samples, volume_truth = self.volume_set[volume_index].getTestBatch(args, end-start)
                trainingSamples[start:end] = volume_samples
                groundTruth[start:end] = volume_truth

        return trainingSamples, groundTruth



    def getPredictionBatch(self, args, starting_coordinates):
        # predict on one volume at a time
        # batches should be from one volume at a time
        v_olap = args["volumes"][self.current_prediction_volume]["prediction_overlap_step"]
        self.current_prediction_total_batches = int(self.volume_set[self.current_prediction_volume].totalPredictions(args, v_olap) / args["prediction_batch_size"])

        if self.current_prediction_batch < self.current_prediction_total_batches:
            # case 1: stay in the current volume
            samples, coordinates, nextCoordinates = self.volume_set[self.current_prediction_volume].getPredictionSample3D(args, starting_coordinates, overlap_step=v_olap)
            self.current_prediction_batch += 1
            if self.current_prediction_batch % int(self.current_prediction_total_batches / 10) == 0:
                print("\tPredicting batch {}/{}...".format(self.current_prediction_batch, self.current_prediction_total_batches))

        elif self.current_prediction_volume is not self.predict_volume_indices[-1]:
            # case 2: finished one volume, and it was not the last volume to predict on
            print("Finished predictions on volume {}...".format(self.current_prediction_volume))
            self.current_prediction_volume += 1
            while not args["volumes"][self.current_prediction_volume]['make_prediction']:
                self.current_prediction_volume += 1
            v_olap = args["volumes"][self.current_prediction_volume]["prediction_overlap_step"]
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
