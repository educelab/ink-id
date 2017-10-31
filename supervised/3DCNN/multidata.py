import numpy as np
import tensorflow as tf
import data


class VolumeSet:
    def __init__(self, args):
        # instantiate all volumes
        self.n_total_volumes = len(args["volumes"])
        self.volume_set = []
        self.current_prediction_batch = 0
        self.current_prediction_total_batches = 0
        self.n_train_volumes = 0
        self.n_predict_volumes = 0
        self.current_prediction_volume = 0

        for i in range(self.n_total_volumes):
            print("Initializing volume {}...".format(i))
            current_vol_args = args['volumes'][i]
            self.volume_set.append(data.Volume(args, volume_number=i))
            if current_vol_args['use_in_training']:
                self.n_train_volumes += 1
            if current_vol_args['make_prediction']:
                self.n_predict_volumes += 1

        print("Initialized {} total volumes, {} to be used for training...".format(self.n_total_volumes, self.n_train_volumes))



    def getTrainingBatch(self, args):
        # gather training samples from other volumes
        # should be as simple as getting batches from each volume and combining them
        trainingSamples = np.zeros((args["batch_size"], args["x_dimension"], args["y_dimension"], args["z_dimension"]), dtype=np.float32)
        groundTruth = np.zeros((args["batch_size"], args["n_classes"]), dtype=np.float32)
        samples_per_volume = int(args["batch_size"] / self.n_train_volumes)
        for i in range(self.n_train_volumes):
            start = i*samples_per_volume
            end = (i+1)*samples_per_volume
            if i == self.n_train_volumes-1: # make sure to 'fill up' all the slots
                end = args["batch_size"]

            volume_samples, volume_truth, volume_epoch = self.volume_set[i].getTrainingBatch(args, n_samples=(end-start))
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

        #TODO return actual epoch instead of 0
        return trainingSamples, groundTruth, 0



    def getTestBatch(self, args):
        # gather testing samples from other volumes
        trainingSamples = np.zeros((args["num_test_cubes"], args["x_dimension"], args["y_dimension"], args["z_dimension"]), dtype=np.float32)
        groundTruth = np.zeros((args["num_test_cubes"], args["n_classes"]), dtype=np.float32)

        if self.n_predict_volumes == 1:
            # all samples come from the same volume
            volume_samples, volume_truth = self.volume_set[-1].getTestBatch(args, args["num_test_cubes"])
            trainingSamples[:] = volume_samples
            groundTruth[:] = volume_truth

        else: #TODO validate this scheme
            samples_per_volume = int(args["num_test_cubes"] / self.n_predict_volumes)
            for i in range(self.n_predict_volumes):
                start = i*samples_per_volume
                end = (i+1)*samples_per_volume
                if i == self.n_predict_volumes-1: # make sure to 'fill up' all the slots
                    end = args["num_test_cubes"]-1

                volume_samples, volume_truth = self.volume_set[i].getTestBatch(args, end-start)
                trainingSamples[start:end] = volume_samples
                groundTruth[start:end] = volume_truth

        return trainingSamples, groundTruth



    def getPredictionBatch(self, args, starting_coordinates):
        # predict on one volume at a time
        # batches should be from one volume at a time
        self.current_prediction_total_batches = int(self.volume_set[self.current_prediction_volume].totalPredictions(args) / args["prediction_batch_size"])

        if self.current_prediction_batch < self.current_prediction_total_batches:
            # case 1: stay in the current volume
            samples, coordinates, nextCoordinates = self.volume_set[self.current_prediction_volume].getPredictionSample3D(args, starting_coordinates)
            self.current_prediction_batch += 1
            if self.current_prediction_batch % int(self.current_prediction_total_batches / 10) == 0:
                print("\tPredicting batch {}/{}...".format(self.current_prediction_batch, self.current_prediction_total_batches))

        elif self.current_prediction_volume + 1 < self.n_total_volumes:
            # case 2: finished volume, if there is another volume, go to it
            print("Finished predictions on volume {}...".format(self.current_prediction_volume))
            self.current_prediction_volume += 1
            self.current_prediction_batch = 0
            starting_coordinates = [0,0,0]
            samples, coordinates, nextCoordinates = self.volume_set[self.current_prediction_volume].getPredictionSample3D(args, starting_coordinates)
            print("\nBeginning predictions on volume {}...".format(self.current_prediction_volume))

        else:
            # case 3: finished volume, no other volume to predict
            samples, coordinates, nextCoordinates = None, None, None
            self.current_prediction_volume = self.n_total_volumes - self.n_predict_volumes
            self.current_prediction_batch = 0

        return samples, coordinates, nextCoordinates



    def reconstruct(self, args, prediction_values, coordinates):
        # fill in the prediction images
        self.volume_set[self.current_prediction_volume].reconstruct3D(args, prediction_values, coordinates)



    def saveAllPredictions(self, args, iteration):
        # save the prediction images to a file
        for i in range(self.n_total_volumes):
            self.volume_set[i].savePrediction3D(args, iteration)



    def saveAllPredictionMetrics(self, args, iteration, minutes):
        # save metrics on performance
        for i in range(self.n_total_volumes):
            self.volume_set[i].savePredictionMetrics(args, iteration, minutes)



    def wobbleVolumes(self, args):
        for i in range(self.n_train_volumes):
            self.volume_set[i].wobbleVolume(args, iteration)
