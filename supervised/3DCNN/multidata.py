import numpy as np
import tensorflow as tf
import data


class VolumeSet:
    def __init__(self, args):
        # instantiate other volumes
        self.volume_set = []
        self.current_prediction_volume = 0
        self.current_prediction_batch = 0
        self.current_prediction_total_batches = 0
        self.n_volumes = len(args["trainingDataPaths"])

        for i in range(self.n_volumes):
            self.volume_set.append(data.Volume(args, volume_number=i, volume_path=args["trainingDataPaths"][i], \
                truth_path=args["groundTruthFiles"][i], surface_mask_path=args["surfaceMaskFiles"][i], surface_point_path=args["surfaceDataFiles"][i]))



    def getTrainingBatch(self, args, testSet=False):
        # gather training samples from other volumes
        # should be as simple as getting batches from each volume and combining them
        trainingSamples = np.zeros((args["batch_size"], args["x_Dimension"], args["y_Dimension"], args["z_Dimension"]), dtype=np.float32)
        groundTruth = np.zeros((args["batch_size"], args["n_classes"]), dtype=np.float32)
        samples_per_volume = int(args["batch_size"] / self.n_volumes)
        for i in range(self.n_volumes):
            start = i*samples_per_volume
            end = (i+1)*samples_per_volume
            if i == self.n_volumes-1: # make sure to 'fill up' all the slots
                end = self.n_volumes-1

            volume_samples, volume_truth, volume_epoch = self.volume_set[i].getTrainingBatch(args, end-start)
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
                print("Predicting batch {}/{}...".format(self.current_prediction_batch, self.current_prediction_total_batches))

        elif current_prediction_volume + 1 != self.n_volumes:
            # case 2: finished volume, if there is another volume, go to it
            print("\nFinished predictions on volume {}...".format(current_prediction_volume))
            self.current_prediction_volume += 1
            self.current_prediction_batch = 0
            samples, coordinates, nextCoordinates = self.volume_set[self.current_prediction_volume].getPredictionSample3D(args, starting_coordinates)
            print("\nBeginning predictions on volume {}...".format(current_prediction_volume))

        else:
            # case 3: finished volume, no other volume to predict
            samples, coordinates, nextCoordinates = None, None, None

        return samples, coordinates, nextCoordinates



    def reconstruct(self, args, prediction_values, coordinates):
        # fill in the prediction images
        self.volume_set[self.current_prediction_volume].reconstruct3D(args, prediction_values, coordinates)



    def saveAllPredictions(self, args, iteration):
        # save the prediction images to a file
        for i in range(self.n_volumes):
            self.volume_set[i].savePrediction3D(args, iteration)



    def saveAllPredictionMetrics(self, args, iteration, minutes):
        # save metrics on performance
        for i in range(self.n_volumes):
            self.volume_set[i].savePredictionMetrics(args, iteration)


    def wobbleVolumes(self, args):
        for i in range(self.n_volumes):
            self.volume_set[i].wobbleVolume(args, iteration)
