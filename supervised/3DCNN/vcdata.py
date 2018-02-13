'''
vcdata.py
A refactored version of data.py
Designed to use volcart for underlying data
'''
#TODO don't pass args in every method, just keep it stored in the instance


import numpy as np
from imageio import imread, imwrite
from volcart import Core
import ops


class Volume:
    def __init__(self, args, volume_number):
        # Part 1: volume metadata
        self.volume_args = args['volumes'][volume_number]
        self.volume_number = volume_number
        self.volume_path = self.volume_args['data_path']
        self.train_bounds = self.volume_args['train_bounds']
        self.train_portion = self.volume_args['train_portion']

        # Part 2 the volume
        self.vc_volpkg = Core.VolumePkg(self.volume_path)
        self.vc_vol = self.vc_volpkg.volume()
        self.vc_vol.setCacheMemory(self.volume_args['cache_bytes'])
        self.volume_shape = [self.vc_vol.height(), self.vc_vol.width(), self.vc_vol.slices()]
        print("Shape of {}: {}".format(self.volume_args['name'], self.volume_shape))
        # eventually, shape of the output data needs to be the shape of the "flattened" volume
        if len(self.volume_args['surface_data']) > 0:
            self.surface_image = imread(self.volume_args['surface_data'])
        if len(self.volume_args['ground_truth']) > 0:
            self.ground_truth = imread(self.volume_args['ground_truth'])
        # Part 3: prediction data
        self.prediction_volume = np.zeros((self.volume_shape[0], self.volume_shape[1], args["predict_depth"]), dtype=np.float32)

        # Part 4: prediction metadata, etc
        self.coordinate_pool = []
        self.epoch = 0
        self.train_index = 0



    def getTrainingBatch(self, args, n_samples):
        """Retrieve samples on the training side of the volume

        Returns:
            An array of length n_samples
            A corresponding array of ground truth
            The current epoch
        """

        if len(self.coordinate_pool) == 0: #initialize training coordinates
            self.coordinate_pool = ops.generateCoordinatePoolVC(
                    args, self.volume_shape, self.volume_args["train_bounds"], self.volume_args["train_portion"])

        coordinates_to_use = self.coordinate_pool[self.train_index:self.train_index+n_samples]
        training_samples, ground_truth = self.getSamplesAtCoordinates(args, coordinates_to_use)
        self.train_index += n_samples

        return training_samples, ground_truth, self.epoch



    def getTestBatch(self, args, n_samples):
        """Retrieve samples on the NON-training side of the volume

        Returns:
            An array of length n_samples
            A corresponding array of ground truth

        """
        coordinates = []
        for i in range(n_samples):
            row_coordinate, col_coordinate = ops.getRandomTestCoordinate(
                args, self.volume_shape, self.volume_args["train_bounds"], self.volume_args["train_portion"])
            coordinates.append([row_coordinate, col_coordinate])

        test_samples, ground_truth = self.getSamplesAtCoordinates(args, coordinates)

        return test_samples, ground_truth



    def getPredictionBatch(self, args, n_samples):
        """Retrieve samples for prediction

        Returns:
            An array of length n_samples
            A corresponding array of the coordinates of those samples
            The coordinate of the next sample to be used

        """
        pass



    def getSamplesAtCoordinates(self, args, coordinates):
        """Generic method to retrieve samples from a list of coordinates
        Args:
            coordinates: the centerpoints of the samples

        Returns:
            An array of samples
            A corresponding array of ground truth

        """

        n_samples = len(coordinates)
        xr = int(args["x_dimension"] / 2)
        yr = int(args["y_dimension"] / 2)
        zr = int(args["z_dimension"] / 2)

        samples = np.zeros((n_samples, args["x_dimension"], args["y_dimension"], args["z_dimension"]), dtype=np.float32)
        ground_truth = np.zeros((n_samples, args["n_classes"]), dtype=np.float32)
        for i in range(n_samples):
            row_coord, col_coord = coordinates[i]
            z_coord = max(24, min(self.volume_shape[2]-24, self.surface_image[row_coord, col_coord] + 24))
            ctr_pt = (col_coord, row_coord, z_coord)
            sample = self.vc_vol.subvolume(center=ctr_pt, x_rad=xr, y_rad = yr, z_rad = zr)
            oriented_sample = np.swapaxes(sample, 0, 2)
            samples[i] = oriented_sample 
            ground_truth[i] = ops.averageTruthInSubvolume(args, row_coord, col_coord, self.ground_truth)
        return samples, ground_truth



    def incrementEpoch(self, args):
        print("finished epoch")
        self.train_index = 0
        self.training_image = np.zeros(self.training_image.shape, dtype=np.uint16)
        self.epoch += 1
        np.random.shuffle(self.coordinate_pool)



    def reconstruct(self, args, prediction_values, coordinates):
        """Fill in the output volume with values

        Args:
            prediction_values: the predictions made
            coordinates: the corresponding location of those predictions
        """
        pass



    def saveOutput(self, args, iteration, minutes):
        self.saveMetrics(args, iteration, minutes)
        self.savePrediction(args, iteration, minutes)



    def saveMetrics(self, args, iteration, minutes):
        pass



    def savePrediction(self, args, iteration, minutes):
        pass
