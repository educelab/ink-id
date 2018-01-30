'''
vcdata.py
A refactored version of data.py
Designed to use volcart for underlying data
'''
#TODO don't pass args in every method, just keep it stored in the instance


from volcart import Core
import ops


class Volume:
    def __init__(self, args, volume_number, volume_path):
        # Part 1: volume metadataf
        self.volume_args = args['volumes'][volume_number]
        self.volume_number = volume_number
        self.train_bounds = self.volume_args['train_bounds']
        self.train_portion = self.volume_args['train_portion']

        # Part 2 the volume
        self.vc_volpkg = Core.VolumePkg(volume_path)
        self.vc_vol = self.vc_volpkg.volume()
        self.volume_shape = [vc_vol.height(), vc_vol.width(), vc_vol.slices()]
        # shape of the output data needs to be the shape of the "flattened" volume
        # how to do that?

        # Part 3: prediction data
        self.prediction_volume = np.zeros((self.volume_shape[0], self_volume.shape[1], args["predict_depth"]), dtype=np.float32)

        # Part 4: prediction metadata, etc
        self.coordinate_pool = []
        self.epoch = 0
        self.train_index = 0



    def getTrainingBatch(self, args, n_samples):
        """Retrieve samples on the training side of the volume

        Returns:
            An array of length n_samples
            A corresponding array of ground truth
        """

        if len(self.coordinate_pool) == 0: #initialize training coordinates
            row_bounds, col_bounds = ops.bounds(args, [self.volume_shape[0], self.volume_shape[1]],\
                identifier=self.train_bounds, train_portion=self.train_portion)
            self.coordinate_pool = ops.generateCoordinatePoolVC(args, row_bounds, col_bounds)

        coordinates_to_use = self.coordinate_pool[self.train_index:self.train_index+n_samples]
        training_samples, ground_truth = self.getSamplesAtCoordinates(args, n_samples)
        self.train_index += n_samples

        return training_samples, ground_truth



    def getTestingBatch(self, args, n_samples):
        """Retrieve samples on the NON-training side of the volume

        Returns:
            An array of length n_samples
            A corresponding array of ground truth

        """
        pass



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

        n = len(coordinates)
        xr = int(args["x_dimension"] / 2)
        yr = int(args["y_dimension"] / 2)
        zr = int(args["z_dimension"] / 2)

        samples = np.zeros((n_samples, args["x_dimension"], args["y_dimension"], args["z_dimension"]), dtype=np.float32)
        ground_truth = np.zeros((n_samples, args["n_classes"]), dtype=np.float32)

        for i in range(n):
            row_coord, col_coord = coordinates[i]
            z_coord = None # should be from surface image
            ctr_pt = (row_coord, col_coord, z_coord)
            sample = self.vc_vol.subvolume(center=ctr_pt, x_rad=xr, y_rad = yr, z_rad = zr)
            samples[i] = sample
            ground_truth[i] = None

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
