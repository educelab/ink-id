import datetime
import inspect
import os
import math
import shutil
import time

import imageio
import numpy as np
from PIL import Image
# from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score
import tensorflow as tf

from inkid import ops
import inkid.model


class Volume:
    def __init__(self, args, volume_ID):
        """Read volume(s), ground truth (single registered .png), and surface segmentation image."""
        # Part 1: volume metadata
        self.args = args
        self.volume_args = args['volumes'][volume_ID]
        self.volume_ID = volume_ID
        self.train_bounds = self.volume_args['train_bounds']
        self.train_portion = self.volume_args['train_portion']
        self.add_augmentation = args["add_augmentation"]
        
        # Part 2: volume data
        data_files = os.listdir(self.volume_args['data_path'])
        data_files.sort()
        volume = []
        for f in data_files:
            slice_data = np.array(Image.open(self.volume_args['data_path']+f))
            volume.append(slice_data)
        self.volume = np.array(volume)
        if args["wobble_volume"]:
            self.wobbled_axes = []
            self.wobbled_angle = 0.0
            self.wobbled_volume = np.array(volume)
        print("{} has shape: {}".format(self.volume_args['name'], self.volume.shape))

        if len(self.volume_args['ground_truth']) > 0:
            self.ground_truth = imageio.imread(self.volume_args['ground_truth'])
        elif 'inked' in self.volume_args['data_path']:
            self.ground_truth = np.ones((self.volume.shape[0:2]), dtype=np.uint16) * 65535
        elif 'blank' in self.volume_args['data_path']:
            self.ground_truth = np.zeros((self.volume.shape[0:2]), dtype=np.uint16)
        else:
            self.ground_truth = np.ones((self.volume.shape[0:2]), dtype=np.uint16) * 65535

        if len(self.volume_args['surface_data']) > 0:
            self.surface_image = imageio.imread(self.volume_args['surface_data'])
        elif self.volume.shape[2] > args["subvolume_shape"][2]:
            print("Approximating surface for {}...".format(self.volume_args['name']))
            surf_start_time = time.time()
            self.surface_image = ops.generateSurfaceApproximation(args, self.volume)
            print("Surface approximation took {:.2f} minutes.".format((time.time()-surf_start_time)/60))
        else:
            self.surface_image = np.zeros((self.volume.shape[0:2]))

        self.surface_mask_image = np.ones((self.volume.shape[0], self.volume.shape[1]), dtype=np.int)
        if len(self.volume_args['surface_mask']) > 0:
            self.surface_mask_image = imageio.imread(self.volume_args['surface_mask'])

        # Part 3: prediction data
        self.prediction_volume = np.zeros((self.volume.shape[0], self.volume.shape[1], args["predict_depth"]), dtype=np.float32)
        self.prediction_image_ink = np.zeros((self.volume.shape[0:2]), dtype=np.float32)
        self.prediction_image_surf = np.zeros((self.volume.shape[0:2]), dtype=np.float32)
        self.prediction_plus_surf = np.zeros((self.volume.shape[0:2]), dtype=np.float32)
        self.training_image = np.zeros(self.prediction_image_ink.shape, dtype=np.uint16)
        try:
            self.prediction_overlap_step = self.args["prediction_overlap_step"]
        except:
            print("No prediction_overlap_step for {}".format(self.volume_args['name']))

        # Part 4: prediction metadata and more
        self.truth_cutoff_high = self.args["truth_cutoff_high"]
        self.max_truth = np.iinfo(self.ground_truth.dtype).max
        self.surface_mask = self.surface_mask_image / np.iinfo(self.surface_mask_image.dtype).max
        self.all_truth, self.all_preds = [], []
        self.test_truth, self.test_preds = [], []
        self.test_results, self.test_results_norm = [], []
        self.all_results, self.all_results_norm = [], []
        self.output_path = args["output_path"]+"/{}/".format(self.volume_args['name'])
        self.coordinate_pool = []
        self.train_index = 0
        self.epoch = 0

        
    def is_on_surface(self, xy_coordinate):
        """Return whether a point is on the surface."""
        x = xy_coordinate[0]
        y = xy_coordinate[1]
        xStep = int(self.args["subvolume_shape"][0] / 2)
        yStep = int(self.args["subvolume_shape"][1] / 2)
        square = self.surface_mask[(x - xStep):(x + xStep), (y - yStep):(y + yStep)]
        return np.size(square) > 0 and np.min(square) != 0


    def yield_coordinates(self, grid_spacing=1, perform_shuffle=True):
        """Walk the 2D space and yield points on the grid.

        Walk the 2D coordinate space and yield the (x, y) points on
        the grid. Skip points in both the x and y directions
        (effectively creating a grid of points) based on the
        grid_spacing argument. If grid_spacing is 1 it will just
        return every (x, y). Return the volume ID along with the
        coordinate values.

        """
        coordinates = []
        for x in range(
                self.args["subvolume_shape"][0],
                self.volume.shape[0] - self.args["subvolume_shape"][0],
                grid_spacing):
            for y in range(
                    self.args["subvolume_shape"][1],
                    self.volume.shape[1] - self.args["subvolume_shape"][1],
                    grid_spacing):
                coordinates.append([x, y])

        coordinates = np.array(coordinates)

        if perform_shuffle:
            np.random.shuffle(coordinates)

        for coordinate in coordinates:
            yield (self.volume_ID, (coordinate))


    def coordinate_to_input(self, xy_coordinate, return_label, augment_samples):
        """Map a coordinate to a tuple of (coordinate, subvolume).

        Given an (x, y) coordinate, return a tuple (coordinate,
        subvolume) of the input coordinate and the subvolume at that
        point, using the surface data to find the z value. If for some
        reason the shape of the requested subvolume is not as
        expected, return None.

        If asked, "augment" the sample by flipping it and rotating it
        randomly.

        """
        # Assume coordinate is in center of subvolume in x and y, and
        # on the top of the subvolume in the z axis.
        x_step = int(self.args["subvolume_shape"][0] / 2)
        y_step = int(self.args["subvolume_shape"][1] / 2)
        z_step = self.args["subvolume_shape"][2]

        x = xy_coordinate[0]
        y = xy_coordinate[1]
        # Cap the z value in case the surface data has a z height that
        # puts the subvolume outside the volume
        z = min(
            max(
                0,
                self.surface_image[x, y] - self.args["surface_cushion"] # + np.random.randint(-2, 2) # TODO add back jitter using params
            ),
            self.volume.shape[2] - z_step
        )

        xyz_coordinate = (x, y, z)

        subvolume = self.volume[(x - x_step):(x + x_step),
                                (y - y_step):(y + y_step),
                                (z):(z + z_step)]

        if augment_samples:
            flip_direction = np.random.randint(4)
            if flip_direction == 0:
                subvolume = np.flip(subvolume, axis=0)
            elif flip_direction == 1:
                subvolume = np.flip(subvolume, axis=1)
            elif flip_direction == 2:
                subvolume = np.flip(subvolume, axis=0)
                subvolume = np.flip(subvolume, axis=1)

            rotate_direction = np.random.randint(4)
            subvolume = np.rot90(subvolume, k=rotate_direction, axes=(0,1))

        if not return_label:
            # Return plain np arrays so this will cooperate with tf.py_func
            return (self.volume_ID, np.asarray(xyz_coordinate, np.int64),
                    np.asarray(subvolume, np.float32))

        average_label = np.mean(self.ground_truth[(x - x_step):(x + x_step),
                                                  (y - y_step):(y + y_step)])
        
        if average_label > (self.truth_cutoff_high * self.max_truth):
            label = [0.0, 1.0]
        else:
            label = [1.0, 0.0]

        return (self.volume_ID, np.asarray(xyz_coordinate, np.int64),
                np.asarray(subvolume, np.float32), np.asarray(label, np.float32))
            

    def adjust_depth_for_wobble(self, x, y, z):
        """Adjust the z value of a point based on the volume wobble.

        Given a starting (x, y, z) 3D point, use the information about
        the wobbled axes and wobbled angle to return an adjusted z
        value for that point.

        """
        if set(self.wobbled_axes) == {0,2}:
            # plane of rotation = yz
            adjacent_length =  (self.volume.shape[0] / 2) - x
            offset = adjacent_length * np.tan(np.deg2rad(self.wobbled_angle))
            new_z = int(z + offset)

        elif set(self.wobbled_axes) == {1,2}:
            # plane of rotation = xz
            adjacent_length = y - (self.volume.shape[1] / 2)
            offset = adjacent_length * np.tan(np.deg2rad(self.wobbled_angle))
            new_z = int(z + offset)

        else:
            # either no wobble or rotation plane = xy (does not affect depth)
            new_z = z

        return new_z


    def reconstruct3D(self, args, predictionValues, coordinates):
        # Important: assume all coordinates as the center of the subvolume
        center_step = int(round(self.prediction_overlap_step / 2))
        row_step = int(args["subvolume_shape"][1]/2)
        col_step = int(args["subvolume_shape"][0]/2)
        for i in range(coordinates.shape[0]):
            rowpoint = coordinates[i,0]
            colpoint = coordinates[i,1]
            zpoint = coordinates[i,2]
            predictionValue = predictionValues[i,1]

            self.all_preds.append(np.argmax(predictionValues[i,:]))
            if(self.ground_truth[rowpoint,colpoint]) > args['truth_cutoff_high']*self.max_truth:
                self.all_truth.append(1.0)
            else:
                self.all_truth.append(0.0)

            if(center_step > 0):
                self.prediction_volume[rowpoint-center_step:rowpoint+center_step, colpoint-center_step:colpoint+center_step, zpoint] = predictionValue
                self.prediction_plus_surf[rowpoint-center_step:rowpoint+center_step, colpoint-center_step:colpoint+center_step] *= predictionValue

            else:
                self.prediction_volume[rowpoint, colpoint, zpoint] = predictionValue
                self.prediction_plus_surf[rowpoint, colpoint] *= predictionValue


            if ops.isInTestSet(args, rowpoint, colpoint, self.volume.shape, self.train_bounds, self.train_portion):
                self.test_preds.append(self.all_preds[-1])
                self.test_truth.append(self.all_truth[-1])


    def savePrediction3D(self, args, iteration, final_flag=False):
        # save individual pictures
        for d in range(args["predict_depth"]):
            self.savePredictionImage(args, iteration, predictValues=self.prediction_volume[:,:,d], predictionName='ink', depth=d)
        # save the average prediction across depths if depth is more than one
        if args["predict_depth"] > 1:
            self.savePredictionImage(args, iteration, predictValues = np.mean(self.prediction_volume, axis=2), predictionName='ink-average')

        # save the output for samples not trained on
        if args["use_grid_training"]:
            # zero out the appropriate column
            if args["grid_test_square"] % 2 == 0:
                # test is on left side
                self.prediction_volume[ :, int(self.volume.shape[1] / 2):] = 0
            else:
                # test is on right side
                self.prediction_volume[ :, :int(self.volume.shape[1] / 2)] = 0

            n_rows = int(args["grid_n_squares"] / 2)
            voxels_per_row = int(self.volume.shape[0] / n_rows)
            start_row_number = int(args["grid_test_square"] / 2)
            end_row_number = start_row_number + 1
            self.prediction_volume[:(start_row_number*voxels_per_row), :] = 0
            self.prediction_volume[(end_row_number*voxels_per_row):, :] = 0

        else:
            rowBounds, colBounds = ops.bounds(args, [self.volume.shape[0], self.volume.shape[1]], self.train_bounds, self.train_portion)
            self.prediction_volume[rowBounds[0]:rowBounds[1], colBounds[0]:colBounds[1]] = 0

        for d in range(args["predict_depth"]):
            self.savePredictionImage(args, iteration, predictValues=self.prediction_volume[:,:,d], predictionName='ink-no-train', depth=d)
        if args["predict_depth"] > 1:
            self.savePredictionImage(args, iteration, predictValues = np.mean(self.prediction_volume, axis=2), predictionName='ink-average-no-train')

        # zero out the volume
        self.prediction_volume = np.zeros((self.volume.shape[0], self.volume.shape[1], args["predict_depth"]), dtype=np.float32)



    def savePredictionImage(self, args, iteration, predictValues=None, predictionName='ink', depth=0, final_flag=False):
        if predictValues is None:
            predictionImage = (65535 * self.predictionImage).astype(np.uint16)
        else:
            predictionImage = (65535 * predictValues).astype(np.uint16)

        mn = np.min(self.prediction_plus_surf)
        mx = np.min(self.prediction_plus_surf)
        predictionPlusSurfImage = (65535 * (self.prediction_plus_surf - mn) / (mx-mn)).astype(np.uint16)

        try:
            os.makedirs(self.output_path + "/{}/".format(predictionName))
        except:
            pass

        # save the ink and surface predictions
        imageio.imsave(self.output_path + "/{}/prediction-iteration{}-depth{}.tif".format(predictionName, iteration, depth), predictionImage)
        #TODO this doesn't work yet:
        # imageio.imsave(self.output_path + "/{}/predictionPlusSurf-iteration{}-depth{}.tif".format(predictionName, iteration, depth), predictionPlusSurfImage)
        imageio.imsave(self.output_path + "/training-{}.tif".format(iteration), self.training_image)

        # zero them out for the next predictions
        self.prediction_image_ink = np.zeros((self.volume.shape[0], self.volume.shape[1]), dtype=np.float32)
        self.prediction_image_surf = np.zeros((self.volume.shape[0], self.volume.shape[1]), dtype=np.float32)



    def savePredictionMetrics(self, args, iteration, minutes):
        print("\n\n\tMetrics for {}:".format(self.volume_args['name']))
        # all_confusion = confusion_matrix(self.all_truth, self.all_preds)
        # test_confusion = confusion_matrix(self.test_truth, self.test_preds)
        # all_confusion_norm = all_confusion.astype('float') / all_confusion.sum(axis=1)[:, np.newaxis]
        # test_confusion_norm = test_confusion.astype('float') / test_confusion.sum(axis=1)[:, np.newaxis]
        # print("\tNormalized confusion matrix for ALL points: \n{}".format(all_confusion_norm))
        # print("\tNormalized confusion matrix for TEST points: \n{}".format(test_confusion_norm))

        # #calculate metrics
        # all_precision = precision_score(self.all_truth, self.all_preds)
        # all_recall = recall_score(self.all_truth, self.all_preds)
        # all_f1 = f1_score(self.all_truth, self.all_preds)
        # test_precision = precision_score(self.test_truth, self.test_preds)
        # test_recall = recall_score(self.test_truth, self.test_preds)
        # test_f1 = f1_score(self.test_truth, self.test_preds)

        # test_confusion_norm_list = [0,0,0,0]
        # all_confusion_norm_list = [0,0,0,0]
        # try:
        #     test_confusion_norm_list = test_confusion_norm.reshape(4).tolist()
        #     all_confusion_norm_list = all_confusion_norm.reshape(4).tolist()
        # except:
        #     pass
        # #save results in csv
        # column_names = 'iteration, minutes, true positive papyrus, false positive ink, false positive papyrus, true positive ink, precision, recall, f1'
        # self.test_results_norm.append([iteration] + [minutes] + test_confusion_norm_list + [test_precision] + [test_recall] + [test_f1])
        # self.all_results_norm.append([iteration] + [minutes] + all_confusion_norm_list + [all_precision] + [all_recall] + [all_f1])
        # np.savetxt(self.output_path + "confusion-all.csv", self.all_results_norm, fmt='%1.4f', header=column_names, delimiter=',')
        # np.savetxt(self.output_path + "confusion-test.csv", self.test_results_norm, fmt='%1.4f', header=column_names, delimiter=',')

        # # save description of this training session
        # description = ""
        # for arg in sorted(args.keys()):
        #     description += arg+": " + str(args[arg]) + "\n"
        # np.savetxt(self.output_path +'description.txt', [description], delimiter=' ', fmt="%s")
        # shutil.copy(os.path.join(os.path.dirname(inspect.getfile(inkid.model)), 'model.py'),
        #             self.output_path + 'network_model.txt')

        # # zero-out predictions & images so next output is correct
        # self.all_truth = []
        # self.all_preds = []
        # self.test_truth = []
        # self.test_preds = []



    def wobbleVolume(self, args):
        # wobble_start_time = time.time()
        # self.wobbled_angle =  ((2*args["wobble_max_degrees"])*np.random.random_sample()) - args["wobble_max_degrees"]
        # print("Wobbling {} volume {:.2f} degrees...".format(self.volume_args['name'], self.wobbled_angle))
        # self.wobbled_axes = np.random.choice(3, 2, replace=False)
        # self.wobbled_volume = rotate(self.volume, self.wobbled_angle, self.wobbled_axes, order=2, mode='nearest', reshape=False)
        # print("Wobbling took {:.2f} minutes".format((time.time() - wobble_start_time)/60))
        pass

        
    def totalPredictions(self, args, overlap_step):
        #TODO don't count predictions off the fragment
        x_slides = (self.volume.shape[0] - args["subvolume_shape"][0]) / overlap_step
        y_slides = (self.volume.shape[1] - args["subvolume_shape"][1]) / overlap_step
        return int(x_slides * y_slides) * args["predict_depth"]
