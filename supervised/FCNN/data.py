import numpy as np
import pdb
import os
from PIL import Image
import math
import datetime
import tifffile as tiff
import ops
from sklearn.metrics import confusion_matrix, recall_score, precision_score
import shutil


class Volume:
    def __init__(self, args):
        # Part 1: volume metadataf
        self.volume_args = args['volumes'][0]
        self.volume_number = volume_number
        self.train_bounds = self.volume_args['train_bounds']
        self.train_portion = self.volume_args['train_portion']


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
        print("  Volume {} shape: {}".format(self.volume_number, self.volume.shape))

        if len(self.volume_args['ground_truth']) > 0:
            self.ground_truth = tiff.imread(self.volume_args['ground_truth'])
        elif 'inked' in self.volume_args['data_path']:
            self.ground_truth = np.ones((self.volume.shape[0:2]), dtype=np.uint16) * 65535
        elif 'blank' in self.volume_args['data_path']:
            self.ground_truth = np.zeros((self.volume.shape[0:2]), dtype=np.uint16)
        else:
            self.ground_truth = np.ones((self.volume.shape[0:2]), dtype=np.uint16) * 65535

        if len(self.volume_args['surface_data']) > 0:
            self.surface_image = tiff.imread(self.volume_args['surface_data'])
            '''
        elif self.volume.shape[2] > args["z_dimension"]:
            print("  Approximating surface for volume {}...".format(self.volume_number))
            surf_start_time = time.time()
            self.surface_image = ops.generateSurfaceApproximation(args, self.volume)
            print("  Surface approximation took {:.2f} minutes.".format((time.time()-surf_start_time)/60))
        '''
        else:
            self.surface_image = np.zeros((self.volume.shape[0:2])) + 50

        self.surface_mask_image = np.ones((self.volume.shape[0], self.volume.shape[1]), dtype=np.int)
        if len(self.volume_args['surface_mask']) > 0:
            self.surface_mask_image = tiff.imread(self.volume_args['surface_mask'])

        # Part 3: prediction data
        self.prediction_volume = np.zeros((self.volume.shape[0], self.volume.shape[1], args["predict_depth"]), dtype=np.float32)
        self.prediction_image_ink = np.zeros((self.volume.shape[0:2]), dtype=np.float32)
        self.prediction_image_surf = np.zeros((self.volume.shape[0:2]), dtype=np.float32)
        self.prediction_plus_surf = np.zeros((self.volume.shape[0:2]), dtype=np.float32)
        self.training_image = np.zeros(self.prediction_image_ink.shape, dtype=np.uint16)
        try:
            self.prediction_overlap_step = self.volume_args["prediction_overlap_step"]
        except:
            print("  No prediction_overlap_step for {}".format(self.volume_args['name']))

        # Part 4: prediction metadata and more
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

        # Part 5: scaling!
        if self.volume_args["scale_factor"] != 1:
            print("  Scaling {}...".format(self.volume_args["name"]))
            self.surface_image = zoom(self.surface_image, self.volume_args["scale_factor"])
            self.surface_image = np.round(self.surface_image *self.volume_args["scale_factor"]).astype(np.uint16)
            self.volume = zoom(self.volume, self.volume_args["scale_factor"])
            self.ground_truth = zoom(self.ground_truth, self.volume_args["scale_factor"])
            self.surface_mask_image = zoom(self.surface_mask_image, self.volume_args["scale_factor"], order=0)
            self.surface_mask = self.surface_mask_image / np.iinfo(self.surface_mask_image.dtype).max
            self.prediction_volume = np.zeros((self.volume.shape[0], self.volume.shape[1], args["predict_depth"]), dtype=np.float32)
            self.prediction_image_ink = zoom(self.prediction_image_ink, self.volume_args["scale_factor"])
            self.prediction_image_surf = zoom(self.prediction_image_surf, self.volume_args["scale_factor"], order=0)
            self.prediction_plus_surf = zoom(self.prediction_plus_surf, self.volume_args["scale_factor"])
            self.training_image = zoom(self.training_image, self.volume_args["scale_factor"])

        # start at a random place on the volume
        self.row_index, self.col_index = ops.getTrainCoordinate(self.row_bounds, self.col_bounds, self.volume.shape)
        self.row_bounds, self.col_bounds = ops.bounds(args, [self.volume.shape[0], self.volume.shape[1]], self.train_bounds, self.train_portion)



    def getMiniBatch(args, x_dimension, y_dimension):
        # lay out the data array
        batch_x = np.zeros((args["minibatch_size"], y_dimension, x_dimension, args["z_dimension"]))
        # lay out the label array
        batch_y = np.zeros((args["minibatch_size"], y_dimension, x_dimension))

        batch_count = 0
        while batch_count < args["minibatch_size"]:
            # make sure the subvolume is within the bounds
            if not (self.row_bounds[0] < (self.row_index + y_dimension) < self.row_bounds[1]):
                # reached bottom edge, go back to top edge
                #TODO add noise to avoid starting at the edge every time
                self.row_index = 0
                self.col_index = 0
                self.epoch += 1
            if not (self.col_bounds[0] < (self.col_index + x_dimension) < self.col_bounds[1])
                # reached right edge, go back to left edge
                #TODO add noise to avoid starting at the edge every time
                self.col_index = 0

            z_coordinate = self.surface_image[self.row_index+int(y_dimension / 2), self.col_index+int(x_dimension / 2)]
            sample = self.volume[self.row_index:self.row_index+y_dimension,
                                    self.col_index:self.col_index+x_dimension,
                                    self.surface_image[z_coordinate:z_coordinate+args["z_dimension"]]
            label = self.ground_truth[self.row_index:self.row_index+y_dimension,
                                    self.col_index:self.col_index+x_dimension]
            # fill in the data
            batch_x[batch_count, 0:sample.shape[0], 0:sample.shape[1], 0:sample.shape[2]] = sample
            batch_y[batch_count] = label
            # move to the next sample
            batch_count += 1
            self.col_index += x_dimension

        return batch_x, batch_y, self.epoch
