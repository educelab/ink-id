import tensorflow as tf
import numpy as np
import pdb
import sys
import datetime
import time

import os
from sklearn.metrics import precision_score, f1_score

import inkid.volumes
import inkid.model

print("Initializing...")
start_time = time.time()

args = {
    ### Input configuration ###
    "model_path" : "/home/jack/devel/fall17/predictions/3dcnn/10-25-11h/models/model.ckpt",

    "trainingDataPath" : "/home/jack/devel/volcart/small-fragment-data/flatfielded-slices/",
    "surfaceDataFile": "/home/jack/devel/volcart/small-fragment-data/polyfit-slices-degree32-cush16-thresh20500/surface.tif",
    "groundTruthFile": "/home/jack/devel/volcart/small-fragment-gt.tif",
    "surfaceMaskFile": "/home/jack/devel/volcart/small-fragment-outline.tif",
    "x_Dimension": 96,
    "y_Dimension": 96,
    "z_Dimension": 48,


    ### Back off from the surface point some distance
    "surface_cushion" : 12,

    ### Network configuration ###
    "prediction_batch_size": 500,
    "filter_size" : [3,3,3],
    "neurons": [16, 8, 4, 2],
    "batch_norm_momentum": .9,

    ### Data configuration ###
    "train_portion" : .6, # Percent of division between train and predict regions
    "balance_samples" : True,
    "use_grid_training": True,
    "grid_n_squares":10,
    "grid_test_square": 9,
    "train_bounds" : 3,
    "restrict_surface": True,

    ### Output configuration ###
    "overlap_step": 4, # during prediction, predict on one sample for each _ by _ voxel square
    "predict_depth" : 1,
    "output_path": "/home/jack/devel/fall17/predictions/3dcnn/P{}-{}-{}h".format(
        datetime.datetime.today().timetuple()[1],
        datetime.datetime.today().timetuple()[2],
        datetime.datetime.today().timetuple()[3]),

    "notes": ""
}


x = tf.placeholder(tf.float32, [None, args["x_Dimension"], args["y_Dimension"], args["z_Dimension"]])
y = tf.placeholder(tf.float32, [None, 2])
drop_rate = tf.placeholder(tf.float32)
training_flag = tf.placeholder(tf.bool)


merged = tf.summary.merge_all()
pred, loss = inkid.model.buildModel(x, y, drop_rate, args, training_flag)
saver = tf.train.Saver()
volume = inkid.volumes.Volume(args)


# automatically dump "sess" once the full loop finishes
with tf.Session() as sess:
    print("Beginning prediction session...")
    print("Output directory: {}".format(args["output_path"]))


    best_acc_iteration = 0
    # make one last prediction after everything finishes
    # use the model that performed best on the test set :)
    saver.restore(sess, args["model_path"])
    print("Beginning predictions from best model (iteration {})...".format(best_acc_iteration))
    startingCoordinates = [0,0,0]
    predictionSamples, coordinates, nextCoordinates = volume.getPredictionSample3D(args, startingCoordinates)
    count = 1
    total_predictions = volume.totalPredictions(args)
    total_prediction_batches = int(total_predictions / args["prediction_batch_size"])
    print("Beginning predictions...")
    while ((count-1)*args["prediction_batch_size"]) < total_predictions:
        if (count % int(total_prediction_batches / 10) == 0):
            #update UI at 10% intervals
            print("Predicting cubes {} of {}".format((count * args["prediction_batch_size"]), total_predictions))
        predictionValues = sess.run(pred, feed_dict={x: predictionSamples, drop_rate: 0.0, training_flag:False})
        volume.reconstruct3D(args, predictionValues, coordinates)
        predictionSamples, coordinates, nextCoordinates = volume.getPredictionSample3D(args, nextCoordinates)
        count += 1
    minutes = ( (time.time() - start_time) /60 )
    volume.savePrediction3D(args, best_acc_iteration, final_flag=True)
    volume.savePredictionMetrics(args, best_acc_iteration, minutes, final_flag=True)



print("full script took {:.2f} minutes".format((time.time() - start_time)/60))
