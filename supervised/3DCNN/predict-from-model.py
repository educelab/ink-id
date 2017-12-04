import tensorflow as tf
import numpy as np
import pdb
import tifffile as tiff
import sys
import datetime
import data
import multidata
import model
import time
import ops
import os
import warnings
from sklearn.metrics import precision_score, f1_score


print("Initializing...")
start_time = time.time()
warnings.filterwarnings('ignore')

args = {
"model_path" : "/home/jack/devel/fall17/predictions/3dcnn/11-28-11h/models/model-90000.ckpt",
    ### Input configuration ###
    "volumes": [
        {
            "name": "pherc2",
            "microns_per_voxel":10,
            "data_path": "/home/jack/devel/volcart/pherc2/oriented-scaled-brightened-slices/",
            "ground_truth":"",
            "surface_mask":"/home/jack/devel/volcart/pherc2/pherc2-surface-mask-scaled.tif",
            "surface_data":"/home/jack/devel/volcart/pherc2/pherc2-surface-points.tif",
            "train_portion":.6,
            "train_bounds":3,# bounds parameters: 0=TOP || 1=RIGHT || 2=BOTTOM || 3=LEFT
            "use_in_training":True,
            "use_in_test_set":True,
            "make_prediction":True,
            "prediction_overlap_step":4,
            "scale_factor": 1,
        },

    ],


    # hackish way to make dimensions even
    "x_dimension": 48,#int(96 * float(sys.argv[1]) / 2) * 2,
    "y_dimension": 48,#int(96 * float(sys.argv[1]) / 2) * 2,
    "z_dimension": 24,#int(48 * float(sys.argv[1]) / 2) * 2,

    ### Back off from the surface point some distance
    "surface_cushion" : 0, #int(8 * float(sys.argv[1])),

    ### Network configuration ###
    "use_multitask_training": False,
    "shallow_learning_rate":.001,
    "learning_rate": .0001,
    "batch_size": 30,
    "prediction_batch_size": 500,
    "filter_size" : [3,3,3],
    "dropout": 0.5,
    "neurons": [16,8,4,2],
    "training_iterations": 100000,
    "training_epochs": 3, #int(3 / pow(float(sys.argv[1]), 2)),
    "n_classes": 2,
    "pos_weight": .5,
    "batch_norm_momentum": .9,
    "fbeta_weight": 0.2,

    ### Data configuration ###
    "wobble_volume" : False,
    "wobble_step" : 1000,
    "wobble_max_degrees" : 2,
    "num_test_cubes" : 500,
    "add_random" : False,
    "random_step" : 10, # one in every randomStep non-ink samples will be a random brick
    "random_range" : 200,
    "use_jitter" : True,
    "jitter_range" : [-6, 6],
    "add_augmentation" : True,
    "balance_samples" : True,
    "use_grid_training": False,
    "grid_n_squares":10,
    "grid_test_square": -1,
    "surface_threshold": 20400,
    "restrict_surface": True,
    "truth_cutoff_low": .2,
    "truth_cutoff_high": .8,

    ### Output configuration ###
    "predict_step": 10000, # make a prediction every x steps
    "display_step": 20, # output stats every x steps
    "predict_depth" : 2,
    "output_path": "/home/jack/devel/fall17/predictions/3dcnn/P{}-{}-{}h".format(
        datetime.datetime.today().timetuple()[1],
        datetime.datetime.today().timetuple()[2],
        datetime.datetime.today().timetuple()[3]),
    "notes": ""
}


x = tf.placeholder(tf.float32, [None, args["x_dimension"], args["y_dimension"], args["z_dimension"]])
y = tf.placeholder(tf.float32, [None, args["n_classes"]])
drop_rate = tf.placeholder(tf.float32)
training_flag = tf.placeholder(tf.bool)


merged = tf.summary.merge_all()
pred, loss = model.buildModel(x, y, drop_rate, args, training_flag)
saver = tf.train.Saver()
volumes = multidata.VolumeSet(args)


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
    predictionSamples, coordinates, nextCoordinates = volumes.getPredictionBatch(args, startingCoordinates, v_olap=4)
    count = 1
    while nextCoordinates is not None:
        #TODO add back the output
        predictionValues = sess.run(pred, feed_dict={x: predictionSamples, drop_rate: 0.0, training_flag:False})
        volumes.reconstruct(args, predictionValues, coordinates)
        predictionSamples, coordinates, nextCoordinates = volumes.getPredictionBatch(args, nextCoordinates, v_olap=4)
    minutes = ( (time.time() - start_time) /60 )
    volumes.saveAllPredictions(args, best_acc_iteration)
    volumes.saveAllPredictionMetrics(args, best_acc_iteration, minutes)


print("full script took {:.2f} minutes".format((time.time() - start_time)/60))
