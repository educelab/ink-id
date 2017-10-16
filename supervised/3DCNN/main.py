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
from sklearn.metrics import precision_score


print("Initializing...")
start_time = time.time()

args = {
    ### Input configuration ###

    "trainingDataPaths" : ["/home/jack/devel/volcart/ReconstructedCarbonSquares/B_blank_0_cropped/", "/home/jack/devel/volcart/ReconstructedCarbonSquares/B_inked_1_cropped/"],
    "groundTruthFiles": ["/home/jack/devel/volcart/ReconstructedCarbonSquares/black.tif", "/home/jack/devel/volcart/ReconstructedCarbonSquares/white.tif"],
    "surfaceMaskFiles": ["", ""],
    "surfaceDataFiles": ["",""],
    "x_Dimension": 20,
    "y_Dimension": 20,
    "z_Dimension": 64,


    ### Back off from the surface point some distance
    "surface_cushion" : 12,

    ### Network configuration ###
    "use_multitask_training": False,
    "shallow_learning_rate":.001,
    "learning_rate": .001,
    "batch_size": 40,
    "prediction_batch_size": 1000,
    "filter_size" : [3,3,3],
    "dropout": 0.5,
    "neurons": [16,8,4,2],
    "training_iterations": 10000,
    "training_epochs": 2,
    "n_classes": 2,
    "pos_weight": .5,
    "batch_norm_momentum": .9,

    ### Data configuration ###
    "wobble_volume" : False,
    "wobble_step" : 1000,
    "wobble_max_degrees" : 3,
    "num_test_cubes" : 1000,
    "add_random" : False,
    "random_step" : 10, # one in every randomStep non-ink samples will be a random brick
    "random_range" : 200,
    "use_jitter" : True,
    "jitter_range" : [-4, 4],
    "add_augmentation" : True,
    "train_portion" : .6, # Percent of division between train and predict regions
    "balance_samples" : False,
    "use_grid_training": False,
    "grid_n_squares":10,
    "grid_test_square": -1,
    "train_bounds" : [3,3], # bounds parameters: 0=TOP || 1=RIGHT || 2=BOTTOM || 3=LEFT
    "surface_threshold": 20400,
    "restrict_surface": False,

    ### Output configuration ###
    "predict_step": 5000, # make a prediction every x steps
    "overlap_step": 4, # during prediction, predict on one sample for each _ by _ voxel square
    "display_step": 50, # output stats every x steps
    "predict_depth" : 1,
    "output_path": "/home/jack/devel/fall17/predictions/3dcnn/{}-{}-{}h".format(
        datetime.datetime.today().timetuple()[1],
        datetime.datetime.today().timetuple()[2],
        datetime.datetime.today().timetuple()[3]),

    "notes": ""
}

if not (len(args["trainingDataPaths"]) == len(args["surfaceDataFiles"]) == len(args["groundTruthFiles"]) == len(args["surfaceMaskFiles"]) == len(args["train_bounds"])):
    print("Please specify an equal number of data paths, surface files, ground truth files, surface masks, and train bounds in the 'args' dictionary")

x = tf.placeholder(tf.float32, [None, args["x_Dimension"], args["y_Dimension"], args["z_Dimension"]])
y = tf.placeholder(tf.float32, [None, args["n_classes"]])
drop_rate = tf.placeholder(tf.float32)
train_flag = tf.placeholder(tf.bool)


if args["use_multitask_training"]:
    pred, shallow_loss, loss = model.buildMultitaskModel(x, y, drop_rate, args, train_flag)
    shallow_optimizer = tf.train.AdamOptimizer(learning_rate=args["shallow_learning_rate"]).minimize(shallow_loss)
else:
    pred, loss = model.buildModel(x, y, drop_rate, args, train_flag)


update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(learning_rate=args["learning_rate"]).minimize(loss)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
false_positives = tf.equal(tf.argmax(y,1) + 1, tf.argmax(pred, 1))
false_positive_rate = tf.reduce_mean(tf.cast(false_positives, tf.float32))
tf.summary.scalar('accuracy', accuracy)
tf.summary.scalar('xentropy-loss', loss)
if args["use_multitask_training"]:
    tf.summary.scalar('xentropy-shallow-loss', loss)
tf.summary.scalar('false_positive_rate', false_positive_rate)


merged = tf.summary.merge_all()
volumes = multidata.VolumeSet(args)

# create summary writer directory
if tf.gfile.Exists(args["output_path"]):
    tf.gfile.DeleteRecursively(args["output_path"])
tf.gfile.MakeDirs(args["output_path"])


# automatically dump "sess" once the full loop finishes
with tf.Session() as sess:
    print("Beginning train session...")
    print("Output directory: {}".format(args["output_path"]))

    train_writer = tf.summary.FileWriter(args["output_path"] + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(args["output_path"] + '/test')

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    predict_flag = False
    iteration = 0
    iterations_since_prediction = 0
    epoch = 0
    predictions_made = 0
    avgOutputVolume = []
    train_accs = []
    train_losses = []
    train_precs = []
    test_accs = []
    test_losses = []
    test_precs = []
    testX, testY = volumes.getTestBatch(args)

    try:
        while epoch < args["training_epochs"]:
        #while iteration < args["training_iterations"]:

            predict_flag = False

            batchX, batchY, epoch = volumes.getTrainingBatch(args)
            if args["use_multitask_training"]:
                summary, _, _ = sess.run([merged, optimizer, shallow_optimizer], feed_dict={x: batchX, y: batchY, drop_rate:args["dropout"], train_flag:True})
            else:
                summary, _ = sess.run([merged, optimizer], feed_dict={x: batchX, y: batchY, drop_rate:args["dropout"], train_flag:True})

            train_writer.add_summary(summary, iteration)

            if iteration % args["display_step"] == 0:
                train_acc, train_loss, train_preds, train_summary = \
                    sess.run([accuracy, loss, pred, merged], feed_dict={x: batchX, y: batchY, drop_rate: 0.0, train_flag:False})
                test_acc, test_loss, test_preds, test_summary = \
                    sess.run([accuracy, loss, pred, merged], feed_dict={x: testX, y: testY, drop_rate:0.0, train_flag:False})
                train_prec = precision_score(np.argmax(batchY, 1), np.argmax(train_preds, 1))
                test_prec = precision_score(np.argmax(testY, 1), np.argmax(test_preds, 1))

                train_accs.append(train_acc)
                test_accs.append(test_acc)
                test_losses.append(test_loss)
                train_losses.append(train_loss)
                train_precs.append(train_prec)
                test_precs.append(test_prec)

                test_writer.add_summary(test_summary, iteration)

                if (test_acc > .9): #or (test_prec > .8) and (iterations_since_prediction > 1000) and (predictions_made < 4): # or (test_prec / args["numCubes"] < .05)
                    # make a full prediction if results are tentatively spectacular
                    predict_flag = True

                print("Iteration: {}\t\tEpoch: {}".format(iteration, epoch))
                print("Train Loss: {:.3f}\tTrain Acc: {:.3f}\tInk Precision: {:.3f}".format(train_loss, train_acc, train_precs[-1]))
                print("Test Loss: {:.3f}\tTest Acc: {:.3f}\t\tInk Precision: {:.3f}".format(test_loss, test_acc, test_precs[-1]))


            if (predict_flag) or (iteration % args["predict_step"] == 0 and iteration > 0):
                iterations_since_prediction = 0
                predictions_made += 1
                print("{} training iterations took {:.2f} minutes".format( \
                    iteration, (time.time() - start_time)/60))
                startingCoordinates = [0,0,0]
                predictionSamples, coordinates, nextCoordinates = volumes.getPredictionBatch(args, startingCoordinates)

                print("Beginning predictions on volumes...")
                while nextCoordinates is not None:
                    #TODO add back the output
                    predictionValues = sess.run(pred, feed_dict={x: predictionSamples, drop_rate: 0.0, train_flag:False})
                    volumes.reconstruct(args, predictionValues, coordinates)
                    predictionSamples, coordinates, nextCoordinates = volumes.getPredictionBatch(args, nextCoordinates)
                minutes = ( (time.time() - start_time) /60 )
                volumes.saveAllPredictions(args, iteration)
                volumes.saveAllPredictionMetrics(args, iteration, minutes)

            if args["wobble_volume"] and iteration >= args["wobble_step"] and (iteration % args["wobble_step"]) == 0:
                # ex. wobble at iteration 1000, or after the prediction for the previous wobble
                volumes.wobbleVolumes(args)

            iteration += 1
            iterations_since_prediction += 1

    except KeyboardInterrupt:
        # make a prediction if interrupted
        pass

    if iterations_since_prediction > 1:
        # make one last prediction after everything finishes
        startingCoordinates = [0,0,0]
        predictionSamples, coordinates, nextCoordinates = volumes.getPredictionBatch(args, startingCoordinates)
        count = 1
        print("Beginning predictions...")
        while nextCoordinates is not None:
            #TODO add back the output
            predictionValues = sess.run(pred, feed_dict={x: predictionSamples, drop_rate: 0.0, train_flag:False})
            volumes.reconstruct(args, predictionValues, coordinates)
            predictionSamples, coordinates, nextCoordinates = volumes.getPredictionBatch(args, nextCoordinates)
        minutes = ( (time.time() - start_time) /60 )
        volumes.saveAllPredictions(args, iteration)
        volumes.saveAllPredictionMetrics(args, iteration, minutes)



print("full script took {:.2f} minutes".format((time.time() - start_time)/60))
