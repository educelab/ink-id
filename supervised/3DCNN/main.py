import tensorflow as tf
import numpy as np
import pdb
import tifffile as tiff
import sys
import datetime
import data
import model
import time
import ops
import os
from sklearn.metrics import precision_score


print("Initializing...")
start_time = time.time()

args = {
    ### Input configuration ###
    "trainingDataPath" : "/home/jack/devel/volcart/small-fragment-data/flatfielded-slices/",
    "surfaceDataFile": "/home/jack/devel/volcart/small-fragment-surface.tif",
    "groundTruthFile": "/home/jack/devel/volcart/small-fragment-data/ink-only-mask.tif",
    "surfaceMaskFile": "/home/jack/devel/volcart/small-fragment-outline.tif",
    "x_Dimension": 48,
    "y_Dimension": 48,
    "z_Dimension": 48,

    ### Back off from the surface point some distance
    "surface_cushion" : 12,

    ### Network configuration ###
    "filter_size" : [3,3,3],
    "learning_rate": 0.001,
    "batch_size": 24,
    "prediction_batch_size": 1000,
    "dropout": 0.5,
    "neurons": [sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]],
    "training_iterations": 10000,
    "training_epochs": 10,
    "n_classes": 2,
    "pos_weight": .5,

    ### Data configuration ###
    "wobble_volume" : True,
    "wobble_step" : 500,
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
    "use_quadrant_training": False,
    "train_quadrant" : -1, # parameters: 0=test top left (else train) || 1=test top right || 2=test bottom left || 3=test bottom right
    "train_bounds" : 3, # bounds parameters: 0=TOP || 1=RIGHT || 2=BOTTOM || 3=LEFT
    "surface_threshold": 20400,
    "restrict_surface": True,

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


x = tf.placeholder(tf.float32, [None, args["x_Dimension"], args["y_Dimension"], args["z_Dimension"]])
y = tf.placeholder(tf.float32, [None, args["n_classes"]])
drop_rate = tf.placeholder(tf.float32)

pred, loss = model.buildModel(x, y, drop_rate, args)

optimizer = tf.train.AdamOptimizer(learning_rate=args["learning_rate"]).minimize(loss)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
false_positives = tf.equal(tf.argmax(y,1) + 1, tf.argmax(pred, 1))
false_positive_rate = tf.reduce_mean(tf.cast(false_positives, tf.float32))
tf.summary.scalar('accuracy', accuracy)
tf.summary.scalar('xentropy-loss', loss)
tf.summary.scalar('false_positive_rate', false_positive_rate)


merged = tf.summary.merge_all()
volume = data.Volume(args)

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
    testX, testY = volume.getTrainingSample(args, testSet=True)

    try:
        while epoch < args["training_epochs"]:

            predict_flag = False

            batchX, batchY, epoch = volume.getTrainingBatch(args)

            summary, _ = sess.run([merged, optimizer], feed_dict={x: batchX, y: batchY, drop_rate:args["dropout"]})
            train_writer.add_summary(summary, iteration)

            if iteration % args["display_step"] == 0:
                train_acc, train_loss, train_preds, train_summary = \
                    sess.run([accuracy, loss, pred, merged], feed_dict={x: batchX, y: batchY, drop_rate: 0.0})
                test_acc, test_loss, test_preds, test_summary = \
                    sess.run([accuracy, loss, pred, merged], feed_dict={x: testX, y: testY, drop_rate:0.0})
                train_prec = precision_score(np.argmax(batchY, 1), np.argmax(train_preds, 1))
                test_prec = precision_score(np.argmax(testY, 1), np.argmax(test_preds, 1))

                train_accs.append(train_acc)
                test_accs.append(test_acc)
                test_losses.append(test_loss)
                train_losses.append(train_loss)
                train_precs.append(train_prec)
                test_precs.append(test_prec)

                test_writer.add_summary(test_summary, iteration)

                if (test_acc > .9) or (test_prec > .8) and (iterations_since_prediction > 1000) and (predictions_made < 4): # or (test_prec / args["numCubes"] < .05)
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
                predictionSamples, coordinates, nextCoordinates = volume.getPredictionSample3D(args, startingCoordinates)

                count = 1
                total_predictions = volume.totalPredictions(args)
                total_prediction_batches = int(total_predictions / args["prediction_batch_size"])
                print("Beginning predictions...")
                while ((count-1)*args["prediction_batch_size"]) < total_predictions:
                    if (count % int(total_prediction_batches / 10) == 0):
                        #update UI at 10% intervals
                        print("Predicting cubes {} of {}".format((count * args["prediction_batch_size"]), total_predictions))
                    predictionValues = sess.run(pred, feed_dict={x: predictionSamples, drop_rate: 1.0})
                    volume.reconstruct3D(args, predictionValues, coordinates)
                    predictionSamples, coordinates, nextCoordinates = volume.getPredictionSample3D(args, nextCoordinates)
                    count += 1
                minutes = ( (time.time() - start_time) /60 )
                volume.savePrediction3D(args, iteration)
                volume.savePredictionMetrics(args, iteration, minutes)

            if args["wobble_volume"] and iteration >= args["wobble_step"] and (iteration % args["wobble_step"]) == 0:
                # ex. wobble at iteration 1000, or after the prediction for the previous wobble
                volume.wobble_volume(args)

            iteration += 1
            iterations_since_prediction += 1

    except KeyboardInterrupt:
        # make a prediction if interrupted
        pass

    if iterations_since_prediction > 1:
        # make one last prediction after everything finishes
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
            predictionValues = sess.run(pred, feed_dict={x: predictionSamples, drop_rate: 1.0})
            volume.reconstruct3D(args, predictionValues, coordinates)
            predictionSamples, coordinates, nextCoordinates = volume.getPredictionSample3D(args, nextCoordinates)
            count += 1
        minutes = ( (time.time() - start_time) /60 )
        volume.savePrediction3D(args, iteration)
        volume.savePredictionMetrics(args, iteration, minutes)



print("full script took {:.2f} minutes".format((time.time() - start_time)/60))
