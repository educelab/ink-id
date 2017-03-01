import tensorflow as tf
import numpy as np
import pdb
import tifffile as tiff
import sys

import data
import model
import time

if len(sys.argv) < 5:
    print("Missing arguments")
    print("Usage: main.py  [xy Dimension]... [z Dimension]... [cushion]... [overlap step]... [dropout probability]...")
    exit()

print("Initializing...")
start_time = time.time()

args = {
    "trainingDataPath": "/home/jack/devel/volcart/small-fragment-data/flatfielded-slices/",
    #"surfaceDataFile": "/home/jack/devel/volcart/small-fragment-data/surf-output-21500/surface-points-21500.tif",
    "surfaceDataFile": "/home/jack/devel/volcart/small-fragment-data/polyfit-slices-degree32-cush16-thresh21500/surface.tif",
    "groundTruthFile": "/home/jack/devel/volcart/small-fragment-data/YZ-aligned-mask.tif",
    "savePredictionPath": "/home/jack/devel/volcart/predictions/3dcnn/",
    "x_Dimension": int(sys.argv[1]),
    "y_Dimension": int(sys.argv[1]),
    "z_Dimension": int(sys.argv[2]),
    "surfaceCushion" : int(sys.argv[3]),
    "overlapStep": int(sys.argv[4]),
    "receptiveField" : [3,3,3],
    "numCubes" : 250,
    "n_Classes": 2,
    "train_portion" : .5,
    "learningRate": 0.001,
    "batchSize": 30,
    "predictBatchSize": 500,
    "dropout": float(sys.argv[5]),
    "trainingIterations": 20001,
    "predictStep": 20000,
    "displayStep": 20,
    "grabNewSamples": 50,
    "surfaceThresh": 21500,
    "notes": "trained on left portion"
}


x = tf.placeholder(tf.float32, [None, args["x_Dimension"], args["y_Dimension"], args["z_Dimension"]])
y = tf.placeholder(tf.float32, [None, args["n_Classes"]])
keep_prob = tf.placeholder(tf.float32)

pred, loss = model.buildModel(x, y, args)
optimizer = tf.train.AdamOptimizer(learning_rate=args["learningRate"]).minimize(loss)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()

volume = data.Volume(args)

with tf.Session() as sess:
    sess.run(init)
    epoch = 0
    avgOutputVolume = []
    while epoch < args["trainingIterations"]:
            if epoch % args["grabNewSamples"] == 0:
                trainingSamples, groundTruth = volume.getTrainingSample(args)


            if epoch % args["grabNewSamples"] % int(args["numCubes"]/8) == 0:
                # periodically shuffle input and labels in parallel
                all_pairs = list(zip(trainingSamples, groundTruth))
                np.random.shuffle(all_pairs)
                trainingSamples, groundTruth = zip(*all_pairs)
                trainingSamples = np.array(trainingSamples)
                groundTruth = np.array(groundTruth)


            randomBatch = np.random.randint(trainingSamples.shape[0] - args["batchSize"])
            batchX = trainingSamples[randomBatch:randomBatch+args["batchSize"]]
            batchY = groundTruth[randomBatch:randomBatch+args["batchSize"]]

            sess.run(optimizer, feed_dict={x: batchX, y: batchY, keep_prob: args["dropout"]})

            if epoch % args["displayStep"] == 0:
                acc = sess.run(accuracy, feed_dict={x: batchX, y: batchY, keep_prob: 1.0})
                evaluatedLoss = sess.run(loss, feed_dict={x: batchX, y: batchY, keep_prob: 1.0})
                print("Epoch: " + str(epoch) + "  Loss: " + str(np.mean(evaluatedLoss)) + "  Acc: " + str(np.mean(acc)))

            if epoch % args["predictStep"] == 0 and epoch > 0:
                print("{} training iterations took {:.2f} minutes".format( \
                    args["predictStep"], (time.time() - start_time)/60))
                startingCoordinates = [0,0]
                predictionSamples, coordinates, nextCoordinates = volume.getPredictionSample(args, startingCoordinates)

                count = 1
                total_predictions = volume.totalPredictions(args)
                while ((count-1)*args["predictBatchSize"]) < total_predictions:
                    print("Predicting cubes {} of {}".format((count * args["predictBatchSize"]), total_predictions))
                    predictionValues = sess.run(pred, feed_dict={x: predictionSamples, keep_prob: 1.0})
                    volume.reconstruct(args, predictionValues, coordinates)
                    predictionSamples, coordinates, nextCoordinates = volume.getPredictionSample(args, nextCoordinates)
                    count += 1
                volume.savePredictionImage(args)

            epoch = epoch + 1

print("full script took {:.2f} minutes".format((time.time() - start_time)/60))
