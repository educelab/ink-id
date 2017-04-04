import tensorflow as tf
import numpy as np
import pdb
# import tifffile as tiff
import sys

import data
import model

# if sys.argv[5] == "true":
#     mulitpower = True
# else:
#     multipower = False

args = {
    "trainingDataPath": "/home/volcart/volumes/packages/CarbonPhantom-Feb2017.volpkg/paths/20170221130948/layered/registered/layers/full-layers/after-rotate/",
    "mulitpower": "false",
    "groundTruthFile": "/home/volcart/volumes/packages/CarbonPhantom-Feb2017.volpkg/paths/20170221130948/layered/registered/ground-truth/GroundTruth-CarbonInk.png",
    "savePredictionPath": "/home/volcart/supervised-results/CarbonPhantom-Feb2017/",
    "saveModelPath": "/home/volcart/prelim-InkDetection/src/results/models/",
    "numChannels": 1,
    "cropX_low": int(sys.argv[1]),
    "cropX_high": int(sys.argv[2]),
    "cropY_low": int(sys.argv[3]),
    "cropY_high": int(sys.argv[4]),
    "x_Dimension": 25,
    "y_Dimension": 25,
    "z_Dimension": 17,
    "stride": 1,
    "numCubes": 500,
    "predictBatchSize": 500,
    "n_Classes": 2,
    "learningRate": 0.0001,
    "batchSize": 30,
    "dropout": 0.75,
    "trainingIterations": 30001,
    "grabNewSamples": 40,
    "predictStep": 2000,
    "displayStep": 10,
    "singleScanPath": "/home/volcart/volumes/packages/CarbonPhantom-Feb2017.volpkg/paths/20170221130948/layered/registered/layers/full-layers/after-rotate/"
}

x = tf.placeholder(tf.float32, [None, args["x_Dimension"], args["y_Dimension"], args["z_Dimension"], args["numChannels"]])
y = tf.placeholder(tf.float32, [None, args["n_Classes"]])
keep_prob = tf.placeholder(tf.float32)

pred, loss = model.buildModel(x, y, args)
optimizer = tf.train.AdamOptimizer(learning_rate=args["learningRate"]).minimize(loss)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()
saver = tf.train.Saver()

volume = data.Volume(args)

with tf.Session() as sess:
    sess.run(init)
    # saver.restore(sess, args["saveModelPath"]+"model-epoch-4000.ckpt")
    epoch = 0
    avgOutputVolume = []
    while epoch < args["trainingIterations"]:
        if epoch % args["grabNewSamples"] == 0:
            trainingSamples, groundTruth = volume.getTrainingSample(args)

        randomBatch = np.random.randint(trainingSamples.shape[0] - args["batchSize"])
        batchX = trainingSamples[randomBatch:randomBatch+args["batchSize"]]
        batchY = groundTruth[randomBatch:randomBatch+args["batchSize"]]

        sess.run(optimizer, feed_dict={x: batchX, y: batchY, keep_prob: args["dropout"]})

        if epoch % args["displayStep"] == 0:
            acc = sess.run(accuracy, feed_dict={x: batchX, y: batchY, keep_prob: 1.0})
            evaluatedLoss = sess.run(loss, feed_dict={x: batchX, y: batchY, keep_prob: 1.0})
            print("Epoch: " + str(epoch) + "  Loss: " + str(np.mean(evaluatedLoss)) + "  Acc: " + str(np.mean(acc)))

        if epoch % args["predictStep"] == 0 and epoch > 0:
            volume.emptyPredictionVolume(args)
            total_num_predictions = volume.totalPredictions(args)

            startingCoordinates = [0,0,0]
            predictionSamples, coordinates, nextCoordinates = volume.getPredictionSample(args, startingCoordinates)

            count = 1
            while ((count-1)*args["predictBatchSize"]) < total_num_predictions: # TODO what about the special case where the volume is a perfect multiple of the numCubes?
                print("Predicting cubes {} of {}".format((count * args["predictBatchSize"]), total_num_predictions))
                predictionValues = sess.run(pred, feed_dict={x: predictionSamples, keep_prob: 1.0})
                volume.reconstruct(args, predictionValues, coordinates)
                predictionSamples, coordinates, nextCoordinates = volume.getPredictionSample(args, nextCoordinates)
                count += 1
                volume.savePredictionImage(args, epoch)
            volume.savePredictionImage(args, epoch)
            # save_path = saver.save(sess, args["saveModelPath"]+"model-epoch-"+str(epoch)+".ckpt")
            # print("Model saved in file: %s" % save_path)


        epoch = epoch + 1
