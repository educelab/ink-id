import tensorflow as tf
import numpy as np
import pdb
import sys
import json
import matplotlib.pyplot as plt

import data
import model

args = {
    "trainingDataPath": "/home/volcart/volumes/packages/CarbonPhantom_MP_2017.volpkg/paths/all-cols/",
    "mulitpower": "true",
    "groundTruthFile": "/home/volcart/volumes/packages/CarbonPhantom_MP_2017.volpkg/paths/all-cols/GroundTruth-CarbonInk.png",
    "savePredictionPath": sys.argv[5],
    "saveModelPath": sys.argv[7],
    "numChannels": 1,
    "numVolumes": 6,
    "cropX_low": int(sys.argv[1]),
    "cropX_high": int(sys.argv[2]),
    "cropY_low": int(sys.argv[3]),
    "cropY_high": int(sys.argv[4]),
    "x_Dimension": 25,
    "y_Dimension": 25,
    "z_Dimension": 71,
    "stride": 1,
    "predictBatchSize": 24, # NOTE: for multipower single channel, this must be a multiple of numVolumes
    "n_Classes": 2,
    "learningRate": 0.0001,
    "batchSize": 24, # NOTE: for multipower single channel, this must be a multiple of numVolumes
    "dropout": 0.75,
    "epochs": 2,
    "predictStep": 200000,
    "displayStep": 100,
    "saveModelStep": 1,
    "singleScanPath": "/home/volcart/volumes/packages/CarbonPhantom-Feb2017.volpkg/paths/20170221130948/layered/registered/layers/full-layers/after-rotate/",
    "experimentType": sys.argv[6],
    "scalingFactor": float(sys.argv[8]),
    "randomTrainingSamples": False
}

with open(args["savePredictionPath"]+'info.txt', 'w') as outfile:
    json.dump(args, outfile, indent=4)

if args["experimentType"] == "multipower-single-channel":
    x = tf.placeholder(tf.float32, [None, args["x_Dimension"], args["y_Dimension"], args["z_Dimension"], args["numChannels"]])
elif args["experimentType"] == "multipower-multinetwork":
    x = tf.placeholder(tf.float32, [None, args["x_Dimension"], args["y_Dimension"], args["z_Dimension"], args["numVolumes"]])
y = tf.placeholder(tf.float32, [None, args["n_Classes"]])
keep_prob = tf.placeholder(tf.float32)

if args["experimentType"] == "multipower-single-channel":
    pred, loss = model.buildModel(x, y, args)
elif args["experimentType"] == "multipower-multinetwork":
    pred, loss = model.buildMultiNetworkModel(x, y, args)
optimizer = tf.train.AdamOptimizer(learning_rate=args["learningRate"]).minimize(loss)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
false_positives = tf.count_nonzero(tf.argmax(pred,1) * (tf.argmax(y,1) - 1))

saver = tf.train.Saver()

volume = data.Volume(args)

with tf.Session() as sess:
    # train_writer = tf.summary.FileWriter('/tmp/tb/', sess.graph)

    init = tf.global_variables_initializer()
    sess.run(init)
    # saver.restore(sess, args["saveModelPath"]+"run1/model-epoch-10000.ckpt") # NOTE: uncomment and change path to restore a graph model
    epoch = 1

    avgOutputVolume = []
    train_accs = []
    train_losses = []
    train_fps = []
    test_accs = []
    test_losses = []
    test_fps = []

    test_coordinates = volume.getRandomTestCoordinates(args)
    testX, testY = volume.getSamples(args, test_coordinates)

    while epoch < args["epochs"]:

        # if args["experimentType"] == "multipower-single-channel":
        #     trainingSamples, groundTruth = volume.getTrainingSample_MultipowerSingleChannel(args)
        # elif args["experimentType"] == "multipower-multinetwork":
        #     trainingSamples, groundTruth = volume.getTrainingSample(args)

        coordinates = volume.getTrainingCoordinates(args)
        for i in range(0,coordinates.shape[0],args["batchSize"]):
            if i < (coordinates.shape[0] - args["batchSize"]):
                batchX, batchY = volume.getSamples(args, coordinates[i:i+args["batchSize"],:])
            else:
                batchX, batchY = volume.getSamples(args, coordinates[i:coordinates.shape[0],:])
            sess.run(optimizer, feed_dict={x: batchX, y: batchY, keep_prob: args["dropout"]})

            if i % (args["batchSize"] * 5) == 0:

                train_acc = sess.run(accuracy, feed_dict={x: batchX, y: batchY, keep_prob: 1.0})
                test_acc = sess.run(accuracy, feed_dict={x:testX, y:testY, keep_prob: 1.0})
                train_loss = sess.run(loss, feed_dict={x: batchX, y: batchY, keep_prob: 1.0})
                test_loss = sess.run(loss, feed_dict={x: testX, y:testY, keep_prob: 1.0})
                train_fp = sess.run(false_positives, feed_dict={x: batchX, y:batchY, keep_prob:1.0})
                test_fp = sess.run(false_positives, feed_dict={x:testX, y:testY, keep_prob:1.0})
                train_accs.append(train_acc)
                test_accs.append(test_acc)
                test_losses.append(test_loss)
                train_losses.append(train_loss)
                test_fps.append(test_fp / args["batchSize"])
                train_fps.append(train_fp / args["batchSize"])

                print("Epoch: {}\tIteration: {}\tTotal # iterations: {}".format(epoch, i, coordinates.shape[0]))
                print("Train Loss: {:.3f}\tTrain Acc: {:.3f}\tFp: {}".format(train_loss, train_acc, train_fp))
                print("Test Loss: {:.3f}\tTest Acc: {:.3f}\t\tFp: {}".format(test_loss, test_acc, test_fp))


        pdb.set_trace()
        plt.figure(1)
        plt.clf()
        plt.subplot(311) # losses
        axes = plt.gca()
        axes.set_ylim([0,np.median(test_losses)+1])
        xs = np.arange(len(train_accs))
        plt.plot(train_losses, 'k.')
        plt.plot(test_losses, 'g.')
        plt.subplot(312) # accuracies
        plt.plot(train_accs, 'k.')
        plt.plot(xs, np.poly1d(np.polyfit(xs, train_accs, 1))(xs), color='k')
        plt.plot(test_accs, 'g.')
        plt.plot(xs, np.poly1d(np.polyfit(xs, test_accs, 1))(xs), color='g')
        plt.subplot(313) # false positives
        plt.plot(train_fps, 'k.')
        plt.plot(test_fps, 'g.')
        plt.savefig(args["savePredictionPath"]+"plots-{}.png".format(epoch))

        # NOTE: uncomment to save model
        if epoch % args["saveModelStep"] == 0 and epoch > 0:
            save_path = saver.save(sess, args["saveModelPath"]+"model-epoch-"+str(epoch)+".ckpt")
            print("Model saved in file: %s" % save_path)

        epoch = epoch + 1


    # NOTE ----------------------------
        # make a single prediction after training has completed

    coordinates = volume.getPredictionCoordinates()
    predictionValues = []
    for i in range(0,coordinates.shape[0],args["batchSize"]):
        if i < (coordinates.shape[0] - args["batchSize"]):
            batchX, batchY = volume.getSamples(args, coordinates[i:i+args["batchSize"],:])
        else:
            batchX, batchY = volume.getSamples(args, coordinates[i:coordinates.shape[0],:])

        predictionValues.append(sess.run(pred, feed_dict={x: batchX, keep_prob: 1.0}))

    total_num_predictions = volume.totalPredictions(args)
    startingCoordinates = [0,0,0]
    count = 1

    if args["experimentType"] == "multipower-single-channel":
        volume.initPredictionImages(args)
        predictionSamples, coordinates, nextCoordinates = volume.getPredictionSample_MultipowerSingleChannel(args, startingCoordinates)
        while ((count-1)*args["predictBatchSize"]) < total_num_predictions:
            print("Predicting cubes {} of {}".format((count * args["predictBatchSize"]), total_num_predictions))
            predictionValues = []
            for i in range(predictionSamples.shape[4]):
                predictionValues.append(sess.run(pred, feed_dict={x: predictionSamples[:,:,:,:,i:i+1], keep_prob: 1.0}))
            volume.reconstruct_MulipowerSingleChannel(args, predictionValues, coordinates)
            predictionSamples, coordinates, nextCoordinates = volume.getPredictionSample_MultipowerSingleChannel(args, nextCoordinates)
            count += 1
            volume.savePredictionImages(args, epoch)
        volume.savePredictionImages(args, epoch)

    elif args["experimentType"] == "multipower-multinetwork":
        volume.emptyPredictionImage(args)
        predictionSamples, coordinates, nextCoordinates = volume.getPredictionSample_MultipowerSingleChannel(args, startingCoordinates)
        while ((count-1)*args["predictBatchSize"]) < total_num_predictions:
            print("Predicting cubes {} of {}".format((count * args["predictBatchSize"]), total_num_predictions))
            predictionValues = sess.run(pred, feed_dict={x: predictionSamples, keep_prob: 1.0})
            volume.reconstruct(args, predictionValues, coordinates)
            predictionSamples, coordinates, nextCoordinates = volume.getPredictionSample_MultipowerSingleChannel(args, nextCoordinates)
            count += 1
            volume.savePredictionImage(args, epoch)
        volume.savePredictionImage(args, epoch)
