import tensorflow as tf
import numpy as np
import pdb
import sys
import json
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score

import data
import model
import ops

args = {
    "trainingDataPath": "/home/volcart/volumes/test/input_MP/",
    "mulitpower": "true",
    "groundTruthFile": "/home/volcart/volumes/test/gt.png",
    "savePredictionPath": sys.argv[1],
    "saveModelPath": sys.argv[1],
    "numChannels": 1,
    "numVolumes": 2,
    "cropX_low": False,
    "cropX_high": False,
    "cropY_low": False,
    "cropY_high": False,
    "x_Dimension": 25,
    "y_Dimension": 25,
    "z_Dimension": 30,
    "stride": 1,
    "predictBatchSize": 24, # NOTE: for multipower single channel, this must be a multiple of numVolumes
    "n_Classes": 2,
    "learningRate": 0.0001,
    "batchSize": 24, # NOTE: for multipower single channel, this must be a multiple of numVolumes
    "dropout": 0.75,
    "epochs": 2,
    "predictStep": 200000,
    "displayStep": 100,
    "saveModelStep": 1000,
    "singleScanPath": "/home/volcart/volumes/packages/CarbonPhantom-Feb2017.volpkg/paths/20170221130948/layered/registered/layers/full-layers/after-rotate/",
    "experimentType": sys.argv[2],
    "scalingFactor": 1.0,
    "randomTrainingSamples": False,
    "graphStep": 1000
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


volume = data.Volume(args)

with tf.Session() as sess:

    # NOTE: if restoring a session, comment out lines tf.global_variables_initializer() and sess.run(init)
    # saver = tf.train.Saver()
    # saver.restore(sess, args["saveModelPath"]+"model-iteration-240000.ckpt") # NOTE: uncomment and change path to restore a graph model

    init = tf.global_variables_initializer()
    sess.run(init)
    # train_writer = tf.summary.FileWriter('/tmp/tb/', sess.graph)

    epoch = 1

    train_accs = []
    train_losses = []
    train_precs = []
    test_accs = []
    test_losses = []
    test_precs = []

    test_coordinates = volume.getRandomTestCoordinates(args)
    testX, testY = volume.getSamples(args, test_coordinates)

    while epoch < args["epochs"]:
        coordinates = volume.getTrainingCoordinates(args)
        # for i in range(0,coordinates.shape[0],args["batchSize"]):
        for i in range(len(coordinates)):
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
                train_preds = sess.run(pred, feed_dict={x: batchX, keep_prob: 1.0})
                test_preds = sess.run(pred, feed_dict={x: testX, y:testY, keep_prob:1.0})
                train_prec = precision_score(np.argmax(batchY, 1), np.argmax(train_preds, 1))
                test_prec = precision_score(np.argmax(testY, 1), np.argmax(test_preds, 1))
                train_accs.append(train_acc)
                test_accs.append(test_acc)
                test_losses.append(test_loss)
                train_losses.append(train_loss)
                train_precs.append(train_prec)
                test_precs.append(test_prec)

                print("Epoch: {}\tIteration: {}\tTotal # iterations: {}".format(epoch, i, coordinates.shape[0]))
                print("Train Loss: {:.3f}\tTrain Acc: {:.3f}\tInk Precision: {:.3f}".format(train_loss, train_acc, train_precs[-1]))
                print("Test Loss: {:.3f}\tTest Acc: {:.3f}\t\tInk Precision: {:.3f}".format(test_loss, test_acc, test_precs[-1]))

            # if i % (args["batchSize"] * args["graphStep"]) == 0 and i > 0:
            #     ops.graph(args, i, test_accs, test_losses, train_accs, train_losses, test_precs, train_precs)

            # NOTE: uncomment/comment to save model
            # if i % (args["batchSize"] * args["saveModelStep"]) == 0 and i > 0:
            #     save_path = saver.save(sess, args["saveModelPath"]+"model-iteration-"+str(i)+".ckpt")
            #     print("Model saved in file: %s" % save_path)

        epoch = epoch + 1

        # save_path = saver.save(sess, args["saveModelPath"]+"model-epoch-"+str(epoch)+".ckpt")
        # print("Model saved in file: %s" % save_path)


    # NOTE ----------------------------
        # make a single prediction after training has completed

    if args["experimentType"] == "multipower-single-channel":
        volume.initPredictionImages(args, args["numVolumes"])
    else:
        volume.initPredictionImages(args, 1)

    coordinates = volume.getPredictionCoordinates()
    predictionValues = []
    for i in range(0,coordinates.shape[0],args["batchSize"]):
        if i < (coordinates.shape[0] - args["batchSize"]):
            batchX, batchY = volume.getSamples(args, coordinates[i:i+args["batchSize"],:])
        else:
            batchX, batchY = volume.getSamples(args, coordinates[i:coordinates.shape[0],:])

        if args["experimentType"] == "multipower-single-channel":
            num_batches = int(batchX.shape[0] / args["numVolumes"])
            # batchX = batchX.reshape((num_batches, args["x_Dimension"], args["y_Dimension"], args["z_Dimension"], args["numVolumes"]))
            batchX = ops.customReshape(args, batchX)
            predictionValues = []
            for j in range(batchX.shape[4]):
                predictionValues.append(sess.run(pred, feed_dict={x: batchX[:,:,:,:,j:j+1], keep_prob: 1.0}))
        else:
            predictionValues = [sess.run(pred, feed_dict={x: batchX, keep_prob: 1.0})]

        if i < (coordinates.shape[0] - args["batchSize"]):
            volume.reconstruct(args, predictionValues, coordinates[i:i+args["batchSize"],:])
        else:
            volume.reconstruct(args, predictionValues, coordinates[i:coordinates.shape[0],:])

        volume.savePredictionImages(args, epoch)
        print("Predicting iteration: {}\t Total number of iterations: {}".format(i, coordinates.shape[0]))
