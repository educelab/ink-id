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

config = {
    "trainingDataPath": "/home/volcart/volumes/test/input_MP/",
    "mulitpower": True,
    "groundTruthFile": "/home/volcart/volumes/test/gt.png",
    "savePredictionPath": "/home/volcart/volumes/test/",
    "saveModelPath": "/home/volcart/volumes/test/",
    "numChannels": 1,
    "numVolumes": 2,
    "crop": False,
    "cropX_low": 550,
    "cropX_high": 600,
    "cropY_low": 450,
    "cropY_high": 500,
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
    "saveModelStep": 100,
    "scalingFactor": 1.0,
    "graphStep": 1000
}

with open(config["savePredictionPath"]+'info.txt', 'w') as outfile:
    json.dump(config, outfile, indent=4)

x = tf.placeholder(tf.float32, [None, config["x_Dimension"], config["y_Dimension"], config["z_Dimension"], config["numChannels"]])
y = tf.placeholder(tf.float32, [None, config["n_Classes"]])
keep_prob = tf.placeholder(tf.float32)

pred, loss = model.buildModel(x, y, config)
optimizer = tf.train.AdamOptimizer(learning_rate=config["learningRate"]).minimize(loss)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
false_positives = tf.count_nonzero(tf.argmax(pred,1) * (tf.argmax(y,1) - 1))

volume = data.Volume(config)

with tf.Session() as sess:

    # NOTE: if restoring a session, comment out lines tf.global_variables_initializer() and sess.run(init)
    saver = tf.train.Saver()
    # saver.restore(sess, config["saveModelPath"]+"model-iteration-240000.ckpt") # NOTE: uncomment and change path to restore a graph model

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

    test_coordinates = volume.getRandomTestCoordinates(config)
    testX, testY = volume.getSamples(config, test_coordinates)

    while epoch < config["epochs"]:
        coordinates = volume.getTrainingCoordinates(config)
        for i in range(0,coordinates.shape[0],config["batchSize"]):
            if i < (coordinates.shape[0] - config["batchSize"]):
                batchX, batchY = volume.getSamples(config, coordinates[i:i+config["batchSize"],:])
            else:
                batchX, batchY = volume.getSamples(config, coordinates[i:coordinates.shape[0],:])

            for j in range(batchX.shape[4]):
                sess.run(optimizer, feed_dict={x: batchX[:,:,:,:,j:j+1], y: batchY, keep_prob: config["dropout"]})

            if i % (config["batchSize"] * 5) == 0:

                for j in range(batchX.shape[4]):
                    train_acc = sess.run(accuracy, feed_dict={x: batchX[:,:,:,:,j:j+1], y: batchY, keep_prob: 1.0})
                    test_acc = sess.run(accuracy, feed_dict={x:testX[:,:,:,:,j:j+1], y:testY, keep_prob: 1.0})
                    train_loss = sess.run(loss, feed_dict={x: batchX[:,:,:,:,j:j+1], y: batchY, keep_prob: 1.0})
                    test_loss = sess.run(loss, feed_dict={x: testX[:,:,:,:,j:j+1], y:testY, keep_prob: 1.0})
                    train_preds = sess.run(pred, feed_dict={x: batchX[:,:,:,:,j:j+1], keep_prob: 1.0})
                    test_preds = sess.run(pred, feed_dict={x: testX[:,:,:,:,j:j+1], y:testY, keep_prob:1.0})
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

            if i % (config["batchSize"] * config["graphStep"]) == 0 and i > 0:
                ops.graph(config, i, test_accs, test_losses, train_accs, train_losses, test_precs, train_precs)

            # NOTE: uncomment/comment to save model
            if i % (config["batchSize"] * config["saveModelStep"]) == 0 and i > 0:
                save_path = saver.save(sess, config["saveModelPath"]+"model-iteration-"+str(i)+".ckpt")
                print("Model saved in file: %s" % save_path)

        epoch = epoch + 1

    # NOTE ----------------------------
        # make a single prediction after training has completed

    volume.initPredictionImages(config, config["numVolumes"])

    coordinates = volume.getPredictionCoordinates()
    predictionValues = []
    for i in range(0,coordinates.shape[0],config["batchSize"]):
        if i < (coordinates.shape[0] - config["batchSize"]):
            batchX, batchY = volume.getSamples(config, coordinates[i:i+config["batchSize"],:])
        else:
            batchX, batchY = volume.getSamples(config, coordinates[i:coordinates.shape[0],:])

        predictionValues = []
        for j in range(batchX.shape[4]):
            predictionValues.append(sess.run(pred, feed_dict={x: batchX[:,:,:,:,j:j+1], keep_prob: 1.0}))

        if i < (coordinates.shape[0] - config["batchSize"]):
            volume.reconstruct(config, predictionValues, coordinates[i:i+config["batchSize"],:])
        else:
            volume.reconstruct(config, predictionValues, coordinates[i:coordinates.shape[0],:])

        volume.savePredictionImages(config, epoch)
        print("Predicting iteration: {}\t Total number of iterations: {}".format(i, coordinates.shape[0]))
