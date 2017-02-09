import tensorflow as tf
import pdb
import numpy as np

import data as data
import model as VAE

args = {
    "dataPath": "/home/volcart/Pictures/LampBlackTest-2016.volpkg/paths/20161205161113/flattened_i1/",
    "x_Dimension": 50,
    "y_Dimension": 50,
    "z_Dimension": 50,
    "learningRate": 0.001,
    "batchSize": 10,
    "dropout": 0.75,
    "trainingIterations": 10001,
    "analyzeStep": 2000,
    "displayStep": 5,
    "saveSamplePath": "/home/volcart/VAE_Layers/",
    "saveVideoPath": "/home/volcart/VAE_Layers/videos/",
    "ffmpegFileListPath": "/home/volcart/VAE_Layers/concatFileList.txt"
}

n_CubesX, n_CubesY, n_CubesZ, trainingData = data.readData(args)

x = tf.placeholder(tf.float32, [None, args["x_Dimension"], args["y_Dimension"], args["z_Dimension"]])
keep_prob = tf.placeholder(tf.float32)

l1, l2, l3, l4, l5, l6, finalLayer, loss = VAE.buildModel(x, args)
correct_prediction = tf.equal(tf.argmax(x,1), tf.argmax(finalLayer,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
optimizer = tf.train.AdamOptimizer(learning_rate=args["learningRate"]).minimize(loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    epoch = 1
    while epoch < args["trainingIterations"]:

        randomBatch = np.random.randint(trainingData.shape[0] - args["batchSize"])
        batchX = trainingData[randomBatch:randomBatch+args["batchSize"]]

        _ = sess.run(optimizer, feed_dict={x: batchX, keep_prob: args["dropout"]})
        if epoch % args["displayStep"] == 0:
            evaluatedLoss = sess.run(loss, feed_dict={x: batchX, keep_prob: 1.0})
            evaluatedAccuracy = sess.run(accuracy, feed_dict={x: batchX, keep_prob: 1.0})
            print("Epoch: " + str(epoch) + "  Loss: " + str(np.mean(evaluatedLoss)) + "  Accuracy: " + str(np.mean(evaluatedAccuracy)))

        if epoch % args["analyzeStep"] == 0:
            cubeCount = 0
            l1_x_Dimension, l2_x_Dimension, l3_x_Dimension, l4_x_Dimension, l5_x_Dimension, l6_x_Dimension = [], [], [], [], [], []
            for i in range(n_CubesX):
                l1_y_Dimension, l2_y_Dimension, l3_y_Dimension, l4_y_Dimension, l5_y_Dimension, l6_y_Dimension = [], [], [], [], [], []
                for j in range(n_CubesY):
                    l1_z_Dimension, l2_z_Dimension, l3_z_Dimension, l4_z_Dimension, l5_z_Dimension, l6_z_Dimension = [], [], [], [], [], []
                    for k in range(n_CubesZ):
                        l1_out, l2_out, l3_out, l4_out, l5_out, l6_out = sess.run([l1, l2, l3, l4, l5, l6], feed_dict={x: trainingData[cubeCount,:,:,:].reshape((1, args["x_Dimension"], args["y_Dimension"], args["z_Dimension"])), keep_prob: 1.0})
                        cubeCount += 1
                        l1_z_Dimension.append(l1_out), l2_z_Dimension.append(l2_out), l3_z_Dimension.append(l3_out), l4_z_Dimension.append(l4_out), l5_z_Dimension.append(l5_out), l6_z_Dimension.append(l6_out)
                    l1_y_Dimension.append(np.concatenate(l1_z_Dimension, axis=3)), l2_y_Dimension.append(np.concatenate(l2_z_Dimension, axis=3)), l3_y_Dimension.append(np.concatenate(l3_z_Dimension, axis=3)), l4_y_Dimension.append(np.concatenate(l4_z_Dimension, axis=3)), l5_y_Dimension.append(np.concatenate(l5_z_Dimension, axis=3)), l6_y_Dimension.append(np.concatenate(l6_z_Dimension, axis=3))
                l1_x_Dimension.append(np.concatenate(l1_y_Dimension, axis=2)), l2_x_Dimension.append(np.concatenate(l2_y_Dimension, axis=2)), l3_x_Dimension.append(np.concatenate(l3_y_Dimension, axis=2)), l4_x_Dimension.append(np.concatenate(l4_y_Dimension, axis=2)), l5_x_Dimension.append(np.concatenate(l5_y_Dimension, axis=2)), l6_x_Dimension.append(np.concatenate(l6_y_Dimension, axis=2))
            l1_Samples = np.squeeze(np.concatenate(l1_x_Dimension, axis=1))
            l2_Samples = np.squeeze(np.concatenate(l2_x_Dimension, axis=1))
            l3_Samples = np.squeeze(np.concatenate(l3_x_Dimension, axis=1))
            l4_Samples = np.squeeze(np.concatenate(l4_x_Dimension, axis=1))
            l5_Samples = np.squeeze(np.concatenate(l5_x_Dimension, axis=1))
            l6_Samples = np.squeeze(np.concatenate(l6_x_Dimension, axis=1), axis=0)

            data.saveSamples(args, np.array([l1_Samples, l2_Samples, l3_Samples, l4_Samples, l5_Samples, l6_Samples]))

        epoch += 1
