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
    "batchSize": 5,
    "dropout": 0.75,
    "trainingIterations": 50000,
    "analyzeStep": 50,
    "displayStep": 5
}

trainingData = data.readData(args)
# TODO: save sample volumes for comparison purposes

x = tf.placeholder(tf.float32, [None, args["x_Dimension"], args["y_Dimension"], args["z_Dimension"]])
keep_prob = tf.placeholder(tf.float32)

l1, l2, l3, l4, l5, l6, loss = VAE.buildModel(x, args)
optimizer = tf.train.AdamOptimizer(learning_rate=args["learningRate"]).minimize(loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    epoch = 0
    while epoch < args["trainingIterations"]:

        randomBatch = np.random.randint(trainingData.shape[0] - args["batchSize"])
        batchX = trainingData[randomBatch:randomBatch+args["batchSize"]]
        # pdb.set_trace()

        _ = sess.run(optimizer, feed_dict={x: batchX, keep_prob: args["dropout"]})

        if epoch % args["displayStep"]:
            evaluatedLoss = sess.run(loss, feed_dict={x: batchX, keep_prob: args["dropout"]})
            print("Epoch: " + str(epoch) + "  Loss: " + str(np.mean(evaluatedLoss)))

        # if epoch % args["analyzeStep"]:
        #     # TODO: layers output shouldn't be activated
        #     l1_out, l2_out, l3_out, l4_out, l5_out, l6_out = sess.run([l1, l2, l3, l4, l5, l6], feed_dict={x: batchX, keep_prob: 1.0})
        #     pdb.set_trace()
            # TODO: save predictions in meaningful way
        epoch += 1
