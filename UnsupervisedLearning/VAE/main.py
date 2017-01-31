import tensorflow as tf
import pdb

import data as data
import model as VAE

args = {
    "dataPath": "/home/volcart/Pictures/LampBlackTest-2016.volpkg/paths/20161205161113/flattened_i1/",
    "x_Dimension": 50,
    "y_Dimension": 50,
    "z_Dimension": 50,
    "learningRate": 0.001,
    "batchSize": 50,
    "dropout": 0.75,
    "trainingIterations": 50000,
    "analyzeStep": 50,
    "displayStep": 5
}

trainingData = data.readData(args)

x = tf.placeholder(tf.float32, [None, args["x_Dimension"], args["y_Dimension"], args["z_Dimension"]])
keep_prob = tf.placeholder(tf.float32)

l1, l2, l3, l4, l5, l6, loss = VAE.buildModel(x, args)
optimizer = tf.train.AdamOptimizer(learning_rate=args["learningRate"]).minimize(loss)
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    epoch = 0
    while epoch < args["trainingIterations"]:
        _ = sess.run(optimizer, feed_dict={x: batchX, keep_prob: args["dropout"]})

        if epoch % args["displayStep"]:
            evaluatedLoss = sess.run(loss, feed_dict={x: batchX, keep_prob: args["dropout"]})
            print("Epoch: " + str(epoch) + "  Loss: " + str(evaluatedLoss))

        if epoch % args["analyzeStep"]:
            l1_out, l2_out, l3_out, l4_out, l5_out, l6_out = sess.run([l1, l2, l3, l4, l5, l6], feed_dict={x: batchX, keep_prob: 1.0})
            # TODO: save predictions in meaningful way
        epoch += 1
