'''
model.py
    - variational autoencoder, modeled after: https://arxiv.org/pdf/1312.6114.pdf
'''

__author__ = "Kendall Weihe"
__email__ = "kendall.weihe@uky.edu"


import tensorflow as tf
import tensorflow.contrib.slim as slim
import pdb
import math
import numpy as np

from ops import *

networkParams = {
    "nFilter0": 1,
    "nFilter1": 2,
    "nFilter2": 4,
    "nFilter3": 8,
    "zSize": 20
}

weights = {
    'wc1' : tf.get_variable("weights_1", shape=[3, 3, 3, networkParams["nFilter0"], networkParams["nFilter1"]],
               initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),

    'wc2' : tf.get_variable("weights_2", shape=[3, 3, 3, networkParams["nFilter1"], networkParams["nFilter2"]],
               initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),

    'wc3' : tf.get_variable("weights_3", shape=[3, 3, 3, networkParams["nFilter2"], networkParams["nFilter3"]],
               initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),

    'wdc1' : tf.get_variable("weights_4", shape=[4, 4, 4, networkParams["nFilter2"], networkParams["nFilter3"]],
               initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),

    'wdc2' : tf.get_variable("weights_5", shape=[4, 4, 4, networkParams["nFilter1"], networkParams["nFilter2"]],
               initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),

    'wdc3' : tf.get_variable("weights_6", shape=[4, 4, 4, networkParams["nFilter0"], networkParams["nFilter1"]],
               initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
}

biases = {
    'bc1': tf.Variable(tf.zeros([networkParams["nFilter1"]], dtype=tf.float32), name="biases_1", dtype=tf.float32),
    'bc2': tf.Variable(tf.zeros([networkParams["nFilter2"]], dtype=tf.float32), name="biases_2", dtype=tf.float32),
    'bc3': tf.Variable(tf.zeros([networkParams["nFilter3"]], dtype=tf.float32), name="biases_3", dtype=tf.float32),
    'bdc1': tf.Variable(tf.zeros([networkParams["nFilter2"]], dtype=tf.float32), name="biases_4", dtype=tf.float32),
    'bdc2': tf.Variable(tf.zeros([networkParams["nFilter1"]], dtype=tf.float32), name="biases_5", dtype=tf.float32),
    'bdc3': tf.Variable(tf.zeros([networkParams["nFilter0"]], dtype=tf.float32), name="biases_6", dtype=tf.float32),
}

def buildModel(x, args):
    l1, l2, l3, z_Mean, z_Stddev = encoder(x, args)
    batchSize = tf.shape(x)[0]
    randomSample = tf.random_normal([batchSize, networkParams["zSize"]], 0, 1, dtype=tf.float32)
    z = z_Mean + (z_Stddev * randomSample)

    l4, l5, l6, finalLayer = decoder(z, args, batchSize)

    flattenedImage = tf.reshape(x, [-1, args["x_Dimension"]*args["y_Dimension"]*args["z_Dimension"]])
    flattenedG = tf.reshape(finalLayer, [-1, args["x_Dimension"]*args["y_Dimension"]*args["z_Dimension"]])

    # KL divergence function
    latentLoss = -0.5 * tf.reduce_sum(1 + tf.log(tf.square(z_Stddev) + 1e-4) - tf.square(z_Mean) - tf.square(z_Stddev), 1)

    # Mean squared error (euclidean distance) function
    # generatorLoss = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(flattenedG, flattenedImage)), 1))

    # Entropy function
    generatorLoss = tf.reduce_sum(flattenedImage * tf.log(flattenedG + 1e-4) + (1 - flattenedImage) * tf.log(1 - flattenedG + 1e-4), 1)

    overallLoss = tf.reduce_sum(latentLoss + generatorLoss)
    return l1, l2, l3, l4, l5, l6, finalLayer, overallLoss

def encoder(x, args):
    x = tf.reshape(x, [-1, args["x_Dimension"], args["y_Dimension"], args["z_Dimension"], 1])
    l1, l1_Forward = conv3d(x, weights["wc1"], biases["bc1"])
    l2, l2_Forward = conv3d(l1_Forward, weights["wc2"], biases["bc2"])
    l3, l3_Forward = conv3d(l2_Forward, weights["wc3"], biases["bc3"])

    return l1, l2, l3, slim.fully_connected(slim.flatten(l3_Forward), networkParams["zSize"], activation_fn=None), slim.fully_connected(slim.flatten(l3_Forward), networkParams["zSize"], activation_fn=None)

def decoder(z, args, batchSize):
    decodeLayer_OutputShape = [math.ceil(args["x_Dimension"]/8), math.ceil(args["y_Dimension"]/8), math.ceil(args["z_Dimension"]/8), networkParams["nFilter3"]]
    decodeLayer = tf.reshape(slim.fully_connected(z, int(np.prod(decodeLayer_OutputShape)), activation_fn=None), [-1, math.ceil(args["x_Dimension"]/8), math.ceil(args["y_Dimension"]/8), math.ceil(args["z_Dimension"]/8), 8])

    l4_OutputShape = [batchSize, math.ceil(args["x_Dimension"]/4), math.ceil(args["y_Dimension"]/4), math.ceil(args["z_Dimension"]/4), networkParams["nFilter2"]]
    l4, l4_Forward = conv3d_transpose(decodeLayer, weights["wdc1"], biases["bdc1"], l4_OutputShape)

    l5_OutputShape = [batchSize, math.ceil(args["x_Dimension"]/2), math.ceil(args["y_Dimension"]/2), math.ceil(args["z_Dimension"]/2), networkParams["nFilter1"]]
    l5, l5_Forward = conv3d_transpose(l4_Forward, weights["wdc2"], biases["bdc2"], l5_OutputShape)

    l6_OutputShape = [batchSize, math.ceil(args["x_Dimension"]), math.ceil(args["y_Dimension"]), math.ceil(args["z_Dimension"]), networkParams["nFilter0"]]
    l6, l6_Forward = conv3d_transpose(l5_Forward, weights["wdc3"], biases["bdc3"], l6_OutputShape, activation_fn="sigmoid")
    l6_Forward = tf.reshape(l6_Forward, [-1, args["x_Dimension"], args["y_Dimension"], args["z_Dimension"]])

    return l4, l5, l6, tf.nn.dropout(l6_Forward, args["dropout"])
