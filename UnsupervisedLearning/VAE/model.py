import tensorflow as tf
import tensorflow.contrib.slim as slim
import pdb
import math
import numpy as np

from ops import *

n_z = 20

weights = {
    'wc1' : tf.get_variable("weights_1", shape=[3, 3, 3, 1, 8],
               initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),

    'wc2' : tf.get_variable("weights_2", shape=[3, 3, 3, 8, 16],
               initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),

    'wc3' : tf.get_variable("weights_3", shape=[3, 3, 3, 16, 32],
               initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),

    'wdc1' : tf.get_variable("weights_4", shape=[4, 4, 4, 16, 32],
               initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),

    'wdc2' : tf.get_variable("weights_5", shape=[4, 4, 4, 8, 16],
               initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),

    'wdc3' : tf.get_variable("weights_6", shape=[4, 4, 4, 1, 8],
               initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
}

biases = {
    'bc1': tf.Variable(tf.zeros([8], dtype=tf.float32), name="biases_1", dtype=tf.float32),
    'bc2': tf.Variable(tf.zeros([16], dtype=tf.float32), name="biases_2", dtype=tf.float32),
    'bc3': tf.Variable(tf.zeros([32], dtype=tf.float32), name="biases_3", dtype=tf.float32),
    'bdc1': tf.Variable(tf.zeros([16], dtype=tf.float32), name="biases_4", dtype=tf.float32),
    'bdc2': tf.Variable(tf.zeros([8], dtype=tf.float32), name="biases_5", dtype=tf.float32),
    'bdc3': tf.Variable(tf.zeros([1], dtype=tf.float32), name="biases_6", dtype=tf.float32),
}

def buildModel(x, args):
    l1, l2, l3, z_Mean, z_Stddev = encoder(x, args)
    batchSize = tf.shape(x)[0]
    randomSample = tf.random_normal([batchSize, n_z], 0, 1, dtype=tf.float32)
    z = z_Mean + (z_Stddev * randomSample)

    l4, l5, l6 = decoder(z, args, batchSize)

    flattenedImage = tf.reshape(x, [-1, args["x_Dimension"]*args["y_Dimension"]*args["z_Dimension"]])
    flattenedG = tf.reshape(l6, [-1, args["x_Dimension"]*args["y_Dimension"]*args["z_Dimension"]])

    latentLoss = 0.5 * tf.reduce_sum(tf.square(z_Mean) + tf.square(z_Stddev) - tf.log(tf.square(z_Stddev)) - 1,1)
    generatorLoss = -tf.reduce_sum(flattenedImage * tf.log(1e-8 + flattenedG) + (1-flattenedImage) * tf.log(1e-8 + 1 - flattenedG),1)
    overallLoss = latentLoss + generatorLoss

    return l1, l2, l3, l4, l5, l6, overallLoss

def encoder(x, args):
    x = tf.reshape(x, [-1, args["x_Dimension"], args["y_Dimension"], args["z_Dimension"], 1])
    l1 = conv3d(x, weights["wc1"], biases["bc1"])
    l2 = conv3d(l1, weights["wc2"], biases["bc2"])
    l3 = conv3d(l2, weights["wc3"], biases["bc3"])

    l3_Flattened = tf.reshape(l3, [-1, l3.get_shape().as_list()[1] * l3.get_shape().as_list()[2] * l3.get_shape().as_list()[3] * l3.get_shape().as_list()[4]])
    w_Mean = slim.fully_connected(l3_Flattened, 20, activation_fn=None)
    w_Stddev = slim.fully_connected(l3_Flattened, 20, activation_fn=None)

    return l1, l2, l3, w_Mean, w_Stddev

def decoder(z, args, batchSize):
    decodeLayer_OutputShape = [math.ceil(args["x_Dimension"]/8), math.ceil(args["y_Dimension"]/8), math.ceil(args["z_Dimension"]/8), 32]
    decodeLayer = tf.reshape(slim.fully_connected(z, int(np.prod(decodeLayer_OutputShape))), [-1, math.ceil(args["x_Dimension"]/8), math.ceil(args["y_Dimension"]/8), math.ceil(args["z_Dimension"]/8), 32])

    l4_OutputShape = [batchSize, math.ceil(args["x_Dimension"]/4), math.ceil(args["y_Dimension"]/4), math.ceil(args["z_Dimension"]/4), 16]
    l4 = conv3d_transpose(decodeLayer, weights["wdc1"], biases["bdc1"], l4_OutputShape)

    l5_OutputShape = [batchSize, math.ceil(args["x_Dimension"]/2), math.ceil(args["y_Dimension"]/2), math.ceil(args["z_Dimension"]/2), 8]
    l5 = conv3d_transpose(l4, weights["wdc2"], biases["bdc2"], l5_OutputShape)

    l6_OutputShape = [batchSize, math.ceil(args["x_Dimension"]), math.ceil(args["y_Dimension"]), math.ceil(args["z_Dimension"]), 1]
    l6 = conv3d_transpose(l5, weights["wdc3"], biases["bdc3"], l6_OutputShape, activation_fn="sigmoid")
    l6 = tf.reshape(l6, [-1, args["x_Dimension"], args["y_Dimension"], args["z_Dimension"]])

    return l4, l5, tf.nn.dropout(l6, args["dropout"])
