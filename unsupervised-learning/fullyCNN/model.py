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
    l1, l2, l3, l3_Forward = encoder(x, args)
    batchSize = tf.shape(x)[0]
    l4, l5, l6, finalLayer = decoder(l3_Forward, args, batchSize)
    loss = tf.nn.softmax_cross_entropy_with_logits(finalLayer, x)
    return l1, l2, l3, l4, l5, l6, finalLayer, loss

def encoder(x, args):
    x = tf.reshape(x, [-1, args["x_Dimension"], args["y_Dimension"], args["z_Dimension"], 1])
    l1, l1_Forward = conv3d(x, weights["wc1"], biases["bc1"])
    l2, l2_Forward = conv3d(l1_Forward, weights["wc2"], biases["bc2"])
    l3, l3_Forward = conv3d(l2_Forward, weights["wc3"], biases["bc3"])
    return l1, l2, l3, l3_Forward

def decoder(l3, args, batchSize):
    l4_OutputShape = [batchSize, math.ceil(args["x_Dimension"]/4), math.ceil(args["y_Dimension"]/4), math.ceil(args["z_Dimension"]/4), networkParams["nFilter2"]]
    l4, l4_Forward = conv3d_transpose(l3, weights["wdc1"], biases["bdc1"], l4_OutputShape)

    l5_OutputShape = [batchSize, math.ceil(args["x_Dimension"]/2), math.ceil(args["y_Dimension"]/2), math.ceil(args["z_Dimension"]/2), networkParams["nFilter1"]]
    l5, l5_Forward = conv3d_transpose(l4_Forward, weights["wdc2"], biases["bdc2"], l5_OutputShape)

    l6_OutputShape = [batchSize, math.ceil(args["x_Dimension"]), math.ceil(args["y_Dimension"]), math.ceil(args["z_Dimension"]), networkParams["nFilter0"]]
    l6, l6_Forward = conv3d_transpose(l5_Forward, weights["wdc3"], biases["bdc3"], l6_OutputShape, activation_fn=None)
    l6_Forward = tf.reshape(l6_Forward, [-1, args["x_Dimension"], args["y_Dimension"], args["z_Dimension"]])

    return l4, l5, l6, tf.nn.dropout(l6_Forward, args["dropout"])
