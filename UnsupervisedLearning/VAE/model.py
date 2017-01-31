import tensorflow as tf
import tensorflow.contrib.slim as slim
from ops import *
import pdb

n_z = 20

weights = {
    'wc1' : tf.get_variable("weights_1_cnn_same", shape=[3, 3, 3, 1, 32],
               initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),

    'wc2' : tf.get_variable("weights_2_cnn_same", shape=[3, 3, 3, 32, 64],
               initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),

    'wc3' : tf.get_variable("weights_3_cnn_same", shape=[3, 3, 3, 64, 128],
               initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),

    'wdc1' : tf.get_variable("weights_4_cnn_same", shape=[3, 3, 3, 128, 256],
               initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),

    'wdc2' : tf.get_variable("weights_5_cnn_same", shape=[3, 3, 3, 64, 128],
               initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
}

biases = {
    'bc1': tf.Variable(tf.zeros([32], dtype=tf.float32), name="biases_1_cnn_same", dtype=tf.float32),
    'bc2': tf.Variable(tf.zeros([64], dtype=tf.float32), name="biases_2_cnn_same", dtype=tf.float32),
    'bc3': tf.Variable(tf.zeros([128], dtype=tf.float32), name="biases_3_cnn_same", dtype=tf.float32),
    'bdc1': tf.Variable(tf.zeros([128], dtype=tf.float32), name="biases_4_cnn_same", dtype=tf.float32),
    'bdc2': tf.Variable(tf.zeros([64], dtype=tf.float32), name="biases_5_cnn_same", dtype=tf.float32),
}

def buildModel(x, args):
    l1, l2, l3, z_Mean, z_Stddev = encoder(x)
    randomSample = tf.random_normal([args["batchSize"], n_z], 0, 1, dtype=tf.float32)
    z = z_Mean + (z_Stddev * randomSample)

    l4, l5, l6 = decoder(z, args)

    flattenedImage = tf.reshape(x, [-1])
    flattenedG = tf.reshape(l6, [-1])

    latentLoss = 0.5 * tf.reduce_sum(tf.square(z_Mean) + tf.square(z_Stddev) - tf.log(tf.square(z_Stddev)) - 1,1)
    generatorLoss = -tf.reduce_sum(flattenedImage * tf.log(1e-8 + flattenedG) + (1-flattenedImage) * tf.log(1e-8 + 1 - flattenedG),1)

    overallLoss = latentLoss + generatorLoss

    return l1, l2, l3, l4, l5, l6, overallLoss

def encoder(x):
    pdb.set_trace()
    l1 = conv3d(x, weights["wc1"], biases["bc1"])
    l2 = conv3d(l1, weights["wc2"], biases["bc2"])
    l3 = conv3d(l2, weights["wc3"], biases["bc3"])

    l3_Flattened = tf.reshape(l3, [-1])
    w_Mean = slim.fully_connected(l3_Flattened, 20, activation_fn=None)
    w_Stddev = slim.fully_connected(l3_Flattened, 20, activation_fn=None)

    return l1, l2, l3, w_Mean, w_Stddev

def decoder(z, args):
    pdb.set_trace()
    l4 = tf.reshape(slim.fully_connected(z, 7*7*32), [-1])
    l5 = conv3d_transpose(l4, weights["wdc1"], biases["bdc1"], [-1])
    l6 = conv3d_transpose(l5, weights["wdc2"], biased["bdc2"], [-1], activation_fn="sigmoid")
    return l4, l5, tf.nn.dropout(l6, args["dropout"])
