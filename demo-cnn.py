'''
demo-cnn.py
from https://github.com/aymericdamien/TensorFlow-Examples
'''

from __future__ import print_function

import tensorflow as tf
import tifffile as tiff
import numpy as np
import time

# Import volume data
import trainvol
start_t = time.time()
CUT_IN = 20
CUT_BACK = 8
NR = 14
THRESH = 20500
RSZ = 2

folder = 'small-fragment-data/'
train_vol = trainvol.TrainVol(
        folder+'volume.npy', folder+'volume-truth.npy', CUT_IN, CUT_BACK, NR, THRESH, RSZ)
train_vol.make_train_with_style('random', prob=.9)
#train_vol.make_train_with_style('rhalf')

# Parameters
learning_rate = 0.001
training_iters = 500000
batch_size = 128
display_step = 10
pred_display_step = 100
dropout = .8
PARAMS = 'lr{}-iters{}-batch{}-drop{}-in{}-back{}-NR{}-thresh{}-resize{}'.format(
        learning_rate, training_iters, batch_size, dropout, CUT_IN, CUT_BACK, NR, THRESH, RSZ)

# Network Parameters
n_input = ((CUT_IN + CUT_BACK) * (2*NR)) # data input size
n_classes = 2 # total classes

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, (CUT_IN+CUT_BACK), 2*NR, 1])
    #pdb.set_trace()

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Convolution Layer
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # Max Pooling (down-sampling)
    conv3 = maxpool2d(conv3, k=2)

    conv3 = tf.nn.relu(conv3)
    conv3 = tf.nn.dropout(conv3, dropout)

    # Output, class prediction
    # reshape to fit the layer
    print("\nconv3: {} \nout: {}".format(tf.shape(conv3), tf.shape(weights['out'])))
    conv3 = tf.reshape(conv3, [-1, weights['out'].get_shape().as_list()[0]])
    print("After reshape:\nconv3: {} \nout: {}".format(tf.shape(conv3), tf.shape(weights['out'])))
    out = tf.add(tf.matmul(conv3, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 16])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 16, 32])),
    # another
    'wc3': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # output layer
    'out': tf.Variable(tf.random_normal([4*4*64, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([16])),
    'bc2': tf.Variable(tf.random_normal([32])),
    'bc3': tf.Variable(tf.random_normal([64])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = train_vol.next_batch(batch_size)
        batch_x = np.reshape(batch_x, (-1, n_input))
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob:dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
            print("{} ink samples in batch".format(np.count_nonzero(batch_y[:,0])))
        step += 1
    print("Optimization Finished!")

    all_samples = train_vol.get_predict_batch()
    all_samples = np.reshape(all_samples, (-1, n_input))
    prediction_rescale = 1
    rows,cols = train_vol.get_output_shape()
    n_rows = int(rows / prediction_rescale)
    n_cols = int(cols / prediction_rescale)
    output_pic = np.zeros((n_rows, n_cols), dtype=np.float64)
    done_count = 0
    '''
    out_count = output_pic.shape[0] * output_pic.shape[1]
    print("making predictions")
    for i in range(1, output_pic.shape[0]):
        for j in range(1, output_pic.shape[1]):
            r_i = np.random.randint((i-1) * prediction_rescale, i * prediction_rescale)
            r_j = np.random.randint((j-1) * prediction_rescale, j * prediction_rescale)
            ind = train_vol.coord_to_ind(r_i, r_j)
            to_predict = np.expand_dims(all_samples[ind], axis=0)
            output_pic[i,j] = sess.run(tf.nn.softmax(pred),
            #output_pic[i,j] = sess.run((pred),
                    feed_dict = {x:to_predict, keep_prob:1.})[:,1]
            done_count += 1
            if (done_count % pred_display_step == 0):
                print("predicted {} of {}".format(done_count, out_count))

    '''
    print("all_feats has {} samples".format(len(all_samples)))
    predictions = np.zeros(all_samples.shape[0])
    prediction_batches = int(len(all_samples) / batch_size) - 1
    for i in range(0, prediction_batches):
        predictions[(i*batch_size):((i+1)*batch_size)] = sess.run(tf.nn.softmax(pred), feed_dict={
                    x:all_samples[(i*batch_size):((i+1)*batch_size)],
                    keep_prob: 1.})[:,1]
        if (done_count % pred_display_step == 0):
            print("predicting round {} of {}".format(i, prediction_batches))

    p_min = np.min(predictions)
    p_max = np.max(predictions)
    print("prediction range: {} - {}".format(p_min, p_max))
    predictions = ((predictions - p_min) / (p_max - p_min) * 65535).astype(np.uint16)
    for i in range(0,n_rows):
        output_pic[i,:] = predictions[i*n_cols:(i+1)*n_cols]
    tiff.imsave('predictions/prediction-{}.tif'.format(PARAMS), output_pic.astype(np.uint16))


    p_min = np.min(output_pic)
    p_max = np.max(output_pic)
    print("prediction range: {} - {}".format(p_min, p_max))
    #output_pic_a = output_pic.astype(np.uint16)
    #output_pic_b = ((output_pic - p_min) / (p_max - p_min) * 65535).astype(np.uint16)
    #output_pic_c = (np.nan_to_num(output_pic - p_min) / (p_max - p_min) * 65535).astype(np.uint16)
    #print("{} ink predictions".format(np.count_nonzero(np.where(output_pic_c > (65535 / 2)))))

    #tiff.imsave('predictions/prediction-a-{}.tif'.format(PARAMS), output_pic_a)
    #tiff.imsave('predictions/prediction-b-{}.tif'.format(PARAMS), output_pic_b)
    #tiff.imsave('predictions/prediction-{}.tif'.format(PARAMS), output_pic_c)


print("full script took {:.2f} seconds".format(time.time() - start_t))
