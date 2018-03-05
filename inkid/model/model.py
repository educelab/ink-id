"""
Functions for building the tf model.
"""
from functools import partial

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow import layers


class Model3dcnn(tf.keras.Model):
    def __init__(self, drop_rate, subvolume_shape, batch_norm_momentum, filters):
        self._drop_rate = drop_rate
        self._subvolume_shape = subvolume_shape
        self._filters = filters
        self._input_shape = [-1, subvolume_shape[0], subvolume_shape[1], subvolume_shape[2], 1]

        self.batch_norm1 = layers.BatchNormalization(
            scale=False, axis=4, momentum=batch_norm_momentum)
        self.batch_norm2 = layers.BatchNormalization(
            scale=False, axis=4, momentum=batch_norm_momentum)
        self.batch_norm3 = layers.BatchNormalization(
            scale=False, axis=4, momentum=batch_norm_momentum)
        self.batch_norm4 = layers.BatchNormalization(
            scale=False, axis=4, momentum=batch_norm_momentum)
        self.conv3d = partial(
            slim.convolution, kernel_size=[3, 3, 3], stride=[2, 2, 2], padding='valid')

    def __call__(self, inputs, training):
        y = tf.reshape(inputs, self._input_shape)
        y = self.batch_norm1(self.conv3d(y, num_outputs=self._filters[0]), training=training)
        y = self.batch_norm2(self.conv3d(y, num_outputs=self._filters[1]), training=training)
        y = self.batch_norm3(self.conv3d(y, num_outputs=self._filters[2]), training=training)
        y = self.batch_norm4(self.conv3d(y, num_outputs=self._filters[3]), training=training)
        y = layers.dropout(slim.fully_connected(slim.flatten(y), 2, activation_fn=None),
                           rate=self._drop_rate)
        return y


def model_fn_3dcnn(features, labels, mode, params):
    model = Model3dcnn(params['drop_rate'],
                       params['subvolume_shape'],
                       params['batch_norm_momentum'],
                       params['filters'])
    
    subvolume = features
    if isinstance(subvolume, dict):
        subvolume = features['Subvolume']

    if mode == tf.estimator.ModeKeys.PREDICT:
        logits = model(subvolume, training=False)
        predictions = {
            'classes': tf.argmax(logits, axis=1),
            'probabilities': tf.nn.softmax(logits),
        }
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=predictions,
            export_outputs={
                'classify': tf.estimator.export.PredictOutput(predictions)
            })

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
        logits = model(subvolume, training=True)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=labels, logits=logits))
        accuracy = tf.metrics.accuracy(
            labels=tf.argmax(labels, axis=1), predictions=tf.argmax(logits, axis=1))
        tf.identity(accuracy[1], name='train_accuracy')
        tf.summary.scalar('train_accuracy', accuracy[1])
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step()))

    if mode == tf.estimator.ModeKeys.EVAL:
        logits = model(subvolume, training=False)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=labels, logits=logits))
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=loss,
            eval_metric_ops={
                # TODO add F1 and precision
                'accuracy': tf.metrics.accuracy(
                    labels=labels,
                    predictions=tf.argmax(logits, axis=1)),
                })
        
