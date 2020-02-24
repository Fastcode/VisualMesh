#!/usr/bin/env python3

import tensorflow as tf


class ConfusionBase(tf.keras.metrics.Metric):
    def __init__(self, name, size, **kwargs):
        super(ConfusionBase, self).__init__(name=name, **kwargs)

        self.confusion = self.add_weight(name="confusion", shape=(size, size), initializer="zeros", dtype=tf.int32)

    def update_state(self, y_true, y_pred, sample_weight=None):

        # Build up an index list that maps each class
        idx = tf.stack([tf.argmax(input=y_true, axis=-1), tf.argmax(input=y_pred, axis=-1)], axis=-1)

        # Trim down the indexes to only those that have a class label
        idx = tf.gather(idx, tf.squeeze(tf.where(tf.reduce_any(tf.greater(y_true, 0.0), axis=-1)), axis=-1))

        # Add them into the corresponding locations
        self.confusion.scatter_nd_add(idx, tf.ones_like(idx[:, 0], dtype=self.confusion.dtype))

    def reset_states(self):
        self.confusion.assign(tf.zeros_like(self.confusion))
