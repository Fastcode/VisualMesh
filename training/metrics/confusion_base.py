#!/usr/bin/env python3

import tensorflow as tf


class ConfusionBase(tf.keras.metrics.Metric):
    def __init__(self, idx, name, **kwargs):
        super(ConfusionBase, self).__init__(name=name, **kwargs)

        self.idx = idx

        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.tn = self.add_weight(name="tn", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):

        # Split the states into boolean values for this class
        predictions = tf.cast(tf.equal(tf.argmax(input=y_pred, axis=-1), self.idx), tf.int32)
        labels = tf.cast(tf.equal(tf.argmax(input=y_true, axis=-1), self.idx), tf.int32)

        self.tp.assign_add(tf.math.count_nonzero(predictions * labels, dtype=tf.float32))
        self.tn.assign_add(tf.math.count_nonzero((predictions - 1) * (labels - 1), dtype=tf.float32))
        self.fp.assign_add(tf.math.count_nonzero(predictions * (labels - 1), dtype=tf.float32))
        self.fn.assign_add(tf.math.count_nonzero((predictions - 1) * labels, dtype=tf.float32))

    def reset_states(self):
        self.tp.assign(0.0)
        self.tn.assign(0.0)
        self.fp.assign(0.0)
        self.fn.assign(0.0)
