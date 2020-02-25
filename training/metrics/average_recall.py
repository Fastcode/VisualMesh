#!/usr/bin/env python3

import tensorflow as tf

from .confusion_base import ConfusionBase


class AverageRecall(ConfusionBase):
    def __init__(self, name, size, **kwargs):
        super(AverageRecall, self).__init__(name, size, **kwargs)

    def result(self):
        # True positives (predicted and labelled true)
        tp = tf.cast(tf.linalg.diag(self.confusion), self.dtype)
        # For all predictions where idx was labelled
        tp_fn = tf.cast(tf.reduce_sum(self.confusion, axis=1), self.dtype)

        return tf.math.reduce_mean(tp / tp_fn)
