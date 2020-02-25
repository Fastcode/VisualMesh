#!/usr/bin/env python3

import tensorflow as tf

from .confusion_base import ConfusionBase


class AveragePrecision(ConfusionBase):
    def __init__(self, name, size, **kwargs):
        super(AveragePrecision, self).__init__(name, size, **kwargs)

    def result(self):
        # True positives (predicted and labelled true)
        tp = tf.cast(tf.linalg.diag_part(self.confusion), self.dtype)
        # For all labels where idx was predicted (all positives)
        p = tf.cast(tf.reduce_sum(self.confusion, axis=0), self.dtype)

        return tf.math.reduce_mean(tp / p)
