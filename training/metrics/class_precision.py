#!/usr/bin/env python3

import tensorflow as tf

from .confusion_base import ConfusionBase


class ClassPrecision(ConfusionBase):
    def __init__(self, name, idx, size, **kwargs):
        super(ClassPrecision, self).__init__(name, size, **kwargs)
        self.idx = idx

    def result(self):
        # True positives (predicted and labelled true)
        tp = tf.cast(self.confusion[self.idx, self.idx], self.dtype)
        # For all labels where idx was predicted (all positives)
        p = tf.cast(tf.reduce_sum(self.confusion[:, self.idx]), self.dtype)

        return tp / p
