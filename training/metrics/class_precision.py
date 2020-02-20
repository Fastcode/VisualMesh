#!/usr/bin/env python3

import tensorflow as tf

from .confusion_base import ConfusionBase


class ClassPrecision(ConfusionBase):
    def __init__(self, idx, name, **kwargs):
        super(ClassPrecision, self).__init__(idx, name=name, **kwargs)

    def result(self):
        return self.tp / (self.tp + self.fp)
