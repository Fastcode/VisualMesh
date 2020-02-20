#!/usr/bin/env python3

import tensorflow as tf


class GraphConvolution(tf.keras.layers.Layer):
    def __init__(self):
        super(GraphConvolution, self).__init__()

    def call(self, X, G):
        return tf.reshape(tf.gather(X, G, name="NetworkGather"), shape=[-1, X.shape[-1] * G.shape[-1]])
