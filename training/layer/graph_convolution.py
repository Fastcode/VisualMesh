#!/usr/bin/env python3

import tensorflow as tf


class GraphConvolution(tf.keras.layers.Layer):

  def __init__(self):
    super(GraphConvolution, self).__init__()

  def build(self):
    super(GraphConvolution, self).build()

  def call(self, inputs):
    logits, G = inputs
    return tf.reshape(tf.gather(logits, G, name='NetworkGather'), shape=[-1, tf.shape(logits)[1] * tf.shape(G)[1]])
