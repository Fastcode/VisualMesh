#!/usr/bin/env python3

import tensorflow as tf
from training.layer import GraphConvolution


class VisualMeshModel(tf.keras.Model):

  def __init__(self, structure, n_classes, activation):
    super(VisualMeshModel, self).__init__()

    self.stages = []

    # Build our network structure
    for i, c in enumerate(structure):

      # Graph convolution
      self.stages.append(GraphConvolution())

      # Dense internal layers
      for j, units in enumerate(c):
        self.stages.append(tf.keras.layers.Dense(units=units, activation=activation))

    # Final dense layer for the number of classes
    self.stages.append(tf.keras.layers.Dense(units=n_classes, activation=tf.nn.softmax))

  def call(self, X, training=False):

    # Split out the graph and logits
    logits, G = X

    # Run through each of our layers in sequence
    for l in self.stages:
      if isinstance(l, GraphConvolution):
        logits = l(logits, G)
      elif isinstance(l, tf.keras.layers.Dense):
        logits = l(logits)

    # At the very end of the network, we remove the offscreen point
    logits = logits[:-1]

    return logits
