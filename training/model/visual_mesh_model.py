#!/usr/bin/env python3

import tensorflow as tf
from training.layer import GraphConvolution


class VisualMeshModel(tf.keras.Model):

  def __init__(self, structure, n_classes):
    super(MyModel, self).__init__()

    self.layers = []

    # Build our network structure
    for i, c in enumerate(structure):

      # Graph convolution
      self.layers.append(GraphConvolution())

      # Dense internal layers
      for j, units in enumerate(c):
        self.layers.append(tf.keras.layers.Dense(units=units))

    # Final dense layer for the number of classes
    self.layers.append(tf.keras.layers.Dense(units=n_classes, activation=tf.nn.softmax))

  def call(self, X, G):

    logits = X

    # Run through each of our layers in sequence
    for l in self.layers:
      if isinstance(l, GraphConvolution):
        logits = l(logits, G)
      elif isinstance(l, tf.keras.layers.Dense):
        logits = l(logits)

    return logits
