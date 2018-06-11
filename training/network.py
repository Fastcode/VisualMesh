#!/usr/bin/env python3

import math
import copy
import tensorflow as tf


def build_network(X, G, groups):

  # Build our tensor
  logits = X
  for i, c in enumerate(groups):

    # Which convolution we are on
    with tf.variable_scope('Conv_{}'.format(i)):

      # The size of the previous output is the size of our input
      prev_last_out = logits.shape[1].value

      with tf.variable_scope('GatherConvolution'):
        # Gather our pixel coordinates based on the graph and flatten them into features
        # Don't worry about the tensorflow warning this creates, this tensor is better as a dense tensor anyway
        logits = tf.reshape(tf.gather(logits, G, name='NetworkGather'), shape=[-1, prev_last_out * G.shape[1].value])

      for j, out_s in enumerate(c):

        # Our input size is the previous output size
        in_s = logits.get_shape()[1].value

        # Which layer we are on
        with tf.variable_scope('Layer_{}'.format(j)):

          # Create weights and biases
          W = tf.get_variable(
            'Weights',
            shape=[in_s, out_s],
            initializer=tf.random_normal_initializer(stddev=math.sqrt(1.0 / in_s)),
            dtype=tf.float32
          )

          b = tf.get_variable(
            'Biases',
            shape=[out_s],
            initializer=tf.random_normal_initializer(stddev=math.sqrt(1.0 / out_s)),
            dtype=tf.float32
          )

          # Apply our weights and biases
          logits = tf.matmul(logits, W)
          logits = tf.add(logits, b)

          # Apply our activation function except for the last layer
          if i + 1 < len(groups) or j + 1 < len(c):
            logits = tf.nn.selu(logits)

  return logits


def build(groups, n_classes, adversary_groups):

  # The last layer is the number of classes we have but the adversary only has two
  mesh_groups = copy.deepcopy(groups)
  mesh_groups[-1].append(n_classes)

  # Multiply the size of the adversary layers
  adversary_groups = copy.deepcopy(adversary_groups)
  adversary_groups[-1].append(1)

  # Number of neighbours for each point
  graph_degree = 7

  # Our iterator handle
  handle = tf.placeholder(tf.string, shape=[])

  # Make our iterator
  iterator = tf.data.Iterator.from_string_handle(
    handle, {
      'X': tf.float32,
      'Y': tf.float32,
      'G': tf.int32,
      'W': tf.float32,
      'n': tf.int32,
      'px': tf.int32,
      'raw': tf.string,
    }, {
      'X': [None, 3],
      'Y': [None, n_classes],
      'G': [None, graph_degree],
      'W': [None],
      'n': [None],
      'px': [None, None, 2],
      'raw': [None],
    }
  )

  # Get values from our iterator
  data = iterator.get_next()
  X = data['X']
  G = data['G']

  # Build our normal mesh, and our adverserial mesh
  with tf.variable_scope('mesh'):
    mesh = build_network(X, G, mesh_groups)
  with tf.variable_scope('adversary'):
    adversary = build_network(X, G, adversary_groups)

  return {
    'handle': handle,  # (input)  Iterator handle
    'mesh': mesh,  # (output) Raw unscaled mesh output
    'adversary': adversary,  # (output) Raw unscaled weights output
    **data,  # Forward all the data from our iterator
  }
