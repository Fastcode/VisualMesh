#!/usr/bin/env python3

import math
import tensorflow as tf


def build(groups, n_classes):

  # The last layer is the number of classes we have
  groups[-1].append(n_classes)

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
    }, {
      'X': [None, 3],
      'Y': [None, n_classes],
      'G': [None, graph_degree],
      'W': [None],
    }
  )

  # Get values from our iterator
  data = iterator.get_next()
  X = data['X']
  G = data['G']

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

          # Apply our activation function
          logits = tf.nn.selu(logits)

  # Softmax our final output for all values in the mesh as our network
  network = tf.nn.softmax(logits, name='Softmax', axis=1)

  return {
    'handle': handle,  # (input)  Iterator handle
    'logits': logits,  # (output) Raw unscaled output
    'network': network,  # (output) network output
    **data,  # Forward all the data from our iterator
  }
