#!/usr/bin/env python3

import os
import re
import math
import json
import yaml
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from . import dataset
from . import network


# Train the network
def test(sess, config, output_path):

  # Extract configuration variables
  test_files = config['dataset']['test']
  geometry = config['geometry']
  classes = config['network']['classes']
  structure = config['network']['structure']
  adversary_structure = config['training']['adversary'].get('structure', None)

  # Build our network and adverserial networks
  net = network.build(structure, len(classes), structure if adversary_structure is None else adversary_structure)

  # Truth labels for the network
  Y = network['Y']

  # The unscaled network output
  X = network['mesh']

  # The alpha channel from the training data to remove unlabelled points from consideration
  W = network['W']
  S = tf.where(tf.greater(W, 0))
  Y = tf.gather_nd(Y, S)
  X = tf.gather_nd(X, S)

  # Softmax X
  X = tf.softmax(X, dim=1)

  tf.max(X, axis=1)
  tf.equal(tf.argmax(Y, axis=1), tf.argmax(X, axis=1))

  tp = tf.gather_nd(max_c, tf.where(correct))
  fp = tf.gather_nd(max_c, tf.where(tf.logical_not(correct)))

  # TODO do a 

  # TODO for each class, get tp and fp?
  # test_op = {
  #   'tp':
  #   'fp':
  # }

  # Get the highest class from both to get a tp/fp list
  # argmax(X) == argmax(Y)
  # Get the max value as the classification
  # max()

  # Initialise global variables
  sess.run(tf.global_variables_initializer())

  save_vars = {v.name: v for v in tf.trainable_variables()}
  saver = tf.train.Saver(save_vars)

  # Get our model directory and load it
  checkpoint_file = tf.train.latest_checkpoint(output_path)
  print('Loading model {}'.format(checkpoint_file))
  saver.restore(sess, checkpoint_file)

  # Load our dataset
  training_dataset = dataset.VisualMeshDataset(
    input_files=test_files,
    classes=classes,
    geometry=geometry,
    batch_size=1,
    shuffle_size=0,
    variants={},
  ).build().make_one_shot_iterator().string_handle()

  # Count how many files are in the test dataset so we can show a progress bar
  # This is slow, but the progress bar is comforting
  print('Counting records in test dataset')
  num_records = sum(1 for _ in tf.python_io.tf_record_iterator(test_files))
  print('{} records in dataset'.format(num_records))

  # Get our iterator handle
  data_handle = sess.run(training_dataset)

  import pudb
  pudb.set_trace()

  # with open(os.path.join(output_path, 'tp.bin'), 'w') as tp:

  #   # Loop through the data
  #   while True:
  #     try:

  #       # Get the difference between our labels and our expectations
  #       # for tp, tn, fp, fn, f in zip(*sess.run(confusion, feed_dict={ network['handle']: data_handle })):
  #       for X, Y, f in zip(*sess.run(classification, feed_dict={network['handle']: data_handle})):

  #     except tf.errors.OutOfRangeError:
  #       print('Testing Done')
  #       break
