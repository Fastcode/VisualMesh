#!/usr/bin/env python3

import os
import re
import math
import json
import rdp
import yaml
import numpy as np
import tensorflow as tf
from tqdm import tqdm, trange

from . import dataset
from . import network


# Train the network
def test(config, output_path):

  # Extract configuration variables
  test_files = config['dataset']['test']
  geometry = config['geometry']
  classes = config['network']['classes']
  structure = config['network']['structure']
  batch_size = config['testing']['batch_size']
  tutor_structure = config['training']['tutor'].get('structure', None)

  # Build our student and tutor networks
  net = network.build(structure, len(classes), structure if tutor_structure is None else tutor_structure)

  # Truth labels for the network
  Y = net['Y']

  # The unscaled network output
  X = net['mesh']

  # The alpha channel from the training data to remove unlabelled points from consideration
  W = net['W']
  S = tf.where(tf.greater(W, 0))
  Y = tf.gather_nd(Y, S)
  X = tf.gather_nd(X, S)

  # Softmax to get actual output
  X = tf.nn.softmax(X, axis=1)

  # This dictionary will store our test queries
  test = {}

  # Get tp/fp per class
  for i, c in enumerate(classes):
    test.update({c[0]: {'X': X[:, i], 'tp': Y[:, i] > 0}})

  # Tensorflow configuration
  tf_config = tf.ConfigProto()
  tf_config.allow_soft_placement = True
  tf_config.graph_options.build_cost_model = 1
  tf_config.gpu_options.allow_growth = True

  with tf.Session(config=tf_config) as sess:
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
      batch_size=batch_size,
      shuffle_size=0,
      variants={},
    ).build().make_one_shot_iterator().string_handle()

    # Count how many files are in the test dataset so we can show a progress bar
    # This is slow, but the progress bar is comforting
    print('Counting records in test dataset')
    num_records = sum(1 for _ in tf.python_io.tf_record_iterator(test_files))
    print('{} records in dataset in {} batches'.format(num_records, num_records // batch_size))

    # Get our iterator handle
    data_handle = sess.run(training_dataset)

    # There can be huge amounts of output data in these, open files to store the data
    # to avoid exhausting the systems ram
    files = {k: open(os.path.join(output_path, '{}.bin'.format(k)), 'wb') for k in test}

    # Process all the results
    for i in trange(num_records // batch_size, unit='batch'):
      results = sess.run(test, feed_dict={net['handle']: data_handle})

      # Write all the results to file
      for k, v in results.items():
        n = np.empty(len(v['X']), dtype=[('X', np.float32), ('tp', np.bool)])
        n['X'] = v['X']
        n['tp'] = v['tp']
        n.tofile(files[k])

    # Close all the files
    for k, v in files.items():
      v.close()

    # Process each of the class files
    results = {}
    for c in classes:
      print('Processing class {}'.format(c[0]))

      # Load the file we just made back into memory
      data = np.memmap(
        os.path.join(output_path, '{}.bin'.format(c[0])),
        dtype=[('c', np.float32), ('tp', np.bool)],
        mode='c',
      )

      # Sort the values by their confidence level
      print('\tSorting {} values by confidence (this may take a while)'.format(len(data)))
      data.sort(order='c')

      # Total number of positive and negative values
      p = np.sum(data['tp'], dtype=np.int64)
      n = len(data) - p

      # Cumulative sum the true and false positives
      print('\tThresholding true positives')
      tp = np.cumsum(data['tp'][::-1], dtype=np.int64)[::-1]
      print('\tThresholding false positives')
      fp = np.cumsum(np.logical_not(data['tp'])[::-1], dtype=np.int64)[::-1]

      # Calculate precision/recall
      print('\tCalculating precision')
      precision = np.true_divide(tp, (tp + fp))
      print('\tCalculating recall')
      recall = np.true_divide(tp, p)

      # Now for a precision recall curve, we don't need billions of points
      # Reducing using normal line reduction techniques is very slow so instead we just
      # reduce the line to a grid of points, averaging all points within the range
      reduced_pr = [np.array([data['c'][0], precision[0], recall[0]])]
      last_i = 0
      epsilon = 1 / 10000
      print('\tReducing PR Curve complexity')
      for i in trange(len(precision)):
        if np.linalg.norm(reduced_pr[-1][1:] - np.array([precision[i], recall[i]])) > epsilon:
          reduced_pr.append(
            np.array([
              np.mean(data['c'][last_i:i]),
              np.mean(precision[last_i:i]),
              np.mean(recall[last_i:i]),
            ])
          )
          last_i = i

      # Work out precision at lowest recall
      lowest_rec = 1
      prec_at_lowest_rec = 1
      for p in reduced_pr:
        if p[2] < lowest_rec:
          lowest_rec = p[2]
          prec_at_lowest_rec = p[1]

      # Add on the ends of the pr curve
      reduced_pr.append(np.array([0, 0, 1]))
      reduced_pr.append(np.array([1, prec_at_lowest_rec, 0]))

      # Sort by recall
      reduced_pr = np.array(reduced_pr)
      reduced_pr = reduced_pr[np.argsort(reduced_pr[:, 2])]

      # Calculate the average precision using this curve
      ap = np.trapz(reduced_pr[:, 1], reduced_pr[:, 2])

      # Do a proper line reduction using Ramer-Douglas-Peucker algorithm
      mask = np.where(rdp.rdp(reduced_pr[:, 1:3], epsilon=epsilon, return_mask=True))
      reduced_pr = reduced_pr[mask]

      # Store the results
      results[c[0]] = {'ap': ap, 'pr': reduced_pr}

    with open('ap.txt', 'w') as apf:
      mean_ap = 0
      for k, v in results.items():
        mean_ap += v['ap']

        print('{} Average Precision: {}'.format(k.title(), v['ap']))
        apf.write('{} {}\n'.format(k, v['ap']))

        np.savetxt(
          '{}_pr.csv'.format(k),
          v['pr'],
          comments='',
          header='Confidence,Precision,Recall',
          delimiter=',',
        )

      mean_ap = mean_ap / len(results)

      print('Mean Average Precision: {}'.format(mean_ap))
      apf.write('mAP {}\n'.format(k, mean_ap))
