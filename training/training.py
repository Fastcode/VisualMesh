#!/usr/bin/env python3

import os
import sys
import random
import tensorflow as tf
import yaml
import re

from . import dataset


def save_yaml_model(sess, output_path, global_step):

  # Run tf to get all our variables
  variables = {v.name: sess.run(v) for v in tf.trainable_variables()}
  output = []

  # So we know when to move to the next list
  conv = -1
  layer = -1

  # Convert the keys into useful data
  items = []
  for k, v in variables.items():
    info = re.match(r'Conv_(\d+)/Layer_(\d+)/(Weights|Biases):0', k)
    if info:
      items.append(((int(info.group(1)), int(info.group(2)), info.group(3).lower()), v))

  # Sorted so we see earlier layers first
  for k, v in sorted(items):
    c = k[0]
    l = k[1]
    var = k[2]

    # If we change convolution add a new element
    if c != conv:
      output.append([])
      conv = c
      layer = -1

    # If we change layer add a new object
    if l != layer:
      output[-1].append({})
      layer = l

    output[conv][layer][var] = v.tolist()

  # Print as yaml
  os.makedirs(os.path.join(output_path, 'yaml_models'), exist_ok=True)
  with open(os.path.join(output_path, 'yaml_models', 'model_{}.yaml'.format(global_step)), 'w') as f:
    f.write(yaml.dump(output, width=120))


def build_training_graph(network, learning_rate):

  # Truth labels for the network
  Y = network['Y']

  # The unscaled network output
  X = network['logits']

  # The weights for the importance of each point
  W = network['W']

  # Gather our individual output for training
  with tf.name_scope('Training'):

    # Global training step
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    # Our loss function
    with tf.name_scope('Loss'):
      loss = tf.reduce_mean(tf.multiply(tf.nn.softmax_cross_entropy_with_logits_v2(logits=X, labels=Y, dim=1), W))

    # Our optimiser
    with tf.name_scope('Optimiser'):
      optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)

    # Calculate accuracy
    with tf.name_scope('Metrics'):
      # Work out which class is larger and make 1 positive and 0 negative
      X = tf.nn.softmax(X)
      predictions = tf.argmin(X, axis=1)
      labels = tf.argmin(Y, axis=1)

      # Get our confusion matrix
      tp = tf.cast(tf.count_nonzero(predictions * labels), tf.float32)
      tn = tf.cast(tf.count_nonzero((predictions - 1) * (labels - 1)), tf.float32)
      fp = tf.cast(tf.count_nonzero(predictions * (labels - 1)), tf.float32)
      fn = tf.cast(tf.count_nonzero((predictions - 1) * labels), tf.float32)

      # Calculate our confusion matrix
      tpr = tp / (tp + fn)
      tnr = tn / (tn + fp)
      ppv = tp / (tp + fp)
      npv = tn / (tn + fn)
      acc = (tp + tn) / (tp + tn + fp + fn)
      f1 = 2.0 * ((ppv * tpr) / (ppv + tpr))
      mcc = (tp * tn - fp * fn) / tf.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

      # How wide the gap is between the two classes
      certainty = tf.reduce_mean(tf.abs(tf.subtract(X[:, 0], X[:, 1])))

  # Monitor loss and metrics
  tf.summary.scalar('Loss', loss)
  tf.summary.scalar('True Positive Rate', tpr)
  tf.summary.scalar('True Negative Rate', tnr)
  tf.summary.scalar('Positive Predictive', ppv)
  tf.summary.scalar('Negative Predictive', npv)
  tf.summary.scalar('Accuracy', acc)
  tf.summary.scalar('F1 Score', f1)
  tf.summary.scalar('Matthews', mcc)
  tf.summary.scalar('Certainty', certainty)

  # TODO make the summary setup for outputting the images

  for v in tf.trainable_variables():
    tf.summary.histogram(v.name, v)

  # Merge all summaries into a single op
  merged_summary_op = tf.summary.merge_all()

  return optimizer, merged_summary_op, global_step


# Train the network
def train(sess, network, config, output_path):

  # Build the training portion of the graph
  optimizer, merged_summary_op, global_step = build_training_graph(network, config['training']['learning_rate'])

  # Setup for tensorboard
  summary_writer = tf.summary.FileWriter(output_path, graph=tf.get_default_graph())

  # Create our model saver to save all the trainable variables and the global_step
  save_vars = {v.name: v for v in tf.trainable_variables()}
  save_vars.update({global_step.name: global_step})
  saver = tf.train.Saver(save_vars)

  # Initialise global variables
  sess.run(tf.global_variables_initializer())

  # Path to model file
  model_path = os.path.join(output_path, 'model.ckpt')

  # If we are loading existing training data do that
  if os.path.isfile(os.path.join(output_path, 'checkpoint')):
    checkpoint_file = tf.train.latest_checkpoint(output_path)
    print('Loading model {}'.format(checkpoint_file))
    saver.restore(sess, checkpoint_file)
  else:
    print('Creating new model {}'.format(model_path))

  if os.path.isfile(config['dataset']['resample']):
    resample_file = config['dataset']['resample']
  else:
    print('Could not find resample file. Training using unweighted points')
    resample_file = None

  # Load our training and validation dataset
  training_dataset = dataset.VisualMeshDataset(
    input_files=config['dataset']['training'],
    classes=config['network']['classes'],
    geometry=config['geometry'],
    batch_size=config['training']['batch_size'],
    variants=config['training']['variants'],
    resample_files=resample_file,
  ).build().make_one_shot_iterator().string_handle()

  # Load our training and validation dataset
  validation_dataset = dataset.VisualMeshDataset(
    input_files=config['dataset']['validation'],
    classes=config['network']['classes'],
    geometry=config['geometry'],
    batch_size=config['validation']['batch_size'],
    variants={},  # No variations for validation
    resample_files=None,  # No resampling in validation
  ).build().repeat().make_one_shot_iterator().string_handle()

  # Get our handles
  training_handle, validation_handle = sess.run([training_dataset, validation_dataset])

  while True:
    try:
      # Run our training step
      sess.run([optimizer], feed_dict={network['handle']: training_handle})

      print("Batch:", tf.train.global_step(sess, global_step))

      # Every N steps do our validation/summary step
      if tf.train.global_step(sess, global_step) % 25 == 0:
        summary, = sess.run([merged_summary_op], feed_dict={network['handle']: validation_handle})
        summary_writer.add_summary(summary, tf.train.global_step(sess, global_step))

      # Every N steps save our model
      if tf.train.global_step(sess, global_step) % 200 == 0:

        # Save the model after every pack
        saver.save(sess, model_path, tf.train.global_step(sess, global_step))

        # Save our model in yaml format
        save_yaml_model(sess, output_path, tf.train.global_step(sess, global_step))

        # TODO run some image outputs for the thing here

    except tf.errors.OutOfRangeError:
      print('Training done')
      break
