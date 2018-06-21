#!/usr/bin/env python3

import os
import sys
import random
import tensorflow as tf
import yaml
import re
import io
import cv2
import time
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from . import network
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
    info = re.match(r'mesh/Conv_(\d+)/Layer_(\d+)/(Weights|Biases):0', k)
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


class MeshDrawer:

  def __init__(self, classes):
    self.classes = classes

  def mesh_image(self, raws, pxs, ns, X):
    # Find the edges of the X values
    cs = np.cumsum(ns)
    cs = np.concatenate([[0], cs]).tolist()
    ranges = list(zip(cs, cs[1:]))

    images = []

    for batch, raw in enumerate(raws):
      img = cv2.imdecode(np.fromstring(raw, np.uint8), cv2.IMREAD_COLOR)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      px = pxs[batch, :ns[batch] - 1]  # Skip the null point (which doesn't exist in px)
      x = X[ranges[batch][0]:ranges[batch][1] - 1]  # Skip the null point

      # Setup the display so everything is all at the correct resolution
      dpi = 80
      height, width, nbands = img.shape
      figsize = width / float(dpi), height / float(dpi)
      fig = plt.figure(figsize=figsize)
      ax = fig.add_axes([0, 0, 1, 1])
      ax.axis('off')

      # Image underlay
      ax.imshow(img, interpolation='nearest')

      # Now for each class, produce a contour plot
      for i, data in enumerate(self.classes):
        r, g, b = data[1]
        r /= 255
        g /= 255
        b /= 255

        ax.tricontour(
          px[:, 1],
          px[:, 0],
          x[:, i],
          levels=[0.5, 0.75, 0.9],
          colors=[(r, g, b, 0.33), (r, g, b, 0.66), (r, g, b, 1.0)]
        )

      ax.set(xlim=[0, width], ylim=[height, 0], aspect=1)
      data = io.BytesIO()
      fig.savefig(data, format='jpg', dpi=dpi)
      data.seek(0)
      result = cv2.imdecode(np.fromstring(data.read(), np.uint8), cv2.IMREAD_COLOR)
      images.append(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

      fig.clf()
      plt.close(fig)

    return np.stack(images)

  def tutor_image(self, raws, pxs, ns, A):
    # Find the edges of the X values
    cs = np.cumsum(ns)
    cs = np.concatenate([[0], cs]).tolist()
    ranges = list(zip(cs, cs[1:]))

    images = []

    for batch, raw in enumerate(raws):
      img = cv2.imdecode(np.fromstring(raw, np.uint8), cv2.IMREAD_COLOR)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      px = pxs[batch, :ns[batch] - 1]  # Skip the null point (which doesn't exist in px)
      a = A[ranges[batch][0]:ranges[batch][1] - 1]  # Skip the null point

      # Setup the display so everything is all at the correct resolution
      dpi = 80
      height, width, nbands = img.shape
      figsize = width / float(dpi), height / float(dpi)
      fig = plt.figure(figsize=figsize)
      ax = fig.add_axes([0, 0, 1, 1])
      ax.axis('off')

      # Image underlay
      ax.imshow(img, interpolation='nearest')

      # Make our tutor plot
      ax.tricontour(
        px[:, 1],
        px[:, 0],
        a,
        levels=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        cmap=plt.get_cmap('jet'),
      )

      ax.set(xlim=[0, width], ylim=[height, 0], aspect=1)
      data = io.BytesIO()
      fig.savefig(data, format='jpg', dpi=dpi)
      data.seek(0)
      result = cv2.imdecode(np.fromstring(data.read(), np.uint8), cv2.IMREAD_COLOR)
      images.append(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

      fig.clf()
      plt.close(fig)

    return np.stack(images)


def build_training_graph(network, classes, learning_rate, tutor_learning_rate, tutor_threshold, base_weight):

  # Truth labels for the network
  Y = network['Y']

  # The unscaled network output
  X = network['mesh']

  # The unscaled tutor networks output
  A = network['tutor'][:, 0]

  # The alpha channel from the training data to remove unlabelled points from the gradients
  W = network['W']
  S = tf.where(tf.greater(W, 0))
  Y = tf.gather_nd(Y, S)
  X = tf.gather_nd(X, S)
  A = tf.nn.sigmoid(tf.gather_nd(A, S))  # Sigmoid as the final layer

  training_summary = []
  validation_summary = []
  image_summary = []

  # Gather our individual output for training
  with tf.name_scope('Training'):

    # Global training step
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    # Our loss function
    with tf.name_scope('Loss'):
      # Unweighted loss, before the tutor decides which samples are more important
      unweighted_mesh_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=X, labels=Y, dim=1)

      # Calculate the labels for the tutor
      a_labels = tf.reduce_sum(tf.abs(tf.subtract(Y, tf.nn.softmax(X, axis=1))), axis=1) / 2.0

      # Only use gradients from areas where the tutor has larger error, this avoids a large number of smaller
      # gradients overpowering the areas where the network has legitimate error.
      # This technique means that the tutor network will never converge, but we don't ever want it to
      a_idx = tf.where(tf.greater(tf.abs(a_labels - A), tutor_threshold))

      # If we have no values that are inaccurate, we will take all the values as normal
      tutor_loss_cut = tf.losses.mean_squared_error(
        predictions=tf.gather_nd(A, a_idx),
        labels=tf.stop_gradient(tf.gather_nd(a_labels, a_idx)),
      )
      tutor_loss_full = tf.losses.mean_squared_error(
        predictions=A,
        labels=tf.stop_gradient(a_labels),
      )
      tutor_loss = tf.cond(tf.equal(tf.size(a_idx), 0), lambda: tutor_loss_full, lambda: tutor_loss_cut)

      # Calculate the loss weights for each of the classes
      scatters = []
      for i in range(len(classes)):
        # Indexes of truth samples for this class
        idx = tf.where(Y[:, i])
        pts = tf.gather_nd(A, idx)
        pts = tf.divide(pts + base_weight, tf.reduce_sum(pts))
        pts = tf.scatter_nd(idx, pts, tf.shape(A, out_type=tf.int64))

        # Either our weights, or if there were none, zeros
        scatters.append(tf.cond(tf.equal(tf.size(idx), 0), lambda: tf.zeros_like(A), lambda: pts))

      # Even if we don't have all classes, the weights should sum to 1
      active_classes = tf.cast(tf.count_nonzero(tf.stack([tf.count_nonzero(s) for s in scatters])), tf.float32)
      W = tf.add_n(scatters)
      W = tf.divide(W, active_classes)

      # Weighted mesh loss, sum rather than mean as we have already normalised based on number of points
      weighted_mesh_loss = tf.reduce_sum(tf.multiply(unweighted_mesh_loss, tf.stop_gradient(W)))

      training_summary.append(tf.summary.scalar('Mesh Loss', weighted_mesh_loss))
      training_summary.append(tf.summary.scalar('Tutor Loss', tutor_loss))

    # Our optimisers
    with tf.name_scope('Optimiser'):
      mesh_optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
        weighted_mesh_loss, global_step=global_step
      )
      tutor_optimiser = tf.train.GradientDescentOptimizer(learning_rate=tutor_learning_rate).minimize(tutor_loss)

  # Calculate accuracy
  with tf.name_scope('Validation'):
    # Work out which class is larger and make 1 positive and 0 negative
    X = tf.nn.softmax(X)

    for i, c in enumerate(classes):
      name = c[0]
      with tf.name_scope(name.title()):
        idx = tf.where(Y[:, i])
        predictions = tf.cast(tf.equal(tf.argmax(X, axis=1), i), tf.int32)
        labels = tf.cast(tf.equal(tf.argmax(Y, axis=1), i), tf.int32)

        weights = tf.gather_nd(W, idx)
        unweighted = tf.gather_nd(unweighted_mesh_loss, idx)

        # Get our confusion matrix
        tp = tf.cast(tf.count_nonzero(predictions * labels), tf.float32)
        tn = tf.cast(tf.count_nonzero((predictions - 1) * (labels - 1)), tf.float32)
        fp = tf.cast(tf.count_nonzero(predictions * (labels - 1)), tf.float32)
        fn = tf.cast(tf.count_nonzero((predictions - 1) * labels), tf.float32)

        # Calculate our confusion matrix
        validation_summary.append(tf.summary.scalar('Loss', tf.reduce_sum(tf.multiply(unweighted, weights))))
        validation_summary.append(tf.summary.scalar('Precision', tp / (tp + fp)))
        validation_summary.append(tf.summary.scalar('Recall', tp / (tp + fn)))
        validation_summary.append(tf.summary.scalar('Accuracy', (tp + tn) / (tp + fp + tn + fn)))

    with tf.name_scope('Global'):
      # Monitor loss and metrics
      validation_summary.append(tf.summary.scalar('Mesh Loss', weighted_mesh_loss))
      validation_summary.append(tf.summary.scalar('Tutor Loss', tutor_loss))

  with tf.name_scope('MeshImage'):
    mesh_drawer = MeshDrawer(classes)
    image_summary.append(
      tf.summary.image(
        'Mesh',
        tf.py_func(
          mesh_drawer.mesh_image, [network['raw'], network['px'], network['n'],
                                   tf.nn.softmax(network['mesh'])], tf.uint8, False
        ),
        max_outputs=10000,  # Doesn't matter as we limit it at the dataset/batch level
      )
    )

  with tf.name_scope('TutorImage'):
    image_summary.append(
      tf.summary.image(
        'Tutor',
        tf.py_func(
          mesh_drawer.tutor_image,
          [network['raw'], network['px'], network['n'],
           tf.nn.sigmoid(network['tutor'][:, 0])],
          tf.uint8,
          False,
        ),
        max_outputs=10000,  # Doesn't matter as we limit it at the dataset/batch level
      )
    )

  for v in tf.trainable_variables():
    validation_summary.append(tf.summary.histogram(v.name, v))

  # Merge all summaries into a single op
  training_summary_op = tf.summary.merge(training_summary)
  validation_summary_op = tf.summary.merge(validation_summary)
  image_summary_op = tf.summary.merge(image_summary)

  return {
    'mesh_optimiser': mesh_optimiser,
    'mesh_loss': weighted_mesh_loss,
    'tutor_optimiser': tutor_optimiser,
    'tutor_loss': tutor_loss,
    'training_summary': training_summary_op,
    'validation_summary': validation_summary_op,
    'image_summary': image_summary_op,
    'global_step': global_step,
  }


# Train the network
def train(sess, config, output_path):

  # Extract configuration variables
  training_files = config['dataset']['training']
  validation_files = config['dataset']['validation']
  geometry = config['geometry']
  classes = config['network']['classes']
  structure = config['network']['structure']
  base_weight = config['training']['tutor']['base_weight']
  tutor_learning_rate = config['training']['tutor']['learning_rate']
  tutor_structure = config['training']['tutor'].get('structure', None)
  tutor_threshold = config['training']['tutor']['threshold']
  training_batch_size = config['training']['batch_size']
  epochs = config['training']['epochs']
  training_learning_rate = config['training']['learning_rate']
  save_frequency = config['training']['save_frequency']
  shuffle_size = config['training']['shuffle_size']
  validation_batch_size = config['training']['validation']['batch_size']
  validation_frequency = config['training']['validation']['frequency']
  image_frequency = config['training']['validation']['image_frequency']
  n_progress_images = config['training']['validation']['progress_images']
  variants = config['training']['variants']

  # Build our student and tutor networks
  net = network.build(structure, len(classes), structure if tutor_structure is None else tutor_structure)

  # Build the training portion of the graph
  training_graph = build_training_graph(
    net, classes, training_learning_rate, tutor_learning_rate, tutor_threshold, base_weight
  )
  mesh_optimiser = training_graph['mesh_optimiser']
  mesh_loss = training_graph['mesh_loss']
  tutor_optimiser = training_graph['tutor_optimiser']
  tutor_loss = training_graph['tutor_loss']
  training_summary = training_graph['training_summary']
  validation_summary = training_graph['validation_summary']
  image_summary = training_graph['image_summary']
  global_step = training_graph['global_step']

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

  # Load our training and validation dataset
  training_dataset = dataset.VisualMeshDataset(
    input_files=training_files,
    classes=classes,
    geometry=geometry,
    batch_size=training_batch_size,
    shuffle_size=shuffle_size,
    variants=variants,
  ).build().repeat(epochs).make_one_shot_iterator().string_handle()

  # Load our training and validation dataset
  validation_dataset = dataset.VisualMeshDataset(
    input_files=validation_files,
    classes=classes,
    geometry=geometry,
    shuffle_size=shuffle_size,
    batch_size=validation_batch_size,
    variants={},  # No variations for validation
  ).build().repeat().make_one_shot_iterator().string_handle()

  # Build our image dataset for drawing images
  image_dataset = dataset.VisualMeshDataset(
    input_files=validation_files,
    classes=classes,
    geometry=geometry,
    shuffle_size=0,  # Don't shuffle so we can resume
    batch_size=n_progress_images,
    variants={},  # No variations for images
  ).build().make_one_shot_iterator().get_next()
  # Make a dataset with a single element for generating images off
  image_dataset = tf.data.Dataset.from_tensors(sess.run(image_dataset)
                                              ).repeat().make_one_shot_iterator().string_handle()

  # Get our handles
  training_handle, validation_handle, image_handle = sess.run([training_dataset, validation_dataset, image_dataset])

  while True:
    try:
      # Run our training step
      start = time.time()
      summary, ml, al, _, __, = sess.run([training_summary, mesh_loss, tutor_loss, mesh_optimiser, tutor_optimiser],
                                         feed_dict={net['handle']: training_handle})
      summary_writer.add_summary(summary, tf.train.global_step(sess, global_step))
      end = time.time()

      # Print batch info
      print(
        'Batch: {} ({:3g}s) Mesh Loss: {:3g} Tutor Loss: {:3g}'.format(
          tf.train.global_step(sess, global_step),
          (end - start),
          ml,
          al,
        )
      )

      # Every N steps do our validation/summary step
      if tf.train.global_step(sess, global_step) % validation_frequency == 0:
        summary, = sess.run([validation_summary], feed_dict={net['handle']: validation_handle})
        summary_writer.add_summary(summary, tf.train.global_step(sess, global_step))

      # Every N steps save our model
      if tf.train.global_step(sess, global_step) % save_frequency == 0:

        # Save the model in tf format
        saver.save(sess, model_path, tf.train.global_step(sess, global_step))

        # Save our model in yaml format
        save_yaml_model(sess, output_path, tf.train.global_step(sess, global_step))

      # Every N steps save our model
      if tf.train.global_step(sess, global_step) % image_frequency == 0:

        # Output the images
        summary = sess.run(image_summary, feed_dict={net['handle']: image_handle})
        summary_writer.add_summary(summary, tf.train.global_step(sess, global_step))

    except tf.errors.OutOfRangeError:
      print('Training done')
      break
