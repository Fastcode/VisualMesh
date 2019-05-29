#!/usr/bin/env python3

import os
import tensorflow as tf
import numpy as np
from tensorflow.python.client import device_lib
import copy
import yaml
import re
import time
import multiprocessing

import training.lr_policy as lr_policy
from . import mesh_drawer
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
    info = re.match(r'Network/Conv_(\d+)/Layer_(\d+)/(Weights|Biases):0', k)
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


def _build_device_training_graph(data, network_structure, config, network_optimiser):
  # Create the network graph ops for this device
  with tf.variable_scope('Network'):
    X = network.build_network(data['X'], data['G'], network_structure, config.network.activation_fn)

  # Capture these before they are filtered for drawing images
  X_0 = tf.nn.softmax(X, axis=-1)

  # First eliminate points that were masked out with alpha
  with tf.name_scope('AlphaMask'):
    S = tf.where(tf.greater(data['W'], 0))
    X = tf.gather_nd(X, S)
    Y = tf.gather_nd(data['Y'], S)

  # Calculate the loss for the batch on this device
  loss = _loss(X, Y, config)

  # Calculate summary information for validation passes
  metrics = _metrics(X, Y, config)

  # Calculate the gradients for this device
  grads = network_optimiser.compute_gradients(loss)

  # Store the ops that have been done on this device
  return {
    'inference': {
      'X': X_0,
      'G': data['G'],
      'n': data['n'],
      'px': data['px'],
      'raw': data['raw']
    },
    'loss': loss,
    'grads': grads,
    'metrics': metrics,
  }


def _loss(X, Y, config):
  '''Calculate the loss for the Network given the provided labels and configuration'''
  with tf.name_scope('Loss'):

    # Unweighted loss, before the losses are balanced per class
    unweighted_mesh_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=X, labels=Y, axis=1)

    # Balance the loss per class
    W = Y
    class_weights = tf.reduce_sum(Y, axis=0)
    W = tf.divide(W, class_weights)
    W = tf.reduce_sum(W, axis=-1)
    W = tf.divide(W, tf.count_nonzero(class_weights, dtype=tf.float32))

    # Weighted mesh loss, sum rather than mean as we have already normalised based on number of points
    loss = tf.reduce_sum(tf.multiply(unweighted_mesh_loss, tf.stop_gradient(W)))

  return loss


def _metrics(X, Y, config):
  with tf.name_scope('Metrics'):
    metrics = {}

    # Calculate our unweighted loss, and the actual prediction from the network
    network_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=X, labels=Y, axis=1)
    X = tf.nn.softmax(X, axis=1)

    for i, c in enumerate(config.network.classes):

      # Get our confusion matrix
      predictions = tf.cast(tf.equal(tf.argmax(X, axis=1), i), tf.int32)
      labels = tf.cast(tf.equal(tf.argmax(Y, axis=1), i), tf.int32)
      tp = tf.count_nonzero(predictions * labels, dtype=tf.float32)
      tn = tf.count_nonzero((predictions - 1) * (labels - 1), dtype=tf.float32)
      fp = tf.count_nonzero(predictions * (labels - 1), dtype=tf.float32)
      fn = tf.count_nonzero((predictions - 1) * labels, dtype=tf.float32)

      # Get the loss for this specific class
      class_loss = tf.reduce_mean(tf.gather_nd(network_loss, tf.where(Y[:, i])))

      # Work out what class this is confused with (precision/recall breakdown)
      precision_dist = tf.reduce_sum(tf.gather(Y, tf.where(predictions)), axis=0)
      recall_dist = tf.reduce_sum(tf.gather(X, tf.where(labels)), axis=0)

      # Add to our metrics object
      metrics[c[0]] = {
        'loss': class_loss,
        'dist': {
          'precision': precision_dist,
          'recall': recall_dist
        },
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
      }

    # Count how many losses were non 0 (0 loss means there were none of this class in the batch)
    class_losses = [m['loss'] for k, m in metrics.items()]
    active_classes = tf.add_n([tf.count_nonzero(l, dtype=tf.float32) for l in class_losses])
    metrics['Global'] = {
      'loss': tf.divide(tf.add_n(class_losses), active_classes),
      'tp': tf.add_n([m['tp'] for k, m in metrics.items()]),
      'tn': tf.add_n([m['tn'] for k, m in metrics.items()]),
      'fp': tf.add_n([m['fp'] for k, m in metrics.items()]),
      'fn': tf.add_n([m['fn'] for k, m in metrics.items()]),
    }

  return metrics


def _merge_ops(device_ops):

  # Always merge on the CPU
  with tf.device('/device:CPU:0'):
    # Merge the results of the operations together
    loss = tf.add_n([op['loss'] for op in device_ops]) / len(device_ops)

    # Merge the gradients together
    grads = []
    for g in zip(*[op['grads'] for op in device_ops]):
      # None gradients don't matter
      if not any([v[0] is None for v in g]):
        grads.append((tf.divide(tf.add_n([v[0] for v in g]), len(device_ops)), g[0][1]))

    # Merge the metrics together
    def _merge_metrics(metrics):
      if type(metrics[0]) == dict:
        return {k: _merge_metrics([m[k] for m in metrics]) for k in metrics[0]}
      else:
        return tf.add_n(metrics)

    metrics = _merge_metrics([op['metrics'] for op in device_ops])

    # Divide all the losses here by the number of GPUs to correct scaling
    metrics = {k: {**m, 'loss': tf.divide(m['loss'], len(device_ops))} for k, m in metrics.items()}

    return {
      'inference': [op['inference'] for op in device_ops],
      'loss': loss,
      'grads': grads,
      'metrics': metrics,
    }


def _progress_images(pool, inferences, config):
  # Arguments for creating each image
  image_args = []

  # For each of the GPUs outputs
  for inference in inferences:
    # Find the edges
    cs = [0] + np.cumsum(inference['n']).tolist()
    ranges = list(zip(cs, cs[1:]))

    # For each image from the GPU
    for i, r in enumerate(ranges):
      image_args.append((
        inference['raw'][i], inference['px'][r[0]:r[1]], inference['X'][r[0]:r[1]],
        [c[1] for c in config.network.classes]
      ))

  # Process all images and sort them so they show up in the same order
  x_imgs = pool.starmap(mesh_drawer.draw, image_args)
  x_imgs.sort(key=lambda k: k[0])

  return tf.Summary(
    value=[
      tf.Summary.Value(
        tag='Mesh/Image/{}'.format(i),
        image=tf.Summary.Image(height=data[1], width=data[2], colorspace=3, encoded_image_string=data[3])
      ) for i, data in enumerate(x_imgs)
    ]
  )


def _build_training_graph(gpus, config):
  # Some variables must exist on the CPU
  with tf.device('/device:CPU:0'):
    # Optimiser, and global_step variables on the CPU
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    # Expose the learning rate as a placeholder so we can utilise them in our training
    learning_rate = tf.placeholder(dtype=tf.float32)

    # Create our optimisers
    network_optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # This iterator is used so we can swap datasets as we go
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
      handle, {
        'X': tf.float32,
        'Y': tf.float32,
        'G': tf.int32,
        'W': tf.float32,
        'n': tf.int32,
        'px': tf.float32,
        'raw': tf.string,
      }, {
        'X': [None, 3],
        'Y': [None, len(config.network.classes)],
        'G': [None, 7],
        'W': [None],
        'n': [None],
        'px': [None, 2],
        'raw': [None],
      }
    )

  # Calculate the structure for the network and add on the number of classes to the final step
  network_structure = copy.deepcopy(config.network.structure)
  network_structure[-1].append(len(config.network.classes))

  # For each GPU build a classification network and a gradients calculator
  device_ops = []
  for i, gpu in enumerate(gpus):
    with tf.device(gpu), tf.name_scope('Tower_{}'.format(i)):
      device_ops.append(_build_device_training_graph(iterator.get_next(), network_structure, config, network_optimiser))

  # If we have multiple GPUs we need to do a merge operation, otherwise just take the element
  ops = _merge_ops(device_ops) if len(device_ops) > 1 else device_ops[0]

  # Accumulate the data over multiple batches
  with tf.device('/device:CPU:0'):
    # Get all trainable variables in the current tower
    x_tvs = tf.trainable_variables(scope='^Network')

    # Create a list of variables with the same shape as the trainable ones initialized with zeros
    x_accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in x_tvs]
    x_zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in x_accum_vars]

    # Accumulate gradients
    x_accum_ops = [x_accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(ops['grads'])]

  # Apply the gradients as part of the optimisation
  with tf.device('/device:CPU:0'):
    # Average the accumulated gradients over the number of batches that we accumulated
    x_divide_ops = [
      (tf.divide(x_accum_vars[i], config.training.batch_accumulation), gv[1]) for i, gv in enumerate(ops['grads'])
    ]

    # Apply the averaged gradients
    optimise_mesh_op = network_optimiser.apply_gradients(x_divide_ops, global_step=global_step)

  # Create the loss summary op
  with tf.name_scope('Training'):
    loss_summary_op = tf.summary.merge([
      tf.summary.scalar('Loss', ops['loss']),
      tf.summary.scalar('Learning_Rate', learning_rate),
    ])

  # Now use the metrics to calculate interesting validation details
  validation_summary_op = []
  for k, m in ops['metrics'].items():
    with tf.name_scope(k.title()):
      validation_summary_op.extend([
        tf.summary.scalar('Loss', m['loss']),
        tf.summary.scalar('Precision', m['tp'] / (m['tp'] + m['fp'])),
        tf.summary.scalar('Recall', m['tp'] / (m['tp'] + m['fn']))
      ])
  validation_summary_op = tf.summary.merge(validation_summary_op)

  # Return the graph operations we will want to run
  return {
    'handle': handle,
    'global_step': global_step,
    'initialisers': x_zero_ops,
    'learning_rate': learning_rate,
    'train': x_accum_ops,
    'update': {
      'update': optimise_mesh_op,
      'loss': ops['loss'],
      'summary': loss_summary_op
    },
    'validate': {
      'summary': validation_summary_op,
      'dist': {
        'precision': {k: m['dist']['precision'] for k, m in ops['metrics'].items() if k != 'Global'},
        'recall': {k: m['dist']['recall'] for k, m in ops['metrics'].items() if k != 'Global'}
      },
    },
    'image': {
      'inference': ops['inference'],
    },
  }


# Train the network
def train(config, output_path):

  # Thread pool for multiprocessing
  with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:

    # Find the GPUs we have available and if we don't have any, fallback to CPU
    gpus = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']
    gpus = ['/device:CPU:0'] if len(gpus) == 0 else gpus

    # Build the training graph operations we need
    ops = _build_training_graph(gpus, config)
    global_step = ops['global_step']

    # Setup for tensorboard
    summary_writer = tf.summary.FileWriter(output_path, graph=tf.get_default_graph())

    # Create our model saver to save all the trainable variables and the global_step
    save_vars = {v.name: v for v in tf.trainable_variables()}
    save_vars.update({global_step.name: global_step})
    saver = tf.train.Saver(save_vars)

    # Load our training and validation dataset
    training_dataset, training_ds_stats = dataset.VisualMeshDataset(
      input_files=config.dataset.training,
      classes=config.network.classes,
      geometry=config.geometry,
      batch_size=max(1, config.training.batch_size // len(gpus)),
      prefetch=tf.data.experimental.AUTOTUNE,
      variants=config.training.variants,
    ).build(stats=True)
    training_dataset = training_dataset.repeat().make_initializable_iterator()

    # Merge in the dataset stats into the training summary
    ops['update']['summary'] = tf.summary.merge([ops['update']['summary'], training_ds_stats])

    # Load our training and validation dataset
    validation_dataset = dataset.VisualMeshDataset(
      input_files=config.dataset.validation,
      classes=config.network.classes,
      geometry=config.geometry,
      batch_size=max(1, config.training.validation.batch_size // len(gpus)),
      prefetch=tf.data.experimental.AUTOTUNE,
      variants={},  # No variations for validation
    ).build().repeat().make_one_shot_iterator()

    # Build our image dataset for drawing images
    image_dataset = dataset.VisualMeshDataset(
      input_files=config.dataset.validation,
      classes=config.network.classes,
      geometry=config.geometry,
      batch_size=max(1, config.training.validation.progress_images // len(gpus)),
      prefetch=tf.data.experimental.AUTOTUNE,
      variants={},  # No variations for images
    ).build()
    image_dataset = image_dataset.take(len(gpus)).repeat().make_one_shot_iterator()

    # Tensorflow session configuration
    tf_config = tf.ConfigProto()
    tf_config.allow_soft_placement = False
    tf_config.gpu_options.allow_growth = True

    with tf.Session(config=tf_config) as sess:

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

      # Initialise our dataset and get our string handles for use
      sess.run([training_dataset.initializer])
      training_handle, validation_handle, image_handle = sess.run([
        training_dataset.string_handle(),
        validation_dataset.string_handle(),
        image_dataset.string_handle()
      ])

      # Our learning rate policy
      rate_policy = {
        'STATIC': lr_policy.Static,
        'ONE_CYCLE': lr_policy.OneCycle
      }[config.training.learning_policy.type](
        config.training.learning_policy
      )

      # We are done messing with the graph
      tf.get_default_graph().finalize()

      while not rate_policy.finished(tf.train.global_step(sess, global_step)):
        start = time.perf_counter()

        # Initialise/reset accumulators
        sess.run(ops['initialisers'])

        # Run minibatches and accumulate gradients
        for _ in range(config.training.batch_accumulation):
          sess.run(
            ops['train'], feed_dict={
              ops['handle']: training_handle,
              ops['learning_rate']: rate_policy.learning_rate
            }
          )

        # Apply accumulated gradients and calculate loss
        output = sess.run(
          ops['update'], feed_dict={
            ops['handle']: training_handle,
            ops['learning_rate']: rate_policy.learning_rate
          }
        )

        # Add summary for tensorboard
        summary_writer.add_summary(output['summary'], tf.train.global_step(sess, global_step))

        end = time.perf_counter()

        # Update the rate policy
        rate_policy.update(tf.train.global_step(sess, global_step))

        # Print batch info
        print(
          'Batch: {} ({:3g}s avg.) Loss: {:3g}'.format(
            tf.train.global_step(sess, global_step),
            (end - start) / config.training.batch_accumulation,
            output['loss'],
          )
        )

        # Every N steps do our validation/summary step
        if tf.train.global_step(sess, global_step) % config.training.validation.frequency == 0:
          output = sess.run(ops['validate'], feed_dict={ops['handle']: validation_handle})
          summary_writer.add_summary(output['summary'], tf.train.global_step(sess, global_step))

          # Histogram summary
          histograms = []
          for name, classes in output['dist'].items():
            for k, vs in classes.items():

              # Normalise the vector so they sum to 1.0
              vs = vs[0] / np.sum(vs[0])

              # Make a pretend bar chart
              edges = []
              buckets = []
              for i, v in enumerate(vs):
                edges.extend([i - 2 / 6, i - 1 / 6, i, i + 1 / 6, i + 2 / 6, i + 3 / 6])
                buckets.extend([0, v, v, v, v, 0])

              # Interleave with 0s so it looks like categories
              histograms.append(
                tf.Summary.Value(
                  tag='{}/Confusion/{}'.format(k.title(), name.title()),
                  histo=tf.HistogramProto(min=-0.5, max=vs.size - 0.5, bucket_limit=edges, bucket=buckets)
                )
              )

          histograms = tf.Summary(value=histograms)
          summary_writer.add_summary(histograms, tf.train.global_step(sess, global_step))

        # Every N steps save our model
        if tf.train.global_step(sess, global_step) % config.training.save_frequency == 0:
          saver.save(sess, model_path, tf.train.global_step(sess, global_step))
          save_yaml_model(sess, output_path, tf.train.global_step(sess, global_step))

        # Every N steps show our image summary
        if tf.train.global_step(sess, global_step) % config.training.validation.image_frequency == 0:
          output = sess.run(ops['image'], feed_dict={ops['handle']: image_handle})
          summary = _progress_images(pool, output['inference'], config)
          summary_writer.add_summary(summary, tf.train.global_step(sess, global_step))

      # Do a validation step
      output = sess.run(ops['validate'], feed_dict={ops['handle']: validation_handle})
      summary_writer.add_summary(output['summary'], tf.train.global_step(sess, global_step))

      # Output some images
      output = sess.run(ops['image'], feed_dict={ops['handle']: image_handle})
      summary = _progress_images(pool, output['inference'], config)
      summary_writer.add_summary(summary, tf.train.global_step(sess, global_step))

      # Save the model
      saver.save(sess, model_path, tf.train.global_step(sess, global_step))
      save_yaml_model(sess, output_path, tf.train.global_step(sess, global_step))

      print('Training done')
