#!/usr/bin/env python3

import os
import tensorflow as tf
from training.model import VisualMeshModel
from training.dataset import VisualMeshDataset


# Convert a dataset into a format that will be accepted by keras fit
def _prepare_dataset(args):

  # The weights given for each value from the training data
  W = tf.math.multiply(args['Y'], args['W'])

  # Balance the weights by class
  class_weights = tf.math.reduce_sum(W, axis=0)
  # Divide each class by how many samples there are (each col sums to 1)
  W = tf.math.divide(W, class_weights)
  # Sum across classes to remove the 0 values (weights by sample)
  W = tf.math.reduce_sum(W, axis=-1)
  # Normalise the weights by how many classes there are so the total weight sums to 1
  W = tf.math.divide(W, tf.math.count_nonzero(class_weights, dtype=tf.float32))

  # Make the average value be 1 for the batch
  W = tf.math.multiply(W, tf.size(args['W'], out_type=tf.float32))

  # Return in the format (x, y, weights)
  return ((args['X'], args['G']), args['Y'], W)


# Train the network
def train(config, output_path):

  # Get the training dataset
  training_dataset = VisualMeshDataset(
    input_files=config.dataset.training,
    classes=config.network.classes,
    geometry=config.geometry,
    batch_size=config.training.batch_size,
    prefetch=tf.data.experimental.AUTOTUNE,
    variants=config.training.variants,
  ).build().map(_prepare_dataset)

  # Get the validation dataset
  validation_dataset = VisualMeshDataset(
    input_files=config.dataset.validation,
    classes=config.network.classes,
    geometry=config.geometry,
    batch_size=config.training.validation.batch_size,
    prefetch=tf.data.experimental.AUTOTUNE,
    variants={},
  ).build().map(_prepare_dataset)

  # Define the model
  model = VisualMeshModel(
    structure=config.network.structure, n_classes=len(config.network.classes), activation=config.network.activation_fn
  )
  model.compile(
    optimizer=tf.optimizers.Adam(),
    loss=tf.losses.CategoricalCrossentropy(),
    metrics=[
      tf.metrics.CategoricalCrossentropy(),
      tf.metrics.Precision(),
      tf.metrics.Recall(),
    ],
  )

  # TODO only load the weights if they exist?
  model.load_weights(output_path)

  # Fit the model
  model.fit(
    training_dataset,
    steps_per_epoch=config.training.batches_per_epoch,
    validation_data=validation_dataset,
    validation_steps=config.training.validation.samples,
    callbacks=[
      tf.keras.callbacks.TensorBoard(log_dir=output_path, write_graph=True, write_images=True),
      tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(output_path, 'model.ckpt'), save_weights_only=True, verbose=1
      ),
    ]
  )
