#!/usr/bin/env python3

import os
import tensorflow as tf
from .model import VisualMeshModel
from .dataset import VisualMeshDataset
from .metrics import ClassPrecision, ClassRecall
from .callbacks import MeshImages


# Convert a dataset into a format that will be accepted by keras fit
def _prepare_dataset(args):

  # The weights given for each value from the training data
  W = tf.math.multiply(args['Y'], tf.expand_dims(args['W'], axis=-1))

  # Balance the weights by class
  class_weights = tf.math.reduce_sum(W, axis=0)
  # Divide each class by how many samples there are (each col sums to 1)
  W = tf.math.divide(W, class_weights)
  # Sum across classes to remove the 0 values (weights by sample)
  W = tf.math.reduce_sum(W, axis=-1)
  # Normalise the weights by how many classes there are so the total weight sums to 1
  W = tf.math.divide(W, tf.math.count_nonzero(class_weights, dtype=tf.float32))

  # Make the average value be 1 for the batch
  W = tf.math.multiply(W, tf.cast(tf.size(args['W']), dtype=tf.float32))

  # Return in the format (x, y, weights)
  return ((args['X'], args['G']), args['Y'], W)


# Train the network
def train(config, output_path):

  # Get the training dataset
  training_dataset = VisualMeshDataset(
    input_files=config.dataset.training,
    classes=config.network.classes,
    model=config.model,
    batch_size=config.training.batch_size,
    prefetch=tf.data.experimental.AUTOTUNE,
    variants=config.training.variants,
  ).build().map(_prepare_dataset)

  # If we are using batches_per_epoch as a number rather than the whole dataset
  if config.training.batches_per_epoch is not None:
    training_dataset = training_dataset.repeat()

  # Get the validation dataset
  validation_dataset = VisualMeshDataset(
    input_files=config.dataset.validation,
    classes=config.network.classes,
    model=config.model,
    batch_size=config.training.validation.batch_size,
    prefetch=tf.data.experimental.AUTOTUNE,
    variants={},
  ).build().map(_prepare_dataset)

  # Build up the list of metrics we want to track
  metrics = [
    tf.metrics.CategoricalCrossentropy(),
    tf.metrics.Precision(name="metrics/global_precision"),
    tf.metrics.Recall(name="metrics/global_recall")
  ]

  for i, k in enumerate(config.network.classes):
    metrics.append(ClassPrecision(i, name='metrics/{}_precision'.format(k[0])))
    metrics.append(ClassRecall(i, name='metrics/{}_recall'.format(k[0])))

  # Define the model
  model = VisualMeshModel(
    structure=config.network.structure, n_classes=len(config.network.classes), activation=config.network.activation_fn
  )
  model.compile(
    optimizer=tf.optimizers.Adam(),
    loss=tf.losses.CategoricalCrossentropy(),
    metrics=metrics,
  )

  # Find the latest checkpoint file
  checkpoint_file = tf.train.latest_checkpoint(output_path)
  if checkpoint_file is not None:
    model.load_weights(checkpoint_file)

  # Fit the model
  model.fit(
    training_dataset,
    epochs=config.training.epochs,
    steps_per_epoch=config.training.batches_per_epoch,
    validation_data=validation_dataset,
    validation_steps=config.training.validation.samples,
    callbacks=[
      tf.keras.callbacks.TensorBoard(log_dir=output_path, update_freq='batch', write_graph=True, histogram_freq=1),
      tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(output_path, 'model.ckpt'), save_weights_only=True, verbose=1
      ),
      MeshImages(
        output_path,
        config.dataset.validation,
        config.network.classes,
        config.model,
        config.training.validation.progress_images,
        [c[1] for c in config.network.classes],
      ),
    ]
  )
