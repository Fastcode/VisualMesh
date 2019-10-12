#!/usr/bin/env python3

import tensorflow as tf
from training.model import VisualMeshModel
from training.dataset import VisualMeshDataset
from training.loss import WeightedLoss


# Train the network
def train(config, output_path):

  # Get the training dataset
  dataset, stats = VisualMeshDataset(
    input_files=config.dataset.training,
    classes=config.network.classes,
    geometry=config.geometry,
    batch_size=config.training.batch_size,
    prefetch=tf.data.experimental.AUTOTUNE,
    variants=config.training.variants,
  ).build()

  # Define the model
  model = VisualMeshModel(config.network.structure, len(config.network.classes))
  model.compile(
    optimizer=optimizer,
    loss=WeightedLoss(),
    metrics=None,
  )

  tensorboard = tf.keras.callbacks.TensorBoard(log_dir=output_path, write_graph=True, write_images=True)

  # Fit the model
  model.fit(dataset, callbacks=[tensorboard])
