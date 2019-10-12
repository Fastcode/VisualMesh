#!/usr/bin/env python3

import tensorflow as tf
from training.model import VisualMeshModel
from training.dataset import VisualMeshDataset
from training.loss import weighted_loss


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
  model = VisualMeshModel(
    structure=config.network.structure, n_classes=len(config.network.classes), activation=config.network.activation_fn
  )
  model.compile(
    optimizer=tf.optimizers.Adam(),
    loss=weighted_loss,
    metrics=None,
  )

  # TODO only load the weights if they exist?
  model.load_weights(output_path)

  # Fit the model
  model.fit(
    dataset,
    callbacks=[
      tf.keras.callbacks.TensorBoard(log_dir=output_path, write_graph=True, write_images=True),
      tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(output_path, 'model.ckpt'), save_weights_only=True, verbose=1
      ),
    ]
  )
