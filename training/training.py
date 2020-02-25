#!/usr/bin/env python3

import os

import tensorflow as tf

import tensorflow_addons as tfa

from .callbacks import MeshImages
from .dataset import VisualMeshDataset
from .loss import FocalLoss
from .metrics import AveragePrecision, AverageRecall, ClassPrecision, ClassRecall
from .model import VisualMeshModel


# Convert a dataset into a format that will be accepted by keras fit
def _prepare_dataset(args):
    # Return in the format (x, y, weights)
    return ((args["X"], args["G"]), args["Y"])


# Train the network
def train(config, output_path):

    # Get the training dataset
    training_dataset = (
        VisualMeshDataset(
            input_files=config.dataset.training,
            classes=config.network.classes,
            model=config.model,
            batch_size=config.training.batch_size,
            prefetch=tf.data.experimental.AUTOTUNE,
            variants=config.training.variants,
        )
        .build()
        .map(_prepare_dataset)
    )

    # If we are using batches_per_epoch as a number rather than the whole dataset
    if config.training.batches_per_epoch is not None:
        training_dataset = training_dataset.repeat()

    # Get the validation dataset
    validation_dataset = (
        VisualMeshDataset(
            input_files=config.dataset.validation,
            classes=config.network.classes,
            model=config.model,
            batch_size=config.training.validation.batch_size,
            prefetch=tf.data.experimental.AUTOTUNE,
            variants={},
        )
        .build()
        .map(_prepare_dataset)
    )

    # Metrics that we want to track
    metrics = [
        AveragePrecision("metrics/average_precision", len(config.network.classes)),
        AverageRecall("metrics/average_precision", len(config.network.classes)),
    ]
    for i, k in enumerate(config.network.classes):
        metrics.append(ClassPrecision("metrics/{}_precision".format(k[0]), i, len(config.network.classes)))
        metrics.append(ClassRecall("metrics/{}_recall".format(k[0]), i, len(config.network.classes)))

    # Define the model
    model = VisualMeshModel(
        structure=config.network.structure,
        n_classes=len(config.network.classes),
        activation=config.network.activation_fn,
    )

    # Setup the optimiser
    if config.training.optimiser.type == "Adam":
        optimiser = tf.optimizers.Adam(learning_rate=float(config.training.optimiser.learning_rate))
    elif config.training.optimiser.type == "Ranger":
        optimiser = tfa.optimizers.Lookahead(
            tfa.optimizers.RectifiedAdam(learning_rate=float(config.training.optimiser.learning_rate)),
            sync_period=int(config.training.optimiser.sync_period),
            slow_step_size=float(config.training.optimiser.slow_step_size),
        )
    else:
        raise RuntimeError("Unknown optimiser type" + config.training.optimiser)

    # Compile the model
    model.compile(
        optimizer=optimiser, loss=FocalLoss(), metrics=metrics,
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
            tf.keras.callbacks.TensorBoard(
                log_dir=output_path, update_freq="batch", profile_batch=0, write_graph=True, histogram_freq=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(output_path, "model.ckpt"), save_weights_only=True, verbose=1
            ),
            MeshImages(
                output_path,
                config.dataset.validation,
                config.network.classes,
                config.model,
                config.training.validation.progress_images,
                [c[1] for c in config.network.classes],
            ),
        ],
    )
