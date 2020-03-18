# Copyright (C) 2017-2020 Trent Houliston <trent@houliston.me>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os

import tensorflow as tf

import tensorflow_addons as tfa

from .callbacks import ClassificationImages
from .dataset import ClassificationDataset, keras_dataset
from .loss import FocalLoss
from .metrics import AveragePrecision, AverageRecall, ClassPrecision, ClassRecall
from .model import VisualMeshModel


# Train the network
def train(config, output_path):

    # Get some arguments that will always be added to datasets
    dataset_args = {
        "mesh": config["model"]["mesh"],
        "geometry": config["model"]["geometry"],
        "prefetch": tf.data.experimental.AUTOTUNE,
    }
    training_dataset_args = {
        **dataset_args,
        **config["dataset"]["training"],
        "batch_size": config["training"]["batch_size"],
    }
    validation_dataset_args = {
        **dataset_args,
        **config["dataset"]["validation"],
        "batch_size": config["training"]["batch_size"],
    }

    # Load the types based on what type of network we are trying to train
    if config["dataset"]["output"]["type"] == "Classification":

        # Load classification datasets
        training_dataset = (
            ClassificationDataset(**training_dataset_args, classes=config["dataset"]["output"]["classes"])
            .build()
            .map(keras_dataset)
        )
        validation_dataset = (
            ClassificationDataset(**validation_dataset_args, classes=config["dataset"]["output"]["classes"])
            .build()
            .map(keras_dataset)
        )

        output_dims = training_dataset.element_spec[1].shape[1]

        # Metrics that we want to track
        metrics = [
            AveragePrecision("metrics/average_precision", output_dims),
            AverageRecall("metrics/average_recall", output_dims),
        ]
        for i, k in enumerate(config["dataset"]["output"]["classes"]):
            metrics.append(ClassPrecision("metrics/{}_precision".format(k["name"]), i, output_dims))
            metrics.append(ClassRecall("metrics/{}_recall".format(k["name"]), i, output_dims))

        # Classification image callback
        callbacks = [
            ClassificationImages(
                output_path,
                config["dataset"]["validation"]["paths"],
                config["dataset"]["output"]["classes"],
                config["model"]["mesh"],
                config["model"]["geometry"],
                config["training"]["validation"]["progress_images"],
                # Draw using the first colour in the list
                [c["colours"][0] for c in config["dataset"]["output"]["classes"]],
            )
        ]

        # Use focal loss for training
        loss = FocalLoss()

    else:
        raise RuntimeError("Unknown classification network type {}".format(config["network"]["output"]["type"]))

    # Get the dimensionality of the Y part of the dataset
    output_dims = training_dataset.element_spec[1].shape[1]

    # Define the model
    model = VisualMeshModel(structure=config["network"]["structure"], output_dims=output_dims)

    # Setup the optimiser
    if config["training"]["optimiser"]["type"] == "Adam":
        optimiser = tf.optimizers.Adam(learning_rate=float(config["training"]["optimiser"]["learning_rate"]))
    elif config["training"]["optimiser"]["type"] == "Ranger":
        optimiser = tfa.optimizers.Lookahead(
            tfa.optimizers.RectifiedAdam(learning_rate=float(config["training"]["optimiser"]["learning_rate"])),
            sync_period=int(config["training"]["optimiser"]["sync_period"]),
            slow_step_size=float(config["training"]["optimiser"]["slow_step_size"]),
        )
    else:
        raise RuntimeError("Unknown optimiser type" + config["training"]["optimiser"]["type"])

    # Compile the model
    model.compile(
        optimizer=optimiser, loss=loss, metrics=metrics,
    )

    # Find the latest checkpoint file and load it
    checkpoint_file = tf.train.latest_checkpoint(output_path)
    if checkpoint_file is not None:
        model.load_weights(checkpoint_file)

    # If we are using batches_per_epoch as a number rather than the whole dataset
    if "batches_per_epoch" in config["training"]:
        training_dataset = training_dataset.repeat()

    # Fit the model
    model.fit(
        training_dataset,
        epochs=config["training"]["epochs"],
        steps_per_epoch=(
            None if "batches_per_epoch" not in config["training"] else config["training"]["batches_per_epoch"]
        ),
        validation_data=validation_dataset,
        validation_steps=config["training"]["validation"]["samples"],
        callbacks=[
            tf.keras.callbacks.TensorBoard(
                log_dir=output_path, update_freq="batch", profile_batch=0, write_graph=True, histogram_freq=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(output_path, "model"),
                monitor="loss",
                save_weights_only=True,
                save_best_only=True,
            ),
            *callbacks,
        ],
    )
