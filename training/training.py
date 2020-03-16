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

from .callbacks import MeshImages
from .dataset import VisualMeshDataset, keras_dataset
from .loss import FocalLoss
from .metrics import AveragePrecision, AverageRecall, ClassPrecision, ClassRecall
from .model import VisualMeshModel


# Train the network
def train(config, output_path):

    n_classes = len(config["network"]["classes"])

    # Define the model
    model = VisualMeshModel(structure=config["network"]["structure"], n_classes=n_classes)

    # Metrics that we want to track
    metrics = [
        AveragePrecision("metrics/average_precision", n_classes),
        AverageRecall("metrics/average_recall", n_classes),
    ]
    for i, k in enumerate(config["network"]["classes"]):
        metrics.append(ClassPrecision("metrics/{}_precision".format(k["name"]), i, n_classes))
        metrics.append(ClassRecall("metrics/{}_recall".format(k["name"]), i, n_classes))

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
        optimizer=optimiser, loss=FocalLoss(), metrics=metrics,
    )

    # Find the latest checkpoint file and load it
    checkpoint_file = tf.train.latest_checkpoint(output_path)
    if checkpoint_file is not None:
        model.load_weights(checkpoint_file)

    # Get the training dataset
    training_dataset = (
        VisualMeshDataset(
            input_files=config["dataset"]["training"],
            classes=config["network"]["classes"],
            mesh=config["model"]["mesh"],
            geometry=config["model"]["geometry"],
            batch_size=config["training"]["batch_size"],
            prefetch=tf.data.experimental.AUTOTUNE,
            variants=config["training"]["variants"],
        )
        .build()
        .map(keras_dataset)
    )

    # If we are using batches_per_epoch as a number rather than the whole dataset
    if "batches_per_epoch" in config["training"]:
        training_dataset = training_dataset.repeat()

    # Get the validation dataset
    validation_dataset = (
        VisualMeshDataset(
            input_files=config["dataset"]["validation"],
            classes=config["network"]["classes"],
            mesh=config["model"]["mesh"],
            geometry=config["model"]["geometry"],
            batch_size=config["training"]["validation"]["batch_size"],
            prefetch=tf.data.experimental.AUTOTUNE,
            variants={},
        )
        .build()
        .map(keras_dataset)
    )

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
            MeshImages(
                output_path,
                config["dataset"]["validation"],
                config["network"]["classes"],
                config["model"]["mesh"],
                config["model"]["geometry"],
                config["training"]["validation"]["progress_images"],
                [c["colours"][0] for c in config["network"]["classes"]],  # Draw using the first colour
            ),
        ],
    )
