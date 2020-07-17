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
import pickle

import tensorflow as tf
import tensorflow_addons as tfa

from .dataset import keras_dataset
from .flavour import get_flavour
from .model import VisualMeshModel


# Train the network
def train(config, output_path):

    # Work out what kind of dataset we are training on and get the data
    datasets, loss, metrics, callbacks = get_flavour(config, output_path)

    # Convert the datasets to keras datasets
    training_dataset, validation_dataset, _ = [d.map(keras_dataset) for d in datasets]

    # Get the dimensionality of the Y part of the dataset
    output_dims = training_dataset.element_spec[1].shape[1]

    # Define the model
    model = VisualMeshModel(structure=config["network"]["structure"], output_dims=output_dims)

    # Setup the optimiser
    if config["training"]["optimiser"]["type"] == "Adam":
        optimiser = tf.optimizers.Adam(learning_rate=float(config["training"]["optimiser"]["learning_rate"]))
    elif config["training"]["optimiser"]["type"] == "SGD":
        optimiser = tf.optimizers.SGD(learning_rate=float(config["training"]["optimiser"]["learning_rate"]))
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
    history = model.fit(
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
            tf.keras.callbacks.TerminateOnNaN(),
            *callbacks,
        ],
    )

    # Pickle the history object
    with open("history.pkl", "wb") as f:
        pickle.dump(history, f)
