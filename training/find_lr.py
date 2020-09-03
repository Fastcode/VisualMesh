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

import math
import os
import pickle

import matplotlib as mpl
import numpy as np
from tqdm import tqdm

import tensorflow as tf

from .dataset import keras_dataset
from .flavour import Dataset, Loss
from .model import VisualMeshModel

mpl.use("Agg")
import matplotlib.pyplot as plt  # isort:skip


class LRProgress(tf.keras.callbacks.Callback):
    def __init__(self, n_steps, lr_schedule, **kwargs):
        super(LRProgress, self).__init__(**kwargs)

        self.lr_schedule = lr_schedule
        self.n_steps = n_steps

        # Tracking loss values
        self.smooth_loss = None
        self.losses = []

        # Progress bar settings
        self.n_history = 20
        self.lr_progress = None
        self.loss_title = None
        self.loss_progress = None

    def on_epoch_end(self, epoch, logs=None):
        # Skip when we explode at the end
        if not math.isfinite(logs["loss"]):
            return

        # Calculate smoothed loss values
        self.smooth_loss = logs["loss"] if self.smooth_loss is None else self.smooth_loss * 0.98 + logs["loss"] * 0.02
        self.losses.append(math.log1p(self.smooth_loss))

        # Create the lr_progress bar so we can see how far it is
        if self.lr_progress is None:
            self.lr_progress = tqdm(total=self.n_steps, dynamic_ncols=True)
        self.lr_progress.update()
        self.lr_progress.set_description("LR:   {:.3e}".format(self.model.optimizer.lr.numpy()))

        # Create our series of moving loss graphs
        if self.loss_progress is None:
            self.loss_title = tqdm(bar_format="{desc}", desc="Loss Graph")
            self.loss_progress = [
                tqdm(bar_format="{desc}|{bar}|", total=self.losses[-1], dynamic_ncols=True,)
                for i in range(self.n_history)
            ]

        valid_i = -min(len(self.losses), self.n_history)

        # Get the maximum of the losses
        loss_max = max(self.losses)

        for bar, loss in zip(self.loss_progress[valid_i:], self.losses[valid_i:]):
            bar.total = loss_max
            bar.n = loss
            bar.set_description("{:.3e}".format(math.expm1(loss)))


# Find valid learning rates for
def find_lr(config, output_path):

    # Open the training dataset and put it on repeat
    training_dataset = Dataset(config, "validation").map(keras_dataset).repeat()

    # Get the dimensionality of the Y part of the dataset
    output_dims = training_dataset.element_spec[1].shape[1]

    # Define the model
    model = VisualMeshModel(structure=config["network"]["structure"], output_dims=output_dims)

    # The max and min lr that we will be searching from
    min_lr = 1e-6
    max_lr = 1e2
    n_steps = 10000
    average_window = 250

    def lr_schedule(epoch, lr):
        return min_lr * (max_lr / min_lr) ** (epoch / n_steps)

    # Setup the optimiser
    if config["training"]["optimiser"]["type"] == "Adam":
        optimiser = tf.optimizers.Adam(learning_rate=min_lr)
    elif config["training"]["optimiser"]["type"] == "SGD":
        optimiser = tf.optimizers.SGD(learning_rate=min_lr)
    elif config["training"]["optimiser"]["type"] == "Ranger":
        import tensorflow_addons as tfa

        optimiser = tfa.optimizers.Lookahead(
            tfa.optimizers.RectifiedAdam(learning_rate=min_lr),
            sync_period=int(config["training"]["optimiser"]["sync_period"]),
            slow_step_size=float(config["training"]["optimiser"]["slow_step_size"]),
        )
    else:
        raise RuntimeError("Unknown optimiser type" + config["training"]["optimiser"]["type"])

    # Compile the model grabbing the flavours for the loss and metrics
    model.compile(optimizer=optimiser, loss=Loss(config))

    # Run the fit function on the model to calculate the learning rates
    history = model.fit(
        training_dataset,
        epochs=n_steps,
        steps_per_epoch=1,
        callbacks=[
            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.LearningRateScheduler(lr_schedule),
            LRProgress(n_steps, lr_schedule),
        ],
        verbose=False,
    )

    # Extract the loss and LR from before it became nan
    loss = np.array(history.history["loss"][:-1])
    lr = np.array(history.history["lr"][:-1])

    # Make a smoothed version of the loss
    smooth_loss = np.convolve(loss, np.ones(average_window), "same") / np.convolve(
        np.ones_like(loss), np.ones(average_window), "same"
    )

    # Find the point where the loss goes statistically off for the first time
    # We then add half our average window since that is where the spike actually starts
    delta = np.log10(smooth_loss)[1:] - np.log10(smooth_loss)[:-1]
    exploding = (average_window // 2) + np.argmax(delta > 3.0 * np.std(delta))

    # Work out the suggested maximum learning rate
    suggested_max = lr[exploding] / 10
    print("Suggested maximum learning rate: {:.0e}".format(suggested_max))

    # Output the details to file
    os.makedirs(os.path.join(output_path, "lr_finder"), exist_ok=True)
    with open(os.path.join(output_path, "lr_finder", "max_lr.csv"), "w") as f:
        f.write("{:.0e}".format(suggested_max))

    np.savetxt(os.path.join(output_path, "lr_finder", "lr_loss.csv"), np.stack([lr, loss], axis=-1), delimiter=",")

    # Draw the plots of raw and smoothed data
    fig, ax = plt.subplots()
    ax.set_title("Learning Rate Finder")
    ax.set_xlabel("log(Learning Rate)")
    ax.set_ylabel("log(Loss)")
    ax.plot(np.log10(lr), np.log10(loss))
    ax.plot(np.log10(lr), np.log10(smooth_loss))
    fig.savefig(os.path.join(output_path, "lr_finder", "lr_loss.png"))
    plt.close(fig)

    # Draw the plots of the delta in the smoothed loss
    fig, ax = plt.subplots()
    ax.set_title("Learning Rate Finder")
    ax.set_xlabel("log(Learning Rate)")
    ax.set_ylabel("Δlog(Loss)")
    ax.plot(np.log10(lr)[:-1], delta)
    fig.savefig(os.path.join(output_path, "lr_finder", "delta_lr_loss.png"))
    plt.close(fig)
